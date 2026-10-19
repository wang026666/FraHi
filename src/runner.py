import torch
import numpy as np

import os
import copy
import json
import pandas as pd
from pprint import pprint

from dataset import BaseDataset, CASMIDataset, data_to_device, get_dset_types
from config_utils import validate_training_runtime
from model import Prediction
from data_utils import th_temp_seed, count_parameters, get_pbar, get_scaler
from losses import get_loss_func, get_sim_func
from peak_losses import (
    combine_protected_training_losses,
    compute_peak_training_loss,
)
from metric_table import MetricTable
from spec_utils import process_spec, unprocess_spec, merge_spec
from training_utils import (
    accumulation_window_size,
    build_run_directory_name,
    cosine_warmup_multiplier,
    num_optimizer_updates,
    scheduled_non_degradation_weight,
    select_protected_validation_metrics,
    should_optimizer_step,
)


def run_train_epoch(step, epoch, model,dl_d, data_d, run_d,optimizer,scheduler):
    
    dev = torch.device(run_d["device"])
    nb = run_d["non_blocking"] 
    loss_func = get_loss_func(run_d["loss"],data_d["mz_bin_res"],agg=run_d["batch_loss_agg"])
    per_sample_loss_func = get_loss_func(
        run_d["loss"], data_d["mz_bin_res"], agg=None
    )
    total_losses = []
    spectrum_losses = []
    peak_metric_history = {}
    scaler = get_scaler(run_d["amp"]) 
    non_degradation_weight = scheduled_non_degradation_weight(
        epoch=epoch,
        target_weight=run_d.get(
            "non_degradation_weight", 0.0
        ),
        warmup_epochs=run_d.get(
            "non_degradation_warmup_epochs", 0
        ),
    )

    train_dl = dl_d["train"]
    num_batches = len(train_dl)
    if num_batches == 0:
        raise ValueError("training dataloader is empty")
    grad_acc_interval = run_d["grad_acc_interval"]

    model.train()
    optimizer.zero_grad()
    for b_idx, b in get_pbar(
            enumerate(train_dl),
            run_d["log_tqdm"],
            desc="> train",
            total=num_batches):
        b = data_to_device(b, dev, nb) 
        b_output = model(data=b, amp=run_d["amp"]) 
        b_pred = b_output.get("full_pred", b_output["pred"])
        b_targ = b["spec"]
        spectrum_loss = loss_func(b_pred, b_targ)
        if getattr(model, "prediction_head", "dense") == "dense":
            b_loss_agg = spectrum_loss
        else:
            peak_weights = {
                "spectrum": run_d.get("peak_spectrum_weight", 1.0),
                "existence": run_d.get("peak_existence_weight", 0.2),
                "mass": run_d.get("peak_mass_weight", 1.0),
                "intensity": run_d.get("peak_intensity_weight", 1.0),
                "consistency": run_d.get(
                    "peak_consistency_weight", 0.1
                ),
            }
            b_loss_agg, peak_metrics = compute_peak_training_loss(
                b_output,
                b,
                spectrum_loss=spectrum_loss,
                weights=peak_weights,
                mz_max=data_d["mz_max"],
                fragment_match_tolerance=run_d.get(
                    "fragment_match_tolerance", 0.2
                ),
            )
            baseline_prediction = b_output["baseline_pred"]
            baseline_loss = loss_func(baseline_prediction, b_targ)
            b_loss_agg, protected_metrics = (
                combine_protected_training_losses(
                    full_loss=b_loss_agg,
                    baseline_loss=baseline_loss,
                    full_per_sample_loss=per_sample_loss_func(
                        b_pred,
                        b_targ,
                    ),
                    baseline_per_sample_loss=per_sample_loss_func(
                        baseline_prediction,
                        b_targ,
                    ),
                    baseline_weight=run_d["baseline_loss_weight"],
                    non_degradation_weight=non_degradation_weight,
                )
            )
            peak_metrics.update(protected_metrics)
            for key, value in peak_metrics.items():
                peak_metric_history.setdefault(key, []).append(
                    float(value.cpu())
                )

        window_size = accumulation_window_size(
            b_idx, num_batches, grad_acc_interval
        )
        scaler.scale(b_loss_agg / window_size).backward()

        if should_optimizer_step(b_idx, num_batches, grad_acc_interval):
            scaler.unscale_(optimizer) 
            parameter_groups = model.protected_parameter_groups()
            clip_norms = {
                "baseline": run_d.get(
                    "baseline_clip_grad_norm",
                    run_d["clip_grad_norm"],
                ),
                "residual": run_d["clip_grad_norm"],
            }
            for group_name, parameters in parameter_groups.items():
                if parameters:
                    torch.nn.utils.clip_grad_norm_(
                        parameters,
                        clip_norms[group_name],
                    )
            scaler.step(optimizer) 
            scaler.update() 
            optimizer.zero_grad()
            if run_d["scheduler"] == "cosine":
                scheduler.step()

        step += 1
        total_losses.append(
            b_loss_agg.detach().to("cpu").item()
        )
        spectrum_losses.append(
            spectrum_loss.detach().to("cpu").item()
        )

    train_total_loss = np.mean(total_losses)
    train_spec_loss = np.mean(spectrum_losses)
    print(f"epoch: {epoch}, train_total_loss: {train_total_loss:.6f}")
    print(f"epoch: {epoch}, train_spec_loss: {train_spec_loss:.6f}")
    if peak_metric_history:
        peak_summary = {
            key: float(np.mean(values))
            for key, values in peak_metric_history.items()
        }
        print(f"epoch: {epoch}, train_peak_metrics: {peak_summary}")
   
    return step, epoch, {}


def compute_metric_tables(pred,targ,mol_id, group_id,prefix,data_d,run_d,auxiliary=False,merge_group=False,compute_agg=False,compute_hist=False,groupby_mol=False,um_batch_size=10000,m_batch_size=1000):

    def merge_group_func(_pred, _targ, _group_id, _mol_id, _transform):

        assert group_id is not None and mol_id is not None

        if _transform == "obj":
            t = data_d["transform"] 
            def pp(x): return x
            n = data_d["spectrum_normalization"]
        elif _transform == "std":
            t = "none"
            def pp(x): return x
            n = "l1"
        elif _transform == "log":
            t = "log10over3"
            def pp(x): return x
            n = "l1"
        else:
            raise ValueError
        m_pred, m_group_id, m_mol_id = merge_spec(_pred, _group_id, t, n, _mol_id)
        m_pred = pp(m_pred)
        m_targ, _ = merge_spec(_targ, _group_id, t, n)
        return m_pred, m_targ, m_mol_id, m_group_id

    def standardize_spectrum(spec):
        return process_spec(
            unprocess_spec(spec, data_d["transform"]),
            "none",
            "l1",
        )

    required_sim_metrics = (
        "cosine",
        "modified_cosine",
        "spectral_entropy",
    )
    sim_metrics = tuple(
        run_d.get("sim_metrics", required_sim_metrics)
    )
    missing_sim_metrics = set(required_sim_metrics) - set(sim_metrics)
    if missing_sim_metrics:
        raise ValueError(
            "run.sim_metrics must include cosine, modified_cosine, "
            "and spectral_entropy; missing: "
            + ", ".join(sorted(missing_sim_metrics))
        )

    um_num_batches = len(pred) // um_batch_size + int(len(pred) % um_batch_size != 0)

    obj_sim_func = get_sim_func(run_d["sim"], data_d["mz_bin_res"])
    obj_loss_func = get_loss_func(run_d["loss"], data_d["mz_bin_res"])
    paper_sim_funcs = {
        metric: get_sim_func(
            metric,
            data_d["mz_bin_res"],
            tolerance=run_d.get("modified_cosine_tolerance", 0.1),
            noise_threshold=run_d.get("sim_noise_cutoff", 0.01),
        )
        for metric in sim_metrics
    }

    sim_obj, loss_obj = [], []
    paper_sim_values = {metric: [] for metric in sim_metrics}
    for b in get_pbar(range(um_num_batches),run_d["log_tqdm"],desc="> unmerged metrics"):
        b_pred = pred[b*um_batch_size:(b+1)*um_batch_size] 
        b_targ = targ[b*um_batch_size:(b+1)*um_batch_size]

        b_sim_obj = obj_sim_func(b_pred, b_targ) 
        b_loss_obj = obj_loss_func(b_pred, b_targ) 
        sim_obj.append(b_sim_obj)
        loss_obj.append(b_loss_obj)
        b_pred_std = standardize_spectrum(b_pred)
        b_targ_std = standardize_spectrum(b_targ)
        for metric, sim_func in paper_sim_funcs.items():
            paper_sim_values[metric].append(
                sim_func(b_pred_std, b_targ_std)
            )
    sim_d = {
        "sim_obj": torch.cat(sim_obj,dim=0),
        "loss_obj": torch.cat(loss_obj,dim=0)
    }
    for metric, values in paper_sim_values.items():
        sim_d[f"sim_{metric}"] = torch.cat(values, dim=0)
    if auxiliary:
        sim_d["sim_cos_std"] = sim_d["sim_cosine"]

    if merge_group:
        un_group_id = torch.unique(group_id) 
       
        m_num_batches = len(un_group_id) // m_batch_size + int(len(un_group_id) % m_batch_size != 0)
       
        m_sim_obj, m_loss_obj, m_group_id, m_mol_id = [], [], [], []
        m_paper_sim_values = {metric: [] for metric in sim_metrics}
        for b in get_pbar(range(m_num_batches),run_d["log_tqdm"],desc="> merged metrics"):
            b_group_id = un_group_id[b*m_batch_size:(b+1)*m_batch_size] 
            b_mask = torch.isin(group_id,b_group_id) 
            b_group_id = group_id[b_mask]
            b_mol_id = mol_id[b_mask]
            b_pred = pred[b_mask]
            b_targ = targ[b_mask]
            b_m_pred, b_m_targ, b_m_mol_id, b_m_group_id = merge_group_func(
                b_pred, b_targ, b_group_id, b_mol_id, "obj"
            )
            b_m_sim_obj = obj_sim_func(b_m_pred, b_m_targ)
            b_m_loss_obj = obj_loss_func(b_m_pred, b_m_targ)
            m_sim_obj.append(b_m_sim_obj)
            m_loss_obj.append(b_m_loss_obj)
            m_group_id.append(b_m_group_id)
            m_mol_id.append(b_m_mol_id)
            b_pred_std = standardize_spectrum(b_pred)
            b_targ_std = standardize_spectrum(b_targ)
            b_m_pred_std, b_m_targ_std, _, _ = merge_group_func(
                b_pred_std,
                b_targ_std,
                b_group_id,
                b_mol_id,
                "std",
            )
            for metric, sim_func in paper_sim_funcs.items():
                m_paper_sim_values[metric].append(
                    sim_func(b_m_pred_std, b_m_targ_std)
                )
        m_group_id = torch.cat(m_group_id,dim=0)
        m_mol_id = torch.cat(m_mol_id,dim=0)
        sim_d["m_sim_obj"] = torch.cat(m_sim_obj,dim=0)
        sim_d["m_loss_obj"] = torch.cat(m_loss_obj,dim=0)
        sim_d["m_group_id"] = m_group_id
        sim_d["m_mol_id"] = m_mol_id
        for metric, values in m_paper_sim_values.items():
            sim_d[f"m_sim_{metric}"] = torch.cat(values, dim=0)
        if auxiliary:
            sim_d["m_sim_cos_std"] = sim_d["m_sim_cosine"]
  
    merged_flags = [False] 
    if merge_group:
        merged_flags.append(True)
    groupby_mol_flags = [False] 
    if groupby_mol:
        groupby_mol_flags.append(True)
    tables = []
    for sl in ["sim", "loss"]:
        for merged in merged_flags:
            keys, vals = [], []
            if merged: 
                _mol_id = m_mol_id
                _group_id = m_group_id
                for k, v in sim_d.items():
                    if k.startswith(f"m_{sl}"): 
                        keys.append(k[len(f"m_{sl}_"):])
                        vals.append(v)
            else:
                _mol_id = mol_id
                _group_id = group_id
                for k, v in sim_d.items():
                    if k.startswith(sl):
                        keys.append(k[len(f"{sl}_"):])
                        vals.append(v)

            table = MetricTable(keys, vals, _mol_id, _group_id, prefix, loss=(sl == "loss"), merged=merged)
            for gm in groupby_mol_flags:
                table.compute(compute_agg=compute_agg,compute_hist=compute_hist,groupby_mol=gm)
            tables.append(table)
    return tables


def run_val(step,epoch,model,dl_d,data_d,run_d):

    if not (dl_d["primary"]["val"] is None):
      
        dev = torch.device(run_d["device"])
        nb = run_d["non_blocking"]
        model.eval()
        pred, baseline_pred, targ, mol_id, group_id = [], [], [], [], []
        with torch.no_grad():
            for b_idx, b in get_pbar(enumerate(dl_d["primary"]["val"]), run_d["log_tqdm"], desc="> val", total=len(dl_d["primary"]["val"])):
                b = data_to_device(b, dev, nb)
                b_output = model(data=b, amp=run_d["amp"])
                b_pred = b_output.get("full_pred", b_output["pred"])
                b_targ = b["spec"]
                b_mol_id = b["mol_id"]
                b_group_id = b["group_id"]
                pred.append(b_pred.detach().to("cpu", non_blocking=nb))
                if "baseline_pred" in b_output:
                    baseline_pred.append(
                        b_output["baseline_pred"].detach().to(
                            "cpu", non_blocking=nb
                        )
                    )
                targ.append(b_targ.detach().to("cpu", non_blocking=nb))
                mol_id.append(b_mol_id.detach().to("cpu", non_blocking=nb))
                group_id.append(b_group_id.detach().to("cpu", non_blocking=nb))
        pred = torch.cat(pred, dim=0)
        targ = torch.cat(targ, dim=0)
        mol_id = torch.cat(mol_id, dim=0)
        group_id = torch.cat(group_id, dim=0)
        def evaluate_prediction(prediction, prefix):
            metric_tables = compute_metric_tables(
                prediction,
                targ,
                mol_id,
                group_id,
                prefix,
                data_d,
                run_d,
                auxiliary=run_d["log_auxiliary"],
                merge_group=True,
                compute_agg=True,
                compute_hist=False,
                groupby_mol=True,
            )
            metrics = {}
            for metric_table in metric_tables:
                metrics = dict(
                    **metrics,
                    **metric_table.unload_cache(
                        prefix=False,
                        agg=True,
                        hist=False,
                    ),
                    **metric_table.export_val("obj"),
                )
            return metric_tables, metrics

        full_tables, full_metrics = evaluate_prediction(pred, "val_full")
        tables = full_tables
        out_d = full_metrics
        if baseline_pred and bool(run_d.get("validation_fallback", False)):
            baseline_pred = torch.cat(baseline_pred, dim=0)
            baseline_tables, baseline_metrics = evaluate_prediction(
                baseline_pred,
                "val_baseline",
            )
            out_d = select_protected_validation_metrics(
                full_metrics,
                baseline_metrics,
                run_d["stop_key"],
            )
            if out_d["prediction_source"] == "baseline":
                tables = baseline_tables
        stop_key = run_d["stop_key"] 
        spec_loss_obj_mean = out_d["spec_loss_obj_mean"]
        loss_mean = out_d[stop_key]
        prediction_source = out_d.get("prediction_source", "full")
        print(
            f"> step {step}, epoch {epoch}: val[{prediction_source}], "
            f"{stop_key}: {loss_mean:.4f}, "
            f"spec_loss_obj_mean:{spec_loss_obj_mean}"
        )
        log_d = {"epoch": epoch, "Epoch": epoch} 
        for table in tables:
            for k, v in table.unload_cache(
                    agg=True, hist=run_d["save_media"]).items():
                log_d[k] = v 
    else:
        out_d = {run_d["stop_key"]: np.nan}
    return step, epoch, out_d


def compute_cross_sims(data_d, run_d, dl_d):
  
    if not run_d["do_test"]:
        return None
    test_dl = dl_d["primary"]["test"]
    sec_dl_d = dl_d["secondary"]
    if test_dl is None or len(sec_dl_d) == 0:
        return None

    test_group_id, test_mol_id, test_prec_type, test_ce, test_spec = [], [], [], [], []
    for b_idx, b in enumerate(test_dl):
        test_group_id.extend(b["group_id"])
        test_mol_id.extend(b["mol_id"])
        test_prec_type.extend(b["prec_type"])
        test_ce.extend(b["col_energy"])
        test_spec.append(b["spec"])
    test_df = pd.DataFrame({
        "group_id": torch.stack(test_group_id, dim=0).numpy(),
        "mol_id": torch.stack(test_mol_id, dim=0).numpy(),
        "prec_type": test_prec_type,
        "col_energy": test_ce
    })
    test_df.loc[:, "idx"] = np.arange(test_df.shape[0])
    test_m_df = test_df.drop( 
        columns=[
            "idx", "col_energy"]).drop_duplicates(
        subset=["group_id"]).sort_values("group_id")
    test_m_df.loc[:, "idx"] = np.arange(test_m_df.shape[0])
    test_spec = torch.cat(test_spec, dim=0)
    test_spec = unprocess_spec(test_spec, data_d["transform"])
    test_spec = process_spec(test_spec, "none", "l1")
    test_m_spec, _ = merge_spec(test_spec, torch.as_tensor(
        test_df["group_id"].to_numpy()), "none", "l1")
    sim_func = get_sim_func("cos", data_d["mz_bin_res"])

   
    log_d = {}
    for sec_key, sec_dl in sec_dl_d.items(): 

        sec_group_id, sec_mol_id, sec_prec_type, sec_ce, sec_spec = [], [], [], [], []
        for b_idx, b in enumerate(sec_dl):
            sec_group_id.extend(b["group_id"])
            sec_mol_id.extend(b["mol_id"])
            sec_prec_type.extend(b["prec_type"])
            sec_ce.extend(b["col_energy"])
            sec_spec.append(b["spec"])
        sec_df = pd.DataFrame({
            "group_id": torch.stack(sec_group_id, dim=0).numpy(),
            "mol_id": torch.stack(sec_mol_id, dim=0).numpy(),
            "prec_type": sec_prec_type,
            "col_energy": sec_ce
        })
        sec_df.loc[:, "idx"] = np.arange(sec_df.shape[0]) 
        sec_m_df = sec_df.drop( 
            columns=[
                "idx", "col_energy"]).drop_duplicates(
            subset=["group_id"]).sort_values("group_id")
        sec_m_df.loc[:, "idx"] = np.arange(sec_m_df.shape[0]) 


        sec_spec = torch.cat(sec_spec, dim=0)
        sec_spec = unprocess_spec(sec_spec, data_d["transform"])
        sec_spec = process_spec(sec_spec, "none", "l1")
        sec_m_spec, _ = merge_spec(sec_spec, torch.as_tensor(sec_df["group_id"].to_numpy()), "none", "l1")
        

        both_df = test_df.merge(sec_df, on=["mol_id", "prec_type", "col_energy"], how="inner") 

        both_test_spec = test_spec[torch.as_tensor(both_df["idx_x"].to_numpy())]
        both_sec_spec = sec_spec[torch.as_tensor(both_df["idx_y"].to_numpy())]


        both_m_df = test_m_df.merge(sec_m_df, on=["mol_id", "prec_type"], how="inner")

        both_test_m_spec = test_m_spec[torch.as_tensor(both_m_df["idx_x"].to_numpy())]
        both_sec_m_spec = sec_m_spec[torch.as_tensor(both_m_df["idx_y"].to_numpy())]

        sim = sim_func(both_test_spec, both_sec_spec) 
        m_sim = sim_func(both_test_m_spec, both_sec_m_spec) 
        log_d[f"test_{sec_key}_cross_sim"] = torch.mean(sim)
        log_d[f"test_{sec_key}_cross_m_sim"] = torch.mean(m_sim) 

    if run_d["print_stats"]:
        pprint(log_d)
    return None


def run_test(step,epoch,model,dl_d,data_d,model_d,run_d,run_dir,test_sets=None):

    if test_sets is None:
        test_sets = ["test"]
    if run_d["do_test"]:

        dev = torch.device(run_d["device"])
        nb = run_d["non_blocking"]
        print(">> test")

        model.to(dev)
        model.eval()
        out_d, save_tables = {}, [] 
        for order in ["primary", "secondary"]:
            out_d[order] = {}
            for dl_key, dl in dl_d[order].items():
                if not (dl_key in test_sets) or dl is None: 
                    continue
                pred, full_pred, baseline_pred = [], [], []
                targ, mol_id, group_id = [], [], []
                with torch.no_grad():
                    for b_idx, b in get_pbar(enumerate(dl), run_d["log_tqdm"], desc=f"> {dl_key}", total=len(dl)):
                        b = data_to_device(b, dev, nb)
                        b_output = model(data=b, amp=run_d["amp"])
                        b_pred = b_output["pred"]
                        b_targ = b["spec"]
                        b_mol_id = b["mol_id"]
                        b_group_id = b["group_id"]
                        pred.append(b_pred.detach().to("cpu", non_blocking=nb))
                        if "baseline_pred" in b_output:
                            full_pred.append(
                                b_output["full_pred"].detach().to(
                                    "cpu", non_blocking=nb
                                )
                            )
                            baseline_pred.append(
                                b_output["baseline_pred"].detach().to(
                                    "cpu", non_blocking=nb
                                )
                            )
                        targ.append(b_targ.detach().to("cpu", non_blocking=nb))
                        mol_id.append(b_mol_id.detach().to("cpu", non_blocking=nb))
                        group_id.append(b_group_id.detach().to("cpu", non_blocking=nb))
                pred = torch.cat(pred, dim=0)
                targ = torch.cat(targ, dim=0)
                mol_id = torch.cat(mol_id, dim=0)
                group_id = torch.cat(group_id, dim=0)
                def evaluate_test_prediction(prediction, prefix):
                    metric_tables = compute_metric_tables(
                        prediction,
                        targ,
                        mol_id,
                        group_id,
                        prefix,
                        data_d,
                        run_d,
                        auxiliary=run_d["log_auxiliary"],
                        merge_group=True,
                        compute_agg=True,
                        compute_hist=False,
                        groupby_mol=True,
                    )
                    metrics = {}
                    for metric_table in metric_tables:
                        metrics = dict(
                            **metrics,
                            **metric_table.unload_cache(
                                prefix=False,
                                agg=True,
                                hist=False,
                            ),
                        )
                    return metric_tables, metrics

                tables, _out_d = evaluate_test_prediction(pred, dl_key)
                stop_key = run_d["stop_key"]
                loss_mean = _out_d[stop_key]
                branch_tables = []
                branch_metric_sets = set(
                    run_d.get("branch_metric_sets", ("val", "test"))
                )
                if baseline_pred and dl_key in branch_metric_sets:
                    full_pred = torch.cat(full_pred, dim=0)
                    baseline_pred = torch.cat(baseline_pred, dim=0)
                    full_tables, full_metrics = evaluate_test_prediction(
                        full_pred,
                        f"{dl_key}_full",
                    )
                    baseline_tables, baseline_metrics = (
                        evaluate_test_prediction(
                            baseline_pred,
                            f"{dl_key}_baseline",
                        )
                    )
                    _out_d = {
                        **_out_d,
                        **{
                            f"full_{key}": value
                            for key, value in full_metrics.items()
                        },
                        **{
                            f"baseline_{key}": value
                            for key, value in baseline_metrics.items()
                        },
                        "prediction_source": (
                            model.inference_prediction_source()
                        ),
                    }
                    branch_tables.extend(full_tables)
                    branch_tables.extend(baseline_tables)
                    print(
                        f"> {dl_key}, selected[{_out_d['prediction_source']}] "
                        f"{stop_key}={loss_mean:.4f}, "
                        f"full={full_metrics[stop_key]:.4f}, "
                        f"baseline={baseline_metrics[stop_key]:.4f}"
                    )
                else:
                    print(f"> {dl_key}, {stop_key} = {loss_mean:.4f}")
                out_d[order][dl_key] = _out_d
                log_d = {"epoch": epoch, "Epoch": epoch}
                for table in tables: 
                    for k, v in table.unload_cache(hist=run_d["save_media"]).items():
                        log_d[k] = v
                if run_d["save_test_sims"]:
                    save_tables.extend(tables)
                    save_tables.extend(branch_tables)
        if run_d["save_test_sims"]:
            save_dp = os.path.join(run_dir, "save_tables")
            os.makedirs(save_dp, exist_ok=True)
            for table in save_tables:
                save_str = table.get_table_str()
                save_fp = os.path.join(save_dp, save_str)
                table.save(save_fp) 
    else:
        out_d = {}
    return step, epoch, out_d


def rank_metrics(rank, total):

    d = {}
    d["rank"] = float(rank)
    d["top01"] = float(rank == 1)
    d["top05"] = float(rank <= 5)
    d["top10"] = float(rank <= 10)
    # d["ndcg"] = 1. / np.log2(float(rank) + 1.)
    norm_rank = float((rank - 1) / total)
    d["norm_rank"] = norm_rank
    d["top01%"] = float(norm_rank <= 0.01) 
    d["top05%"] = float(norm_rank <= 0.05)
    d["top10%"] = float(norm_rank <= 0.10) 
    d["total"] = total
    return d


def sims_to_rank_metrics(sim, sim2, key_prefix, cand_match_mask):
   
    rm_d = {}
    key = f"{key_prefix}"
    cand_match_idx = torch.argmax(cand_match_mask.float()) 
    rm_d[f"{key}_sim"] = sim[cand_match_idx].item() 
    noisey_sim = sim + 0.00001 * torch.rand_like(sim) 
    sim_argsorted = torch.argsort(-noisey_sim, dim=0)
    rank = torch.argmax(cand_match_mask.float()[sim_argsorted]) + 1
    _rm_d = rank_metrics(rank, cand_match_mask.shape[0]) 
    rm_d.update({f"{key}_{k}":v for k,v in _rm_d.items()})

    rm_d[f"{key}_sim2"] = sim2[cand_match_idx].item() 
    noisey_sim2 = sim2 + 0.00001 * torch.rand_like(sim2) 
    sim2_argsorted = torch.argsort(-noisey_sim2, dim=0) 

    num_20p = max(1, int(np.round(0.2*sim2_argsorted.shape[0])))
    sim2_t20p = sim2_argsorted[:num_20p] 
    if cand_match_idx not in sim2_t20p: 
        sim2_t20p = torch.cat([cand_match_idx.reshape(1,),sim2_t20p[1:]],dim=0)
    sim_t20p_argsorted = torch.argsort(-noisey_sim[sim2_t20p], dim=0)
    rank_t20p = torch.argmax(cand_match_mask.float()[sim2_t20p][sim_t20p_argsorted]) + 1
    total_t20p = sim_t20p_argsorted.shape[0] 
    _rm_d = rank_metrics(rank_t20p, total_t20p) 
    rm_d.update({f"{key}_t20p_{k}":v for k,v in _rm_d.items()})

    sim2_b20p = sim2_argsorted[-num_20p:]
    if cand_match_idx not in sim2_b20p:
        sim2_b20p = torch.cat([cand_match_idx.reshape(1,),sim2_b20p[1:]],dim=0)
    sim_b20p_argsorted = torch.argsort(-noisey_sim[sim2_b20p], dim=0)
    rank_b20p = torch.argmax(cand_match_mask.float()[sim2_b20p][sim_b20p_argsorted]) + 1
    total_b20p = sim_b20p_argsorted.shape[0]
    _rm_d = rank_metrics(rank_b20p, total_b20p)
    rm_d.update({f"{key}_b20p_{k}":v for k,v in _rm_d.items()})
    return rm_d


def run_casmi(step,epoch,model,casmi_ds,casmi_type,data_d,model_d,run_d,run_dir,mr_d,update_mr_d):

    assert casmi_type in ["casmi22"]
    log_dict = {}
    if run_d[f"do_{casmi_type}"]:
        if casmi_ds is None:
            raise ValueError(
                f"{casmi_type} evaluation is enabled but its dataset is unavailable"
            )
        pred_all = run_d[f"{casmi_type}_pred_all"]
        casmi_d = mr_d[f"{casmi_type}_d"]
        print(f">> {casmi_type}")

        dev = torch.device(run_d["device"])
        nb = run_d["non_blocking"]

        model.to(dev)
        model.eval()
        model.set_mode(casmi_type)

        spec_dl = casmi_ds.get_dataloader(run_d, "spec")
       
        spec, spec_spec_id, spec_group_id, spec_mol_id, spec_casmi_fp = [], [], [], [], []
        for b_idx, b in get_pbar(enumerate(spec_dl), run_d["log_tqdm"], desc="> spec", total=len(spec_dl)):
            spec.append(b["spec"])
            spec_spec_id.append(b["spec_id"])
            spec_group_id.append(b["group_id"])
            spec_mol_id.append(b["mol_id"])
            spec_casmi_fp.append(b["casmi_fp"])
        spec = torch.cat(spec, dim=0)
        spec_spec_id = torch.cat(spec_spec_id, dim=0)
        spec_group_id = torch.cat(spec_group_id, dim=0)
        spec_mol_id = torch.cat(spec_mol_id, dim=0)
        spec_casmi_fp = torch.cat(spec_casmi_fp, dim=0)
      
        spec_merge, spec_group_id_merge, spec_mol_id_merge, spec_casmi_fp_merge = merge_spec(
            spec, spec_group_id, casmi_ds.transform, casmi_ds.spectrum_normalization, spec_mol_id, spec_casmi_fp)
       
        if pred_all:
            
            group_dl = casmi_ds.get_dataloader(run_d, "group")
           
            cand_pred, cand_group_id, cand_mol_id, cand_spec_id, cand_casmi_fp = [], [], [], [], []
            with torch.no_grad():
                for b_idx, b in get_pbar(enumerate(group_dl), run_d["log_tqdm"], desc="> group all", total=len(group_dl)):
                    b = data_to_device(b, dev, nb)
                    b_group_id = b["group_id"]
                    b_mol_id = b["mol_id"]
                    b_spec_id = b["spec_id"]
                    b_casmi_fp = b["casmi_fp"]
                    b_pred = model(data=b, amp=run_d["amp"])["pred"] 
                    cand_pred.append(b_pred.cpu())
                    cand_group_id.append(b_group_id.cpu())
                    cand_mol_id.append(b_mol_id.cpu())
                    cand_spec_id.append(b_spec_id.cpu())
                    cand_casmi_fp.append(b_casmi_fp.cpu())
            cand_pred_all = torch.cat(cand_pred, dim=0)
            cand_group_id_all = torch.cat(cand_group_id, dim=0)
            cand_mol_id_all = torch.cat(cand_mol_id, dim=0)
            cand_spec_id_all = torch.cat(cand_spec_id, dim=0)
            cand_casmi_fp_all = torch.cat(cand_casmi_fp, dim=0)
            assert torch.isin(spec_mol_id, cand_mol_id_all).all() 
            assert torch.isin(spec_mol_id_merge, cand_mol_id_all).all() 

        rm_ds, um_rm_ds = casmi_d["rm_ds"], casmi_d["um_rm_ds"]
        sims, sims2, group_ids = casmi_d["sims"], casmi_d["sims2"], casmi_d["group_ids"]
        sim_func = get_sim_func(run_d["sim"], data_d["mz_bin_res"])
        fp_sim_func = get_sim_func("jacc", None) 
        for i in range(spec_group_id_merge.shape[0]): 
            query_group_id = int(spec_group_id_merge[i]) 
            query_mol_id = int(spec_mol_id_merge[i]) 
            if query_group_id in casmi_d["query_group_ids"]: 
                continue
            if pred_all:
            
                cand_group_mask = torch.as_tensor(np.isin(cand_group_id_all.numpy(),query_group_id),dtype=torch.bool)
                cand_pred = cand_pred_all[cand_group_mask]
                cand_group_id = cand_group_id_all[cand_group_mask]
                cand_mol_id = cand_mol_id_all[cand_group_mask]
                cand_spec_id = cand_spec_id_all[cand_group_mask]
                cand_casmi_fp = cand_casmi_fp_all[cand_group_mask]
                cand_pred_merge, cand_mol_id_merge, cand_casmi_fp_merge = merge_spec(cand_pred, cand_mol_id, casmi_ds.transform, casmi_ds.spectrum_normalization,cand_casmi_fp)
            else:

                group_dl = casmi_ds.get_dataloader(run_d, "group", group_id=query_group_id)
                cand_pred, cand_mol_id, cand_spec_id, cand_casmi_fp = [], [], [], []
                with torch.no_grad():
                    for b_idx, b in get_pbar(enumerate(group_dl), run_d["log_tqdm"], desc=f"> group {i+1} / {spec_group_id_merge.shape[0]}", total=len(group_dl)):
                        b = data_to_device(b, dev, nb)
                        b_mol_id = b["mol_id"]
                        b_spec_id = b["spec_id"]
                        b_casmi_fp = b["casmi_fp"]
                        b_pred = model(data=b, amp=run_d["amp"])["pred"]
                        cand_pred.append(b_pred.cpu())
                        cand_mol_id.append(b_mol_id.cpu())
                        cand_spec_id.append(b_spec_id.cpu())
                        cand_casmi_fp.append(b_casmi_fp.cpu())
                cand_pred = torch.cat(cand_pred, dim=0)
                cand_mol_id = torch.cat(cand_mol_id, dim=0)
              
                cand_spec_id = torch.cat(cand_spec_id, dim=0)
                cand_casmi_fp = torch.cat(cand_casmi_fp, dim=0)
             
                cand_pred = torch.nan_to_num(cand_pred, nan=0.0)
                cand_mol_id = torch.nan_to_num(cand_mol_id, nan=0.0)
                cand_casmi_fp = torch.nan_to_num(cand_casmi_fp, nan=0.0)
                cand_pred_merge, cand_mol_id_merge, cand_casmi_fp_merge = merge_spec(cand_pred, cand_mol_id, casmi_ds.transform, casmi_ds.spectrum_normalization,cand_casmi_fp)
           
            cand_match_mask = torch.as_tensor(cand_mol_id_merge == query_mol_id,dtype=torch.bool)  
            assert cand_match_mask.sum() == 1, cand_match_mask.sum() 
            cand_spec = cand_pred_merge 
          
            targ_spec = spec_merge[i].unsqueeze(0).expand(cand_spec.shape[0], -1)
            sim_obj = sim_func(cand_spec, targ_spec) 
            cand_fp = cand_casmi_fp_merge 
            targ_fp = spec_casmi_fp_merge[i].unsqueeze(0).expand(cand_fp.shape[0], -1)
            sim_fp = fp_sim_func(cand_fp, targ_fp) 
            if run_d["casmi_save_sim"]:
                sims.append(sim_obj)
                sims2.append(sim_fp)
                group_ids.append(query_group_id)
            rm_d = sims_to_rank_metrics(sim_obj, sim_fp, casmi_type, cand_match_mask)
            rm_ds.append(rm_d)

          
            casmi_d["query_group_ids"].add(query_group_id)
            
            update_mr_d(mr_d,**{f"{casmi_type}_d": casmi_d})
        if not rm_ds:
            raise RuntimeError(f"{casmi_type} produced no ranking results")
        rm_d = {k: np.array([d[k] for d in rm_ds]) for k in rm_ds[0]} 
        if len(um_rm_ds) > 0: 
            um_rm_d = {k: np.array([d[k] for d in um_rm_ds]) for k in um_rm_ds[0]}
        else:
            um_rm_d = {}
 
        for k, v in rm_d.items():
            log_dict[k] = np.mean(v)
        for k, v in um_rm_d.items():
            log_dict[k] = np.mean(v)
   
        if sims:
            sims_all = torch.cat(sims,dim=0).cpu().numpy()
            print(f">> {casmi_type} sims_all shape: {sims_all.shape}")

       
        if run_d["print_stats"]:
            pprint(log_dict)
    return step, epoch, log_dict


def get_ds_model(data_d, model_d, run_d):

    with th_temp_seed(model_d["model_seed"]):
        embed_types = model_d["embed_types"]
        dset_types = get_dset_types(embed_types)
        assert len(dset_types) > 0, dset_types
        ds = BaseDataset(*dset_types, **data_d)
        dim_d = ds.get_data_dims()
    
        model = Prediction(
            dim_d,
            mz_max=data_d["mz_max"],
            mz_bin_res=data_d["mz_bin_res"],
            **model_d,
        )
       
        if run_d["do_casmi22"]:
            casmi22_ds = CASMIDataset(ds, "casmi22", *dset_types, **data_d)
            ds.update_casmi_info(casmi22_ds)
        else:
            casmi22_ds = None
    dev = torch.device(run_d["device"])
    model.to(dev)
    return ds, model, casmi22_ds


def init_casmi_d():
    d = {}
    d["query_group_ids"] = set()
    for k in ["rm_ds","um_rm_ds","sims","sims2","group_ids"]:
        d[k] = []
    return d


def train_and_eval(data_d, model_d, run_d):

    validate_training_runtime(model_d, run_d)
    base_dir = run_d["base_dir"]
    custom_name = run_d["custom_name"]
    resume_dir = run_d.get("resume_dir")
    if resume_dir:
        run_dir = resume_dir
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"resume directory does not exist: {run_dir}")
        resume_checkpoint = os.path.join(run_dir, "chkpt.pkl")
        if not os.path.isfile(resume_checkpoint):
            raise FileNotFoundError(
                f"resume checkpoint does not exist: {resume_checkpoint}"
            )
    else:
        run_name = build_run_directory_name(
            custom_name,
            run_d["split_seed"],
            model_d["model_seed"],
            run_d["train_seed"],
        )
        run_dir = os.path.join(base_dir, run_name)
        os.makedirs(run_dir, exist_ok=False)
    print(f"Run directory: {run_dir}")

    # set seeds
    torch.manual_seed(run_d["train_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_d["train_seed"])
    np.random.seed(run_d["train_seed"] // 2)
    if run_d["cuda_deterministic"]:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

    resolved_config_fp = os.path.join(run_dir, "resolved_config.json")
    with open(resolved_config_fp, "w", encoding="utf-8") as config_file:
        json.dump(
            {"data": data_d, "model": model_d, "run": run_d},
            config_file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

   
    ds, model, casmi22_ds = get_ds_model(data_d, model_d, run_d)
    if run_d["pretrained"] is not None:
        state_dict_path = run_d["pretrained"]
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_checkpoint(state_dict)
        print(
            ">>> successfully loaded same-architecture model from "
            f"{state_dict_path}: {model.prediction_head}"
        )
    trainable_names = model.configure_joint_trainability()
    if not trainable_names:
        raise ValueError("model has no trainable parameters")
    num_params = count_parameters(model, requires_grad=False)
    mol_embed_params, mlp_params, total_params = model.count_parameters()
    assert num_params == total_params, (num_params, total_params)
    print(f">>> mol_embed_params = {mol_embed_params}, mlp_params = {mlp_params}, total_params = {total_params}")
    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f">>> trainable_params = {trainable_params}")
    dl_d, split_id_d = ds.get_dataloaders(run_d) 

    if run_d["optimizer"] == "adam":
        optimizer_fn = torch.optim.Adam
    elif run_d["optimizer"] == "adamw":
        optimizer_fn = torch.optim.AdamW
    else:
        raise NotImplementedError
    optimizer = optimizer_fn(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=run_d["learning_rate"],
        weight_decay=run_d["weight_decay"],
    )

    if run_d["scheduler"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,run_d["scheduler_period"],gamma=run_d["scheduler_ratio"])
    elif run_d["scheduler"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",patience=run_d["scheduler_period"],factor=run_d["scheduler_ratio"])
    elif run_d["scheduler"] == "cosine":
        if dl_d["train"] is None:
            num_batches = 0
        else:
            num_batches = len(dl_d["train"])
        updates_per_epoch = num_optimizer_updates(
            num_batches, run_d["grad_acc_interval"]
        )
        tot_updates = run_d["num_epochs"] * updates_per_epoch
        warmup_updates = int(
            run_d.get("scheduler_warmup_epochs", 0)
        ) * updates_per_epoch
        end_ratio = float(run_d.get("scheduler_end_lr", 0.0)) / float(
            run_d["learning_rate"]
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda update_index: cosine_warmup_multiplier(
                update_index=update_index,
                total_updates=tot_updates,
                warmup_updates=warmup_updates,
                end_ratio=end_ratio,
            ),
        )
    elif run_d["scheduler"] == "none":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,1,gamma=1.0)
    else:
        raise NotImplementedError

    best_val_loss_mean = np.inf 
    best_val_metrics = {} 
    best_epoch = -1 
    best_prediction_source = "full"
    best_state_dict = copy.deepcopy(model.state_dict())
    early_stop_count = 0 
    early_stop_thresh = run_d["early_stop_thresh"] 
    step = 0 
    epoch = -1 
    casmi22_d = init_casmi_d()
    dev = torch.device(run_d["device"])

    mr_fp = os.path.join(run_dir, "chkpt.pkl") 
    temp_mr_fp = os.path.join(run_dir, "temp_chkpt.pkl") 
    split_id_fp = os.path.join(run_dir, "split_id.pkl")
    if os.path.isfile(mr_fp):
        print(">>> reloading model from most recent checkpoint")
        mr_d = torch.load(mr_fp,map_location="cpu")
        model.load_state_dict(mr_d["mr_model_sd"])
        best_state_dict = copy.deepcopy(mr_d["best_model_sd"])
        optimizer.load_state_dict(mr_d["optimizer_sd"]) 
        scheduler.load_state_dict(mr_d["scheduler_sd"]) 
        best_val_loss_mean = mr_d["best_val_loss_mean"]
        best_val_metrics = mr_d["best_val_metrics"]
        best_epoch = mr_d["best_epoch"]
        best_prediction_source = mr_d.get(
            "best_prediction_source", "full"
        )
        early_stop_count = mr_d["early_stop_count"]
        step = mr_d["step"]
        epoch = mr_d["epoch"]
        casmi22_d = mr_d.get("casmi22_d", init_casmi_d())
    else:
        print(">>> no checkpoint detected")
        mr_d = { 
            "mr_model_sd": model.state_dict(),
            "best_model_sd": best_state_dict,
            "optimizer_sd": optimizer.state_dict(),
            "scheduler_sd": scheduler.state_dict(),
            "best_val_loss_mean": best_val_loss_mean,
            "best_val_metrics": best_val_metrics,
            "best_epoch": best_epoch,
            "best_prediction_source": best_prediction_source,
            "early_stop_count": early_stop_count,
            "step": step,
            "epoch": epoch,
            "test": False,
            "test_d": {},
            "casmi22": False,
            "casmi22_d": casmi22_d
        }
        if run_d["save_split"]:
            torch.save(split_id_d, split_id_fp)

        if run_d["save_state"]:

            torch.save(mr_d,temp_mr_fp) 
            os.replace(temp_mr_fp,mr_fp) #

    model.to(dev)

    epoch += 1

    while epoch < run_d["num_epochs"]:

        print(f">>> start epoch {epoch}")
        should_early_stop = False
        
        step, epoch, _ = run_train_epoch(step, epoch, model, dl_d, data_d, run_d, optimizer, scheduler)
        
        step, epoch, val_d = run_val(step, epoch, model, dl_d, data_d, run_d)

        if run_d["scheduler"] == "step":
            scheduler.step()
        elif run_d["scheduler"] == "plateau":
            scheduler_key = run_d["stop_key"]
            if run_d.get("scheduler_monitor") == "baseline":
                scheduler_key = f"baseline_{scheduler_key}"
            scheduler.step(val_d[scheduler_key])

        val_loss_mean = val_d[run_d["stop_key"]]
        prediction_source = val_d.get("prediction_source", "full")
        if not np.isfinite(float(val_loss_mean)):
            raise RuntimeError(
                f"validation metric {run_d['stop_key']} is not finite: "
                f"{val_loss_mean}"
            )
        if best_val_loss_mean == np.inf: 
            print("> val loss delta: N/A")
        else:
            print(f"> val loss delta: {val_loss_mean-best_val_loss_mean}") 
        if run_d["use_val_info"]:
            if val_loss_mean >= best_val_loss_mean:
                early_stop_count += 1
                print( 
                    f"> val loss DID NOT decrease, early stop count at {early_stop_count}/{early_stop_thresh}")
            else:
                best_val_loss_mean = val_loss_mean
                best_val_metrics = {k: v for k, v in val_d.items() if ("_mean" in k)} 
                best_epoch = epoch
                best_prediction_source = prediction_source
                early_stop_count = 0 
              
                if model.prediction_head != "dense":
                    model.set_inference_prediction(prediction_source)
                model.to("cpu") # 
                best_state_dict = copy.deepcopy(model.state_dict())
             
                save_path = os.path.join(run_dir, "best_model_state.pth")
              
                torch.save(best_state_dict, save_path)
                print(f"Model state dict saved to {save_path}")

                model.to(dev)
                if model.prediction_head != "dense":
                    model.set_inference_prediction("full")
                print("> val loss DID decrease, early stop count reset")
            if early_stop_count >= early_stop_thresh:
                print("> early stopping NOW")
                should_early_stop = True
        else:
            
            best_val_loss_mean = val_loss_mean
            best_val_metrics = {k: v for k,v in val_d.items() if ("_mean" in k)}
            best_epoch = epoch
            best_prediction_source = prediction_source
            early_stop_count = 0
            
            if model.prediction_head != "dense":
                model.set_inference_prediction(prediction_source)
            model.to("cpu")
            best_state_dict = copy.deepcopy(model.state_dict())
            model.to(dev)
            if model.prediction_head != "dense":
                model.set_inference_prediction("full")

       
        mr_d = {
            "mr_model_sd": model.state_dict(), 
            "best_model_sd": best_state_dict, 
            "optimizer_sd": optimizer.state_dict(),
            "scheduler_sd": scheduler.state_dict(),
            "best_val_loss_mean": best_val_loss_mean,
            "best_val_metrics": best_val_metrics,
            "best_epoch": best_epoch,
            "best_prediction_source": best_prediction_source,
            "early_stop_count": early_stop_count,
            "step": step,
            "epoch": epoch,
            "test": False,
            "casmi22": False,
            "casmi22_d": casmi22_d
        }
        if run_d["save_state"]:
            torch.save(mr_d, temp_mr_fp)
            os.replace(temp_mr_fp,mr_fp)
     
        print(f"Finished epoch{epoch}")
        if should_early_stop:
            break
        epoch += 1

    
    def update_mr_d(mr_d,**kwargs):

        for k, v in kwargs.items():
            mr_d[k] = v
        if run_d["save_state"]:
            torch.save(mr_d, temp_mr_fp)
            os.replace(temp_mr_fp, mr_fp)
           

    test_d = mr_d.get("test_d", {})
    if not mr_d["test"]:
        compute_cross_sims(data_d, run_d, dl_d) 
        model.load_state_dict(best_state_dict)
        step, epoch, test_d = run_test(step, epoch, model, dl_d, data_d,model_d, run_d, run_dir, test_sets=run_d["test_sets"])
        update_mr_d(mr_d, test=True, test_d=test_d)


    casmi22_metrics = mr_d.get("casmi22_metrics", {})
    if not mr_d["casmi22"]:
        model.load_state_dict(best_state_dict)
        step, epoch, casmi22_metrics = run_casmi(
            step,
            epoch,
            model,
            casmi22_ds,
            "casmi22",
            data_d,
            model_d,
            run_d,
            run_dir,
            mr_d,
            update_mr_d,
        )
        update_mr_d(
            mr_d,
            casmi22=True,
            casmi22_d={},
            casmi22_metrics=casmi22_metrics,
        )


    mr_d.update({
        "mr_model_sd": model.state_dict(),
        "best_model_sd": best_state_dict,
        "best_val_loss_mean": best_val_loss_mean,
        "best_val_metrics": best_val_metrics,
        "best_epoch": best_epoch,
        "best_prediction_source": best_prediction_source,
        "epoch": epoch,
        "step": step,
        "test": True,
        "test_d": test_d,
        "casmi22": True,
        "casmi22_metrics": casmi22_metrics,
        "status": "completed",
    })
    if run_d["save_state"]:
        torch.save(mr_d, temp_mr_fp)
        os.replace(temp_mr_fp,mr_fp)

    best_model_fp = os.path.join(run_dir, "best_model_state.pth")
    if not os.path.isfile(best_model_fp):
        torch.save(best_state_dict, best_model_fp)

    def json_safe(value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.detach().cpu().item()
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    summary_fp = os.path.join(run_dir, "run_summary.json")
    with open(summary_fp, "w", encoding="utf-8") as summary_file:
        json.dump(
            json_safe({
                "status": "completed",
                "best_epoch": best_epoch,
                "best_prediction_source": best_prediction_source,
                "best_val_loss_mean": best_val_loss_mean,
                "best_val_metrics": best_val_metrics,
                "test_metrics": test_d,
                "casmi22_metrics": casmi22_metrics,
                "last_epoch": epoch,
                "batches_seen": step,
            }),
            summary_file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    if run_d["device"] != "cpu" and torch.cuda.is_available():
        cuda_max_memory = torch.cuda.max_memory_allocated(device=dev)/1e9
        print(f"> GPU memory: {cuda_max_memory:.2f} GB")

    return
