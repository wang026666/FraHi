"""Configuration validation and project-relative path handling."""

from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def normalize_prediction_head(prediction_head):
    """Map the public FraHi name to the checkpoint-compatible architecture."""
    return "hipeak" if prediction_head == "frahi" else prediction_head


def processed_data_filenames(data_d):
    """Return the processed spectrum/molecule filenames for a NIST release."""
    raw_version = data_d.get("nist_version")
    version = str(raw_version).strip()
    if not version.isdigit():
        raise ValueError(
            "data.nist_version must be a numeric NIST release, e.g. 20"
        )
    return f"spec_{version}_df.pkl", f"mol_{version}_df.pkl"


def _resolve_path(value, default, project_root):
    raw_value = default if value is None or str(value).strip() == "" else value
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = Path(project_root) / path
    return str(path.resolve())


def normalize_config_paths(data_d, run_d, project_root=PROJECT_ROOT):
    """Return copies with all runtime paths made absolute."""
    data_d = deepcopy(data_d)
    run_d = deepcopy(run_d)
    project_root = Path(project_root).resolve()

    output_dir = _resolve_path(
        run_d.get("base_dir") or data_d.get("base_dir"),
        "outputs",
        project_root,
    )
    data_d["proc_dp"] = _resolve_path(
        data_d.get("proc_dp"), "data/proc", project_root
    )
    data_d["base_dir"] = _resolve_path(
        data_d.get("base_dir"), output_dir, project_root
    )
    run_d["base_dir"] = _resolve_path(
        run_d.get("base_dir"), output_dir, project_root
    )
    if run_d.get("resume_dir"):
        run_d["resume_dir"] = _resolve_path(
            run_d["resume_dir"], run_d["resume_dir"], project_root
        )
    else:
        run_d["resume_dir"] = None
    return data_d, run_d


def _require_keys(section_name, section, required_keys):
    for key in required_keys:
        if key not in section:
            raise ValueError(
                f"missing required configuration key: {section_name}.{key}"
            )


def validate_training_runtime(model_d, run_d):
    """Reject checkpoint inheritance for protected joint training runs."""
    prediction_head = normalize_prediction_head(
        model_d.get("prediction_head", "dense")
    )
    if (
        prediction_head in {"plain_peak", "hipeak"}
        and int(run_d["num_epochs"]) > 0
        and bool(model_d.get("protect_baseline", False))
        and run_d.get("pretrained")
    ):
        raise ValueError(
            "run.pretrained must be empty for protected joint training"
        )


def validate_config(data_d, model_d, run_d):
    """Validate the minimum contract required by the released pipeline."""
    _require_keys(
        "data",
        data_d,
        ("proc_dp", "mz_bin_res", "mz_max", "nist_version"),
    )
    _require_keys("model", model_d, ("model_seed", "embed_types"))
    _require_keys(
        "run",
        run_d,
        (
            "device",
            "batch_size",
            "grad_acc_interval",
            "num_epochs",
            "do_casmi22",
            "pretrained",
        ),
    )

    if float(data_d["mz_bin_res"]) <= 0:
        raise ValueError("data.mz_bin_res must be greater than zero")
    if float(data_d["mz_max"]) <= 0:
        raise ValueError("data.mz_max must be greater than zero")
    processed_data_filenames(data_d)
    batch_size = int(run_d["batch_size"])
    grad_acc_interval = int(run_d["grad_acc_interval"])
    num_epochs = int(run_d["num_epochs"])
    if batch_size <= 0:
        raise ValueError("run.batch_size must be greater than zero")
    if grad_acc_interval <= 0:
        raise ValueError("run.grad_acc_interval must be greater than zero")
    if batch_size % grad_acc_interval != 0:
        raise ValueError(
            "run.batch_size must be divisible by run.grad_acc_interval"
        )
    if num_epochs > 0 and batch_size // grad_acc_interval < 2:
        raise ValueError(
            "training microbatch must contain at least two examples "
            "because the prediction head uses BatchNorm"
        )
    if num_epochs < 0:
        raise ValueError("run.num_epochs cannot be negative")
    required_sim_metrics = {
        "cosine",
        "modified_cosine",
        "spectral_entropy",
    }
    sim_metrics = run_d.get(
        "sim_metrics",
        sorted(required_sim_metrics),
    )
    if not isinstance(sim_metrics, (list, tuple)):
        raise ValueError("run.sim_metrics must be a list")
    missing_sim_metrics = required_sim_metrics - set(sim_metrics)
    if missing_sim_metrics:
        raise ValueError(
            "run.sim_metrics is incomplete; missing: "
            + ", ".join(sorted(missing_sim_metrics))
        )
    modified_cosine_tolerance = float(
        run_d.get("modified_cosine_tolerance", 0.1)
    )
    if modified_cosine_tolerance < 0:
        raise ValueError(
            "run.modified_cosine_tolerance cannot be negative"
        )
    sim_noise_cutoff = float(run_d.get("sim_noise_cutoff", 0.01))
    if not 0 <= sim_noise_cutoff < 1:
        raise ValueError(
            "run.sim_noise_cutoff must be in the range [0, 1)"
        )
    if not model_d["embed_types"]:
        raise ValueError("model.embed_types cannot be empty")
    prediction_head = normalize_prediction_head(
        model_d.get("prediction_head", "dense")
    )
    valid_prediction_heads = {"dense", "plain_peak", "hipeak"}
    if prediction_head not in valid_prediction_heads:
        raise ValueError(
            "model.prediction_head must be one of: "
            + ", ".join(sorted(valid_prediction_heads | {"frahi"}))
        )
    if prediction_head in {"plain_peak", "hipeak"}:
        if bool(data_d.get("merge_collision_energies", True)):
            raise ValueError(
                "peak prediction heads require "
                "data.merge_collision_energies=false"
            )
        if "peak_target_max" not in data_d:
            raise ValueError(
                "data.peak_target_max is required for peak prediction heads"
            )
        if not bool(data_d.get("require_fragment_candidates", False)):
            raise ValueError(
                "data.require_fragment_candidates must be true for "
                "peak prediction heads"
            )
        target_capacity = int(data_d["peak_target_max"])
        if target_capacity <= 0:
            raise ValueError(
                "data.peak_target_max must be positive"
            )
        hydrogen_shifts = data_d.get("fragment_hydrogen_shifts")
        if (
            not isinstance(hydrogen_shifts, (list, tuple))
            or not hydrogen_shifts
            or 0 not in [int(value) for value in hydrogen_shifts]
        ):
            raise ValueError(
                "data.fragment_hydrogen_shifts must be a non-empty "
                "list containing zero"
            )
        if int(data_d.get("fragment_max_candidates", 0)) <= 0:
            raise ValueError(
                "data.fragment_max_candidates must be positive"
            )
        if model_d["embed_types"] != ["hdse"]:
            raise ValueError(
                "peak prediction heads currently require embed_types=['hdse']"
            )
        if not bool(model_d.get("protect_baseline", False)):
            raise ValueError(
                "model.protect_baseline must be true for joint peak training"
            )
        if "anchor_source_head" in model_d:
            raise ValueError(
                "model.anchor_source_head is not allowed for joint training"
            )
        if "freeze_anchor" in run_d:
            raise ValueError(
                "run.freeze_anchor is not allowed for joint training"
            )
        baseline_loss_weight = float(
            run_d.get("baseline_loss_weight", 0.0)
        )
        if baseline_loss_weight != 1.0:
            raise ValueError(
                "run.baseline_loss_weight must equal 1.0 so the online "
                "baseline follows the standalone Dense training trajectory"
            )
        if not bool(run_d.get("validation_fallback", False)):
            raise ValueError(
                "run.validation_fallback must be true for joint training"
            )
        baseline_clip_grad_norm = float(
            run_d.get("baseline_clip_grad_norm", 0.0)
        )
        if baseline_clip_grad_norm <= 0.0:
            raise ValueError(
                "run.baseline_clip_grad_norm must be greater than zero"
            )
        if run_d.get("scheduler") != "plateau":
            raise ValueError(
                "joint training requires the baseline plateau scheduler"
            )
        if run_d.get("scheduler_monitor") != "baseline":
            raise ValueError(
                "run.scheduler_monitor must be baseline for joint training"
            )
        branch_metric_sets = run_d.get(
            "branch_metric_sets", ["val", "test"]
        )
        if not isinstance(branch_metric_sets, (list, tuple)):
            raise ValueError("run.branch_metric_sets must be a list")
        missing_branch_sets = {"val", "test"} - set(branch_metric_sets)
        if missing_branch_sets:
            raise ValueError(
                "run.branch_metric_sets must include val and test"
            )
        non_degradation_weight = float(
            run_d.get("non_degradation_weight", 0.0)
        )
        if non_degradation_weight < 0:
            raise ValueError(
                "run.non_degradation_weight cannot be negative"
            )
        decoder_layers = int(model_d.get("peak_decoder_layers", 1))
        if decoder_layers <= 0:
            raise ValueError(
                "peak_decoder_layers must be positive"
            )
        local_residual_log_ratio_max = float(
            model_d.get("local_residual_log_ratio_max", 0.0)
        )
        if not 0.0 < local_residual_log_ratio_max <= 2.0:
            raise ValueError(
                "model.local_residual_log_ratio_max must be in the "
                "range (0, 2]"
            )
        if prediction_head == "hipeak":
            hierarchical_residual_max = float(
                model_d.get(
                    "hierarchical_residual_log_ratio_max", 0.0
                )
            )
            if not (
                    0.0
                    < hierarchical_residual_max
                    <= local_residual_log_ratio_max):
                raise ValueError(
                    "model.hierarchical_residual_log_ratio_max must "
                    "be in the range (0, local_residual_log_ratio_max]"
                )
            hierarchical_adapter_ratio = float(
                model_d.get("hierarchical_adapter_ratio", 0.0)
            )
            if not 0.0 < hierarchical_adapter_ratio <= 1.0:
                raise ValueError(
                    "model.hierarchical_adapter_ratio must be in the "
                    "range (0, 1]"
                )
        anchor_warmup_epochs = int(
            run_d.get("non_degradation_warmup_epochs", 0)
        )
        if anchor_warmup_epochs < 0:
            raise ValueError(
                "run.non_degradation_warmup_epochs cannot be "
                "negative"
            )
        fragment_match_tolerance = float(
            run_d.get("fragment_match_tolerance", 0.0)
        )
        if fragment_match_tolerance <= 0:
            raise ValueError(
                "run.fragment_match_tolerance must be positive"
            )
        renderer_gradient_sigmas = model_d.get(
            "renderer_gradient_sigmas_mz",
            [0.1, 0.5, 2.0, 5.0],
        )
        if (
            not isinstance(renderer_gradient_sigmas, (list, tuple))
            or not renderer_gradient_sigmas
            or any(
                float(value) <= 0.0
                for value in renderer_gradient_sigmas
            )
        ):
            raise ValueError(
                "model.renderer_gradient_sigmas_mz must contain "
                "positive values"
            )
    if (
        num_epochs == 0
        and bool(run_d["do_casmi22"])
        and not run_d["pretrained"]
    ):
        raise ValueError(
            "run.pretrained is required when num_epochs is 0 and "
            "CASMI2022 evaluation is enabled"
        )


def apply_runtime_overrides(data_d, run_d, data_dir=None, output_dir=None):
    """Apply CLI path overrides without mutating the caller's dictionaries."""
    data_d = deepcopy(data_d)
    run_d = deepcopy(run_d)
    if data_dir is not None:
        data_d["proc_dp"] = data_dir
    if output_dir is not None:
        data_d["base_dir"] = output_dir
        run_d["base_dir"] = output_dir
    return data_d, run_d
