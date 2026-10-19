import argparse
import os
from pathlib import Path

import yaml

from config_utils import (
    PROJECT_ROOT,
    apply_runtime_overrides,
    normalize_config_paths,
    validate_config,
    validate_training_runtime,
)
from training_utils import select_split_seeds


def load_config(
        custom_fp,
        device_id=None,
        checkpoint_name=None,
        data_dir=None,
        output_dir=None,
        resume_dir=None):
    assert custom_fp, "custom_fp cannot be empty"
    assert os.path.isfile(custom_fp), custom_fp

    with open(custom_fp, "r", encoding="utf-8") as custom_file:
        config_d = yaml.safe_load(custom_file)
    if not isinstance(config_d, dict):
        raise ValueError(f"configuration must be a mapping: {custom_fp}")

    run_name = config_d.get("run_name")
    if not run_name:
        run_name = os.path.splitext(os.path.basename(custom_fp))[0]
        config_d["run_name"] = run_name

    data_d = config_d["data"]
    model_d = config_d["model"]
    run_d = config_d["run"]

    if device_id is not None:
        if device_id < 0:
            run_d["device"] = "cpu"
        else:
            run_d["device"] = f"cuda:{device_id}"

    if checkpoint_name:
        checkpoint_path = Path(checkpoint_name).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path
        run_d["pretrained"] = str(checkpoint_path.resolve())
    if resume_dir:
        run_d["resume_dir"] = resume_dir

    data_d, run_d = apply_runtime_overrides(
        data_d,
        run_d,
        data_dir=data_dir,
        output_dir=output_dir,
    )
    data_d, run_d = normalize_config_paths(data_d, run_d)
    validate_config(data_d, model_d, run_d)

    custom_name = os.path.splitext(os.path.basename(custom_fp))[0]
    run_d["custom_name"] = custom_name

    return run_name, data_d, model_d, run_d

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device_id", type=int, required=False, help="device id (-1 for cpu)")
    parser.add_argument("-c", "--custom_fp", type=str, required=True, help="path to custom config file")
    parser.add_argument(
        "-k",
        "--checkpoint_name",
        type=str,
        required=False,
        help="path to a pretrained state_dict checkpoint",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="processed data directory (overrides data.proc_dp)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="run output directory (overrides base_dir)",
    )
    parser.add_argument(
        "--resume_dir",
        type=str,
        help="existing run directory containing chkpt.pkl to resume",
    )
    parser.add_argument("-n", "--num_splits", type=int, default=1, help="number of different split seeds to run")
    parser.add_argument(
        "--split_seed",
        type=int,
        help="run exactly one split with this seed; requires --num_splits 1",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="validate and print the resolved configuration without importing the model stack",
    )
    flags = parser.parse_args()
    if flags.resume_dir and flags.num_splits != 1:
        parser.error("--resume_dir can only be used with --num_splits 1")
    if flags.split_seed is not None and flags.num_splits != 1:
        parser.error("--split_seed can only be used with --num_splits 1")
    if flags.resume_dir and flags.checkpoint_name:
        parser.error(
            "--resume_dir and --checkpoint_name are mutually exclusive"
        )

    run_name, data_d, model_d, run_d = load_config(
        flags.custom_fp,
        flags.device_id,
        flags.checkpoint_name,
        data_dir=flags.data_dir,
        output_dir=flags.output_dir,
        resume_dir=flags.resume_dir,
    )
    validate_training_runtime(model_d, run_d)

    if flags.split_seed is not None:
        run_d["split_seed"] = flags.split_seed
    split_seeds = select_split_seeds(run_d["split_seed"], flags.num_splits)

    if flags.dry_run:
        resolved = {
            "run_name": run_name,
            "data": data_d,
            "model": model_d,
            "run": run_d,
            "split_seeds": split_seeds,
        }
        print(yaml.safe_dump(resolved, sort_keys=False))
        raise SystemExit(0)

    print(f"Will run training {len(split_seeds)} times, using the following split_seeds: {split_seeds}")

    from runner import train_and_eval

    for i, split_seed in enumerate(split_seeds):
        print(f"\n\n===== Running training {i+1}/{len(split_seeds)}, split_seed: {split_seed} =====\n")

        run_d["split_seed"] = split_seed

        train_and_eval(data_d, model_d, run_d)

        print(f"\n===== Finished training {i+1}/{len(split_seeds)} =====\n")
