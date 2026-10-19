"""Pure helpers for optimizer and scheduler step accounting."""

import math
from datetime import datetime


def _validate_counts(num_batches, grad_acc_interval):
    if num_batches < 0:
        raise ValueError("num_batches cannot be negative")
    if grad_acc_interval <= 0:
        raise ValueError("grad_acc_interval must be greater than zero")


def should_optimizer_step(batch_index, num_batches, grad_acc_interval):
    """Return True at each full accumulation window and at the final batch."""
    _validate_counts(num_batches, grad_acc_interval)
    if not 0 <= batch_index < num_batches:
        raise ValueError("batch_index must identify a batch in the epoch")
    completed_batches = batch_index + 1
    return (
        completed_batches % grad_acc_interval == 0
        or completed_batches == num_batches
    )


def accumulation_window_size(batch_index, num_batches, grad_acc_interval):
    """Return the divisor for the accumulation window containing a batch."""
    _validate_counts(num_batches, grad_acc_interval)
    if not 0 <= batch_index < num_batches:
        raise ValueError("batch_index must identify a batch in the epoch")
    window_start = (batch_index // grad_acc_interval) * grad_acc_interval
    return min(grad_acc_interval, num_batches - window_start)


def num_optimizer_updates(num_batches, grad_acc_interval):
    """Return optimizer updates per epoch, including a partial final window."""
    _validate_counts(num_batches, grad_acc_interval)
    if num_batches == 0:
        return 0
    return math.ceil(num_batches / grad_acc_interval)


def scheduled_non_degradation_weight(
        epoch, target_weight, warmup_epochs):
    """Linearly introduce the anchor penalty after residual learning starts."""
    epoch = int(epoch)
    target_weight = float(target_weight)
    warmup_epochs = int(warmup_epochs)
    if epoch < 0:
        raise ValueError("epoch cannot be negative")
    if target_weight < 0.0:
        raise ValueError("target_weight cannot be negative")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs cannot be negative")
    if warmup_epochs == 0:
        return target_weight
    return target_weight * min(1.0, epoch / warmup_epochs)


def select_protected_validation_metrics(
        full_metrics, baseline_metrics, stop_key):
    """Choose the lower-loss validation output and preserve both reports."""
    if stop_key not in full_metrics:
        raise KeyError(f"full metrics are missing stop key: {stop_key}")
    if stop_key not in baseline_metrics:
        raise KeyError(f"baseline metrics are missing stop key: {stop_key}")
    use_baseline = baseline_metrics[stop_key] < full_metrics[stop_key]
    source = "baseline" if use_baseline else "full"
    selected_metrics = baseline_metrics if use_baseline else full_metrics
    return {
        **selected_metrics,
        **{f"full_{key}": value for key, value in full_metrics.items()},
        **{
            f"baseline_{key}": value
            for key, value in baseline_metrics.items()
        },
        "prediction_source": source,
    }


def cosine_warmup_multiplier(
        update_index, total_updates, warmup_updates, end_ratio):
    """Return a linear-warmup then cosine-decay learning-rate factor."""
    update_index = int(update_index)
    total_updates = int(total_updates)
    warmup_updates = int(warmup_updates)
    end_ratio = float(end_ratio)
    if update_index < 0:
        raise ValueError("update_index cannot be negative")
    if total_updates <= 0:
        raise ValueError("total_updates must be positive")
    if not 0 <= warmup_updates < total_updates:
        raise ValueError(
            "warmup_updates must be in the range [0, total_updates)"
        )
    if not 0.0 <= end_ratio <= 1.0:
        raise ValueError("end_ratio must be in the range [0, 1]")
    update_index = min(update_index, total_updates - 1)
    if warmup_updates and update_index < warmup_updates:
        return (update_index + 1) / warmup_updates
    decay_intervals = max(1, total_updates - warmup_updates - 1)
    progress = min(
        1.0,
        (update_index - warmup_updates) / decay_intervals,
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return end_ratio + (1.0 - end_ratio) * cosine


def select_split_seeds(base_split_seed, num_splits):
    """Return the paper's registered split seeds without silent truncation."""
    if not 1 <= num_splits <= 5:
        raise ValueError("num_splits must be between 1 and 5")
    if num_splits == 1:
        return [int(base_split_seed)]
    return [520, 521, 522, 523, 524][:num_splits]


def build_run_directory_name(
        custom_name,
        split_seed,
        model_seed,
        train_seed,
        timestamp=None):
    """Build a collision-resistant, evidence-friendly run directory name."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (
        f"run_{custom_name}"
        f"_ss{int(split_seed)}"
        f"_ms{int(model_seed)}"
        f"_ts{int(train_seed)}"
        f"_{timestamp}"
    )
