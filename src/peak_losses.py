"""Hungarian set supervision for continuous mass-spectrum peaks."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


DEFAULT_PEAK_LOSS_WEIGHTS = {
    "spectrum": 1.0,
    "existence": 0.2,
    "mass": 1.0,
    "intensity": 1.0,
    "consistency": 0.1,
}


def non_degradation_penalty(
        candidate_loss, anchor_loss, reduction="none"):
    """Penalize per-sample degradation before applying a reduction."""
    if not isinstance(candidate_loss, torch.Tensor):
        raise TypeError("candidate_loss must be a torch.Tensor")
    anchor_loss = torch.as_tensor(
        anchor_loss,
        dtype=candidate_loss.dtype,
        device=candidate_loss.device,
    ).detach()
    penalty = F.relu(candidate_loss - anchor_loss)
    if reduction == "none":
        return penalty
    if reduction == "mean":
        return penalty.mean()
    if reduction == "sum":
        return penalty.sum()
    raise ValueError(f"invalid reduction: {reduction}")


def combine_protected_training_losses(
        full_loss,
        baseline_loss,
        full_per_sample_loss,
        baseline_per_sample_loss,
        baseline_weight,
        non_degradation_weight):
    """Build the joint objective without coupling the two parameter paths."""
    baseline_weight = float(baseline_weight)
    non_degradation_weight = float(non_degradation_weight)
    if baseline_weight <= 0.0:
        raise ValueError("baseline_weight must be greater than zero")
    if non_degradation_weight < 0.0:
        raise ValueError("non_degradation_weight cannot be negative")
    non_degradation = non_degradation_penalty(
        full_per_sample_loss,
        baseline_per_sample_loss,
        reduction="mean",
    )
    total_loss = (
        full_loss
        + baseline_weight * baseline_loss
        + non_degradation_weight * non_degradation
    )
    return total_loss, {
        "baseline_spectrum_loss": baseline_loss.detach(),
        "non_degradation_loss": non_degradation.detach(),
        "baseline_loss_weight": baseline_loss.detach().new_tensor(
            baseline_weight
        ),
        "non_degradation_weight": baseline_loss.detach().new_tensor(
            non_degradation_weight
        ),
        "full_worse_fraction": (
            full_per_sample_loss > baseline_per_sample_loss.detach()
        ).float().mean().detach(),
    }


def _assignment(predicted_mz, predicted_intensity, target_mz, target_intensity,
                mz_max):
    mass_cost = torch.abs(
        predicted_mz.unsqueeze(1) - target_mz.unsqueeze(0)
    ) / float(mz_max)
    intensity_cost = torch.abs(
        predicted_intensity.unsqueeze(1) - target_intensity.unsqueeze(0)
    )
    detached_cost = (mass_cost + intensity_cost).detach().cpu().numpy()
    row, column = linear_sum_assignment(detached_cost)
    return (
        torch.as_tensor(row, dtype=torch.long, device=predicted_mz.device),
        torch.as_tensor(column, dtype=torch.long, device=predicted_mz.device),
    )


def compute_peak_training_loss(
        output,
        batch,
        spectrum_loss,
        weights=None,
        mz_max=500.0,
        fragment_match_tolerance=0.2):
    """Combine rendered-spectrum loss with permutation-invariant peak loss."""
    weights = {
        **DEFAULT_PEAK_LOSS_WEIGHTS,
        **({} if weights is None else weights),
    }
    predicted_mz = output["peak_mz"]
    predicted_intensity = output["peak_intensity"]
    exist_logits = output["peak_exist_logits"]
    target_mz = batch["peak_target_mz"]
    target_intensity = batch["peak_target_intensity"]
    target_mask = batch["peak_target_mask"].bool()
    precursor_mz = batch["prec_mz_value"].float()
    candidate_mask = output.get(
        "peak_candidate_mask",
        torch.ones_like(predicted_mz, dtype=torch.bool),
    ).bool()
    fragment_constrained = bool(output.get("fragment_constrained", False))
    if predicted_mz.ndim != 2:
        raise ValueError("predicted peaks must have shape [batch, queries]")
    if target_mz.shape[0] != predicted_mz.shape[0]:
        raise ValueError(
            "peak targets and predictions must have the same batch size"
        )
    if candidate_mask.shape != predicted_mz.shape:
        raise ValueError("peak_candidate_mask must match predicted peaks")

    differentiable_zero = (
        predicted_intensity.sum() + exist_logits.sum()
    ) * 0.0
    existence_losses = []
    mass_losses = []
    intensity_losses = []
    consistency_losses = []
    for batch_index in range(predicted_mz.shape[0]):
        valid_target_mz = target_mz[batch_index][target_mask[batch_index]]
        valid_target_intensity = target_intensity[batch_index][
            target_mask[batch_index]
        ]
        valid_predicted_index = torch.nonzero(
            candidate_mask[batch_index], as_tuple=False
        ).flatten()
        valid_predicted_mz = predicted_mz[
            batch_index, valid_predicted_index
        ]
        valid_predicted_intensity = predicted_intensity[
            batch_index, valid_predicted_index
        ]
        valid_exist_logits = exist_logits[
            batch_index, valid_predicted_index
        ]
        existence_target = torch.zeros_like(valid_exist_logits)
        if (
            valid_target_mz.numel() == 0
            or valid_predicted_index.numel() == 0
        ):
            predicted_index = valid_predicted_index.new_empty((0,))
            target_index = valid_predicted_index.new_empty((0,))
            mass_loss = differentiable_zero
            intensity_loss = differentiable_zero
            consistency_loss = differentiable_zero
        else:
            if fragment_constrained:
                mass_cost = torch.abs(
                    valid_predicted_mz.unsqueeze(1)
                    - valid_target_mz.unsqueeze(0)
                )
                nearest_distance, nearest_target = torch.min(
                    mass_cost, dim=1
                )
                local_predicted_index = torch.nonzero(
                    nearest_distance <= float(fragment_match_tolerance),
                    as_tuple=False,
                ).flatten()
                target_index = nearest_target[local_predicted_index]
            else:
                local_predicted_index, target_index = _assignment(
                    valid_predicted_mz,
                    valid_predicted_intensity,
                    valid_target_mz,
                    valid_target_intensity,
                    mz_max=mz_max,
                )
            predicted_index = valid_predicted_index[
                local_predicted_index
            ]
            existence_target[local_predicted_index] = 1.0
            if predicted_index.numel() == 0:
                mass_loss = differentiable_zero
                intensity_loss = differentiable_zero
                consistency_loss = differentiable_zero
            else:
                mass_loss = F.l1_loss(
                    predicted_mz[
                        batch_index, predicted_index
                    ] / float(mz_max),
                    valid_target_mz[target_index] / float(mz_max),
                )
                if fragment_constrained:
                    unique_target_index = torch.unique(target_index)
                    grouped_predicted_intensity = torch.stack(
                        [
                            predicted_intensity[
                                batch_index,
                                predicted_index[
                                    target_index == group_target
                                ],
                            ].sum()
                            for group_target in unique_target_index
                        ]
                    )
                    intensity_loss = F.l1_loss(
                        grouped_predicted_intensity,
                        valid_target_intensity[unique_target_index],
                    )
                    consistency_loss = differentiable_zero
                else:
                    intensity_loss = F.l1_loss(
                        predicted_intensity[
                            batch_index, predicted_index
                        ],
                        valid_target_intensity[target_index],
                    )
                    matched_target_mz = valid_target_mz[target_index]
                    below_precursor = (
                        matched_target_mz
                        <= precursor_mz[batch_index] + 1e-6
                    )
                    coordinate_terms = []
                    if below_precursor.any():
                        coordinate_terms.append(
                            torch.abs(
                                output["peak_mz_absolute"][
                                    batch_index,
                                    predicted_index[below_precursor],
                                ]
                                - output["peak_mz_neutral_loss"][
                                    batch_index,
                                    predicted_index[below_precursor],
                                ]
                            ) / float(mz_max)
                        )
                    above_precursor = ~below_precursor
                    if above_precursor.any():
                        coordinate_terms.append(
                            F.binary_cross_entropy(
                                output["peak_mass_gate"][
                                    batch_index,
                                    predicted_index[above_precursor],
                                ],
                                torch.ones_like(
                                    output["peak_mass_gate"][
                                        batch_index,
                                        predicted_index[above_precursor],
                                    ]
                                ),
                                reduction="none",
                            )
                        )
                    consistency_loss = torch.cat(
                        coordinate_terms
                    ).mean()
        if valid_exist_logits.numel() == 0:
            existence_loss = differentiable_zero
        else:
            existence_loss = F.binary_cross_entropy_with_logits(
                valid_exist_logits,
                existence_target,
            )
        existence_losses.append(existence_loss)
        mass_losses.append(mass_loss)
        intensity_losses.append(intensity_loss)
        consistency_losses.append(consistency_loss)

    existence_loss = torch.stack(existence_losses).mean()
    mass_loss = torch.stack(mass_losses).mean()
    intensity_loss = torch.stack(intensity_losses).mean()
    consistency_loss = torch.stack(consistency_losses).mean()
    if not isinstance(spectrum_loss, torch.Tensor):
        spectrum_loss = predicted_mz.new_tensor(float(spectrum_loss))
    peak_loss = (
        float(weights["existence"]) * existence_loss
        + float(weights["mass"]) * mass_loss
        + float(weights["intensity"]) * intensity_loss
        + float(weights["consistency"]) * consistency_loss
    )
    total_loss = float(weights["spectrum"]) * spectrum_loss + peak_loss
    metrics = {
        "spectrum_loss": spectrum_loss.detach(),
        "peak_loss": peak_loss.detach(),
        "peak_existence_loss": existence_loss.detach(),
        "peak_mass_loss": mass_loss.detach(),
        "peak_intensity_loss": intensity_loss.detach(),
        "peak_consistency_loss": consistency_loss.detach(),
    }
    if "peak_log_ratio" in output:
        valid_log_ratio = output["peak_log_ratio"][candidate_mask]
        if valid_log_ratio.numel() == 0:
            valid_log_ratio = predicted_mz.new_zeros((1,))
        absolute_log_ratio = valid_log_ratio.abs()
        metrics.update(
            {
                "peak_log_ratio_abs_mean": (
                    absolute_log_ratio.mean().detach()
                ),
                "peak_log_ratio_abs_max": (
                    absolute_log_ratio.max().detach()
                ),
            }
        )
    if "peak_route_weights" in output:
        route_weights = output["peak_route_weights"]
        valid_route_weights = route_weights[candidate_mask]
        if valid_route_weights.numel() == 0:
            valid_route_weights = route_weights.new_zeros((1, 3))
        route_mean = valid_route_weights.mean(dim=0)
        route_entropy = -(
            valid_route_weights.clamp_min(1e-12)
            * valid_route_weights.clamp_min(1e-12).log()
        ).sum(dim=-1).mean()
        metrics.update(
            {
                "peak_route_atom": route_mean[0].detach(),
                "peak_route_community": route_mean[1].detach(),
                "peak_route_graph": route_mean[2].detach(),
                "peak_route_entropy": route_entropy.detach(),
            }
        )
    return total_loss, metrics
