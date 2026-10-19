"""Spectrum-row protocols shared by FraHi and its baseline controls."""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


MERGE_GROUP_COLUMNS = [
    "mol_id",
    "prec_type",
    "inst_type",
    "frag_mode",
    "spec_type",
    "ion_mode",
    "dset",
    "col_gas",
]

MOLECULE_GROUP_COLUMNS = [
    "mol_id",
    "prec_type",
    "inst_type",
    "frag_mode",
    "spec_type",
    "ion_mode",
    "dset",
]


def _merge_peak_list(peaks, tolerance=1e-4):
    """Merge near-identical masses using the released mean aggregation."""
    ordered = sorted(peaks, key=lambda item: item[0])
    if not ordered:
        raise ValueError("a spectrum must contain at least one peak")
    merged = []
    current = [ordered[0]]
    for peak in ordered[1:]:
        if abs(peak[0] - current[-1][0]) <= tolerance:
            current.append(peak)
        else:
            merged.append(
                (
                    float(np.mean([item[0] for item in current])),
                    float(np.mean([item[1] for item in current])),
                )
            )
            current = [peak]
    merged.append(
        (
            float(np.mean([item[0] for item in current])),
            float(np.mean([item[1] for item in current])),
        )
    )
    return merged


def merge_collision_energy_rows(spec_df):
    """Reproduce the released molecule-level collision-energy merge."""
    merged_rows = []
    for values, group in spec_df.groupby(
            MERGE_GROUP_COLUMNS, sort=True, dropna=False):
        peaks = [
            peak
            for spectrum_peaks in group["peaks"]
            for peak in spectrum_peaks
        ]
        row = dict(zip(MERGE_GROUP_COLUMNS, values))
        row.update(
            {
                "peaks": _merge_peak_list(peaks),
                "prec_mz": float(group["prec_mz"].mean()),
                "ace": float(group["ace"].mean()),
                "nce": float(group["nce"].max()),
                "ri": np.nan,
                "res": int(group["res"].min()),
            }
        )
        merged_rows.append(row)
    return pd.DataFrame(merged_rows)


def assign_spectrum_group_ids(spec_df):
    """Assign one group ID per molecular condition, independent of NCE."""
    result = spec_df.copy(deep=True).reset_index(drop=True)
    group_frame = result[MOLECULE_GROUP_COLUMNS].drop_duplicates()
    group_frame = group_frame.reset_index(drop=True)
    group_frame["group_id"] = np.arange(len(group_frame), dtype=np.int64)
    result = result.merge(
        group_frame,
        on=MOLECULE_GROUP_COLUMNS,
        how="left",
        validate="many_to_one",
        sort=False,
    )
    result["spec_id"] = np.arange(len(result), dtype=np.int64)
    return result


def prepare_spectrum_rows(spec_df, merge_collision_energies):
    """Apply either the released merge or the CE-resolved main protocol."""
    if not isinstance(merge_collision_energies, (bool, np.bool_)):
        raise ValueError("merge_collision_energies must be boolean")
    rows = spec_df.copy(deep=True)
    if merge_collision_energies:
        rows = merge_collision_energy_rows(rows)
    return assign_spectrum_group_ids(rows)


def _transform_peak_intensities(intensity, transform, normalization):
    intensity = intensity.float()
    if intensity.numel() == 0:
        return intensity
    intensity = intensity / intensity.max().clamp_min(1e-12) * 1000.0
    if transform == "log10":
        intensity = torch.log10(intensity + 1.0)
    elif transform == "log10over3":
        intensity = torch.log10(intensity + 1.0) / 3.0
    elif transform == "log10over4":
        intensity = torch.log10(intensity + 1.0) / 4.0
    elif transform == "loge":
        intensity = torch.log(intensity + 1.0)
    elif transform == "sqrt":
        intensity = torch.sqrt(intensity)
    elif transform == "none":
        pass
    else:
        raise ValueError(f"invalid transform: {transform}")
    if normalization == "l1":
        intensity = F.normalize(intensity, p=1, dim=0)
    elif normalization == "l2":
        intensity = F.normalize(intensity, p=2, dim=0)
    elif normalization == "none":
        pass
    else:
        raise ValueError(f"invalid normalization: {normalization}")
    return intensity


def build_peak_targets(peaks, max_peaks, transform, normalization):
    """Return strongest-first, fixed-capacity continuous peak targets."""
    if int(max_peaks) <= 0:
        raise ValueError("max_peaks must be greater than zero")
    max_peaks = int(max_peaks)
    selected = sorted(peaks, key=lambda item: item[1], reverse=True)[:max_peaks]
    mz = torch.zeros(max_peaks, dtype=torch.float32)
    intensity = torch.zeros(max_peaks, dtype=torch.float32)
    mask = torch.zeros(max_peaks, dtype=torch.bool)
    if selected:
        count = len(selected)
        mz[:count] = torch.tensor(
            [item[0] for item in selected], dtype=torch.float32
        )
        raw_intensity = torch.tensor(
            [item[1] for item in selected], dtype=torch.float32
        )
        intensity[:count] = _transform_peak_intensities(
            raw_intensity,
            transform,
            normalization,
        )
        mask[:count] = True
    return {"mz": mz, "intensity": intensity, "mask": mask}
