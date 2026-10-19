import numpy as np
import os
import pandas as pd
import time
from tqdm import tqdm
import argparse
import glob
import requests
import json
import re

from input_utils import resolve_msp_files

key_dict = {
    "Precursor_type": "prec_type",
    "Spectrum_type": "spec_type",
    "PrecursorMZ": "prec_mz",
    "Instrument_type": "inst_type",
    "Collision_energy": "col_energy",
    "Ion_mode": "ion_mode",
    "Ionization": "ion_type",
    "ID": "spec_id",
    "Collision_gas": "col_gas",
    "Pressure": "pressure",
    "Num peaks": "num_peaks",
    "MW": "mw",
    "ExactMass": "exact_mass",
    "CAS#": "cas_num",
    "NIST#": "nist_num",
    "Name": "name",
    "MS": "peaks",
    "SMILES": "smiles",
    "Rating": "rating",
    "Frag_mode": "frag_mode",
    "Instrument": "inst",
    "RI": "ri",
    "InChIKey": "inchi_key"
}

KEY_ALIASES = {
    "CASNO": "CAS#",
    "NISTNO": "NIST#",
    "Num Peaks": "Num peaks",
}


def inchi_key_to_smiles(inchi_key, local_mapping_file="", delay=1, missing_keys=None):
    """
    Query SMILES for a given InChIKey, first from a local file, then from the PubChem API if not found.
    :param inchi_key: str, InChIKey
    :param local_mapping_file: str, path to local InChIKey-SMILES JSON mapping
    :param delay: float, delay between API requests in seconds (default: 1)
    :param missing_keys: set, collect InChIKeys not found locally
    :return: str or None, SMILES or None if failed
    """
    if os.path.exists(local_mapping_file):
        try:
            with open(local_mapping_file, "r") as f:
                local_mapping = json.load(f)
            if inchi_key in local_mapping:
                return local_mapping[inchi_key]
        except Exception as e:
            print(f"Error reading local mapping file: {e}")

    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchi_key}/property/CanonicalSMILES/JSON"
    try:
        response = requests.get(url, timeout=60)
        time.sleep(delay)
        if response.status_code == 200:
            data = response.json()
            return data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
        else:
            print(f"HTTP Error {response.status_code} for InChIKey {inchi_key}: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"RequestException for InChIKey {inchi_key}: {e}")
        return None

def inchi_to_smiles(inchi_keys):
    """
    Sequentially convert a list of InChIKeys to SMILES, count failures.
    :param inchi_keys: list of InChIKeys
    :return: tuple, (smiles_list, failed_count)
    """
    failed_count = 0
    smiles_list = []
    for inchi_key in tqdm(inchi_keys, desc="Processing InChIKey to SMILES"):
        try:
            smiles = inchi_key_to_smiles(inchi_key)
            if smiles is None:
                failed_count += 1
            smiles_list.append(smiles)
        except Exception:
            failed_count += 1
            smiles_list.append(None)
    return smiles_list, failed_count

def extract_info_from_comments(comments, key):
    match = re.search(
        rf"(?:^|\s){re.escape(key)}\s*=\s*\"([^\"]*)\"",
        comments,
    )
    return None if match is None else match.group(1)

def validate_peaks(peaks_str, line_number):
    """
    Validate peaks data format.
    :param peaks_str: string of 'm/z intensity' peaks
    :param line_number: line number in source file (for debug)
    :return: True if valid, False otherwise
    """
    if not peaks_str.strip():
        print(f"Empty peaks data at line {line_number}")
        return False
    for line in peaks_str.strip().split('\n'):
        parts = line.split()
        try:
            if len(parts) < 2:
                raise ValueError("peak row must contain m/z and intensity")
            m_z, intensity = float(parts[0]), float(parts[1])
            if m_z < 0 or intensity < 0:
                print(f"Negative value in peaks at line {line_number}: {line}")
                return False
        except (ValueError, IndexError):
            print(f"Non-numeric peak data at line {line_number}: {line}")
            return False
    return True

def preproc_msp(msp_inputs, keys, num_entries):
    """
    Parse MSP files from multiple folders into a DataFrame.
    :param msp_dirs: list of directories containing MSP files
    :param keys: list of field names to extract
    :param num_entries: max number of entries to read, -1 for all
    """
    msp_files = resolve_msp_files(msp_inputs)
    print(f"Found {len(msp_files)} MSP files to process.")

    raw_data_list = []
    sum_invalid = 0
    total_entries = 0

    def finalize_record(raw_data_item, line_number, msp_fp):
        nonlocal sum_invalid, total_entries
        if not any(value is not None for value in raw_data_item.values()):
            return
        peaks = raw_data_item.get("MS")
        peak_count = len(peaks.strip().splitlines()) if peaks else 0
        declared_peak_count = raw_data_item.get("Num peaks")
        peak_count_matches = (
            declared_peak_count is None
            or int(declared_peak_count) == peak_count
        )
        if (
            not peaks
            or not peak_count_matches
            or not validate_peaks(peaks, line_number)
        ):
            sum_invalid += 1
            print(f"Invalid peaks data at line {line_number} in file {msp_fp}")
        else:
            raw_data_list.append(raw_data_item.copy())
        total_entries += 1

    for msp_fp in tqdm(msp_files, desc="Processing MSP files"):
        if num_entries > -1 and total_entries >= num_entries:
            break
        raw_data_item = {key: None for key in keys}
        read_ms = False
        line_number = 0

        with open(msp_fp, encoding="utf-8", errors="replace") as f:
            for raw_l in f:
                if num_entries > -1 and total_entries >= num_entries:
                    break
                raw_l = raw_l.rstrip("\r\n")
                line_number += 1
                if raw_l == '':
                    finalize_record(raw_data_item, line_number, msp_fp)
                    raw_data_item = {key: None for key in keys}
                    read_ms = False
                elif read_ms:
                    raw_data_item['MS'] = raw_data_item['MS'] + raw_l + '\n'
                else:
                    key, separator, value = raw_l.partition(":")
                    if not separator:
                        print(
                            f"Skipping unrecognized metadata at line "
                            f"{line_number}: {raw_l}"
                        )
                        continue
                    key = KEY_ALIASES.get(key.strip(), key.strip())
                    value = value.strip()
                    if key == "Num peaks":
                        raw_data_item['Num peaks'] = int(value)
                        raw_data_item['MS'] = ''
                        read_ms = True
                    elif key == "Comments":
                        comments = value
                        smiles = extract_info_from_comments(comments, "computed SMILES")
                        rating = extract_info_from_comments(comments, "MoNA Rating")
                        frag_mode = extract_info_from_comments(comments, "fragmentation mode")
                        if not (smiles is None):
                            raw_data_item["SMILES"] = smiles
                        if not (rating is None):
                            raw_data_item["Rating"] = rating
                        if not (frag_mode is None):
                            raw_data_item["Frag_mode"] = frag_mode
                    elif key == "CAS#":
                        cas_nist_data = value.split(";")
                        cas_value = cas_nist_data[0].strip()
                        raw_data_item["CAS#"] = cas_value
                        if len(cas_nist_data) > 1:
                            raw_data_item["NIST#"] = cas_nist_data[1].strip()
                    elif key in keys:
                        raw_data_item[key] = value
        if num_entries == -1 or total_entries < num_entries:
            finalize_record(raw_data_item, line_number, msp_fp)

    msp_df = pd.DataFrame(raw_data_list)
    if not msp_df.empty:
        msp_df = msp_df.dropna(axis=0, how="all")
    print(f"{sum_invalid} invalid spectra filtered")
    print(f"Total {len(raw_data_list)} valid entries processed")
    return msp_df

def resolve_nist_mol_files(mol_dp, spec_ids=None):
    """Resolve local NIST MOL files, optionally limiting them to MSP IDs."""
    mol_dp = os.fspath(mol_dp)
    if not os.path.isdir(mol_dp):
        raise NotADirectoryError(mol_dp)

    if spec_ids is None:
        mol_fp_list = sorted(glob.glob(os.path.join(mol_dp, "ID*.MOL")))
    else:
        normalized_ids = [str(spec_id).strip() for spec_id in spec_ids]
        mol_fp_list = [
            os.path.join(mol_dp, f"ID{spec_id}.MOL")
            for spec_id in normalized_ids
        ]
        missing = [path for path in mol_fp_list if not os.path.isfile(path)]
        if missing:
            print(
                f"Warning: {len(missing)} NIST spectra have no local MOL "
                "file and will be skipped"
            )
            missing_set = set(missing)
            mol_fp_list = [
                path for path in mol_fp_list if path not in missing_set
            ]

    if not mol_fp_list:
        raise FileNotFoundError(f"no ID*.MOL files found in {mol_dp}")
    return mol_fp_list


def preproc_nist_mol(mol_dp, spec_ids=None):
    """
    Read all .MOL files in a directory and return a DataFrame with spec_id and SMILES.
    """
    from data_utils import par_apply, rdkit_import

    mol_fp_list = resolve_nist_mol_files(mol_dp, spec_ids=spec_ids)

    def proc_mol_file(mol_fp):
        Chem, = rdkit_import("rdkit.Chem")
        mol_stem = os.path.splitext(os.path.basename(mol_fp))[0]
        spec_id = mol_stem[2:] if mol_stem.startswith("ID") else mol_stem
        mol = Chem.MolFromMolFile(mol_fp, sanitize=True)
        if not (mol is None):
            smiles = Chem.MolToSmiles(mol)
        else:
            smiles = None
        entry = dict(
            spec_id=spec_id,
            smiles=smiles
        )
        return entry
    mol_df_entries = par_apply(mol_fp_list, proc_mol_file)
    mol_df = pd.DataFrame(mol_df_entries)
    return mol_df

def merge_and_check(msp_df, mol_df, rename_dict):
    """
    Clean and merge msp_df with mol_df, validate completeness.
    :param msp_df: DataFrame, raw msp data
    :param mol_df: DataFrame, optional, molecule data
    :param rename_dict: dict, column rename mapping
    :return: DataFrame, processed spec_df
    """
    msp_bad_cols = set(msp_df.columns) - set(rename_dict.keys())
    msp_df = msp_df.drop(columns=msp_bad_cols)
    msp_df = msp_df.rename(columns=rename_dict)

    if mol_df is None:
        assert not msp_df["smiles"].isna().all(), "All values in smiles column are null!"
        if (
            msp_df["spec_id"].isna().any()
            or not msp_df["spec_id"].is_unique
        ):
            msp_df.loc[:, "spec_id"] = np.arange(msp_df.shape[0])
        spec_df = msp_df
    else:
        required_mol_columns = {"spec_id", "smiles"}
        missing_mol_columns = required_mol_columns - set(mol_df.columns)
        if missing_mol_columns:
            raise ValueError(
                "molecule table is missing required columns: "
                f"{sorted(missing_mol_columns)}"
            )
        if msp_df["spec_id"].isna().any():
            raise ValueError("NIST MSP contains records without ID")

        msp_df = msp_df.copy()
        mol_df = mol_df.loc[:, ["spec_id", "smiles"]].copy()
        msp_df["spec_id"] = msp_df["spec_id"].astype(str).str.strip()
        mol_df["spec_id"] = mol_df["spec_id"].astype(str).str.strip()
        mol_df = mol_df.dropna(subset=["smiles"])
        if mol_df["spec_id"].duplicated().any():
            raise ValueError("NIST MOL directory contains duplicate IDs")

        msp_df = msp_df.drop(columns=["smiles"], errors="ignore")
        spec_df = msp_df.merge(
            mol_df,
            how="left",
            on="spec_id",
            validate="one_to_one",
        )
        missing_smiles = int(spec_df["smiles"].isna().sum())
        if missing_smiles:
            print(
                f"Warning: dropping {missing_smiles} NIST spectra without "
                "a valid local molecular structure"
            )
            spec_df = spec_df.dropna(subset=["smiles"])
        if spec_df.empty:
            raise ValueError("no NIST spectra have a valid local MOL structure")

    print("Missing values per column:")
    print(spec_df.isna().sum())
    spec_df = spec_df.reset_index(drop=True)
    return spec_df

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--msp_file',
        '--msp-file',
        dest='msp_files',
        nargs='+',
        required=True,
        help='one or more .msp/.MSPEC files or directories',
    )
    parser.add_argument(
        '--mol_dir',
        type=str,
        help='directory containing NIST ID*.MOL files',
    )
    parser.add_argument(
        '--output_name',
        type=str,
        default='nist_20_hr_records_df',
    )
    parser.add_argument('--output_dp', type=str, default='data/df')
    parser.add_argument('--num_entries', type=int, default=-1)
    parser.add_argument(
        '--output_type',
        type=str,
        default="json",
        choices=[
            "json",
            "csv"])
    args = parser.parse_args()
    os.makedirs(args.output_dp, exist_ok=True)

    msp_df = preproc_msp(args.msp_files, key_dict.keys(), args.num_entries)
    mol_df = (
        preproc_nist_mol(args.mol_dir, spec_ids=msp_df["ID"])
        if args.mol_dir
        else None
    )
    spec_df = merge_and_check(msp_df, mol_df, key_dict)
    spec_df_fp = os.path.join(args.output_dp,
                              f"{args.output_name}.{args.output_type}")
    if args.output_type == "json":
        spec_df.to_json(spec_df_fp)
    else:
        assert args.output_type == "csv"
        spec_df.to_csv(spec_df_fp, index=False)
