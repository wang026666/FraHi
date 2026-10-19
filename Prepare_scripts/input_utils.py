"""Input discovery helpers for spectrum preprocessing scripts."""

from pathlib import Path


SUPPORTED_MSP_SUFFIXES = {".msp", ".mspec"}


def normalize_casmi_ion_mode(value):
    """Convert CASMI POSITIVE/NEGATIVE labels to P/N, ignoring space and case."""
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in {"POSITIVE", "NEGATIVE"}:
            return {"POSITIVE": "P", "NEGATIVE": "N"}[normalized]
    raise ValueError(
        "CASMI ION_MODE must be POSITIVE or NEGATIVE "
        f"(case-insensitive, surrounding whitespace allowed); got {value!r}"
    )


def resolve_msp_files(inputs):
    """Resolve file and directory inputs into a sorted, unique MSP file list."""
    if not inputs:
        raise ValueError("at least one MSP file or directory is required")

    resolved = []
    for raw_input in inputs:
        path = Path(raw_input).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"MSP input does not exist: {raw_input}")
        if path.is_file():
            if path.suffix.lower() not in SUPPORTED_MSP_SUFFIXES:
                raise ValueError(f"unsupported MSP file extension: {path}")
            resolved.append(path.resolve())
        elif path.is_dir():
            resolved.extend(
                candidate.resolve()
                for candidate in path.rglob("*")
                if candidate.is_file()
                and candidate.suffix.lower() in SUPPORTED_MSP_SUFFIXES
            )
        else:
            raise ValueError(f"MSP input is neither a file nor directory: {path}")

    unique = sorted(set(resolved))
    if not unique:
        raise FileNotFoundError("no .msp or .MSPEC files were found in the inputs")
    return unique
