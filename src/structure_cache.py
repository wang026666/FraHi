"""Reusable molecular HDSE graph caches for CE-resolved experiments."""

import os
import tempfile
from pathlib import Path

import torch


HDSE_GRAPH_CACHE_VERSION = 1


def resolve_hdse_graph_cache_path(proc_dp, cache_path):
    """Resolve a cache path relative to the processed-data directory."""
    if cache_path is None or str(cache_path).strip() == "":
        return None
    path = Path(cache_path).expanduser()
    if not path.is_absolute():
        path = Path(proc_dp) / path
    return path.resolve()


def save_hdse_graph_cache(
        path,
        graphs,
        metadata=None,
        overwrite=False):
    """Atomically save one preprocessed HDSE graph per molecule."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"HDSE graph cache already exists: {path}")
    if not graphs:
        raise ValueError("cannot save an empty HDSE graph cache")
    normalized_graphs = {int(key): value for key, value in graphs.items()}
    payload = {
        "format": "hipeak-hdse-graph-cache",
        "version": HDSE_GRAPH_CACHE_VERSION,
        "graphs": normalized_graphs,
        "metadata": {} if metadata is None else dict(metadata),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(file_descriptor)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return path


def load_hdse_graph_cache(
        path,
        required_mol_ids=None,
        required_graph_attributes=()):
    """Load and validate a graph cache before DataLoader workers are forked."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"HDSE graph cache is missing: {path}. "
            "Build it with scripts/build_hdse_graph_cache.py."
        )
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"invalid HDSE graph cache payload: {path}")
    if payload.get("format") != "hipeak-hdse-graph-cache":
        raise ValueError(f"invalid HDSE graph cache format: {path}")
    if payload.get("version") != HDSE_GRAPH_CACHE_VERSION:
        raise ValueError(
            f"unsupported HDSE graph cache version: "
            f"{payload.get('version')}"
        )
    graphs = payload.get("graphs")
    if not isinstance(graphs, dict) or not graphs:
        raise ValueError(f"HDSE graph cache contains no graphs: {path}")
    graphs = {int(key): value for key, value in graphs.items()}
    invalid_hierarchy = [
        mol_id
        for mol_id, graph in graphs.items()
        if not hasattr(graph, "hdse_community")
    ]
    if invalid_hierarchy:
        preview = ", ".join(
            str(item) for item in sorted(invalid_hierarchy)[:10]
        )
        raise ValueError(
            "HDSE graph cache entries are missing hdse_community, "
            f"including molecules: {preview}. Rebuild the cache."
        )
    for attribute in required_graph_attributes:
        missing_attribute = [
            mol_id
            for mol_id, graph in graphs.items()
            if not hasattr(graph, attribute)
        ]
        if missing_attribute:
            preview = ", ".join(
                str(item) for item in sorted(missing_attribute)[:10]
            )
            raise ValueError(
                f"HDSE graph cache entries are missing {attribute}, "
                f"including molecules: {preview}. Augment or rebuild "
                "the cache."
            )
    if required_mol_ids is not None:
        required = {int(mol_id) for mol_id in required_mol_ids}
        missing = sorted(required - set(graphs))
        if missing:
            preview = ", ".join(str(item) for item in missing[:10])
            raise ValueError(
                f"HDSE graph cache is missing {len(missing)} required "
                f"molecules, including: {preview}"
            )
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("HDSE graph cache metadata must be a mapping")
    return graphs, metadata
