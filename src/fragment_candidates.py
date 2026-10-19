"""Chemically constrained fragment-candidate tensor construction."""

import torch


PROTON_MASS = 1.007276466621
HYDROGEN_ATOM_MASS = 1.00782503223


def compute_bridge_fragment_masses(atom_masses, bonds):
    """Compute fragment masses by deleting graph bridges without sanitizing.

    Atom masses must already include the hydrogens attached in the original
    molecule.  Ring bonds are rejected because their endpoints remain
    connected after a single deletion.
    """
    atom_masses = tuple(float(value) for value in atom_masses)
    num_atoms = len(atom_masses)
    adjacency = [set() for _ in range(num_atoms)]
    normalized_bonds = []
    for atom_a, atom_b in bonds:
        atom_a = int(atom_a)
        atom_b = int(atom_b)
        if not (
            0 <= atom_a < num_atoms
            and 0 <= atom_b < num_atoms
            and atom_a != atom_b
        ):
            raise ValueError("bond endpoint is outside the molecular graph")
        adjacency[atom_a].add(atom_b)
        adjacency[atom_b].add(atom_a)
        normalized_bonds.append((atom_a, atom_b))

    all_atoms = set(range(num_atoms))

    def component(start, removed_a, removed_b):
        visited = {start}
        stack = [start]
        while stack:
            atom_index = stack.pop()
            for neighbor in adjacency[atom_index]:
                if (
                    (atom_index == removed_a and neighbor == removed_b)
                    or (
                        atom_index == removed_b
                        and neighbor == removed_a
                    )
                ):
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        return visited

    bond_fragments = []
    for atom_a, atom_b in normalized_bonds:
        component_a = component(atom_a, atom_a, atom_b)
        if atom_b in component_a:
            continue
        component_b = component(atom_b, atom_a, atom_b)
        if (
            component_a & component_b
            or component_a | component_b != all_atoms
        ):
            continue
        bond_fragments.append(
            (
                atom_a,
                atom_b,
                sum(atom_masses[index] for index in component_a),
                sum(atom_masses[index] for index in component_b),
            )
        )
    return bond_fragments


def _empty_candidates():
    return {
        "fragment_edge_index": torch.empty((2, 0), dtype=torch.long),
        "fragment_mz": torch.empty((0,), dtype=torch.float32),
        "fragment_h_shift": torch.empty((0,), dtype=torch.float32),
        "fragment_side": torch.empty((0,), dtype=torch.float32),
        "fragment_is_inter_community": torch.empty(
            (0,), dtype=torch.float32
        ),
    }


def assemble_fragment_candidates(
        bond_fragments,
        community_ids,
        hydrogen_shifts=(-2, -1, 0, 1, 2),
        max_candidates=512,
        mz_max=500.0):
    """Create side-specific product-ion candidates from bridge cleavages.

    ``bond_fragments`` contains ``(atom_a, atom_b, mass_a, mass_b)`` for
    bonds whose removal splits the molecular graph.  The two endpoint orders
    identify which component supplies the candidate product ion.
    """
    community_ids = torch.as_tensor(community_ids, dtype=torch.long)
    hydrogen_shifts = tuple(int(value) for value in hydrogen_shifts)
    if int(max_candidates) <= 0:
        raise ValueError("max_candidates must be greater than zero")
    if float(mz_max) <= 0:
        raise ValueError("mz_max must be greater than zero")
    if not hydrogen_shifts:
        raise ValueError("hydrogen_shifts cannot be empty")

    rows = []
    for atom_a, atom_b, mass_a, mass_b in bond_fragments:
        atom_a = int(atom_a)
        atom_b = int(atom_b)
        if not (
            0 <= atom_a < community_ids.numel()
            and 0 <= atom_b < community_ids.numel()
        ):
            raise ValueError("fragment bond endpoint is outside the graph")
        is_inter = float(
            community_ids[atom_a] != community_ids[atom_b]
        )
        for side, source, target, neutral_mass in (
            (0.0, atom_a, atom_b, float(mass_a)),
            (1.0, atom_b, atom_a, float(mass_b)),
        ):
            for hydrogen_shift in hydrogen_shifts:
                mz = (
                    neutral_mass
                    + PROTON_MASS
                    + hydrogen_shift * HYDROGEN_ATOM_MASS
                )
                if 0.0 < mz < float(mz_max):
                    rows.append(
                        (
                            source,
                            target,
                            mz,
                            float(hydrogen_shift),
                            side,
                            is_inter,
                        )
                    )
    if not rows:
        return _empty_candidates()

    rows = rows[:int(max_candidates)]
    values = torch.tensor(rows, dtype=torch.float64)
    return {
        "fragment_edge_index": values[:, :2].long().t().contiguous(),
        "fragment_mz": values[:, 2].float(),
        "fragment_h_shift": values[:, 3].float(),
        "fragment_side": values[:, 4].float(),
        "fragment_is_inter_community": values[:, 5].float(),
    }
