"""Fragment and peak-set decoders used by FraHi and its baseline controls."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter


class _HardForwardSoftBackward(torch.autograd.Function):
    """Return the hard tensor exactly and backpropagate via the surrogate."""

    @staticmethod
    def forward(ctx, hard_value, surrogate_value):
        return hard_value

    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output


def _normalize_rendered_spectrum(spectrum, normalization):
    spectrum = spectrum.clamp_min(0.0)
    if normalization == "l1":
        return F.normalize(spectrum, p=1, dim=1)
    if normalization == "l2":
        return F.normalize(spectrum, p=2, dim=1)
    if normalization == "none":
        return spectrum
    raise ValueError(f"invalid output_normalization: {normalization}")


def pool_communities(node_embeddings, graph_ids, local_community_ids):
    """Mean-pool atoms by graph-scoped HDSE community."""
    if node_embeddings.ndim != 2:
        raise ValueError("node_embeddings must have shape [num_nodes, hidden]")
    if graph_ids.ndim != 1 or local_community_ids.ndim != 1:
        raise ValueError("graph and community IDs must be one-dimensional")
    if not (
        node_embeddings.shape[0]
        == graph_ids.shape[0]
        == local_community_ids.shape[0]
    ):
        raise ValueError("node embeddings and IDs must have equal lengths")
    pairs = torch.stack(
        [graph_ids.long(), local_community_ids.long()],
        dim=1,
    )
    unique_pairs, inverse = torch.unique(
        pairs,
        sorted=True,
        return_inverse=True,
        dim=0,
    )
    pooled = scatter(
        node_embeddings,
        inverse,
        dim=0,
        dim_size=unique_pairs.shape[0],
        reduce="mean",
    )
    dense, mask = to_dense_batch(pooled, unique_pairs[:, 0])
    return dense, mask


def build_hierarchy_context(
        node_embeddings,
        graph_ids,
        local_community_ids,
):
    """Build padded atom, community, and graph contexts."""
    node_tokens, node_mask = to_dense_batch(node_embeddings, graph_ids)
    community_tokens, community_mask = pool_communities(
        node_embeddings,
        graph_ids,
        local_community_ids,
    )
    num_graphs = int(graph_ids.max().item()) + 1
    graph_tokens = scatter(
        node_embeddings,
        graph_ids,
        dim=0,
        dim_size=num_graphs,
        reduce="mean",
    ).unsqueeze(1)
    graph_mask = torch.ones(
        graph_tokens.shape[:2],
        dtype=torch.bool,
        device=graph_tokens.device,
    )
    return {
        "node_tokens": node_tokens,
        "node_mask": node_mask,
        "community_tokens": community_tokens,
        "community_mask": community_mask,
        "graph_tokens": graph_tokens,
        "graph_mask": graph_mask,
    }


def _pair_features(source, target):
    return torch.cat(
        [source, target, torch.abs(source - target), source * target],
        dim=-1,
    )


def build_fragment_context(
        node_embeddings,
        graph_ids,
        local_community_ids,
        fragment_edge_index,
        fragment_mz,
        fragment_h_shift,
        fragment_side,
        fragment_is_inter_community):
    """Pad graph-local cleavage candidates and their hierarchy features."""
    num_graphs = int(graph_ids.max().item()) + 1
    hidden_dim = node_embeddings.shape[1]
    num_candidates = fragment_edge_index.shape[1]
    if num_candidates == 0:
        empty_pair = node_embeddings.new_zeros(
            (num_graphs, 0, hidden_dim * 4)
        )
        empty_token = node_embeddings.new_zeros(
            (num_graphs, 0, hidden_dim)
        )
        empty_value = node_embeddings.new_zeros((num_graphs, 0))
        return {
            "fragment_local_tokens": empty_pair,
            "fragment_community_tokens": empty_pair.clone(),
            "fragment_graph_tokens": empty_token,
            "fragment_mz": empty_value,
            "fragment_h_shift": empty_value.clone(),
            "fragment_side": empty_value.clone(),
            "fragment_is_inter_community": empty_value.clone(),
            "fragment_mask": torch.zeros(
                (num_graphs, 0), dtype=torch.bool,
                device=node_embeddings.device,
            ),
        }

    source, target = fragment_edge_index.long()
    if torch.any(graph_ids[source] != graph_ids[target]):
        raise ValueError("fragment candidate cannot cross molecular graphs")
    candidate_graph_ids = graph_ids[source].long()
    graph_tokens = scatter(
        node_embeddings,
        graph_ids,
        dim=0,
        dim_size=num_graphs,
        reduce="mean",
    )
    scoped_communities = torch.stack(
        [graph_ids.long(), local_community_ids.long()],
        dim=1,
    )
    _, community_inverse = torch.unique(
        scoped_communities,
        sorted=True,
        return_inverse=True,
        dim=0,
    )
    pooled_communities = scatter(
        node_embeddings,
        community_inverse,
        dim=0,
        reduce="mean",
    )
    node_communities = pooled_communities[community_inverse]

    local_pair = _pair_features(
        node_embeddings[source], node_embeddings[target]
    )
    community_pair = _pair_features(
        node_communities[source], node_communities[target]
    )
    candidate_graph_tokens = graph_tokens[candidate_graph_ids]

    def pad(values):
        dense, mask = to_dense_batch(
            values,
            candidate_graph_ids,
            batch_size=num_graphs,
        )
        return dense, mask

    local_dense, candidate_mask = pad(local_pair)
    community_dense, _ = pad(community_pair)
    graph_dense, _ = pad(candidate_graph_tokens)
    mz_dense, _ = pad(fragment_mz.float().unsqueeze(-1))
    shift_dense, _ = pad(fragment_h_shift.float().unsqueeze(-1))
    side_dense, _ = pad(fragment_side.float().unsqueeze(-1))
    inter_dense, _ = pad(
        fragment_is_inter_community.float().unsqueeze(-1)
    )
    return {
        "fragment_local_tokens": local_dense,
        "fragment_community_tokens": community_dense,
        "fragment_graph_tokens": graph_dense,
        "fragment_mz": mz_dense.squeeze(-1),
        "fragment_h_shift": shift_dense.squeeze(-1),
        "fragment_side": side_dense.squeeze(-1),
        "fragment_is_inter_community": inter_dense.squeeze(-1),
        "fragment_mask": candidate_mask,
    }


def render_peak_set(
        peak_mz,
        peak_weight,
        mz_max,
        mz_bin_res,
        output_normalization="l1",
        gradient_sigmas_mz=(0.1, 0.5, 2.0, 5.0),
):
    """Render hard-bin peaks with a multiscale straight-through gradient.

    Exact masses are supervised separately by the set loss.  Hard rendering
    keeps the forward spectrum and reported metrics aligned with
    ``spec_utils.bin_func``.  The backward pass blurs an adjacent-bin linear
    splat at several mass scales, allowing coarse mass correction before
    fine-bin alignment.
    """
    if peak_mz.shape != peak_weight.shape:
        raise ValueError("peak_mz and peak_weight must have matching shapes")
    if peak_mz.ndim != 2:
        raise ValueError("peak tensors must have shape [batch, queries]")
    if float(mz_max) <= 0 or float(mz_bin_res) <= 0:
        raise ValueError("mz_max and mz_bin_res must be positive")
    gradient_sigmas_mz = tuple(float(value) for value in gradient_sigmas_mz)
    if not gradient_sigmas_mz or any(
            value <= 0.0 for value in gradient_sigmas_mz):
        raise ValueError("gradient_sigmas_mz must contain positive values")
    num_bins_float = float(mz_max) / float(mz_bin_res)
    num_bins = int(round(num_bins_float))
    if not math.isclose(num_bins, num_bins_float, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("mz_max must be divisible by mz_bin_res")

    bin_position = peak_mz / float(mz_bin_res)
    bin_index = torch.floor(bin_position).long()
    hard_spectrum = peak_weight.new_zeros(
        (peak_weight.shape[0], num_bins)
    )
    valid = (bin_index >= 0) & (bin_index < num_bins)
    hard_spectrum.scatter_add_(
        1,
        bin_index.clamp(0, num_bins - 1),
        peak_weight * valid.to(peak_weight.dtype),
    )
    hard_spectrum = _normalize_rendered_spectrum(
        hard_spectrum,
        output_normalization,
    )
    if (
        not torch.is_grad_enabled()
        or not (peak_mz.requires_grad or peak_weight.requires_grad)
    ):
        return hard_spectrum

    fraction = bin_position - torch.floor(bin_position)
    upper_index = bin_index + 1
    lower_valid = (bin_index >= 0) & (bin_index < num_bins)
    upper_valid = (upper_index >= 0) & (upper_index < num_bins)
    soft_spectrum = peak_weight.new_zeros(
        (peak_weight.shape[0], num_bins)
    )
    soft_spectrum = soft_spectrum.scatter_add(
        1,
        bin_index.clamp(0, num_bins - 1),
        peak_weight
        * (1.0 - fraction)
        * lower_valid.to(peak_weight.dtype),
    )
    soft_spectrum = soft_spectrum.scatter_add(
        1,
        upper_index.clamp(0, num_bins - 1),
        peak_weight
        * fraction
        * upper_valid.to(peak_weight.dtype),
    )
    multiscale_spectrum = torch.zeros_like(soft_spectrum)
    for sigma_mz in gradient_sigmas_mz:
        sigma_bins = sigma_mz / float(mz_bin_res)
        radius = max(1, int(math.ceil(3.0 * sigma_bins)))
        offsets = torch.arange(
            -radius,
            radius + 1,
            dtype=soft_spectrum.dtype,
            device=soft_spectrum.device,
        )
        kernel = torch.exp(
            -0.5 * torch.square(offsets / sigma_bins)
        )
        kernel = kernel / kernel.sum()
        multiscale_spectrum = multiscale_spectrum + F.conv1d(
            soft_spectrum.unsqueeze(1),
            kernel.view(1, 1, -1),
            padding=radius,
        ).squeeze(1)
    soft_spectrum = multiscale_spectrum / len(gradient_sigmas_mz)
    soft_spectrum = _normalize_rendered_spectrum(
        soft_spectrum,
        output_normalization,
    )
    return _HardForwardSoftBackward.apply(
        hard_spectrum,
        soft_spectrum,
    )


def apply_fragment_log_residual(
        anchor,
        peak_mz,
        peak_intensity,
        peak_log_ratio,
        candidate_mask,
        mz_max,
        mz_bin_res,
        output_normalization="l1"):
    """Apply bounded candidate-local log-ratio corrections to an anchor.

    Each candidate contributes ``intensity * expm1(log_ratio)`` only to
    its hard mass bin.  A zero log-ratio therefore reproduces the anchor
    exactly while retaining a non-zero derivative for the residual head.
    """
    candidate_tensors = (
        peak_mz,
        peak_intensity,
        peak_log_ratio,
        candidate_mask,
    )
    if anchor.ndim != 2:
        raise ValueError("anchor must have shape [batch, bins]")
    if any(tensor.shape != peak_mz.shape for tensor in candidate_tensors):
        raise ValueError("fragment candidate tensors must have equal shapes")
    if peak_mz.ndim != 2 or peak_mz.shape[0] != anchor.shape[0]:
        raise ValueError(
            "fragment candidates must have shape [batch, candidates]"
        )
    if float(mz_max) <= 0.0 or float(mz_bin_res) <= 0.0:
        raise ValueError("mz_max and mz_bin_res must be positive")
    expected_bins = float(mz_max) / float(mz_bin_res)
    if not math.isclose(
            anchor.shape[1], expected_bins, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("anchor width must equal mz_max / mz_bin_res")

    bin_index = torch.floor(peak_mz / float(mz_bin_res)).long()
    valid = (
        candidate_mask.bool()
        & (bin_index >= 0)
        & (bin_index < anchor.shape[1])
    )
    candidate_delta = (
        peak_intensity
        * torch.expm1(peak_log_ratio)
        * valid.to(peak_intensity.dtype)
    )
    dense_delta = anchor.new_zeros(anchor.shape).scatter_add(
        1,
        bin_index.clamp(0, anchor.shape[1] - 1),
        candidate_delta,
    )
    refined = anchor + dense_delta
    if not torch.any(dense_delta.detach()):
        return refined, dense_delta
    return (
        _normalize_rendered_spectrum(
            refined,
            output_normalization,
        ),
        dense_delta,
    )


class PeakQueryLayer(nn.Module):
    """One query-update block with optional HDSE hierarchy routing."""

    def __init__(self, mode, hidden_dim, num_heads, dropout):
        super().__init__()
        if mode not in {"plain_peak", "hipeak"}:
            raise ValueError(f"invalid peak decoder mode: {mode}")
        self.mode = mode
        self.atom_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        if mode == "hipeak":
            self.community_attention = nn.MultiheadAttention(
                hidden_dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.graph_attention = nn.MultiheadAttention(
                hidden_dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.route = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 3),
            )
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.ff_norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _attend(module, query, tokens, mask):
        output, _ = module(
            query,
            tokens,
            tokens,
            key_padding_mask=~mask,
            need_weights=False,
        )
        return output

    def forward(self, query, context):
        atom_output = self._attend(
            self.atom_attention,
            query,
            context["node_tokens"],
            context["node_mask"],
        )
        route_weights = None
        if self.mode == "plain_peak":
            routed = atom_output
        else:
            community_output = self._attend(
                self.community_attention,
                query,
                context["community_tokens"],
                context["community_mask"],
            )
            graph_output = self._attend(
                self.graph_attention,
                query,
                context["graph_tokens"],
                context["graph_mask"],
            )
            route_features = torch.cat(
                [
                    query,
                    atom_output,
                    community_output,
                    graph_output,
                ],
                dim=-1,
            )
            route_weights = torch.softmax(
                self.route(route_features),
                dim=-1,
            )
            candidates = torch.stack(
                [atom_output, community_output, graph_output],
                dim=2,
            )
            routed = torch.sum(
                candidates * route_weights.unsqueeze(-1),
                dim=2,
            )
        query = self.context_norm(query + routed)
        query = self.ff_norm(query + self.ff(query))
        return query, route_weights


class ContinuousPeakDecoder(nn.Module):
    """Decode a fixed set of continuous peaks and render a dense spectrum."""

    def __init__(
            self,
            mode,
            hidden_dim,
            metadata_dim,
            num_queries,
            num_heads,
            num_layers,
            mz_max,
            mz_bin_res,
            output_normalization,
            gradient_sigmas_mz=(0.1, 0.5, 2.0, 5.0),
            dropout=0.1):
        super().__init__()
        if mode not in {"plain_peak", "hipeak"}:
            raise ValueError(f"invalid peak decoder mode: {mode}")
        if int(num_queries) <= 0:
            raise ValueError("num_queries must be greater than zero")
        if int(num_layers) <= 0:
            raise ValueError("num_layers must be greater than zero")
        self.mode = mode
        self.num_queries = int(num_queries)
        self.mz_max = float(mz_max)
        self.mz_bin_res = float(mz_bin_res)
        self.output_normalization = output_normalization
        self.gradient_sigmas_mz = tuple(
            float(value) for value in gradient_sigmas_mz
        )
        if not self.gradient_sigmas_mz or any(
                value <= 0.0 for value in self.gradient_sigmas_mz):
            raise ValueError(
                "gradient_sigmas_mz must contain positive values"
            )

        self.query_embedding = nn.Embedding(self.num_queries, hidden_dim)
        self.metadata_projection = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [
                PeakQueryLayer(mode, hidden_dim, num_heads, dropout)
                for _ in range(int(num_layers))
            ]
        )
        self.exist_head = nn.Linear(hidden_dim, 1)
        self.absolute_mass_head = nn.Linear(hidden_dim, 1)
        self.neutral_loss_head = nn.Linear(hidden_dim, 1)
        self.mass_gate_head = nn.Linear(hidden_dim, 1)
        self.intensity_head = nn.Linear(hidden_dim, 1)

    def forward(self, context, metadata, precursor_mz):
        batch_size = metadata.shape[0]
        query = self.query_embedding.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        query = query + self.metadata_projection(metadata).unsqueeze(1)
        route_weights = None
        for layer in self.layers:
            query, route_weights = layer(query, context)

        precursor_limit = precursor_mz.float().clamp(
            min=0.0,
            max=self.mz_max - 1e-4,
        ).unsqueeze(1)
        absolute_mz = torch.sigmoid(
            self.absolute_mass_head(query).squeeze(-1)
        ) * (self.mz_max - 1e-4)
        neutral_loss = torch.sigmoid(
            self.neutral_loss_head(query).squeeze(-1)
        ) * precursor_limit
        neutral_loss_mz = precursor_limit - neutral_loss
        mass_gate = torch.sigmoid(
            self.mass_gate_head(query).squeeze(-1)
        )
        peak_mz = (
            mass_gate * absolute_mz
            + (1.0 - mass_gate) * neutral_loss_mz
        )

        exist_logits = self.exist_head(query).squeeze(-1)
        probability = torch.sigmoid(exist_logits)
        intensity = F.softplus(
            self.intensity_head(query).squeeze(-1)
        )
        intensity = F.normalize(intensity, p=1, dim=1)
        rendered = render_peak_set(
            peak_mz,
            probability * intensity,
            mz_max=self.mz_max,
            mz_bin_res=self.mz_bin_res,
            output_normalization=self.output_normalization,
            gradient_sigmas_mz=self.gradient_sigmas_mz,
        )
        output = {
            "pred": rendered,
            "peak_exist_logits": exist_logits,
            "peak_probability": probability,
            "peak_mz": peak_mz,
            "peak_intensity": intensity,
            "peak_mz_absolute": absolute_mz,
            "peak_mz_neutral_loss": neutral_loss_mz,
            "peak_mass_gate": mass_gate,
        }
        if route_weights is not None:
            output["peak_route_weights"] = route_weights
        return output


class FragmentCandidateDecoder(nn.Module):
    """Score exact-mass bond-cleavage candidates as a sparse spectrum."""

    def __init__(
            self,
            mode,
            hidden_dim,
            metadata_dim,
            num_layers,
            mz_max,
            mz_bin_res,
            output_normalization,
            gradient_sigmas_mz=(0.1, 0.5, 2.0, 5.0),
            local_residual_log_ratio_max=0.5,
            dropout=0.1):
        super().__init__()
        if mode not in {"plain_peak", "hipeak"}:
            raise ValueError(f"invalid fragment decoder mode: {mode}")
        if int(num_layers) <= 0:
            raise ValueError("num_layers must be greater than zero")
        self.mode = mode
        self.mz_max = float(mz_max)
        self.mz_bin_res = float(mz_bin_res)
        self.output_normalization = output_normalization
        self.local_residual_log_ratio_max = float(
            local_residual_log_ratio_max
        )
        if self.local_residual_log_ratio_max <= 0.0:
            raise ValueError(
                "local_residual_log_ratio_max must be positive"
            )
        self.gradient_sigmas_mz = tuple(
            float(value) for value in gradient_sigmas_mz
        )
        self.chemical_projection = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.metadata_projection = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.local_projection = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        if mode == "hipeak":
            self.community_projection = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.route = nn.Sequential(
                nn.Linear(hidden_dim * 5, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 3),
            )
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.exist_head = nn.Linear(hidden_dim, 1)
        self.intensity_head = nn.Linear(hidden_dim, 1)
        self.log_ratio_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.log_ratio_head.weight)
        nn.init.zeros_(self.log_ratio_head.bias)

    def forward(self, context, metadata, precursor_mz):
        mask = context["fragment_mask"].bool()
        peak_mz = context["fragment_mz"].float()
        chemical_features = torch.stack(
            [
                peak_mz / self.mz_max,
                context["fragment_h_shift"].float() / 2.0,
                context["fragment_side"].float(),
                context["fragment_is_inter_community"].float(),
            ],
            dim=-1,
        )
        chemical = self.chemical_projection(chemical_features)
        metadata_token = self.metadata_projection(metadata).unsqueeze(1)
        local = self.local_projection(context["fragment_local_tokens"])
        route_weights = None
        if self.mode == "plain_peak":
            routed = local
        else:
            community = self.community_projection(
                context["fragment_community_tokens"]
            )
            graph = context["fragment_graph_tokens"]
            route_input = torch.cat(
                [local, community, graph, chemical, metadata_token.expand_as(local)],
                dim=-1,
            )
            route_weights = torch.softmax(self.route(route_input), dim=-1)
            route_candidates = torch.stack(
                [local, community, graph],
                dim=2,
            )
            routed = torch.sum(
                route_candidates * route_weights.unsqueeze(-1),
                dim=2,
            )
            route_weights = route_weights * mask.unsqueeze(-1)
        hidden = self.input_norm(routed + chemical + metadata_token)
        for layer in self.layers:
            hidden = hidden + layer(hidden)

        exist_logits = self.exist_head(hidden).squeeze(-1)
        raw_intensity = F.softplus(
            self.intensity_head(hidden).squeeze(-1)
        )
        probability = torch.sigmoid(exist_logits)
        peak_log_ratio = (
            self.local_residual_log_ratio_max
            * torch.tanh(self.log_ratio_head(hidden).squeeze(-1))
            * mask.to(hidden.dtype)
        )
        candidate_weight = (
            probability
            * raw_intensity
            * mask.to(raw_intensity.dtype)
        )
        intensity = F.normalize(candidate_weight, p=1, dim=1)
        rendered = render_peak_set(
            peak_mz,
            intensity,
            mz_max=self.mz_max,
            mz_bin_res=self.mz_bin_res,
            output_normalization=self.output_normalization,
            gradient_sigmas_mz=self.gradient_sigmas_mz,
        )
        output = {
            "pred": rendered,
            "peak_exist_logits": exist_logits,
            "peak_probability": probability * mask,
            "peak_mz": peak_mz,
            "peak_intensity": intensity,
            "peak_mz_absolute": peak_mz,
            "peak_mz_neutral_loss": (
                precursor_mz.float().unsqueeze(1) - peak_mz
            ),
            "peak_mass_gate": torch.ones_like(peak_mz),
            "peak_candidate_mask": mask,
            "peak_log_ratio": peak_log_ratio,
            "peak_hidden": hidden,
            "fragment_constrained": True,
        }
        if route_weights is not None:
            output["peak_route_weights"] = route_weights
        return output


class HierarchicalResidualAdapter(nn.Module):
    """Jointly refine local fragment candidates with bounded HDSE evidence.

    Local, community, and graph branches train in one run.  The community,
    graph, and log-ratio outputs start at exactly zero, so the complete model
    initially reproduces its concurrently trained dense baseline.
    """

    def __init__(
            self,
            hidden_dim,
            metadata_dim,
            residual_log_ratio_max=0.15,
            adapter_ratio=0.25,
            dropout=0.1):
        super().__init__()
        self.residual_log_ratio_max = float(residual_log_ratio_max)
        if not 0.0 < self.residual_log_ratio_max <= 2.0:
            raise ValueError(
                "residual_log_ratio_max must be in the range (0, 2]"
            )
        adapter_ratio = float(adapter_ratio)
        if not 0.0 < adapter_ratio <= 1.0:
            raise ValueError("adapter_ratio must be in the range (0, 1]")
        adapter_dim = max(8, int(round(hidden_dim * adapter_ratio)))

        self.community_adapter = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, adapter_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim, hidden_dim),
        )
        self.graph_adapter = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, adapter_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim, hidden_dim),
        )
        for adapter in (self.community_adapter, self.graph_adapter):
            nn.init.zeros_(adapter[-1].weight)
            nn.init.zeros_(adapter[-1].bias)

        self.local_route_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, adapter_dim),
            nn.SiLU(),
        )
        self.community_route_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, adapter_dim),
            nn.SiLU(),
        )
        self.graph_route_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, adapter_dim),
            nn.SiLU(),
        )
        self.metadata_route_projection = nn.Sequential(
            nn.Linear(metadata_dim, adapter_dim),
            nn.SiLU(),
        )
        self.route = nn.Linear(adapter_dim * 4, 3)
        with torch.no_grad():
            self.route.bias.copy_(
                self.route.bias.new_tensor([2.0, 0.0, 0.0])
            )

        self.log_ratio_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.log_ratio_head.weight)
        nn.init.zeros_(self.log_ratio_head.bias)

    @staticmethod
    def _reuse_plain_candidates(plain_output):
        required = (
            "pred",
            "peak_exist_logits",
            "peak_probability",
            "peak_mz",
            "peak_intensity",
            "peak_mz_absolute",
            "peak_mz_neutral_loss",
            "peak_mass_gate",
            "peak_candidate_mask",
            "peak_hidden",
        )
        missing = [key for key in required if key not in plain_output]
        if missing:
            raise ValueError(
                "plain_output is missing adapter inputs: "
                + ", ".join(missing)
            )
        return {
            key: plain_output[key]
            for key in required
        }

    def forward(self, context, metadata, precursor_mz, plain_output):
        del precursor_mz
        anchor = self._reuse_plain_candidates(plain_output)
        mask = anchor["peak_candidate_mask"].bool()
        local_hidden = anchor["peak_hidden"]
        community_tokens = context["fragment_community_tokens"]
        graph_tokens = context["fragment_graph_tokens"]
        if community_tokens.shape[:2] != local_hidden.shape[:2]:
            raise ValueError(
                "community candidates must match Plain-Peak candidates"
            )
        if graph_tokens.shape != local_hidden.shape:
            raise ValueError(
                "graph candidates must match Plain-Peak hidden features"
            )

        metadata_tokens = metadata.unsqueeze(1).expand(
            -1, local_hidden.shape[1], -1
        )
        route_input = torch.cat(
            [
                self.local_route_projection(local_hidden),
                self.community_route_projection(community_tokens),
                self.graph_route_projection(graph_tokens),
                self.metadata_route_projection(metadata_tokens),
            ],
            dim=-1,
        )
        route_weights = torch.softmax(self.route(route_input), dim=-1)
        community_delta = self.community_adapter(community_tokens)
        graph_delta = self.graph_adapter(graph_tokens)
        adapted_hidden = (
            local_hidden
            + route_weights[..., 1:2] * community_delta
            + route_weights[..., 2:3] * graph_delta
        )
        peak_log_ratio = (
            self.residual_log_ratio_max
            * torch.tanh(
                self.log_ratio_head(adapted_hidden).squeeze(-1)
            )
            * mask.to(adapted_hidden.dtype)
        )

        return {
            "pred": anchor["pred"],
            "peak_exist_logits": anchor["peak_exist_logits"],
            "peak_probability": anchor["peak_probability"],
            "peak_mz": anchor["peak_mz"],
            "peak_intensity": anchor["peak_intensity"],
            "peak_mz_absolute": anchor["peak_mz_absolute"],
            "peak_mz_neutral_loss": anchor["peak_mz_neutral_loss"],
            "peak_mass_gate": anchor["peak_mass_gate"],
            "peak_candidate_mask": mask,
            "peak_log_ratio": peak_log_ratio,
            "peak_route_weights": (
                route_weights * mask.unsqueeze(-1).to(route_weights.dtype)
            ),
            "fragment_constrained": True,
        }
