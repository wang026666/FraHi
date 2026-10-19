import os
import math
from collections import defaultdict
from types import SimpleNamespace
from typing import List, Optional, Tuple
from torch import _VF

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pygnn
from torch_geometric.nn import Linear as Linear_pyg, global_mean_pool
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch, to_dense_adj, degree
import networkx as nx


from data_utils import DummyContext
from config_utils import normalize_prediction_head

Tensor = torch.Tensor  
from torch_scatter import scatter
import torch_geometric.nn as pyg_nn
from peak_decoder import (
    FragmentCandidateDecoder,
    HierarchicalResidualAdapter,
    apply_fragment_log_residual,
    build_fragment_context,
    build_hierarchy_context,
)


config = {
    "gt": {
        "layer_type": "GINE+Transformer",  # CustomGatedGCN+Performer
        "layers": 8,
        "n_heads": 8,
        "dim_hidden": 320,  # `gt.dim_hidden` must match `gnn.dim_inner`
        "dropout": 0.2,
        "attn_dropout": 0.5,
        "layer_norm": False,
        "batch_norm": True,
    },
    "gnn": {
        "head": "san_graph",
        "layers_pre_mp": 0,
        "layers_post_mp": 3,  # Not used when `gnn.head: san_graph`
        "dim_inner": 320,  # `gt.dim_hidden` must match `gnn.dim_inner`
        "batchnorm": True,
        "act": "relu",
        "dropout": 0.0,
        "agg": "mean",
        "normalize_adj": False,
    },
}


class GatedGCNLayer(pyg_nn.conv.MessagePassing):
    """
        GatedGCN layer
        Residual Gated Graph ConvNets
        https://arxiv.org/pdf/1711.07553.pdf
    """
    def __init__(self, in_dim, out_dim, dropout, residual,
                 equivstable_pe=False, **kwargs):
        super().__init__(**kwargs)
        self.A = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.B = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.C = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.D = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.E = pyg_nn.Linear(in_dim, out_dim, bias=True)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        self.EquivStablePE = equivstable_pe
        if self.EquivStablePE:
            self.mlp_r_ij = nn.Sequential(
                nn.Linear(1, out_dim), nn.ReLU(),
                nn.Linear(out_dim, 1),
                nn.Sigmoid())

        self.bn_node_x = nn.BatchNorm1d(out_dim)
        self.bn_edge_e = nn.BatchNorm1d(out_dim)
        self.dropout = dropout
        self.residual = residual
        self.e = None

    def forward(self, batch):
        x, e, edge_index = batch.x, batch.edge_attr, batch.edge_index

        """
        x               : [n_nodes, in_dim]
        e               : [n_edges, in_dim]
        edge_index      : [2, n_edges]
        """
        if self.residual:
            x_in = x
            e_in = e

        Ax = self.A(x)
        Bx = self.B(x)
        Ce = self.C(e)
        Dx = self.D(x)
        Ex = self.E(x)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        pe_LapPE = batch.pe_EquivStableLapPE if self.EquivStablePE else None

        x, e = self.propagate(edge_index,
                              Bx=Bx, Dx=Dx, Ex=Ex, Ce=Ce,
                              e=e, Ax=Ax,
                              PE=pe_LapPE)

        x = self.bn_node_x(x)
        e = self.bn_edge_e(e)

        x = F.relu(x)
        e = F.relu(e)

        x = F.dropout(x, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        if self.residual:
            x = x_in + x
            e = e_in + e

        batch.x = x
        batch.edge_attr = e

        return batch

    def message(self, Dx_i, Ex_j, PE_i, PE_j, Ce):
        """
        {}x_i           : [n_edges, out_dim]
        {}x_j           : [n_edges, out_dim]
        {}e             : [n_edges, out_dim]
        """
        e_ij = Dx_i + Ex_j + Ce
        sigma_ij = torch.sigmoid(e_ij)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        if self.EquivStablePE:
            r_ij = ((PE_i - PE_j) ** 2).sum(dim=-1, keepdim=True)
            r_ij = self.mlp_r_ij(r_ij)  # the MLP is 1 dim --> hidden_dim --> 1 dim
            sigma_ij = sigma_ij * r_ij

        self.e = e_ij
        return sigma_ij

    def aggregate(self, sigma_ij, index, Bx_j, Bx):
        """
        sigma_ij        : [n_edges, out_dim]  ; is the output from message() function
        index           : [n_edges]
        {}x_j           : [n_edges, out_dim]
        """
        dim_size = Bx.shape[0]  # or None ??   <--- Double check this

        sum_sigma_x = sigma_ij * Bx_j
        numerator_eta_xj = scatter(sum_sigma_x, index, 0, None, dim_size,
                                   reduce='sum')

        sum_sigma = sigma_ij
        denominator_eta_xj = scatter(sum_sigma, index, 0, None, dim_size,
                                     reduce='sum')

        out = numerator_eta_xj / (denominator_eta_xj + 1e-6)
        return out

    def update(self, aggr_out, Ax):
        """
        aggr_out        : [n_nodes, out_dim] ; is the output from aggregate() function after the aggregation
        {}x             : [n_nodes, out_dim]
        """
        x = Ax + aggr_out
        e_out = self.e
        del self.e
        return x, e_out


def dropout(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
    if p < 0.0 or p > 1.0:
        raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
    return _VF.dropout_(input, p, training) if inplace else _VF.dropout(input, p, training)


def softmax(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype: Optional[int] = None) -> Tensor:
    if dtype is None:
        ret = input.softmax(dim)
    else:
        ret = input.softmax(dim, dtype=dtype)
    return ret

def unbatch_edge_index(edge_index: Tensor, batch: Tensor) -> List[Tensor]:

    deg = degree(batch, dtype=torch.int64)
    ptr = torch.cat([deg.new_zeros(1), deg.cumsum(dim=0)[:-1]], dim=0)

    edge_batch = batch[edge_index[0]]
    edge_index = edge_index - ptr[edge_batch]
    sizes = degree(edge_batch, dtype=torch.int64).cpu().tolist()
    return edge_index.split(sizes, dim=1)


def bias_scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0, dist = None
) -> Tuple[Tensor, Tensor]:
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2, -1))
    attn_ = softmax(attn, dim=-1)
    # print("ok!!!!!!!!!")
    structure_query = dist
    tgt_len = structure_query.shape[1]
    structure_query = structure_query.contiguous().transpose(2, 3)
    structure_query = structure_query.contiguous().transpose(1, 2)
    structure_query = structure_query.contiguous().view(attn.shape[0], tgt_len, tgt_len)
    attn = attn + structure_query
    
    attn_ = softmax(structure_query, dim=-1)
    if attn_mask is not None:
        attn += attn_mask
    attn = softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn, attn_

def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    dist = None
) -> Tuple[Tensor, Optional[Tensor]]:

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
       
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    if not use_separate_proj_weight:
        q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = F._in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)
    
   
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")


    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

 
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    
    if static_k is None:
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    src_len = k.size(1)


    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))


    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    if not training:
        dropout_p = 0.0


    attn_output, attn_output_weights, attn_output_weights_ = bias_scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, dist)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights_ = attn_output_weights_.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, (attn_output_weights.sum(dim=1) / num_heads, attn_output_weights_.sum(dim=1) / num_heads)
    else:
        return attn_output, None


def draw_graph_with_attn(
    graph,
    outdir,
    filename,
    nodecolor=("tag", "attn"),
    dpi=300
):
    adj_matrix = nx.to_numpy_array(graph)
    partitions, _ = girvan_newman(adj_matrix)
    # print(partitions)
    color_map = []
    for node in graph.nodes:
        for i, cluster in enumerate(partitions):
            if node in cluster:
                color_map.append(i)
    if len(graph.edges) == 0:
        return
    import matplotlib.pyplot as plt
    plt.switch_backend("agg")
    fig = plt.figure(figsize=(4*len(nodecolor), 4), dpi=dpi)

    node_colors = defaultdict(list)

    titles = {
        'tag': '4-Chloro-N-phenylbenzamide',
        'attn1': 'Transformer attention',
        'attn2': 'HDSE attention'
    }


    for i in graph.nodes():
        for key in nodecolor:
            node_colors[key].append(graph.nodes[i][key])

    node_colors['tag'] = color_map
    node_colors['tag'][0] = 100
    vmax = {}
    cmap = {}
    for key in nodecolor:
        vmax[key] = 19
        cmap[key] = 'tab20'
        if 'attn' in key:
            vmax[key] = max(node_colors[key])
            cmap[key] = 'viridis'

    pos_layout = nx.spring_layout(graph, weight=None)
    # kamada_kawai_layout
    for i, key in enumerate(nodecolor):
        ax = fig.add_subplot(1, len(nodecolor), i+1)
        ax.set_title(titles.get(key, key), fontweight='bold')
        nx.draw(
            graph,
            pos=pos_layout,
            with_labels=False,
            font_size=4,
            node_color=node_colors[key],
            vmin=0,
            vmax=vmax[key],
            cmap=cmap[key],
            width=1.3,
            node_size=20,
            alpha=1.0,
        )
        if 'attn' in key:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            sm = plt.cm.ScalarMappable(cmap=cmap[key], norm=plt.Normalize(vmin=0, vmax=vmax[key]))
            sm._A = []
            plt.colorbar(sm, cax=cax)

    fig.axes[0].xaxis.set_visible(False)
    try:
        fig.canvas.draw()
    except:
        return

    save_path = os.path.join(outdir, filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(save_path)
    plt.savefig(save_path,dpi=1200)
  
    save_path_eps = os.path.splitext(save_path)[0] + ".eps"
    plt.savefig(save_path_eps, format='eps',dpi=1200)

    plt.close(fig)
    
class biasMultiheadAttention(nn.MultiheadAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None, dist=None) -> Tuple[Tensor, Optional[Tensor]]:
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
        
        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, dist=dist)
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, dist=dist)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

class UpdateLayer(nn.Module):
    """Local MPNN + full graph attention x-former layer.
    """

    def __init__(self,  dim_h,
                 local_gnn_type, global_model_type, num_heads, dropout=0.0,
                 attn_dropout=0.0, layer_norm=False, batch_norm=True,
                 bigbird_cfg=None):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = False


        # Local message-passing model.
        if local_gnn_type == 'None':
            self.local_model = None
        elif local_gnn_type == 'GENConv':
            self.local_model = pygnn.GENConv(dim_h, dim_h)
        elif local_gnn_type == 'GINE':
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                   nn.ReLU(),
                                   Linear_pyg(dim_h, dim_h))
            self.local_model = pygnn.GINEConv(gin_nn)
        elif local_gnn_type == 'GAT':
            self.local_model = pygnn.GATConv(in_channels=dim_h,
                                             out_channels=dim_h // num_heads,
                                             heads=num_heads,
                                             edge_dim=dim_h)
        elif local_gnn_type == 'CustomGatedGCN':
            equivstable_pe = False
            self.local_model = GatedGCNLayer(dim_h, dim_h,
                                             dropout=dropout,
                                             residual=True,
                                             equivstable_pe=equivstable_pe)
        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
        self.local_gnn_type = local_gnn_type

        # torch.nn.functional._scaled_dot_product_attention = bias_scaled_dot_product_attention
        # Global attention transformer-style model.
        if global_model_type == 'None':
            self.self_attn = None
        elif global_model_type == 'Transformer':
            self.self_attn = biasMultiheadAttention(
                dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
            # self.global_model = torch.nn.TransformerEncoderLayer(
            #     d_model=dim_h, nhead=num_heads,
            #     dim_feedforward=2048, dropout=0.1, activation=F.relu,
            #     layer_norm_eps=1e-5, batch_first=True)
        # elif global_model_type == 'Performer':
        #     self.self_attn = SelfAttention(
        #         dim=dim_h, heads=num_heads,
        #         dropout=self.attn_dropout, causal=False)
        else:
            raise ValueError(f"Unsupported global x-former model: "
                             f"{global_model_type}")
        self.global_model_type = global_model_type

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            # self.norm1_local = pygnn.norm.LayerNorm(dim_h)
            # self.norm1_attn = pygnn.norm.LayerNorm(dim_h)
            self.norm1_local = pygnn.norm.GraphNorm(dim_h)
            self.norm1_attn = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_local = pygnn.norm.InstanceNorm(dim_h)
            # self.norm1_attn = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
        self.activation = F.relu
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        if self.layer_norm:
            # self.norm2 = pygnn.norm.LayerNorm(dim_h)
            self.norm2 = pygnn.norm.GraphNorm(dim_h)
            # self.norm2 = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)
        
        
        self.structure_map_query = nn.Sequential(
            nn.Linear(dim_h, num_heads))
        self.structure_map_key = nn.Sequential(
            nn.Linear(dim_h, num_heads))
        # self.cnn = nn.Sequential(nn.Conv2d(1, num_heads, kernel_size=1), nn.ReLU(), nn.Conv2d(num_heads, num_heads, kernel_size=1))

    def forward(self, batch):
        h = batch.x
        h_in1 = h  # for first residual connection

        h_out_list = []
        # Local MPNN with edge attributes.
        if self.local_model is not None:
            self.local_model: pygnn.conv.MessagePassing  # Typing hint.
            if self.local_gnn_type == 'CustomGatedGCN':
                es_data = None
                if self.equivstable_pe:
                    es_data = batch.pe_EquivStableLapPE
                local_out = self.local_model(Batch(batch=batch,
                                                   x=h,
                                                   edge_index=batch.edge_index,
                                                   edge_attr=batch.edge_attr,
                                                   pe_EquivStableLapPE=es_data))
                # GatedGCN does residual connection and dropout internally.
                h_local = local_out.x
                batch.edge_attr = local_out.edge_attr
            else:
                if self.equivstable_pe:
                    h_local = self.local_model(h, batch.edge_index, batch.edge_attr,
                                               batch.pe_EquivStableLapPE)
                else:
                    h_local = self.local_model(h, batch.edge_index, batch.edge_attr)
                h_local = self.dropout_local(h_local)
                h_local = h_in1 + h_local  # Residual connection.

            if self.layer_norm:
                h_local = self.norm1_local(h_local, batch.batch)
            if self.batch_norm:
                h_local = self.norm1_local(h_local)
            h_out_list.append(h_local)

        # Multi-head attention.
        if self.self_attn is not None:
            h_dense, mask = to_dense_batch(h, batch.batch)
            # dist = 1.0 / dist
            # dist_conv = (self.structure_map_query(batch.dist),self.structure_map_key(batch.dist))
            dist_conv = self.structure_map_query(batch.dist)
            # dist_conv = batch.dist

            if self.global_model_type == 'Transformer':
                h_attn_ = self._sa_block(h_dense, None, ~mask, dist_conv)
                h_attn = h_attn_[0][mask]
                
                # if ((self.training)):
                # # if self.training is False:
                #     print(batch, batch.ptr.shape[0], self.training)
                #     for index in range(batch.ptr.shape[0]-1): 
                        
                #         h_attn_1 = h_attn_[0][index][mask[index]] 
                #         attn_1 = h_attn_[1][0][index][mask[index]][:,mask[index]] 
                #         attn_2 = h_attn_[1][1][index][mask[index]][:,mask[index]]
               

                #         global global_plot_counter  

                #         os.makedirs(save_dir, exist_ok=True)

                #         plt.clf() 
                #         plt.imshow(attn_1.cpu().detach().numpy(), cmap='viridis', interpolation='nearest')
                #         plt.colorbar() 
                #         plt.savefig(os.path.join(save_dir, f'attn_1_{index}_{global_plot_counter}.png'), dpi=1200)
                #         plt.savefig(os.path.join(save_dir, f'attn_1_{index}_{global_plot_counter}.eps'), format='eps', dpi=1200)
                #         plt.clf()

                #         plt.imshow(attn_2.cpu().detach().numpy(), cmap='viridis', interpolation='nearest')
                #         plt.colorbar()
                #         plt.savefig(os.path.join(save_dir, f'attn_2_{index}_{global_plot_counter}.png'), dpi=1200)
                #         plt.savefig(os.path.join(save_dir, f'attn_2_{index}_{global_plot_counter}.eps'), format='eps', dpi=1200)
                #         plt.clf()
                #         batch_egde_1 = unbatch_edge_index(batch.edge_index, batch.batch)[index] 
                #         graph = Data(x=h_attn_1, edge_index=batch_egde_1) 
                #         graph.tag = graph.x.argmax(dim=-1) 
                #         graph.attn1 = attn_1[0] 
                #         graph.attn2 = attn_2[0]
                        
                #         graph = utils.to_networkx(graph, node_attrs=['tag', 'attn1', 'attn2'], to_undirected=True)
                #         draw_graph_with_attn(
                #             graph,
                #             save_dir, 
                #             f'graph{index}_{global_plot_counter}.png',
                #             nodecolor=['tag', 'attn1', 'attn2']
                #         )
                #         global_plot_counter += 1  
            elif self.global_model_type == 'Performer':
                h_attn = self.self_attn(h_dense, mask=mask)[mask]
            elif self.global_model_type == 'BigBird':
                h_attn = self.self_attn(h_dense, attention_mask=mask)
            else:
                raise RuntimeError(f"Unexpected {self.global_model_type}")

            h_attn = self.dropout_attn(h_attn)
            h_attn = h_in1 + h_attn  # Residual connection.
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn, batch.batch)
            if self.batch_norm:
                h_attn = self.norm1_attn(h_attn)
            h_out_list.append(h_attn)

        # Combine local and global outputs.
        # h = torch.cat(h_out_list, dim=-1)
        h = sum(h_out_list)

        # Feed Forward block.
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)

        batch.x = h
        return batch

    def _sa_block(self, x, attn_mask, key_padding_mask, dist_conv):
        """Self-attention block.
        """
        x, _ = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True, dist=dist_conv)
        
        return x, _

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = f'summary: dim_h={self.dim_h}, ' \
            f'local_gnn_type={self.local_gnn_type}, ' \
            f'global_model_type={self.global_model_type}, ' \
            f'heads={self.num_heads}'
        return s




class CateFeatureEmbedding(nn.Module):
    def __init__(self, num_uniq_values, embed_dim, dropout=0.0):
        '''
        '''
        super().__init__()
        if len(num_uniq_values)==1:
            self.linear = nn.Linear(embed_dim, embed_dim)
        else:
            self.linear = nn.Linear(embed_dim*2, embed_dim)
        num_uniq_values = torch.LongTensor(num_uniq_values)
        csum = torch.cumsum(num_uniq_values, dim=0)
        num_emb = csum[-1]
        num_uniq_values = torch.LongTensor(num_uniq_values).reshape(1, 1, -1)
        self.register_buffer('num_uniq_values', num_uniq_values) 
        
        starts = torch.cat((torch.LongTensor([0]), csum[:-1])).reshape(1, -1)
        self.register_buffer('starts', starts)
        
        self.embeddings = nn.Embedding(
            num_emb, embed_dim)
        
        self.dropout_proba = dropout
        
        self.layer_norm_output = nn.LayerNorm(embed_dim)
        pass

    def forward(self, x):
        # x = x + 1
        if torch.any(x < 0):
            raise RuntimeError(str(x))

        if torch.any(torch.ge(x, self.num_uniq_values)):
            print(torch.max(x[:,:,:,0]))
            print(torch.max(x[:,:,:,1]))
            raise RuntimeError(str(x))
            pass
        
        x = x + self.starts
        
        if self.training:
            # x[torch.rand(size=x.shape, device=x.device) < self.dropout_proba] = 0
            pass
        # print(self.embeddings(x).shape)
        # emb = self.embeddings(x).sum(dim=-2)
        emb = self.embeddings(x)
        emb = torch.reshape(emb,(emb.shape[0],emb.shape[1],emb.shape[2],-1))
        emb = self.linear(emb)
        return emb # (batch_size,N,N,embed_dim)
    pass

class LinearNodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        
        self.encoder = torch.nn.Linear(62, emb_dim)

    def forward(self, batch):
        batch.x = self.encoder(batch.x)
        return batch
    
class LinearEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.encoder = torch.nn.Linear(21, emb_dim)

    def forward(self, batch):
        batch.edge_attr = self.encoder(batch.edge_attr)
        return batch
    
class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        self.walk_step = 20
        self.pe_lin =  torch.nn.Linear(self.walk_step, 72) 
        self.pe_norm = torch.nn.BatchNorm1d(self.walk_step)
        # Encode integer node features via nn.Embeddings
        self.node_encoder = LinearNodeEncoder(self.dim_in-72)
        
        self.edge_encoder = LinearEdgeEncoder(self.dim_in)
        
    def forward(self, batch):
        x_pe = self.pe_norm(batch.pestat_RWSE)
        x_pe = self.pe_lin(x_pe)
        batch = self.node_encoder(batch)
        batch = self.edge_encoder(batch)
        batch.x = torch.cat((batch.x, x_pe), dim=1)

        return batch

class EquivStableLapPENodeEncoder(torch.nn.Module):
    """Equivariant and Stable Laplace Positional Embedding node encoder.

    This encoder simply transforms the k-dim node LapPE to d-dim to be
    later used at the local GNN module as edge weights.
    Based on the approach proposed in paper https://openreview.net/pdf?id=e95i1IHcWj
    
    Args:
        dim_emb: Size of final node embedding
    """

    def __init__(self, dim_emb):
        super().__init__()


        max_freqs = 8
        norm_type = "none"  # Raw PE normalization layer type

        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(max_freqs)
        else:
            self.raw_norm = None

        self.linear_encoder_eigenvec = nn.Linear(max_freqs, dim_emb)

    def forward(self, batch):
        if not (hasattr(batch, 'EigVals') and hasattr(batch, 'EigVecs')):
            raise ValueError("Precomputed eigen values and vectors are "
                             f"required for {self.__class__.__name__}; set "
                             f"config 'posenc_EquivStableLapPE.enable' to True")
        pos_enc = batch.EigVecs

        empty_mask = torch.isnan(pos_enc)  # (Num nodes) x (Num Eigenvectors)
        pos_enc[empty_mask] = 0.  # (Num nodes) x (Num Eigenvectors)

        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)

        pos_enc = self.linear_encoder_eigenvec(pos_enc)

        # Keep PE separate in a variable
        batch.pe_EquivStableLapPE = pos_enc

        return batch


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d

cfg = dict_to_namespace(config)


class HDSE_MS_Model(torch.nn.Module):
    """Multi-scale graph x-former.
    """

    def __init__(self, dim_in=320, layers=None, layer_type="CustomGatedGCN+Transformer"):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        # self.lap_pe_encoder = EquivStableLapPENodeEncoder(dim_emb=cfg.gnn.dim_inner)
        self.lap_pe_encoder = None  # 
        dim_in = self.encoder.dim_in
      

        if layers is not None:
            cfg.gt.layers = layers
        if layer_type is not None:
            cfg.gt.layer_type = layer_type

        dim_in = cfg.gnn.dim_inner

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        try:
            local_gnn_type, global_model_type = cfg.gt.layer_type.split('+')
        except:
            raise ValueError(f"Unexpected layer type: {cfg.gt.layer_type}")
        # layers = []
        self.layers = nn.ModuleList()

        self.catelist = [11, 31]
        for _ in range(cfg.gt.layers):
            self.layers.append(UpdateLayer(
                dim_h=cfg.gt.dim_hidden,
                local_gnn_type=local_gnn_type,
                global_model_type=global_model_type,
                num_heads=cfg.gt.n_heads,
                dropout=cfg.gt.dropout,
                attn_dropout=cfg.gt.attn_dropout,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
            ))
        # self.layers = torch.nn.Sequential(*layers)
        self.layer_structure_embed_cate = CateFeatureEmbedding(self.catelist, cfg.gt.dim_hidden, dropout=0.1)
        # self.layer_structure_embed_cate = CateFeatureEmbedding(self.catelist, cfg.gt.dim_hidden, dropout=0.1)

    def _encode_nodes(self, data):
        batch = data.get("hdse", None)
        batch = self.encoder(batch)
        if self.lap_pe_encoder is not None:
            batch = self.lap_pe_encoder(batch)
        # print(batch)
        dist = to_dense_adj(batch.complete_edge_index, batch.batch, edge_attr = batch.complete_edge_dist)
        SPD = to_dense_adj(batch.complete_edge_index, batch.batch, edge_attr = batch.complete_edge_SPD)
        if(len(self.catelist)==1): 
            dist = torch.stack([SPD], axis=3)
            # print(dist.shape)
        else: 
            dist = torch.stack([dist,SPD], axis=3)

        batch.dist = self.layer_structure_embed_cate(dist)
        for module in self.layers:
            batch = module(batch)
        return batch

    def forward_context(self, data):
        batch = self._encode_nodes(data)
        return self._build_context(batch)

    @staticmethod
    def _build_context(batch):
        if not hasattr(batch, "hdse_community"):
            raise ValueError(
                "HDSE graph is missing hdse_community; regenerate it with "
                "HDSE_MS_preprocess"
            )
        required_fragment_fields = (
            "fragment_edge_index",
            "fragment_mz",
            "fragment_h_shift",
            "fragment_side",
            "fragment_is_inter_community",
        )
        missing_fragment_fields = [
            field
            for field in required_fragment_fields
            if not hasattr(batch, field)
        ]
        if missing_fragment_fields:
            raise ValueError(
                "HDSE graph is missing fragment candidates: "
                + ", ".join(missing_fragment_fields)
                + ". Augment or rebuild the graph cache."
            )
        context = build_hierarchy_context(
            batch.x,
            batch.batch,
            batch.hdse_community,
        )
        context.update(
            build_fragment_context(
                node_embeddings=batch.x,
                graph_ids=batch.batch,
                local_community_ids=batch.hdse_community,
                fragment_edge_index=batch.fragment_edge_index,
                fragment_mz=batch.fragment_mz,
                fragment_h_shift=batch.fragment_h_shift,
                fragment_side=batch.fragment_side,
                fragment_is_inter_community=(
                    batch.fragment_is_inter_community
                ),
            )
        )
        return context

    def forward_context_and_graph(self, data):
        """Encode nodes once and return graph pooling plus hierarchy tokens."""
        batch = self._encode_nodes(data)
        graph_embedding = global_mean_pool(batch.x, batch.batch)
        return graph_embedding, self._build_context(batch)

    def forward(self, data):
        batch = self._encode_nodes(data)
        return global_mean_pool(batch.x, batch.batch)
    
class HDSE_MS_Embedder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = HDSE_MS_Model(dim_in=kwargs.get("dim_in", 320), layers=kwargs.get("layers", 8), layer_type=kwargs.get("layer_type", "CustomGatedGCN+Transformer"))
        self.embed_dim = 320
    
    def get_embed_dim(self):

        return self.embed_dim

    def get_split_params(self):

        nopt_params, pt_params = [], []
        for k, v in self.named_parameters():
            nopt_params.append(v)
        return nopt_params, pt_params

    def forward(self, data):
        return self.model(data)

    def forward_context(self, data):
        return self.model.forward_context(data)

    def forward_context_and_graph(self, data):
        return self.model.forward_context_and_graph(data)




def mask_prediction_by_mass(raw_prediction, prec_mass_idx, prec_mass_offset, mask_value=0.):
    # adapted from NEIMS

    max_idx = raw_prediction.shape[1]
    assert torch.all(prec_mass_idx < max_idx) 
    idx = torch.arange(max_idx, device=prec_mass_idx.device) 
    mask = (
        idx.unsqueeze(0) <= (
            prec_mass_idx.unsqueeze(1) +
            prec_mass_offset)).float() 
    return mask * raw_prediction + (1. - mask) * mask_value 


def reverse_prediction(raw_prediction, prec_mass_idx, prec_mass_offset):
    # adapted from NEIMS
   
    batch_size = raw_prediction.shape[0]
    max_idx = raw_prediction.shape[1]
    assert torch.all(prec_mass_idx < max_idx)
    rev_prediction = torch.flip(raw_prediction, dims=(1,)) 

    offset_idx = torch.minimum(
        max_idx * torch.ones_like(prec_mass_idx),
        prec_mass_idx + prec_mass_offset + 1) 
    shifts = - (max_idx - offset_idx)
    gather_idx = torch.arange( 
        max_idx,
        device=raw_prediction.device).unsqueeze(0).expand(
        batch_size,
        max_idx)
    gather_idx = (gather_idx - shifts.unsqueeze(1)) % max_idx 
    offset_rev_prediction = torch.gather(rev_prediction, 1, gather_idx) 

    return offset_rev_prediction





class BottleneckBlock(nn.Module):
    """ from the NEIMS paper (uses LeakyReLU instead of ReLU) """

    def __init__(self, in_dim, out_dim, dropout):

        super(BottleneckBlock, self).__init__()
        bottleneck_factor = 0.5
        bottleneck_size = int(round(bottleneck_factor * out_dim))
        self.in_batch_norm = nn.BatchNorm1d(in_dim)
        self.in_activation = nn.LeakyReLU()
        self.in_linear = nn.Linear(in_dim, bottleneck_size)
        self.out_batch_norm = nn.BatchNorm1d(bottleneck_size)
        self.out_linear = nn.Linear(bottleneck_size, out_dim)
        self.out_activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):

        h = x
        h = self.in_batch_norm(h)
        h = self.in_activation(h)
        h = self.dropout(h)
        h = self.in_linear(h)
        h = self.out_batch_norm(h)
        h = self.out_activation(h)
        h = self.out_linear(h)
        return h


class Prediction(nn.Module):

    def __init__(self, dim_d, **kwargs):

        super(Prediction, self).__init__()
        self.g_dim = dim_d["g_dim"] 
        self.o_dim = dim_d["o_dim"] 
        for k, v in kwargs.items():
            setattr(self, k, v)
        # Preserve legacy module/state-dict names when users select FraHi.
        self.prediction_head = normalize_prediction_head(
            kwargs.get("prediction_head", "dense")
        )
        self.protect_baseline = bool(
            kwargs.get(
                "protect_baseline",
                self.prediction_head != "dense",
            )
        )
        self.register_buffer(
            "_use_baseline_fallback",
            torch.tensor(False, dtype=torch.bool),
            persistent=self.prediction_head != "dense",
        )
        self.embedders = nn.ModuleList([])
        if "pyg_data" in self.embed_types: # gps
            from gps_model import GPSEmbedder
            self.embedders.append(GPSEmbedder(**kwargs))
        if "hdse" in self.embed_types:
            self.embedders.append(HDSE_MS_Embedder(**kwargs))
        assert len(self.embedders) > 0, len(self.embedders)
        embeds_dims = [embedder.get_embed_dim() for embedder in self.embedders] 
        if self.g_dim > 0:
            embeds_dims.append(self.g_dim)
        if self.embed_dim == -1:
            self.embed_dim = sum(embeds_dims)
        if self.prediction_head not in {
                "dense", "plain_peak", "hipeak"}:
            raise ValueError(
                f"invalid prediction_head: {self.prediction_head}"
            )
        if self.embed_linear:
            self.embed_layers = nn.ModuleList(
                [
                    nn.Linear(embed_dim, self.embed_dim)
                    for embed_dim in embeds_dims
                ]
            )
        else:
            self.embed_layers = nn.ModuleList(
                [nn.Identity() for _ in embeds_dims]
            )
        self.ff_layers = nn.ModuleList([])
        self.ff_layers.append(
            nn.Linear(self.embed_dim, self.ff_h_dim)
        )
        for _ in range(self.ff_num_layers):
            self.ff_layers.append(
                BottleneckBlock(
                    self.ff_h_dim,
                    self.ff_h_dim,
                    self.dropout,
                )
            )
        self.forw_out_layer = nn.Linear(self.ff_h_dim, self.o_dim)
        self.rev_out_layer = nn.Linear(self.ff_h_dim, self.o_dim)
        self.out_gate = nn.Sequential(
            nn.Linear(self.ff_h_dim, self.o_dim),
            nn.Sigmoid(),
        )
        self.out_modules = [
            "ff_layers",
            "forw_out_layer",
            "rev_out_layer",
            "out_gate",
        ]

        if self.prediction_head != "dense":
            if len(self.embedders) != 1 or "hdse" not in self.embed_types:
                raise ValueError(
                    "peak prediction heads require exactly one HDSE embedder"
                )
            if self.g_dim <= 0:
                raise ValueError(
                    "peak prediction heads require global spectrum metadata"
                )
            mz_bin_res = float(kwargs.get("mz_bin_res", 0.1))
            mz_max = float(
                kwargs.get("mz_max", self.o_dim * mz_bin_res)
            )
            decoder_kwargs = {
                "hidden_dim": self.embedders[0].get_embed_dim(),
                "metadata_dim": self.g_dim,
                "num_layers": int(
                    kwargs.get("peak_decoder_layers", 2)
                ),
                "mz_max": mz_max,
                "mz_bin_res": mz_bin_res,
                "output_normalization": self.output_normalization,
                "gradient_sigmas_mz": kwargs.get(
                    "renderer_gradient_sigmas_mz",
                    (0.1, 0.5, 2.0, 5.0),
                ),
                "local_residual_log_ratio_max": float(
                    kwargs.get("local_residual_log_ratio_max", 0.5)
                ),
                "dropout": float(
                    kwargs.get("peak_decoder_dropout", 0.1)
                ),
            }
            self.plain_peak_decoder = FragmentCandidateDecoder(
                mode="plain_peak",
                **decoder_kwargs,
            )
            self.out_modules.append("plain_peak_decoder")
            if self.prediction_head == "hipeak":
                self.hipeak_decoder = HierarchicalResidualAdapter(
                    hidden_dim=self.embedders[0].get_embed_dim(),
                    metadata_dim=self.g_dim,
                    residual_log_ratio_max=float(
                        kwargs.get(
                            "hierarchical_residual_log_ratio_max",
                            0.15,
                        )
                    ),
                    adapter_ratio=float(
                        kwargs.get("hierarchical_adapter_ratio", 0.25)
                    ),
                    dropout=float(
                        kwargs.get("peak_decoder_dropout", 0.1)
                    ),
                )
                self.out_modules.append("hipeak_decoder")

    @staticmethod
    def _normalize_prediction(prediction, normalization):
        prediction = F.relu(prediction)
        if normalization == "l1":
            return F.normalize(prediction, p=1, dim=1)
        if normalization == "l2":
            return F.normalize(prediction, p=2, dim=1)
        if normalization == "none":
            return prediction
        raise ValueError(
            f"invalid output_normalization: {normalization}"
        )

    def _forward_dense_from_embeddings(self, embeddings, data):
        if self.g_dim > 0:
            embeddings.append(data["spec_meta"])
        embeddings = [
            self.embed_layers[index](embedding)
            for index, embedding in enumerate(embeddings)
        ]
        hidden = self.ff_layers[0](torch.cat(embeddings, dim=1))
        for ff_layer in self.ff_layers[1:]:
            hidden = hidden + ff_layer(hidden)
        forward_prediction = self.forw_out_layer(hidden)
        reverse = reverse_prediction(
            self.rev_out_layer(hidden),
            data["prec_mz_idx"],
            self.prec_mass_offset,
        )
        gate = self.out_gate(hidden)
        prediction = (
            forward_prediction * gate
            + reverse * (1.0 - gate)
        )
        prediction = mask_prediction_by_mass(
            prediction,
            data["prec_mz_idx"],
            self.prec_mass_offset,
        )
        prediction = self._normalize_prediction(
            prediction,
            self.output_normalization,
        )
        return prediction, hidden

    @staticmethod
    def _detach_context(value):
        if isinstance(value, torch.Tensor):
            return value.detach()
        if isinstance(value, dict):
            return {
                key: Prediction._detach_context(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [Prediction._detach_context(item) for item in value]
        if isinstance(value, tuple):
            return tuple(
                Prediction._detach_context(item) for item in value
            )
        return value

    def set_inference_prediction(self, source):
        """Select the persisted evaluation output for a protected model."""
        if source not in {"full", "baseline"}:
            raise ValueError("source must be 'full' or 'baseline'")
        if source == "baseline" and self.prediction_head == "dense":
            raise ValueError("dense models do not have a residual prediction")
        self._use_baseline_fallback.fill_(source == "baseline")

    def inference_prediction_source(self):
        if self.prediction_head == "dense":
            return "full"
        return (
            "baseline"
            if bool(self._use_baseline_fallback.item())
            else "full"
        )

    def _select_prediction(self, full_prediction, baseline_prediction):
        if (
            not self.training
            and bool(self._use_baseline_fallback.item())
        ):
            return baseline_prediction
        return full_prediction

    @staticmethod
    def _checkpoint_head(state_dict):
        keys = tuple(state_dict)
        if any(key.startswith("hipeak_decoder.") for key in keys):
            return "hipeak"
        if any(key.startswith("plain_peak_decoder.") for key in keys):
            return "plain_peak"
        if any(key.startswith("peak_decoder.") for key in keys):
            return "legacy_peak"
        return "dense"

    def load_checkpoint(self, state_dict):
        """Load only a checkpoint produced by the same model architecture."""
        source_head = self._checkpoint_head(state_dict)
        if source_head != self.prediction_head:
            raise ValueError(
                "checkpoint loading requires the same architecture: "
                f"expected {self.prediction_head}, found {source_head}"
            )
        self.load_state_dict(state_dict, strict=True)

    def configure_joint_trainability(self):
        """Train the protected baseline and residual branches together."""
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        return [
            name for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]

    def protected_parameter_groups(self):
        """Partition parameters so residual clipping cannot scale baseline grads."""
        baseline_modules = (
            self.embedders,
            self.embed_layers,
            self.ff_layers,
            self.forw_out_layer,
            self.rev_out_layer,
            self.out_gate,
        )
        baseline = [
            parameter
            for module in baseline_modules
            for parameter in module.parameters()
        ]
        baseline_ids = {id(parameter) for parameter in baseline}
        residual = [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in baseline_ids
        ]
        return {"baseline": baseline, "residual": residual}

    def _forward_joint_residual(
            self,
            residual_context,
            residual_anchor,
            dense_prediction,
            data):
        plain_output = self.plain_peak_decoder(
            residual_context,
            data["spec_meta"],
            data["prec_mz_value"],
        )
        plain_prediction, plain_local_residual = (
            apply_fragment_log_residual(
                anchor=residual_anchor,
                peak_mz=plain_output["peak_mz"],
                peak_intensity=plain_output["peak_intensity"],
                peak_log_ratio=plain_output["peak_log_ratio"],
                candidate_mask=plain_output["peak_candidate_mask"],
                mz_max=self.mz_max,
                mz_bin_res=self.mz_bin_res,
                output_normalization=self.output_normalization,
            )
        )
        if self.prediction_head == "plain_peak":
            output = dict(plain_output)
            output.update({
                "pred": self._select_prediction(
                    plain_prediction,
                    dense_prediction,
                ),
                "full_pred": plain_prediction,
                "baseline_pred": dense_prediction,
                "plain_peak_pred": plain_output["pred"],
                "plain_local_residual": plain_local_residual,
            })
            return output

        hipeak_output = self.hipeak_decoder(
            residual_context,
            data["spec_meta"],
            data["prec_mz_value"],
            plain_output,
        )
        hipeak_prediction, hipeak_local_residual = (
            apply_fragment_log_residual(
                anchor=plain_prediction,
                peak_mz=hipeak_output["peak_mz"],
                peak_intensity=hipeak_output["peak_intensity"],
                peak_log_ratio=hipeak_output["peak_log_ratio"],
                candidate_mask=hipeak_output["peak_candidate_mask"],
                mz_max=self.mz_max,
                mz_bin_res=self.mz_bin_res,
                output_normalization=self.output_normalization,
            )
        )
        output = dict(hipeak_output)
        output.update({
            "pred": self._select_prediction(
                hipeak_prediction,
                dense_prediction,
            ),
            "full_pred": hipeak_prediction,
            "baseline_pred": dense_prediction,
            "plain_peak_pred": plain_output["pred"],
            "local_residual_pred": plain_prediction,
            "plain_local_residual": plain_local_residual,
            "hipeak_local_residual": hipeak_local_residual,
        })
        return output

    def forward(self, data, amp=False):

        if amp:
            amp_context = torch.cuda.amp.autocast()
        else:
            amp_context = DummyContext()
        with amp_context:
            if self.prediction_head == "dense":
                embeddings = [
                    embedder(data) for embedder in self.embedders
                ]
                prediction, _ = self._forward_dense_from_embeddings(
                    embeddings,
                    data,
                )
                return {"pred": prediction}

            graph_embedding, context = (
                self.embedders[0].forward_context_and_graph(data)
            )
            dense_prediction, hidden = (
                self._forward_dense_from_embeddings(
                    [graph_embedding],
                    data,
                )
            )
            if self.protect_baseline:
                residual_context = self._detach_context(context)
                residual_anchor = dense_prediction.detach()
            else:
                residual_context = context
                residual_anchor = dense_prediction
            rng_devices = (
                [dense_prediction.device.index]
                if dense_prediction.is_cuda
                else []
            )
            with torch.random.fork_rng(
                    devices=rng_devices,
                    enabled=self.protect_baseline):
                return self._forward_joint_residual(
                    residual_context,
                    residual_anchor,
                    dense_prediction,
                    data,
                )



    def count_parameters(self):
        def is_initialized(param):
            return not isinstance(param, torch.nn.parameter.UninitializedParameter)

        mol_params = 0
        for embedder in self.embedders:
            mol_params += sum(p.numel() for p in embedder.parameters() if is_initialized(p))
        for embed_layer in self.embed_layers:
            mol_params += sum(p.numel() for p in embed_layer.parameters() if is_initialized(p))

        mlp_params = 0
        for out_module in self.out_modules:
            mlp_params += sum(p.numel() for p in getattr(self, out_module).parameters() if is_initialized(p))

        total_params = sum(p.numel() for p in self.parameters() if is_initialized(p))
        return mol_params, mlp_params, total_params
    def set_mode(self, mode):
        self.mode = mode
