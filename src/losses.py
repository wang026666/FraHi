import torch as th
import torch.nn.functional as F
import math

from data_utils import EPS


def kl(p, q): return th.sum(p * (th.log(p + EPS) - th.log(q + EPS)), dim=1)
def mse(x, y): return 0.5 * th.mean((x - y)**2, dim=1)
def normal_nll(x, y): return -th.sum(-math.log(0.08 * math.sqrt(2 * math.pi)) - 0.5 * (x - y)**2 / (0.08**2), dim=1)
ln2 = math.log(2)
def compute_weights(x, mz_bin_res):
    mass_pow = 0.5
    weights = (
        mz_bin_res *
        th.arange(
            1,
            x.shape[1] +
            1,
            device=x.device,
            dtype=th.float32))**mass_pow
    weights = (weights / th.sum(weights)).unsqueeze(0)
    return weights

def get_loss_func(loss_type, mz_bin_res, agg=None):
    if loss_type == "mse":
        loss_func = mse
    elif loss_type == "wmse":
        def w_mse(pred, targ):
            weights = compute_weights(pred, mz_bin_res)
            return th.sum(weights * (pred - targ)**2, dim=1)
        loss_func = w_mse
    elif loss_type == "js":
        def js(pred, targ):
            pred = F.normalize(pred, dim=1, p=1)
            targ = F.normalize(targ, dim=1, p=1)
            z = 0.5 * (pred + targ)
            return th.sqrt(F.relu(0.5 * kl(pred, z) + 0.5 * kl(targ, z)))
        loss_func = js
    elif loss_type == "forw_kl":
        def forw_kl(pred, targ):
            pred = F.normalize(pred, dim=1, p=1)
            targ = F.normalize(targ, dim=1, p=1)
            return kl(targ, pred)
        loss_func = forw_kl
    elif loss_type == "rev_kl":
        def rev_kl(pred, targ):
            pred = F.normalize(pred, dim=1, p=1)
            targ = F.normalize(targ, dim=1, p=1)
            return kl(pred, targ)
        loss_func = rev_kl
    elif loss_type == "normal_nll":
        def myloss(input, target):
            from torch.distributions import Normal
            nd = Normal(input, 0.08)
            pdf_term = nd.log_prob(target)
            return -th.sum(pdf_term, dim=1)
        loss_func = myloss
    elif loss_type == "wass":
        def wass(pred, targ):
            pred = F.normalize(pred, dim=1, p=1)
            targ = F.normalize(targ, dim=1, p=1)
            pred_cdf = th.cumsum(pred, dim=1)
            targ_cdf = th.cumsum(targ, dim=1)
            return th.sum(th.abs(pred_cdf - targ_cdf), dim=1)
        loss_func = wass
    elif loss_type == "cos":
        def cos(pred, targ):
            pred = F.normalize(pred, dim=1, p=2).unsqueeze(1)
            targ = F.normalize(targ, dim=1, p=2).unsqueeze(2)
            return 1. - th.matmul(pred, targ).squeeze(-1).squeeze(-1)
        loss_func = cos
    elif loss_type == "wcos":
        def w_cos(pred, targ):
            weights = compute_weights(pred, mz_bin_res)
            w_pred = F.normalize(weights * pred, dim=1, p=2).unsqueeze(1)
            w_targ = F.normalize(weights * targ, dim=1, p=2).unsqueeze(2)
            return 1. - th.matmul(w_pred, w_targ).squeeze(-1).squeeze(-1)
        loss_func = w_cos
    else:
        raise NotImplementedError
    if not (agg is None):
        if agg == "mean":
            def loss_func_agg(p, t): return th.mean(loss_func(p, t), dim=0)
        elif agg == "sum":
            def loss_func_agg(p, t): return th.sum(loss_func(p, t), dim=0)
    else:
        loss_func_agg = loss_func
    return loss_func_agg

def _relative_noise_filter(spec, noise_threshold):
    """Remove peaks below a spectrum-relative intensity threshold."""
    spec = th.clamp(spec, min=0.0)
    if noise_threshold <= 0:
        return spec
    max_intensity = th.max(spec, dim=1, keepdim=True).values
    keep = (max_intensity > 0) & (
        spec >= noise_threshold * max_intensity
    )
    return th.where(keep, spec, th.zeros_like(spec))


def _entropy_weighted_probability(spec, noise_threshold):
    probability = F.normalize(
        _relative_noise_filter(spec, noise_threshold),
        p=1,
        dim=1,
    )
    entropy = -th.sum(th.xlogy(probability, probability), dim=1)
    exponent = th.where(
        entropy < 3.0,
        0.25 + 0.25 * entropy,
        th.ones_like(entropy),
    )
    weighted = th.pow(probability, exponent.unsqueeze(1))
    return F.normalize(weighted, p=1, dim=1)


def spectral_entropy_similarity(pred, targ, noise_threshold=0.01):
    """Return entropy similarity for aligned, binned mass spectra."""
    pred = _entropy_weighted_probability(pred, noise_threshold)
    targ = _entropy_weighted_probability(targ, noise_threshold)
    mixture = 0.5 * (pred + targ)
    pred_entropy = -th.sum(th.xlogy(pred, pred), dim=1)
    targ_entropy = -th.sum(th.xlogy(targ, targ), dim=1)
    mixture_entropy = -th.sum(th.xlogy(mixture, mixture), dim=1)
    divergence = (
        mixture_entropy - 0.5 * pred_entropy - 0.5 * targ_entropy
    )
    similarity = 1.0 - divergence / ln2
    nonempty = (
        (th.sum(pred, dim=1) > 0) & (th.sum(targ, dim=1) > 0)
    )
    return th.where(
        nonempty,
        th.clamp(similarity, min=0.0, max=1.0),
        th.zeros_like(similarity),
    )


def modified_cosine_similarity(
        pred,
        targ,
        mz_bin_res,
        tolerance=0.1,
        noise_threshold=0.01):
    """Return greedy one-to-one, tolerance-aware cosine similarity."""
    if mz_bin_res <= 0:
        raise ValueError("mz_bin_res must be greater than zero")
    if tolerance < 0:
        raise ValueError("modified cosine tolerance cannot be negative")

    pred = _relative_noise_filter(pred, noise_threshold)
    targ = _relative_noise_filter(targ, noise_threshold)
    similarities = []
    tolerance_bins = int(math.floor(tolerance / mz_bin_res + 1e-7))
    for pred_row, targ_row in zip(pred, targ):
        pred_indices = th.nonzero(
            pred_row > 0, as_tuple=False
        ).flatten()
        targ_indices = th.nonzero(
            targ_row > 0, as_tuple=False
        ).flatten()
        denominator = (
            th.linalg.vector_norm(pred_row)
            * th.linalg.vector_norm(targ_row)
        )
        if (
            pred_indices.numel() == 0
            or targ_indices.numel() == 0
            or denominator <= 0
        ):
            similarities.append(pred_row.new_zeros(()))
            continue

        offsets = th.arange(
            -tolerance_bins,
            tolerance_bins + 1,
            device=targ_indices.device,
        )
        candidate_targ = targ_indices.repeat_interleave(offsets.numel())
        candidate_pred = (
            targ_indices.unsqueeze(1) + offsets.unsqueeze(0)
        ).reshape(-1)
        valid = (
            (candidate_pred >= 0)
            & (candidate_pred < pred_row.numel())
        )
        candidate_pred = candidate_pred[valid]
        candidate_targ = candidate_targ[valid]
        nonzero_pred = pred_row[candidate_pred] > 0
        candidate_pred = candidate_pred[nonzero_pred]
        candidate_targ = candidate_targ[nonzero_pred]
        if candidate_pred.numel() == 0:
            similarities.append(pred_row.new_zeros(()))
            continue

        candidate_scores = (
            pred_row[candidate_pred] * targ_row[candidate_targ]
        )
        order = th.argsort(candidate_scores, descending=True, stable=True)
        used_pred = set()
        used_targ = set()
        matched_score = pred_row.new_zeros(())
        for candidate_index in order.tolist():
            pred_index = int(candidate_pred[candidate_index])
            targ_index = int(candidate_targ[candidate_index])
            if pred_index in used_pred or targ_index in used_targ:
                continue
            used_pred.add(pred_index)
            used_targ.add(targ_index)
            matched_score = (
                matched_score + candidate_scores[candidate_index]
            )
        similarities.append(
            th.clamp(matched_score / denominator, min=0.0, max=1.0)
        )
    return th.stack(similarities)


def get_sim_func(
        sim_type,
        mz_bin_res,
        tolerance=0.1,
        noise_threshold=0.01):
    if sim_type in {"cos", "cosine"}:
        def cos(pred, targ):
            n_pred = F.normalize(pred, p=2, dim=1).unsqueeze(1)
            n_targ = F.normalize(targ, p=2, dim=1).unsqueeze(2)
            return th.bmm(n_pred, n_targ).squeeze(-1).squeeze(-1)
        sim_func = cos
    elif sim_type == "modified_cosine":
        def modified_cosine(pred, targ):
            return modified_cosine_similarity(
                pred,
                targ,
                mz_bin_res=mz_bin_res,
                tolerance=tolerance,
                noise_threshold=noise_threshold,
            )
        sim_func = modified_cosine
    elif sim_type == "spectral_entropy":
        def spectral_entropy(pred, targ):
            return spectral_entropy_similarity(
                pred,
                targ,
                noise_threshold=noise_threshold,
            )
        sim_func = spectral_entropy
    elif sim_type == "wcos":
        def w_cos(pred, targ):
            weights = compute_weights(pred, mz_bin_res)
            n_pred = F.normalize(weights * pred, p=2, dim=1).unsqueeze(1)
            n_targ = F.normalize(weights * targ, p=2, dim=1).unsqueeze(2)
            return th.bmm(n_pred, n_targ).squeeze(-1).squeeze(-1)
        sim_func = w_cos
    elif sim_type == "wass":
        def wass(pred, targ):
            pred = F.normalize(pred, dim=1, p=1)
            targ = F.normalize(targ, dim=1, p=1)
            pred_cdf = th.cumsum(pred, dim=1)
            targ_cdf = th.cumsum(targ, dim=1)
            max_val = float(pred.shape[1] - 1)
            return (max_val - th.sum(th.abs(pred_cdf - targ_cdf), dim=1)) / max_val
        sim_func = wass
    elif sim_type == "sdot":
        def s_dot(pred, targ):
            pred = (pred**0.5).unsqueeze(1)
            targ = (targ**0.5).unsqueeze(2)
            numerator = th.matmul(pred, targ).squeeze(-1).squeeze(-1)
            denominator = th.sum(pred, dim=1) * th.sum(targ, dim=1)
            return numerator / th.clamp(denominator, min=EPS)
        sim_func = s_dot
    elif sim_type == "wsdot":
        def w_s_dot(pred, targ):
            weights = compute_weights(pred, mz_bin_res)
            pred = ((weights * pred)**0.5).unsqueeze(1)
            targ = ((weights * targ)**0.5).unsqueeze(2)
            numerator = th.matmul(pred, targ).squeeze(-1).squeeze(-1)
            denominator = th.sum(pred, dim=1) * th.sum(targ, dim=1)
            return numerator / th.clamp(denominator, min=EPS)
        sim_func = w_s_dot
    elif sim_type == "js":
        def js(pred, targ):
            pred = F.normalize(pred, dim=1, p=1)
            targ = F.normalize(targ, dim=1, p=1)
            z = 0.5 * (pred + targ)
            return ln2 - (0.5 * kl(pred, z) + 0.5 * kl(targ, z))
        sim_func = js
    elif sim_type == "jacc":
        def jacc(pred, targ):
            pred = pred > 0.
            targ = targ > 0.
            numerator = th.bmm(
                pred.float().unsqueeze(1), 
                targ.float().unsqueeze(2)
            ).squeeze(-1).squeeze(-1)
            denominator = th.sum((pred | targ).float(), dim=1)
            return numerator / th.clamp(denominator, min=EPS)
        sim_func = jacc
    else:
        raise ValueError
    return sim_func
