from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn

@dataclass(frozen=True)
class GeneratorConfig:
    history_length: int = 30
    horizon: int = 5
    width: int = 16
    timesteps: int = 40
    steps: int = 4800
    batch: int = 8
    beta: float = 1.0
    rho: float = 0.2
    jitter: float = 1e-6
    scale_floor: float = 1e-3
    min_reference_observations: int = 30
    min_history_observations: int = 2
    learning_rate: float = 0.001
    gradient_clip: float = 5.0


# ═══ A. 六通道表示与训练期归一化 ═══════════════════════════════════════
FEATURES = ("open_gap_log_return", "close_open_log_return", "upper_log_range",
            "lower_log_range", "log1p_volume", "log1p_amount")

def encode_features(open_, high, low, close, previous_close, volume, amount):
    """Encode fully observed bars; broadcasting is supported, invalid bars fail.

    Prices must share the same backward-adjustment convention. Volume and
    amount retain the vendor's units. No returns, ranges or volumes are clipped.
    """
    o, h, l, c, p, v, a = np.broadcast_arrays(*(
        np.asarray(x, dtype=np.float64)
        for x in (open_, high, low, close, previous_close, volume, amount)
    ))
    good = (np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
            & np.isfinite(p) & np.isfinite(v) & np.isfinite(a)
            & (o > 0) & (h > 0) & (l > 0) & (c > 0) & (p > 0)
            & (h >= np.maximum(o, c)) & (l <= np.minimum(o, c))
            & (v >= 0) & (a >= 0))
    if not np.all(good):
        raise ValueError("Invalid OHLCV bar supplied to feature encoder")
    return np.stack((np.log(o / p), np.log(c / o),
                     np.log(h / np.maximum(o, c)),
                     np.log(np.minimum(o, c) / l), np.log1p(v), np.log1p(a)), axis=-1)

def decode_features(features, previous_close, time_axis: int = -3) -> dict:
    """Decode physical features into OHLCV paths, recursively along time_axis.

    Default shape is [..., time, asset, 6], with previous_close [..., asset].
    The last four feature channels must already be nonnegative. Models may
    produce them through a square-coordinate map; this decoder never clips.
    All prices use the anchor's same adjusted-price coordinate. Overflow fails
    explicitly instead of silently replacing or deleting generated samples.
    """
    x = np.asarray(features, dtype=np.float64)
    if x.ndim < 2 or x.shape[-1] != 6 or not np.isfinite(x).all():
        raise ValueError("Expected finite [..., time, ..., 6] features")
    axis = time_axis % x.ndim
    if axis == x.ndim - 1:
        raise ValueError("The feature axis cannot be the time axis")
    if np.any(x[..., 2:] < 0):
        raise ValueError("The four physical range/volume/amount channels must be nonnegative")
    seq = np.moveaxis(x, axis, 0)
    shape = seq.shape[1:-1]
    prev = np.broadcast_to(np.asarray(previous_close, dtype=np.float64), shape).copy()
    if not np.isfinite(prev).all() or np.any(prev <= 0):
        raise ValueError("A finite positive anchor close is required")
    result = {key: np.empty(seq.shape[:-1], np.float64)
              for key in ("open", "high", "low", "close", "volume", "amount")}
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for t, bar in enumerate(seq):
                o = prev * np.exp(bar[..., 0])
                c = o * np.exp(bar[..., 1])
                result["open"][t] = o
                result["close"][t] = c
                result["high"][t] = np.maximum(o, c) * np.exp(bar[..., 2])
                result["low"][t] = np.minimum(o, c) * np.exp(-bar[..., 3])
                result["volume"][t] = np.expm1(bar[..., 4])
                result["amount"][t] = np.expm1(bar[..., 5])
                prev = c
    except FloatingPointError as exc:
        raise ValueError("Nonfinite generated OHLCV path") from exc
    for key, values in result.items():
        if not np.isfinite(values).all() or (key in ("open", "high", "low", "close")
                                              and np.any(values <= 0)):
            raise ValueError("Nonfinite or nonpositive generated OHLCV prices")
        result[key] = np.moveaxis(values, 0, axis)
    return result

def _broadcast_mask(mask, shape, name):
    mask = np.asarray(mask, dtype=bool)
    if mask.shape == shape[:-1]:
        return np.broadcast_to(mask[..., None], shape)
    if mask.shape != shape:
        raise ValueError(f"{name} must have shape {shape[:-1]} or {shape}")
    return mask

def _inputs(H, H_mask, eligible, Y=None, Y_mask=None):
    H = np.asarray(H, dtype=np.float32)
    if H.ndim != 4 or min(H.shape) < 1 or H.shape[-1] != len(FEATURES):
        raise ValueError("H must be a nonempty [A,W,K,6] array")
    H_mask = _broadcast_mask(H_mask, H.shape, "H_mask")
    eligible = np.asarray(eligible, dtype=bool)
    if eligible.shape != (len(H), H.shape[2]):
        raise ValueError("eligible must have shape [N,K]")
    if not np.isfinite(H[H_mask]).all():
        raise ValueError("Observed history must be finite")
    if np.any(H[..., 2:][H_mask[..., 2:]] < 0):
        raise ValueError("Observed upper/lower/volume/amount channels must be nonnegative")
    H = np.where(H_mask, H, 0.0)
    if Y is None:
        return H, H_mask, eligible
    Y = np.asarray(Y, dtype=np.float32)
    if Y.ndim != 4 or Y.shape[0] != len(H) or Y.shape[2:] != H.shape[2:] or Y.shape[1] < 1:
        raise ValueError("Y must have shape [A,L,K,6] matching H")
    Y_mask = _broadcast_mask(Y_mask, Y.shape, "Y_mask")
    if not np.isfinite(Y[Y_mask]).all():
        raise ValueError("Observed targets must be finite")
    if np.any(Y[..., 2:][Y_mask[..., 2:]] < 0):
        raise ValueError("Observed positive target channels must be nonnegative")
    effective = Y_mask & eligible[:, None, :, None]
    if not effective.any():
        raise ValueError("No eligible observed targets")
    return H, H_mask, eligible, np.where(Y_mask, Y, 0.0), effective

def to_model_coordinates(values):
    """Keep signed returns; take the positive square root of four positive fields."""
    result = np.array(values, dtype=np.float32, copy=True)
    if result.shape[-1] != len(FEATURES) or not np.isfinite(result).all():
        raise ValueError("Finite six-channel physical features required")
    if (result[..., 2:] < 0).any():
        raise ValueError("Physical positive channels cannot be negative")
    result[..., 2:] = np.sqrt(result[..., 2:])
    return result

def from_model_coordinates(values):
    result = np.array(values, dtype=np.float32, copy=True)
    if result.shape[-1] != len(FEATURES) or not np.isfinite(result).all():
        raise ValueError("Finite six-channel model coordinates required")
    with np.errstate(over="raise", invalid="raise"):
        result[..., 2:] = np.square(result[..., 2:])
    return result

class OHLCVNormalizer:
    def __init__(self, H, H_mask, Y, Y_mask, eligible):
        H, H_mask, eligible, Y, effective = _inputs(H, H_mask, eligible, Y, Y_mask)
        history = to_model_coordinates(H)
        future = to_model_coordinates(Y)
        history_mask = H_mask & eligible[:, None, :, None]
        self.history_length, self.horizon, self.assets = H.shape[1], Y.shape[1], H.shape[2]
        means, scales = [], []
        for channel in range(len(FEATURES)):
            observed = np.concatenate((history[..., channel][history_mask[..., channel]],
                                       future[..., channel][effective[..., channel]])).astype(np.float64)
            if not len(observed):
                raise ValueError(f"No observed training values for channel {channel}")
            means.append(float(observed.mean()))
            scales.append(float(max(observed.std(), 1e-6)))
        self.mean, self.scale = means, scales

    def context(self, H, H_mask):
        if H.shape[1:] != (self.history_length, self.assets, len(FEATURES)):
            raise ValueError("History dimensions differ from fitted panel")
        H_mask = _broadcast_mask(H_mask, H.shape, "H_mask")
        physical = np.where(H_mask, H, 0.0)
        normalized = ((to_model_coordinates(physical) - np.asarray(self.mean, dtype=np.float32))
                      / np.asarray(self.scale, dtype=np.float32))
        normalized = np.where(H_mask, normalized, 0.0)
        return np.concatenate((normalized, H_mask.astype(np.float32)), axis=-1)

    def encode(self, Y):
        return ((to_model_coordinates(Y) - np.asarray(self.mean, dtype=np.float32))
                / np.asarray(self.scale, dtype=np.float32))

    def decode(self, paths):
        coordinates = (np.asarray(paths, dtype=np.float32) * np.asarray(self.scale, dtype=np.float32)
                       + np.asarray(self.mean, dtype=np.float32))
        return from_model_coordinates(coordinates)



class HistoricalVolatilityScaling:
    """B1：教师期唯一日参考协方差 + 当前历史尺度；不接收真实未来。"""
    def __init__(self, normalizer, reference_data, config: GeneratorConfig):
        self.config = {key: getattr(config, key) for key in (
            "rho", "jitter", "scale_floor", "min_reference_observations", "min_history_observations")}
        if not 0 < config.rho <= 1 or min(config.jitter, config.scale_floor) <= 0:
            raise ValueError("rho must be in (0,1]; jitter/scale_floor must be positive")
        self.normalizer = normalizer
        if not isinstance(reference_data, dict) or set(reference_data) != {"values", "mask", "dates"}:
            raise ValueError("reference_data requires exactly values, mask and dates")
        values = np.asarray(reference_data["values"], dtype=np.float32)
        if values.ndim != 3 or values.shape[1:] != (normalizer.assets, 6):
            raise ValueError("Reference values must be unique daily [D,K,6] features")
        mask = np.asarray(reference_data["mask"], dtype=bool)
        if mask.shape == values.shape[:-1]:
            mask = np.broadcast_to(mask[..., None], values.shape).copy()
        if mask.shape != values.shape or not np.isfinite(values[mask]).all():
            raise ValueError("Invalid observed reference values/mask")
        dates = np.asarray(reference_data["dates"], dtype="datetime64[D]")
        if dates.shape != (len(values),) or len(dates) < 2 or np.isnat(dates).any():
            raise ValueError("Reference dates must identify every unique daily row")
        if np.any(dates[1:] <= dates[:-1]):
            raise ValueError("Reference dates must be strictly increasing and unique")
        if dates[0] < np.datetime64("2019-01-01") or dates[-1] > np.datetime64("2021-12-31"):
            raise ValueError("D reference may use only teacher dates 2019--2021")
        s = np.asarray(normalizer.scale[:2], dtype=np.float64)
        self.w = s / np.linalg.norm(s)
        q = self._q(values)
        paired = mask[..., 0] & mask[..., 1]
        counts = paired.sum(0)
        means = np.where(paired, q, 0.).sum(0) / np.maximum(counts, 1)
        residual = np.where(paired, q - means[None], 0.)
        pooled_dof = np.maximum(counts - 1, 0).sum()
        if pooled_dof == 0:
            raise ValueError("No reference residual variance is identifiable")
        pooled = float(np.square(residual).sum() / pooled_dof)
        pooled = max(pooled, self.config["scale_floor"] ** 2)
        good = counts >= self.config["min_reference_observations"]
        E = residual / np.sqrt(np.maximum(counts - 1, 1))[None]
        E[:, ~good] = 0.
        covariance = E.T @ E
        covariance[np.diag_indices_from(covariance)] += (~good) * pooled + self.config["jitter"]
        self.covariance = np.ascontiguousarray(covariance, dtype=np.float64)
        self.reference_fallback_count = int((~good).sum())
        self._validate()

    def _q(self, values):
        # Normalizer location does not alter centered covariance, but including it
        # makes this definition exactly the physical return direction of x.
        s = np.asarray(self.normalizer.scale[:2], dtype=np.float64)
        return ((values[..., 0].astype(np.float64) + values[..., 1].astype(np.float64)
                 - self.normalizer.mean[0] - self.normalizer.mean[1]) / np.linalg.norm(s))

    def _validate(self):
        k = self.normalizer.assets
        if self.covariance.shape != (k, k) or not np.isfinite(self.covariance).all():
            raise ValueError("Invalid reference covariance shape or finiteness")
        if not np.allclose(self.covariance, self.covariance.T, atol=1e-12, rtol=1e-12):
            raise ValueError("Reference covariance is not symmetric")
        try:
            np.linalg.cholesky(self.covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Reference covariance must be positive definite") from exc
        scales = np.asarray(self.normalizer.scale[:2], dtype=float)
        if not np.allclose(self.w, scales / np.linalg.norm(scales), atol=1e-15, rtol=1e-15):
            raise ValueError("Reference direction differs from teacher normalization")

    def scales(self, H, H_mask, eligible):
        H, H_mask, eligible = _inputs(H, H_mask, eligible)
        if H.shape[2] != self.normalizer.assets:
            raise ValueError("History stock coordinates differ from reference")
        paired = H_mask[..., 0] & H_mask[..., 1] & eligible[:, None]
        q = self._q(H)
        count = paired.sum(1)
        mean = np.where(paired, q, 0.).sum(1) / np.maximum(count, 1)
        good = count >= self.config["min_history_observations"]
        E = np.where(paired & good[:, None], q - mean[:, None], 0.) / np.sqrt(np.maximum(count - 1, 1))[:, None]
        fallback = eligible & ~good
        diagonal = np.diag(self.covariance)
        fallback_trace = (fallback * diagonal[None]).sum(1)
        n = eligible.sum(1)
        trace_h = np.square(E).sum((1, 2)) + fallback_trace
        common_h = (np.square(E.sum(2)).sum(1) + fallback_trace) / np.maximum(n, 1)
        trace_ref = (eligible * diagonal[None]).sum(1)
        # O(K^2) per history only, precomputed before training; B/loss remain O(K).
        reference_common = np.einsum("bi,ij,bj->b", eligible.astype(float), self.covariance,
                                     eligible.astype(float), optimize=True) / np.maximum(n, 1)
        rho, jitter = self.config["rho"], self.config["jitter"]
        trace = (1-rho)*trace_h + rho*trace_ref + jitter*n
        common_var = (1-rho)*common_h + rho*reference_common + jitter
        residual_var = np.where(n > 1, (trace-common_var) / np.maximum(n-1, 1), 1.)
        tolerance = 1e-10 * np.maximum(trace, 1.)
        if np.any(common_var[n > 0] <= 0) or np.any(residual_var[n > 1] < -tolerance[n > 1]):
            raise FloatingPointError("Invalid history preconditioner variance")
        floor = self.config["scale_floor"] ** 2
        sc = np.sqrt(np.maximum(common_var, floor))
        su = np.sqrt(np.maximum(residual_var, floor))
        sc[n == 0], su[n <= 1] = 1., 1.
        if not np.isfinite(sc).all() or not np.isfinite(su).all():
            raise FloatingPointError("Nonfinite history preconditioner scale")
        eigen = np.stack((np.ones_like(sc), sc ** 2, su ** 2), axis=1)
        return {"sc": sc, "su": su, "active_stocks": n,
                "history_fallback_stocks": fallback.sum(1),
                "history_observed_pairs": paired.sum((1, 2)),
                "all_active_observed_days": (paired.sum(2) == n[:, None]).sum(1),
                "history_pair_coverage": paired.sum((1, 2)) / np.maximum(n * H.shape[1], 1),
                "common_floor_applied": (common_var < floor) & (n > 0),
                "residual_floor_applied": (residual_var < floor) & (n > 1),
                "B_squared_condition_bound": eigen.max(1) / eigen.min(1)}

def scale_coordinates(values, active, sc, su, w, inverse=False):
    if values.ndim != 4 or values.shape[-1] != 6:
        raise ValueError("Expected [B,F,K,6] tensor")
    active = torch.as_tensor(active, dtype=torch.bool, device=values.device)
    sc, su = (torch.as_tensor(x, dtype=values.dtype, device=values.device) for x in (sc, su))
    w = torch.as_tensor(w, dtype=values.dtype, device=values.device)
    if active.shape != (len(values), values.shape[2]) or sc.shape != (len(values),) or su.shape != sc.shape or w.shape != (2,):
        raise ValueError("Invalid scale transform shape")
    if not bool(torch.isfinite(sc).all() & torch.isfinite(su).all()) or bool((sc <= 0).any() | (su <= 0).any()):
        raise ValueError("Preconditioner scales must be finite and positive")
    if not bool(torch.isfinite(w).all()) or not torch.allclose(w.square().sum(), torch.ones((), dtype=w.dtype, device=w.device), atol=1e-6, rtol=1e-6):
        raise ValueError("Return direction must be a unit vector")
    if inverse:
        sc, su = sc.reciprocal(), su.reciprocal()
    q = (values[..., :2] * w).sum(-1) * active[:, None]
    mean = q.sum(-1, keepdim=True) / active.sum(-1).clamp_min(1)[:, None, None]
    delta = ((su[:, None, None]-1)*q + (sc-su)[:, None, None]*mean) * active[:, None]
    return torch.cat((values[..., :2] + delta[..., None]*w, values[..., 2:]), dim=-1)

class CrossStockFactorMixer(nn.Module):
    def __init__(self, width, factors=4):
        super().__init__()
        # A shared key bias cancels inside softmax, so omit that inert parameter.
        self.query, self.key = nn.Linear(width, width), nn.Linear(width, width, bias=False)
        self.value, self.output = nn.Linear(width, width), nn.Linear(width, width)
        self.inducing = nn.Parameter(torch.randn(factors, width) / math.sqrt(width))
        self.width = width

    def forward(self, x, eligible):
        query = self.query(x)
        logits = torch.einsum("rw,bkw->brk", self.inducing, self.key(x)) / math.sqrt(self.width)
        weights = torch.softmax(logits.masked_fill(~eligible[:, None], -1e4), dim=-1)
        weights = weights * eligible[:, None]
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        factors = torch.einsum("brk,bkw->brw", weights, self.value(x))
        reads = torch.softmax(torch.einsum("bkw,brw->bkr", query, factors) / math.sqrt(self.width), dim=-1)
        message = torch.einsum("bkr,brw->bkw", reads, factors)
        return x + self.output(message)

class JointDenoiser(nn.Module):
    def __init__(self, history_length=30, horizon=5, width=16):
        super().__init__()
        if min(history_length, horizon, width) < 1:
            raise ValueError("Network dimensions must be positive")
        self.history_length, self.horizon, self.width = history_length, horizon, width
        self.channel_mixer = nn.Linear(2 * len(FEATURES), width)
        self.history_conv = nn.Conv1d(width, width, 3, padding=1, groups=width)
        self.history_projection = nn.Linear(2 * width, width)
        self.noisy_projection = nn.Linear(width + horizon * len(FEATURES) + 6, width)
        self.factor_mixer = CrossStockFactorMixer(width)
        self.head = nn.Sequential(nn.SiLU(), nn.Linear(width, 4 * width), nn.SiLU(),
                                  nn.Linear(4 * width, horizon * len(FEATURES)))
        with torch.no_grad():
            self.head[-1].weight.mul_(0.05)
            self.head[-1].bias.zero_()

    def encode_history(self, context):
        # context [B,W,K,12]
        b, length, stocks, _ = context.shape
        mixed = torch.nn.functional.silu(self.channel_mixer(context)).permute(0, 2, 3, 1)

        windows = torch.nn.functional.pad(mixed, (1, 1)).unfold(-1, 3, 1)
        weights = self.history_conv.weight[:, 0]
        local = (windows * weights[None, None, :, None, :]).sum(-1)
        local = torch.nn.functional.silu(local + self.history_conv.bias[None, None, :, None])
        observed = context[..., len(FEATURES):].mean(-1).permute(0, 2, 1)
        pooled = (local * observed[:, :, None]).sum(-1) / observed.sum(-1)[:, :, None].clamp_min(1.)
        last = local[..., -1]
        return self.history_projection(torch.cat((pooled, last), dim=-1)).reshape(b, stocks, self.width)

    def forward(self, noisy, context, eligible, t, schedule, history_encoded=False):
        history = context if history_encoded else self.encode_history(context)
        frac = (t.float() + 0.5) / len(schedule["abar"])
        ab = schedule["abar"][t]
        time_features = torch.stack((frac, torch.log(ab / (1 - ab)) / 10,
                                     torch.sin(math.pi * frac), torch.cos(math.pi * frac),
                                     torch.sin(2 * math.pi * frac), torch.cos(2 * math.pi * frac)), -1)
        future = noisy.permute(0, 2, 1, 3).flatten(-2)
        tokens = torch.nn.functional.silu(self.noisy_projection(torch.cat(
            (history, future, time_features[:, None].expand(-1, history.shape[1], -1)), -1)))
        tokens = self.factor_mixer(tokens, eligible)
        output = self.head(tokens).reshape(len(noisy), noisy.shape[2], self.horizon, len(FEATURES))
        return output.permute(0, 2, 1, 3) * eligible[:, None, :, None]


def cosine_schedule(timesteps):
    if not isinstance(timesteps, int) or timesteps < 2:
        raise ValueError("timesteps must be an integer >= 2")
    u = np.arange(timesteps + 1, dtype=np.float64) / timesteps
    cumulative = np.cos((u + 0.008) / 1.008 * np.pi / 2) ** 2
    cumulative /= cumulative[0]
    beta = np.minimum(1 - cumulative[1:] / cumulative[:-1], 0.999)
    alpha = 1 - beta
    abar = np.cumprod(alpha)
    previous = np.r_[1.0, abar[:-1]]
    posterior = beta * (1 - previous) / (1 - abar)
    return {key: torch.as_tensor(value, dtype=torch.float32) for key, value in
            {"beta": beta, "alpha": alpha, "abar": abar, "posterior_variance": posterior}.items()}

def balanced_denoising_loss(output, target, observed, return_scales, active, sc, su, beta=1.0):
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    if output.shape != target.shape or observed.shape != output.shape or observed.dtype != torch.bool:
        raise ValueError("Matching tensors and boolean observations required")
    active = torch.as_tensor(active, dtype=torch.bool, device=output.device)
    if bool((observed & ~active[:, None, :, None]).any()):
        raise ValueError("Observed loss support cannot contain inactive stocks")
    sc, su = (torch.as_tensor(x, dtype=torch.float64, device=output.device) for x in (sc, su))
    scales = torch.as_tensor(return_scales, dtype=torch.float64, device=output.device)
    if scales.shape != (2,) or bool((scales <= 0).any()) or not bool(torch.isfinite(scales).all()):
        raise ValueError("Two positive finite return scales required")
    w = scales / scales.square().sum().sqrt()
    error = scale_coordinates((output-target).double(), active, sc, su, w) * observed
#
    wo = observed[..., :2].to(error.dtype) * w
    r = wo.square().sum(-1)
    dot = (error[..., :2] * wo).sum(-1)
    a, b = (su.square()-1)[:, None, None], (sc.square()-su.square())[:, None]
    local_den = 1+a*r
    n = active.sum(-1).clamp_min(1).to(error.dtype)[:, None]
    pdinvp = (r/local_den).sum(-1) / n
    woodbury_den = 1+b*pdinvp
    if bool((local_den <= 0).any() | (woodbury_den <= 0).any()) or not bool(torch.isfinite(woodbury_den).all()):
        raise FloatingPointError("Observed covariance solve lost positive definiteness")
    edinvp = (dot/local_den).sum(-1) / n.sqrt()
    base = error.square().sum((-1, -2)) - (a*dot.square()/local_den).sum(-1)
    cell_day = base - b*edinvp.square()/woodbury_den
    paired = observed[..., 0] & observed[..., 1]
    n_pair = paired.sum(-1)
    valid_day = n_pair > 0
    projection = ((error[..., :2]*w).sum(-1)*paired).sum(-1) / n_pair.clamp_min(1).to(error.dtype).sqrt()

    common_den = su.square()[:, None] + b*n_pair/n
    common_day = projection.square() / common_den * valid_day
    count = observed.flatten(1).sum(1)
    valid = count > 0
    if not bool(valid.any()):
        raise ValueError("Sampled training batch contains no observed target")
    cell_sum, common_sum = cell_day.sum(-1), common_day.sum(-1)

    if bool((cell_day < -1e-9*base.abs().clamp_min(1)).any()):
        raise FloatingPointError("Negative observed-subspace squared norm")
    days = valid_day.sum(-1)
    denominator = count[valid].to(error.dtype) + beta*days[valid]
    cell_contribution = (cell_sum[valid]/denominator).mean()
    common_contribution = (beta*common_sum[valid]/denominator).mean()
    common_valid = days > 0
    return {"balanced_total": ((cell_sum[valid]+beta*common_sum[valid])/denominator).mean(),
            "cell_contribution": cell_contribution, "common_contribution": common_contribution,
            "cell": (cell_sum[valid]/count[valid]).mean(),
            "common": (common_sum[common_valid]/days[common_valid]).mean() if bool(common_valid.any()) else output.sum()*0,
            "valid_anchors": int(valid.sum()), "common_anchors": int(common_valid.sum()),
            "common_days": int(valid_day.sum()), "observed_cells": int(count.sum()), "beta": beta}


