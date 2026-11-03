from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class AugmentationConfig:
    draws: int = 64
    amplitude: float = 2.0
    synthetic_weight: float = 0.1
    rv_floor: float = 1e-12
    min_history_days: int = 24
    min_training_samples: int = 60


def _boolean(values, shape, name):
    result = np.asarray(values)
    if result.dtype.kind != "b" or result.shape != shape:
        raise ValueError(f"{name} must be a boolean array with shape {shape}")
    return result


def _mask(values, shape, name):
    result = np.asarray(values)
    if result.dtype.kind != "b":
        raise ValueError(f"{name} must contain boolean observation flags")
    if result.shape == shape[:-1]:
        result = np.broadcast_to(result[..., None], shape)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape[:-1]} or {shape}")
    return result


def _physical(values, dimensions, name):
    result = np.asarray(values)
    if result.ndim != dimensions or min(result.shape) < 1 or result.shape[-1] != 6:
        raise ValueError(f"{name} must be a nonempty {dimensions}-D six-channel array")
    if result.dtype.kind != "f" or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite floating-point physical features")
    if np.any(result[..., 2:] < 0):
        raise ValueError(f"The last four channels of {name} must be nonnegative")
    return result


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def transform_anchor(paths, eligible, amplitude=2.0):
    x = _physical(paths, 4, "paths [M,L,K,6]")
    active = _boolean(eligible, (x.shape[2],), "eligible")
    if np.ndim(amplitude) != 0 or not np.isfinite(amplitude) or amplitude < 1:
        raise ValueError("amplitude must be finite and at least one")
    a = float(amplitude)
    out = x.astype(np.float64, copy=True)
    count = int(active.sum())
    info = {"active_count": count, "empty_support": count == 0,
            "amplitude": a, "draw_count": len(x)}
    names = ("base_stock_rv", "stress_stock_rv", "base_basket_rv", "stress_basket_rv",
             "base_residual_rv", "stress_residual_rv", "residual_mean_energy",
             "residual_draw_variation", "expected_stress_stock_rv",
             "mean_preservation_error", "basket_preservation_error", "energy_identity_error")
    if count == 0:
        return out, {**info, **{key: 0.0 for key in names}}
    with np.errstate(over="raise", invalid="raise"):
        selected = (out[..., 0] + out[..., 1])[:, :, active]
        basket = selected.mean(-1, keepdims=True)
        residual = selected - basket
        residual_mean = residual.mean(0, keepdims=True)
        centered = residual - residual_mean
        delta = (a - 1.0) * centered
        out[:, :, active, 0] += delta / 2.0
        out[:, :, active, 1] += delta / 2.0
        changed = (out[..., 0] + out[..., 1])[:, :, active]
        new_basket = changed.mean(-1, keepdims=True)
        new_residual = changed - new_basket

        def energy(values):
            return float(np.square(values).sum(1).mean())

        info.update(base_stock_rv=energy(selected), stress_stock_rv=energy(changed),
                    base_basket_rv=energy(basket), stress_basket_rv=energy(new_basket),
                    base_residual_rv=energy(residual), stress_residual_rv=energy(new_residual),
                    residual_mean_energy=energy(residual_mean),
                    residual_draw_variation=energy(centered),
                    mean_preservation_error=float(np.max(np.abs(changed.mean(0)-selected.mean(0)))),
                    basket_preservation_error=float(np.max(np.abs(new_basket-basket))))
        expected = (info["base_basket_rv"] + info["residual_mean_energy"]
                    + a*a*info["residual_draw_variation"])
        info.update(expected_stress_stock_rv=expected,
                    energy_identity_error=abs(info["stress_stock_rv"]-expected))
    if not np.isfinite(out).all() or not all(np.isfinite(value) for value in info.values()):
        raise FloatingPointError("Nonfinite stress scenario or diagnostic")
    return out, info


def augment_pool(paths, eligible, amplitude=2.0):
    x = _physical(paths, 5, "paths [A,M,L,K,6]")
    active = _boolean(eligible, (x.shape[0], x.shape[3]), "eligible")
    out = np.empty(x.shape, dtype=np.float64)
    rows = []
    for index in range(len(x)):
        out[index], info = transform_anchor(x[index], active[index], amplitude)
        rows.append(info)
    return out, {key: np.asarray([row[key] for row in rows]) for key in rows[0]}


def build_targets(Y, Y_mask, H_mask, eligible, stress_paths, *,
                  synthetic_weight=0.1, min_history_days=24,
                  min_training_samples=60, rv_floor=1e-12):
    if np.ndim(synthetic_weight) != 0 or not np.isfinite(synthetic_weight) or not 0 <= synthetic_weight <= 1:
        raise ValueError("synthetic_weight must be finite and between zero and one")
    if np.ndim(rv_floor) != 0 or not np.isfinite(rv_floor) or rv_floor <= 0:
        raise ValueError("rv_floor must be finite and positive")
    _positive_integer(min_history_days, "min_history_days")
    _positive_integer(min_training_samples, "min_training_samples")
    pool = _physical(stress_paths, 5, "stress_paths [A,M,L,K,6]")
    n, _, horizon, stocks, _ = pool.shape
    y = np.asarray(Y, dtype=np.float64)
    if y.shape != (n, horizon, stocks, 6):
        raise ValueError("Y must have shape [A,L,K,6] matching stress_paths")
    ym = _mask(Y_mask, y.shape, "Y_mask")
    if not np.isfinite(y[ym]).all() or np.any(y[..., 2:][ym[..., 2:]] < 0):
        raise ValueError("Observed Y must contain valid physical features")
    active = _boolean(eligible, (n, stocks), "eligible")
    hm = np.asarray(H_mask)
    if hm.ndim not in (3, 4) or hm.shape[0] != n or hm.shape[2] != stocks or hm.shape[1] < 1:
        raise ValueError("H_mask must have shape [A,W,K] or [A,W,K,6]")
    hm = _mask(hm, (n, hm.shape[1], stocks, 6), "H_mask")
    if min_history_days > hm.shape[1]:
        raise ValueError("min_history_days cannot exceed the historical window length")
    observed_history = hm[..., 0] & hm[..., 1]
    valid = (active & (ym[..., 0] & ym[..., 1]).all(1)
             & (observed_history.sum(1) >= min_history_days) & observed_history[:, -1])
    supported = valid.sum(0) >= min_training_samples
    valid &= supported[None]
    with np.errstate(over="raise", invalid="raise"):
        real = np.where(ym[..., :2], y[..., :2], 0.).sum(-1)
        generated = pool[..., 0] + pool[..., 1]
        ret = real.sum(1)
        real_rv = np.maximum(np.square(real).sum(1), rv_floor)
        synthetic_rv = np.maximum(np.square(generated).sum(2), rv_floor).mean(1)
        mixed_rv = (1-synthetic_weight)*real_rv + synthetic_weight*synthetic_rv
    targets = {"return": ret, "real_rv": real_rv,
               "synthetic_rv": synthetic_rv, "rv": mixed_rv}
    for key, values in targets.items():
        if not np.isfinite(values[valid]).all():
            raise FloatingPointError(f"Nonfinite supervised target: {key}")
    return {**{key: np.where(valid, values, np.nan) for key, values in targets.items()},
            "valid": valid, "supported": supported}


def main(argv=None):
    defaults = AugmentationConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="Student training NPZ with Y,Y_mask,H_mask,eligible,codes,dates")
    parser.add_argument("--paths", type=Path, required=True,
                        help="Raw pool from utils.infer, conditioned on exactly the same histories")
    parser.add_argument("--output", type=Path, default=Path("output/augmentation.npz"))
    parser.add_argument("--amplitude", type=float, default=defaults.amplitude)
    parser.add_argument("--synthetic-weight", type=float, default=defaults.synthetic_weight)
    parser.add_argument("--min-history-days", type=int, default=defaults.min_history_days)
    parser.add_argument("--min-training-samples", type=int, default=defaults.min_training_samples)
    args = parser.parse_args(argv)
    if args.output.resolve() in (args.data.resolve(), args.paths.resolve()):
        parser.error("Output must differ from the real data and raw path pool")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    with np.load(args.data, allow_pickle=False) as data, np.load(args.paths, allow_pickle=False) as bank:
        for key in ("dates", "codes", "eligible"):
            if key not in data or key not in bank or not np.array_equal(data[key], bank[key]):
                raise ValueError(f"Data and pool must have identical {key}; regenerate paths for this split")
        for key in ("Y", "Y_mask", "H_mask"):
            if key not in data:
                raise ValueError(f"Student data is missing {key}")
        if "paths" not in bank:
            raise ValueError("Path pool is missing paths")
        paths = _physical(bank["paths"], 5, "paths [A,M,L,K,6]")
        for key, size in (("dates", paths.shape[0]), ("codes", paths.shape[3])):
            identifiers = data[key]
            if (identifiers.shape != (size,) or identifiers.dtype.kind not in "US"
                    or np.unique(identifiers).size != size
                    or np.any(np.char.str_len(np.char.strip(identifiers)) == 0)):
                raise ValueError(f"{key} must contain {size} unique, nonempty string identifiers")
        stressed, diagnostics = augment_pool(paths, data["eligible"], args.amplitude)
        targets = build_targets(data["Y"], data["Y_mask"], data["H_mask"], data["eligible"], stressed,
                                synthetic_weight=args.synthetic_weight,
                                min_history_days=args.min_history_days,
                                min_training_samples=args.min_training_samples)
        if not targets["valid"].any():
            raise ValueError("No valid student labels meet history and training-sample support thresholds")
        metadata = {key: data[key] for key in ("dates", "codes", "eligible")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, stress_paths=stressed, **targets, **metadata,
                            amplitude=np.asarray(args.amplitude),
                            synthetic_weight=np.asarray(args.synthetic_weight),
                            **{"diagnostic_"+key: value for key, value in diagnostics.items()})
    print(f"Saved {args.output}: {int(targets['valid'].sum())} valid labels, "
          f"{int(targets['supported'].sum())} supported stocks")


if __name__ == "__main__":
    main()
