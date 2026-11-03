from __future__ import annotations
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.model import (
    FEATURES, GeneratorConfig, OHLCVNormalizer, HistoricalVolatilityScaling,
    JointDenoiser, _inputs, scale_coordinates, cosine_schedule,
)

class DiffAlpha:
    def __init__(self, normalizer, scaling, net, schedule):
        self.normalizer, self.scaling = normalizer, scaling
        self.net, self.schedule = net.cpu().eval(), schedule
        self.net.requires_grad_(False)

    @torch.no_grad()
    def sample(self, H, H_mask, eligible, draws=64, seed=0, sample_batch=2):
        H, H_mask, eligible = _inputs(H, H_mask, eligible)
        if not isinstance(draws, int) or draws < 1 or not isinstance(sample_batch, int) or sample_batch < 1:
            raise ValueError("Positive integer draws/sample_batch required")
        if next(self.net.parameters()).device.type != "cpu":
            raise ValueError("Readable reference uses the original CPU execution path")
        self.net.eval()
        scales = self.scaling.scales(H, H_mask, eligible)
        rng = torch.Generator(device="cpu").manual_seed(int(seed))
        result = np.empty((len(H), draws, self.normalizer.horizon, H.shape[2], 6), np.float32)
        for begin in range(0, len(H), sample_batch):
            end = min(len(H), begin + sample_batch)
            context = torch.as_tensor(self.normalizer.context(H[begin:end], H_mask[begin:end]))
            encoded = self.net.encode_history(context).repeat_interleave(draws, dim=0)
            active = torch.as_tensor(eligible[begin:end]).repeat_interleave(draws, dim=0)
            sc, su = [torch.as_tensor(scales[key][begin:end], dtype=torch.float32).repeat_interleave(draws)
                      for key in ("sc", "su")]
            z = torch.randn((len(encoded), self.normalizer.horizon, H.shape[2], 6), generator=rng)
            z *= active[:, None, :, None]
            for step in range(len(self.schedule["abar"]) - 1, -1, -1):
                t = torch.full((len(z),), step, dtype=torch.long)
                velocity = self.net(z, encoded, active, t, self.schedule, history_encoded=True)
                ab = self.schedule["abar"][step]
                z0 = ab.sqrt() * z - (1-ab).sqrt() * velocity
                previous = self.schedule["abar"][step-1] if step else torch.ones_like(ab)
                z = (self.schedule["beta"][step] * previous.sqrt() / (1-ab) * z0
                     + self.schedule["alpha"][step].sqrt() * (1-previous) / (1-ab) * z)
                if step:
                    z += self.schedule["posterior_variance"][step].sqrt() * torch.randn(z.shape, generator=rng)
                z *= active[:, None, :, None]
            x = scale_coordinates(z, active, sc, su, self.scaling.w)
            paths = x.numpy().reshape(end-begin, draws, self.normalizer.horizon, H.shape[2], 6)
            decoded = self.normalizer.decode(paths)
            result[begin:end] = np.where(eligible[begin:end, None, None, :, None], decoded, 0.)
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite paths; no clipping or rejection performed")
        return result


def save_checkpoint(model, path, config=GeneratorConfig(), codes=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    norm, scaling = model.normalizer, model.scaling
    if (config.history_length, config.horizon, config.width, config.timesteps) != (
        norm.history_length, norm.horizon, model.net.width, len(model.schedule["abar"])
    ):
        raise ValueError("Checkpoint config differs from fitted model")
    if codes is not None:
        codes = np.asarray(codes, dtype=str)
        if codes.shape != (norm.assets,) or len(set(codes.tolist())) != norm.assets:
            raise ValueError("codes must uniquely identify the fitted stock coordinates")
    payload = {
        "format_version": 1,
        "config": asdict(config),
        "features": list(FEATURES),
        "codes": None if codes is None else codes.tolist(),
        "normalizer": {key: getattr(norm, key) for key in (
            "history_length", "horizon", "assets", "mean", "scale")},
        "scaling": {
            "config": scaling.config,
            "w": torch.tensor(scaling.w, dtype=torch.float64),
            "covariance": torch.tensor(scaling.covariance, dtype=torch.float64),
            "reference_fallback_count": scaling.reference_fallback_count,
        },
        "state_dict": model.net.state_dict(),
    }
    torch.save(payload, path)


def load_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or payload.get("features") != list(FEATURES):
        raise ValueError("Unsupported checkpoint format or feature order")
    config = GeneratorConfig(**payload["config"])
    norm = OHLCVNormalizer.__new__(OHLCVNormalizer)
    for key in ("history_length", "horizon", "assets", "mean", "scale"):
        setattr(norm, key, payload["normalizer"][key])
    if (norm.history_length, norm.horizon) != (config.history_length, config.horizon):
        raise ValueError("Inconsistent checkpoint dimensions")
    if (np.shape(norm.mean) != (6,) or np.shape(norm.scale) != (6,)
            or not np.isfinite(norm.mean).all() or not np.isfinite(norm.scale).all()
            or np.any(np.asarray(norm.scale) <= 0)):
        raise ValueError("Invalid checkpoint normalization")
    scaling = HistoricalVolatilityScaling.__new__(HistoricalVolatilityScaling)
    scaling.normalizer = norm
    scaling.config = payload["scaling"]["config"]
    scaling.w = payload["scaling"]["w"].numpy()
    scaling.covariance = payload["scaling"]["covariance"].numpy()
    scaling.reference_fallback_count = payload["scaling"]["reference_fallback_count"]
    scaling._validate()
    with torch.random.fork_rng(devices=[]):
        net = JointDenoiser(config.history_length, config.horizon, config.width)
    net.load_state_dict(payload["state_dict"], strict=True)
    model = DiffAlpha(norm, scaling, net, cosine_schedule(config.timesteps))
    model.config = config
    model.codes = payload["codes"]
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="NPZ with H, H_mask, eligible, codes")
    parser.add_argument("--output", type=Path, default=Path("output/raw_paths.npz"))
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--sample-batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=9811)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    torch.set_num_threads(args.threads)
    model = load_checkpoint(args.checkpoint)
    with np.load(args.data, allow_pickle=False) as data:
        required = {"H", "H_mask", "eligible", "codes", "dates"}
        if not required.issubset(data.files):
            raise ValueError(f"Missing inference keys: {sorted(required - set(data.files))}")
        if model.codes is None or data["codes"].tolist() != model.codes:
            raise ValueError("Input stock codes/order differ from checkpoint")
        if data["dates"].shape != (len(data["H"]),):
            raise ValueError("dates must identify every history anchor")
        paths = model.sample(data["H"], data["H_mask"], data["eligible"],
                             draws=args.draws, seed=args.seed, sample_batch=args.sample_batch)
        result = {"paths": paths, "eligible": data["eligible"], "codes": data["codes"],
                  "dates": data["dates"], "seed": np.array(args.seed)}
        if "anchor_close" in data.files:
            result["anchor_close"] = data["anchor_close"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, **result)
    print(json.dumps({"output": str(args.output), "shape": list(paths.shape)}))


if __name__ == "__main__":
    main()


