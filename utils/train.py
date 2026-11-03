from __future__ import annotations
import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.model import (
    GeneratorConfig, OHLCVNormalizer, HistoricalVolatilityScaling,
    JointDenoiser, _inputs, scale_coordinates, cosine_schedule,
    balanced_denoising_loss,
)
from utils.infer import DiffAlpha, save_checkpoint

def fit_generator(H, H_mask, Y, Y_mask, eligible, reference_data,
                  config=GeneratorConfig(), seed=9811, progress=None):
    
    H, H_mask, eligible, Y, observed = _inputs(H, H_mask, eligible, Y, Y_mask)
    if H.shape[1] != config.history_length or Y.shape[1] != config.horizon:
        raise ValueError("Data dimensions differ from GeneratorConfig")
    if min(config.steps, config.batch) < 1:
        raise ValueError("Positive training steps and batch size required")
    if not observed.reshape(len(H), -1).any(axis=1).all():
        raise ValueError("Every teacher anchor needs at least one eligible observed target")
    if (not math.isfinite(config.learning_rate) or config.learning_rate <= 0
            or not math.isfinite(config.gradient_clip) or config.gradient_clip <= 0):
        raise ValueError("learning_rate and gradient_clip must be finite and positive")
    norm = OHLCVNormalizer(H, H_mask, Y, observed, eligible)
    scaling = HistoricalVolatilityScaling(norm, reference_data, config)
    scales = scaling.scales(H, H_mask, eligible)
    sc, su = [torch.as_tensor(scales[key], dtype=torch.float32) for key in ("sc", "su")]
    context = torch.as_tensor(norm.context(H, H_mask))
    targets = torch.as_tensor(norm.encode(Y))
    observed, active = torch.as_tensor(observed), torch.as_tensor(eligible)
    schedule = cosine_schedule(config.timesteps)
##
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed) + 10100)
        net = JointDenoiser(config.history_length, config.horizon, config.width)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    rng = torch.Generator(device="cpu").manual_seed(int(seed) + 20200)
    logs = []
    net.train()
    for step in range(1, config.steps + 1):
        ids = torch.randint(len(H), (config.batch,), generator=rng)
        t = torch.randint(config.timesteps, (config.batch,), generator=rng)
        noise = torch.randn((config.batch, config.horizon, H.shape[2], 6), generator=rng)
        imputed = torch.randn(noise.shape, generator=rng)
        clean_x = torch.where(observed[ids], targets[ids], imputed)
        clean_z = scale_coordinates(clean_x, active[ids], sc[ids], su[ids], scaling.w, inverse=True)
        ab = schedule["abar"][t, None, None, None]
        noisy_z = (ab.sqrt() * clean_z + (1-ab).sqrt() * noise) * active[ids, None, :, None]
        velocity_target = ab.sqrt() * noise - (1-ab).sqrt() * clean_z
        velocity_prediction = net(noisy_z, context[ids], active[ids], t, schedule)
        components = balanced_denoising_loss(
            velocity_prediction, velocity_target, observed[ids], norm.scale[:2],
            active[ids], sc[ids], su[ids], config.beta)
        loss = components["balanced_total"]
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Nonfinite generator loss at update {step}")
        lr = config.learning_rate * (.1 + .9 * .5 * (1 + math.cos(math.pi * (step-1) / config.steps)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  
        torch.nn.utils.clip_grad_norm_(net.parameters(), config.gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == config.steps:
            row = {"step": step, "loss": float(loss.detach()), "learning_rate": lr,
                   "cell": float(components["cell"].detach()),
                   "common": float(components["common"].detach())}
            logs.append(row)
            if progress is not None:
                progress(row)
    return DiffAlpha(norm, scaling, net, schedule), logs


def load_training_data(path):
    required = {"H", "H_mask", "Y", "Y_mask", "eligible", "reference_values",
                "reference_mask", "reference_dates", "dates", "target_end", "codes"}
    with np.load(path, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Missing training keys: {sorted(missing)}")
        data = {key: archive[key] for key in required}
    n = len(data["H"])
    start, end = np.datetime64("2019-01-01"), np.datetime64("2021-12-31")
    for key in ("dates", "target_end"):
        dates = data[key].astype("datetime64[D]")
        if dates.shape != (n,) or np.isnat(dates).any() or np.any((dates < start) | (dates > end)):
            raise ValueError(f"{key} must lie inside teacher period 2019--2021")
    anchors = data["dates"].astype("datetime64[D]")
    ends = data["target_end"].astype("datetime64[D]")
    if np.any(anchors[1:] <= anchors[:-1]) or np.any(ends <= anchors):
        raise ValueError("Unique ascending anchor dates and strictly later targets required")
    codes = data["codes"]
    if data["H"].ndim != 4 or codes.shape != (data["H"].shape[2],) or len(set(codes.tolist())) != len(codes):
        raise ValueError("codes must uniquely identify the H stock axis")
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Prepared teacher.npz (2019--2021)")
    parser.add_argument("--output", type=Path, default=Path("output/teacher"))
    parser.add_argument("--steps", type=int, default=4800)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=9811)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Output directory is not empty: {args.output}")
    torch.set_num_threads(args.threads)
    data = load_training_data(args.data)
    config = GeneratorConfig(
        history_length=data["H"].shape[1], horizon=data["Y"].shape[1],
        width=args.width, steps=args.steps, batch=args.batch, timesteps=args.timesteps,
        learning_rate=args.learning_rate, beta=args.beta, rho=args.rho,
    )
    model, logs = fit_generator(
        data["H"], data["H_mask"], data["Y"], data["Y_mask"], data["eligible"],
        {"values": data["reference_values"], "mask": data["reference_mask"], "dates": data["reference_dates"]},
        config=config, seed=args.seed, progress=lambda row: print(json.dumps(row), flush=True),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, args.output / "generator.pt", config, codes=data["codes"])
    (args.output / "training_log.json").write_text(json.dumps(logs, indent=2) + "\n")
    (args.output / "config.json").write_text(json.dumps({
        **asdict(config), "seed": args.seed, "threads": args.threads, "data": str(args.data),
        "parameters": sum(p.numel() for p in model.net.parameters()),
        "numpy_version": np.__version__, "torch_version": str(torch.__version__),
    }, indent=2) + "\n")
    print(f"Saved frozen teacher to {args.output / 'generator.pt'}")


if __name__ == "__main__":
    main()


