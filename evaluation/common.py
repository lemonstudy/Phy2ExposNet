"""Shared dataset, checkpoint, and physics setup for evaluation commands."""
import math
import json
import warnings
from pathlib import Path
from collections.abc import Mapping

import torch

from data.loaders import get_dataset, split_metadata
from training.phy2_exposnet import VIEOperator


def checkpoint_protocol_status(path, dataset):
    expected = split_metadata(dataset)
    metadata = Path(path).with_suffix(".json")
    result = {"expected": expected, "verified": False}
    if not metadata.is_file():
        result["reason"] = "No training split metadata accompanies this checkpoint."
        return result
    record = json.loads(metadata.read_text(encoding="utf-8"))
    saved = record.get("splits", {}).get("test", {})
    result["recorded_test_split"] = saved
    result["verified"] = saved.get("protocol") == expected["protocol"] and saved.get("membership_sha256") == expected["membership_sha256"]
    if not result["verified"]:
        result["reason"] = "Checkpoint test-split metadata does not match the requested dataset split."
    return result


def load_checkpoint(model, path, device, dataset=None):
    """Load weights strictly; never accept a partially matched paper model."""
    state = torch.load(path, map_location=device, weights_only=True)
    if isinstance(state, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state and isinstance(state[key], Mapping):
                state = state[key]
                break
    if not isinstance(state, Mapping):
        raise ValueError(f"Expected a state_dict checkpoint: {path}")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    if dataset is not None and getattr(dataset, "split_mode", None) == "grouped":
        status = checkpoint_protocol_status(path, dataset)
        if not status["verified"]:
            warnings.warn(
                f"USC grouped evaluation: {status['reason']} Legacy-trained weights must be retrained "
                "under the grouped split before scores establish held-out performance.",
                RuntimeWarning, stacklevel=2,
            )
    return model.to(device).eval()


def observe_from_complex(psi_ri, eps=1e-08):
    re = psi_ri[:, 0:1]
    im = psi_ri[:, 1:2]
    mag = torch.sqrt((re * re + im * im).clamp_min(eps))
    return mag

def _to_gt_pl(targets, device):
    gt = targets.unsqueeze(0).to(device)
    if gt.dim() == 4 and gt.size(1) == 2:
        gt_pl = observe_from_complex(gt)
    else:
        if gt.dim() == 3:
            gt = gt.unsqueeze(1)
        gt_pl = gt
    return gt_pl

def _run_baseline(model, inputs_raw):
    out = model(inputs_raw)
    out = out[0] if isinstance(out, (list, tuple)) else out
    if out.dim() == 3:
        out = out.unsqueeze(1)
    if not (out.dim() == 4 and out.size(1) == 1):
        raise RuntimeError(f'Baseline output expected (B,1,H,W), got {tuple(out.shape)}')
    return out


def resolve_checkpoints(parser, args, defaults):
    """Fill omitted weight paths from checkpoints/<dataset> and validate them."""
    directory = Path(args.checkpoint_root) / args.dataset
    for name, filename in defaults.items():
        supplied = getattr(args, name)
        path = Path(supplied) if supplied is not None else directory / filename
        if not path.is_file():
            parser.error(f"Checkpoint not found: {path}. Supply an explicit weight path or fetch the Git LFS objects.")
        setattr(args, name, str(path))


def add_data_arguments(parser, metrics=True):
    parser.add_argument("--dataset", choices=["radiomapseer", "usc"], default="radiomapseer")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--usc_split", choices=["grouped", "legacy"], default="legacy")
    if metrics:
        parser.add_argument("--aggregation", choices=["per_map", "global"], default="per_map")
        parser.add_argument("--ssim_range", choices=["fixed", "legacy"], default="fixed")
    parser.add_argument("--dx", type=float, default=1.0)
    parser.add_argument("--dy", type=float, default=1.0)
    parser.add_argument("--vie_freq_hz", type=float, default=None)
    parser.add_argument("--vie_lambda_eff", type=float, default=0.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def setup_data(args, phase="test"):
    device = torch.device(args.device)
    dataset = get_dataset(args.dataset, phase, args.data_dir, usc_split=args.usc_split)
    if not len(dataset):
        raise ValueError(f"Empty {phase} dataset")
    inputs, _ = dataset[0]
    height, width = inputs.shape[-2:]
    frequency = args.vie_freq_hz
    if frequency is None:
        frequency = 2.5e9 if args.dataset == "usc" else 5.9e9
    effective_k = 2 * math.pi / args.vie_lambda_eff if args.vie_lambda_eff > 0 else None
    operator = VIEOperator(
        height, width, args.dx, args.dy, frequency,
        eps0=8.854187817e-12, mu0=4e-7 * math.pi,
        dtype=torch.complex64, k_eff=effective_k,
    ).to(device)
    return dataset, operator, device, frequency
