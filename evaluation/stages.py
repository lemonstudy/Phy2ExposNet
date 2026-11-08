"""Plot the local nine-panel incident-field and two-stage diagnostic figure.

Recovered from test_compde_vie_pef.py:visualize_random_test_sample_with_stage1.
"""
import argparse
import json
import random
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data.loaders import split_metadata
from evaluation.common import add_data_arguments, checkpoint_protocol_status, load_checkpoint, resolve_checkpoints, setup_data
from models.baseline_pef import PEFNet
from models.baseline_resnet import ResNet
from models.phy2_exposnet import Phy2ExposNet
from training.phy2_exposnet import build_phy2_input, pl_from_etot


@torch.inference_mode()
def plot_stages(model, dataset, device, operator, out_dir, index=None, seed=42, tx_dilate_size=7):
    if tx_dilate_size < 1 or tx_dilate_size % 2 == 0:
        raise ValueError("Transmitter display size must be a positive odd integer")
    if index is None:
        index = random.Random(seed).randint(0, len(dataset) - 1)
    if not 0 <= index < len(dataset):
        raise ValueError(f"Sample index must be within [0, {len(dataset) - 1}]")
    model.eval()
    inputs, _ = dataset[index]
    raw = inputs.unsqueeze(0).to(device)
    features, _ = build_phy2_input(raw, operator, device)
    field = model(features)
    initial = pl_from_etot(field, building_mask=raw[:, :1])
    prediction = model.refine(initial)
    transmitter = F.max_pool2d(raw[:, 1:2], tx_dilate_size, stride=1, padding=tx_dilate_size // 2)
    amplitude = torch.sqrt(field[:, :1].square() + field[:, 1:2].square() + 1e-12)
    items = [
        ("Base (Tx mask)", transmitter), ("Building", raw[:, :1]),
        ("Einc Real", features[:, 2:3]), ("Einc Imag", features[:, 3:4]),
        ("Stage1 Etot Real", field[:, :1]), ("Stage1 Etot Imag", field[:, 1:2]),
        ("Stage1 |Etot|", amplitude), ("Stage1 pl0", initial), ("Final Prediction", prediction),
    ]
    if not all(torch.isfinite(value).all() for _, value in items):
        raise ValueError("Non-finite model output in stage visualization")
    cmap = mcolors.LinearSegmentedColormap.from_list("trunc_plasma", plt.get_cmap("plasma")(np.linspace(0, 1, 256)))
    figure, axes = plt.subplots(3, 3, figsize=(15, 13))
    for panel, (title, value) in enumerate(items):
        axis = axes.ravel()[panel]
        img = axis.imshow(value[0, 0].cpu().float().numpy(), cmap="gray" if panel < 2 else cmap)
        axis.set_title(title, fontsize=11)
        axis.axis("off")
        colorbar = figure.colorbar(img, ax=axis, fraction=0.046, pad=0.04)
        colorbar.ax.tick_params(labelsize=8)
    figure.tight_layout()
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"stages_{index:05d}.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path, index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser, metrics=False)
    parser.add_argument("--checkpoint_root", default="checkpoints")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model", choices=["phy2", "pef", "resnet"], default="phy2")
    parser.add_argument("--variant", choices=["full", "light", "ul"], default="full")
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tx_dilate_size", type=int, default=7)
    parser.add_argument("--out_dir", default="outputs/stages")
    args = parser.parse_args()
    if args.model != "phy2" and args.variant != "full":
        parser.error("--variant light/ul applies only to --model phy2")
    filename = f"phy2_{args.variant}.pt" if args.model == "phy2" else f"{args.model}.pt"
    resolve_checkpoints(parser, args, {"checkpoint": filename})
    if args.tx_dilate_size < 1 or args.tx_dilate_size % 2 == 0:
        parser.error("--tx_dilate_size must be a positive odd integer")
    dataset, operator, device, frequency = setup_data(args)
    if args.index is not None and not 0 <= args.index < len(dataset):
        parser.error(f"--index must be within [0, {len(dataset) - 1}]")
    factories = {"phy2": lambda: Phy2ExposNet(variant=args.variant), "pef": PEFNet, "resnet": ResNet}
    model = load_checkpoint(factories[args.model](), args.checkpoint, device, dataset=dataset)
    figure, index = plot_stages(model, dataset, device, operator, args.out_dir, args.index, args.seed, args.tx_dilate_size)
    record = {"arguments": vars(args), "figure": str(figure), "sample_index": index,
              "frequency_hz": frequency, "split": split_metadata(dataset),
              "checkpoint_protocol": checkpoint_protocol_status(args.checkpoint, dataset),
              "display_scaling": "Each panel retains the original local autoscaling convention."}
    (Path(args.out_dir) / "figure.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Saved {figure}")


if __name__ == "__main__":
    main()
