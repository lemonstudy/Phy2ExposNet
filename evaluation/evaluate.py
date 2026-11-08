"""Evaluate one supplied checkpoint using the recovered per-map metrics."""
import argparse
import json
from pathlib import Path

import torch

from evaluation.common import _run_baseline, _to_gt_pl, add_data_arguments, load_checkpoint, setup_data, checkpoint_protocol_status
from evaluation.metrics import MetricAccumulator
from data.loaders import split_metadata
from models.baseline_pef import PEFNet
from models.baseline_resnet import ResNet
from models.phy2_exposnet import Phy2ExposNet
from models.phy2_exposnet_ablation import Stage1VIEModel, Stage2MSEModel, LegacyTwoStageModel
from models.radiounet import RadioWNet
from training.phy2_exposnet import build_phy2_input, pl_from_etot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=["phy2", "pef", "resnet", "radiounet", "stage1", "stage2", "legacy_full"], default="phy2")
    parser.add_argument("--variant", choices=["full", "light", "ul"], default="full")
    parser.add_argument("--input_mode", choices=["full", "no_tx"], default="full")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--out_dir", default="outputs/eval")
    args = parser.parse_args()
    if args.max_samples < 0:
        parser.error("--max_samples must be non-negative")
    if args.model != "phy2" and args.input_mode != "full":
        parser.error("--input_mode no_tx is supported only by --model phy2")
    factories = {
        "phy2": lambda: Phy2ExposNet(in_ch=4 if args.input_mode == "full" else 3, variant=args.variant),
        "pef": PEFNet, "resnet": ResNet,
        "radiounet": lambda: RadioWNet(inputs=2, phase="firstU"),
        "stage1": Stage1VIEModel, "stage2": Stage2MSEModel,
        "legacy_full": LegacyTwoStageModel,
    }
    dataset, operator, device, frequency = setup_data(args)
    model = load_checkpoint(factories[args.model](), args.checkpoint, device, dataset=dataset)
    count = min(args.max_samples, len(dataset)) if args.max_samples else len(dataset)
    accumulator = MetricAccumulator(ssim_range=args.ssim_range)
    with torch.inference_mode():
        for index in range(count):
            inputs, target = dataset[index]
            raw = inputs.unsqueeze(0).to(device)
            gt = _to_gt_pl(target, device)
            if args.model == "radiounet":
                prediction = _run_baseline(model, raw)
            else:
                features, _ = build_phy2_input(raw, operator, device, input_mode=args.input_mode)
                if args.model == "stage2":
                    prediction = model(features)
                elif args.model == "legacy_full":
                    initial = pl_from_etot(model.forward_stage1(features), building_mask=raw[:, :1])
                    prediction = model.forward_stage2(initial)
                else:
                    initial = pl_from_etot(model(features), building_mask=raw[:, :1])
                    prediction = initial if args.model == "stage1" else model.refine(initial)
            accumulator.update({args.model: prediction}, gt)
            if (index + 1) % 100 == 0 or index + 1 == count:
                print(f"[Progress] {args.model}/{args.variant}: {index + 1}/{count} test samples", flush=True)
    summary = accumulator.summary()
    metrics = summary[args.aggregation][args.model]
    result = {
        "arguments": vars(args), "frequency_hz": frequency, "num_samples": count,
        "aggregation": args.aggregation, "building_pixels_in_metrics": True,
        "split": split_metadata(dataset),
        "checkpoint_protocol": checkpoint_protocol_status(args.checkpoint, dataset),
        "metrics": metrics, "metrics_by_aggregation": summary,
    }
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
