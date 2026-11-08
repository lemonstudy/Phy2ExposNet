"""Evaluate the supplied checkpoints and compare Tables 2 and 3 with the paper.

The default run uses every test sample. A positive --max_samples is available
for smoke checks; incomplete test coverage never establishes full reproduction.
"""
import argparse
import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("NMSE_dB", "RMSE", "MAE", "SSIM")
# Published references: DOI 10.1109/OJAP.2026.3707005, Tables 2 and 3.
TABLE2 = {
    "radiomapseer": {
        "ITU-R IMT-2020": (0.88, 0.2001, 0.1469, 0.5277),
        "3GPP38.901": (-2.51, 0.1355, 0.0968, 0.5789),
        "XGBoost": (-1.16, 0.1559, 0.1217, 0.4601),
        "KNN": (-0.87, 0.1624, 0.1183, 0.5106),
        "RadioUNet": (-18.49, 0.0230, 0.0120, 0.9022),
        "PEFNet": (-17.89, 0.0239, 0.0129, 0.9006),
        "ResNet": (-18.78, 0.0221, 0.0124, 0.8992),
        "Phy2-ExposNet": (-19.87, 0.0195, 0.0102, 0.9194),
    },
    "usc": {
        "ITU-R IMT-2020": (-4.92, 0.3743, 0.3069, 0.6089),
        "3GPP38.901": (-3.57, 0.4376, 0.3715, 0.5072),
        "XGBoost": (-15.74, 0.1046, 0.0802, 0.7960),
        "KNN": (-14.57, 0.1182, 0.0834, 0.8592),
        "RadioUNet": (-25.81, 0.0324, 0.0217, 0.9261),
        "PEFNet": (-28.08, 0.0235, 0.0157, 0.9304),
        "ResNet": (-28.27, 0.0236, 0.0155, 0.9378),
        "Phy2-ExposNet": (-30.79, 0.0173, 0.0112, 0.9540),
    },
}
TABLE3 = {
    "RadioUNet": (13.27, 50.64, -18.49, 0.0230),
    "PEFNet": (67.41, 259.12, -17.89, 0.0239),
    "ResNet": (12.60, 50.04, -18.78, 0.0221),
    "Phy2-ExposNet": (11.12, 44.41, -19.87, 0.0195),
    "Phy2-ExposNet-Light": (8.67, 33.07, -19.45, 0.0203),
    "Phy2-ExposNet-UL": (3.67, 14.01, -19.35, 0.0206),
}


def build_jobs(args):
    """Reuse existing evaluation entry points without changing their formulas."""
    output = Path(args.out_dir)
    checkpoint_root = Path(args.checkpoint_root)
    common = ["--device", args.device, "--dx", str(args.dx), "--dy", str(args.dy),
              "--aggregation", args.aggregation, "--ssim_range", args.ssim_range,
              "--max_samples", str(args.max_samples)]
    jobs = [("model_info", [sys.executable, "-m", "evaluation.model_info",
                            "--out_dir", str(output / "model_info")])]
    for dataset, directory, frequency in [
        ("radiomapseer", args.rmap_data_dir, 5.9e9), ("usc", args.usc_data_dir, 2.5e9),
    ]:
        name = "table2_" + dataset
        jobs.append((name, [sys.executable, "-m", "evaluation.compare",
                           "--dataset", dataset, "--checkpoint_root", str(checkpoint_root),
                           "--data_dir", str(directory), "--usc_split", "legacy",
                           "--vie_freq_hz", str(frequency), "--vis_n", "0",
                           "--out_dir", str(output / name), *common]))
    for variant in ("light", "ul"):
        jobs.append((variant, [sys.executable, "-m", "evaluation.evaluate",
                               "--dataset", "radiomapseer", "--model", "phy2",
                               "--variant", variant, "--checkpoint",
                               str(checkpoint_root / "radiomapseer" / f"phy2_{variant}.pt"),
                               "--data_dir", str(args.rmap_data_dir), "--vie_freq_hz", "5900000000.0",
                               "--out_dir", str(output / variant), *common]))
    return jobs


def validate_inputs(args):
    directories = [(Path(args.rmap_data_dir), ("gain/DPM", "png/buildings_complete", "png/antennas")),
                   (Path(args.usc_data_dir), ("city_map", "tx_map", "power_map"))]
    for directory, children in directories:
        for child in children:
            if not (directory / child).is_dir():
                raise ValueError(f"Dataset directory not found: {directory / child}")
    for dataset, names in [("radiomapseer", ("radio", "pef", "resnet", "phy2_full", "phy2_light", "phy2_ul")),
                           ("usc", ("radio", "pef", "resnet", "phy2_full"))]:
        for name in names:
            path = Path(args.checkpoint_root) / dataset / f"{name}.pt"
            if not path.is_file():
                raise ValueError(f"Checkpoint not found: {path}. Fetch the checkpoint Git LFS objects first.")


def comparison_row(table, dataset, model, metric, paper, actual):
    decimals = 4 if metric in ("RMSE", "MAE", "SSIM") else 2
    tolerance = 0.5 * 10 ** -decimals
    delta = actual - paper
    return {"Table": table, "Dataset": dataset, "Model": model, "Metric": metric,
            "Paper": paper, "Actual": actual, "Delta": delta, "Tolerance": tolerance,
            "Match": math.isfinite(actual) and abs(delta) <= tolerance + 1e-12}


def summarize(output):
    """Require full test coverage before reporting agreement with the paper."""
    output = Path(output)
    records = {name: json.loads((output / name / "metrics.json").read_text(encoding="utf-8"))
               for name in ("table2_radiomapseer", "table2_usc", "light", "ul")}
    sizes = json.loads((output / "model_info/models.json").read_text(encoding="utf-8"))
    sizes = {row["Model"]: row for row in sizes["models"]}
    rows = []
    for dataset, references in TABLE2.items():
        metrics = records["table2_" + dataset]["metrics"]
        for model, values in references.items():
            measured = metrics["Baseline" if model == "RadioUNet" else model]
            for metric, paper in zip(METRICS, values):
                rows.append(comparison_row(2, dataset, model, metric, paper, measured[metric]))
    for model, expected in TABLE3.items():
        if model.endswith("-Light") or model.endswith("-UL"):
            measured = records["light" if model.endswith("-Light") else "ul"]["metrics"]
        else:
            key = "Baseline" if model == "RadioUNet" else model
            measured = records["table2_radiomapseer"]["metrics"][key]
        actual = (sizes[model]["Params (M)"], sizes[model]["Size (MB)"], measured["NMSE_dB"], measured["RMSE"])
        for metric, paper, value in zip(("Params (M)", "Size (MB)", "NMSE_dB", "RMSE"), expected, actual):
            rows.append(comparison_row(3, "radiomapseer", model, metric, paper, value))
    expected_counts = {"table2_radiomapseer": 7920, "table2_usc": 2000, "light": 7920, "ul": 7920}
    counts = {name: {"expected": count, "evaluated": records[name]["num_samples"]}
              for name, count in expected_counts.items()}
    full_test = all(row["expected"] == row["evaluated"] for row in counts.values())
    matches = sum(row["Match"] for row in rows)
    status = "smoke_only" if not full_test else ("within_published_rounding" if matches == len(rows) else "differences_found")
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"status": status, "scope": "Tables 2 and 3 using the supplied checkpoints",
              "paper_doi": "10.1109/OJAP.2026.3707005", "full_test_coverage": full_test,
              "full_table_match": full_test and matches == len(rows),
              "matched_values": matches, "total_values": len(rows), "test_coverage": counts,
              "comparison": rows,
              "note": "Match uses half the final decimal place printed in the paper. Best-case figure examples are not used as aggregate test scores."}
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    # Avoid loading a second Intel OpenMP runtime through conda's NumPy/MKL.
    # Keep an explicitly selected threading layer unchanged.
    if sys.platform == "win32":
        os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rmap_data_dir", default="data/RadioMapSeer")
    parser.add_argument("--usc_data_dir", default="data/USC")
    parser.add_argument("--checkpoint_root", default="checkpoints")
    parser.add_argument("--out_dir", default="outputs/paper_check")
    parser.add_argument("--device", default=None, help="cpu, cuda, or a specific CUDA device")
    parser.add_argument("--max_samples", type=int, default=0, help="0 evaluates every test sample")
    parser.add_argument("--aggregation", choices=["per_map", "global"], default="per_map")
    parser.add_argument("--ssim_range", choices=["fixed", "legacy"], default="legacy")
    parser.add_argument("--dx", type=float, default=1.0)
    parser.add_argument("--dy", type=float, default=1.0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry_run", action="store_true")
    mode.add_argument("--summarize_only", action="store_true", help="Rebuild the comparison from completed stage outputs")
    args = parser.parse_args()
    if args.max_samples < 0 or not math.isfinite(args.dx) or not math.isfinite(args.dy) or min(args.dx, args.dy) <= 0:
        parser.error("--max_samples must be non-negative; --dx and --dy must be finite and positive")
    for name in ("rmap_data_dir", "usc_data_dir", "checkpoint_root", "out_dir"):
        setattr(args, name, str(Path(getattr(args, name)).expanduser().resolve()))
    if args.device is None and not args.summarize_only:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dry_run:
        for name, command in build_jobs(args):
            print(f"{name}: {subprocess.list2cmdline(command) if sys.platform == 'win32' else shlex.join(command)}")
        return
    output = Path(args.out_dir)
    if not args.summarize_only:
        try:
            validate_inputs(args)
        except ValueError as error:
            parser.error(str(error))
        output.mkdir(parents=True, exist_ok=True)
        versions = {}
        for package in ("torch", "torchvision", "numpy", "scikit-image", "scikit-learn", "xgboost"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        threading_environment = {name: os.environ.get(name) for name in
                                 ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_THREADING_LAYER")}
        record = {"arguments": vars(args), "python": sys.version, "platform": platform.platform(),
                  "packages": versions, "threading_environment": threading_environment}
        (output / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        jobs = build_jobs(args)
        child_environment = os.environ.copy()
        child_environment["PYTHONIOENCODING"] = "utf-8"
        for index, (name, command) in enumerate(jobs, 1):
            log = output / f"{name}.log"
            print(f"[{index}/{len(jobs)}] {name}: {log}", flush=True)
            with log.open("w", encoding="utf-8") as handle:
                result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                        env=child_environment, check=False)
            if result.returncode:
                parser.exit(result.returncode, f"Evaluation failed at {name}. See {log}\n")
    result = summarize(output)
    print(f"Status: {result['status']}; matched {result['matched_values']}/{result['total_values']} values.")
    print(f"Comparison: {output / 'comparison.csv'}")
    print(f"Summary: {output / 'summary.json'}")


if __name__ == "__main__":
    main()
