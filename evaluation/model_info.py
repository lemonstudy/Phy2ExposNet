"""Report neural parameter counts and model sizes using the paper's Table 3 convention."""
import argparse
import csv
import gc
import json
import math
from pathlib import Path

from models.phy2_exposnet import Phy2ExposNet
from models.baseline_pef import PEFNet
from models.baseline_resnet import ResNet
from models.radiounet import RadioWNet
from training.phy2_exposnet import VIEOperator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--out_dir", default="outputs/model_info")
    args = parser.parse_args()
    operator = VIEOperator(args.height, args.width, 1., 1., 5.9e9, 8.854187817e-12, 4e-7 * math.pi)
    physics_bytes = sum(buffer.numel() * buffer.element_size() for buffer in operator.buffers())
    factories = {
        "RadioUNet": lambda: RadioWNet(inputs=2, phase="firstU"),
        "PEFNet": PEFNet, "ResNet": ResNet,
        "Phy2-ExposNet": Phy2ExposNet,
        "Phy2-ExposNet-Light": lambda: Phy2ExposNet(variant="light"),
        "Phy2-ExposNet-UL": lambda: Phy2ExposNet(variant="ul"),
    }
    rows = []
    for name, factory in factories.items():
        model = factory()
        parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
        vie_bytes = physics_bytes if name in ("PEFNet", "ResNet", "Phy2-ExposNet") else 0
        rows.append({
            "Model": name, "Parameters": sum(p.numel() for p in model.parameters()),
            "Params (M)": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
            "Size (MB)": round((parameter_bytes + buffer_bytes + vie_bytes) / 1024**2, 2),
        })
        del model
        gc.collect()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "models.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"grid": [args.height, args.width], "unit": "MB as reported in Table 3 (bytes / 1024**2)", "models": rows,
              "note": "Uses the original Table 3 reporting convention. Serialized checkpoint overhead and peak runtime memory are excluded."}
    (output / "models.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
