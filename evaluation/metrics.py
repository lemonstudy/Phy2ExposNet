"""Explicit per-map and pooled-pixel metrics for normalized exposure maps."""
import math
from collections import defaultdict

import torch
from skimage.metrics import structural_similarity


class MetricAccumulator:
    """Accumulate whole samples, independent of evaluation batch size."""

    def __init__(self, ssim_range="fixed"):
        if ssim_range not in ("fixed", "legacy"):
            raise ValueError(f"Unknown SSIM range convention: {ssim_range}")
        self.ssim_range = ssim_range
        self.count = 0
        self.totals = defaultdict(lambda: defaultdict(float))
        self.per_map = defaultdict(lambda: defaultdict(float))

    def update(self, predictions, target, valid=None):
        if target.ndim != 4 or target.shape[1] != 1:
            raise ValueError("Expected target shape (B, 1, H, W)")
        if valid is None:
            valid = torch.ones_like(target)
        current = {}
        for batch_index in range(target.shape[0]):
            gt = target[batch_index:batch_index + 1].detach().double()
            mask = valid[batch_index:batch_index + 1].detach().double()
            n = float(mask.sum())
            if n <= 0:
                raise ValueError("Metric mask contains no valid pixels")
            truth_sum = float((gt * mask).sum())
            truth_sq = float((gt.square() * mask).sum())
            truth_mean = truth_sum / n
            sst = float(((gt - truth_mean).square() * mask).sum())
            for name, prediction in predictions.items():
                if prediction.shape != target.shape:
                    raise ValueError(f"Prediction shape mismatch for {name}")
                pred = prediction[batch_index:batch_index + 1].detach().double()
                if not torch.isfinite(pred).all() or not torch.isfinite(gt).all():
                    raise ValueError(f"Non-finite prediction or target for {name}")
                error = pred - gt
                sse = float((error.square() * mask).sum())
                sae = float((error.abs() * mask).sum())
                ape = float((error.abs() / gt.abs().clamp_min(0.05) * mask).sum())
                x = (pred * mask)[0, 0].cpu().numpy()
                y = (gt * mask)[0, 0].cpu().numpy()
                data_range = 1.0
                if self.ssim_range == "legacy":
                    data_range = max(float(max(x.max(), y.max()) - min(x.min(), y.min())), 1e-8)
                score = float(structural_similarity(y, x, data_range=data_range))
                values = {
                    "NMSE": sse / max(truth_sq, 1e-12),
                    "RMSE": math.sqrt(sse / n), "MAE": sae / n,
                    "R2": 1 - sse / max(sst, 1e-12), "SSIM": score, "MAPE": ape / n,
                }
                for metric, value in values.items():
                    self.per_map[name][metric] += value
                for key, value in {"sse": sse, "sae": sae, "ape": ape, "n": n,
                                   "truth_sum": truth_sum, "truth_sq": truth_sq, "ssim": score}.items():
                    self.totals[name][key] += value
                current[name] = values
            self.count += 1
        return current

    def summary(self):
        if self.count == 0:
            raise ValueError("No samples have been evaluated")
        per_map = {}
        pooled = {}
        for name, sums in self.totals.items():
            per_map[name] = {key: value / self.count for key, value in self.per_map[name].items()}
            n = sums["n"]
            total_variance = max(sums["truth_sq"] - sums["truth_sum"] ** 2 / n, 1e-12)
            pooled[name] = {
                "NMSE": sums["sse"] / max(sums["truth_sq"], 1e-12),
                "RMSE": math.sqrt(sums["sse"] / n), "MAE": sums["sae"] / n,
                "R2": 1 - sums["sse"] / total_variance,
                "SSIM": sums["ssim"] / self.count, "MAPE": sums["ape"] / n,
            }
            for metrics in (per_map[name], pooled[name]):
                metrics["NMSE_dB"] = 10 * math.log10(max(metrics["NMSE"], 1e-12))
        return {"per_map": per_map, "global": pooled, "num_samples": self.count,
                "ssim_range": self.ssim_range, "ssim_window": 7,
                "nmse_db_convention": "10*log10(aggregated linear NMSE)"}
