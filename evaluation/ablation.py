"""Reproduce the paper's two-sample, four-column ablation figure."""
import argparse
import json
from pathlib import Path

import torch

from data.loaders import split_metadata
from evaluation.common import add_data_arguments, checkpoint_protocol_status, resolve_checkpoints, setup_data
from evaluation.common import load_checkpoint as _load_ckpt_strict
from models import phy2_exposnet_ablation
from training.phy2_exposnet_ablation import observe_from_complex, pl_from_etot


@torch.inference_mode()
def select_best_rmse_indices(model_full, dataset, device, vie_op, top_k=2,
                             two_stage_output='residual', max_samples=0):
    """Rank the test samples by the full model's RMSE before display clipping."""
    model_full.eval()
    count = min(max_samples, len(dataset)) if max_samples else len(dataset)
    if count < top_k:
        raise ValueError(f'At least {top_k} test samples are needed for the figure')
    scores = []
    for index in range(count):
        inputs, targets = dataset[index][:2]
        raw = inputs.unsqueeze(0).to(device)
        building = raw[:, :1]
        tx_mask = raw[:, 1:2]
        incident = vie_op.incident_field(tx_mask)
        features = torch.cat([building, tx_mask, incident.real.unsqueeze(1), incident.imag.unsqueeze(1)], dim=1)
        initial = pl_from_etot(model_full.forward_stage1(features), building_mask=building)
        prediction = model_full.forward_stage2(initial)
        if two_stage_output == 'residual':
            prediction = initial + prediction
        truth = targets.unsqueeze(0).to(device)
        if truth.shape[1] == 2:
            truth = observe_from_complex(truth)
        rmse = float((prediction - truth).double().square().mean().sqrt())
        if not torch.isfinite(prediction).all():
            raise ValueError(f'Non-finite full-model prediction at test index {index}')
        scores.append({'index': index, 'rmse': rmse})
    scores.sort(key=lambda row: (row['rmse'], row['index']))
    selected = scores[:top_k]
    return [row['index'] for row in selected], {
        'method': 'Phy2-ExposNet', 'metric': 'RMSE', 'order': 'ascending',
        'evaluated_samples': count, 'selected': selected,
    }


@torch.no_grad()
def visualize_ablation_2samples_2x4_from_ckpts(dataset, device, vie_op, out_dir, stage1_ckpt, stage2_ckpt, full_ckpt, sample_indices=None, vmax=1.0, boxes=None, two_stage_output='residual', max_samples=0, return_selection=False):
    import os
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.patches import Rectangle
    os.makedirs(out_dir, exist_ok=True)
    if two_stage_output not in ('residual', 'direct'):
        raise ValueError(f'Unknown two-stage output convention: {two_stage_output}')
    model_stage1 = phy2_exposnet_ablation.LegacyTwoStageModel(in_ch=4).to(device)
    model_stage2 = phy2_exposnet_ablation.Stage2MSEModel(in_ch=4).to(device)
    model_full = phy2_exposnet_ablation.LegacyTwoStageModel(in_ch=4).to(device)
    _load_ckpt_strict(model_stage1, stage1_ckpt, device)
    _load_ckpt_strict(model_stage2, stage2_ckpt, device)
    _load_ckpt_strict(model_full, full_ckpt, device)
    model_stage1.eval()
    model_stage2.eval()
    model_full.eval()
    if sample_indices is None:
        sample_indices, selection = select_best_rmse_indices(
            model_full, dataset, device, vie_op, top_k=2,
            two_stage_output=two_stage_output, max_samples=max_samples,
        )
    else:
        assert len(sample_indices) == 2, 'sample_indices must contain exactly 2 indices.'
        selection = {'method': 'explicit_indices', 'selected': [{'index': index} for index in sample_indices]}
    if boxes is not None:
        assert len(boxes) == 2, 'boxes must contain 2 boxes, one for each row.'
    all_panels = []
    all_building_masks = []
    for idx in sample_indices:
        sample = dataset[idx]
        if isinstance(sample, (list, tuple)) and len(sample) == 3:
            inputs, targets, _ = sample
        else:
            inputs, targets = sample
        inputs_raw = inputs.unsqueeze(0).to(device)
        building = inputs_raw[:, 0:1]
        tx_mask = inputs_raw[:, 1:2]
        gt = targets.unsqueeze(0).to(device)
        if gt.dim() == 4 and gt.size(1) == 2:
            gt_pl = observe_from_complex(gt)
        else:
            if gt.dim() == 3:
                gt = gt.unsqueeze(1)
            gt_pl = gt
        e_inc = vie_op.incident_field(tx_mask)
        inputs_4ch = torch.cat([building, tx_mask, e_inc.real.unsqueeze(1), e_inc.imag.unsqueeze(1)], dim=1)
        etot_stage1 = model_stage1.forward_stage1(inputs_4ch)
        pl0_stage1 = pl_from_etot(etot_stage1, building_mask=building).clamp(0, 1)
        pred_stage1 = model_stage1.forward_stage2(pl0_stage1)
        if two_stage_output == 'residual':
            pred_stage1 = pl0_stage1 + pred_stage1
        pred_stage1 = pred_stage1.clamp(0, 1)
        pred_stage2 = model_stage2(inputs_4ch).clamp(0, 1)
        etot_full = model_full.forward_stage1(inputs_4ch)
        pl0_full = pl_from_etot(etot_full, building_mask=building)
        pred_full = model_full.forward_stage2(pl0_full)
        if two_stage_output == 'residual':
            pred_full = pl0_full + pred_full
        pred_full = pred_full.clamp(0, 1)
        st1_np = pred_stage1[0, 0].detach().cpu().numpy()
        st2_np = pred_stage2[0, 0].detach().cpu().numpy()
        full_np = pred_full[0, 0].detach().cpu().numpy()
        gt_np = gt_pl[0, 0].detach().cpu().numpy()
        building_np = building[0, 0].detach().cpu().numpy()
        building_mask = building_np > 0.5
        all_panels.append([st1_np, st2_np, full_np, gt_np])
        all_building_masks.append(building_mask)
    xticks = [0, 50, 100, 150, 200, 250]
    yticks = [0, 50, 100, 150, 200, 250]
    col_tags = ['(a)', '(b)', '(c)', '(d)']
    new_cmap = mcolors.LinearSegmentedColormap.from_list('trunc_plasma', plt.get_cmap('plasma')(np.linspace(0.0, 1.0, 256)))
    fig, axs = plt.subplots(2, 4, figsize=(7.16, 3.45))
    axs = axs.ravel()
    for row in range(2):
        for col in range(4):
            ax = axs[row * 4 + col]
            img = all_panels[row][col]
            building_mask = all_building_masks[row]
            img_masked = np.ma.array(img, mask=building_mask)
            cmap_local = new_cmap.copy()
            cmap_local.set_bad(color='#000000')
            im = ax.imshow(img_masked, cmap=cmap_local, vmin=0.0, vmax=vmax, origin='upper')
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.tick_params(axis='both', labelsize=6.0, length=1.5, pad=1)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontname('Times New Roman')
            ax.set_title('')
            if boxes is not None:
                x, y, w, h = boxes[row]
                rect = Rectangle((x, y), w, h, fill=False, edgecolor='#ff3b30', linewidth=1.8)
                ax.add_patch(rect)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='3.2%', pad=0.03)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_ticks([0.0, 0.5, 1.0])
            cbar.ax.tick_params(labelsize=5.8, length=1.2, pad=1)
            for t in cbar.ax.get_yticklabels():
                t.set_fontname('Times New Roman')
            if row == 1:
                ax.text(0.5, -0.16, col_tags[col], transform=ax.transAxes, ha='center', va='top', fontsize=7.2, fontname='Times New Roman')
    plt.subplots_adjust(left=0.035, right=0.995, bottom=0.08, top=0.995, wspace=0.22, hspace=0.28)
    idx0, idx1 = sample_indices
    save_path = os.path.join(out_dir, f'ablation_2x4_idx{idx0:04d}_{idx1:04d}.png')
    plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)
    print(f'[VIS] saved 2x4 figure to: {save_path}')
    print(f'[VIS] sample indices: {sample_indices}')
    return (save_path, sample_indices, selection) if return_selection else (save_path, sample_indices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser, metrics=False)
    parser.add_argument("--checkpoint_root", default="checkpoints")
    parser.add_argument("--no_phys_ckpt", default=None, help="Original abla/stage1.pt: a two-stage checkpoint")
    parser.add_argument("--stage2_ckpt", default=None)
    parser.add_argument("--full_ckpt", default=None)
    parser.add_argument("--indices", nargs=2, type=int, default=None,
                        help="Override automatic selection, e.g. 248 6052 for the original Fig. 3 samples")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="0 ranks the complete test split; positive values limit smoke checks")
    parser.add_argument("--two_stage_output", choices=["residual", "direct"], default="residual",
                        help="residual matches the main trainer; direct reproduces the original local plotting convention")
    parser.add_argument("--boxes", nargs=8, type=float, default=None,
                        help="Two highlight rectangles: x y width height for each sample")
    parser.add_argument("--out_dir", default="outputs/ablation")
    args = parser.parse_args()
    if args.max_samples < 0:
        parser.error("--max_samples must be non-negative")
    if args.indices is None and args.max_samples == 1:
        parser.error("--max_samples must allow at least two samples for automatic selection")
    resolve_checkpoints(parser, args, {
        "no_phys_ckpt": "ablation/no_phys.pt", "stage2_ckpt": "ablation/stage2.pt",
        "full_ckpt": "phy2_full.pt",
    })
    dataset, operator, device, _ = setup_data(args)
    if args.indices is not None and any(index < 0 or index >= len(dataset) for index in args.indices):
        parser.error(f"--indices must be within [0, {len(dataset) - 1}]")
    figure, indices, selection = visualize_ablation_2samples_2x4_from_ckpts(
        dataset=dataset, device=device, vie_op=operator, out_dir=args.out_dir,
        stage1_ckpt=args.no_phys_ckpt, stage2_ckpt=args.stage2_ckpt,
        full_ckpt=args.full_ckpt, sample_indices=args.indices,
        two_stage_output=args.two_stage_output,
        boxes=None if args.boxes is None else [args.boxes[:4], args.boxes[4:]],
        max_samples=args.max_samples, return_selection=True,
    )
    record = {"arguments": vars(args), "figure": str(figure), "sample_indices": indices,
              "figure_selection": selection,
              "split": split_metadata(dataset), "checkpoint_protocols": {
                  name: checkpoint_protocol_status(path, dataset)
                  for name, path in [("no_phys", args.no_phys_ckpt), ("stage2", args.stage2_ckpt), ("full", args.full_ckpt)]}}
    (Path(args.out_dir) / "figure.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
