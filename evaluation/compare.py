"""Evaluate eight comparison methods and export the paper's 3 x 3 figures.

Prediction and figure implementations come from the original local paper scripts.
"""
import argparse
import csv
import json
import math
import os
from collections import OrderedDict
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.neighbors import KNeighborsRegressor
from torch.utils.data import Subset
from xgboost import XGBRegressor

from data.loaders import get_dataset, split_metadata
from evaluation.common import (
    _run_baseline, _to_gt_pl, add_data_arguments, checkpoint_protocol_status,
    load_checkpoint, resolve_checkpoints, setup_data,
)
from evaluation.metrics import MetricAccumulator
from models.baseline_pef import PEFNet
from models.baseline_resnet import ResNet
from models.phy2_exposnet import Phy2ExposNet
from models.radiounet import RadioWNet
from training.phy2_exposnet import pl_from_etot


def _extract_tx_xy(tx_mask: torch.Tensor):
    B, _, H, W = tx_mask.shape
    flat = tx_mask.view(B, -1).argmax(dim=1)
    ty = flat // W
    tx = flat % W
    return (ty, tx)


def _make_coord_features(inputs_raw: torch.Tensor, dx=1.0, dy=1.0):
    B, _, H, W = inputs_raw.shape
    device = inputs_raw.device
    tx_mask = inputs_raw[:, 1:2]
    ty, tx = _extract_tx_xy(tx_mask)
    yy = torch.arange(H, device=device).view(1, H, 1).expand(B, H, W)
    xx = torch.arange(W, device=device).view(1, 1, W).expand(B, H, W)
    dy_m = (yy - ty.view(B, 1, 1)).float() * dy
    dx_m = (xx - tx.view(B, 1, 1)).float() * dx
    d2d = torch.sqrt(dx_m * dx_m + dy_m * dy_m).clamp_min(1.0)
    return {'yy': yy.float(), 'xx': xx.float(), 'ty': ty.float(), 'tx': tx.float(), 'd2d': d2d, 'H': H, 'W': W}


def _invert_and_normalize01(x: torch.Tensor, eps=1e-08):
    x = -x
    xmin = x.amin(dim=(2, 3), keepdim=True)
    xmax = x.amax(dim=(2, 3), keepdim=True)
    return ((x - xmin) / (xmax - xmin).clamp_min(eps)).clamp(0, 1)


def _apply_building_mask_for_vis(pl_map: torch.Tensor, building: torch.Tensor, building_value=0.0):
    return pl_map * (1.0 - building) + building_value * building


def _run_refine_model(model, inputs_4ch, building):
    try:
        out_stage1 = model(inputs_4ch, pl0=None)
    except TypeError:
        out_stage1 = model(inputs_4ch)
    etot_ri = out_stage1[0] if isinstance(out_stage1, (list, tuple)) else out_stage1
    pl0 = pl_from_etot(etot_ri, building_mask=building)
    try:
        out_stage2 = model(inputs_4ch, pl0=pl0)
        pl = out_stage2[1] if isinstance(out_stage2, (list, tuple)) else out_stage2
    except TypeError:
        if hasattr(model, 'refine'):
            pl = model.refine(pl0)
        else:
            raise RuntimeError(f'{model.__class__.__name__} neither supports (x4, pl0=...) nor refine(pl0).')
    if pl.dim() == 3:
        pl = pl.unsqueeze(1)
    if not (pl.dim() == 4 and pl.size(1) == 1):
        raise RuntimeError(f'Refine model output expected (B,1,H,W), got {tuple(pl.shape)}')
    return pl


def _pl_itur_m2412_imt2020_uma_nlos(inputs_raw, fc_ghz=2.5, h_bs=25.0, h_ut=1.5, dx=1.0, dy=1.0):
    geo = _make_coord_features(inputs_raw, dx=dx, dy=dy)
    d2d = geo['d2d'].clamp_min(10.0)
    d3d = torch.sqrt(d2d ** 2 + (h_bs - h_ut) ** 2).clamp_min(10.0)
    c = 300000000.0
    fc_hz = fc_ghz * 1000000000.0
    d_bp = 4.0 * h_bs * h_ut * fc_hz / c
    d_bp = max(float(d_bp), 10.0)
    pl_los = torch.where(d2d <= d_bp, 28.0 + 22.0 * torch.log10(d3d) + 20.0 * math.log10(fc_ghz), 28.0 + 40.0 * torch.log10(d3d) + 20.0 * math.log10(fc_ghz) - 9.0 * math.log10(d_bp ** 2 + (h_bs - h_ut) ** 2))
    pl_nlos = 13.54 + 39.08 * torch.log10(d3d) + 20.0 * math.log10(fc_ghz) - 0.6 * (h_ut - 1.5)
    building = inputs_raw[:, 0:1]
    pl = torch.maximum(pl_los, pl_nlos).unsqueeze(1)
    pl = _invert_and_normalize01(pl)
    pl = _apply_building_mask_for_vis(pl, building, building_value=0.0)
    return pl


def _pl_3gpp_38901_uma_nlos(inputs_raw, fc_ghz=2.5, h_bs=1.5, h_ut=1.5, dx=1.0, dy=1.0):
    geo = _make_coord_features(inputs_raw, dx=dx, dy=dy)
    d2d = geo['d2d']
    d3d = torch.sqrt(d2d ** 2 + (h_bs - h_ut) ** 2).clamp_min(10.0)
    c = 300000000.0
    fc_hz = fc_ghz * 1000000000.0
    d_bp = 4.0 * h_bs * h_ut * fc_hz / c
    d_bp = max(float(d_bp), 10.0)
    pl_los = torch.where(d2d <= d_bp, 28.0 + 22.0 * torch.log10(d3d) + 20.0 * math.log10(fc_ghz), 28.0 + 40.0 * torch.log10(d3d) + 20.0 * math.log10(fc_ghz) - 9.0 * math.log10(d_bp ** 2 + (h_bs - h_ut) ** 2))
    pl_nlos_prime = 13.54 + 39.08 * torch.log10(d3d) + 20.0 * math.log10(fc_ghz) - 0.6 * (h_ut - 1.5)
    building = inputs_raw[:, 0:1]
    pl = torch.maximum(pl_los, pl_nlos_prime).unsqueeze(1)
    pl = _invert_and_normalize01(pl)
    pl = _apply_building_mask_for_vis(pl, building, building_value=0.0)
    return pl


def _sample_ml_features_from_one(inputs_raw, gt_pl, dx=1.0, dy=1.0, stride=4):
    assert inputs_raw.shape[0] == 1
    geo = _make_coord_features(inputs_raw, dx=dx, dy=dy)
    building = inputs_raw[:, 0:1]
    H, W = (geo['H'], geo['W'])
    xx = geo['xx'][0].cpu().numpy()
    yy = geo['yy'][0].cpu().numpy()
    d2d = geo['d2d'][0].cpu().numpy()
    bld = building[0, 0].cpu().numpy()
    txx = float(geo['tx'][0].item())
    tyy = float(geo['ty'][0].item())
    feat = np.stack([xx / max(W - 1, 1), yy / max(H - 1, 1), np.log10(d2d + 1.0), bld, np.full_like(xx, txx / max(W - 1, 1), dtype=np.float32), np.full_like(yy, tyy / max(H - 1, 1), dtype=np.float32)], axis=-1)
    y = gt_pl[0, 0].detach().cpu().numpy()[..., None]
    feat = feat[::stride, ::stride, :].reshape(-1, feat.shape[-1])
    y = y[::stride, ::stride, :].reshape(-1)
    return (feat.astype(np.float32), y.astype(np.float32))


def build_knn_regressor(dataset, device, dx=1.0, dy=1.0, n_neighbors=5, stride=4, max_points=200000, seed=42):
    rng = np.random.default_rng(seed)
    X_all, y_all = ([], [])
    n_total = 0
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if isinstance(sample, (list, tuple)) and len(sample) == 3:
            inputs, targets, _ = sample
        else:
            inputs, targets = sample
        inputs_raw = inputs.unsqueeze(0).to(device)
        gt_pl = _to_gt_pl(targets, device=device)
        Xi, yi = _sample_ml_features_from_one(inputs_raw, gt_pl, dx=dx, dy=dy, stride=stride)
        X_all.append(Xi)
        y_all.append(yi)
        n_total += Xi.shape[0]
        if n_total >= max_points:
            break
    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    if X.shape[0] > max_points:
        keep = rng.choice(X.shape[0], size=max_points, replace=False)
        X = X[keep]
        y = y[keep]
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance', algorithm='auto', n_jobs=-1)
    knn.fit(X, y)
    print(f'[INFO] KNN fitted with {X.shape[0]} samples, n_neighbors={n_neighbors}')
    return knn


def build_xgb_regressor(dataset, device, dx=1.0, dy=1.0, stride=4, max_points=200000, n_estimators=300, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, seed=42):
    rng = np.random.default_rng(seed)
    X_all, y_all = ([], [])
    n_total = 0
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if isinstance(sample, (list, tuple)) and len(sample) == 3:
            inputs, targets, _ = sample
        else:
            inputs, targets = sample
        inputs_raw = inputs.unsqueeze(0).to(device)
        gt_pl = _to_gt_pl(targets, device=device)
        Xi, yi = _sample_ml_features_from_one(inputs_raw, gt_pl, dx=dx, dy=dy, stride=stride)
        X_all.append(Xi)
        y_all.append(yi)
        n_total += Xi.shape[0]
        if n_total >= max_points:
            break
    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    if X.shape[0] > max_points:
        keep = rng.choice(X.shape[0], size=max_points, replace=False)
        X = X[keep]
        y = y[keep]
    xgb = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, subsample=subsample, colsample_bytree=colsample_bytree, objective='reg:squarederror', tree_method='hist', random_state=seed, n_jobs=-1)
    xgb.fit(X, y)
    print(f'[INFO] XGBoost fitted with {X.shape[0]} samples, n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate}')
    return xgb


def _predict_knn_map(knn_model, inputs_raw, device, dx=1.0, dy=1.0):
    geo = _make_coord_features(inputs_raw, dx=dx, dy=dy)
    building = inputs_raw[:, 0:1]
    B, _, H, W = inputs_raw.shape
    assert B == 1, 'KNN prediction helper currently expects batch size 1.'
    xx = geo['xx'][0].cpu().numpy()
    yy = geo['yy'][0].cpu().numpy()
    d2d = geo['d2d'][0].cpu().numpy()
    bld = building[0, 0].cpu().numpy()
    txx = float(geo['tx'][0].item())
    tyy = float(geo['ty'][0].item())
    feat = np.stack([xx / max(W - 1, 1), yy / max(H - 1, 1), np.log10(d2d + 1.0), bld, np.full_like(xx, txx / max(W - 1, 1), dtype=np.float32), np.full_like(yy, tyy / max(H - 1, 1), dtype=np.float32)], axis=-1).reshape(-1, 6).astype(np.float32)
    pred = knn_model.predict(feat).reshape(H, W)
    pred = torch.from_numpy(pred).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return pred.clamp(0, 1)


def _predict_xgb_map(xgb_model, inputs_raw, device, dx=1.0, dy=1.0):
    geo = _make_coord_features(inputs_raw, dx=dx, dy=dy)
    building = inputs_raw[:, 0:1]
    B, _, H, W = inputs_raw.shape
    assert B == 1, 'XGBoost prediction helper currently expects batch size 1.'
    xx = geo['xx'][0].cpu().numpy()
    yy = geo['yy'][0].cpu().numpy()
    d2d = geo['d2d'][0].cpu().numpy()
    bld = building[0, 0].cpu().numpy()
    txx = float(geo['tx'][0].item())
    tyy = float(geo['ty'][0].item())
    feat = np.stack([xx / max(W - 1, 1), yy / max(H - 1, 1), np.log10(d2d + 1.0), bld, np.full_like(xx, txx / max(W - 1, 1), dtype=np.float32), np.full_like(yy, tyy / max(H - 1, 1), dtype=np.float32)], axis=-1).reshape(-1, 6).astype(np.float32)
    pred = xgb_model.predict(feat).reshape(H, W)
    pred = torch.from_numpy(pred).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return pred.clamp(0, 1)


def _predict_all_methods(model_base, model_pef, model_resnet, model_propose, knn_model, xgb_model, inputs_raw, device, vie_op, dx=1.0, dy=1.0, fc_ghz=2.5):
    building = inputs_raw[:, 0:1]
    tx_mask = inputs_raw[:, 1:2]
    pred_base = _run_baseline(model_base, inputs_raw)
    e_inc = vie_op.incident_field(tx_mask)
    inputs_4ch = torch.cat([building, tx_mask, e_inc.real.unsqueeze(1), e_inc.imag.unsqueeze(1)], dim=1)
    pred_pef = _run_refine_model(model_pef, inputs_4ch, building)
    pred_resnet = _run_refine_model(model_resnet, inputs_4ch, building)
    pred_proposed = _run_refine_model(model_propose, inputs_4ch, building)
    pred_imt2020 = _pl_itur_m2412_imt2020_uma_nlos(inputs_raw, fc_ghz=fc_ghz, dx=dx, dy=dy)
    pred_3gpp38 = _pl_3gpp_38901_uma_nlos(inputs_raw, fc_ghz=fc_ghz, dx=dx, dy=dy)
    pred_xgb = _predict_xgb_map(xgb_model, inputs_raw, device=device, dx=dx, dy=dy)
    pred_knn = _predict_knn_map(knn_model, inputs_raw, device=device, dx=dx, dy=dy)
    return OrderedDict([('Baseline', pred_base), ('PEFNet', pred_pef), ('ResNet', pred_resnet), ('ITU-R IMT-2020', pred_imt2020), ('3GPP38.901', pred_3gpp38), ('XGBoost', pred_xgb), ('KNN', pred_knn), ('Phy2-ExposNet', pred_proposed)])


@torch.no_grad()
def visualize_compare(model_pef, model_base, model_resnet, model_propose, knn_model, xgb_model, dataset, device, vie_op, out_dir, n=10, start_idx=0, vmax=1.0, dx=1.0, dy=1.0, fc_ghz=2.5, aggregation='per_map', ssim_range='fixed', return_summary=False):
    os.makedirs(out_dir, exist_ok=True)
    model_base.eval()
    model_pef.eval()
    model_resnet.eval()
    model_propose.eval()
    eps = 1e-12
    N = len(dataset)
    use_valid_mask = False
    method_names = ['Baseline', 'PEFNet', 'ResNet', 'ITU-R IMT-2020', '3GPP38.901', 'XGBoost', 'KNN', 'Phy2-ExposNet']
    global_count = 0
    accumulator = MetricAccumulator(ssim_range=ssim_range)
    sample_scores = []
    for idx in range(start_idx, N):
        sample = dataset[idx]
        if isinstance(sample, (list, tuple)) and len(sample) == 3:
            inputs, targets, _ = sample
        else:
            inputs, targets = sample
        inputs_raw = inputs.unsqueeze(0).to(device)
        gt_pl = _to_gt_pl(targets, device=device)
        building = inputs_raw[:, 0:1]
        if use_valid_mask:
            valid = (1.0 - building).clamp(0, 1)
        else:
            valid = torch.ones_like(gt_pl)
        preds = _predict_all_methods(model_base=model_base, model_pef=model_pef, model_resnet=model_resnet, model_propose=model_propose, knn_model=knn_model, xgb_model=xgb_model, inputs_raw=inputs_raw, device=device, vie_op=vie_op, dx=dx, dy=dy, fc_ghz=fc_ghz)
        metrics_this = accumulator.update(preds, gt_pl, valid)
        global_count += 1
        sample_scores.append({'idx': idx, 'sort_rmse': metrics_this['Phy2-ExposNet']['RMSE']})
        if global_count % 100 == 0 or idx + 1 == N:
            print(f'[Progress] comparison: {global_count}/{N - start_idx} test samples', flush=True)
    print('[VIS] Method order in 3x3 figure:')
    print('[VIS] GT | ITU-R IMT-2020 | 3GPP38.901 | XGBoost | KNN | Baseline | PEFNet | ResNet | Phy2-ExposNet')
    if global_count == 0:
        raise ValueError('No samples were evaluated; check the dataset and start index.')
    results = {}
    summary = accumulator.summary()
    for name in method_names:
        metrics = summary[aggregation][name]
        nmse, rmse, r2, ssim_mean, mae, mape = [metrics[key] for key in ['NMSE', 'RMSE', 'R2', 'SSIM', 'MAE', 'MAPE']]
        print(f'[Eval][{aggregation}] {name}: NMSE={nmse:.6e}, RMSE={rmse:.6e}, R2={r2:.6f}, SSIM={ssim_mean:.6f}, MAE={mae:.6f}, MAPE={mape:.6f}')
        results[name] = {
            'NMSE': nmse, 'NMSE_dB': 10 * math.log10(max(nmse, eps)),
            'RMSE': rmse, 'R2': r2, 'SSIM': ssim_mean, 'MAE': mae, 'MAPE': mape,
        }
    sample_scores = sorted(sample_scores, key=lambda x: (x['sort_rmse'], x['idx']))
    topk = sample_scores[:min(n, len(sample_scores))]
    summary['figure_selection'] = {
        'method': 'Phy2-ExposNet', 'metric': 'RMSE', 'order': 'ascending',
        'evaluated_samples': global_count,
        'selected': [{'index': item['idx'], 'rmse': item['sort_rmse']} for item in topk],
    }
    fig_w = 7.16
    fig_h = 7.1
    xticks = [0, 50, 100, 150, 200, 250]
    yticks = [0, 50, 100, 150, 200, 250]
    for rank, item in enumerate(topk, 1):
        idx = item['idx']
        sample = dataset[idx]
        if isinstance(sample, (list, tuple)) and len(sample) == 3:
            inputs, targets, _ = sample
        else:
            inputs, targets = sample
        inputs_raw = inputs.unsqueeze(0).to(device)
        gt_pl = _to_gt_pl(targets, device=device)
        building_np = inputs_raw[0, 0].detach().cpu().numpy()
        building_mask = building_np > 0.5
        preds = _predict_all_methods(model_base=model_base, model_pef=model_pef, model_resnet=model_resnet, model_propose=model_propose, knn_model=knn_model, xgb_model=xgb_model, inputs_raw=inputs_raw, device=device, vie_op=vie_op, dx=dx, dy=dy, fc_ghz=fc_ghz)
        panel_order = ['ITU-R IMT-2020', '3GPP38.901', 'XGBoost', 'KNN', 'Baseline', 'PEFNet', 'ResNet', 'Phy2-ExposNet']
        panels = [gt_pl] + [preds[k] for k in panel_order]
        panels = [p.clamp(0, 1) for p in panels]
        panel_tags = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']
        fig, axs = plt.subplots(3, 3, figsize=(7.16, 7.1))
        axs = axs.ravel()
        xticks = [0, 50, 100, 150, 200, 250]
        yticks = [0, 50, 100, 150, 200, 250]
        new_cmap = mcolors.LinearSegmentedColormap.from_list('trunc_plasma', plt.get_cmap('plasma')(np.linspace(0.0, 1.0, 256)))
        for ax, panel, tag in zip(axs, panels, panel_tags):
            img = panel[0, 0].detach().cpu().numpy()
            img_masked = np.ma.array(img, mask=building_mask)
            cmap_local = new_cmap.copy()
            cmap_local.set_bad(color='#000000')
            im = ax.imshow(img_masked, cmap=cmap_local, vmin=0.0, vmax=1.0, origin='upper')
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.tick_params(axis='both', labelsize=8, length=2, pad=1)
            ax.set_title('')
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontname('Times New Roman')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='3%', pad=0.06)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            cbar.ax.tick_params(labelsize=8, length=2, pad=1)
            for t in cbar.ax.get_yticklabels():
                t.set_fontname('Times New Roman')
            ax.text(0.5, -0.16, tag, transform=ax.transAxes, ha='center', va='top', fontsize=10, fontname='Times New Roman')
        plt.subplots_adjust(left=0.06, right=0.995, bottom=0.06, top=0.995, wspace=0.32, hspace=0.28)
        save_path = os.path.join(out_dir, f'paper_3x3_top{rank:02d}_idx{idx:04d}.png')
        plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.01)
        plt.close(fig)
    print(f'[VIS] Saved top-{len(topk)} IEEE-style 3x3 figures to: {out_dir}')
    return (results, summary) if return_summary else results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--checkpoint_root", default="checkpoints")
    parser.add_argument("--radio_ckpt", default=None)
    parser.add_argument("--pef_ckpt", default=None)
    parser.add_argument("--resnet_ckpt", default=None)
    parser.add_argument("--phy2_ckpt", default=None)
    parser.add_argument("--variant", choices=["full", "light", "ul"], default="full")
    parser.add_argument("--out_dir", default="outputs/compare")
    parser.add_argument("--vis_n", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=0, help="0 evaluates the complete test split")
    parser.add_argument("--knn_neighbors", type=int, default=5)
    parser.add_argument("--knn_stride", type=int, default=8)
    parser.add_argument("--knn_max_samples", type=int, default=20000)
    parser.add_argument("--xgb_n_estimators", type=int, default=200)
    parser.add_argument("--xgb_max_depth", type=int, default=8)
    parser.add_argument("--xgb_learning_rate", type=float, default=0.05)
    parser.add_argument("--xgb_subsample", type=float, default=0.8)
    parser.add_argument("--xgb_colsample_bytree", type=float, default=0.8)
    args = parser.parse_args()
    if args.max_samples < 0 or args.vis_n < 0:
        parser.error("--max_samples and --vis_n must be non-negative")
    resolve_checkpoints(parser, args, {
        "radio_ckpt": "radio.pt", "pef_ckpt": "pef.pt",
        "resnet_ckpt": "resnet.pt", "phy2_ckpt": f"phy2_{args.variant}.pt",
    })

    dataset, operator, device, frequency = setup_data(args)
    split = split_metadata(dataset)
    checkpoint_protocols = {
        name: checkpoint_protocol_status(path, dataset)
        for name, path in [("RadioUNet", args.radio_ckpt), ("PEFNet", args.pef_ckpt),
                           ("ResNet", args.resnet_ckpt), ("Phy2-ExposNet", args.phy2_ckpt)]
    }
    checkpoint_dataset = dataset
    training = get_dataset(args.dataset, "train", args.data_dir, usc_split=args.usc_split)
    if args.max_samples:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    radio = load_checkpoint(RadioWNet(inputs=2, phase="firstU"), args.radio_ckpt, device, dataset=checkpoint_dataset)
    pef = load_checkpoint(PEFNet(in_ch=4), args.pef_ckpt, device, dataset=checkpoint_dataset)
    resnet = load_checkpoint(ResNet(in_ch=4), args.resnet_ckpt, device, dataset=checkpoint_dataset)
    phy2 = load_checkpoint(Phy2ExposNet(in_ch=4, variant=args.variant), args.phy2_ckpt, device, dataset=checkpoint_dataset)
    knn = build_knn_regressor(
        training, device, dx=args.dx, dy=args.dy, n_neighbors=args.knn_neighbors,
        stride=args.knn_stride, max_points=args.knn_max_samples, seed=42,
    )
    xgb = build_xgb_regressor(
        training, device, dx=args.dx, dy=args.dy, stride=args.knn_stride,
        max_points=args.knn_max_samples, n_estimators=args.xgb_n_estimators,
        max_depth=args.xgb_max_depth, learning_rate=args.xgb_learning_rate,
        subsample=args.xgb_subsample, colsample_bytree=args.xgb_colsample_bytree,
        seed=42,
    )
    results, summary = visualize_compare(
        model_base=radio, model_pef=pef, model_resnet=resnet, model_propose=phy2,
        knn_model=knn, xgb_model=xgb, dataset=dataset, device=device,
        vie_op=operator, out_dir=args.out_dir, n=args.vis_n,
        dx=args.dx, dy=args.dy, fc_ghz=frequency / 1e9,
        aggregation=args.aggregation, ssim_range=args.ssim_range, return_summary=True,
    )
    output = Path(args.out_dir)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Method", *next(iter(results.values()))])
        writer.writeheader()
        writer.writerows({"Method": name, **metrics} for name, metrics in results.items())
    record = {
        "arguments": vars(args), "frequency_hz": frequency,
        "num_samples": len(dataset), "aggregation": args.aggregation,
        "split": split,
        "checkpoint_protocols": checkpoint_protocols,
        "building_pixels_in_metrics": True,
        "figure_selection": summary["figure_selection"],
        "metrics": results, "metrics_by_aggregation": summary,
    }
    (output / "metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
