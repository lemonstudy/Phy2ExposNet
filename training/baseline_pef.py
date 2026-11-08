import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import time
import copy
import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from data import loaders
from models import baseline_pef

class LaplacianH(nn.Module):
    """
    H(f) = Δf using 2nd-order finite difference implemented by conv2d.
    f: (B,1,H,W)
    """
    def __init__(self):
        super().__init__()
        k = torch.tensor([[0.,  1., 0.],
                          [1., -4., 1.],
                          [0.,  1., 0.]], dtype=torch.float32)
        self.register_buffer("kernel", k.view(1, 1, 3, 3))

    def forward(self, f):
        f_pad = F.pad(f, (1, 1, 1, 1), mode="replicate")
        return F.conv2d(f_pad, self.kernel)

def hankel2_0(x: torch.Tensor) -> torch.Tensor:
    j0 = torch.special.bessel_j0(x)
    y0 = torch.special.bessel_y0(x)
    return torch.complex(j0, -y0)

def hankel2_1(x: torch.Tensor) -> torch.Tensor:
    j1 = torch.special.bessel_j1(x)
    y1 = torch.special.bessel_y1(x)
    return torch.complex(j1, -y1)

class VIEOperator(nn.Module):
    """
    v = W u, where W matches the 'cell-averaged' dense matrix implementation:
      off-diagonal: (j/4) k^2 (π a/2) J1(ka) H0^(2)(k r)
      diagonal:     (j/2) [ π k a H1^(2)(k a) - 2j ]
    Uses LINEAR convolution (zero-padding to (2H-1, 2W-1)), no periodic wrap-around.
    """

    def __init__(self, H: int, W: int, dx: float, dy: float,
                 freq_hz: float, eps0: float, mu0: float,
                 dtype=torch.complex64, device=None, k_eff: float | None = None):
        super().__init__()
        self.H = int(H)
        self.W = int(W)
        self.dx = float(dx)
        self.dy = float(dy)
        self.freq_hz = float(freq_hz)

        if k_eff is None:

            k = 2.0 * math.pi * self.freq_hz * math.sqrt(eps0 * mu0)
        else:

            k = float(k_eff)
        k_t = torch.tensor(k, dtype=torch.float32, device=device)

        A = self.dx * self.dy
        a = math.sqrt(A / math.pi)
        a_t = torch.tensor(a, dtype=torch.float32, device=device)

        Hp = 2 * self.H - 1
        Wp = 2 * self.W - 1
        self.Hp, self.Wp = Hp, Wp

        K = torch.zeros((Hp, Wp), dtype=dtype, device=device)

        ka = k_t * a_t
        J1_ka = torch.special.bessel_j1(ka)

        scale_off = (1j / 4.0) * (k_t ** 2) * (math.pi * a / 2.0) * J1_ka
        scale_off = scale_off.to(dtype)

        H1_ka = hankel2_1(ka)
        diag = (1j / 2.0) * (math.pi * ka * H1_ka - 2j)
        diag = diag.to(dtype)

        for dy_idx in range(-(self.H - 1), self.H):
            for dx_idx in range(-(self.W - 1), self.W):
                iy = dy_idx % Hp
                ix = dx_idx % Wp
                if dy_idx == 0 and dx_idx == 0:
                    K[iy, ix] = diag
                else:
                    r = math.hypot(dy_idx * self.dy, dx_idx * self.dx)
                    z = (k_t * torch.tensor(r, dtype=torch.float32, device=device)).clamp_min(1e-12)
                    H0 = hankel2_0(z).to(dtype)
                    K[iy, ix] = scale_off * H0

        K_fft = torch.fft.fft2(K)
        self.register_buffer("k", k_t)
        self.register_buffer("K_fft", K_fft)

    @torch.no_grad()
    def incident_field(self, tx_mask: torch.Tensor) -> torch.Tensor:
        """
        E_inc(p) = G(|p - p_tx|) with G(r) = -(j/4) H0^(2)(k r)
        tx_mask: (B,1,H,W) one-hot Tx
        return:  (B,H,W) complex
        """
        B, _, H, W = tx_mask.shape
        assert (H, W) == (self.H, self.W)

        flat = tx_mask.view(B, -1).argmax(dim=1)
        ty = (flat // W).to(tx_mask.device)
        tx = (flat % W).to(tx_mask.device)

        yy = torch.arange(H, device=tx_mask.device).view(1, H, 1).expand(B, H, W)
        xx = torch.arange(W, device=tx_mask.device).view(1, 1, W).expand(B, H, W)

        dy = (yy - ty.view(B, 1, 1)).float() * self.dy
        dx = (xx - tx.view(B, 1, 1)).float() * self.dx
        r = torch.sqrt(dy * dy + dx * dx).clamp_min(min(self.dx, self.dy) * 0.5)
        kr = self.k.to(tx_mask.device) * r
        G = -0.25j * hankel2_0(kr)
        return G.to(self.K_fft.dtype)

    def apply(self, u: torch.Tensor) -> torch.Tensor:
        """
        v = W u via FFT linear convolution.
        u: (B,H,W) complex OR (B,1,H,W)
        """
        if u.dim() == 4:
            u = u[:, 0, :, :]
        assert u.shape[-2:] == (self.H, self.W)

        if not torch.is_complex(u):
            u = torch.complex(u, torch.zeros_like(u))

        B = u.shape[0]
        U_pad = torch.zeros((B, self.Hp, self.Wp), dtype=self.K_fft.dtype, device=u.device)
        U_pad[:, :self.H, :self.W] = u.to(self.K_fft.dtype)

        V_pad = torch.fft.ifft2(torch.fft.fft2(U_pad) * self.K_fft.to(u.device))
        return V_pad[:, :self.H, :self.W]

def _to_complex_from_2ch(x_ri: torch.Tensor) -> torch.Tensor:
    """x_ri: (B,2,H,W) -> (B,H,W) complex."""
    return torch.complex(x_ri[:, 0], x_ri[:, 1])

def pl_from_etot(etot_ri: torch.Tensor, building_mask: torch.Tensor | None = None, eps: float = 1e-8):
    """Convert complex field to an initial PL-like map (normalized to [0,1]).

    The paper converts |E| -> received power -> PL (Sec.III-A.3, Eq.(16) + text).
    Since RadioMapSeer labels are already normalized, we use a stable surrogate:

      pl0 = -log10(|E| + eps)  (bigger attenuation -> bigger value)
      then per-sample min-max normalization to [0,1].
    """
    etot = _to_complex_from_2ch(etot_ri)
    amp = torch.abs(etot).clamp_min(eps)      # (B,H,W)
    pl0 = -torch.log10(amp + 1e-12)
    pl0 = pl0.unsqueeze(1)

    if building_mask is not None:
        bm = building_mask if building_mask.dim() == 4 else building_mask.unsqueeze(1)
        valid = (1.0 - bm).clamp(0, 1)

        bad = (valid.sum(dim=(1, 2, 3)) == 0)
        if bad.any():
            valid = valid.clone()
            valid[bad] = 1.0
        vmin = (pl0 * valid).amin(dim=(2, 3), keepdim=True)
        vmax = (pl0 * valid).amax(dim=(2, 3), keepdim=True)
    else:
        vmin = pl0.amin(dim=(2, 3), keepdim=True)
        vmax = pl0.amax(dim=(2, 3), keepdim=True)

    pl0n = (pl0 - vmin) / (vmax - vmin).clamp_min(eps)
    return pl0n.clamp(0, 1)

def pefnet_physics_loss(
    etot_ri: torch.Tensor,
    inputs_raw: torch.Tensor,
    vie_op: VIEOperator,
    device,
    eps_r_air: float = 1.0,
    eps_r_building: float = 4.0,
    ring_exclude_radius: int = 2,
    boundary_width: int = 1,
):
    """Physical knowledge loss L_PHY (paper Eq.(15)).

    L_PHY = (1/N) || (I + Wχ)E'_tot - E_inc ||_2^2

    Where χ is a *diagonal* matrix in the discretized system, implemented as a
    pointwise χ-map multiplied with E.
    """
    inputs_raw = inputs_raw.to(device)
    building = inputs_raw[:, 0:1]
    tx_mask = inputs_raw[:, 1:2]

    eps_r = torch.where(
        building > 0.5,
        torch.tensor(eps_r_building, device=device, dtype=building.dtype),
        torch.tensor(eps_r_air, device=device, dtype=building.dtype),
    )
    chi = (eps_r - 1.0)[:, 0]  # (B,H,W)

    etot = _to_complex_from_2ch(etot_ri.to(device))
    e_inc = vie_op.incident_field(tx_mask.to(device))
    we = vie_op.apply(torch.complex(chi, torch.zeros_like(chi)) * etot)

    r = (etot + we) - e_inc
    res2 = (r.real ** 2 + r.imag ** 2)  # (B,H,W)

    mask = torch.ones_like(building)
    if boundary_width and boundary_width > 0:
        b, _, h, w = mask.shape
        bmask = make_boundary_mask(h, w, width=boundary_width, device=mask.device, dtype=mask.dtype)
        mask = mask * (1.0 - bmask)
    if ring_exclude_radius and ring_exclude_radius > 0:
        tx_excl = F.max_pool2d(
            tx_mask,
            kernel_size=2 * ring_exclude_radius + 1,
            stride=1,
            padding=ring_exclude_radius,
        )
        mask = mask * (1.0 - tx_excl)

    num = (res2.unsqueeze(1) * mask).sum()
    den = mask.sum().clamp_min(1.0)
    return num / den

def make_boundary_mask(h, w, width=1, device=None, dtype=torch.float32):
    """Binary mask (1 on boundary ring of given width, 0 elsewhere). Shape (1,1,h,w)."""
    m = torch.zeros((1, 1, h, w), dtype=dtype, device=device)
    width = max(int(width), 1)
    m[:, :, :width, :] = 1
    m[:, :, -width:, :] = 1
    m[:, :, :, :width] = 1
    m[:, :, :, -width:] = 1
    return m

def observe_from_complex(psi_ri, eps=1e-8):
    """
    psi_ri: (B,2,H,W) where [:,0]=Re, [:,1]=Im
    return: (B,1,H,W) aligned with current target domain
    """
    re = psi_ri[:, 0:1]
    im = psi_ri[:, 1:2]
    mag = torch.sqrt((re * re + im * im).clamp_min(eps))  # (B,1,H,W)

    return mag

def _max_pool_mask(x, k):

    return F.max_pool2d(x, kernel_size=k, stride=1, padding=k//2)

def _min_pool_mask(x, k):

    return -F.max_pool2d(-x, kernel_size=k, stride=1, padding=k//2)

@torch.no_grad()
def _make_boundary_band(building_mask, width=5):
    """
    building_mask: (B,1,H,W) 1 inside building, 0 outside
    returns boundary_band: (B,1,H,W) 1 near boundary (both sides), 0 elsewhere
    """
    if width <= 0:
        return torch.zeros_like(building_mask)

    k = 2 * width + 1
    dil = _max_pool_mask(building_mask, k)  # expands building
    ero = _min_pool_mask(building_mask, k)  # shrinks building
    band = (dil - ero).clamp(0, 1)          # ring around boundary
    return band

def masked_mse(pred, target, mask, eps=1e-8):
    """
    pred/target: (B,1,H,W)
    mask: (B,1,H,W) in [0,1]
    """
    mse_map = (pred - target).pow(2)
    num = (mse_map * mask).sum()
    den = mask.sum().clamp_min(eps)
    return num / den

def print_metrics(metrics, epoch_samples, phase):
    outs = []
    for k in metrics.keys():
        outs.append(f"{k}: {metrics[k] / max(epoch_samples,1):.6f}")
    print(f"{phase}: " + ", ".join(outs))

@torch.no_grad()
def eval_nmse_rmse(model, dataloader, device, WNetPhase="firstU",vie_op=None,use_VIE=False):
    model.eval()
    se_sum = 0.0
    gt2_sum = 0.0
    mse_sum = 0.0
    n_batches = 0

    if vie_op is None:
        raise ValueError("eval_nmse_rmse for PEFNet requires vie_op.")

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        building = inputs[:, 0:1]
        tx_mask = inputs[:, 1:2]

        e_inc = vie_op.incident_field(tx_mask)
        inputs_4ch = torch.cat([building, tx_mask, e_inc.real.unsqueeze(1), e_inc.imag.unsqueeze(1)], dim=1)

        etot_ri = model(inputs_4ch)
        pl0 = pl_from_etot(etot_ri, building_mask=building)
        pl_ref = model(inputs_4ch)

        if targets.dim() == 4 and targets.size(1) == 2:
            pldat = observe_from_complex(targets)
        else:
            pldat = targets

        pred = pl_ref
        diff2 = (pred - pldat).pow(2)

        se_sum += diff2.sum().item()
        gt2_sum += pldat.pow(2).sum().item()
        mse_sum += diff2.mean().item()
        n_batches += 1

    nmse = se_sum / max(gt2_sum, 1e-12)
    rmse = (mse_sum / max(n_batches, 1)) ** 0.5
    return nmse, rmse

@torch.no_grad()
def visualize_pef_gt_vs_pred(
    model_pef,
    dataset,
    device,
    vie_op,
    out_dir,
    n=5,
    start_idx=0,
    vmax=1.0,
):
    """
    Plot ground truth and refined predictions from the test dataset.
    Include the initial map to inspect Stage 2 refinement.

    Each dataset sample returns:
      inputs: (2,H,W) [building, tx]
      targets: (1,H,W) or (2,H,W)
    """
    os.makedirs(out_dir, exist_ok=True)
    model_pef.eval()

    for k in range(n):
        idx = start_idx + k
        inputs, targets = dataset[idx]

        inputs_raw = inputs.unsqueeze(0).to(device)
        building = inputs_raw[:, 0:1]
        tx_mask = inputs_raw[:, 1:2]

        e_inc = vie_op.incident_field(tx_mask)  # (B,H,W) complex
        inputs_4ch = torch.cat(
            [building, tx_mask, e_inc.real.unsqueeze(1), e_inc.imag.unsqueeze(1)], dim=1
        )  # (B,4,H,W)

        etot_ri = model_pef(inputs_4ch)                        # (B,2,H,W)
        pl0 = pl_from_etot(etot_ri, building_mask=building)    # (B,1,H,W)
        pl_ref = model_pef.refine(pl0)                         # (B,1,H,W)

        gt = targets.unsqueeze(0).to(device)
        if gt.dim() == 4 and gt.size(1) == 2:
            gt_pl = observe_from_complex(gt)
        else:
            if gt.dim() == 3:
                gt = gt.unsqueeze(1)
            gt_pl = gt

        gt_img = gt_pl[0, 0].detach().cpu()
        pr_img = pl_ref[0, 0].detach().cpu()
        pl0_img = -pl0[0, 0].detach().cpu()
        eps = 1e-12
        pl0_min = pl0_img.min()
        pl0_max_ = pl0_img.max()
        pl0_img = (pl0_img - pl0_min) / (pl0_max_ - pl0_min + eps)
        pl0_img = pl0_img.clamp(0, 1)

        fig, axs = plt.subplots(1, 4, figsize=(18, 4))
        axs[0].imshow(gt_img, vmin=0, vmax=vmax)
        axs[0].set_title(f"GT (idx={idx})")
        axs[1].imshow(pl0_img, vmin=0, vmax=vmax)
        axs[1].set_title("PL0 (from Etot)")
        axs[2].imshow(pr_img, vmin=0, vmax=vmax)
        axs[2].set_title("PEF Pred (PL_ref)")
        axs[3].imshow((pr_img - gt_img).abs())
        axs[3].set_title("|Pred - GT|")

        for ax in axs:
            ax.axis("off")

        plt.tight_layout()
        fpath = os.path.join(out_dir, f"pred_{idx:05d}.png")
        plt.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"[VIS] saved {n} samples to: {out_dir}")

def train_model(
    model,
    dataloaders,
    device,
    optimizer,
    scheduler,
    H_op,
    vie_op=None,
    num_epochs=50,
    WNetPhase="firstU",

    use_H=False,
    lam_H=0.05,
    beta_H=0.0,
    ring_exclude_radius=3,

    use_VIE=False,
    lam_VIE=0.05,
    eps_r_building=4.0,

    lam_BC=0.0,
    lam_SRC=0.0,
    boundary_width=1,
    src_radius=3
):

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    if vie_op is None:
        raise ValueError("PEFNet training requires vie_op (for Einc inputs + L_PHY).")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}/{num_epochs-1}")
        print("-" * 30)
        since = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                print("learning rate:", optimizer.param_groups[0]['lr'])
                model.train()
            else:
                model.eval()

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, targets in dataloaders[phase]:
                inputs_raw = inputs.to(device)  # (B,2,H,W): [building, tx]
                targets = targets.to(device)
                building = inputs_raw[:, 0:1]
                tx_mask = inputs_raw[:, 1:2]

                with torch.no_grad():
                    e_inc = vie_op.incident_field(tx_mask)
                inputs_4ch = torch.cat(
                    [building, tx_mask, e_inc.real.unsqueeze(1), e_inc.imag.unsqueeze(1)],
                    dim=1,
                )

                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == 'train'):

                    etot_ri = model(inputs_4ch)

                    pl0 = pl_from_etot(etot_ri, building_mask=building)

                    pl_ref = model.refine(pl0)  

                    if targets.dim() == 4 and targets.size(1) == 2:
                        pldat = observe_from_complex(targets)
                    else:
                        pldat = targets

                    loss_dat = F.mse_loss(pl_ref, pldat)
                    loss_phy = pefnet_physics_loss(
                        etot_ri=etot_ri,
                        inputs_raw=inputs_raw,
                        vie_op=vie_op,
                        device=device,
                        eps_r_building=eps_r_building,
                        ring_exclude_radius=ring_exclude_radius,
                        boundary_width=boundary_width,
                    )
                    loss = lam_VIE * loss_phy + loss_dat

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                metrics['loss'] += loss.detach().cpu().item() * inputs.size(0)
                metrics['loss_phy'] += loss_phy.detach().cpu().item() * inputs.size(0)
                metrics['loss_dat'] += loss_dat.detach().cpu().item() * inputs.size(0)

                with torch.no_grad():
                    bmask = (building > 0.5).float()
                    inside = bmask
                    outside = 1.0 - bmask
                    band = _make_boundary_band(bmask, width=3)
                    metrics['mse_inside'] += masked_mse(pl_ref, pldat, inside).item() * inputs.size(0)
                    metrics['mse_outside'] += masked_mse(pl_ref, pldat, outside).item() * inputs.size(0)
                    metrics['mse_band'] += masked_mse(pl_ref, pldat, band).item() * inputs.size(0)

                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / max(epoch_samples, 1)
            if phase == 'val' and scheduler is not None:
                scheduler.step(epoch_loss)
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model (val loss improved)")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        dt = time.time() - since
        print(f"epoch time: {dt//60:.0f}m {dt%60:.0f}s")

    print(f"\nBest val loss: {best_loss:.6f}")
    model.load_state_dict(best_model_wts)
    return model, best_loss

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--step_size", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.1)

    parser.add_argument("--wnet_phase", type=str, default="firstU", choices=["firstU", "secondU"])

    parser.add_argument("--lam_H", type=float, default=0.05)
    parser.add_argument("--beta_H", type=float, default=0.0)
    parser.add_argument("--ring_exclude_radius", type=int, default=3)

    parser.add_argument("--lam_VIE", type=float, default=0.05, help="VIE residual loss weight (0 to disable)")
    parser.add_argument("--vie_freq_hz", type=float, default=None, help="frequency for Green's function (Hz)")
    parser.add_argument("--dx", type=float, default=1.0)
    parser.add_argument("--dy", type=float, default=1.0)
    parser.add_argument("--vie_eps_r_building", type=float, default=4.0, help="relative permittivity for buildings (air=1)")

    parser.add_argument("--lam_BC", type=float, default=0.0, help="boundary (Dirichlet) loss weight")
    parser.add_argument("--boundary_width", type=int, default=1, help="boundary ring width (pixels) for BC/PDE masks")

    parser.add_argument("--lam_SRC", type=float, default=0.0, help="Tx neighborhood (source) loss weight")
    parser.add_argument("--src_radius", type=int, default=3, help="Tx neighborhood radius (pixels) for source loss")

    parser.add_argument("--run_compare", action="store_true")
    parser.add_argument("--vis_n", type=int, default=5, help="number of samples to visualize")
    parser.add_argument("--vis_split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--out_dir", type=str, default="outputs/baseline_pef")
    parser.add_argument("--eps0", type=float, default=8.854187817e-12)
    parser.add_argument("--mu0", type=float, default=4.0e-7 * math.pi)
    parser.add_argument("--vie_lambda_eff", type=float, default=0.0,
                    help="If >0, use effective wavelength (meters) for VIE (k_eff=2π/λ_eff) instead of true freq.")

    parser.add_argument("--dataset", choices=["radiomapseer", "usc"], default="usc")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--usc_split", choices=["grouped", "legacy"], default="legacy")
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()
    if args.vie_freq_hz is None:
        args.vie_freq_hz = 2.5e9 if args.dataset == "usc" else 5.9e9

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True

    Radio_train = loaders.get_dataset(args.dataset, "train", args.data_dir, usc_split=args.usc_split)
    Radio_val = loaders.get_dataset(args.dataset, "val", args.data_dir, usc_split=args.usc_split)
    Radio_test = loaders.get_dataset(args.dataset, "test", args.data_dir, usc_split=args.usc_split)

    dataloaders = {
        'train': DataLoader(Radio_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type=="cuda")),
        'val': DataLoader(Radio_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type=="cuda")),
    }
    test_loader = DataLoader(Radio_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type=="cuda"))

    H_op = LaplacianH().to(device)

    sample_in, _ = Radio_train[0]
    _, H, W = sample_in.shape
    k_eff = None
    if args.vie_lambda_eff and args.vie_lambda_eff > 0:
        k_eff = 2.0 * math.pi / args.vie_lambda_eff  # rad/m

    vie_op = VIEOperator(H, W, dx=args.dx, dy=args.dy,freq_hz=args.vie_freq_hz,
                         eps0=args.eps0, mu0=args.mu0, dtype=torch.complex64, k_eff=k_eff).to(device)

    os.makedirs(args.out_dir, exist_ok=True)

    def build_model(tag):

        return baseline_pef.PEFNet(in_ch=4).to(device)

    def run_one(tag):
        print(f"\n==================== {tag} ====================")
        model = build_model(tag)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            threshold=0.0,
            min_lr=1e-7,
        )

        model, best_val_loss = train_model(
            model=model,
            dataloaders=dataloaders,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            H_op=H_op,
            vie_op=vie_op,
            num_epochs=args.epochs,
            WNetPhase=args.wnet_phase,
            use_H=False,
            lam_H=0.0,
            beta_H=0.0,
            use_VIE=True,

            lam_VIE=args.lam_VIE,
            eps_r_building=args.vie_eps_r_building,
            ring_exclude_radius=args.ring_exclude_radius,
            lam_BC=0.0,
            lam_SRC=0.0,
            boundary_width=args.boundary_width,
            src_radius=args.src_radius,
        )

        ckpt = os.path.join(args.out_dir, f"{tag}_best.pt")
        torch.save(model.state_dict(), ckpt)
        loaders.save_run_metadata(ckpt, args, {"train": Radio_train, "val": Radio_val, "test": Radio_test})

        nmse_val, rmse_val = eval_nmse_rmse(model, dataloaders['val'], device, WNetPhase=args.wnet_phase, vie_op=vie_op, use_VIE=True)
        nmse_test, rmse_test = eval_nmse_rmse(model, test_loader, device, WNetPhase=args.wnet_phase, vie_op=vie_op, use_VIE=True)

        print(f"[{tag}] BestValLoss={best_val_loss:.6f}")
        print(f"[{tag}] VAL  NMSE={nmse_val:.6e}, RMSE={rmse_val:.6e}")
        print(f"[{tag}] TEST NMSE={nmse_test:.6e}, RMSE={rmse_test:.6e}")
        print("saved:", ckpt)

        return model, {
            "tag": tag,
            "ckpt": ckpt,
            "best_val_loss": best_val_loss,
            "val_nmse": nmse_val,
            "val_rmse": rmse_val,
            "test_nmse": nmse_test,
            "test_rmse": rmse_test,
        }

    model_pef, _ = run_one("PEFNet")
    vis_dir = os.path.join(args.out_dir, "vis")
    visualize_pef_gt_vs_pred(
        model_pef=model_pef,
        dataset=Radio_test,
        device=device,
        vie_op=vie_op,
        out_dir=vis_dir,
        n=args.vis_n,
        start_idx=0,
        vmax=1.0,
    )

if __name__ == "__main__":
    main()
