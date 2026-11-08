# Phy2-ExposNet

Code for paper [**Phy2-ExposNet: A Physics-Informed Neural Network for Urban EMF Exposure Mapping**](https://ieeexplore.ieee.org/document/11578056).

## Structure

```text
models/                  Main model, stage ablations, and paper baselines
training/                Training and evaluation entry points
evaluation/              Checkpoint metrics and recovered paper figures
checkpoints/             Pretrained neural weights (Git LFS)
results_pic/             Original paper figures: Fig1.png, Fig2.png, Fig3.png
data/
  loaders.py             RadioMapSeer and USC loaders
  RadioMapSeer/          Download RadioMapSeer here
  USC/                   Download USC data here
outputs/                 Generated results
```

## Environment

Use Python 3.10 or later. Install a compatible PyTorch/torchvision pair for your CUDA environment, then install the remaining dependencies:

The requirements list dependencies. A CUDA GPU is recommended for training.

## Datasets

Download and extract the datasets before training. They are not included in this repository:

- **RadioMapSeer:** [official dataset page](https://radiomapseer.github.io/). Place its contents under `data/RadioMapSeer/`.
- **USC:** follow the **USC** dataset download link in [PMNet](https://github.com/abman23/PMNet). Place the data under `data/USC/`.

The loaders expect the following layout, without an extra nested dataset directory:

```text
data/RadioMapSeer/
  gain/DPM/<map>_<tx>.png
  png/buildings_complete/<map>.png
  png/antennas/<map>_<tx>.png
data/USC/
  city_map/<sample>.png
  tx_map/<sample>.png
  power_map/<sample>.png
```

USC samples must have matching filenames across the three directories.

## Training

Run all commands from the repository root. Main and ablation entry points default to RadioMapSeer; use `--dataset usc --data_dir path/to/USC_2` for the prepared USC dataset. All trainers accept `--data_dir` and `--num_workers`. Frequency defaults to 5.9 GHz for RadioMapSeer and 2.5 GHz for USC. The historical scripts and current entry points default to `--dx 1 --dy 1`;

### Ablations

The same two-stage model without physics losses:

```bash
python -m training.phy2_exposnet_ablation --ablation_mode no_phys \
  --epochs 50 --batch_size 16 --lr 1e-4 \
  --out_dir outputs/no_phys
```

The single-stage refinement model supervised by MSE:

```bash
python -m training.phy2_exposnet_ablation \
  --ablation_mode stage2_mse --epochs 50 --batch_size 16 \
  --lr 1e-4 --out_dir outputs/stage2
```

The three paper modes are `full`, `no_phys`, and `stage2_mse`. Full/no-physics reuse the main model, including its Dropout and residual refinement; they differ only in whether the physical losses are added. Full uses the main training objective, and both MSE-only ablations use MSE. `stage1_vie` is an additional physics-only experiment outside the three Fig. 3 variants. `phy2_exposnet_loss` also supports PDE-only (`--lam_H 0.5 --lam_VIE 0`) and VIE-only (`--lam_H 0 --lam_VIE 0.5`) experiments.

## Citation

```bibtex
@article{li2026phy2,
  title={Phy2-ExposNet: A Physics-Informed Neural Network for Urban EMF Exposure Mapping},
  author={Li, Shuangning and Zhang, Yarui and Wang, Shanshan and Wiart, Joe},
  journal={IEEE Open Journal of Antennas and Propagation},
  year={2026},
  publisher={IEEE}
}
```

Please also cite the dataset authors and relevant baseline methods when using their work.
