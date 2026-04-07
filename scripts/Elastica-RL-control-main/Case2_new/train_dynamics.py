"""
Main entry point for Cosserat rod dynamics learning.

Usage
-----
# Default training (all settings from DynamicsConfig):
    python train_dynamics.py

# Override key settings via command line:
    python train_dynamics.py --n_epochs 150 --batch_size 512 --lr 5e-4
    python train_dynamics.py --no_rollout          # single-step only
    python train_dynamics.py --resume checkpoints_dynamics/ckpt_last.pt
    python train_dynamics.py --device cpu

# Inspect data only (no training):
    python train_dynamics.py --dry_run

Full pipeline:
  1. Build DynamicsConfig (+ CLI overrides)
  2. Build DiscreteOperators (L matrices, IMEX matrices)
  3. Build CosseratDataset (train + val, preprocessing)
  4. Build CosseratSurrogate (FNO + IMEX buffers)
  5. Build Trainer and run training loop
"""

import argparse
import os
import sys
import torch

# Make sure the Case2 directory is on the path
sys.path.insert(0, os.path.dirname(__file__))

from dynamics_learning.config     import DynamicsConfig
from dynamics_learning.operators  import DiscreteOperators
from dynamics_learning.dataset    import create_dataloaders
from dynamics_learning.surrogate  import CosseratSurrogate
from dynamics_learning.trainer    import Trainer


# --------------------------------------------------------------------------- #
#  Argument parser                                                             #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser("Train Cosserat rod dynamics surrogate (FNO + IMEX)")

    # Data
    p.add_argument("--data_path",  type=str, default=None,
                   help="Path to .npz dataset (default: from DynamicsConfig)")
    p.add_argument("--train_ratio", type=float, default=None)

    # Dissipative operator
    p.add_argument("--alpha_v",     type=float, default=None,
                   help="Spatial diffusivity for v  (default: 1e-3)")
    p.add_argument("--alpha_omega", type=float, default=None)
    p.add_argument("--beta_v",      type=float, default=None,
                   help="Damping rate for v (= NU, default: 10.0)")
    p.add_argument("--beta_omega",  type=float, default=None)

    # FNO architecture
    p.add_argument("--fno_modes",    type=int, default=None)
    p.add_argument("--fno_hidden",   type=int, default=None)
    p.add_argument("--fno_layers",   type=int, default=None)
    p.add_argument("--fno_dropout",  type=float, default=None,
                   help="Dropout rate in FNO blocks (default: 0.1)")

    # Training
    p.add_argument("--n_epochs",    type=int,   default=None)
    p.add_argument("--batch_size",  type=int,   default=None)
    p.add_argument("--lr",          type=float, default=None)
    p.add_argument("--no_rollout",  action="store_true",
                   help="Disable multi-step rollout loss")
    p.add_argument("--rollout_k",   type=int,   default=None)
    p.add_argument("--rollout_start", type=int, default=None,
                   help="Epoch to switch on rollout loss")

    # Checkpointing
    p.add_argument("--checkpoint_dir", type=str,  default=None)
    p.add_argument("--resume",         type=str,  default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--save_every",     type=int,  default=None)

    # System
    p.add_argument("--device",       type=str,  default=None,
                   help="'cuda', 'cpu', or 'auto'")
    p.add_argument("--num_workers",  type=int,  default=0,
                   help="DataLoader num_workers (0 = main process, safe on Windows)")

    # Subsampling
    p.add_argument("--subsample", type=int, default=None,
                   help="Temporal subsampling factor (default: 1, recommended: 5)")
    p.add_argument("--lr_schedule", type=str, default=None,
                   help="LR schedule: 'cosine' or 'step'")

    # Misc
    p.add_argument("--dry_run", action="store_true",
                   help="Only build dataset and print stats, do not train")
    p.add_argument("--seed",    type=int, default=42)

    return p.parse_args()


# --------------------------------------------------------------------------- #
#  Build config from defaults + CLI overrides                                 #
# --------------------------------------------------------------------------- #

def build_config(args) -> DynamicsConfig:
    cfg = DynamicsConfig()

    # Data
    if args.data_path    is not None: cfg.data_path    = args.data_path
    if args.train_ratio  is not None: cfg.train_ratio  = args.train_ratio

    # Operator
    if args.alpha_v      is not None: cfg.alpha_v      = args.alpha_v
    if args.alpha_omega  is not None: cfg.alpha_omega  = args.alpha_omega
    if args.beta_v       is not None: cfg.beta_v       = args.beta_v
    if args.beta_omega   is not None: cfg.beta_omega   = args.beta_omega

    # FNO
    if args.fno_modes    is not None: cfg.fno_modes           = args.fno_modes
    if args.fno_hidden   is not None: cfg.fno_hidden_channels  = args.fno_hidden
    if args.fno_layers   is not None: cfg.fno_n_layers         = args.fno_layers
    if args.fno_dropout  is not None: cfg.fno_dropout          = args.fno_dropout

    # Training
    if args.n_epochs     is not None: cfg.n_epochs       = args.n_epochs
    if args.batch_size   is not None: cfg.batch_size      = args.batch_size
    if args.lr           is not None: cfg.learning_rate   = args.lr
    if args.no_rollout:               cfg.use_rollout_loss = False
    if args.rollout_k    is not None: cfg.rollout_k       = args.rollout_k
    if args.rollout_start is not None: cfg.rollout_start_epoch = args.rollout_start

    # Checkpointing
    if args.checkpoint_dir is not None: cfg.checkpoint_dir = args.checkpoint_dir
    if args.save_every     is not None: cfg.save_every      = args.save_every

    # Device
    if args.device is not None: cfg.device = args.device

    # Subsampling & LR schedule
    if args.subsample    is not None: cfg.subsample_factor = args.subsample
    if args.lr_schedule  is not None: cfg.lr_schedule      = args.lr_schedule

    return cfg


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)

    # ---------------------------------------------------------------------- #
    #  1. Config                                                               #
    # ---------------------------------------------------------------------- #
    cfg = build_config(args)
    device = cfg.resolved_device
    print(cfg.summary())
    print(f"[Main] Device: {device}")

    # ---------------------------------------------------------------------- #
    #  2. Discrete Operators                                                   #
    # ---------------------------------------------------------------------- #
    print("[Main] Building discrete L operators ...")
    ops = DiscreteOperators(
        n_elem       = cfg.n_elem,
        ds           = cfg.ds,
        dt           = cfg.effective_dt,
        alpha_v      = cfg.alpha_v,
        beta_v       = cfg.beta_v,
        alpha_omega  = cfg.alpha_omega,
        beta_omega   = cfg.beta_omega,
    )
    print(ops.summary())
    ops.to_torch(device=device)

    # ---------------------------------------------------------------------- #
    #  3. Datasets and DataLoaders                                             #
    # ---------------------------------------------------------------------- #
    print("[Main] Building datasets ...")
    loaders, norm_stats = create_dataloaders(
        cfg,
        ops,
        rollout_k   = cfg.rollout_k,
        num_workers = args.num_workers,
    )

    if args.dry_run:
        print("\n[Main] --dry_run: dataset stats only, no training.")
        _print_dataset_stats(loaders, norm_stats)
        return

    # ---------------------------------------------------------------------- #
    #  4. Surrogate model                                                      #
    # ---------------------------------------------------------------------- #
    print("[Main] Building surrogate model ...")
    surrogate = CosseratSurrogate(cfg, ops, norm_stats)
    print(f"[Main] FNO parameters: {surrogate.fno.count_parameters():,}")

    # ---------------------------------------------------------------------- #
    #  5. Trainer                                                              #
    # ---------------------------------------------------------------------- #
    trainer = Trainer(surrogate, loaders, ops, norm_stats, cfg)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # ---------------------------------------------------------------------- #
    #  6. Train                                                                #
    # ---------------------------------------------------------------------- #
    trainer.train()

    # ---------------------------------------------------------------------- #
    #  7. Final rollout evaluation on validation set                          #
    # ---------------------------------------------------------------------- #
    print("\n[Main] Running final rollout evaluation ...")
    rollout_metrics = trainer.evaluate_rollout_error(n_episodes=50)
    for k, v in rollout_metrics.items():
        print(f"  {k}: {v:.4e}")


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _print_dataset_stats(loaders: dict, norm_stats: dict):
    print("\n--- Dataset info ---")
    for name, loader in loaders.items():
        ds = loader.dataset
        print(f"  {name}: {len(ds):,} samples, batch_size={loader.batch_size}")

    if norm_stats is not None:
        import numpy as np
        print("\n--- Normalisation statistics (per-channel) ---")
        for key, (mean, std) in norm_stats.items():
            mean_str = ", ".join(f"{m:+.4e}" for m in np.atleast_1d(mean))
            std_str  = ", ".join(f"{s:.4e}" for s in np.atleast_1d(std))
            print(f"  {key:20s}: mean=[{mean_str}]  std=[{std_str}]")

    print("\n--- Quick batch test ---")
    batch = next(iter(loaders["single_train"]))
    print(f"  input shape:  {batch['input'].shape}")
    print(f"  target shape: {batch['target'].shape}")
    print(f"  v_elem shape: {batch['v_elem'].shape}")
    print(f"  omega shape:  {batch['omega'].shape}")

    rollout_batch = next(iter(loaders["rollout_train"]))
    print(f"  rollout inputs shape:  {rollout_batch['inputs'].shape}")
    print(f"  rollout targets shape: {rollout_batch['targets'].shape}")


if __name__ == "__main__":
    main()
