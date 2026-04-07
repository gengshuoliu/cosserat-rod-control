"""
Trainer for the Cosserat rod dynamics surrogate.

Training schedule:
  Phase 1 (epoch 0 .. rollout_start_epoch-1):
      Single-step loss only.

  Phase 2 (epoch rollout_start_epoch .. n_epochs-1):
      Single-step loss  +  multi-step rollout loss.

Checkpoint format:
    {
        "epoch":       int,
        "model_state": ...,
        "optim_state": ...,
        "sched_state": ...,
        "norm_stats":  dict,
        "best_val":    float,
        "cfg":         dict,
    }
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from itertools import cycle

from .config import DynamicsConfig
from .surrogate import CosseratSurrogate
from .operators import DiscreteOperators
from .losses import compute_total_loss


class Trainer:
    """
    Manages the training loop for CosseratSurrogate.

    Parameters
    ----------
    surrogate    : CosseratSurrogate
    loaders      : dict — keys "single_train", "single_val",
                           "rollout_train", "rollout_val"
    ops          : DiscreteOperators (already moved to device)
    norm_stats   : dict  (from training dataset)
    cfg          : DynamicsConfig
    """

    def __init__(self,
                 surrogate:   CosseratSurrogate,
                 loaders:     dict,
                 ops:         DiscreteOperators,
                 norm_stats:  dict,
                 cfg:         DynamicsConfig):

        self.surrogate  = surrogate
        self.loaders    = loaders
        self.ops        = ops
        self.norm_stats = norm_stats
        self.cfg        = cfg
        self.device     = cfg.resolved_device

        # Move model to device
        self.surrogate.to(self.device)

        # Optimiser: AdamW with weight decay
        self.optim = torch.optim.AdamW(
            surrogate.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        # LR scheduler
        if cfg.lr_schedule == "cosine":
            self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optim, T_max=cfg.n_epochs, eta_min=1e-6,
            )
        else:
            self.sched = torch.optim.lr_scheduler.MultiStepLR(
                self.optim,
                milestones=list(cfg.lr_milestones),
                gamma=cfg.lr_gamma,
            )

        # Checkpoint directory
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

        # Training state
        self.start_epoch = 0
        self.best_val    = float("inf")
        self.history     = []   # list of dicts (one per epoch)

    # ---------------------------------------------------------------------- #
    #  Checkpoint I/O                                                         #
    # ---------------------------------------------------------------------- #

    def save_checkpoint(self, epoch: int, tag: str = "last"):
        path = os.path.join(self.cfg.checkpoint_dir, f"ckpt_{tag}.pt")
        torch.save({
            "epoch":       epoch,
            "model_state": self.surrogate.state_dict(),
            "optim_state": self.optim.state_dict(),
            "sched_state": self.sched.state_dict(),
            "norm_stats":  self.norm_stats,
            "best_val":    self.best_val,
            "cfg":         self.cfg.__dict__,
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.surrogate.load_state_dict(ckpt["model_state"])
        self.optim.load_state_dict(ckpt["optim_state"])
        self.sched.load_state_dict(ckpt["sched_state"])
        self.norm_stats = ckpt["norm_stats"]
        self.best_val   = ckpt.get("best_val", float("inf"))
        self.start_epoch = ckpt["epoch"] + 1
        print(f"[Trainer] Loaded checkpoint from {path} (epoch {ckpt['epoch']})")

    # ---------------------------------------------------------------------- #
    #  Single epoch helpers                                                   #
    # ---------------------------------------------------------------------- #

    def _train_epoch(self, epoch: int) -> dict:
        """Run one training epoch, return dict of metrics."""
        self.surrogate.train()
        use_rollout = (self.cfg.use_rollout_loss and
                       epoch >= self.cfg.rollout_start_epoch)

        single_loader  = self.loaders["single_train"]
        rollout_loader = self.loaders["rollout_train"] if use_rollout else None
        rollout_iter   = cycle(rollout_loader) if rollout_loader else None

        total_single = 0.0
        total_ro     = 0.0
        n_batches    = 0

        for single_batch in single_loader:
            self.optim.zero_grad(set_to_none=True)

            rollout_batch = next(rollout_iter) if rollout_iter else None

            loss, info = compute_total_loss(
                self.surrogate,
                single_batch,
                rollout_batch,
                self.ops,
                self.norm_stats,
                self.cfg,
                self.device,
                use_rollout=use_rollout,
            )

            loss.backward()

            if self.cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.surrogate.parameters(), self.cfg.grad_clip)

            self.optim.step()

            total_single += info["loss_single"]
            total_ro     += info["loss_rollout"]
            n_batches    += 1

        return {
            "train/loss_single":  total_single / max(n_batches, 1),
            "train/loss_rollout": total_ro     / max(n_batches, 1),
        }

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> dict:
        """Run one validation epoch, return dict of metrics."""
        self.surrogate.eval()
        use_rollout = (self.cfg.use_rollout_loss and
                       epoch >= self.cfg.rollout_start_epoch)

        single_loader  = self.loaders["single_val"]
        rollout_loader = self.loaders["rollout_val"] if use_rollout else None
        rollout_iter   = cycle(rollout_loader) if rollout_loader else None

        total_single = 0.0
        total_ro     = 0.0
        n_batches    = 0

        for single_batch in single_loader:
            rollout_batch = next(rollout_iter) if rollout_iter else None

            _, info = compute_total_loss(
                self.surrogate,
                single_batch,
                rollout_batch,
                self.ops,
                self.norm_stats,
                self.cfg,
                self.device,
                use_rollout=use_rollout,
            )

            total_single += info["loss_single"]
            total_ro     += info["loss_rollout"]
            n_batches    += 1

        return {
            "val/loss_single":  total_single / max(n_batches, 1),
            "val/loss_rollout": total_ro     / max(n_batches, 1),
        }

    # ---------------------------------------------------------------------- #
    #  Rollout evaluation (physical-unit velocity error)                      #
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def evaluate_rollout_error(self, n_episodes: int = 20) -> dict:
        """
        Evaluate tip velocity error over full episodes using IMEX
        dynamics on the element grid.

        Returns mean tip velocity RMSE at selected time steps.
        """
        self.surrogate.eval()

        val_ds = self.loaders["single_val"].dataset

        ep_indices = np.random.choice(val_ds.n_split,
                                      size=min(n_episodes, val_ds.n_split),
                                      replace=False)
        T  = val_ds.T
        ne = val_ds.n_elem

        eval_steps = sorted(set([T // 4, T // 2, T]))
        errors_per_step = {s: [] for s in eval_steps}

        for ep in ep_indices:
            # Initialise from GT state at t=0 (element grid)
            v_elem = torch.from_numpy(val_ds.v_elem[ep, 0]).unsqueeze(0).to(self.device)
            omega  = torch.from_numpy(val_ds.omega[ep, 0]).unsqueeze(0).to(self.device)
            v_elem_prev = v_elem.clone()
            omega_prev  = omega.clone()

            for t in range(T):
                fno_input = val_ds._build_input(ep, t)
                fno_input_t = fno_input.unsqueeze(0).to(self.device)

                # Replace velocity and delta channels with predicted values
                if val_ds.norm_stats is not None:
                    mv  = torch.from_numpy(val_ds.norm_stats["v_elem"][0]).to(self.device).unsqueeze(0).unsqueeze(-1)
                    sv  = torch.from_numpy(val_ds.norm_stats["v_elem"][1]).to(self.device).unsqueeze(0).unsqueeze(-1)
                    mo  = torch.from_numpy(val_ds.norm_stats["omega"][0]).to(self.device).unsqueeze(0).unsqueeze(-1)
                    so  = torch.from_numpy(val_ds.norm_stats["omega"][1]).to(self.device).unsqueeze(0).unsqueeze(-1)
                    mdv = torch.from_numpy(val_ds.norm_stats["delta_v"][0]).to(self.device).unsqueeze(0).unsqueeze(-1)
                    sdv = torch.from_numpy(val_ds.norm_stats["delta_v"][1]).to(self.device).unsqueeze(0).unsqueeze(-1)
                    mdw = torch.from_numpy(val_ds.norm_stats["delta_omega"][0]).to(self.device).unsqueeze(0).unsqueeze(-1)
                    sdw = torch.from_numpy(val_ds.norm_stats["delta_omega"][1]).to(self.device).unsqueeze(0).unsqueeze(-1)

                    fno_input_t[:, 0:3, :]  = (v_elem - mv) / sv
                    fno_input_t[:, 3:6, :]  = (omega  - mo) / so

                    dv = v_elem - v_elem_prev
                    dw = omega  - omega_prev
                    fno_input_t[:, 24:27, :] = (dv - mdv) / sdv
                    fno_input_t[:, 27:30, :] = (dw - mdw) / sdw

                # FNO forward
                pred_norm = self.surrogate(fno_input_t)
                ab_v_p, ab_om_p = self.surrogate._denorm_prediction(
                    pred_norm, val_ds.norm_stats
                )

                # Dynamics step (element grid, no node interpolation)
                v_elem_prev = v_elem.clone()
                omega_prev  = omega.clone()
                v_elem, omega = self.surrogate.dynamics_step(
                    v_elem, omega, ab_v_p, ab_om_p
                )

                # Compare tip velocity (last element) at selected steps
                step_idx = t + 1
                if step_idx in errors_per_step:
                    v_elem_gt = torch.from_numpy(
                        val_ds.v_elem[ep, step_idx, :, -1]
                    ).to(self.device)
                    tip_v_err = torch.norm(
                        v_elem[0, :, -1] - v_elem_gt
                    ).item()
                    errors_per_step[step_idx].append(tip_v_err)

        result = {}
        for step, errs in errors_per_step.items():
            if errs:
                result[f"val/tip_v_err_step{step}"] = float(np.mean(errs))

        return result

    # ---------------------------------------------------------------------- #
    #  Main training loop                                                     #
    # ---------------------------------------------------------------------- #

    def train(self):
        cfg = self.cfg
        print(cfg.summary())
        print(f"[Trainer] Starting training on {self.device}")
        print(f"[Trainer] Model: {self.surrogate.fno}")
        print(f"[Trainer] Train batches: {len(self.loaders['single_train'])}")
        print(f"[Trainer]   Val batches: {len(self.loaders['single_val'])}")

        for epoch in range(self.start_epoch, cfg.n_epochs):
            t0 = time.time()

            # --- Training ---
            train_metrics = self._train_epoch(epoch)

            # --- Validation ---
            val_metrics = self._val_epoch(epoch)

            # --- LR step ---
            self.sched.step()

            # --- Logging ---
            elapsed = time.time() - t0
            val_single = val_metrics["val/loss_single"]

            metrics = {**train_metrics, **val_metrics,
                       "epoch": epoch,
                       "lr": self.sched.get_last_lr()[0],
                       "elapsed_s": elapsed}
            self.history.append(metrics)

            if epoch % cfg.log_every == 0:
                ro_phase = epoch >= cfg.rollout_start_epoch and cfg.use_rollout_loss
                print(
                    f"[Ep {epoch:03d}/{cfg.n_epochs}] "
                    f"trn_single={train_metrics['train/loss_single']:.4e}  "
                    f"val_single={val_single:.4e}  "
                    f"{'[+rollout] ' if ro_phase else ''}"
                    f"lr={self.sched.get_last_lr()[0]:.2e}  "
                    f"t={elapsed:.1f}s"
                )

            # --- Best model checkpoint ---
            if val_single < self.best_val:
                self.best_val = val_single
                self.save_checkpoint(epoch, tag="best")

            # --- Periodic checkpoint ---
            if epoch % cfg.save_every == 0 or epoch == cfg.n_epochs - 1:
                self.save_checkpoint(epoch, tag="last")

        # --- Save final history ---
        history_path = os.path.join(cfg.checkpoint_dir, "history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\n[Trainer] Training complete. Best val_single={self.best_val:.4e}")
        print(f"[Trainer] Checkpoints saved to: {cfg.checkpoint_dir}/")
        print(f"[Trainer] History saved to:     {history_path}")
