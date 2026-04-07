from __future__ import annotations
"""
Loss functions for the Cosserat rod dynamics surrogate.

Two loss modes:
  1. single_step_loss
       MSE between FNO output and precomputed IMEX targets (normalised space).

  2. rollout_loss
       K-step autoregressive rollout using IMEX dynamics on element grid.
       Uses teacher forcing for Q and r; propagates v_elem and omega.
       IMEX step: v_next = (I - dt*L_v)^{-1} (v + dt*N_theta)

Total loss = single_step_loss  +  rollout_weight * rollout_loss
"""

import numpy as np
import torch
import torch.nn.functional as F
from .operators import DiscreteOperators
from .config import DynamicsConfig


# --------------------------------------------------------------------------- #
#  Single-step loss                                                            #
# --------------------------------------------------------------------------- #

def single_step_loss(pred: torch.Tensor,
                     target: torch.Tensor,
                     mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    MSE loss between predicted and target N_theta (normalised).

    Parameters
    ----------
    pred   : (B, 6, n_elem)  normalised FNO output
    target : (B, 6, n_elem)  normalised training target
    mask   : (B,) bool        optional sample mask

    Returns
    -------
    loss : scalar
    """
    err = F.mse_loss(pred, target, reduction="none")   # (B, 6, n_elem)
    loss = err.mean(dim=[1, 2])                        # (B,)

    if mask is not None:
        loss = loss[mask]

    return loss.mean()


# --------------------------------------------------------------------------- #
#  Rollout loss (damped-residual dynamics on element grid)                     #
# --------------------------------------------------------------------------- #

def rollout_loss(surrogate,
                 batch: dict,
                 ops: DiscreteOperators,
                 norm_stats: dict | None,
                 cfg: DynamicsConfig,
                 device: str) -> torch.Tensor:
    """
    K-step rollout loss using IMEX dynamics on element grid.

    Strategy:
      - Start from GT v_elem^0, omega^0
      - At each step k, use GT FNO input but replace v_elem, omega,
        and delta channels with predicted values
      - IMEX step: v_next = A_v_elem_inv @ (v + dt*FNO_v)
      - Loss = discounted MSE of (v_elem, omega) against GT
    """
    K      = cfg.rollout_k
    gamma  = cfg.rollout_gamma
    dt     = cfg.effective_dt

    inputs_seq  = batch["inputs"].to(device)     # (K+1, B, 30, n_elem)
    v_elem_seq  = batch["v_elem"].to(device)     # (K+1, B,  3, n_elem)
    omega_seq   = batch["omega"].to(device)      # (K+1, B,  3, n_elem)

    B  = inputs_seq.shape[1]

    # Normalisation tensors
    if norm_stats is not None:
        def _to(arr):
            if isinstance(arr, np.ndarray):
                return torch.from_numpy(arr).to(device).unsqueeze(0).unsqueeze(-1)
            return torch.tensor(arr, device=device)

        mean_ve = _to(norm_stats["v_elem"][0])
        std_ve  = _to(norm_stats["v_elem"][1])
        mean_om = _to(norm_stats["omega"][0])
        std_om  = _to(norm_stats["omega"][1])
        mean_dv = _to(norm_stats["delta_v"][0])
        std_dv  = _to(norm_stats["delta_v"][1])
        mean_dw = _to(norm_stats["delta_omega"][0])
        std_dw  = _to(norm_stats["delta_omega"][1])
        mean_tv = _to(norm_stats["tgt_v_elem"][0])
        std_tv  = _to(norm_stats["tgt_v_elem"][1])
        mean_tw = _to(norm_stats["tgt_omega"][0])
        std_tw  = _to(norm_stats["tgt_omega"][1])
    else:
        z = torch.zeros(1, 1, 1, device=device)
        o = torch.ones(1, 1, 1, device=device)
        mean_ve = mean_om = mean_dv = mean_dw = mean_tv = mean_tw = z
        std_ve = std_om = std_dv = std_dw = std_tv = std_tw = o

    # Initialise with GT state at t=0
    v_elem_pred = v_elem_seq[0]    # (B, 3, n_elem)  physical
    omega_pred  = omega_seq[0]     # (B, 3, n_elem)   physical
    v_elem_prev = v_elem_seq[0]    # for delta at step 0
    omega_prev  = omega_seq[0]

    total_loss   = torch.tensor(0.0, device=device)
    total_weight = 0.0

    for k in range(K):
        # Build FNO input: GT context with predicted velocity channels
        fno_input_k = inputs_seq[k].clone()   # (B, 30, n_elem)

        # Replace v_elem and omega channels (0:3 and 3:6)
        fno_input_k[:, 0:3, :] = (v_elem_pred - mean_ve) / std_ve
        fno_input_k[:, 3:6, :] = (omega_pred  - mean_om) / std_om

        # Replace delta channels (24:27 and 27:30)
        if k > 0:
            dv = v_elem_pred - v_elem_prev
            dw = omega_pred  - omega_prev
            fno_input_k[:, 24:27, :] = (dv - mean_dv) / std_dv
            fno_input_k[:, 27:30, :] = (dw - mean_dw) / std_dw
        # k==0: delta channels remain as GT from dataset

        # FNO forward
        pred_norm = surrogate(fno_input_k)          # (B, 6, n_elem) normalised

        # Denorm FNO output to physical units
        ab_v_phys  = pred_norm[:, :3, :] * std_tv + mean_tv
        ab_om_phys = pred_norm[:, 3:, :] * std_tw + mean_tw

        # IMEX step: (I - dt*L)^{-1} @ (u + dt*N_theta)
        rhs_v = v_elem_pred + dt * ab_v_phys
        rhs_w = omega_pred  + dt * ab_om_phys
        v_elem_next = ops.solve_v_elem(rhs_v)
        omega_next  = ops.solve_omega(rhs_w)

        # Loss against GT at step k+1
        v_elem_gt = v_elem_seq[k + 1]
        omega_gt  = omega_seq[k + 1]
        loss_v    = F.mse_loss(v_elem_next, v_elem_gt)
        loss_omega = F.mse_loss(omega_next,  omega_gt)
        step_loss = loss_v + loss_omega

        weight = gamma ** (K - 1 - k)
        total_loss   = total_loss   + weight * step_loss
        total_weight = total_weight + weight

        # Update state for next step (detach to prevent gradient explosion)
        v_elem_prev = v_elem_pred.detach()
        omega_prev  = omega_pred.detach()
        v_elem_pred = v_elem_next.detach()
        omega_pred  = omega_next.detach()

    return total_loss / max(total_weight, 1e-8)


# --------------------------------------------------------------------------- #
#  Combined loss                                                               #
# --------------------------------------------------------------------------- #

def compute_total_loss(surrogate,
                       single_batch: dict,
                       rollout_batch: dict | None,
                       ops: DiscreteOperators,
                       norm_stats: dict | None,
                       cfg: DynamicsConfig,
                       device: str,
                       use_rollout: bool = False,
                       ) -> tuple[torch.Tensor, dict]:
    """Compute total training loss = single_step + rollout (if enabled)."""
    info = {}

    # --- Single-step loss ---
    fno_input = single_batch["input"].to(device)
    target    = single_batch["target"].to(device)

    pred = surrogate(fno_input)
    loss_single = single_step_loss(pred, target)
    info["loss_single"] = loss_single.item()

    total = loss_single

    # --- Rollout loss ---
    if use_rollout and rollout_batch is not None:
        def to_seq(t):
            """Transpose DataLoader output from (B, K+1, ...) to (K+1, B, ...)."""
            return t.transpose(0, 1)

        rb = {
            "inputs":  to_seq(rollout_batch["inputs"]),    # (K+1, B, 30, ne)
            "targets": to_seq(rollout_batch["targets"]),   # (K,   B,  6, ne)
            "v_elem":  to_seq(rollout_batch["v_elem"]),    # (K+1, B,  3, ne)
            "omega":   to_seq(rollout_batch["omega"]),     # (K+1, B,  3, ne)
        }

        loss_ro = rollout_loss(surrogate, rb, ops, norm_stats, cfg, device)
        info["loss_rollout"] = loss_ro.item()
        total = total + cfg.rollout_weight * loss_ro
    else:
        info["loss_rollout"] = 0.0

    info["loss_total"] = total.item()
    return total, info
