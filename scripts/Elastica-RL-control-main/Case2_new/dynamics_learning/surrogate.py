from __future__ import annotations
"""
CosseratSurrogate: the full learned dynamics model.

Combines:
  - FNO1d             : neural network  N_theta(u, a, phi)
  - IMEX time step    : dissipative operator L treated implicitly on element grid

IMEX dynamics step (element grid, no node↔element interpolation):
    (I − Δt·L_v)  v_elem^{n+1}  =  v_elem^n  +  Δt · N_theta^v(FNO)
    (I − Δt·L_ω)  omega^{n+1}   =  omega^n   +  Δt · N_theta^ω(FNO)

where L = α·∂_ss − β·I is the dissipative linear operator.

Kinematic reconstruction (for position / orientation):
    v_node^{n+1}  = elem_to_node(v_elem^{n+1})
    r^{n+1}       = r^n  +  Δt * v_node^{n+1}
    Q^{n+1}       ≈ orthonorm( Q^n + Δt * skew(omega^{n+1}) @ Q^n )
"""

import torch
import torch.nn as nn
import numpy as np

from .fno1d import FNO1d
from .operators import DiscreteOperators
from .config import DynamicsConfig


# --------------------------------------------------------------------------- #
#  Kinematic helpers                                                           #
# --------------------------------------------------------------------------- #

def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    """Build skew-symmetric matrix from 3-vectors.  v: (..., 3) -> S: (..., 3, 3)"""
    *lead, _ = v.shape
    S = torch.zeros(*lead, 3, 3, dtype=v.dtype, device=v.device)
    S[..., 0, 1] = -v[..., 2]
    S[..., 0, 2] =  v[..., 1]
    S[..., 1, 0] =  v[..., 2]
    S[..., 1, 2] = -v[..., 0]
    S[..., 2, 0] = -v[..., 1]
    S[..., 2, 1] =  v[..., 0]
    return S


def update_directors(Q: torch.Tensor, omega: torch.Tensor, dt: float) -> torch.Tensor:
    """
    First-order director update with orthonormalisation via SVD.
    Q: (B, 3, 3, n_elem), omega: (B, 3, n_elem), returns Q_new: (B, 3, 3, n_elem)
    """
    B, _, _, n_elem = Q.shape
    omega_t = omega.permute(0, 2, 1)                   # (B, n_elem, 3)
    S = skew_symmetric(omega_t)                         # (B, n_elem, 3, 3)
    Q_t = Q.permute(0, 3, 1, 2)                        # (B, n_elem, 3, 3)

    Q_new_t = Q_t + dt * torch.bmm(
        S.reshape(B * n_elem, 3, 3),
        Q_t.reshape(B * n_elem, 3, 3)
    ).reshape(B, n_elem, 3, 3)

    U, _, Vh = torch.linalg.svd(Q_new_t)
    Q_orth_t = torch.bmm(
        U.reshape(B * n_elem, 3, 3),
        Vh.reshape(B * n_elem, 3, 3)
    ).reshape(B, n_elem, 3, 3)

    return Q_orth_t.permute(0, 2, 3, 1)


# --------------------------------------------------------------------------- #
#  CosseratSurrogate                                                           #
# --------------------------------------------------------------------------- #

class CosseratSurrogate(nn.Module):
    """
    Full dynamics surrogate = FNO + damped-residual solver.

    Parameters
    ----------
    cfg : DynamicsConfig
    ops : DiscreteOperators   (kept for static utilities)
    norm_stats : dict | None  — normalisation stats from training dataset
    """

    def __init__(self,
                 cfg: DynamicsConfig,
                 ops: DiscreteOperators,
                 norm_stats: dict | None = None):
        super().__init__()

        self.cfg  = cfg
        self.ops  = ops
        self.norm_stats = norm_stats
        self.dt   = cfg.effective_dt

        # FNO (trainable)
        self.fno = FNO1d(
            in_channels     = cfg.fno_in_channels,
            out_channels    = cfg.fno_out_channels,
            modes           = cfg.fno_modes,
            hidden_channels = cfg.fno_hidden_channels,
            n_layers        = cfg.fno_n_layers,
            dropout         = cfg.fno_dropout,
        )

    # ---------------------------------------------------------------------- #
    #  Forward: predict normalised N_theta from normalised FNO input          #
    # ---------------------------------------------------------------------- #

    def forward(self, fno_input: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass.
        fno_input : (B, 30, n_elem)  normalised
        returns   : (B, 6, n_elem)   normalised N_theta prediction
        """
        return self.fno(fno_input)

    # ---------------------------------------------------------------------- #
    #  Damped-residual dynamics step (element level)                          #
    # ---------------------------------------------------------------------- #

    def dynamics_step(self,
                      v_elem:    torch.Tensor,
                      omega:     torch.Tensor,
                      ab_v_elem: torch.Tensor,
                      ab_omega:  torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        One-step IMEX dynamics (all on element grid):
            (I − Δt·L_v)  v_next  =  v_elem  +  Δt · ab_v_elem
            (I − Δt·L_ω)  ω_next  =  omega   +  Δt · ab_omega

        Solved via precomputed inverses A⁻¹.

        Parameters
        ----------
        v_elem    : (B, 3, n_elem)
        omega     : (B, 3, n_elem)
        ab_v_elem : (B, 3, n_elem)  — FNO output (physical units)
        ab_omega  : (B, 3, n_elem)

        Returns
        -------
        v_elem_next, omega_next : (B, 3, n_elem) each
        """
        rhs_v = v_elem + self.dt * ab_v_elem
        rhs_w = omega  + self.dt * ab_omega
        v_elem_next = self.ops.solve_v_elem(rhs_v)
        omega_next  = self.ops.solve_omega(rhs_w)
        return v_elem_next, omega_next

    # ---------------------------------------------------------------------- #
    #  Rollout step for multi-step training loss (teacher forcing on Q, r)    #
    # ---------------------------------------------------------------------- #

    def rollout_step_training(self,
                              fno_input:   torch.Tensor,
                              v_elem:      torch.Tensor,
                              omega:       torch.Tensor,
                              norm_stats:  dict | None = None
                              ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        One rollout step in training mode.
        Teacher forcing for context (Q, r, target come from GT FNO input).
        Only v_elem and omega are propagated from predictions.

        Parameters
        ----------
        fno_input : (B, 30, n_elem)  normalised (with v/omega channels replaced)
        v_elem    : (B, 3, n_elem)   physical units
        omega     : (B, 3, n_elem)   physical units

        Returns
        -------
        v_elem_next, omega_next : (B, 3, n_elem) physical units
        """
        pred_norm = self.fno(fno_input)
        ab_v_phys, ab_omega_phys = self._denorm_prediction(pred_norm, norm_stats)
        return self.dynamics_step(v_elem, omega, ab_v_phys, ab_omega_phys)

    # ---------------------------------------------------------------------- #
    #  Full inference rollout (for MPC / evaluation)                          #
    # ---------------------------------------------------------------------- #

    def rollout_step_inference(self,
                               v_elem:      torch.Tensor,
                               omega:       torch.Tensor,
                               r_node:      torch.Tensor,
                               Q_elem:      torch.Tensor,
                               a_spatial:   torch.Tensor,
                               tgt_pos:     torch.Tensor,
                               tgt_quat:    torch.Tensor,
                               delta_v:     torch.Tensor,
                               delta_omega: torch.Tensor,
                               norm_stats:  dict | None = None,
                               ) -> tuple[torch.Tensor, torch.Tensor,
                                          torch.Tensor, torch.Tensor]:
        """
        Single full-state surrogate step for MPC rollouts.

        Parameters
        ----------
        v_elem      : (B, 3, n_elem)
        omega       : (B, 3, n_elem)
        r_node      : (B, 3, n_nodes)
        Q_elem      : (B, 3, 3, n_elem)
        a_spatial   : (B, 3, n_elem)
        tgt_pos     : (B, 3)
        tgt_quat    : (B, 4)
        delta_v     : (B, 3, n_elem)   temporal velocity change
        delta_omega : (B, 3, n_elem)   temporal omega change
        norm_stats  : dict

        Returns
        -------
        v_elem_next, omega_next, r_next, Q_next
        """
        B, _, n_elem = omega.shape
        device = v_elem.device

        r_elem = DiscreteOperators.node_to_elem(r_node)
        Q_quat = _rotmat_to_quat_torch(Q_elem)

        s_norm = torch.linspace(0.5 / n_elem, 1.0 - 0.5 / n_elem,
                                n_elem, device=device)
        s_norm = s_norm.unsqueeze(0).unsqueeze(0).expand(B, 1, n_elem)

        tgt_pos_b  = tgt_pos.unsqueeze(-1).expand(-1, -1, n_elem)
        tgt_quat_b = tgt_quat.unsqueeze(-1).expand(-1, -1, n_elem)

        fno_raw = torch.cat([v_elem, omega, r_elem, Q_quat,
                             a_spatial, tgt_pos_b, tgt_quat_b, s_norm,
                             delta_v, delta_omega], dim=1)

        fno_input = self._norm_input(fno_raw, norm_stats)
        pred_norm = self.fno(fno_input)
        ab_v_phys, ab_omega_phys = self._denorm_prediction(pred_norm, norm_stats)

        v_elem_next, omega_next = self.dynamics_step(
            v_elem, omega, ab_v_phys, ab_omega_phys)

        # Kinematic update
        v_node_next = DiscreteOperators.elem_to_node(v_elem_next, bc_start=0.0)
        r_next = r_node + self.dt * v_node_next
        Q_next = update_directors(Q_elem, omega_next, self.dt)

        return v_elem_next, omega_next, r_next, Q_next

    # ---------------------------------------------------------------------- #
    #  Normalisation helpers                                                   #
    # ---------------------------------------------------------------------- #

    def _to_bcast(self, arr, device):
        """Convert numpy array (C,) to tensor (1, C, 1) for broadcasting."""
        return torch.from_numpy(arr).to(device).unsqueeze(0).unsqueeze(-1)

    def _norm_input(self, x_raw: torch.Tensor, norm_stats: dict | None) -> torch.Tensor:
        """Normalise a (B, 30, n_elem) raw input tensor."""
        if norm_stats is None:
            return x_raw

        device = x_raw.device
        out = x_raw.clone()
        slices_keys = [
            (slice(0,  3),  "v_elem"),
            (slice(3,  6),  "omega"),
            (slice(6,  9),  "r_elem"),
            (slice(9,  13), "Q_quat"),
            (slice(13, 16), "a_spatial"),
            (slice(16, 19), "tgt_pos"),
            (slice(19, 23), "tgt_quat"),
            # s_norm (channel 23) not normalised — already in [0, 1]
            (slice(24, 27), "delta_v"),
            (slice(27, 30), "delta_omega"),
        ]
        for sl, key in slices_keys:
            mean, std = norm_stats[key]
            m = self._to_bcast(mean, device)
            s = self._to_bcast(std, device)
            out[:, sl, :] = (out[:, sl, :] - m) / s
        return out

    def _denorm_prediction(self,
                           pred_norm: torch.Tensor,
                           norm_stats: dict | None
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        """Denormalise FNO output (B, 6, n_elem) → physical-unit (ab_v, ab_omega)."""
        ab_v_norm  = pred_norm[:, :3, :]
        ab_om_norm = pred_norm[:, 3:, :]

        if norm_stats is None:
            return ab_v_norm, ab_om_norm

        device = pred_norm.device
        mean_v = self._to_bcast(norm_stats["tgt_v_elem"][0], device)
        std_v  = self._to_bcast(norm_stats["tgt_v_elem"][1], device)
        mean_w = self._to_bcast(norm_stats["tgt_omega"][0], device)
        std_w  = self._to_bcast(norm_stats["tgt_omega"][1], device)

        ab_v_phys  = ab_v_norm  * std_v  + mean_v
        ab_om_phys = ab_om_norm * std_w  + mean_w
        return ab_v_phys, ab_om_phys


# --------------------------------------------------------------------------- #
#  Torch quaternion utility                                                    #
# --------------------------------------------------------------------------- #

def _rotmat_to_quat_torch(Q: torch.Tensor) -> torch.Tensor:
    """Q: (B, 3, 3, n_elem) → quat: (B, 4, n_elem)"""
    B, _, _, n_elem = Q.shape
    R = Q.permute(0, 3, 1, 2)
    tr = R[:, :, 0, 0] + R[:, :, 1, 1] + R[:, :, 2, 2]
    qw = 0.5 * torch.sqrt(torch.clamp(1.0 + tr, min=1e-10))
    denom = torch.clamp(4.0 * qw, min=1e-10)
    qx = (R[:, :, 2, 1] - R[:, :, 1, 2]) / denom
    qy = (R[:, :, 0, 2] - R[:, :, 2, 0]) / denom
    qz = (R[:, :, 1, 0] - R[:, :, 0, 1]) / denom
    quat = torch.stack([qw, qx, qy, qz], dim=-1)
    return quat.permute(0, 2, 1)
