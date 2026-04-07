"""
Configuration dataclass for Cosserat rod dynamics learning.

All physical parameters come from the PyElastica simulation setup in set_environment.py.
Damping beta_v, beta_omega come from AnalyticalLinearDamper(damping_constant=NU=10).
"""

from dataclasses import dataclass, field
import math
import os


@dataclass
class DynamicsConfig:
    # ------------------------------------------------------------------ #
    #  Physical / rod geometry  (must match the dataset generation setup)  #
    # ------------------------------------------------------------------ #
    n_elem: int = 40           # number of rod elements
    n_nodes: int = 41          # number of rod nodes  (= n_elem + 1)
    base_length: float = 1.0   # rod length [m]
    dt: float = 2.5006235969e-3   # [s] — from meta["env"]["dt_effective"]
    subsample_factor: int = 1     # temporal subsampling (1=no subsampling, 5=recommended)

    # ------------------------------------------------------------------ #
    #  Dissipative operator  L = α·∂_ss − β·I                            #
    #  IMEX step: (I − Δt·L) u^{n+1} = u^n + Δt·N_θ(u^n)              #
    #  β values from AnalyticalLinearDamper (NU=10)                      #
    # ------------------------------------------------------------------ #
    alpha_v: float = 1e-3        # spatial diffusivity for v  [m²/s]
    alpha_omega: float = 1e-3    # spatial diffusivity for ω  [m²/s]
    beta_v: float = 10.0         # damping rate for linear velocity  (= NU)
    beta_omega: float = 10.0     # damping rate for angular velocity (= NU)

    # ------------------------------------------------------------------ #
    #  B-spline actuation                                                  #
    # ------------------------------------------------------------------ #
    n_ctrl: int = 6            # number of interior control points per direction
    alpha_scale: float = 140.0 # torque scale [N·m] for bending
    beta_scale: float = 140.0  # torque scale [N·m] for twist

    # ------------------------------------------------------------------ #
    #  FNO architecture                                                    #
    #                                                                      #
    #  Input channels (30, per element):                                   #
    #    v_elem     (3)  linear velocity at element centres                #
    #    omega      (3)  angular velocity                                  #
    #    r_elem     (3)  position at element centres                       #
    #    Q_quat     (4)  quaternion from director_collection               #
    #    a_spatial  (3)  B-spline expanded torque (3 directions)           #
    #    tgt_pos    (3)  target position  broadcast to all elements        #
    #    tgt_quat   (4)  target quaternion broadcast to all elements       #
    #    s_norm     (1)  normalised arc-length                             #
    #    delta_v    (3)  temporal velocity change  v(t)-v(t-1)             #
    #    delta_omega(3)  temporal omega change  omega(t)-omega(t-1)        #
    #  Total = 3+3+3+4+3+3+4+1+3+3 = 30                                  #
    #                                                                      #
    #  Output channels (6, per element):                                   #
    #    ab_v    (3)  nonlinear acceleration for v                         #
    #    ab_w    (3)  nonlinear acceleration for omega                     #
    # ------------------------------------------------------------------ #
    fno_in_channels: int = 30
    fno_out_channels: int = 6
    fno_modes: int = 16
    fno_hidden_channels: int = 128
    fno_n_layers: int = 6
    fno_dropout: float = 0.05

    # ------------------------------------------------------------------ #
    #  Data                                                                #
    # ------------------------------------------------------------------ #
    data_path: str = "data_case2/case2_new_500.npz"
    train_ratio: float = 0.8

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    n_epochs: int = 100

    # LR scheduler
    lr_schedule: str = "cosine"
    lr_milestones: tuple = (40, 70, 90)
    lr_gamma: float = 0.3

    # Gradient clipping
    grad_clip: float = 1.0

    # ------------------------------------------------------------------ #
    #  Multi-step rollout loss                                             #
    # ------------------------------------------------------------------ #
    use_rollout_loss: bool = True
    rollout_k: int = 16
    rollout_weight: float = 0.5
    rollout_gamma: float = 0.9
    rollout_start_epoch: int = 10

    # ------------------------------------------------------------------ #
    #  Normalisation                                                       #
    # ------------------------------------------------------------------ #
    normalize: bool = True
    norm_eps: float = 1e-6

    # ------------------------------------------------------------------ #
    #  Checkpointing & logging                                             #
    # ------------------------------------------------------------------ #
    checkpoint_dir: str = "checkpoints_dynamics"
    save_every: int = 10
    log_every: int = 1

    # ------------------------------------------------------------------ #
    #  Device                                                              #
    # ------------------------------------------------------------------ #
    device: str = "auto"

    # ------------------------------------------------------------------ #
    #  Derived properties                                                  #
    # ------------------------------------------------------------------ #
    @property
    def ds(self) -> float:
        """Spatial step size [m]."""
        return self.base_length / self.n_elem

    @property
    def effective_dt(self) -> float:
        """Time step after temporal subsampling [s]."""
        return self.dt * self.subsample_factor

    @property
    def gamma_v(self) -> float:
        """Exponential damping factor for v (not used in IMEX dynamics, kept for analysis)."""
        return math.exp(-self.beta_v * self.effective_dt)

    @property
    def gamma_w(self) -> float:
        """Exponential damping factor for omega (not used in IMEX dynamics, kept for analysis)."""
        return math.exp(-self.beta_omega * self.effective_dt)

    @property
    def resolved_device(self) -> str:
        import torch
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  DynamicsConfig",
            "=" * 60,
            f"  Rod:      n_elem={self.n_elem}, L={self.base_length} m, dt={self.dt:.4e} s"
            + (f" (subsample={self.subsample_factor}, dt_eff={self.effective_dt:.4e})"
               if self.subsample_factor > 1 else ""),
            f"  L-operator: alpha_v={self.alpha_v}, alpha_w={self.alpha_omega}",
            f"              beta_v={self.beta_v}, beta_w={self.beta_omega}",
            f"  FNO:      in={self.fno_in_channels}, out={self.fno_out_channels}",
            f"            modes={self.fno_modes}, hidden={self.fno_hidden_channels}",
            f"            layers={self.fno_n_layers}, dropout={self.fno_dropout}",
            f"  Training: epochs={self.n_epochs}, batch={self.batch_size}",
            f"            lr={self.learning_rate}, rollout_k={self.rollout_k}",
            f"            rollout_weight={self.rollout_weight}, start_epoch={self.rollout_start_epoch}",
            f"  Device:   {self.resolved_device}",
            "=" * 60,
        ]
        return "\n".join(lines)
