"""
CosseratDataset: loads the PyElastica .npz dataset and preprocesses it for
training the FNO-based dynamics surrogate.

Key design choices:
  - IMEX targets on element grid: target = (A_v_elem @ v_next - v_curr) / dt
    preserving the full dissipative operator L = α∂_ss − βI.
  - All dynamics on element grid (40 pts), eliminating node-element
    interpolation errors that accumulate in rollout.
  - Temporal context: delta_v and delta_omega channels in input (30 total).

Two dataset modes:
  - "single" : returns one (state, action, target) transition at a time
  - "rollout": returns K+1 consecutive steps for multi-step rollout loss
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .operators import DiscreteOperators
from .bspline_utils import BSplineExpander
from .config import DynamicsConfig


# --------------------------------------------------------------------------- #
#  Quaternion utilities (numpy)                                                #
# --------------------------------------------------------------------------- #

def rotmat_to_quat_np(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix (or batch) to unit quaternion [qw, qx, qy, qz].

    Parameters
    ----------
    R : (..., 3, 3)

    Returns
    -------
    q : (..., 4)
    """
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    qw = 0.5 * np.sqrt(np.maximum(1.0 + tr, 1e-10))
    denom = np.maximum(4.0 * qw, 1e-10)
    qx = (R[..., 2, 1] - R[..., 1, 2]) / denom
    qy = (R[..., 0, 2] - R[..., 2, 0]) / denom
    qz = (R[..., 1, 0] - R[..., 0, 1]) / denom
    return np.stack([qw, qx, qy, qz], axis=-1)


# --------------------------------------------------------------------------- #
#  Main Dataset class                                                           #
# --------------------------------------------------------------------------- #

class CosseratDataset(Dataset):
    """
    Dataset for Cosserat rod dynamics learning.

    Parameters
    ----------
    cfg         : DynamicsConfig
    ops         : DiscreteOperators  (kept for backward compat; static methods used)
    mode        : "single" | "rollout"
    split       : "train" | "val"
    rollout_k   : int  — number of steps in rollout window (mode="rollout")
    """

    def __init__(self,
                 cfg: DynamicsConfig,
                 ops: DiscreteOperators,
                 mode: str = "single",
                 split: str = "train",
                 rollout_k: int = 5):

        super().__init__()
        self.cfg = cfg
        self.ops = ops
        self.mode = mode
        self.split = split
        self.rollout_k = rollout_k

        assert mode in ("single", "rollout"), f"Unknown mode: {mode}"
        assert split in ("train", "val"),     f"Unknown split: {split}"

        # ------------------------------------------------------------------ #
        #  1. Load raw .npz                                                    #
        # ------------------------------------------------------------------ #
        print(f"[Dataset] Loading {cfg.data_path} ...")
        raw = np.load(cfg.data_path, allow_pickle=True)

        meta = json.loads(str(raw["meta_json"]))
        self._T     = meta["horizon_steps"]

        rod_pos    = raw["rod_pos"].astype(np.float32)
        rod_vel    = raw["rod_vel"].astype(np.float32)
        rod_dir    = raw["rod_dir"].astype(np.float32)
        rod_omega  = raw["rod_omega"].astype(np.float32)
        actions    = raw["action"].astype(np.float32)
        target_pos = raw["target_pos"].astype(np.float32)
        target_dir = raw["target_dir"].astype(np.float32)
        valid_mask = raw["valid_mask"]

        # ------------------------------------------------------------------ #
        #  Temporal subsampling (e.g. factor=5 keeps every 5th step)         #
        # ------------------------------------------------------------------ #
        s = cfg.subsample_factor
        if s > 1:
            print(f"[Dataset] Temporal subsampling factor={s} "
                  f"(T: {actions.shape[1]} -> {actions.shape[1]//s})")
            rod_pos    = rod_pos[:, ::s]
            rod_vel    = rod_vel[:, ::s]
            rod_dir    = rod_dir[:, ::s]
            rod_omega  = rod_omega[:, ::s]
            actions    = actions[:, ::s]
            target_pos = target_pos[:, ::s]
            target_dir = target_dir[:, ::s]
            valid_mask = valid_mask[:, ::s]

        n_ep, Tp1, _, n_nodes = rod_vel.shape
        n_elem = n_nodes - 1
        self._T = Tp1 - 1   # number of transitions
        T = self._T

        # ------------------------------------------------------------------ #
        #  2. Quaternion from rod_dir                                          #
        # ------------------------------------------------------------------ #
        print("[Dataset] Computing quaternions from director matrices ...")
        R = rod_dir.transpose(0, 1, 4, 2, 3)           # (N_ep, T+1, n_elem, 3, 3)
        Q_quat_raw = rotmat_to_quat_np(R)               # (N_ep, T+1, n_elem, 4)
        Q_quat = Q_quat_raw.transpose(0, 1, 3, 2).astype(np.float32)  # (N_ep,T+1,4,n_elem)

        R_tgt2 = target_dir[:, :, :, :, 0]              # (N_ep, T+1, 3, 3)
        tgt_quat = rotmat_to_quat_np(R_tgt2)            # (N_ep, T+1, 4)

        # ------------------------------------------------------------------ #
        #  3. r_elem from rod_pos (node -> element interpolation)              #
        # ------------------------------------------------------------------ #
        r_elem = 0.5 * (rod_pos[:, :, :, :-1] + rod_pos[:, :, :, 1:])

        # ------------------------------------------------------------------ #
        #  4. v_elem from rod_vel (node -> element interpolation)              #
        # ------------------------------------------------------------------ #
        v_elem = 0.5 * (rod_vel[:, :, :, :-1] + rod_vel[:, :, :, 1:])

        # ------------------------------------------------------------------ #
        #  5. B-spline action expansion                                        #
        # ------------------------------------------------------------------ #
        print("[Dataset] Expanding actions via B-spline ...")
        expander = BSplineExpander(n_ctrl=cfg.n_ctrl,
                                   base_length=cfg.base_length,
                                   n_elem=n_elem)
        a_spatial = expander.expand(
            actions,
            alpha_scale=cfg.alpha_scale,
            beta_scale=cfg.beta_scale,
        ).astype(np.float32)    # (N_ep, T, 3, n_elem)

        # ------------------------------------------------------------------ #
        #  6. Compute IMEX training targets on element grid                    #
        #     tgt_v = (A_v_elem @ v_elem_next  -  v_elem_curr) / dt           #
        #     tgt_w = (A_omega  @ omega_next   -  omega_curr)  / dt           #
        # ------------------------------------------------------------------ #
        print("[Dataset] Computing IMEX targets (element grid) ...")
        v_elem_curr = v_elem[:, :-1, :, :]     # (N_ep, T, 3, n_elem)
        v_elem_next = v_elem[:, 1:,  :, :]     # (N_ep, T, 3, n_elem)
        w_curr      = rod_omega[:, :-1, :, :]   # (N_ep, T, 3, n_elem)
        w_next      = rod_omega[:, 1:,  :, :]   # (N_ep, T, 3, n_elem)

        tgt_v_elem = ops.compute_target_v_elem_direct(v_elem_curr, v_elem_next).astype(np.float32)
        tgt_omega  = ops.compute_target_omega(w_curr, w_next).astype(np.float32)

        # ------------------------------------------------------------------ #
        #  6b. Compute temporal deltas for FNO input context                   #
        #      delta_v[t] = v_elem[t] - v_elem[t-1],  delta_v[0] = 0         #
        # ------------------------------------------------------------------ #
        print("[Dataset] Computing temporal deltas ...")
        delta_v = np.zeros((n_ep, T, 3, n_elem), dtype=np.float32)
        delta_v[:, 1:, :, :] = v_elem[:, 1:T, :, :] - v_elem[:, 0:T-1, :, :]

        delta_omega = np.zeros((n_ep, T, 3, n_elem), dtype=np.float32)
        delta_omega[:, 1:, :, :] = rod_omega[:, 1:T, :, :] - rod_omega[:, 0:T-1, :, :]

        # ------------------------------------------------------------------ #
        #  7. Episode split  (train / val by episode index, shuffled)         #
        # ------------------------------------------------------------------ #
        n_train = int(n_ep * cfg.train_ratio)
        rng = np.random.RandomState(42)
        perm = rng.permutation(n_ep)
        if split == "train":
            ep_idx = perm[:n_train]
        else:
            ep_idx = perm[n_train:]

        # Store arrays indexed by selected episodes
        self.v_node     = rod_vel[ep_idx]       # (n_split, T+1, 3, n_nodes)  kept for compat
        self.v_elem     = v_elem[ep_idx]        # (n_split, T+1, 3, n_elem)
        self.omega      = rod_omega[ep_idx]     # (n_split, T+1, 3, n_elem)
        self.r_elem     = r_elem[ep_idx]        # (n_split, T+1, 3, n_elem)
        self.Q_quat     = Q_quat[ep_idx]        # (n_split, T+1, 4, n_elem)
        self.a_spatial  = a_spatial[ep_idx]      # (n_split, T,   3, n_elem)
        self.tgt_pos    = target_pos[ep_idx, :, :, 0]  # (n_split, T+1, 3)
        self.tgt_quat   = tgt_quat[ep_idx]      # (n_split, T+1, 4)
        self.tgt_v_elem = tgt_v_elem[ep_idx]    # (n_split, T,   3, n_elem)
        self.tgt_omega  = tgt_omega[ep_idx]      # (n_split, T,   3, n_elem)
        self.delta_v    = delta_v[ep_idx]        # (n_split, T,   3, n_elem)
        self.delta_omega = delta_omega[ep_idx]   # (n_split, T,   3, n_elem)
        self.valid_mask = valid_mask[ep_idx]     # (n_split, T+1)

        self.n_split  = len(ep_idx)
        self.T        = T
        self.n_elem   = n_elem
        self.n_nodes  = n_nodes

        # Precompute arc-length coordinate
        s_arr = np.linspace(0.5 / n_elem, 1.0 - 0.5 / n_elem, n_elem, dtype=np.float32)
        self.s_norm = s_arr[None, :]   # (1, n_elem)

        # ------------------------------------------------------------------ #
        #  8. Normalisation statistics (only from training split)              #
        # ------------------------------------------------------------------ #
        self.norm_stats = None
        if cfg.normalize and split == "train":
            self.norm_stats = self._compute_norm_stats()

        print(f"[Dataset] Split={split}, episodes={self.n_split}, "
              f"transitions={self.n_split * self.T}")

    # ---------------------------------------------------------------------- #
    #  Normalisation                                                           #
    # ---------------------------------------------------------------------- #

    def _compute_norm_stats(self) -> dict:
        """Compute per-channel mean and std from all training samples."""
        print("[Dataset] Computing normalisation statistics ...")
        eps = self.cfg.norm_eps

        def stats_per_channel(arr):
            """Per-channel stats for arrays of shape (..., C, N_spatial)."""
            shape = arr.shape
            C = shape[-2]
            flat = arr.reshape(-1, C, shape[-1]).astype(np.float64)
            mean = flat.mean(axis=(0, 2))
            std  = flat.std(axis=(0, 2)) + eps
            return mean.astype(np.float32), std.astype(np.float32)

        def stats_per_channel_nospatial(arr):
            """Per-channel stats for arrays of shape (..., C) (no spatial dim)."""
            shape = arr.shape
            C = shape[-1]
            flat = arr.reshape(-1, C).astype(np.float64)
            mean = flat.mean(axis=0)
            std  = flat.std(axis=0) + eps
            return mean.astype(np.float32), std.astype(np.float32)

        s = {}
        s["v_elem"]      = stats_per_channel(self.v_elem)
        s["omega"]       = stats_per_channel(self.omega)
        s["r_elem"]      = stats_per_channel(self.r_elem)
        s["Q_quat"]      = stats_per_channel(self.Q_quat)
        s["a_spatial"]   = stats_per_channel(self.a_spatial)
        s["tgt_pos"]     = stats_per_channel_nospatial(self.tgt_pos)
        s["tgt_quat"]    = stats_per_channel_nospatial(self.tgt_quat)
        s["tgt_v_elem"]  = stats_per_channel(self.tgt_v_elem)
        s["tgt_omega"]   = stats_per_channel(self.tgt_omega)
        s["delta_v"]     = stats_per_channel(self.delta_v)
        s["delta_omega"] = stats_per_channel(self.delta_omega)
        return s

    def set_norm_stats(self, stats: dict):
        """Called by trainer to set val dataset stats from train dataset."""
        self.norm_stats = stats

    def _normalize(self, x: torch.Tensor, key: str) -> torch.Tensor:
        if self.norm_stats is None:
            return x
        mean, std = self.norm_stats[key]
        mean_t = torch.from_numpy(mean).to(x.device)
        std_t  = torch.from_numpy(std).to(x.device)
        if x.ndim == 2:  # (C, N_spatial)
            return (x - mean_t.unsqueeze(-1)) / std_t.unsqueeze(-1)
        else:
            return (x - mean_t) / std_t

    def _denormalize_target(self, x: torch.Tensor, key: str) -> torch.Tensor:
        if self.norm_stats is None:
            return x
        mean, std = self.norm_stats[key]
        mean_t = torch.from_numpy(mean).to(x.device)
        std_t  = torch.from_numpy(std).to(x.device)
        if x.ndim == 2:
            return x * std_t.unsqueeze(-1) + mean_t.unsqueeze(-1)
        else:
            return x * std_t + mean_t

    # ---------------------------------------------------------------------- #
    #  Build FNO input tensor (30 channels)                                    #
    # ---------------------------------------------------------------------- #

    def _build_input(self, ep: int, t: int) -> torch.Tensor:
        """
        Build the (30, n_elem) FNO input tensor for episode ep, timestep t.

        Channels:
            0:3   v_elem    (3, n_elem)
            3:6   omega     (3, n_elem)
            6:9   r_elem    (3, n_elem)
            9:13  Q_quat    (4, n_elem)
            13:16 a_spatial (3, n_elem)
            16:19 tgt_pos   (3, n_elem)  broadcast
            19:23 tgt_quat  (4, n_elem)  broadcast
            23:24 s_norm    (1, n_elem)
            24:27 delta_v   (3, n_elem)
            27:30 delta_omega (3, n_elem)
        """
        ne = self.n_elem

        v_e  = torch.from_numpy(self.v_elem[ep, t])     # (3, ne)
        om   = torch.from_numpy(self.omega[ep, t])       # (3, ne)
        r_e  = torch.from_numpy(self.r_elem[ep, t])      # (3, ne)
        Q_e  = torch.from_numpy(self.Q_quat[ep, t])      # (4, ne)
        a_sp = torch.from_numpy(self.a_spatial[ep, t])    # (3, ne)
        dv   = torch.from_numpy(self.delta_v[ep, t])      # (3, ne)
        dw   = torch.from_numpy(self.delta_omega[ep, t])  # (3, ne)

        tgt_pos = torch.from_numpy(self.tgt_pos[ep, t])   # (3,)
        tgt_q   = torch.from_numpy(self.tgt_quat[ep, t])  # (4,)
        tgt_pos_b = tgt_pos.unsqueeze(-1).expand(-1, ne)   # (3, ne)
        tgt_q_b   = tgt_q.unsqueeze(-1).expand(-1, ne)     # (4, ne)

        s_b = torch.from_numpy(self.s_norm)   # (1, ne)

        # Normalise each field
        if self.norm_stats is not None:
            v_e       = self._normalize(v_e,       "v_elem")
            om        = self._normalize(om,        "omega")
            r_e       = self._normalize(r_e,       "r_elem")
            Q_e       = self._normalize(Q_e,       "Q_quat")
            a_sp      = self._normalize(a_sp,      "a_spatial")
            tgt_pos_b = self._normalize(tgt_pos_b, "tgt_pos")
            tgt_q_b   = self._normalize(tgt_q_b,   "tgt_quat")
            dv        = self._normalize(dv,        "delta_v")
            dw        = self._normalize(dw,        "delta_omega")

        return torch.cat([v_e, om, r_e, Q_e, a_sp, tgt_pos_b, tgt_q_b, s_b, dv, dw], dim=0)

    # ---------------------------------------------------------------------- #
    #  Build target tensor for a single transition                            #
    # ---------------------------------------------------------------------- #

    def _build_target(self, ep: int, t: int) -> torch.Tensor:
        """
        Build the (6, n_elem) target tensor:
            0:3  target_v_elem   (damped residual for v)
            3:6  target_omega    (damped residual for omega)
        """
        tv = torch.from_numpy(self.tgt_v_elem[ep, t])  # (3, n_elem)
        tw = torch.from_numpy(self.tgt_omega[ep, t])    # (3, n_elem)

        if self.norm_stats is not None:
            tv = self._normalize(tv, "tgt_v_elem")
            tw = self._normalize(tw, "tgt_omega")

        return torch.cat([tv, tw], dim=0)

    # ---------------------------------------------------------------------- #
    #  Dataset interface                                                       #
    # ---------------------------------------------------------------------- #

    def __len__(self) -> int:
        if self.mode == "single":
            return self.n_split * self.T
        else:
            return self.n_split * (self.T - self.rollout_k)

    def __getitem__(self, idx: int):
        if self.mode == "single":
            return self._get_single(idx)
        else:
            return self._get_rollout(idx)

    def _get_single(self, idx: int):
        ep = idx // self.T
        t  = idx  % self.T

        fno_input = self._build_input(ep, t)
        target    = self._build_target(ep, t)

        v_elem_t = torch.from_numpy(self.v_elem[ep, t])  # (3, n_elem)
        omega_t  = torch.from_numpy(self.omega[ep, t])    # (3, n_elem)

        return {
            "input":  fno_input,    # (30, n_elem)
            "target": target,       # (6,  n_elem)
            "v_elem": v_elem_t,     # (3,  n_elem) — for dynamics step in rollout
            "omega":  omega_t,      # (3,  n_elem)
            "ep": ep,
            "t": t,
        }

    def _get_rollout(self, idx: int):
        K  = self.rollout_k
        ep = idx // (self.T - K)
        t0 = idx  % (self.T - K)

        inputs_list  = []
        targets_list = []
        v_elem_list  = []
        omega_list   = []

        for k in range(K + 1):
            t = t0 + k
            inputs_list.append(self._build_input(ep, t))
            v_elem_list.append(torch.from_numpy(self.v_elem[ep, t]))
            omega_list.append(torch.from_numpy(self.omega[ep, t]))
            if k < K:
                targets_list.append(self._build_target(ep, t))

        return {
            "inputs":  torch.stack(inputs_list),   # (K+1, 30, n_elem)
            "targets": torch.stack(targets_list),  # (K,   6,  n_elem)
            "v_elem":  torch.stack(v_elem_list),   # (K+1, 3,  n_elem)
            "omega":   torch.stack(omega_list),    # (K+1, 3,  n_elem)
            "ep": ep,
            "t0": t0,
        }


# --------------------------------------------------------------------------- #
#  DataLoader factory                                                          #
# --------------------------------------------------------------------------- #

def create_dataloaders(cfg: DynamicsConfig,
                       ops: DiscreteOperators,
                       rollout_k: int = 5,
                       num_workers: int = 0):
    """Create train and val DataLoaders for both modes."""
    train_single  = CosseratDataset(cfg, ops, mode="single",  split="train", rollout_k=rollout_k)
    val_single    = CosseratDataset(cfg, ops, mode="single",  split="val",   rollout_k=rollout_k)
    train_rollout = CosseratDataset(cfg, ops, mode="rollout", split="train", rollout_k=rollout_k)
    val_rollout   = CosseratDataset(cfg, ops, mode="rollout", split="val",   rollout_k=rollout_k)

    train_stats = train_single.norm_stats
    val_single.set_norm_stats(train_stats)
    val_rollout.set_norm_stats(train_stats)
    train_rollout.set_norm_stats(train_stats)

    kw = dict(num_workers=num_workers, pin_memory=True)
    loaders = {
        "single_train":  DataLoader(train_single,  batch_size=cfg.batch_size, shuffle=True,  **kw),
        "single_val":    DataLoader(val_single,    batch_size=cfg.batch_size, shuffle=False, **kw),
        "rollout_train": DataLoader(train_rollout, batch_size=max(16, cfg.batch_size // 8),
                                    shuffle=True, **kw),
        "rollout_val":   DataLoader(val_rollout,   batch_size=max(16, cfg.batch_size // 8),
                                    shuffle=False, **kw),
    }

    return loaders, train_stats
