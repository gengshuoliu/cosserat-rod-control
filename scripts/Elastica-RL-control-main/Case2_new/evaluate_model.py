"""
Evaluate trained Cosserat rod dynamics surrogate model.

Usage (Colab):
    !python evaluate_model.py --ckpt checkpoints_dynamics/ckpt_best.pt

Outputs (eval_results/):
    - single_step_r2.png            per-channel R2
    - mpc_horizon_error.png         error vs prediction horizon (core plot)
    - rollout_trajectory.png        one episode: pred vs GT tip velocity
    - spatial_field.png             single-step velocity field along rod
    - tip_trajectory_3d.png         tip 3D path: GT vs Pred
    - error_heatmap.png             per-channel error over time
    - eval_metrics.json             all numbers
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dynamics_learning.config import DynamicsConfig
from dynamics_learning.operators import DiscreteOperators
from dynamics_learning.surrogate import CosseratSurrogate
from dynamics_learning.dataset import CosseratDataset


# ========================================================================== #
#  Load model & data                                                         #
# ========================================================================== #

def load_model(ckpt_path, device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = DynamicsConfig(**{
        k: v for k, v in ckpt["cfg"].items()
        if k in DynamicsConfig.__dataclass_fields__
    })

    ops = DiscreteOperators(
        n_elem=cfg.n_elem, ds=cfg.ds, dt=cfg.effective_dt,
        alpha_v=cfg.alpha_v, beta_v=cfg.beta_v,
        alpha_omega=cfg.alpha_omega, beta_omega=cfg.beta_omega,
    )
    ops.to_torch(device)

    norm_stats = ckpt["norm_stats"]
    surrogate = CosseratSurrogate(cfg, ops, norm_stats)
    surrogate.load_state_dict(ckpt["model_state"])
    surrogate.to(device)
    surrogate.eval()

    print(f"[Eval] Loaded: epoch {ckpt['epoch']}, best_val={ckpt.get('best_val', '?')}")
    return surrogate, ops, norm_stats, cfg, device


# ========================================================================== #
#  Core: rollout H steps from GT state at (ep, t_start)                      #
# ========================================================================== #

@torch.no_grad()
def rollout_from(surrogate, val_ds, ep, t_start, H, device):
    """Roll out H steps from GT. Returns tip velocity errors (H,)."""
    T = val_ds.T
    ns = val_ds.norm_stats

    v_elem = torch.from_numpy(val_ds.v_elem[ep, t_start]).unsqueeze(0).to(device)
    omega = torch.from_numpy(val_ds.omega[ep, t_start]).unsqueeze(0).to(device)
    v_prev = v_elem.clone()
    w_prev = omega.clone()

    if ns is not None:
        mv  = torch.from_numpy(ns["v_elem"][0]).to(device).unsqueeze(0).unsqueeze(-1)
        sv  = torch.from_numpy(ns["v_elem"][1]).to(device).unsqueeze(0).unsqueeze(-1)
        mo  = torch.from_numpy(ns["omega"][0]).to(device).unsqueeze(0).unsqueeze(-1)
        so  = torch.from_numpy(ns["omega"][1]).to(device).unsqueeze(0).unsqueeze(-1)
        mdv = torch.from_numpy(ns["delta_v"][0]).to(device).unsqueeze(0).unsqueeze(-1)
        sdv = torch.from_numpy(ns["delta_v"][1]).to(device).unsqueeze(0).unsqueeze(-1)
        mdw = torch.from_numpy(ns["delta_omega"][0]).to(device).unsqueeze(0).unsqueeze(-1)
        sdw = torch.from_numpy(ns["delta_omega"][1]).to(device).unsqueeze(0).unsqueeze(-1)

    errs, v_preds, v_gts, w_preds, w_gts = [], [], [], [], []

    for k in range(H):
        t = t_start + k
        if t >= T:
            break

        fno_input = val_ds._build_input(ep, t).unsqueeze(0).to(device)
        if ns is not None:
            fno_input[:, 0:3, :]   = (v_elem - mv) / sv
            fno_input[:, 3:6, :]   = (omega - mo) / so
            fno_input[:, 24:27, :] = ((v_elem - v_prev) - mdv) / sdv
            fno_input[:, 27:30, :] = ((omega - w_prev) - mdw) / sdw

        pred_norm = surrogate(fno_input)
        ab_v, ab_w = surrogate._denorm_prediction(pred_norm, ns)

        v_prev, w_prev = v_elem.clone(), omega.clone()
        v_elem, omega = surrogate.dynamics_step(v_elem, omega, ab_v, ab_w)

        vp = v_elem[0, :, -1].cpu().numpy()
        vg = val_ds.v_elem[ep, t + 1, :, -1]
        wp = omega[0, :, -1].cpu().numpy()
        wg = val_ds.omega[ep, t + 1, :, -1]
        v_preds.append(vp.copy())
        v_gts.append(vg.copy())
        w_preds.append(wp.copy())
        w_gts.append(wg.copy())
        errs.append(np.linalg.norm(vp - vg))

    return (np.array(errs),
            np.array(v_preds), np.array(v_gts),
            np.array(w_preds), np.array(w_gts))


# ========================================================================== #
#  1. Single-step R2                                                         #
# ========================================================================== #

@torch.no_grad()
def evaluate_single_step(surrogate, val_ds, device, n_samples=5000):
    n = min(n_samples, len(val_ds))
    indices = np.random.RandomState(0).choice(len(val_ds), n, replace=False)

    all_pred, all_true = [], []
    for idx in indices:
        sample = val_ds[idx]
        pred = surrogate(sample["input"].unsqueeze(0).to(device)).squeeze(0).cpu()
        all_pred.append(pred)
        all_true.append(sample["target"])

    pred = torch.stack(all_pred)
    true = torch.stack(all_true)

    names = ["v_x", "v_y", "v_z", "w_x", "w_y", "w_z"]
    results = {}
    for c in range(6):
        p, t = pred[:, c, :].flatten(), true[:, c, :].flatten()
        ss_res = ((p - t) ** 2).sum().item()
        ss_tot = ((t - t.mean()) ** 2).sum().item()
        results[names[c]] = 1 - ss_res / max(ss_tot, 1e-10)

    return results


# ========================================================================== #
#  2. MPC horizon: sliding-window rollout                                    #
# ========================================================================== #

@torch.no_grad()
def evaluate_mpc_horizon(surrogate, val_ds, device, n_episodes=20, max_H=20):
    T = val_ds.T
    n_ep = min(n_episodes, val_ds.n_split)
    rng = np.random.RandomState(42)
    eps = rng.choice(val_ds.n_split, n_ep, replace=False)

    all_errs = []
    for ep in eps:
        max_start = T - max_H
        if max_start <= 0:
            continue
        starts = rng.choice(max_start, min(15, max_start), replace=False)
        for t0 in starts:
            e, _, _, _, _ = rollout_from(surrogate, val_ds, ep, t0, max_H, device)
            if len(e) == max_H:
                all_errs.append(e)

    arr = np.array(all_errs)
    return arr.mean(axis=0), arr.std(axis=0), len(all_errs)


# ========================================================================== #
#  Plots                                                                     #
# ========================================================================== #

def plot_r2(r2_dict, save_dir):
    names = list(r2_dict.keys())
    vals = list(r2_dict.values())
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, vals, color=colors)
    ax.axhline(y=0.8, color="green", linestyle="--", alpha=0.6, label="MPC target (0.8)")
    ax.set_ylabel("R2")
    ax.set_title(f"Single-Step R2   (avg = {np.mean(vals):.3f})")
    ax.set_ylim(min(0, min(vals) - 0.1), 1.05)
    ax.legend()
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "single_step_r2.png"), dpi=150)
    plt.close()


def plot_horizon(mean_err, std_err, mean_speed, save_dir):
    H = len(mean_err)
    x = np.arange(1, H + 1)
    rel = mean_err / mean_speed * 100

    fig, ax = plt.subplots(figsize=(10, 5))

    # Color zones
    ax.axhspan(0, 15, color="green", alpha=0.08)
    ax.axhspan(15, 30, color="yellow", alpha=0.08)
    ax.axhspan(30, 50, color="orange", alpha=0.08)
    ax.axhspan(50, max(100, rel.max() * 1.1), color="red", alpha=0.05)

    ax.plot(x, rel, "o-", color="#333", linewidth=2, markersize=5, zorder=5)
    ax.fill_between(x,
                    (mean_err - std_err) / mean_speed * 100,
                    (mean_err + std_err) / mean_speed * 100,
                    alpha=0.15, color="#333")

    # Find usable horizon
    usable = H
    for i, r in enumerate(rel):
        if r > 30:
            usable = i
            break
    if usable < H:
        ax.axvline(x=usable + 0.5, color="red", linestyle=":", linewidth=2)
        ax.text(usable + 0.7, rel.max() * 0.9, f"Usable: {usable} steps",
                fontsize=11, color="red", fontweight="bold")

    # Annotate each point
    for i in range(H):
        ax.annotate(f"{rel[i]:.0f}%", xy=(x[i], rel[i]),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8)

    ax.set_xlabel("Prediction Horizon (steps)", fontsize=12)
    ax.set_ylabel("Relative Error (%)", fontsize=12)
    ax.set_title("MPC Horizon Error  (re-init from GT each window)", fontsize=13)
    ax.set_xlim(0.5, H + 0.5)
    ax.set_ylim(0, max(60, rel.max() * 1.15))
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)

    # Legend for zones
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="green", alpha=0.15, label="<15% Excellent"),
        Patch(color="yellow", alpha=0.15, label="15-30% Good"),
        Patch(color="orange", alpha=0.15, label="30-50% Marginal"),
    ], loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mpc_horizon_error.png"), dpi=150)
    plt.close()
    return usable


def plot_trajectory(surrogate, val_ds, ep, device, save_dir, mpc_horizon=15):
    """One episode: pred vs GT tip velocity and angular velocity."""
    errs, v_preds, v_gts, w_preds, w_gts = rollout_from(
        surrogate, val_ds, ep, 0, val_ds.T, device)
    T = len(v_gts)
    t = np.arange(1, T + 1)

    fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
    v_labels = ["v_x (m/s)", "v_y (m/s)", "v_z (m/s)"]
    w_labels = ["w_x (rad/s)", "w_y (rad/s)", "w_z (rad/s)"]

    for c in range(3):
        axes[c].plot(t, v_gts[:, c], color="#2196F3", linewidth=1.5, label="GT")
        axes[c].plot(t, v_preds[:, c], color="#F44336", linewidth=1.5, alpha=0.8, label="Pred")
        axes[c].set_ylabel(v_labels[c])
        axes[c].legend(loc="upper right")
        axes[c].grid(True, alpha=0.3)
        # MPC usable zone
        axes[c].axvspan(0, mpc_horizon, color="green", alpha=0.06)

    for c in range(3):
        axes[3+c].plot(t, w_gts[:, c], color="#2196F3", linewidth=1.5, label="GT")
        axes[3+c].plot(t, w_preds[:, c], color="#F44336", linewidth=1.5, alpha=0.8, label="Pred")
        axes[3+c].set_ylabel(w_labels[c])
        axes[3+c].legend(loc="upper right")
        axes[3+c].grid(True, alpha=0.3)
        axes[3+c].axvspan(0, mpc_horizon, color="green", alpha=0.06)

    # MPC boundary line on all axes
    for ax in axes:
        ax.axvline(x=mpc_horizon, color="green", linestyle="--", linewidth=1.5, alpha=0.7)
    axes[0].text(mpc_horizon + 1, axes[0].get_ylim()[1] * 0.85,
                 f"MPC horizon ({mpc_horizon} steps)",
                 fontsize=10, color="green", fontweight="bold")

    axes[-1].set_xlabel("Time Step")
    axes[0].set_title(f"Open-Loop Rollout: Tip Velocity (Episode {ep})")
    axes[3].set_title(f"Open-Loop Rollout: Tip Angular Velocity (Episode {ep})")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "rollout_trajectory.png"), dpi=150)
    plt.close()



def plot_spatial_field(surrogate, val_ds, ep, device, save_dir):
    """
    Velocity & angular velocity spatial distribution along the rod
    at selected time steps. Uses single-step prediction (no accumulation).
    """
    ns = val_ds.norm_stats
    n_elem = val_ds.n_elem
    s = np.linspace(0, 1, n_elem)  # normalised arc-length

    frames = [0, 5, 10, 15]  # MPC-relevant time steps
    fig, axes = plt.subplots(3, len(frames), figsize=(16, 9), sharex=True)

    v_labels = ["v_x (m/s)", "v_y (m/s)", "v_z (m/s)"]

    for j, t in enumerate(frames):
        if t >= val_ds.T:
            continue

        # Single-step: GT input → FNO → predict → IMEX → compare with GT next
        fno_input = val_ds._build_input(ep, t).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_norm = surrogate(fno_input)
            ab_v, ab_w = surrogate._denorm_prediction(pred_norm, ns)

        v_curr = torch.from_numpy(val_ds.v_elem[ep, t]).unsqueeze(0).to(device)
        w_curr = torch.from_numpy(val_ds.omega[ep, t]).unsqueeze(0).to(device)
        v_pred, w_pred = surrogate.dynamics_step(v_curr, w_curr, ab_v, ab_w)

        v_pred_np = v_pred[0].cpu().numpy()       # (3, n_elem)
        v_gt_np   = val_ds.v_elem[ep, t + 1]      # (3, n_elem)

        for c in range(3):
            axes[c, j].plot(s, v_gt_np[c], color='#2196F3', linewidth=2, label='GT')
            axes[c, j].plot(s, v_pred_np[c], color='#F44336', linewidth=2,
                           alpha=0.8, linestyle='--', label='Pred')
            axes[c, j].grid(True, alpha=0.3)
            if j == 0:
                axes[c, j].set_ylabel(v_labels[c], fontsize=11)
            if c == 0:
                axes[c, j].set_title(f't = {t} → {t+1}', fontsize=12, fontweight='bold')
                if j == 0:
                    axes[c, j].legend(fontsize=9)

    for j in range(len(frames)):
        axes[-1, j].set_xlabel('Arc-length s/L', fontsize=10)

    fig.suptitle(f'Single-Step Velocity Field Along Rod (Episode {ep})',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "spatial_field.png"), dpi=150, bbox_inches='tight')
    plt.close()


# ========================================================================== #
#  Plot: Tip 3D trajectory                                                    #
# ========================================================================== #

@torch.no_grad()
def plot_tip_3d(surrogate, val_ds, ep, device, save_dir, H=15):
    """
    Tip 3D trajectory for H steps using autoregressive rollout (velocity only).
    Position is reconstructed step-by-step: r_{n+1} = r_n + dt * v_{n+1}_tip_node.
    Uses the same rollout logic as rollout_from() to stay consistent.
    """
    from dynamics_learning.operators import DiscreteOperators

    T = val_ds.T
    ns = val_ds.norm_stats
    dt = surrogate.dt
    H = min(H, T)

    v_elem = torch.from_numpy(val_ds.v_elem[ep, 0]).unsqueeze(0).to(device)
    omega  = torch.from_numpy(val_ds.omega[ep, 0]).unsqueeze(0).to(device)
    v_prev = v_elem.clone()
    w_prev = omega.clone()

    if ns is not None:
        mv  = torch.from_numpy(ns["v_elem"][0]).to(device).unsqueeze(0).unsqueeze(-1)
        sv  = torch.from_numpy(ns["v_elem"][1]).to(device).unsqueeze(0).unsqueeze(-1)
        mo  = torch.from_numpy(ns["omega"][0]).to(device).unsqueeze(0).unsqueeze(-1)
        so  = torch.from_numpy(ns["omega"][1]).to(device).unsqueeze(0).unsqueeze(-1)
        mdv = torch.from_numpy(ns["delta_v"][0]).to(device).unsqueeze(0).unsqueeze(-1)
        sdv = torch.from_numpy(ns["delta_v"][1]).to(device).unsqueeze(0).unsqueeze(-1)
        mdw = torch.from_numpy(ns["delta_omega"][0]).to(device).unsqueeze(0).unsqueeze(-1)
        sdw = torch.from_numpy(ns["delta_omega"][1]).to(device).unsqueeze(0).unsqueeze(-1)

    # GT tip position trajectory (from dataset)
    gt_tips = [val_ds.r_elem[ep, t, :, -1].copy() for t in range(H + 1)]

    # Pred: integrate tip position from predicted velocity
    # Start from GT position at t=0
    pred_tip = val_ds.r_elem[ep, 0, :, -1].copy()
    pred_tips = [pred_tip.copy()]

    for k in range(H):
        t = k
        if t >= T:
            break

        fno_input = val_ds._build_input(ep, t).unsqueeze(0).to(device)
        if ns is not None:
            fno_input[:, 0:3, :]   = (v_elem - mv) / sv
            fno_input[:, 3:6, :]   = (omega - mo) / so
            fno_input[:, 24:27, :] = ((v_elem - v_prev) - mdv) / sdv
            fno_input[:, 27:30, :] = ((omega - w_prev) - mdw) / sdw

        pred_norm = surrogate(fno_input)
        ab_v, ab_w = surrogate._denorm_prediction(pred_norm, ns)

        v_prev, w_prev = v_elem.clone(), omega.clone()
        v_elem, omega = surrogate.dynamics_step(v_elem, omega, ab_v, ab_w)

        # Tip velocity: elem_to_node, take last node
        v_tip_node = v_elem[0, :, -1].cpu().numpy()  # approximate: last element ≈ last node
        pred_tip = pred_tip + dt * v_tip_node
        pred_tips.append(pred_tip.copy())

    gt_tips = np.array(gt_tips)
    pred_tips = np.array(pred_tips)

    # --- Plot ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(gt_tips[:, 0], gt_tips[:, 1], gt_tips[:, 2],
            'o-', color='#2196F3', linewidth=2.5, markersize=5, label='GT')
    ax.plot(pred_tips[:, 0], pred_tips[:, 1], pred_tips[:, 2],
            's-', color='#F44336', linewidth=2.5, markersize=5, alpha=0.85, label='Pred')

    ax.scatter(*gt_tips[0], color='green', s=150, marker='o', edgecolors='k',
               zorder=5, label='Start (t=0)')
    ax.scatter(*gt_tips[-1], color='#2196F3', s=120, marker='^', edgecolors='k',
               zorder=5, label=f'End GT (t={H})')
    ax.scatter(*pred_tips[-1], color='#F44336', s=120, marker='^', edgecolors='k',
               zorder=5, label=f'End Pred (t={H})')

    for i in range(0, len(gt_tips), 5):
        ax.text(gt_tips[i, 0], gt_tips[i, 1], gt_tips[i, 2],
                f' t={i}', fontsize=8, color='#555')

    ax.set_xlabel('x (m)', fontsize=11)
    ax.set_ylabel('y (m)', fontsize=11)
    ax.set_zlabel('z (m)', fontsize=11)
    ax.set_title(f'Tip 3D Trajectory — MPC Window ({H} steps, Episode {ep})',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "tip_trajectory_3d.png"), dpi=150, bbox_inches='tight')
    plt.close()


# ========================================================================== #
#  Plot: Per-channel error heatmap                                            #
# ========================================================================== #

def plot_error_heatmap(surrogate, val_ds, ep, device, save_dir):
    """Heatmap: per-channel absolute error over time."""
    errs, v_preds, v_gts, w_preds, w_gts = rollout_from(
        surrogate, val_ds, ep, 0, val_ds.T, device)
    T = len(v_gts)

    # Build error matrix (T, 6)
    v_err = np.abs(np.array(v_preds) - np.array(v_gts))   # (T, 3)
    w_err = np.abs(np.array(w_preds) - np.array(w_gts))   # (T, 3)
    err_matrix = np.concatenate([v_err, w_err], axis=1).T  # (6, T)

    labels = ["v_x", "v_y", "v_z", "w_x", "w_y", "w_z"]

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(err_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_yticks(range(6))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_title(f"Per-Channel Absolute Error (Episode {ep})", fontsize=13)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Absolute Error", fontsize=10)

    # Mark the MPC usable zone
    ax.axvline(x=10, color='white', linestyle='--', linewidth=2, alpha=0.8)
    ax.text(11, -0.7, "MPC horizon", color='white', fontsize=9,
            fontweight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "error_heatmap.png"), dpi=150, bbox_inches='tight')
    plt.close()


# ========================================================================== #
#  Main                                                                      #
# ========================================================================== #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints_dynamics/ckpt_best.pt")
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--max_horizon", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="eval_results")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    surrogate, ops, norm_stats, cfg, device = load_model(args.ckpt)
    val_ds = CosseratDataset(cfg, ops, mode="single", split="val")
    val_ds.set_norm_stats(norm_stats)

    mean_speed = np.linalg.norm(val_ds.v_elem[:, :, :, -1], axis=-1).mean()

    # --- 1. Single-step R2 ---
    print("\n[1] Single-Step R2")
    r2 = evaluate_single_step(surrogate, val_ds, device)
    avg_r2 = np.mean(list(r2.values()))
    for k, v in r2.items():
        print(f"  {k}: {v:.4f}")
    print(f"  avg: {avg_r2:.4f}")
    plot_r2(r2, args.save_dir)

    # --- 2. MPC horizon ---
    print("\n[2] MPC Horizon (sliding-window)")
    mpc_mean, mpc_std, nw = evaluate_mpc_horizon(
        surrogate, val_ds, device, args.n_episodes, args.max_horizon)
    print(f"  {nw} windows evaluated")
    for h in [1, 5, 10, 15, 20]:
        if h <= len(mpc_mean):
            r = mpc_mean[h-1] / mean_speed * 100
            print(f"  H={h:>2}: {r:>5.1f}%")
    usable = plot_horizon(mpc_mean, mpc_std, mean_speed, args.save_dir)

    # --- 3. One trajectory ---
    print("\n[3] Trajectory plot")
    rng = np.random.RandomState(42)
    ep = rng.choice(val_ds.n_split)
    plot_trajectory(surrogate, val_ds, ep, device, args.save_dir)

    # --- 4. Spatial velocity field ---
    print("\n[4] Spatial velocity field along rod")
    plot_spatial_field(surrogate, val_ds, ep, device, args.save_dir)

    # --- 5. Error heatmap ---
    print("\n[5] Error heatmap")
    plot_error_heatmap(surrogate, val_ds, ep, device, args.save_dir)

    # --- Summary ---
    summary = {
        "r2": r2, "avg_r2": float(avg_r2),
        "mean_tip_speed": float(mean_speed),
        "mpc_usable_horizon": int(usable),
        "mpc_horizon": {f"H{h}": float(mpc_mean[h-1] / mean_speed * 100)
                        for h in range(1, len(mpc_mean)+1)},
    }
    with open(os.path.join(args.save_dir, "eval_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*40}")
    print(f"  avg R2:          {avg_r2:.3f}  {'OK' if avg_r2>0.8 else 'NEED >0.8'}")
    print(f"  MPC usable:      {usable} steps  {'OK' if usable>=10 else 'NEED >=10'}")
    print(f"  Results:         {args.save_dir}/")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
