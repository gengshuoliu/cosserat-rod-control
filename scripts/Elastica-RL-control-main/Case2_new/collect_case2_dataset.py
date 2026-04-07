import os
import json
import time
import argparse
import numpy as np

from set_environment import Environment


def snapshot_full_state(env):
    """Full-field snapshot from Elastica systems."""
    rod = env.shearable_rod
    sph = env.sphere
    return {
        "rod_pos": rod.position_collection.copy(),      # (3, n_nodes)
        "rod_vel": rod.velocity_collection.copy(),      # (3, n_nodes)
        "rod_dir": rod.director_collection.copy(),      # (3, 3, n_elems)
        "rod_omega": rod.omega_collection.copy(),       # (3, n_elems)
        "target_pos": sph.position_collection.copy(),   # (3, 1)
        "target_dir": sph.director_collection.copy() if hasattr(sph, "director_collection") else None,  # (3,3,1)
        "target_vel": sph.velocity_collection.copy() if hasattr(sph, "velocity_collection") else None,  # (3,1)
    }


def assert_shapes(x, n_elems: int):
    n_nodes = n_elems + 1
    assert x["rod_pos"].shape == (3, n_nodes), f"rod_pos {x['rod_pos'].shape} != (3,{n_nodes})"
    assert x["rod_vel"].shape == (3, n_nodes), f"rod_vel {x['rod_vel'].shape} != (3,{n_nodes})"
    assert x["rod_dir"].shape == (3, 3, n_elems), f"rod_dir {x['rod_dir'].shape} != (3,3,{n_elems})"
    assert x["rod_omega"].shape == (3, n_elems), f"rod_omega {x['rod_omega'].shape} != (3,{n_elems})"
    assert x["target_pos"].shape == (3, 1), f"target_pos {x['target_pos'].shape} != (3,1)"


def is_finite_dict(x: dict) -> bool:
    for v in x.values():
        if v is None:
            continue
        if not np.all(np.isfinite(v)):
            return False
    return True


def sample_action_random_walk(rng, prev_action, a_max, sigma):
    """a_t = clip(a_{t-1} + eps, -a_max, a_max), eps~N(0,sigma^2)"""
    eps = rng.normal(0.0, sigma, size=prev_action.shape)
    return np.clip(prev_action + eps, -a_max, a_max)


def collect_dataset(
    out_path: str,
    seed: int,
    n_episodes: int,
    horizon_steps: int,
    hold_steps: int,
    a_max: float,
    sigma: float,
    mode: int,
    dim: float,
    sim_dt: float,
    num_steps_per_update: int,
    n_elem: int,
    number_of_control_points: int,
    alpha: float,
    beta: float,
    boundary: tuple,
    save_float32: bool,
):
    rng = np.random.default_rng(seed)
    
    final_time = float(sim_dt) * float(num_steps_per_update) * float(horizon_steps + 1)



    env = Environment(
        final_time=final_time,
        num_steps_per_update=int(num_steps_per_update),
        number_of_control_points=int(number_of_control_points),
        alpha=float(alpha),
        beta=float(beta),
        target_position=np.array([0.0, 0.0, 0.0], dtype=np.float64),  
        COLLECT_DATA_FOR_POSTPROCESSING=False,
        sim_dt=float(sim_dt),
        n_elem=int(n_elem),
        mode=int(mode),
        dim=float(dim),
        boundary=np.array(boundary, dtype=np.float64),
    )

    # Infer shapes
    env.reset()
    x0 = snapshot_full_state(env)
    assert_shapes(x0, n_elems=n_elem)

    n_nodes = n_elem + 1
    action_dim = int(env.action_space.shape[0])
    T = int(horizon_steps)

    dtype = np.float32 if save_float32 else np.float64

    rod_pos = np.zeros((n_episodes, T + 1, 3, n_nodes), dtype=dtype)
    rod_vel = np.zeros((n_episodes, T + 1, 3, n_nodes), dtype=dtype)
    rod_dir = np.zeros((n_episodes, T + 1, 3, 3, n_elem), dtype=dtype)
    rod_omega = np.zeros((n_episodes, T + 1, 3, n_elem), dtype=dtype)

    target_pos = np.zeros((n_episodes, T + 1, 3, 1), dtype=dtype)
    target_dir = np.zeros((n_episodes, T + 1, 3, 3, 1), dtype=dtype)

    actions = np.zeros((n_episodes, T, action_dim), dtype=dtype)
    valid_mask = np.zeros((n_episodes, T + 1), dtype=np.int8)

    dt_effective = float(env.time_step) * float(env.num_steps_per_update)
    meta = {
        "seed": int(seed),
        "n_episodes": int(n_episodes),
        "horizon_steps": int(horizon_steps),
        "hold_steps": int(hold_steps),
        "action": {"a_max": float(a_max), "sigma": float(sigma), "type": "random_walk"},
        "env": {
            "mode": int(mode),
            "dim": float(dim),
            "sim_dt": float(sim_dt),
            "num_steps_per_update": int(num_steps_per_update),
            "dt_effective": float(dt_effective),
            "n_elem": int(n_elem),
            "n_nodes": int(n_nodes),
            "number_of_control_points": int(number_of_control_points),
            "alpha": float(alpha),
            "beta": float(beta),
            "boundary": list(map(float, boundary)),
            "final_time": float(final_time),
        },
        "created_unix_time": time.time(),
        "strict_alignment": "Stores x_{t+1} BEFORE checking done, ensuring T actions align with T next-states when valid.",
    }

    hold_steps = max(int(hold_steps), 1)

    for ep in range(n_episodes):
        env.reset()

        # init action
        a_prev = rng.uniform(-a_max, a_max, size=(action_dim,)).astype(np.float64)
        a_hold = a_prev.copy()
        hold_ctr = 0

        # store frame 0
        x = snapshot_full_state(env)
        if not is_finite_dict(x):
            print(f"[ep {ep+1:03d}] invalid at reset -> skipped")
            continue

        rod_pos[ep, 0] = x["rod_pos"].astype(dtype)
        rod_vel[ep, 0] = x["rod_vel"].astype(dtype)
        rod_dir[ep, 0] = x["rod_dir"].astype(dtype)
        rod_omega[ep, 0] = x["rod_omega"].astype(dtype)

        target_pos[ep, 0] = x["target_pos"].astype(dtype)
        if x["target_dir"] is not None:
            target_dir[ep, 0] = x["target_dir"].astype(dtype)

        valid_mask[ep, 0] = 1

        # steps t=0..T-1
        for t in range(T):
            if hold_ctr % hold_steps == 0:
                a_hold = sample_action_random_walk(rng, a_prev, a_max=a_max, sigma=sigma)
                a_prev = a_hold.copy()
            hold_ctr += 1

            actions[ep, t] = a_hold.astype(dtype)

            # take env step
            _, _, done, _ = env.step(a_hold)

            # snapshot next state
            x_next = snapshot_full_state(env)
            if not is_finite_dict(x_next):
                
                break

    
            rod_pos[ep, t + 1] = x_next["rod_pos"].astype(dtype)
            rod_vel[ep, t + 1] = x_next["rod_vel"].astype(dtype)
            rod_dir[ep, t + 1] = x_next["rod_dir"].astype(dtype)
            rod_omega[ep, t + 1] = x_next["rod_omega"].astype(dtype)

            target_pos[ep, t + 1] = x_next["target_pos"].astype(dtype)
            if x_next["target_dir"] is not None:
                target_dir[ep, t + 1] = x_next["target_dir"].astype(dtype)

            valid_mask[ep, t + 1] = 1

            # ✅ 再判断 done（即使 done=True，也不丢最后一帧）
            if done:
                break

        print(f"[ep {ep+1:03d}/{n_episodes}] valid frames: {int(valid_mask[ep].sum())}/{T+1}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        rod_pos=rod_pos,
        rod_vel=rod_vel,
        rod_dir=rod_dir,
        rod_omega=rod_omega,
        target_pos=target_pos,
        target_dir=target_dir,
        action=actions,
        valid_mask=valid_mask,
        meta_json=json.dumps(meta, indent=2),
    )
    print(f"\nSaved dataset to: {out_path}")


def parse_args():
    ap = argparse.ArgumentParser("Collect STRICT-aligned (x_t, a_t, x_{t+1}) dataset for mode=2 random target.")

    ap.add_argument("--out_path", type=str, default="data_case2/case2_fullfield_strict.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_episodes", type=int, default=30)
    ap.add_argument("--horizon_steps", type=int, default=400)
    ap.add_argument("--hold_steps", type=int, default=5)

    ap.add_argument("--a_max", type=float, default=0.8)
    ap.add_argument("--sigma", type=float, default=0.05)

    ap.add_argument("--mode", type=int, default=2)
    ap.add_argument("--dim", type=float, default=3.5)
    ap.add_argument("--sim_dt", type=float, default=2.5e-4)
    ap.add_argument("--num_steps_per_update", type=int, default=10)
    ap.add_argument("--n_elem", type=int, default=40)
    ap.add_argument("--number_of_control_points", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=140.0)
    ap.add_argument("--beta", type=float, default=140.0)

    ap.add_argument("--boundary", type=float, nargs=6,
                    default=(-0.35, 0.35, 0.55, 1.10, -0.35, 0.35))

    ap.add_argument("--save_float32", action="store_true")

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect_dataset(
        out_path=args.out_path,
        seed=args.seed,
        n_episodes=args.n_episodes,
        horizon_steps=args.horizon_steps,
        hold_steps=args.hold_steps,
        a_max=args.a_max,
        sigma=args.sigma,
        mode=args.mode,
        dim=args.dim,
        sim_dt=args.sim_dt,
        num_steps_per_update=args.num_steps_per_update,
        n_elem=args.n_elem,
        number_of_control_points=args.number_of_control_points,
        alpha=args.alpha,
        beta=args.beta,
        boundary=tuple(args.boundary),
        save_float32=bool(args.save_float32),
    )