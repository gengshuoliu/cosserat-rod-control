"""
B-spline action expansion utilities.

Replicates the MuscleTorquesWithVaryingBetaSplines logic:
  - n_ctrl+2 control points placed at  np.linspace(0, base_length, n_ctrl+2)
  - First and last control point values are ALWAYS 0  (boundary condition)
  - Middle n_ctrl values come from the RL action
  - Cubic B-spline (make_interp_spline, degree k=3)
  - Evaluated at cumulative element lengths  [ds, 2*ds, ..., base_length]
    (i.e., at element END positions, NOT midpoints)
  - Scaled by muscle_torque_scale (alpha or beta)

For the dataset, action layout is:
    a[0:6]   -> normal   direction (scale = alpha = 140.0)
    a[6:12]  -> binormal direction (scale = alpha = 140.0)
    a[12:18] -> twist    direction (scale = beta  = 140.0)
"""

import numpy as np
from scipy.interpolate import make_interp_spline


# --------------------------------------------------------------------------- #
#  Core single-direction expansion                                             #
# --------------------------------------------------------------------------- #

def expand_bspline(ctrl_values: np.ndarray,
                   n_ctrl: int,
                   base_length: float,
                   n_elem: int,
                   scale: float = 1.0) -> np.ndarray:
    """
    Expand n_ctrl control-point values into a spatial torque profile over n_elem elements.

    Parameters
    ----------
    ctrl_values : (n_ctrl,) array   — raw RL action values (typically in [-1, 1])
    n_ctrl      : int               — number of interior control points  (e.g. 6)
    base_length : float             — rod length  [m]                    (e.g. 1.0)
    n_elem      : int               — number of rod elements             (e.g. 40)
    scale       : float             — torque scale factor                (e.g. 140.0)

    Returns
    -------
    torque : (n_elem,) float64  — spatial torque at each element
    """
    # Positions of n_ctrl+2 control points (including boundary zeros)
    ctrl_pos = np.linspace(0.0, base_length, n_ctrl + 2)

    # Values: boundary values are zero, middle n_ctrl come from action
    ctrl_val = np.zeros(n_ctrl + 2, dtype=np.float64)
    ctrl_val[1:-1] = ctrl_values

    # Build cubic B-spline
    spline = make_interp_spline(ctrl_pos, ctrl_val, k=3)

    # Evaluation points: cumulative element lengths  (element END positions)
    # same as  np.cumsum(element_lengths)  for a uniform rod
    x_eval = np.linspace(base_length / n_elem, base_length, n_elem)  # [ds, 2ds, ..., L]

    return scale * spline(x_eval)


# --------------------------------------------------------------------------- #
#  Batch expansion: all 3 directions from an 18-dim action vector             #
# --------------------------------------------------------------------------- #

def expand_action_to_spatial(action: np.ndarray,
                              n_ctrl: int = 6,
                              base_length: float = 1.0,
                              n_elem: int = 40,
                              alpha_scale: float = 140.0,
                              beta_scale: float = 140.0) -> np.ndarray:
    """
    Expand a single 18-dim action vector to a spatial torque field (3, n_elem).

    Parameters
    ----------
    action      : (18,) array  — [normal_6, binormal_6, twist_6]
    n_ctrl      : int          — interior control points per direction (6)
    base_length : float        — rod length [m]
    n_elem      : int          — number of elements
    alpha_scale : float        — torque scale for bending  (normal + binormal)
    beta_scale  : float        — torque scale for twist

    Returns
    -------
    a_spatial : (3, n_elem) float64
        Row 0: normal   torque profile
        Row 1: binormal torque profile
        Row 2: twist    torque profile
    """
    assert action.shape[-1] == 3 * n_ctrl, (
        f"Expected action dim {3*n_ctrl}, got {action.shape[-1]}"
    )

    a_normal   = expand_bspline(action[:n_ctrl],         n_ctrl, base_length, n_elem, scale=alpha_scale)
    a_binormal = expand_bspline(action[n_ctrl:2*n_ctrl], n_ctrl, base_length, n_elem, scale=alpha_scale)
    a_twist    = expand_bspline(action[2*n_ctrl:],       n_ctrl, base_length, n_elem, scale=beta_scale)

    return np.stack([a_normal, a_binormal, a_twist], axis=0)  # (3, n_elem)


# --------------------------------------------------------------------------- #
#  Vectorised batch expansion  (for offline preprocessing of full dataset)     #
# --------------------------------------------------------------------------- #

def expand_actions_batch(actions: np.ndarray,
                         n_ctrl: int = 6,
                         base_length: float = 1.0,
                         n_elem: int = 40,
                         alpha_scale: float = 140.0,
                         beta_scale: float = 140.0) -> np.ndarray:
    """
    Expand a batch of actions to spatial torque fields.

    Parameters
    ----------
    actions : (..., 18)  any leading batch dimensions

    Returns
    -------
    a_spatial : (..., 3, n_elem)
    """
    leading = actions.shape[:-1]
    flat_actions = actions.reshape(-1, 3 * n_ctrl)
    N = flat_actions.shape[0]

    result = np.zeros((N, 3, n_elem), dtype=np.float64)
    for i in range(N):
        result[i] = expand_action_to_spatial(
            flat_actions[i], n_ctrl, base_length, n_elem, alpha_scale, beta_scale
        )

    return result.reshape(*leading, 3, n_elem)


# --------------------------------------------------------------------------- #
#  Precompute the B-spline basis matrix (for ultra-fast batch expansion)       #
# --------------------------------------------------------------------------- #

def build_bspline_basis(n_ctrl: int = 6,
                        base_length: float = 1.0,
                        n_elem: int = 40) -> np.ndarray:
    """
    Precompute the B-spline basis matrix  B  of shape  (n_elem, n_ctrl+2).

    The expansion can then be done as:
        torque = B  @  [0, ctrl_values, 0]   (fast matrix multiply)

    Returns
    -------
    B : (n_elem, n_ctrl+2) float64
    """
    ctrl_pos = np.linspace(0.0, base_length, n_ctrl + 2)
    x_eval   = np.linspace(base_length / n_elem, base_length, n_elem)

    B = np.zeros((n_elem, n_ctrl + 2), dtype=np.float64)
    for j in range(n_ctrl + 2):
        # Unit impulse at control point j
        e_j = np.zeros(n_ctrl + 2)
        e_j[j] = 1.0
        spline = make_interp_spline(ctrl_pos, e_j, k=3)
        B[:, j] = spline(x_eval)

    return B


class BSplineExpander:
    """
    Fast B-spline expander using a precomputed basis matrix.

    Usage
    -----
    expander = BSplineExpander(n_ctrl=6, base_length=1.0, n_elem=40)
    a_spatial = expander.expand(action_18dim, alpha_scale=140.0, beta_scale=140.0)
    """

    def __init__(self, n_ctrl: int = 6, base_length: float = 1.0, n_elem: int = 40):
        self.n_ctrl = n_ctrl
        self.n_elem = n_elem
        # Basis matrix shape: (n_elem, n_ctrl+2)
        self.B = build_bspline_basis(n_ctrl, base_length, n_elem)
        # The interior part (columns 1..n_ctrl):  (n_elem, n_ctrl)
        # boundary columns are multiplied by 0 always, so we can skip them
        self.B_interior = self.B[:, 1:-1]   # (n_elem, n_ctrl)

    def expand_direction(self, ctrl_6: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """
        ctrl_6 : (n_ctrl,)  — one direction's action values
        Returns: (n_elem,)  spatial torque
        """
        return scale * (self.B_interior @ ctrl_6.astype(np.float64))

    def expand(self, action: np.ndarray,
               alpha_scale: float = 140.0,
               beta_scale: float = 140.0) -> np.ndarray:
        """
        action : (..., 18)
        Returns: (..., 3, n_elem)
        """
        nc = self.n_ctrl
        leading = action.shape[:-1]
        flat = action.reshape(-1, 3 * nc).astype(np.float64)
        N = flat.shape[0]
        out = np.empty((N, 3, self.n_elem), dtype=np.float64)

        # Vectorised: B_interior (n_elem, nc) @ flat[..., :nc].T  -> (n_elem, N)
        out[:, 0, :] = (self.B_interior @ flat[:, :nc].T).T       * alpha_scale
        out[:, 1, :] = (self.B_interior @ flat[:, nc:2*nc].T).T   * alpha_scale
        out[:, 2, :] = (self.B_interior @ flat[:, 2*nc:].T).T     * beta_scale

        return out.reshape(*leading, 3, self.n_elem)
