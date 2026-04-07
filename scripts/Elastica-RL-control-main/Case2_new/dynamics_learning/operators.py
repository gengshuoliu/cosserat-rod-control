"""
Discrete dissipative linear operators  L_v  and  L_omega.

Implements the spatially-discrete version of:
    (L_v  v)_i = alpha_v  * d^2v/ds^2|_i  - beta_v  * v_i
    (L_w  w)_i = alpha_w  * d^2w/ds^2|_i  - beta_w  * w_i

Using second-order centred finite differences with:
  - Dirichlet BC at index 0  (clamped end : v=0, omega=0)
  - Neumann   BC at last idx  (free  end  : dv/ds=0, domega/ds=0)

The IMEX time step solves:
    (I - dt * L) * u^{n+1}  =  u^n + dt * N_theta(u^n, ...)

Since L is fixed, we precompute:
    A_v     = I - dt * L_v     (41x41)
    A_omega = I - dt * L_omega (40x40)
and their inverses (small dense matrices, O(N^2) solve per step).

All matrices are returned as numpy float64 arrays.
"""

import numpy as np
import torch


# --------------------------------------------------------------------------- #
#  Build the raw L matrix (numpy)                                              #
# --------------------------------------------------------------------------- #

def build_L(n: int, ds: float, alpha: float, beta: float,
            dirichlet_start: bool = True,
            neumann_end: bool = True) -> np.ndarray:
    """
    Build an  n×n  tridiagonal dissipative operator matrix L.

    Interior row i  (1 <= i <= n-2):
        L[i, i-1] =  alpha / ds^2
        L[i, i  ] = -2*alpha/ds^2  - beta
        L[i, i+1] =  alpha / ds^2

    Row 0 (dirichlet_start=True):
        All zeros  ->  (I - dt*L)[0, :] = [1, 0, ...]  -> u[0] fixed at 0

    Row n-1 (neumann_end=True):
        Ghost node: u[n] = u[n-2]  ->  d^2u/ds^2 at n-1 = 2*(u[n-2]-u[n-1])/ds^2
        L[n-1, n-2] = 2*alpha/ds^2
        L[n-1, n-1] = -2*alpha/ds^2 - beta

    Returns
    -------
    L : np.ndarray  shape (n, n), dtype float64
    """
    L = np.zeros((n, n), dtype=np.float64)
    c = alpha / ds**2

    # Interior rows
    for i in range(1, n - 1):
        L[i, i - 1] = c
        L[i, i]     = -(2.0 * c + beta)
        L[i, i + 1] = c

    # Last row: Neumann (zero-slope) via ghost node
    if neumann_end:
        L[n - 1, n - 2] = 2.0 * c
        L[n - 1, n - 1] = -(2.0 * c + beta)

    # Row 0: Dirichlet (leave as all zeros → identity row in A = I - dt*L)
    # If dirichlet_start=False, treat row 0 as Neumann instead
    if not dirichlet_start:
        L[0, 0] = -(2.0 * c + beta)
        L[0, 1] = 2.0 * c   # ghost node: u[-1] = u[1]

    return L


# --------------------------------------------------------------------------- #
#  High-level: build both operators and IMEX matrices                          #
# --------------------------------------------------------------------------- #

class DiscreteOperators:
    """
    Precomputed discrete operators for the IMEX surrogate.

    Attributes
    ----------
    L_v      : (n_nodes, n_nodes)  np.float64   raw L for v
    L_omega  : (n_elem,  n_elem)   np.float64   raw L for omega
    A_v      : (n_nodes, n_nodes)  np.float64   I - dt*L_v
    A_omega  : (n_elem,  n_elem)   np.float64   I - dt*L_omega
    A_v_inv  : (n_nodes, n_nodes)  np.float64   inv(A_v)
    A_w_inv  : (n_elem,  n_elem)   np.float64   inv(A_omega)

    PyTorch tensor buffers (same matrices, for GPU operations):
    A_v_t, A_omega_t, A_v_inv_t, A_w_inv_t
    """

    def __init__(self, n_elem: int, ds: float, dt: float,
                 alpha_v: float, beta_v: float,
                 alpha_omega: float, beta_omega: float):
        """
        Parameters
        ----------
        n_elem   : number of rod elements  (e.g. 40)
        ds       : spatial step size [m]   (= base_length / n_elem)
        dt       : effective time step [s]
        alpha_v  : spatial diffusivity for v
        beta_v   : uniform damping rate for v
        alpha_omega : spatial diffusivity for omega
        beta_omega  : uniform damping rate for omega
        """
        self.n_elem  = n_elem
        self.n_nodes = n_elem + 1
        self.ds = ds
        self.dt = dt

        # --- Build L matrices ---
        # L_v : node grid (n_nodes=41),  row 0 Dirichlet,  last row Neumann
        self.L_v = build_L(self.n_nodes, ds, alpha_v, beta_v,
                           dirichlet_start=True, neumann_end=True)

        # L_v_elem : element grid (n_elem=40), same BCs
        #   Eliminates node↔element interpolation in the dynamics loop
        #   while preserving the full dissipative operator L = α∂_ss − βI.
        self.L_v_elem = build_L(n_elem, ds, alpha_v, beta_v,
                                dirichlet_start=True, neumann_end=True)

        # L_omega : element grid (n_elem=40),  row 0 Dirichlet (omega[0]=0,
        #           clamped end constrains director[0]), last row Neumann
        self.L_omega = build_L(n_elem, ds, alpha_omega, beta_omega,
                               dirichlet_start=True, neumann_end=True)

        # --- IMEX matrices  A = I - dt * L ---
        self.A_v        = np.eye(self.n_nodes) - dt * self.L_v
        self.A_v_elem   = np.eye(n_elem)       - dt * self.L_v_elem
        self.A_omega    = np.eye(n_elem)        - dt * self.L_omega

        # --- Inverses (small dense matrices) ---
        self.A_v_inv      = np.linalg.inv(self.A_v)
        self.A_v_elem_inv = np.linalg.inv(self.A_v_elem)
        self.A_w_inv      = np.linalg.inv(self.A_omega)

        # Verify positive-definiteness (A should be SPD since L is neg-semidefinite)
        self._check_positive_definite()

    def _check_positive_definite(self):
        for name, A in [("A_v", self.A_v),
                        ("A_v_elem", self.A_v_elem),
                        ("A_omega", self.A_omega)]:
            eig = np.linalg.eigvalsh(A)
            assert eig.min() > 0, (
                f"{name} is not positive definite! min eigenvalue = {eig.min():.4e}"
            )

    def to_torch(self, device: str = "cpu"):
        """
        Register torch tensor versions of the matrices.
        Call this once before training.
        """
        self.A_v_t          = torch.tensor(self.A_v,          dtype=torch.float32, device=device)
        self.A_v_elem_t     = torch.tensor(self.A_v_elem,     dtype=torch.float32, device=device)
        self.A_omega_t      = torch.tensor(self.A_omega,      dtype=torch.float32, device=device)
        self.A_v_inv_t      = torch.tensor(self.A_v_inv,      dtype=torch.float32, device=device)
        self.A_v_elem_inv_t = torch.tensor(self.A_v_elem_inv, dtype=torch.float32, device=device)
        self.A_w_inv_t      = torch.tensor(self.A_w_inv,      dtype=torch.float32, device=device)
        return self

    # ---------------------------------------------------------------------- #
    #  Key operations used in training / rollout                              #
    # ---------------------------------------------------------------------- #

    def apply_A_v(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute  A_v @ v  for batched v.

        Parameters
        ----------
        v : (B, 3, n_nodes) float32

        Returns
        -------
        (B, 3, n_nodes) float32
        """
        # einsum: result[b,c,i] = sum_j A_v[i,j] * v[b,c,j]
        return torch.einsum("ij,bcj->bci", self.A_v_t, v)

    def apply_A_omega(self, omega: torch.Tensor) -> torch.Tensor:
        """
        Compute  A_omega @ omega  for batched omega.

        Parameters
        ----------
        omega : (B, 3, n_elem) float32

        Returns
        -------
        (B, 3, n_elem) float32
        """
        return torch.einsum("ij,bcj->bci", self.A_omega_t, omega)

    def solve_v_elem(self, rhs: torch.Tensor) -> torch.Tensor:
        """
        Solve  A_v_elem @ v_next  =  rhs   via precomputed inverse (element grid).

        Parameters
        ----------
        rhs : (B, 3, n_elem) float32    rhs = v_elem_curr + dt * ab_v_elem

        Returns
        -------
        v_next : (B, 3, n_elem) float32
        """
        return torch.einsum("ij,bcj->bci", self.A_v_elem_inv_t, rhs)

    def solve_v(self, rhs: torch.Tensor) -> torch.Tensor:
        """
        Solve  A_v @ v_next  =  rhs   via precomputed inverse.

        Parameters
        ----------
        rhs : (B, 3, n_nodes) float32    rhs = v_curr + dt * ab_v_node

        Returns
        -------
        v_next : (B, 3, n_nodes) float32
        """
        return torch.einsum("ij,bcj->bci", self.A_v_inv_t, rhs)

    def solve_omega(self, rhs: torch.Tensor) -> torch.Tensor:
        """
        Solve  A_omega @ omega_next  =  rhs  via precomputed inverse.

        Parameters
        ----------
        rhs : (B, 3, n_elem) float32    rhs = omega_curr + dt * ab_omega

        Returns
        -------
        omega_next : (B, 3, n_elem) float32
        """
        return torch.einsum("ij,bcj->bci", self.A_w_inv_t, rhs)

    def compute_target_v_node(self, v_curr: np.ndarray, v_next: np.ndarray) -> np.ndarray:
        """
        Compute the training target for N_theta^v  at node grid.

            target_v_node = (A_v @ v_next  -  v_curr) / dt

        Parameters
        ----------
        v_curr : (..., 3, n_nodes) float32/64
        v_next : (..., 3, n_nodes) float32/64

        Returns
        -------
        target : (..., 3, n_nodes) float64
        """
        # Apply A_v along the last axis (spatial):
        # A_v @ v_next[..., :] -> (..., 3, n_nodes)
        Av_v_next = np.einsum("ij,...j->...i", self.A_v, v_next.astype(np.float64))
        return (Av_v_next - v_curr.astype(np.float64)) / self.dt

    def compute_target_omega(self, omega_curr: np.ndarray, omega_next: np.ndarray) -> np.ndarray:
        """
        Compute the training target for N_theta^omega  at element grid.

            target_omega = (A_omega @ omega_next  -  omega_curr) / dt

        Parameters
        ----------
        omega_curr : (..., 3, n_elem)
        omega_next : (..., 3, n_elem)

        Returns
        -------
        target : (..., 3, n_elem) float64
        """
        Aw_o_next = np.einsum("ij,...j->...i", self.A_omega, omega_next.astype(np.float64))
        return (Aw_o_next - omega_curr.astype(np.float64)) / self.dt

    def compute_target_v_elem_direct(self, v_elem_curr: np.ndarray, v_elem_next: np.ndarray) -> np.ndarray:
        """
        Compute the training target for N_theta^v directly on the element grid.

        Uses the element-grid IMEX formula (no node interpolation):
            target = (A_v_elem @ v_elem_next  -  v_elem_curr) / dt

        Parameters
        ----------
        v_elem_curr : (..., 3, n_elem) float32/64
        v_elem_next : (..., 3, n_elem) float32/64

        Returns
        -------
        target : (..., 3, n_elem) float64
        """
        Av_v_next = np.einsum("ij,...j->...i", self.A_v_elem, v_elem_next.astype(np.float64))
        return (Av_v_next - v_elem_curr.astype(np.float64)) / self.dt

    def compute_target_v_elem(self, v_curr: np.ndarray, v_next: np.ndarray) -> np.ndarray:
        """
        Convenience: compute target_v at NODE grid, then interpolate to ELEMENT grid.
        This is what the FNO loss uses (element-level comparison).

            target_v_elem[..., i] = 0.5*(target_v_node[..., i] + target_v_node[..., i+1])

        Parameters
        ----------
        v_curr : (..., 3, n_nodes)
        v_next : (..., 3, n_nodes)

        Returns
        -------
        target_v_elem : (..., 3, n_elem)
        """
        target_node = self.compute_target_v_node(v_curr, v_next)
        # Average adjacent nodes -> element centres
        return 0.5 * (target_node[..., :-1] + target_node[..., 1:])

    # ---------------------------------------------------------------------- #
    #  Utility: node <-> element interpolation                                #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def node_to_elem(x_node: torch.Tensor) -> torch.Tensor:
        """
        Interpolate field from nodes (n_nodes=41) to elements (n_elem=40).
            x_elem[i] = 0.5 * (x_node[i] + x_node[i+1])

        Parameters
        ----------
        x_node : (..., n_nodes) or (..., C, n_nodes)

        Returns
        -------
        x_elem : (..., n_elem) or (..., C, n_elem)
        """
        return 0.5 * (x_node[..., :-1] + x_node[..., 1:])

    @staticmethod
    def elem_to_node(x_elem: torch.Tensor, bc_start: float = 0.0) -> torch.Tensor:
        """
        Interpolate FNO output from elements (40) back to nodes (41).

        Convention:
          node[0]       = bc_start   (Dirichlet, e.g. 0 for clamped end)
          node[i]       = 0.5*(elem[i-1] + elem[i])   for i = 1,...,39
          node[40]      = elem[39]    (Neumann extrapolation at free end)

        Parameters
        ----------
        x_elem : (B, C, n_elem) float32
        bc_start : float  value at node 0 (default 0 for clamped end)

        Returns
        -------
        x_node : (B, C, n_nodes) float32
        """
        B, C, N = x_elem.shape
        x_node = torch.empty(B, C, N + 1, dtype=x_elem.dtype, device=x_elem.device)
        # Node 0: Dirichlet
        x_node[:, :, 0] = bc_start
        # Nodes 1..N-1: average adjacent elements
        x_node[:, :, 1:N] = 0.5 * (x_elem[:, :, :-1] + x_elem[:, :, 1:])
        # Node N: Neumann (free end), extrapolate from last element
        x_node[:, :, N] = x_elem[:, :, -1]
        return x_node

    def summary(self) -> str:
        eig_v_min      = np.linalg.eigvalsh(self.A_v).min()
        eig_v_elem_min = np.linalg.eigvalsh(self.A_v_elem).min()
        eig_w_min      = np.linalg.eigvalsh(self.A_omega).min()
        return (
            f"DiscreteOperators:\n"
            f"  n_nodes={self.n_nodes}, n_elem={self.n_elem}, ds={self.ds:.4f}, dt={self.dt:.4e}\n"
            f"  A_v      (node) min_eig={eig_v_min:.4f}  (should be > 0)\n"
            f"  A_v_elem (elem) min_eig={eig_v_elem_min:.4f}  (should be > 0)\n"
            f"  A_omega  (elem) min_eig={eig_w_min:.4f}  (should be > 0)\n"
        )
