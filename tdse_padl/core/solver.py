"""
solver.py
=========
Crank-Nicolson (CN) implicit solver for the 1D Time-Dependent Schrödinger
Equation (TDSE) in natural units  ℏ = m = 1.

Governing equation
------------------
    iℏ ∂ψ/∂t = H ψ,    H = -ℏ²/(2m) ∂²/∂x² + V(x)

With ℏ = m = 1 this simplifies to:

    i ∂ψ/∂t = [-½ ∂²/∂x² + V(x)] ψ

Crank-Nicolson discretisation
------------------------------
The implicit midpoint rule gives a unitary (norm-conserving) evolution:

    (1 + i·(dt/2)·H) ψ^{n+1} = (1 - i·(dt/2)·H) ψ^n
     ↑ LHS matrix (A)                ↑ RHS matrix (B)

The Hamiltonian is tridiagonal on an N-point uniform grid (dx = L/N):

    H_{j,j}   = 1/dx²  +  V[j]          (diagonal)
    H_{j,j±1} = -1/(2·dx²)              (off-diagonal)

where the factor 1/dx² comes from ℏ²/(m·dx²) = 1/dx² in natural units, and
-1/(2·dx²) from -ℏ²/(2·m·dx²) = -1/(2·dx²).

Thomas algorithm (tridiagonal solver)
--------------------------------------
At each timestep we solve the complex tridiagonal system A·x = d using the
Thomas (forward-elimination / back-substitution) algorithm.  This runs in
O(N) time vs O(N²) for a dense solver, giving a dramatic speedup for large
grids.

Boundary conditions
--------------------
Dirichlet (absorbing walls): ψ[0] = ψ[N-1] = 0 at all times.
The first and last rows of the tridiagonal matrices are therefore set to the
identity row (coefficient 1, RHS 0).

References
----------
* Crank, J. & Nicolson, P. (1947). Proc. Camb. Phil. Soc. 43, 50-67.
* Press et al. "Numerical Recipes in C", §17.2 (Schrödinger equation).
"""

from __future__ import annotations

import numpy as np

from .potential import zero_potential
from .wavepacket import gaussian_wavepacket


# ---------------------------------------------------------------------------
# Thomas algorithm (tridiagonal solver)
# ---------------------------------------------------------------------------

def _thomas(
    lower: np.ndarray,
    main: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """
    Solve a tridiagonal linear system  A·x = rhs  using the Thomas algorithm.

    The system is stored in three 1-D arrays:

        lower[i]  → A[i, i-1]   (lower diagonal;  lower[0]  unused)
        main[i]   → A[i, i]     (main diagonal)
        upper[i]  → A[i, i+1]   (upper diagonal;  upper[-1] unused)

    All four arrays must have the same length N.  The algorithm works in-place
    on temporary copies so the caller's arrays are not modified.

    Parameters
    ----------
    lower, main, upper : np.ndarray, shape (N,), dtype complex128
        Diagonals of the tridiagonal matrix.
    rhs : np.ndarray, shape (N,), dtype complex128
        Right-hand side vector.

    Returns
    -------
    x : np.ndarray, shape (N,), dtype complex128
        Solution vector.
    """
    N = len(main)
    # Work on copies to avoid mutating caller's arrays
    c = upper.copy()   # modified upper diagonal
    d = rhs.copy()     # modified RHS

    # ------------------------------------------------------------------
    # Forward sweep (elimination)
    # ------------------------------------------------------------------
    # Normalise first row
    c[0] /= main[0]
    d[0] /= main[0]

    for i in range(1, N):
        denom = main[i] - lower[i] * c[i - 1]
        if i < N - 1:
            c[i] /= denom
        d[i] = (d[i] - lower[i] * d[i - 1]) / denom

    # ------------------------------------------------------------------
    # Back substitution
    # ------------------------------------------------------------------
    x = d.copy()
    for i in range(N - 2, -1, -1):
        x[i] -= c[i] * x[i + 1]

    return x


# ---------------------------------------------------------------------------
# Crank-Nicolson solver
# ---------------------------------------------------------------------------

class CrankNicolsonSolver:
    """
    Implicit Crank-Nicolson solver for the 1D TDSE.

    Solves  i ∂ψ/∂t = [-½ ∂²/∂x² + V(x)] ψ  on a uniform grid of N points
    spanning [0, L] with Dirichlet (absorbing) boundary conditions.

    The CN scheme is unconditionally stable and unitary for the pure
    Schrödinger equation, so the norm ∫|ψ|²dx should drift by less than
    machine-epsilon per step (in the absence of absorption / loss).

    Parameters
    ----------
    N : int, optional
        Number of spatial grid points (default 512).
    L : float, optional
        Spatial domain length (default 1.0).
    dt : float, optional
        Timestep in natural units (default 1e-5).
    V : np.ndarray or None, optional
        Potential energy array of shape (N,).  If None, defaults to zero
        potential (free particle).

    Attributes
    ----------
    N, L, dt : int / float
        Grid and time parameters.
    dx : float
        Spatial step size dx = L / N.
    x : np.ndarray, shape (N,)
        Spatial grid x[i] = i · dx.
    V : np.ndarray, shape (N,)
        Potential energy profile.
    lhs_lower, lhs_main, lhs_upper : np.ndarray, shape (N,), complex128
        Diagonals of the LHS matrix  A = I + i·(dt/2)·H.
    rhs_lower, rhs_main, rhs_upper : np.ndarray, shape (N,), complex128
        Diagonals of the RHS matrix  B = I - i·(dt/2)·H.
    """

    def __init__(
        self,
        N: int = 512,
        L: float = 1.0,
        dt: float = 1e-5,
        V: np.ndarray | None = None,
    ) -> None:
        self.N = N
        self.L = L
        self.dt = dt
        self.dx = L / N
        self.x = np.linspace(0.0, L, N, endpoint=False)

        # Potential
        if V is None:
            self.V = zero_potential(N, L)
        else:
            if V.shape != (N,):
                raise ValueError(
                    f"Potential V must have shape ({N},), got {V.shape}."
                )
            self.V = np.asarray(V, dtype=np.float64)

        # Build tridiagonal Hamiltonian elements (interior rows)
        # --------------------------------------------------------
        # H diagonal:     1/dx² + V[i]
        # H off-diagonal: -1/(2·dx²)
        dx2 = self.dx ** 2
        h_diag = 1.0 / dx2 + self.V           # shape (N,)
        h_off  = -0.5 / dx2                    # scalar (same for all off-diags)

        alpha = 1j * (dt / 2.0)                # complex prefactor

        # ------------------------------------------------------------------
        # LHS matrix  A = I + alpha·H    (coefficient of ψ^{n+1})
        # ------------------------------------------------------------------
        self.lhs_main  = (1.0 + alpha * h_diag).astype(np.complex128)
        self.lhs_upper = (      alpha * h_off  * np.ones(N)).astype(np.complex128)
        self.lhs_lower = (      alpha * h_off  * np.ones(N)).astype(np.complex128)

        # ------------------------------------------------------------------
        # RHS matrix  B = I - alpha·H    (coefficient of ψ^n)
        # ------------------------------------------------------------------
        self.rhs_main  = (1.0 - alpha * h_diag).astype(np.complex128)
        self.rhs_upper = (    - alpha * h_off   * np.ones(N)).astype(np.complex128)
        self.rhs_lower = (    - alpha * h_off   * np.ones(N)).astype(np.complex128)

        # ------------------------------------------------------------------
        # Enforce Dirichlet BCs on boundary rows
        # Boundary rows: main = 1, off-diags = 0, RHS value = 0
        # ------------------------------------------------------------------
        for mat_main, mat_upper, mat_lower in [
            (self.lhs_main, self.lhs_upper, self.lhs_lower),
            (self.rhs_main, self.rhs_upper, self.rhs_lower),
        ]:
            # Row 0
            mat_main[0]  = 1.0 + 0j
            mat_upper[0] = 0.0 + 0j
            mat_lower[0] = 0.0 + 0j   # lower[0] unused but kept consistent
            # Row N-1
            mat_main[-1]  = 1.0 + 0j
            mat_lower[-1] = 0.0 + 0j
            mat_upper[-1] = 0.0 + 0j  # upper[-1] unused but kept consistent

    # ------------------------------------------------------------------
    # Single time step
    # ------------------------------------------------------------------

    def step(self, psi: np.ndarray) -> np.ndarray:
        """
        Advance the wavefunction by one timestep dt using Crank-Nicolson.

        Algorithm
        ---------
        1. Compute  d = B · ψ^n  (tridiagonal matrix-vector product with RHS).
        2. Solve    A · ψ^{n+1} = d  (Thomas algorithm on LHS).
        3. Enforce Dirichlet BCs: ψ^{n+1}[0] = ψ^{n+1}[-1] = 0.

        The norm is intentionally *not* renormalised so that its drift can
        serve as an accuracy diagnostic.

        Parameters
        ----------
        psi : np.ndarray, shape (N,), dtype complex128
            Current wavefunction.

        Returns
        -------
        psi_new : np.ndarray, shape (N,), dtype complex128
            Wavefunction at time t + dt.
        """
        psi = np.asarray(psi, dtype=np.complex128)

        # Step 1: compute d = B · ψ  (tridiagonal matvec)
        d = np.zeros(self.N, dtype=np.complex128)
        d[0] = self.rhs_main[0] * psi[0]  # boundary row (no off-diag contrib)
        d[1:-1] = (
            self.rhs_lower[1:-1] * psi[:-2]
            + self.rhs_main[1:-1] * psi[1:-1]
            + self.rhs_upper[1:-1] * psi[2:]
        )
        d[-1] = self.rhs_main[-1] * psi[-1]  # boundary row

        # Force boundary RHS to 0 (ψ[0] = ψ[-1] = 0 at all times)
        d[0]  = 0.0 + 0j
        d[-1] = 0.0 + 0j

        # Step 2: solve A · ψ^{n+1} = d
        psi_new = _thomas(self.lhs_lower, self.lhs_main, self.lhs_upper, d)

        # Step 3: enforce Dirichlet BCs (belt-and-suspenders)
        psi_new[0]  = 0.0 + 0j
        psi_new[-1] = 0.0 + 0j

        return psi_new

    # ------------------------------------------------------------------
    # Multi-step helper
    # ------------------------------------------------------------------

    def step_n(self, psi: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Advance the wavefunction by n_steps timesteps.

        Parameters
        ----------
        psi : np.ndarray, shape (N,), dtype complex128
            Initial wavefunction.
        n_steps : int
            Number of CN steps to take.

        Returns
        -------
        psi : np.ndarray, shape (N,), dtype complex128
            Wavefunction after n_steps · dt time has elapsed.
        """
        psi = np.asarray(psi, dtype=np.complex128)
        for _ in range(n_steps):
            psi = self.step(psi)
        return psi


# ---------------------------------------------------------------------------
# Standalone test / demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from .potential import rectangular_barrier
    from .wavepacket import gaussian_wavepacket

    # -----------------------------------------------------------------------
    # Simulation parameters
    # -----------------------------------------------------------------------
    N      = 512
    L      = 1.0
    dt     = 1e-5
    dx     = L / N
    x      = np.linspace(0.0, L, N, endpoint=False)

    # -----------------------------------------------------------------------
    # 1. Build a rectangular barrier at centre=0.6, width=0.05, height=200
    # -----------------------------------------------------------------------
    barrier_center = 0.6
    barrier_width  = 0.05
    barrier_height = 200.0

    V = rectangular_barrier(N, L, barrier_center, barrier_width, barrier_height)

    # Locate barrier edge indices for transmission / reflection diagnostics
    half = barrier_width / 2.0
    barrier_start_idx = int((barrier_center - half) / dx)
    barrier_end_idx   = int((barrier_center + half) / dx) + 1

    # -----------------------------------------------------------------------
    # 2. Initialise Gaussian wavepacket
    # -----------------------------------------------------------------------
    psi = gaussian_wavepacket(N, L, x0=0.25, k0=50.0, sigma=0.05)

    # -----------------------------------------------------------------------
    # 3. Compute initial norm
    # -----------------------------------------------------------------------
    norm_initial = np.sum(np.abs(psi) ** 2) * dx

    # -----------------------------------------------------------------------
    # 4. Run 2000 Crank-Nicolson steps
    # -----------------------------------------------------------------------
    solver = CrankNicolsonSolver(N=N, L=L, dt=dt, V=V)
    psi = solver.step_n(psi, n_steps=2_000)

    # -----------------------------------------------------------------------
    # 5. Compute diagnostics
    # -----------------------------------------------------------------------
    norm_final   = np.sum(np.abs(psi) ** 2) * dx
    trans_coeff  = np.sum(np.abs(psi[barrier_end_idx:]) ** 2) * dx
    refl_coeff   = np.sum(np.abs(psi[:barrier_start_idx]) ** 2) * dx
    norm_conserved = abs(norm_final - norm_initial) < 1e-4

    # -----------------------------------------------------------------------
    # 6. Print results
    # -----------------------------------------------------------------------
    print(f"Initial norm: {norm_initial:.5f}")
    print(f"Final norm:   {norm_final:.5f}")
    print(f"Transmission: {trans_coeff * 100:.1f}%")
    print(f"Reflection:   {refl_coeff  * 100:.1f}%")
    print(f"Norm conserved: {norm_conserved}")

    if not norm_conserved:
        print(
            f"WARNING: norm drift = {abs(norm_final - norm_initial):.2e} "
            f"exceeds 1e-4 tolerance.",
            file=sys.stderr,
        )
        sys.exit(1)