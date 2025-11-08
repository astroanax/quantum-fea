import numpy as np
from typing import List, Tuple

class IdentityPreconditioner:
    """Identity (no-op) preconditioner for 1D examples."""
    def apply(self, K: np.ndarray, f: np.ndarray):
        return K, f


class BPXPreconditioner2D:
    """BPX-style multilevel preconditioner for 2D Poisson on a unit square with Q1 elements.

    This is a simplified pedagogical implementation:
    - Builds nested uniform grids: level ℓ has n_ℓ = 2^ℓ intervals per dimension.
    - Interior DOFs only (Dirichlet boundary); count m_ℓ = (n_ℓ - 1)^2.
    - Assembles stiffness matrices A_ℓ via 5-point Laplacian (finite difference equivalent to Q1 FE).
    - Constructs prolongations P_ℓ: interior fine -> coarse using bilinear interpolation via 1D Kronecker product.
    - Defines an additive BPX operator B = Σ_ℓ ω_ℓ P_{ℓ→L} D_ℓ^{-1} P_{ℓ→L}^T where D_ℓ is diag(A_ℓ) and P_{ℓ→L}
      is the composite prolongation from level ℓ to finest L.

    We then return the preconditioned matrix M = B @ A_L to demonstrate improved conditioning.
    Note: For actual solves one would apply B as a preconditioner inside an iterative method; here
    we expose M for eigenvalue/condition number inspection.
    """

    def __init__(self, finest_intervals: int, levels: int | None = None):
        if levels is None:
            levels = int(np.log2(finest_intervals))
        self.finest_intervals = finest_intervals
        self.levels = levels
        self.level_ns = [2 ** l for l in range(levels + 1)]  # intervals per side
        if self.level_ns[-1] != finest_intervals:
            raise ValueError("finest_intervals must be a power of two matching levels")
        self.A_levels: List[np.ndarray] = []
        self.P_to_fine: List[np.ndarray] = []  # prolongations from level ℓ to finest
        self._build_hierarchy()

    @staticmethod
    def _laplacian_matrix_2d(n: int) -> np.ndarray:
        """Assemble interior Laplacian matrix for n intervals per dimension with Dirichlet BCs.
        Interior grid has (n-1)^2 points. 5-point stencil: 4 on center, -1 on N,E,S,W.
        """
        m = (n - 1)
        N = m * m
        A = np.zeros((N, N))
        def idx(i, j):
            return i * m + j  # i,j in [0,m-1]
        for i in range(m):
            for j in range(m):
                p = idx(i, j)
                A[p, p] = 4.0
                if i > 0:
                    A[p, idx(i - 1, j)] = -1.0
                if i < m - 1:
                    A[p, idx(i + 1, j)] = -1.0
                if j > 0:
                    A[p, idx(i, j - 1)] = -1.0
                if j < m - 1:
                    A[p, idx(i, j + 1)] = -1.0
        # Scale by h^{-2} with h = 1/n, but uniform scaling does not alter condition number; omit.
        return A

    @staticmethod
    def _prolongation_1d(n_coarse: int, n_fine: int) -> np.ndarray:
        """1D interior prolongation from coarse (n_coarse intervals) to fine (n_fine=2*n_coarse intervals).
        Interior counts: m_c = n_coarse -1, m_f = n_fine -1.
        Maps coarse interior vector v_c to fine interior v_f via linear interpolation.
        """
        if n_fine != 2 * n_coarse:
            raise ValueError("n_fine must be 2 * n_coarse")
        m_c = n_coarse - 1
        m_f = n_fine - 1
        P = np.zeros((m_f, m_c))
        # Fine interior indices i=1..n_fine-1; 0-based i' = i-1
        for i in range(1, n_fine):
            i_f = i - 1
            if i % 2 == 0:  # even i = 2k aligns with coarse interior k
                k = i // 2
                if 1 <= k <= n_coarse - 1:
                    P[i_f, k - 1] = 1.0
            else:  # odd i = 2k -1 lies between coarse nodes k-1 and k
                k = (i + 1) // 2
                if 1 <= k - 1 <= n_coarse - 1 and 1 <= k <= n_coarse - 1:
                    P[i_f, k - 2] = 0.5
                    P[i_f, k - 1] = 0.5
        return P

    @staticmethod
    def _prolongation_2d(n_coarse: int, n_fine: int) -> np.ndarray:
        P1 = BPXPreconditioner2D._prolongation_1d(n_coarse, n_fine)
        # 2D is Kronecker product of 1D in y then x (or vice versa, consistent ordering)
        return np.kron(P1, P1)

    def _build_hierarchy(self):
        # Assemble A_ℓ and composite prolongations to finest.
        prolongations_level: List[np.ndarray] = [np.eye((self.level_ns[0]-1)**2)]  # P_0→0
        for l, n in enumerate(self.level_ns):
            A_l = self._laplacian_matrix_2d(n)
            self.A_levels.append(A_l)
            if l == 0:
                continue
            n_coarse = self.level_ns[l-1]
            P_lc = self._prolongation_2d(n_coarse, n)
            # Composite prolongation to finest: P_{l→L} = P_{l→l} if l == L else chain
            prolongations_level.append(P_lc @ prolongations_level[-1])
        # Adjust composite prolongations to map each level directly to finest (last level index L)
        P_to_finest = []
        P_L = prolongations_level[-1]
        for l, P_comp in enumerate(prolongations_level):
            # If l == L, composite is identity on finest; ensure shape matches A_L
            if l == len(self.level_ns) - 1:
                P_to_finest.append(np.eye(P_L.shape[0]))
            else:
                # Need mapping from level l interior to finest: multiply by remaining prolongations
                remaining = np.eye(P_L.shape[0]) @ P_L  # P_L is mapping from level 0 to L
                # P_comp maps level 0 to level l; so level l to L is P_L @ (P_comp)^+ (use least squares pseudo-inverse)
                # Simplify: store direct product chain built incrementally: P_l = P_{l-1}→l ... ; we can build direct by chaining fine prolongations.
                # For clarity in this pedagogical code, recompute forward chain.
                chain = np.eye((self.level_ns[l]-1)**2)
                for k in range(l, len(self.level_ns)-1):
                    chain = self._prolongation_2d(self.level_ns[k], self.level_ns[k+1]) @ chain
                P_to_finest.append(chain)
        self.P_to_fine = P_to_finest

    def apply(self, K: np.ndarray, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return preconditioned matrix M = B K and unchanged right-hand side f.
        Assumes K corresponds to finest level Laplacian with matching size.
        """
        A_fine = self.A_levels[-1]
        if K.shape != A_fine.shape:
            raise ValueError("Input K size does not match finest level Laplacian dimensions")
        # Build additive BPX operator B
        B = self.compute_B()
        M = B @ K
        return M, f

    def compute_B(self) -> np.ndarray:
        """Compute and cache the additive BPX operator B = sum_l P_l D_l^{-1} P_l^T.
        Returns B for reuse by apply_B.
        """
        if hasattr(self, "_B_cache") and self._B_cache is not None:
            return self._B_cache
        N = self.A_levels[-1].shape[0]
        B = np.zeros((N, N), dtype=float)
        for l, A_l in enumerate(self.A_levels):
            D_inv = 1.0 / np.diag(A_l)
            D_inv_mat = np.diag(D_inv)
            P_l = self.P_to_fine[l]
            B += P_l @ D_inv_mat @ P_l.T
        self._B_cache = B
        return B

    def apply_B(self, v: np.ndarray) -> np.ndarray:
        """Apply the BPX operator B to a vector v via cached matrix multiply (small problems)."""
        B = self.compute_B()
        return B @ v
