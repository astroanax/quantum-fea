import numpy as np
from typing import Tuple

class PoissonQ1Mesh2D:
    """Uniform Cartesian Q1 (bilinear) element mesh on [0,1]^2 with Dirichlet boundary (u=0).

    Provides stiffness matrix assembly via 5-point Laplacian stencil equivalent for uniform grid.
    """
    def __init__(self, intervals: int):
        if intervals < 2:
            raise ValueError("Need at least 2 intervals per dimension")
        self.n = intervals  # intervals per side
        self.m = intervals - 1  # interior points per side
        self.N = self.m * self.m
        self.h = 1.0 / intervals

    def stiffness_matrix(self) -> np.ndarray:
        """Return the interior Laplacian stiffness matrix (without h^{-2} scaling)."""
        m = self.m
        N = self.N
        A = np.zeros((N, N))
        def idx(i, j):
            return i * m + j
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
        return A

    def load_vector(self, f_handle=None) -> np.ndarray:
        """Assemble load vector for source term f(x,y); default constant 1.
        Dirichlet boundary assumed zero, simple midpoint quadrature per interior node.
        """
        if f_handle is None:
            f_handle = lambda x, y: 1.0
        m = self.m
        h = self.h
        b = np.zeros(self.N)
        xs = np.linspace(h, 1 - h, m)
        ys = np.linspace(h, 1 - h, m)
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                p = i * m + j
                b[p] = f_handle(x, y) * h * h  # integration weight ~ h^2
        return b


def classical_solve(K: np.ndarray, f: np.ndarray) -> np.ndarray:
    return np.linalg.solve(K, f)
