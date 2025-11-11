import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class Beam1DMesh:
    """Euler-Bernoulli beam: DOFs = [w, theta] per node."""
    length: float
    elements: int
    E: float = 1.0
    I: float = 1.0

    def __post_init__(self):
        if self.elements < 1:
            raise ValueError("elements must be >= 1")
        self.Nn = self.elements + 1
        self.h = self.length / self.elements

    @property
    def dof_count(self) -> int:
        return 2 * self.Nn

    def element_stiffness(self) -> np.ndarray:
        h = self.h
        EI = self.E * self.I
        ke = (EI / h**3) * np.array(
            [[12,    6*h,  -12,    6*h],
             [ 6*h, 4*h*h, -6*h, 2*h*h],
             [-12,  -6*h,   12,  -6*h],
             [ 6*h, 2*h*h, -6*h, 4*h*h]], dtype=float)
        return ke

    def assemble_stiffness(self) -> np.ndarray:
        ndof = self.dof_count
        K = np.zeros((ndof, ndof), dtype=float)
        ke = self.element_stiffness()
        for e in range(self.elements):
            n0, n1 = e, e + 1
            idx = [2*n0, 2*n0+1, 2*n1, 2*n1+1]
            for a in range(4):
                for b in range(4):
                    K[idx[a], idx[b]] += ke[a, b]
        return K

    def load_vector(self, tip_force: float = 0.0, tip_moment: float = 0.0) -> np.ndarray:
        f = np.zeros(self.dof_count, dtype=float)
        last = self.Nn - 1
        f[2*last] += tip_force
        f[2*last + 1] += tip_moment
        return f

    def apply_cantilever_bc(self, K: np.ndarray, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        fixed = np.array([0, 1], dtype=int)
        free = np.setdiff1d(np.arange(self.dof_count), fixed)
        K_red = K[np.ix_(free, free)]
        f_red = f[free]
        return K_red, f_red, free

    def reconstruct_full(self, u_red: np.ndarray, free: np.ndarray) -> np.ndarray:
        u_full = np.zeros(self.dof_count, dtype=float)
        u_full[free] = u_red
        return u_full


def classical_solve(K: np.ndarray, f: np.ndarray) -> np.ndarray:
    return np.linalg.solve(K, f)
