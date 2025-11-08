import numpy as np

class OneDCantileverMesh:
    """Simple 1D mesh for a cantilever beam discretized into linear elements.

    We approximate a bending or axial stiffness surrogate over length L with Ne elements.
    Degrees of freedom: node displacements u_i. Fixed at left end (node 0). A point load at the right end.
    """
    def __init__(self, length: float, elements: int, young_modulus: float = 1.0, area_moment: float = 1.0):
        self.L = length
        self.Ne = elements
        self.Nn = elements + 1
        self.E = young_modulus
        self.I = area_moment  # Using I as a surrogate; in axial case interpret as area A.
        self.h = length / elements

    def stiffness_matrix(self, bending: bool = False) -> np.ndarray:
        """Assemble a simplistic stiffness matrix.

        For axial bar: local k = (E*A / h) [[1,-1],[-1,1]]
        For bending surrogate (very crude): scale by (E*I / h**3)
        Returns full K with boundary condition rows/cols for fixed node 0 removed.
        """
        if bending:
            coeff = self.E * self.I / (self.h ** 3)
        else:
            coeff = self.E * self.I / self.h
        K = np.zeros((self.Nn, self.Nn))
        for e in range(self.Ne):
            K[e, e] += coeff
            K[e, e+1] -= coeff
            K[e+1, e] -= coeff
            K[e+1, e+1] += coeff
        # Apply fixed boundary at node 0: remove first row/col
        K_red = K[1:, 1:]
        return K_red

    def load_vector(self, end_load: float) -> np.ndarray:
        """Point load at free end translated to reduced system (excluding fixed node)."""
        f = np.zeros(self.Nn)
        f[-1] = end_load
        # Remove fixed node 0 entry
        return f[1:]


def classical_solve(K: np.ndarray, f: np.ndarray) -> np.ndarray:
    return np.linalg.solve(K, f)
