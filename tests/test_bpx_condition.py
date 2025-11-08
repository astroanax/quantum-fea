import numpy as np
from quantum_fem.fem2d import PoissonQ1Mesh2D
from quantum_fem.preconditioner import BPXPreconditioner2D


def cond(A: np.ndarray) -> float:
    w = np.linalg.eigvalsh(A)
    w = w[w > 1e-12]
    return float(w[-1] / w[0])


def test_bpx_condition_number_reduction():
    n = 8  # finest intervals per side
    mesh = PoissonQ1Mesh2D(intervals=n)
    A = mesh.stiffness_matrix()
    bpx = BPXPreconditioner2D(finest_intervals=n)
    M, _ = bpx.apply(A, np.zeros_like(A[:,0]))
    cA = cond(A)
    cM = cond(M)
    # Expect at least 30% reduction for this setup.
    assert cM < 0.7 * cA
