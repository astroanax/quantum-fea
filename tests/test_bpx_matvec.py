import numpy as np
from quantum_fem.fem2d import PoissonQ1Mesh2D
from quantum_fem.preconditioner import BPXPreconditioner2D

def test_bpx_apply_B_shapes_and_effect():
    n = 8
    mesh = PoissonQ1Mesh2D(intervals=n)
    A = mesh.stiffness_matrix()
    bpx = BPXPreconditioner2D(finest_intervals=n)
    v = np.random.rand((n-1)*(n-1))
    Bv = bpx.apply_B(v)
    assert Bv.shape == v.shape
    # Check that B is positive definite-ish on random vectors: v^T B v > 0
    assert float(v @ Bv) > 0
