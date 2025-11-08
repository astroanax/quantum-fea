import numpy as np
from quantum_fem.block_encoding import laplacian_1d_4x4, hhl_like_4x4


def test_hhl_like_4x4_runs_and_dimensions():
    A = laplacian_1d_4x4()
    b = np.array([0.0, 1.0, 1.0, 0.0])
    res = hhl_like_4x4(A, b, reps=500)
    assert res.solution_amplitudes.shape == (4,)
    assert res.success_prob >= 0.0
