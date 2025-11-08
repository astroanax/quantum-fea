import numpy as np
from quantum_fem.quantum_linear_functional import hadamard_test_overlap


def test_hadamard_test_overlap_basic():
    x = np.array([1.0, 0.0, 0.0, 0.0])
    g = np.array([1.0, 0.0, 0.0, 0.0])
    est = hadamard_test_overlap(x, g, shots=1000)
    # Expect near nx*ng = 1.0
    assert est.value > 0.7
