import numpy as np
from quantum_fem.block_encoding import laplacian_1d_4x4, hhl_like_4x4
from quantum_fem.quantum_linear_functional import hadamard_test_overlap

"""Demo: 4x4 Laplacian, HHL-like multi-eigenvalue rotation, and a functional estimate.
"""

def main():
    A = laplacian_1d_4x4()
    # Choose a simple b and g
    b = np.array([0.0, 1.0, 1.0, 0.0], dtype=float)
    g = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)  # average functional

    res = hhl_like_4x4(A, b, reps=4000)
    x_est = res.solution_amplitudes
    print("Eigenvalues:", res.eigenvalues)
    print("Success prob:", res.success_prob)
    print("Estimated x (scaled):", x_est)

    func = hadamard_test_overlap(x_est, g, shots=4000)
    print("Estimated <g,x> via Hadamard test:", func.value)

if __name__ == "__main__":
    main()
