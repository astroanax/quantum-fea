"""Test proper QPE implementation."""
import numpy as np
from src.quantum_fem.phase_estimation import quantum_phase_estimation

# Test 1: Simple 2x2 matrix with known eigenvalues
print("=" * 60)
print("Test 1: 2x2 Matrix A = [[3, 1], [1, 3]]")
print("Expected eigenvalues: λ₁=2, λ₂=4")
print("=" * 60)

A = np.array([[3.0, 1.0],
              [1.0, 3.0]])

# Run QPE with different precision levels
for n_bits in [4, 6, 8]:
    print(f"\n--- QPE with {n_bits} bits precision ---")
    result = quantum_phase_estimation(A, precision_bits=n_bits)  # Auto t-selection
    
    print(f"t parameter: {result.t_parameter:.6f}")
    print(f"Measured phases: {result.phases}")
    print(f"Estimated eigenvalues: {result.eigenvalues}")
    print(f"Counts: {result.counts}")
    
    # Compare with exact eigenvalues
    exact_eigvals = np.linalg.eigvalsh(A)
    print(f"Exact eigenvalues: {exact_eigvals}")
    
    if len(result.eigenvalues) >= 2:
        errors = np.abs(np.sort(result.eigenvalues)[:2] - exact_eigvals)
        print(f"Errors: {errors}")
        print(f"Max error: {np.max(errors):.6f}")

# Test 2: FEM-like matrix
print("\n" + "=" * 60)
print("Test 2: FEM-like matrix")
print("=" * 60)

# Typical small stiffness matrix
k = 1000.0  # stiffness
A_fem = np.array([[k, -k],
                  [-k, k]])

# Add small diagonal term to make it positive definite
A_fem = A_fem + np.eye(2) * 0.01

print(f"\nMatrix:\n{A_fem}")

result = quantum_phase_estimation(A_fem, precision_bits=8)  # Auto t-selection
exact_eigvals = np.linalg.eigvalsh(A_fem)

print(f"\nt parameter: {result.t_parameter:.6f}")
print(f"Estimated eigenvalues: {result.eigenvalues}")
print(f"Exact eigenvalues: {exact_eigvals}")

# Test 3: Test with specific initial state (eigenvector)
print("\n" + "=" * 60)
print("Test 3: QPE with eigenvector as initial state")
print("=" * 60)

A = np.array([[3.0, 1.0],
              [1.0, 3.0]])
eigvals, eigvecs = np.linalg.eigh(A)

print(f"Exact eigenvalues: {eigvals}")

# Prepare in first eigenvector
init_state = eigvecs[:, 0]
print(f"Initial state (eigenvector 1): {init_state}")

result = quantum_phase_estimation(A, precision_bits=8, initial_state=init_state)  # Auto t

print(f"t parameter: {result.t_parameter:.6f}")
print(f"Estimated eigenvalues: {result.eigenvalues}")
print(f"Should predominantly measure λ={eigvals[0]:.4f}")
print(f"Measurement counts: {result.counts}")

# Check which eigenvalue was measured most
if len(result.eigenvalues) > 0:
    most_common_eigval = result.eigenvalues[0]
    error = np.abs(most_common_eigval - eigvals[0])
    print(f"Most common measurement: λ={most_common_eigval:.4f}")
    print(f"Error: {error:.6f}")
