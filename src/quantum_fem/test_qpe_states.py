"""Test QPE without initial state (should use |+⟩)."""
import numpy as np
from quantum_fem.phase_estimation import quantum_phase_estimation_2x2

A = np.array([[2.0, -1.0], [-1.0, 2.0]])
vals_true, _ = np.linalg.eigh(A)

print("Testing QPE with NO initial state (uses |+⟩)")
print(f"Matrix eigenvalues: {vals_true}")

# Expected: should measure λ=1 since |+⟩ is eigenvector for λ=1
result = quantum_phase_estimation_2x2(A, precision_bits=6, t=1.0, initial_state=None)

print(f"\nEstimated phases: {result.phases}")
print(f"Estimated eigenvalues: {result.eigenvalues}")
print(f"True eigenvalues: {vals_true}")

# Now test with explicit |+⟩ = [1, 1]/√2
print("\n" + "="*70)
print("Testing QPE with explicit |+⟩ initial state")

b_plus = np.array([1.0, 1.0]) / np.sqrt(2)
result2 = quantum_phase_estimation_2x2(A, precision_bits=6, t=1.0, initial_state=b_plus)

print(f"\nEstimated phases: {result2.phases}")
print(f"Estimated eigenvalues: {result2.eigenvalues}")
print(f"True eigenvalues: {vals_true}")

# Now test with |0⟩
print("\n" + "="*70)
print("Testing QPE with |0⟩ initial state")
print("|0⟩ = (|+⟩ + |-⟩)/√2, so should measure both λ=1 and λ=3")

b_zero = np.array([1.0, 0.0])
result3 = quantum_phase_estimation_2x2(A, precision_bits=6, t=1.0, initial_state=b_zero)

print(f"\nEstimated phases: {result3.phases}")
print(f"Estimated eigenvalues: {result3.eigenvalues}")
print(f"True eigenvalues: {vals_true}")
print(f"\nExpected to see both λ≈1 and λ≈3")
