"""Minimal test: call function vs inline with identical parameters."""
import numpy as np
from quantum_fem.phase_estimation import quantum_phase_estimation_2x2

A = np.array([[2.0, -1.0], [-1.0, 2.0]])

print("Calling quantum_phase_estimation_2x2(A, precision_bits=4, t=1.0, initial_state=None)")
result = quantum_phase_estimation_2x2(A, precision_bits=4, t=1.0, initial_state=None)

print(f"\nResult:")
print(f"  Phases: {result.phases}")
print(f"  Eigenvalues: {result.eigenvalues}")

print(f"\nExpected (from inline version that works):")
print(f"  Phases: [0.1875, 0.125]")
print(f"  Eigenvalues: [1.178, 0.785]")

print(f"\nTrue eigenvalues: {np.linalg.eigvalsh(A)}")

print(f"\n" + "="*70)
print("If phases are WRONG, then there's a bug in the function filtering/sorting logic")
print("The inline circuit construction is identical, so the issue is post-processing")
