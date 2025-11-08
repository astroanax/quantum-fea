"""Detailed QPE testing to understand phase wrapping."""
import numpy as np
from src.quantum_fem.phase_estimation import quantum_phase_estimation

print("=" * 70)
print("QPE Analysis: Understanding Phase Wrapping")
print("=" * 70)

# Test matrix with eigenvalues 2 and 4
A = np.array([[3.0, 1.0],
              [1.0, 3.0]])

eigvals, eigvecs = np.linalg.eigh(A)
print(f"\nExact eigenvalues: λ₁={eigvals[0]:.4f}, λ₂={eigvals[1]:.4f}")
print(f"Eigenvectors:")
print(f"  v₁ = {eigvecs[:, 0]}")
print(f"  v₂ = {eigvecs[:, 1]}")

# Test different t values
print("\n" + "=" * 70)
print("Testing different t parameter values")
print("=" * 70)

for t_val in [0.5, 1.0, 1.5, 2.0, 3.0]:
    print(f"\n--- t = {t_val} ---")
    
    # Calculate expected phases
    phase1 = (eigvals[0] * t_val) / (2 * np.pi)
    phase2 = (eigvals[1] * t_val) / (2 * np.pi)
    
    # Wrap to [0, 1)
    phase1_wrapped = phase1 % 1.0
    phase2_wrapped = phase2 % 1.0
    
    print(f"Expected phases (before wrap): ϕ₁={phase1:.4f}, ϕ₂={phase2:.4f}")
    print(f"Expected phases (after wrap):  ϕ₁={phase1_wrapped:.4f}, ϕ₂={phase2_wrapped:.4f}")
    
    # Run QPE with uniform superposition (should see both eigenvalues)
    result = quantum_phase_estimation(A, precision_bits=8, t=t_val, initial_state=None)
    
    print(f"Measured phases: {result.phases[:3]}")  # Top 3
    print(f"Counts: {result.counts[:3]}")
    print(f"Recovered eigenvalues: {result.eigenvalues[:3]}")
    
    # Check if we can distinguish both eigenvalues
    if len(result.eigenvalues) >= 2:
        est_eigvals = np.sort(result.eigenvalues[:2])
        true_eigvals = np.sort(eigvals)
        errors = np.abs(est_eigvals - true_eigvals)
        print(f"Errors: {errors}")
        if np.all(errors < 0.1):
            print("✓ Both eigenvalues correctly resolved!")
        else:
            print("✗ Eigenvalues not well separated")

# Test with eigenvector preparation
print("\n" + "=" * 70)
print("Testing with eigenvector initialization")
print("=" * 70)

for i in [0, 1]:
    print(f"\n--- Initialized in eigenvector {i+1} (λ={eigvals[i]:.4f}) ---")
    
    result = quantum_phase_estimation(
        A, 
        precision_bits=8, 
        t=1.0, 
        initial_state=eigvecs[:, i]
    )
    
    print(f"Measured phases: {result.phases[:2]}")
    print(f"Counts: {result.counts[:2]}")
    print(f"Recovered eigenvalue: {result.eigenvalues[0]:.4f}")
    print(f"Expected eigenvalue: {eigvals[i]:.4f}")
    print(f"Error: {np.abs(result.eigenvalues[0] - eigvals[i]):.6f}")

# Recommendations
print("\n" + "=" * 70)
print("Optimal t Selection")
print("=" * 70)

print(f"""
For eigenvalues λ₁={eigvals[0]:.4f}, λ₂={eigvals[1]:.4f}:

1. To measure both eigenvalues without wrapping:
   - Need: λ_max × t < 2π
   - Condition: {eigvals[1]:.4f} × t < {2*np.pi:.4f}
   - Max t: {2*np.pi/eigvals[1]:.4f}

2. To separate eigenvalues with n={8} bits:
   - Resolution: 2π / 2^{8} = {2*np.pi/256:.6f}
   - Phase separation needed: Δλ × t / (2π) ≥ 1/256
   - Min t: {256/(eigvals[1]-eigvals[0]):.4f}

3. Optimal t (satisfies both):
   - t ∈ [{256/(eigvals[1]-eigvals[0]):.4f}, {2*np.pi/eigvals[1]:.4f}]
   - Recommended: t = {np.pi/eigvals[1]:.4f} (middle of range)
""")

# Test with optimal t
t_optimal = np.pi / eigvals[1]
print(f"Testing with optimal t = {t_optimal:.4f}:")
result = quantum_phase_estimation(A, precision_bits=8, t=t_optimal)
print(f"Measured phases: {result.phases[:3]}")
print(f"Recovered eigenvalues: {result.eigenvalues[:3]}")
est_eigvals = np.sort(result.eigenvalues[:2])
true_eigvals = np.sort(eigvals)
errors = np.abs(est_eigvals - true_eigvals)
print(f"Errors: {errors}")
print(f"Max error: {np.max(errors):.6f}")
