"""Understand eigenvector to qubit state mapping."""
import numpy as np

# Matrix
A = np.array([[2.0, -1.0], [-1.0, 2.0]])
vals, vecs = np.linalg.eigh(A)

print("Eigenvalues:", vals)
print("\nEigenvectors (columns):")
print(vecs)

# Eigenvector for λ=1
v0 = vecs[:, 0]
print(f"\nv0 (λ=1): {v0}")
print(f"  Component 0 (|0⟩): {v0[0]}")
print(f"  Component 1 (|1⟩): {v0[1]}")

# Eigenvector for λ=3  
v1 = vecs[:, 1]
print(f"\nv1 (λ=3): {v1}")
print(f"  Component 0 (|0⟩): {v1[0]}")
print(f"  Component 1 (|1⟩): {v1[1]}")

# Qubit states
print("\n" + "="*60)
print("Standard qubit states:")
print("|0⟩ = [1, 0]")
print("|1⟩ = [0, 1]")
print("|+⟩ = [1/√2, 1/√2] = [0.707, 0.707]")
print("|-⟩ = [1/√2, -1/√2] = [0.707, -0.707]")

print("\n" + "="*60)
print("Our eigenvectors (up to global phase):")
print(f"v0 = [-0.707, -0.707] = -1 * [0.707, 0.707] = -|+⟩")
print(f"v1 = [-0.707,  0.707] = -1 * [0.707, -0.707] = -|-⟩")

print("\nGlobal phase doesn't matter in QM, so:")
print("  v0 ≡ |+⟩ (eigenvalue λ=1)")
print("  v1 ≡ |-⟩ (eigenvalue λ=3)")

# Verify by applying A
print("\n" + "="*60)
print("Verification:")
A_v0 = A @ np.array([1, 1]) / np.sqrt(2)
print(f"A |+⟩ = {A_v0}")
print(f"λ₀ |+⟩ = {vals[0] * np.array([1, 1]) / np.sqrt(2)}")
print(f"Match? {np.allclose(A_v0, vals[0] * np.array([1, 1]) / np.sqrt(2))}")

A_v1 = A @ np.array([1, -1]) / np.sqrt(2)
print(f"\nA |-⟩ = {A_v1}")
print(f"λ₁ |-⟩ = {vals[1] * np.array([1, -1]) / np.sqrt(2)}")
print(f"Match? {np.allclose(A_v1, vals[1] * np.array([1, -1]) / np.sqrt(2))}")
