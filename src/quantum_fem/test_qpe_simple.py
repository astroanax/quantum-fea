"""Simplest possible QPE test."""
import numpy as np
import cirq

# Matrix with eigenvalues [1, 3]
A = np.array([[2.0, -1.0], [-1.0, 2.0]])
vals, vecs = np.linalg.eigh(A)

print(f"Eigenvalues: {vals}")
print(f"Eigenvectors:\n{vecs}")
print(f"v0 (λ=1): {vecs[:, 0]}")
print(f"v1 (λ=3): {vecs[:, 1]}")

# Create U = e^{iAt}
t = 1.0
vals_U = np.exp(1j * vals * t)
U = vecs @ np.diag(vals_U) @ vecs.T

print(f"\nU eigenvalues: {vals_U}")
print(f"Expected phases (θ=λt/2π): {vals * t / (2*np.pi)}")

# Simple QPE: use |+> which is actually an eigenvector!
# For A = [[2,-1],[-1,2]], eigenvector for λ=1 is [-1/√2, -1/√2] = |->
# eigenvector for λ=3 is [-1/√2, 1/√2]

# Test with |+> = [1/√2, 1/√2]
# This is a superposition of both eigenvectors

# Actually, let's prepare exactly v0
theta = 2 * np.arctan2(abs(vecs[1, 0]), abs(vecs[0, 0]))
print(f"\nAngle to prepare v0: θ = {theta} rad = {np.degrees(theta)} deg")
print(f"Expected: 90 deg for |-> state")

# Build circuit
n_bits = 4
phase_qubits = [cirq.LineQubit(i) for i in range(n_bits)]
eigen_qubit = cirq.LineQubit(n_bits)

circuit = cirq.Circuit()

# Prepare eigenstate |+> (v0 for λ=1)
circuit.append(cirq.H(eigen_qubit))  # |+> = (|0> + |1>)/√2

# Initialize phase register
circuit.append([cirq.H(q) for q in phase_qubits])

# Controlled-U^{2^k}
for k in range(n_bits):
    power = 2**k
    U_pow = np.linalg.matrix_power(U, power)
    
    # Check if still unitary
    unitarity = np.linalg.norm(U_pow @ U_pow.conj().T - np.eye(2))
    if unitarity > 1e-6:
        print(f"WARNING: U^{power} not unitary, error={unitarity:.2e}")
    
    gate = cirq.MatrixGate(U_pow)
    circuit.append(gate(eigen_qubit).controlled_by(phase_qubits[k]))

# Inverse QFT
circuit.append(cirq.qft(*phase_qubits, without_reverse=True, inverse=True))

# Measure
circuit.append(cirq.measure(*phase_qubits, key='phase'))

print("\nRunning QPE...")
sim = cirq.Simulator()
result = sim.run(circuit, repetitions=1000)

# Analyze
measurements = result.measurements['phase']
phases = []
for meas in measurements:
    binary = sum(bit * (2**i) for i, bit in enumerate(reversed(meas)))
    phase = binary / (2**n_bits)
    phases.append(phase)

unique, counts = np.unique(phases, return_counts=True)

print(f"\nResults:")
for ph, cnt in zip(unique, counts):
    if cnt > 20:  # Only show significant
        eigenval = (2 * np.pi * ph) / t
        print(f"  Phase={ph:.4f} ({cnt:4d} counts) → λ={eigenval:.4f}")

print(f"\nExpected phase for λ=1: {1 * t / (2*np.pi):.4f}")
print(f"Closest quantized value (4 bits): {round(16 * 1 * t / (2*np.pi)) / 16:.4f}")
