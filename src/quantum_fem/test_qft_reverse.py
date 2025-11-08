"""Test QFT with and without reverse."""
import numpy as np
import cirq
from quantum_fem.phase_estimation import matrix_to_unitary_2x2

A = np.array([[2.0, -1.0], [-1.0, 2.0]])
t = 1.0
U = matrix_to_unitary_2x2(A, t)

n_bits = 4
phase_qubits = [cirq.LineQubit(i) for i in range(n_bits)]
eigen_qubit = cirq.LineQubit(n_bits)

# Test 1: WITH reverse (without_reverse=False, which is default behavior)
print("="*70)
print("TEST 1: QFT with reverse (standard textbook QPE)")
print("="*70)

circuit1 = cirq.Circuit()
circuit1.append(cirq.H(eigen_qubit))  # |+⟩
circuit1.append([cirq.H(q) for q in phase_qubits])

for k in range(n_bits):
    t_eff = t * (2**k)
    U_pow = matrix_to_unitary_2x2(A, t_eff)
    gate = cirq.MatrixGate(U_pow)
    circuit1.append(gate(eigen_qubit).controlled_by(phase_qubits[k]))

circuit1.append(cirq.qft(*phase_qubits, without_reverse=False, inverse=True))
circuit1.append(cirq.measure(*phase_qubits, key='phase'))

sim = cirq.Simulator()
result1 = sim.run(circuit1, repetitions=1000)

phases1 = []
for meas in result1.measurements['phase']:
    binary = sum(bit * (2**i) for i, bit in enumerate(reversed(meas)))
    phase = binary / (2**n_bits)
    phases1.append(phase)

unique1, counts1 = np.unique(phases1, return_counts=True)
print("Top measurements:")
top_idx = np.argsort(-counts1)[:5]
for idx in top_idx:
    ph = unique1[idx]
    eigenval = (2 * np.pi * ph) / t
    print(f"  Phase={ph:.4f} ({counts1[idx]:4d}) → λ={eigenval:.4f}")

# Test 2: WITHOUT reverse (without_reverse=True)
print("\n" + "="*70)
print("TEST 2: QFT without reverse (cheaper, needs different bit order)")
print("="*70)

circuit2 = cirq.Circuit()
circuit2.append(cirq.H(eigen_qubit))
circuit2.append([cirq.H(q) for q in phase_qubits])

for k in range(n_bits):
    t_eff = t * (2**k)
    U_pow = matrix_to_unitary_2x2(A, t_eff)
    gate = cirq.MatrixGate(U_pow)
    circuit2.append(gate(eigen_qubit).controlled_by(phase_qubits[k]))

circuit2.append(cirq.qft(*phase_qubits, without_reverse=True, inverse=True))
circuit2.append(cirq.measure(*phase_qubits, key='phase'))

result2 = sim.run(circuit2, repetitions=1000)

phases2 = []
for meas in result2.measurements['phase']:
    binary = sum(bit * (2**i) for i, bit in enumerate(reversed(meas)))
    phase = binary / (2**n_bits)
    phases2.append(phase)

unique2, counts2 = np.unique(phases2, return_counts=True)
print("Top measurements:")
top_idx2 = np.argsort(-counts2)[:5]
for idx in top_idx2:
    ph = unique2[idx]
    eigenval = (2 * np.pi * ph) / t
    print(f"  Phase={ph:.4f} ({counts2[idx]:4d}) → λ={eigenval:.4f}")

print("\n" + "="*70)
print(f"Expected: λ=1.0 (phase=0.1592) and λ=3.0 (phase=0.4775)")
print("="*70)
