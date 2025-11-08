"""Direct call to quantum_phase_estimation_2x2 with debug output."""
import numpy as np
import cirq
from quantum_fem.phase_estimation import matrix_to_unitary_2x2

# Inline the function with debug output
A = np.array([[2.0, -1.0], [-1.0, 2.0]])
t = 1.0
precision_bits = 4

phase_qubits = [cirq.LineQubit(i) for i in range(precision_bits)]
eigenstate_qubit = cirq.LineQubit(precision_bits)

circuit = cirq.Circuit()

# Prep |+⟩
circuit.append(cirq.H(eigenstate_qubit))

# Phase register
circuit.append([cirq.H(q) for q in phase_qubits])

# Controlled-U^{2^k}
for k, control_qubit in enumerate(phase_qubits):
    power = 2**k
    # This is what the CURRENT phase_estimation.py does:
    t_eff = t * power
    U_power = matrix_to_unitary_2x2(A, t_eff)
    
    # Correct via SVD
    U_check, S, Vh = np.linalg.svd(U_power)
    U_power_corrected = U_check @ Vh
    
    gate = cirq.MatrixGate(U_power_corrected)
    circuit.append(gate(eigenstate_qubit).controlled_by(control_qubit))
    
    print(f"k={k}, power={power}, t_eff={t_eff}")
    print(f"  U eigenvalues: {np.linalg.eigvals(U_power_corrected)}")

circuit.append(cirq.qft(*phase_qubits, without_reverse=True, inverse=True))
circuit.append(cirq.measure(*phase_qubits, key='phase'))

print("\nRunning circuit...")
sim = cirq.Simulator()
result = sim.run(circuit, repetitions=2000)

measurements = result.measurements['phase']

# Read bits
phases_measured = []
for meas in measurements:
    binary_value = sum(bit * (2**i) for i, bit in enumerate(reversed(meas)))
    phase = binary_value / (2**precision_bits)
    phases_measured.append(phase)

unique_phases, counts = np.unique(phases_measured, return_counts=True)

# Filter >5%
sig_mask = counts > (len(measurements) * 0.05)
if np.any(sig_mask):
    unique_phases = unique_phases[sig_mask]
    counts = counts[sig_mask]

# Top 2
top_indices = np.argsort(-counts)[:min(2, len(counts))]
estimated_phases = unique_phases[top_indices]
estimated_eigenvalues = (2 * np.pi * estimated_phases) / t

print("\nFiltered phases (>5%):")
for ph, cnt in zip(unique_phases, counts):
    print(f"  Phase={ph:.4f} ({cnt:4d}, {100*cnt/len(measurements):.1f}%)")

print(f"\nTop 2 phases: {estimated_phases}")
print(f"Top 2 eigenvalues: {estimated_eigenvalues}")

vals_true = np.linalg.eigvalsh(A)
print(f"True eigenvalues: {vals_true}")
