"""Compare circuits from working vs broken QPE."""
import numpy as np
import cirq
from quantum_fem.phase_estimation import quantum_phase_estimation_2x2, matrix_to_unitary_2x2

# Run through phase_estimation.py function and print circuit
print("CIRCUIT FROM phase_estimation.py")
print("="*70)

A = np.array([[2.0, -1.0], [-1.0, 2.0]])
n_bits = 4
t = 1.0

# Manually build what the function does
phase_qubits = [cirq.LineQubit(i) for i in range(n_bits)]
eigenstate_qubit = cirq.LineQubit(n_bits)

circuit_func = cirq.Circuit()

# |+⟩ prep (initial_state=None)
circuit_func.append(cirq.H(eigenstate_qubit))

# Phase register to |+⟩^n
circuit_func.append([cirq.H(q) for q in phase_qubits])

# Controlled-U^{2^k}
for k, control_qubit in enumerate(phase_qubits):
    power = 2**k
    t_eff = t * power  # This is what the function does now
    U_power = matrix_to_unitary_2x2(A, t_eff)
    
    # Check 
    U_check, S, Vh = np.linalg.svd(U_power)
    U_power_corrected = U_check @ Vh
    
    gate = cirq.MatrixGate(U_power_corrected)
    circuit_func.append(gate(eigenstate_qubit).controlled_by(control_qubit))

# QFT
circuit_func.append(cirq.qft(*phase_qubits, without_reverse=True, inverse=True))
circuit_func.append(cirq.measure(*phase_qubits, key='phase'))

print(circuit_func)
print()

# Run it
sim = cirq.Simulator()
result_func = sim.run(circuit_func, repetitions=1000)

phases_func = []
for meas in result_func.measurements['phase']:
    phase = sum(bit * (0.5 ** (i+1)) for i, bit in enumerate(meas))
    phases_func.append(phase)

unique_func, counts_func = np.unique(phases_func, return_counts=True)
print("Top results:")
top = np.argsort(-counts_func)[:3]
for idx in top:
    print(f"  Phase={unique_func[idx]:.4f} ({counts_func[idx]:4d}) → λ={(2*np.pi*unique_func[idx])/t:.4f}")

print("\n" + "="*70)
print("CIRCUIT FROM test_qpe_simple.py")
print("="*70)

# The working version
U_base = matrix_to_unitary_2x2(A, t)

circuit_working = cirq.Circuit()
circuit_working.append(cirq.H(eigenstate_qubit))
circuit_working.append([cirq.H(q) for q in phase_qubits])

for k in range(n_bits):
    # THIS IS THE KEY DIFFERENCE!
    # Working version: U^{2^k} = (U)^{2^k}
    # My version: U^{2^k} = e^{i A t_{eff}} where t_eff = t * 2^k
    # These should be the same! e^{iA(t*2^k)} = (e^{iAt})^{2^k}
    
    U_working_power = np.linalg.matrix_power(U_base, 2**k)
    gate_working = cirq.MatrixGate(U_working_power)
    circuit_working.append(gate_working(eigenstate_qubit).controlled_by(phase_qubits[k]))

circuit_working.append(cirq.qft(*phase_qubits, without_reverse=True, inverse=True))
circuit_working.append(cirq.measure(*phase_qubits, key='phase'))

print(circuit_working)
print()

result_working = sim.run(circuit_working, repetitions=1000)

phases_working = []
for meas in result_working.measurements['phase']:
    binary = sum(bit * (2**i) for i, bit in enumerate(reversed(meas)))
    phase = binary / (2**n_bits)
    phases_working.append(phase)

unique_working, counts_working = np.unique(phases_working, return_counts=True)
print("Top results:")
top = np.argsort(-counts_working)[:3]
for idx in top:
    print(f"  Phase={unique_working[idx]:.4f} ({counts_working[idx]:4d}) → λ={(2*np.pi*unique_working[idx])/t:.4f}")

print("\n" + "="*70)
print("Expected: λ=1 (phase=0.1592), λ=3 (phase=0.4775)")
