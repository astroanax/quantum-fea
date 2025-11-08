"""Debug quantum phase estimation - understand why phase = 0."""
import numpy as np
import cirq
from quantum_fem.phase_estimation import matrix_to_unitary_2x2

def test_qpe_manually():
    """Manually construct and debug QPE circuit."""
    
    # Simple test matrix
    A = np.array([[2.0, -1.0],
                  [-1.0, 2.0]])
    
    print("Matrix A:")
    print(A)
    
    # Eigendecompose
    vals, vecs = np.linalg.eigh(A)
    print(f"\nEigenvalues: {vals}")
    print(f"Eigenvectors:")
    print(vecs)
    
    # Test U = e^{iAt}
    # IMPORTANT: t should be chosen so eigenvalues don't wrap around (mod 2π)
    # For eigenvalues [1, 3], if t=2π then e^{iλt} = e^{i2π}, e^{i6π} = 1, 1 (both identity!)
    # Better choice: t = 1 or t = π/max(λ)
    t = 1.0  # Simple choice
    U = matrix_to_unitary_2x2(A, t)
    
    print(f"\nU = e^{{iAt}} with t={t}:")
    print(U)
    print(f"Is unitary? {np.allclose(U @ U.conj().T, np.eye(2))}")
    
    # Check eigenvalues of U
    # If U = e^{iAt} and A has eigenvalue λ, then U has eigenvalue e^{iλt}
    vals_U, vecs_U = np.linalg.eig(U)
    print(f"\nEigenvalues of U: {vals_U}")
    print(f"Expected: e^{{i λ t}} = {np.exp(1j * vals * t)}")
    
    # Extract phases from U eigenvalues
    # e^{i θ} => θ = angle(eigenvalue)
    phases_U = np.angle(vals_U)  # in [-π, π]
    phases_normalized = phases_U / (2 * np.pi)  # in [-0.5, 0.5]
    phases_normalized[phases_normalized < 0] += 1  # map to [0, 1)
    
    print(f"\nPhases from U eigenvalues: {phases_normalized}")
    print(f"These encode: λ = 2π * phase / t = {2 * np.pi * phases_normalized / t}")
    
    # Now test QPE circuit step by step
    print("\n" + "="*70)
    print("TESTING QPE CIRCUIT")
    print("="*70)
    
    precision_bits = 4
    phase_qubits = [cirq.LineQubit(i) for i in range(precision_bits)]
    eigenstate_qubit = cirq.LineQubit(precision_bits)
    
    circuit = cirq.Circuit()
    
    # Try to prepare eigenstate |v_0⟩ (smallest eigenvalue)
    # For A = [[2,-1],[-1,2]], eigenvalues are [1, 3]
    # Eigenvector for λ=1 is [1/√2, 1/√2]
    # This corresponds to |+⟩ state!
    print("\nPreparing eigenstate |+⟩ (eigenvector of λ=1)")
    circuit.append(cirq.H(eigenstate_qubit))
    
    # Initialize phase register
    circuit.append([cirq.H(q) for q in phase_qubits])
    
    # Apply controlled-U^{2^k}
    print("\nApplying controlled-U^{2^k} operations...")
    for k, control_qubit in enumerate(phase_qubits):
        power = 2**k
        U_power = np.linalg.matrix_power(U, power)
        gate = cirq.MatrixGate(U_power)
        circuit.append(gate(eigenstate_qubit).controlled_by(control_qubit))
        print(f"  k={k}, power={power}")
    
    # Apply inverse QFT
    circuit.append(cirq.qft(*phase_qubits, without_reverse=True, inverse=True))
    
    # Measure
    circuit.append(cirq.measure(*phase_qubits, key='phase'))
    
    print("\nCircuit:")
    print(circuit)
    
    # Run
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=1000)
    
    measurements = result.measurements['phase']
    
    # Convert to phases
    phases_measured = []
    for meas in measurements:
        binary_value = sum(bit * (2**i) for i, bit in enumerate(reversed(meas)))
        phase = binary_value / (2**precision_bits)
        phases_measured.append(phase)
    
    unique_phases, counts = np.unique(phases_measured, return_counts=True)
    
    print(f"\nMeasurement results:")
    for phase, count in zip(unique_phases, counts):
        prob = count / len(measurements)
        eigenvalue = (2 * np.pi * phase) / t
        print(f"  Phase={phase:.4f}, Count={count}, Prob={prob:.3f}, λ={eigenvalue:.4f}")
    
    print(f"\nExpected phase for λ=1: {1.0 * t / (2*np.pi) % 1:.4f}")
    print(f"Expected phase for λ=3: {3.0 * t / (2*np.pi) % 1:.4f}")

if __name__ == "__main__":
    test_qpe_manually()
