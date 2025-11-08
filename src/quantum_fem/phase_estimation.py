"""Quantum Phase Estimation for extracting eigenvalues."""

import numpy as np
import cirq
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class PhaseEstimationResult:
    phases: np.ndarray
    eigenvalues: np.ndarray
    phase_register: np.ndarray
    precision_bits: int


def matrix_to_unitary_2x2(A: np.ndarray, t: float = 1.0) -> np.ndarray:
    """Convert 2x2 Hermitian matrix to unitary: U = e^{iAt}."""
    if A.shape != (2, 2):
        raise ValueError("A must be 2x2")
    vals, vecs = np.linalg.eigh(A)
    exp_lambda = np.diag(np.exp(1j * vals * t))
    U = vecs @ exp_lambda @ vecs.conj().T
    return U


def controlled_unitary_2x2(
    matrix: np.ndarray,
    t: float,
    control: cirq.Qid,
    target: cirq.Qid,
    power: int = 1
) -> cirq.Operation:
    """Create controlled-U^power operation for U = e^{iAt}."""
    t_effective = t * power
    U_power = matrix_to_unitary_2x2(matrix, t_effective)
    U_check, S, Vh = np.linalg.svd(U_power)
    U_power = U_check @ Vh
    gate = cirq.MatrixGate(U_power)
    return gate(target).controlled_by(control)


def quantum_phase_estimation_2x2(
    matrix: np.ndarray,
    precision_bits: int = 4,
    t: float = 1.0,
    initial_state: Optional[np.ndarray] = None
) -> PhaseEstimationResult:
    """Perform quantum phase estimation on a 2x2 Hermitian matrix.
    Estimates eigenvalues by encoding as U = e^{iAt} and measuring phase register.
    """
    phase_qubits = [cirq.LineQubit(i) for i in range(precision_bits)]
    eigenstate_qubit = cirq.LineQubit(precision_bits)
    circuit = cirq.Circuit()
    
    if initial_state is not None:
        norm = np.linalg.norm(initial_state)
        normalized = initial_state / norm
        if abs(normalized[0]) > 1e-10:
            theta = 2 * np.arctan2(abs(normalized[1]), abs(normalized[0]))
            circuit.append(cirq.ry(theta)(eigenstate_qubit))
        else:
            circuit.append(cirq.X(eigenstate_qubit))
    else:
        circuit.append(cirq.H(eigenstate_qubit))
    
    circuit.append([cirq.H(q) for q in phase_qubits])
    
    for k, control_qubit in enumerate(phase_qubits):
        power = 2**k
        controlled_U = controlled_unitary_2x2(matrix, t, control_qubit, eigenstate_qubit, power)
        circuit.append(controlled_U)
    
    circuit.append(cirq.qft(*phase_qubits, without_reverse=True, inverse=True))
    circuit.append(cirq.measure(*phase_qubits, key='phase'))
    
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=2000)
    measurements = result.measurements['phase']
    
    phases_measured = []
    for meas in measurements:
        binary_value = sum(bit * (2**i) for i, bit in enumerate(reversed(meas)))
        phase = binary_value / (2**precision_bits)
        phases_measured.append(phase)
    
    unique_phases, counts = np.unique(phases_measured, return_counts=True)
    significant_mask = counts > (len(measurements) * 0.05)
    if np.any(significant_mask):
        unique_phases = unique_phases[significant_mask]
        counts = counts[significant_mask]
    
    top_indices = np.argsort(-counts)[:min(2, len(counts))]
    estimated_phases = unique_phases[top_indices]
    estimated_eigenvalues = (2 * np.pi * estimated_phases) / t
    phase_register_state = unique_phases[top_indices]
    
    return PhaseEstimationResult(
        phases=estimated_phases,
        eigenvalues=estimated_eigenvalues,
        phase_register=phase_register_state,
        precision_bits=precision_bits
    )
