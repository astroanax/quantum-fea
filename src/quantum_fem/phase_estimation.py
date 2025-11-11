import numpy as np
import cirq
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class PhaseEstimationResult:
    phases: np.ndarray
    eigenvalues: np.ndarray
    counts: np.ndarray
    precision_bits: int
    t_parameter: float
    measured_phases_all: List[float]


def matrix_to_unitary(A: np.ndarray, t: float = 1.0) -> np.ndarray:
    """U = e^{iAt}"""
    if not np.allclose(A, A.conj().T):
        raise ValueError("A must be Hermitian")
    vals, vecs = np.linalg.eigh(A)
    exp_lambda = np.diag(np.exp(1j * vals * t))
    U = vecs @ exp_lambda @ vecs.conj().T
    return U


def controlled_unitary_power(U: np.ndarray, control: cirq.Qid, targets: List[cirq.Qid], power: int = 1) -> cirq.Operation:
    """Controlled-U^power"""
    U_power = np.linalg.matrix_power(U, power)
    U_check, S, Vh = np.linalg.svd(U_power)
    U_power = U_check @ Vh
    if len(targets) == 1:
        gate = cirq.MatrixGate(U_power)
        return gate(targets[0]).controlled_by(control)
    gate = cirq.MatrixGate(U_power)
    return gate(*targets).controlled_by(control)


def quantum_phase_estimation(matrix: np.ndarray, precision_bits: int = 8, t: Optional[float] = None,
                             initial_state: Optional[np.ndarray] = None, repetitions: int = 2000) -> PhaseEstimationResult:
    """QPE: measures eigenvalues via phase φ where U|ψ⟩ = e^{iλt}|ψ⟩, λ = 2πφ/t"""
    n_target = int(np.log2(matrix.shape[0]))
    if 2**n_target != matrix.shape[0]:
        raise ValueError(f"Matrix size must be power of 2, got {matrix.shape[0]}")
    
    if t is None:
        eigvals = np.linalg.eigvalsh(matrix)
        lam_max = np.max(np.abs(eigvals))
        lam_min = np.min(np.abs(eigvals[np.abs(eigvals) > 1e-10]))
        if lam_max > 1e-10:
            t = (2 * np.pi) / (1.5 * lam_max)
        else:
            t = 1.0
        print(f"[QPE] Auto-selected t={t:.6e} (λ_min={lam_min:.4f}, λ_max={lam_max:.4f})")
        print(f"[QPE] Expected phase range: [{lam_min*t/(2*np.pi):.4f}, {lam_max*t/(2*np.pi):.4f}]")
    
    phase_qubits = [cirq.LineQubit(i) for i in range(precision_bits)]
    target_qubits = [cirq.LineQubit(precision_bits + i) for i in range(n_target)]
    circuit = cirq.Circuit()
    
    if initial_state is not None:
        norm = np.linalg.norm(initial_state)
        if norm < 1e-10:
            raise ValueError("Initial state must be non-zero")
        normalized = initial_state / norm
        if n_target == 1:
            theta = 2 * np.arctan2(abs(normalized[1]), abs(normalized[0]))
            circuit.append(cirq.ry(theta)(target_qubits[0]))
        else:
            circuit.append(cirq.StatePreparationChannel(normalized)(*target_qubits))
    else:
        circuit.append([cirq.H(q) for q in target_qubits])
    
    circuit.append([cirq.H(q) for q in phase_qubits])
    
    U = matrix_to_unitary(matrix, t)
    for k, control_qubit in enumerate(phase_qubits):
        power = 2**k
        controlled_U = controlled_unitary_power(U, control_qubit, target_qubits, power)
        circuit.append(controlled_U)
    
    circuit.append(cirq.qft(*phase_qubits, without_reverse=True, inverse=True))
    circuit.append(cirq.measure(*phase_qubits, key='phase'))
    
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=repetitions)
    measurements = result.measurements['phase']
    
    phases_measured = []
    for meas in measurements:
        binary_value = sum(bit * (2**i) for i, bit in enumerate(reversed(meas)))
        phase = binary_value / (2**precision_bits)
        phases_measured.append(phase)
    
    unique_phases, counts = np.unique(phases_measured, return_counts=True)
    significant_mask = counts > (repetitions * 0.05)
    if np.any(significant_mask):
        unique_phases = unique_phases[significant_mask]
        counts = counts[significant_mask]
    
    sorted_indices = np.argsort(-counts)
    estimated_phases = unique_phases[sorted_indices]
    sorted_counts = counts[sorted_indices]
    estimated_eigenvalues = (2 * np.pi * estimated_phases) / t
    
    return PhaseEstimationResult(phases=estimated_phases, eigenvalues=estimated_eigenvalues,
                                counts=sorted_counts, precision_bits=precision_bits,
                                t_parameter=t, measured_phases_all=phases_measured)


def quantum_phase_estimation_2x2(matrix: np.ndarray, precision_bits: int = 8, t: float = 1.0,
                                 initial_state: Optional[np.ndarray] = None) -> PhaseEstimationResult:
    """Wrapper for 2x2 matrices"""
    if matrix.shape != (2, 2):
        raise ValueError("Use quantum_phase_estimation() for non-2x2 matrices")
    return quantum_phase_estimation(matrix, precision_bits, t, initial_state)
