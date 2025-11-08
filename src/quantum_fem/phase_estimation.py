"""Quantum Phase Estimation for extracting eigenvalues."""

import numpy as np
import cirq
from dataclasses import dataclass
from typing import Tuple, Optional, List


@dataclass
class PhaseEstimationResult:
    phases: np.ndarray
    eigenvalues: np.ndarray
    counts: np.ndarray
    precision_bits: int
    t_parameter: float
    measured_phases_all: List[float]


def matrix_to_unitary(A: np.ndarray, t: float = 1.0) -> np.ndarray:
    """Convert Hermitian matrix to unitary: U = e^{iAt}.
    
    Args:
        A: Hermitian matrix (any size)
        t: Time parameter for evolution
    
    Returns:
        U: Unitary matrix
    """
    if not np.allclose(A, A.conj().T):
        raise ValueError("A must be Hermitian")
    
    vals, vecs = np.linalg.eigh(A)
    exp_lambda = np.diag(np.exp(1j * vals * t))
    U = vecs @ exp_lambda @ vecs.conj().T
    return U


def controlled_unitary_power(
    U: np.ndarray,
    control: cirq.Qid,
    targets: List[cirq.Qid],
    power: int = 1
) -> cirq.Operation:
    """Create controlled-U^power operation.
    
    Args:
        U: Unitary matrix (size 2^n × 2^n)
        control: Control qubit
        targets: Target qubits (n qubits)
        power: Power to raise U to
    
    Returns:
        Controlled operation
    """
    U_power = np.linalg.matrix_power(U, power)
    
    # Ensure unitarity (numerical errors can break this)
    U_check, S, Vh = np.linalg.svd(U_power)
    U_power = U_check @ Vh
    
    # For single qubit targets, use standard controlled gate
    if len(targets) == 1:
        gate = cirq.MatrixGate(U_power)
        return gate(targets[0]).controlled_by(control)
    
    # For multi-qubit targets, need to construct controlled gate properly
    # Cirq's MatrixGate handles this automatically
    gate = cirq.MatrixGate(U_power)
    return gate(*targets).controlled_by(control)


def quantum_phase_estimation(
    matrix: np.ndarray,
    precision_bits: int = 8,
    t: Optional[float] = None,
    initial_state: Optional[np.ndarray] = None,
    repetitions: int = 2000
) -> PhaseEstimationResult:
    """Perform quantum phase estimation on a Hermitian matrix.
    
    This is the general QPE algorithm that works for any size Hermitian matrix.
    It encodes eigenvalues as phases: U|ψ⟩ = e^{iλt}|ψ⟩ where U = e^{iAt}.
    
    Args:
        matrix: Hermitian matrix (size 2^n × 2^n)
        precision_bits: Number of qubits in phase register (more = better precision)
        t: Time evolution parameter (if None, auto-selected to avoid phase wrapping)
        initial_state: Initial state vector (if None, uses uniform superposition)
        repetitions: Number of measurement shots
    
    Returns:
        PhaseEstimationResult with estimated phases and eigenvalues
    
    Algorithm:
        1. Prepare |0⟩^⊗n_precision ⊗ |ψ⟩ where |ψ⟩ is eigenvector superposition
        2. Apply Hadamard to phase register
        3. Apply controlled-U^(2^k) for k = 0, 1, ..., n_precision-1
        4. Apply inverse QFT to phase register
        5. Measure phase register to get eigenvalue estimate
        
    Phase wrapping: Choose t such that λ_max * t < 2π to avoid wrapping.
    QPE measures phase ϕ ∈ [0,1), which corresponds to eigenvalue λ = 2πϕ/t.
    """
    # Determine number of target qubits from matrix size
    n_target = int(np.log2(matrix.shape[0]))
    if 2**n_target != matrix.shape[0]:
        raise ValueError(f"Matrix size must be power of 2, got {matrix.shape[0]}")
    
    # Auto-select t to avoid phase wrapping
    if t is None:
        eigvals = np.linalg.eigvalsh(matrix)
        lam_max = np.max(np.abs(eigvals))
        lam_min = np.min(np.abs(eigvals[np.abs(eigvals) > 1e-10]))  # Ignore ~zero eigenvalues
        
        if lam_max > 1e-10:
            # Choose t such that we can distinguish eigenvalues
            # Resolution is 2π/2^n, so we need: (λ_max - λ_min)*t / (2π) ≥ 1/2^n
            # For safety, use: t = 2π / (1.5 * λ_max) which gives max phase ~0.67
            t = (2 * np.pi) / (1.5 * lam_max)
        else:
            t = 1.0
        print(f"[QPE] Auto-selected t={t:.6f} (λ_min={lam_min:.4f}, λ_max={lam_max:.4f})")
        print(f"[QPE] Expected phase range: [{lam_min*t/(2*np.pi):.4f}, {lam_max*t/(2*np.pi):.4f}]")
    
    # Create qubits
    phase_qubits = [cirq.LineQubit(i) for i in range(precision_bits)]
    target_qubits = [cirq.LineQubit(precision_bits + i) for i in range(n_target)]
    
    # Build circuit
    circuit = cirq.Circuit()
    
    # Step 1: Initialize target register with |ψ⟩
    if initial_state is not None:
        norm = np.linalg.norm(initial_state)
        if norm < 1e-10:
            raise ValueError("Initial state must be non-zero")
        normalized = initial_state / norm
        
        # For 2-level system (1 qubit), use Ry gate
        if n_target == 1:
            theta = 2 * np.arctan2(abs(normalized[1]), abs(normalized[0]))
            circuit.append(cirq.ry(theta)(target_qubits[0]))
        else:
            # For larger systems, use state preparation
            # This is a simplified approach - could be optimized
            circuit.append(cirq.StatePreparationChannel(normalized)(*target_qubits))
    else:
        # Uniform superposition over all eigenstates
        circuit.append([cirq.H(q) for q in target_qubits])
    
    # Step 2: Apply Hadamard to phase register
    circuit.append([cirq.H(q) for q in phase_qubits])
    
    # Step 3: Apply controlled-U^(2^k) operations
    U = matrix_to_unitary(matrix, t)
    
    for k, control_qubit in enumerate(phase_qubits):
        power = 2**k
        controlled_U = controlled_unitary_power(U, control_qubit, target_qubits, power)
        circuit.append(controlled_U)
    
    # Step 4: Apply inverse QFT to phase register
    circuit.append(cirq.qft(*phase_qubits, without_reverse=True, inverse=True))
    
    # Step 5: Measure phase register
    circuit.append(cirq.measure(*phase_qubits, key='phase'))
    
    # Execute circuit
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=repetitions)
    measurements = result.measurements['phase']
    
    # Convert binary measurements to phases
    phases_measured = []
    for meas in measurements:
        # Convert binary to decimal (reversed because of qubit ordering)
        binary_value = sum(bit * (2**i) for i, bit in enumerate(reversed(meas)))
        phase = binary_value / (2**precision_bits)
        phases_measured.append(phase)
    
    # Count occurrences of each phase
    unique_phases, counts = np.unique(phases_measured, return_counts=True)
    
    # Filter out noise (keep phases that appear in >5% of shots)
    significant_mask = counts > (repetitions * 0.05)
    if np.any(significant_mask):
        unique_phases = unique_phases[significant_mask]
        counts = counts[significant_mask]
    
    # Sort by count (most common first)
    sorted_indices = np.argsort(-counts)
    estimated_phases = unique_phases[sorted_indices]
    sorted_counts = counts[sorted_indices]
    
    # Convert phases to eigenvalues: λ = 2π·phase / t
    estimated_eigenvalues = (2 * np.pi * estimated_phases) / t
    
    return PhaseEstimationResult(
        phases=estimated_phases,
        eigenvalues=estimated_eigenvalues,
        counts=sorted_counts,
        precision_bits=precision_bits,
        t_parameter=t,
        measured_phases_all=phases_measured
    )


def quantum_phase_estimation_2x2(
    matrix: np.ndarray,
    precision_bits: int = 8,
    t: float = 1.0,
    initial_state: Optional[np.ndarray] = None
) -> PhaseEstimationResult:
    """Wrapper for 2x2 matrices for backward compatibility."""
    if matrix.shape != (2, 2):
        raise ValueError("Use quantum_phase_estimation() for non-2x2 matrices")
    return quantum_phase_estimation(matrix, precision_bits, t, initial_state)
