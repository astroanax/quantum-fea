import numpy as np
import cirq
from typing import Callable
from dataclasses import dataclass

"""Estimate a linear functional g^T x where x approximates A^{-1} b using a prepared quantum state |x>.
For demo: treat |x> amplitudes (normalized) as proportional to solution entries; then perform Hadamard test to
estimate inner product with a classical vector g (loaded as amplitudes |g>). This is pedagogical and not a full
amplitude estimation implementation.
"""

@dataclass
class FunctionalEstimate:
    value: float
    samples: int
    raw_expectation: float


def hadamard_test_overlap(x_state: np.ndarray, g_vec: np.ndarray, shots: int = 2000) -> FunctionalEstimate:
    if x_state.shape != g_vec.shape:
        raise ValueError("Shape mismatch between x_state and g_vec")
    n = x_state.size
    # Normalize both
    nx = np.linalg.norm(x_state)
    ng = np.linalg.norm(g_vec)
    if nx == 0 or ng == 0:
        return FunctionalEstimate(0.0, shots, 0.0)
    x_norm = x_state / nx
    g_norm = g_vec / ng

    # We build |psi> = |0>|x> + |1>|g> and use Hadamard on ancilla to estimate Re(<x|g>)
    # For 4 entries we need 2 qubits; generalize by log2(n)
    qubits = cirq.LineQubit.range(int(np.ceil(np.log2(n))) + 1)  # ancilla + system qubits
    anc = qubits[0]
    sys = qubits[1:]

    # Prepare superposition manually by building full state vector
    full_dim = 2 ** len(sys)
    if full_dim != n:
        raise ValueError("n must be a power of two for this demo")
    state_plus = np.zeros(2 * full_dim, dtype=complex)
    # |0>|x>
    state_plus[0:full_dim] = x_norm
    # |1>|g>
    state_plus[full_dim:2*full_dim] = g_norm
    # Normalize global state
    state_plus /= np.linalg.norm(state_plus)

    prep = cirq.StatePreparationChannel(state_plus)
    circuit = cirq.Circuit(prep.on(*qubits))
    circuit.append(cirq.H(anc))
    circuit.append(cirq.measure(anc, key="m"))

    sim = cirq.Simulator()
    result = sim.run(circuit, repetitions=shots)
    meas = result.measurements["m"][:, 0]
    # P(0) - P(1) = Re(<x|g>) after proper normalization factor 2; approximate scaling here
    p0 = np.mean(meas == 0)
    expectation = 2 * p0 - 1
    # Recover functional estimate scaled by norms
    estimate = expectation * nx * ng
    return FunctionalEstimate(value=float(estimate), samples=shots, raw_expectation=float(expectation))
