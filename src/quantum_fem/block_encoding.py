import numpy as np
import cirq
from dataclasses import dataclass

"""Block encoding demo for a 4x4 1D Laplacian-like SPD matrix A.

We consider a uniform 1D Dirichlet problem with 4 interior points giving a 4x4 matrix.
Simplified A (scaled) via eigen-decomposition A = V Λ V^T with Λ bounded in [λ_min, λ_max].
We prepare |b>, apply approximate inverse via controlled rotations (multi-eigenvalue extension of small HHL),
not full phase estimation (pedagogical stride from 2x2 to 4x4).
"""

@dataclass
class BlockEncodingResult:
    solution_amplitudes: np.ndarray
    success_prob: float
    raw_state_norm: float
    eigenvalues: np.ndarray


def laplacian_1d_4x4() -> np.ndarray:
    # Classic tridiagonal with 2,-1 off diagonals, size 4
    A = 2 * np.eye(4) - np.diag(np.ones(3), 1) - np.diag(np.ones(3), -1)
    return A


def hhl_like_4x4(A: np.ndarray, b: np.ndarray, reps: int = 4000) -> BlockEncodingResult:
    if A.shape != (4, 4) or b.shape != (4,):
        raise ValueError("Expect A 4x4 and b shape (4,)")
    # Eigendecompose
    vals, vecs = np.linalg.eigh(A)
    # Scale by max eigenvalue for stability
    s = np.max(vals)
    vals_s = vals / s

    # Normalize b
    nb = np.linalg.norm(b)
    if nb == 0:
        raise ValueError("b must be non-zero")
    b_norm = b / nb

    # Expand b in eigenbasis: coefficients c_i = v_i^T b_norm
    coeffs = vecs.T @ b_norm

    # For each eigenvalue λ_i, amplitude for ancilla rotation ~ C / λ_i with C = 1/max(vals_s)
    C = 1.0 / np.max(vals_s)
    angles = [2 * np.arcsin(min(1.0, C / v)) for v in vals_s]

    # We map 4 basis states to two system qubits; need a circuit that prepares eigenbasis superposition with coeffs.
    # Represent basis |i> on 2 qubits. We'll prepare state |psi_b> = Σ c_i |v_i> via explicit unitary U_b whose columns are vecs.
    # For simplicity: use state preparation by initializing vector of length 4 and applying cirq.StatePreparation.
    qubits = cirq.LineQubit.range(2)  # system qubits
    anc = cirq.LineQubit(2)

    prep = cirq.StatePreparationChannel(vecs @ coeffs)
    circuit = cirq.Circuit(prep.on(*qubits))

    # Apply controlled rotations; need projection onto each computational basis state |i>.
    # Implement via multiplexed rotations using X gates pattern.
    for i in range(4):
        angle = angles[i]
        # Control pattern: flip qubits so that |i> maps to |11>, apply controlled rotation, flip back.
        bits = [(i >> k) & 1 for k in range(2)]  # k=0 LSB
        # Flip qubits where bit == 0 so target state becomes |11>
        ops = []
        for k, bval in enumerate(bits):
            if bval == 0:
                ops.append(cirq.X(qubits[k]))
        circuit.append(ops)
        circuit.append(cirq.ry(angle)(anc).controlled_by(*qubits))
        # Flip back
        circuit.append(ops[::-1])

    # Measure ancilla for success probability
    circuit.append(cirq.measure(anc, key="anc"))

    sim = cirq.Simulator()
    result = sim.run(circuit, repetitions=reps)
    anc_meas = np.array(result.measurements["anc"])[:, 0]
    p_success = np.mean(anc_meas == 1)

    # Postselected statevector
    circuit_nomeas = cirq.Circuit(prep.on(*qubits))
    for i in range(4):
        angle = angles[i]
        bits = [(i >> k) & 1 for k in range(2)]
        ops = []
        for k, bval in enumerate(bits):
            if bval == 0:
                ops.append(cirq.X(qubits[k]))
        circuit_nomeas.append(ops)
        circuit_nomeas.append(cirq.ry(angle)(anc).controlled_by(*qubits))
        circuit_nomeas.append(ops[::-1])

    final = sim.simulate(circuit_nomeas)
    state = final.final_state_vector.reshape(8)
    # system basis interleaved with ancilla; ancilla=1 amplitudes indices with bit2=1
    amp_sys_anc1 = state[1::2]
    norm1 = np.linalg.norm(amp_sys_anc1)
    if norm1 < 1e-12:
        return BlockEncodingResult(solution_amplitudes=np.zeros(4), success_prob=0.0, raw_state_norm=0.0, eigenvalues=vals)
    sol_amps = amp_sys_anc1 / norm1

    # Rescale to match classical solution magnitude roughly
    x_classical = np.linalg.solve(A, b)
    scale_back = np.linalg.norm(x_classical) / np.linalg.norm(sol_amps)
    sol = sol_amps.real * scale_back

    return BlockEncodingResult(solution_amplitudes=sol, success_prob=float(p_success), raw_state_norm=float(norm1), eigenvalues=vals)
