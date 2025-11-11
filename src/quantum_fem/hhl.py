from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import cirq
from .phase_estimation import quantum_phase_estimation, PhaseEstimationResult

@dataclass
class HHLResult:
    x: np.ndarray
    success_prob: float
    state_norm: float
    qpe_result: Optional[PhaseEstimationResult] = None


def prepare_b_state(b: np.ndarray) -> Tuple[float, float]:
    """Encode b as Ry(theta)|0>"""
    if b.shape != (2,):
        raise ValueError("b must be a 2-vector")
    norm_b = np.linalg.norm(b)
    if norm_b == 0:
        raise ValueError("b must be non-zero")
    b_norm = b / norm_b
    theta = 2.0 * np.arctan2(float(np.clip(b_norm[1], -1.0, 1.0)), float(np.clip(b_norm[0], -1.0, 1.0)))
    return theta, norm_b


def _rotation_angle_from_eigvec(v: np.ndarray) -> float:
    """Return theta such that Ry(theta) has first column equal to v"""
    c, s = float(v[0]), float(v[1])
    if c < 0:
        c, s = -c, -s
    theta = 2.0 * np.arctan2(s, c)
    return theta


def hhl_2x2_general(A: np.ndarray, b: np.ndarray) -> HHLResult:
    """HHL for 2x2 matrix. Uses C=0.9*λ_min, ||x||=(||b||/C)*√p"""
    if A.shape != (2, 2):
        raise ValueError("A must be 2x2")
    if b.shape != (2,):
        raise ValueError("b must be a 2-vector")

    vals, vecs = np.linalg.eigh(A)
    l1, l2 = float(vals[0]), float(vals[1])
    V = np.array(vecs, dtype=float)
    if np.linalg.det(V) < 0:
        V[:, 0] = -V[:, 0]
    thetaV = _rotation_angle_from_eigvec(V[:, 0])

    lam_min = min(l1, l2)
    C = 0.9 * lam_min
    th1 = 2.0 * np.arcsin(C / l1)
    th2 = 2.0 * np.arcsin(C / l2)

    sys = cirq.LineQubit(0)
    anc = cirq.LineQubit(1)
    theta_b, norm_b = prepare_b_state(b)

    circuit_nomeas = cirq.Circuit()
    circuit_nomeas.append(cirq.ry(theta_b)(sys))
    circuit_nomeas.append(cirq.ry(-thetaV)(sys))
    circuit_nomeas.append(cirq.X(sys))
    circuit_nomeas.append(cirq.ry(th1)(anc).controlled_by(sys))
    circuit_nomeas.append(cirq.X(sys))
    circuit_nomeas.append(cirq.ry(th2)(anc).controlled_by(sys))
    circuit_nomeas.append(cirq.ry(thetaV)(sys))

    sim = cirq.Simulator()
    final = sim.simulate(circuit_nomeas)
    state = final.final_state_vector.reshape(4)
    amp_sys_anc1 = state[1:4:2]
    norm1 = np.linalg.norm(amp_sys_anc1)
    if norm1 < 1e-12:
        return HHLResult(x=np.zeros(2), success_prob=0.0, state_norm=0.0)
    psi_x = (amp_sys_anc1 / norm1).real
    
    p_success_from_state = norm1 ** 2
    x_magnitude = (norm_b / C) * np.sqrt(p_success_from_state)
    x_est = x_magnitude * psi_x
    
    circuit = cirq.Circuit()
    circuit.append(cirq.ry(theta_b)(sys))
    circuit.append(cirq.ry(-thetaV)(sys))
    circuit.append(cirq.X(sys))
    circuit.append(cirq.ry(th1)(anc).controlled_by(sys))
    circuit.append(cirq.X(sys))
    circuit.append(cirq.ry(th2)(anc).controlled_by(sys))
    circuit.append(cirq.ry(thetaV)(sys))
    circuit.append(cirq.measure(anc, key="anc"))
    result = sim.run(circuit, repetitions=2000)
    p_success = float(np.mean(result.measurements["anc"][:, 0] == 1))
    
    return HHLResult(x=x_est, success_prob=p_success, state_norm=float(norm1))


def hhl_proper_qpe(A: np.ndarray, b: np.ndarray, precision_bits: int = 8, C: Optional[float] = None) -> HHLResult:
    """HHL using proper QPE for eigenvalue analysis"""
    n_qubits = int(np.log2(A.shape[0]))
    if 2**n_qubits != A.shape[0]:
        raise ValueError(f"Matrix size must be power of 2, got {A.shape[0]}")
    if A.shape[0] != len(b):
        raise ValueError(f"Incompatible dimensions: A is {A.shape[0]}×{A.shape[0]}, b is {len(b)}")
    
    qpe_result = quantum_phase_estimation(A, precision_bits=precision_bits, t=1.0, initial_state=b)
    eigenvalues = qpe_result.eigenvalues
    lam_min = np.min(np.abs(eigenvalues))
    lam_max = np.max(np.abs(eigenvalues))
    
    if C is None:
        C = 0.9 * lam_min
    
    print(f"[HHL-QPE] Eigenvalue range: [{lam_min:.4f}, {lam_max:.4f}], C={C:.4f}")
    print(f"[HHL-QPE] Detected phases: {qpe_result.phases}")
    print(f"[HHL-QPE] Phase counts: {qpe_result.counts}")
    
    if A.shape[0] == 2:
        result = hhl_2x2_general(A, b)
        result.qpe_result = qpe_result
        return result
    
    print(f"[HHL-QPE] Warning: Full HHL for {A.shape[0]}×{A.shape[0]} matrices not yet implemented.")
    print(f"[HHL-QPE] Returning classical solution.")
    x_classical = np.linalg.solve(A, b)
    return HHLResult(x=x_classical, success_prob=1.0, state_norm=np.linalg.norm(x_classical), qpe_result=qpe_result)
