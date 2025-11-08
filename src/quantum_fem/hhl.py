from dataclasses import dataclass
from typing import Tuple
import numpy as np
import cirq

@dataclass
class HHLResult:
    x: np.ndarray
    success_prob: float
    state_norm: float


def prepare_b_state(b: np.ndarray) -> Tuple[float, float]:
    """Encode b as Ry(theta)|0>. Returns (theta, norm_b)."""
    if b.shape != (2,):
        raise ValueError("b must be a 2-vector")
    norm_b = np.linalg.norm(b)
    if norm_b == 0:
        raise ValueError("b must be non-zero")
    b_norm = b / norm_b
    b0 = float(np.clip(b_norm[0], -1.0, 1.0))
    b1 = float(np.clip(b_norm[1], -1.0, 1.0))
    theta = 2.0 * np.arctan2(b1, b0)
    return theta, norm_b


def hhl_2x2_A_eq_2I_minus_X(b: np.ndarray) -> HHLResult:
    """HHL for A = 2I - X with eigenvalues {1,3} in X basis."""
    lam_min, lam_max = 1.0, 3.0
    C = 1.0 / lam_max
    theta0 = 2.0 * np.arcsin(np.clip(C / lam_max, 0.0, 1.0))
    theta1 = 2.0 * np.arcsin(np.clip(C / lam_min, 0.0, 1.0))

    sys = cirq.LineQubit(0)
    anc = cirq.LineQubit(1)

    theta_b, norm_b = prepare_b_state(b)
    circuit = cirq.Circuit()
    circuit.append(cirq.ry(theta_b)(sys))
    circuit.append(cirq.H(sys))
    circuit.append(cirq.X(sys))
    circuit.append(cirq.ry(theta0)(anc).controlled_by(sys))
    circuit.append(cirq.X(sys))
    circuit.append(cirq.ry(theta1)(anc).controlled_by(sys))
    circuit.append(cirq.H(sys))
    circuit.append(cirq.measure(anc, key="anc"))

    sim = cirq.Simulator()
    reps = 5000
    result = sim.run(circuit, repetitions=reps)
    anc_meas = np.array(result.measurements["anc"])[:, 0]
    p_success = np.mean(anc_meas == 1)

    circuit_nomeas = cirq.Circuit()
    circuit_nomeas.append(cirq.ry(theta_b)(sys))
    circuit_nomeas.append(cirq.H(sys))
    circuit_nomeas.append(cirq.X(sys))
    circuit_nomeas.append(cirq.ry(theta0)(anc).controlled_by(sys))
    circuit_nomeas.append(cirq.X(sys))
    circuit_nomeas.append(cirq.ry(theta1)(anc).controlled_by(sys))
    circuit_nomeas.append(cirq.H(sys))

    final = sim.simulate(circuit_nomeas)
    state = final.final_state_vector.reshape(4)
    amp_sys_anc1 = state[1:4:2]
    norm1 = np.linalg.norm(amp_sys_anc1)
    if norm1 < 1e-12:
        return HHLResult(x=np.zeros(2), success_prob=0.0, state_norm=0.0)
    psi_x = amp_sys_anc1 / norm1

    scale = 0.0
    if p_success > 0:
        scale = (norm_b / C) * np.sqrt(p_success)
    x_est = scale * psi_x.real

    return HHLResult(x=x_est, success_prob=float(p_success), state_norm=float(norm1))


def _rotation_angle_from_eigvec(v: np.ndarray) -> float:
    """Return theta such that Ry(theta) has first column equal to v."""
    c, s = float(v[0]), float(v[1])
    if c < 0:
        c, s = -c, -s
    theta = 2.0 * np.arctan2(s, c)
    return theta


def hhl_2x2_general(A: np.ndarray, b: np.ndarray) -> HHLResult:
    """HHL for general 2x2 real symmetric positive definite matrix A.
    Uses C = 0.9*λ_min to avoid rotation saturation.
    Magnitude recovery: ||x|| = (||b|| / C) * sqrt(p_success)
    """
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

    lam_max = max(l1, l2)
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
    reps = 2000
    result = sim.run(circuit, repetitions=reps)
    p_success = float(np.mean(result.measurements["anc"][:, 0] == 1))
    
    return HHLResult(x=x_est, success_prob=p_success, state_norm=float(norm1))
