"""Debug script to understand what the quantum circuit actually produces."""
import numpy as np
from quantum_fem.beam import Beam1DMesh

# Setup beam problem
L = 1.0
A = 0.01
a = A ** 0.5
I = a**4 / 12.0
E = 200e9
P = 1000.0

beam = Beam1DMesh(length=L, elements=1, E=E, I=I)
K = beam.assemble_stiffness()
f = beam.load_vector(tip_force=P)
K_red, f_red, free = beam.apply_cantilever_bc(K, f)

# Scale system
w_eig = np.linalg.eigvalsh(K_red)
s = float(np.max(w_eig))
A_mat = K_red / s
b_vec = f_red / s

print("Scaled system:")
print(f"A = \n{A_mat}")
print(f"b = {b_vec}")
print(f"scaling factor s = {s:.3e}")

# Classical solution
x_classical = np.linalg.solve(A_mat, b_vec)
print(f"\nClassical solution x = {x_classical}")
print(f"||x|| = {np.linalg.norm(x_classical):.6e}")

# Eigendecomposition
vals, vecs = np.linalg.eigh(A_mat)
print(f"\nEigenvalues: {vals}")
print(f"Condition number: {vals[1]/vals[0]:.2f}")

# What HHL produces
from quantum_fem.hhl import prepare_b_state, _rotation_angle_from_eigvec
import cirq

l1, l2 = float(vals[0]), float(vals[1])
V = np.array(vecs, dtype=float)
if np.linalg.det(V) < 0:
    V[:, 0] = -V[:, 0]
thetaV = _rotation_angle_from_eigvec(V[:, 0])
lam_max = max(l1, l2)
lam_min = min(l1, l2)
C_max_allowed = 0.9 * lam_min
C = C_max_allowed
th1 = 2.0 * np.arcsin(C / l1)
th2 = 2.0 * np.arcsin(C / l2)

sys = cirq.LineQubit(0)
anc = cirq.LineQubit(1)
theta_b, norm_b = prepare_b_state(b_vec)

circuit = cirq.Circuit()
circuit.append(cirq.ry(theta_b)(sys))
circuit.append(cirq.ry(-thetaV)(sys))
circuit.append(cirq.X(sys))
circuit.append(cirq.ry(th1)(anc).controlled_by(sys))
circuit.append(cirq.X(sys))
circuit.append(cirq.ry(th2)(anc).controlled_by(sys))
circuit.append(cirq.ry(thetaV)(sys))

sim = cirq.Simulator()
final = sim.simulate(circuit)
state = final.final_state_vector.reshape(4)

print(f"\nFull quantum state (|00>, |01>, |10>, |11>):")
print(state)

amp_sys_anc1 = state[1:4:2]
norm1 = np.linalg.norm(amp_sys_anc1)
psi_x = (amp_sys_anc1 / norm1).real

print(f"\nPostselected state (anc=1): {amp_sys_anc1}")
print(f"Norm of anc=1 component: {norm1:.6f}")
print(f"Normalized psi_x: {psi_x}")
print(f"Success probability: {norm1**2:.6f}")

print(f"\nC = 1/λ_max = {C:.6f}")
print(f"C/λ_1 = {C/l1:.6f}")
print(f"C/λ_2 = {C/l2:.6f}")

# Theoretical analysis
c_i = vecs.T @ (b_vec / norm_b)  # coefficients of b in eigenbasis
print(f"\nEigenbasis coefficients of b: {c_i}")

# After HHL, amplitudes should be proportional to c_i * (C/λ_i)
expected_amplitudes = c_i * np.array([C/l1, C/l2])
print(f"Expected HHL amplitudes (unnormalized): {expected_amplitudes}")
expected_norm = np.linalg.norm(expected_amplitudes)
print(f"Expected norm: {expected_norm:.6f}")
print(f"Expected success prob: {expected_norm**2:.6f}")

# The solution in eigenbasis is c_i / λ_i
solution_eigenbasis = c_i / vals
print(f"\nSolution in eigenbasis (c_i/λ_i): {solution_eigenbasis}")
solution_computational = vecs @ solution_eigenbasis
print(f"Solution in computational basis: {solution_computational}")
print(f"Matches classical? {np.allclose(solution_computational, x_classical)}")

# Quantum state (normalized) should match direction of solution
direction_quantum = psi_x / np.linalg.norm(psi_x)
direction_classical = x_classical / np.linalg.norm(x_classical)
print(f"\nQuantum direction: {direction_quantum}")
print(f"Classical direction: {direction_classical}")
print(f"Directions match? {np.allclose(direction_quantum, direction_classical)}")

# Now the KEY question: can we recover magnitude from success probability alone?
print(f"\n{'='*60}")
print("MAGNITUDE RECOVERY ANALYSIS")
print('='*60)

# Theoretical relationship:
# success_prob = ||Σ c_i * (C/λ_i) |v_i>||^2
# But we want ||Σ c_i / λ_i |v_i>||^2 = ||x||^2

# The ratio is (C)^2, so:
# ||x|| = (1/C) * sqrt(success_prob)  ???

predicted_norm_v1 = (1.0 / C) * np.sqrt(norm1**2)
print(f"Predicted ||x|| using (1/C)*sqrt(p_success): {predicted_norm_v1:.6e}")
print(f"Actual ||x||: {np.linalg.norm(x_classical):.6e}")
print(f"Ratio: {predicted_norm_v1 / np.linalg.norm(x_classical):.3f}")

# But wait - we normalized b, so need to account for that too
predicted_norm_v2 = (norm_b / C) * np.sqrt(norm1**2)
print(f"\nPredicted ||x|| using (norm_b/C)*sqrt(p_success): {predicted_norm_v2:.6e}")
print(f"Actual ||x||: {np.linalg.norm(x_classical):.6e}")
print(f"Ratio: {predicted_norm_v2 / np.linalg.norm(x_classical):.3f}")

# Try reconstructing
x_quantum_v1 = predicted_norm_v1 * psi_x
x_quantum_v2 = predicted_norm_v2 * psi_x

print(f"\nQuantum solution v1: {x_quantum_v1}")
print(f"Error v1: {np.linalg.norm(x_quantum_v1 - x_classical) / np.linalg.norm(x_classical) * 100:.2f}%")

print(f"\nQuantum solution v2: {x_quantum_v2}")
print(f"Error v2: {np.linalg.norm(x_quantum_v2 - x_classical) / np.linalg.norm(x_classical) * 100:.2f}%")
