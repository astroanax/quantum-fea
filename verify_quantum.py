"""Verify EXACTLY what classical information is being used."""
import numpy as np
from quantum_fem.beam import Beam1DMesh

# Setup the exact same problem
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

w_eig = np.linalg.eigvalsh(K_red)
s = float(np.max(w_eig))
A_mat = K_red / s
b_vec = f_red / s

print("="*70)
print("CLASSICAL INFORMATION AUDIT")
print("="*70)

print("\n1. INPUT (Always classical in any quantum algorithm):")
print(f"   A matrix: {A_mat.tolist()}")
print(f"   b vector: {b_vec.tolist()}")

print("\n2. CIRCUIT DESIGN (Required for HHL on 2×2):")
vals, vecs = np.linalg.eigh(A_mat)
print(f"   Eigenvalues: {vals.tolist()}")
print(f"   Eigenvectors:\n{vecs}")
print(f"   → Used to design rotation angle θ_V for basis transformation")
print(f"   → In phase estimation, this would be quantum!")

print("\n3. SOLUTION RECOVERY:")
x_classical = np.linalg.solve(A_mat, b_vec)
print(f"   Classical solution: {x_classical}")
print(f"   Classical solution norm: {np.linalg.norm(x_classical):.10e}")

print("\n" + "="*70)
print("NOW RUNNING QUANTUM CIRCUIT")
print("="*70)

# Now run the quantum circuit step by step
from quantum_fem.hhl import prepare_b_state, _rotation_angle_from_eigvec
import cirq

l1, l2 = float(vals[0]), float(vals[1])
V = np.array(vecs, dtype=float)
if np.linalg.det(V) < 0:
    V[:, 0] = -V[:, 0]

# This is the classical info used to design the circuit:
thetaV = _rotation_angle_from_eigvec(V[:, 0])
print(f"\nθ_V (basis change angle): {thetaV:.6f} rad")

lam_min = min(l1, l2)
C = 0.9 * lam_min
print(f"C (scaling constant): {C:.6f}")

th1 = 2.0 * np.arcsin(C / l1)
th2 = 2.0 * np.arcsin(C / l2)
print(f"θ_1 (rotation for λ_1): {th1:.6f} rad")
print(f"θ_2 (rotation for λ_2): {th2:.6f} rad")

sys = cirq.LineQubit(0)
anc = cirq.LineQubit(1)
theta_b, norm_b = prepare_b_state(b_vec)
print(f"θ_b (state prep angle): {theta_b:.6f} rad")
print(f"‖b‖: {norm_b:.10e}")

# Build and run circuit
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

print("\n" + "="*70)
print("QUANTUM STATE OUTPUT")
print("="*70)
print(f"Full state: {state}")

amp_sys_anc1 = state[1:4:2]
norm1 = np.linalg.norm(amp_sys_anc1)
psi_x = (amp_sys_anc1 / norm1).real

print(f"Postselected amplitudes (anc=1): {amp_sys_anc1}")
print(f"Success probability: {norm1**2:.10f}")
print(f"Normalized direction ψ_x: {psi_x}")

print("\n" + "="*70)
print("MAGNITUDE RECOVERY (THE CRITICAL PART)")
print("="*70)

# The QUANTUM formula
p_success = norm1**2
x_magnitude_quantum = (norm_b / C) * np.sqrt(p_success)
print(f"\nUsing QUANTUM formula: ‖x‖ = (‖b‖ / C) * √p")
print(f"  ‖b‖ / C = {norm_b:.10e} / {C:.6f} = {norm_b/C:.10e}")
print(f"  √p = √{p_success:.10f} = {np.sqrt(p_success):.10f}")
print(f"  ‖x‖_quantum = {x_magnitude_quantum:.10e}")

x_quantum = x_magnitude_quantum * psi_x
print(f"\nQuantum solution: {x_quantum}")
print(f"Classical solution: {x_classical}")

error = np.linalg.norm(x_quantum - x_classical) / np.linalg.norm(x_classical) * 100
print(f"\nRelative error: {error:.10f}%")

print("\n" + "="*70)
print("CHEATING CHECK")
print("="*70)

print("\nDid we use classical solution x_classical for scaling?")
print("  NO - we used (‖b‖ / C) * √p")

print("\nDid we use eigenvalues for scaling?")
print("  YES for C = 0.9 * λ_min")
print("  BUT eigenvalues are needed to DESIGN the circuit!")
print("  In full quantum algorithm, phase estimation extracts eigenvalues")

print("\nDid we use eigenvectors for scaling?")
print("  NO - eigenvectors only used for basis transformation (θ_V)")
print("  This is circuit design, not solution recovery")

print("\n" + "="*70)
print("VERDICT")
print("="*70)

print("""
For a 2×2 system:
- Eigendecomposition is REQUIRED to design the HHL circuit (get θ_V, θ_1, θ_2)
- This is O(1) cost for 2×2, so acceptable for pedagogical demo
- Solution magnitude comes from QUANTUM measurement: √p
- Solution direction comes from QUANTUM state: ψ_x

For larger systems (n ≥ 4):
- Would use PHASE ESTIMATION to get eigenvalues quantum-mechanically
- Would still need basis transformation (done via Hamiltonian simulation)
- Magnitude recovery formula (‖b‖/C)√p would be the same

CONCLUSION: This IS the quantum way for 2×2!
The eigendecomposition is circuit construction, not solution computation.
""")
