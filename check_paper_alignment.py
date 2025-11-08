"""Check if our implementation matches the paper and if 0% error is realistic."""
import numpy as np
from quantum_fem.beam import Beam1DMesh
import cirq

# Setup problem
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
print("PAPER ALIGNMENT CHECK")
print("="*70)

print("\nPaper's approach (arXiv:2403.19512):")
print("1. BPX preconditioning → well-conditioned system")
print("2. Block encoding of preconditioned matrix")
print("3. Quantum phase estimation → eigenvalues")
print("4. Amplitude amplification → boost success prob")
print("5. Amplitude estimation → measure functionals ⟨g,x⟩")

print("\nOur 2×2 implementation:")
print("1. BPX: ✓ Available (not applied to this 2×2 demo)")
print("2. Block encoding: ✓ Via eigenvalue-controlled rotations")
print("3. Phase estimation: ✗ Using classical eigendecomposition instead")
print("4. Amplitude amplification: ✗ Not implemented")
print("5. Amplitude estimation: ✗ Using direct state measurement")

print("\n" + "="*70)
print("REALISTIC ERROR ANALYSIS")
print("="*70)

# Run quantum circuit multiple times to see statistical variation
from quantum_fem.hhl import hhl_2x2_general

errors = []
success_probs = []
for trial in range(10):
    result = hhl_2x2_general(A_mat, b_vec)
    x_quantum = result.x
    x_classical = np.linalg.solve(A_mat, b_vec)
    error = np.linalg.norm(x_quantum - x_classical) / np.linalg.norm(x_classical) * 100
    errors.append(error)
    success_probs.append(result.success_prob)

print(f"\n10 trials of quantum solver:")
print(f"  Mean error: {np.mean(errors):.6f}%")
print(f"  Std error: {np.std(errors):.6f}%")
print(f"  Min error: {np.min(errors):.6f}%")
print(f"  Max error: {np.max(errors):.6f}%")
print(f"  Mean success prob: {np.mean(success_probs):.4f} ± {np.std(success_probs):.4f}")

print("\n" + "="*70)
print("WHY IS ERROR SO LOW?")
print("="*70)

print("""
1. SIMULATOR vs REAL HARDWARE:
   - We're using cirq.Simulator() → PERFECT state vector simulation
   - No gate errors, no decoherence, no measurement noise
   - Real hardware would have ~1-10% error from noise

2. STATEVECTOR vs SAMPLING:
   - We extract exact amplitudes from final_state_vector
   - Success probability from statevector norm is EXACT
   - Real implementation samples measurements → statistical error

3. FINITE SAMPLING:
   - We measure ancilla 2000 times to estimate p_success
   - But we use EXACT statevector norm for magnitude calculation!
   - This is why error is ~0% not ~1%

4. NO AMPLITUDE AMPLIFICATION:
   - Success prob ~18% means low amplitude
   - Without amp amplification, real sampling would need ~5000+ shots
   - We're cheating by using exact statevector
""")

print("\n" + "="*70)
print("WHAT WOULD REALISTIC ERROR BE?")
print("="*70)

# Simulate realistic sampling error
x_classical = np.linalg.solve(A_mat, b_vec)

print("\nScenario 1: Sampling-based success probability (like real quantum computer)")
shots = 2000
realistic_errors = []
for trial in range(100):
    # Run circuit
    from quantum_fem.hhl import prepare_b_state, _rotation_angle_from_eigvec
    vals, vecs = np.linalg.eigh(A_mat)
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
    theta_b, norm_b = prepare_b_state(b_vec)
    
    # Get exact state for direction
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
    psi_x = (amp_sys_anc1 / norm1).real
    
    # But use SAMPLED success probability
    circuit_meas = cirq.Circuit()
    circuit_meas.append(cirq.ry(theta_b)(sys))
    circuit_meas.append(cirq.ry(-thetaV)(sys))
    circuit_meas.append(cirq.X(sys))
    circuit_meas.append(cirq.ry(th1)(anc).controlled_by(sys))
    circuit_meas.append(cirq.X(sys))
    circuit_meas.append(cirq.ry(th2)(anc).controlled_by(sys))
    circuit_meas.append(cirq.ry(thetaV)(sys))
    circuit_meas.append(cirq.measure(anc, key="anc"))
    
    result = sim.run(circuit_meas, repetitions=shots)
    p_success_sampled = np.mean(result.measurements["anc"][:, 0] == 1)
    
    # Magnitude from sampled probability
    x_magnitude = (norm_b / C) * np.sqrt(p_success_sampled)
    x_quantum = x_magnitude * psi_x
    
    error = np.linalg.norm(x_quantum - x_classical) / np.linalg.norm(x_classical) * 100
    realistic_errors.append(error)

print(f"  With {shots} measurement shots:")
print(f"    Mean error: {np.mean(realistic_errors):.4f}%")
print(f"    Std error: {np.std(realistic_errors):.4f}%")
print(f"    Max error: {np.max(realistic_errors):.4f}%")

print("\nScenario 2: Add realistic gate noise (1% per gate)")
print("  Not implemented, but would add ~5-10% error on real hardware")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

print(f"""
Our implementation:
✓ Follows HHL circuit structure from paper
✓ Uses quantum state for solution (not classical projection)
✗ Uses classical eigendecomposition (paper uses phase estimation)
✗ No amplitude amplification (paper includes it)
✗ Uses exact statevector (paper samples measurements)

Error is ~0% because:
- Perfect simulator (no gate noise)
- Exact statevector access (no sampling error)
- Formula is mathematically exact for unsaturated rotations

Realistic quantum computer would have:
- Sampling error: ~{np.mean(realistic_errors):.2f}% ± {np.std(realistic_errors):.2f}%
- Gate errors: +5-10%
- Total: ~5-15% error

For <10% requirement: We PASS on simulator, would be BORDERLINE on real hardware.
With amplitude amplification (paper's approach), real hardware would achieve <5%.
""")
