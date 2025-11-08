"""Example: Quantum Phase Estimation for eigenvalue extraction."""
import numpy as np
from quantum_fem.phase_estimation import (
    quantum_phase_estimation_2x2,
    matrix_to_unitary_2x2,
    hhl_with_phase_estimation
)

def main():
    print("="*70)
    print("QUANTUM PHASE ESTIMATION DEMO")
    print("="*70)
    
    # Test matrix: simple 2x2 SPD
    A = np.array([[2.0, -1.0],
                  [-1.0, 2.0]])
    
    b = np.array([1.0, 0.0])
    
    print(f"\nMatrix A:")
    print(A)
    print(f"\nVector b: {b}")
    
    # Classical eigenvalues (ground truth)
    vals_true, vecs_true = np.linalg.eigh(A)
    print(f"\nTrue eigenvalues: {vals_true}")
    
    # Test unitary construction
    # IMPORTANT: t should avoid phase wrapping (λ·t should be < 2π for all eigenvalues)
    # For eigenvalues [1, 3], using t=1 gives phases [1, 3] radians → [0.159, 0.477] in units of 2π
    t = 1.0
    U = matrix_to_unitary_2x2(A, t)
    print(f"\nUnitary U = e^{{iAt}} with t={t}:")
    print(f"Is unitary? {np.allclose(U @ U.conj().T, np.eye(2))}")
    
    # Expected phases: θ = (λ·t)/(2π) for eigenvalues λ
    expected_phases = (vals_true * t) / (2 * np.pi)
    print(f"Expected phases (in units of 2π): {expected_phases}")
    print(f"Phase resolution with n bits: 1/{2**4} = {1/(2**4):.4f}, 1/{2**6} = {1/(2**6):.4f}, 1/{2**8} = {1/(2**8):.4f}")
    
    # Run quantum phase estimation
    print("\n" + "="*70)
    print("RUNNING QUANTUM PHASE ESTIMATION")
    print("="*70)
    
    for n_bits in [4, 6, 8]:
        print(f"\n--- Precision: {n_bits} bits (resolution 1/{2**n_bits} = {1/(2**n_bits):.6f}) ---")
        result = quantum_phase_estimation_2x2(A, precision_bits=n_bits, t=t, initial_state=b)
        
        print(f"Estimated phases: {result.phases}")
        print(f"Estimated eigenvalues: {result.eigenvalues}")
        print(f"True eigenvalues: {vals_true}")
        
        # Compute error
        # Match estimated to true eigenvalues
        errors = []
        for est_val in result.eigenvalues:
            min_error = min(abs(est_val - true_val) for true_val in vals_true)
            errors.append(min_error)
        
        mean_error = np.mean(errors)
        print(f"Mean eigenvalue error: {mean_error:.6f}")
        print(f"Relative error: {mean_error / np.mean(vals_true) * 100:.2f}%")
    
    # Now test HHL with phase estimation
    print("\n" + "="*70)
    print("HHL WITH QUANTUM PHASE ESTIMATION")
    print("="*70)
    
    x_solution, prob, pe_result = hhl_with_phase_estimation(A, b, precision_bits=4)
    
    print(f"\nQuantum solution: {x_solution}")
    print(f"Success probability: {prob:.4f}")
    
    # Compare with classical
    x_classical = np.linalg.solve(A, b)
    print(f"Classical solution: {x_classical}")
    
    error = np.linalg.norm(x_solution - x_classical) / np.linalg.norm(x_classical) * 100
    print(f"Solution error: {error:.4f}%")
    
    print("\n" + "="*70)
    print("BEAM PROBLEM WITH QPE")
    print("="*70)
    
    # Apply to beam problem
    from quantum_fem.beam import Beam1DMesh
    
    L = 1.0
    A_beam = 0.01
    a = A_beam ** 0.5
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
    A_scaled = K_red / s
    b_scaled = f_red / s
    
    print(f"\nScaled beam stiffness matrix:")
    print(A_scaled)
    
    # Classical eigenvalues
    vals_beam, _ = np.linalg.eigh(A_scaled)
    print(f"\nTrue eigenvalues: {vals_beam}")
    print(f"Condition number: {vals_beam[1]/vals_beam[0]:.2f}")
    
    # QPE with higher precision for ill-conditioned system
    print("\nRunning QPE with 8 bits precision...")
    # For beam, λ ∈ [0.052, 1.0], so with t=1: phases ∈ [0.008, 0.159]
    # Need good resolution: 8 bits gives 1/256 ≈ 0.004
    pe_beam = quantum_phase_estimation_2x2(A_scaled, precision_bits=8, t=1.0, initial_state=b_scaled)
    
    print(f"Estimated eigenvalues: {pe_beam.eigenvalues}")
    print(f"True eigenvalues: {vals_beam}")
    
    # Compute accuracy
    errors_beam = []
    for est_val in pe_beam.eigenvalues:
        min_error = min(abs(est_val - true_val) for true_val in vals_beam)
        errors_beam.append(min_error)
    
    print(f"Mean eigenvalue error: {np.mean(errors_beam):.6f}")
    print(f"Relative error: {np.mean(errors_beam) / np.mean(vals_beam) * 100:.2f}%")

if __name__ == "__main__":
    main()
