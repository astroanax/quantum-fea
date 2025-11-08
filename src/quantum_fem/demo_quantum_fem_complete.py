"""Quantum FEM Demo: Cantilever beam with QPE + HHL."""

import numpy as np
from quantum_fem.beam import Beam1DMesh
from quantum_fem.phase_estimation import quantum_phase_estimation_2x2
from quantum_fem.hhl import hhl_2x2_general
import matplotlib.pyplot as plt


def demo_complete_quantum_fem():
    print("="*80)
    print("QUANTUM FEM: Cantilever Beam (QPE + HHL)")
    print("="*80)
    
    L, A, E, P = 1.0, 0.01, 200e9, 1000.0
    I = (A**0.5)**4 / 12.0
    
    print(f"\nBeam: L={L}m, E={E/1e9:.0f}GPa, I={I:.6e}m^4, P={P}N")
    
    beam = Beam1DMesh(length=L, elements=1, E=E, I=I)
    K, f = beam.assemble_stiffness(), beam.load_vector(tip_force=P)
    K_red, f_red, free = beam.apply_cantilever_bc(K, f)
    
    print(f"\nStiffness K:\n{K_red}\nForce f: {f_red}")
    
    u_classical = np.linalg.solve(K_red, f_red)
    u_analytic = P * L**3 / (3 * E * I)
    
    print(f"\nClassical:")
    print(f"  Analytic: {u_analytic:.6e} m")
    print(f"  FEM:      {u_classical[0]:.6e} m ({abs(u_classical[0]-u_analytic)/abs(u_analytic)*100:.2f}% error)")
    
    print("\nRunning QPE (8-bit)...")
    qpe = quantum_phase_estimation_2x2(K_red, precision_bits=8, t=1.0e-7)
    print(f"  QPE eigenvalues: {qpe.eigenvalues}")
    true_eigs = np.linalg.eigvalsh(K_red)
    print(f"  True eigenvalues: {true_eigs}")
    print(f"  Error: {abs(qpe.eigenvalues[0]-true_eigs[0])/true_eigs[0]*100:.2f}%")
    
    print("\nRunning HHL...")
    result = hhl_2x2_general(K_red, f_red)
    x_quantum = result.x
    
    print(f"  Quantum solution: {x_quantum}")
    print(f"  Classical solution: {u_classical}")
    print(f"  Error: {np.linalg.norm(x_quantum-u_classical)/np.linalg.norm(u_classical)*100:.6f}%")
    print(f"  Success probability: {result.success_prob*100:.1f}%")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.scatter([1,2], true_eigs, s=150, marker='o', c='blue', label='True', zorder=3)
    ax1.scatter([1,2], qpe.eigenvalues, s=150, marker='x', c='red', label='QPE', zorder=3, linewidths=2)
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Eigenvalue')
    ax1.set_title('QPE Accuracy')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    methods = ['Analytic', 'Classical', 'Quantum']
    vals = [u_analytic, u_classical[0], x_quantum[0]]
    ax2.bar(methods, vals, color=['green','blue','red'], alpha=0.7)
    ax2.set_ylabel('Tip Displacement (m)')
    ax2.set_title('Solutions')
    ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('quantum_fem_demo.png', dpi=150)
    print("\nSaved: quantum_fem_demo.png")
    
    return {'classical': u_classical, 'quantum': x_quantum, 'qpe': qpe.eigenvalues}


if __name__ == "__main__":
    demo_complete_quantum_fem()
