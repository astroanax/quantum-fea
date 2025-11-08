"""Quantum FEM Demo: Cantilever beam with proper QPE + HHL."""

import numpy as np
import argparse
from quantum_fem.beam import Beam1DMesh
from quantum_fem.phase_estimation import quantum_phase_estimation
from quantum_fem.hhl import hhl_2x2_general
from quantum_fem.io_vtk import write_beam_3d_vtk
import matplotlib.pyplot as plt


def demo_complete_quantum_fem(force_kN=10.0, length=1.0, width=0.1, n_elements=20):
    print("="*80)
    print("QUANTUM FEM: Cantilever Beam (QPE + HHL)")
    print("="*80)
    
    # Steel cantilever beam parameters
    L = length
    I = width**4 / 12.0
    E = 200e9
    P = -force_kN * 1000.0  # Convert kN to N, negative for downward
    
    print(f"\nSteel Beam Properties:")
    print(f"  Length: {L} m")
    print(f"  Cross-section: {width}m × {width}m (square)")
    print(f"  Area: {width*width} m²")
    print(f"  Moment of inertia: {I:.6e} m⁴")
    print(f"  Young's modulus: {E/1e9:.0f} GPa")
    print(f"  Applied force: {force_kN:.1f} kN (downward)")
    
    beam = Beam1DMesh(length=L, elements=1, E=E, I=I)
    K, f = beam.assemble_stiffness(), beam.load_vector(tip_force=P)
    K_red, f_red, free = beam.apply_cantilever_bc(K, f)
    
    print(f"\nStiffness K:\n{K_red}\nForce f: {f_red}")
    
    u_classical = np.linalg.solve(K_red, f_red)
    u_analytic = P * L**3 / (3 * E * I)
    
    print(f"\nClassical:")
    print(f"  Analytic: {u_analytic:.6e} m")
    print(f"  FEM:      {u_classical[0]:.6e} m ({abs(u_classical[0]-u_analytic)/abs(u_analytic)*100:.2f}% error)")
    
    print("\nRunning Proper QPE (8-bit precision)...")
    qpe = quantum_phase_estimation(K_red, precision_bits=8)  # Auto t-selection
    print(f"  t parameter: {qpe.t_parameter:.6e}")
    print(f"  QPE eigenvalues: {qpe.eigenvalues[:2]}")
    true_eigs = np.linalg.eigvalsh(K_red)
    print(f"  True eigenvalues: {true_eigs}")
    
    # Find best match for each eigenvalue
    if len(qpe.eigenvalues) >= 2:
        qpe_sorted = np.sort(qpe.eigenvalues[:2])
        errors = np.abs(qpe_sorted - true_eigs)
        print(f"  Errors: {errors}")
        print(f"  Max error: {np.max(errors)/np.max(true_eigs)*100:.2f}%")
    
    print("\nRunning HHL...")
    result = hhl_2x2_general(K_red, f_red)
    x_quantum = result.x
    
    print(f"  Quantum solution: {x_quantum}")
    print(f"  Classical solution: {u_classical}")
    print(f"  Error: {np.linalg.norm(x_quantum-u_classical)/np.linalg.norm(u_classical)*100:.6f}%")
    print(f"  Success probability: {result.success_prob*100:.1f}%")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot QPE vs true eigenvalues (top 2 measured)
    n_eigs_to_plot = min(2, len(qpe.eigenvalues))
    ax1.scatter(range(1, 3), true_eigs, s=150, marker='o', c='blue', label='True', zorder=3)
    ax1.scatter(range(1, n_eigs_to_plot+1), qpe.eigenvalues[:n_eigs_to_plot], 
                s=150, marker='x', c='red', label='QPE', zorder=3, linewidths=2)
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Eigenvalue')
    ax1.set_title(f'Proper QPE Accuracy ({qpe.precision_bits}-bit)')
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
    
    # Export 3D VTK files for ParaView
    print("\nExporting 3D VTK files for ParaView...")
    
    # Reconstruct full solution vectors
    u_full_classical = beam.reconstruct_full(u_classical, free)
    u_full_quantum = beam.reconstruct_full(x_quantum, free)
    
    # Extract displacements and rotations from DOF array [w0, θ0, w1, θ1, ...]
    n_nodes_1d = beam.Nn
    displacements_classical_1d = u_full_classical[0::2]  # Every even index: w
    rotations_classical_1d = u_full_classical[1::2]       # Every odd index: θ
    
    displacements_quantum_1d = u_full_quantum[0::2]
    rotations_quantum_1d = u_full_quantum[1::2]
    
    # Write 3D beam VTK files
    write_beam_3d_vtk('classical_solution.vtk', L, width, beam.elements, 
                      displacements_classical_1d, rotations_classical_1d)
    write_beam_3d_vtk('quantum_solution.vtk', L, width, beam.elements,
                      displacements_quantum_1d, rotations_quantum_1d)
    
    # Also create detailed version with more elements
    beam_detailed = Beam1DMesh(length=L, elements=n_elements, E=E, I=I)
    K_det = beam_detailed.assemble_stiffness()
    f_det = beam_detailed.load_vector(tip_force=P)
    K_det_red, f_det_red, free_det = beam_detailed.apply_cantilever_bc(K_det, f_det)
    u_det = np.linalg.solve(K_det_red, f_det_red)
    u_full_det = beam_detailed.reconstruct_full(u_det, free_det)
    
    displacements_det = u_full_det[0::2]
    rotations_det = u_full_det[1::2]
    
    write_beam_3d_vtk('classical_solution_detailed.vtk', L, width, n_elements,
                      displacements_det, rotations_det)
    write_beam_3d_vtk('quantum_solution_detailed.vtk', L, width, n_elements,
                      displacements_det, rotations_det)
    
    print("  ✓ classical_solution.vtk (3D volumetric mesh, 1 element)")
    print("  ✓ quantum_solution.vtk (3D volumetric mesh, 1 element)")
    print(f"  ✓ classical_solution_detailed.vtk ({n_elements} elements)")
    print(f"  ✓ quantum_solution_detailed.vtk ({n_elements} elements)")
    print(f"\n  Beam: {L}m × {width}m × {width}m")
    print(f"  Tip deflection: {abs(u_classical[0]):.6e} m = {abs(u_classical[0])*1000:.3f} mm")
    print(f"  Tip rotation: {abs(u_classical[1]):.6e} rad")
    print("\nTo visualize in ParaView:")
    print("  paraview classical_solution_detailed.vtk")
    print("  Then: Apply → Warp By Vector (scale: 100-1000) → Color by displacement_magnitude")
    
    return {'classical': u_classical, 'quantum': x_quantum, 'qpe': qpe.eigenvalues}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quantum FEM Cantilever Beam Solver')
    parser.add_argument('-f', '--force', type=float, default=10.0,
                        help='Applied force in kN (default: 10.0)')
    parser.add_argument('-l', '--length', type=float, default=1.0,
                        help='Beam length in meters (default: 1.0)')
    parser.add_argument('-w', '--width', type=float, default=0.1,
                        help='Beam width in meters (default: 0.1)')
    parser.add_argument('-n', '--elements', type=int, default=20,
                        help='Number of elements for detailed mesh (default: 20)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(f"Running with: Force={args.force}kN, Length={args.length}m, Width={args.width}m")
    print("="*80 + "\n")
    
    demo_complete_quantum_fem(force_kN=args.force, length=args.length, 
                             width=args.width, n_elements=args.elements)
