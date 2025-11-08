import argparse
import numpy as np
from quantum_fem.beam import Beam1DMesh, classical_solve
from quantum_fem.io_vtk import write_vtp_polyline, write_vtp_beam_thick
from quantum_fem.hhl import hhl_2x2_general, hhl_2x2_general_calibrated

"""Cantilever beam (Euler-Bernoulli) with end load.
- Fixed at x=0: w=0, theta=0
- Tip load P at x=L (vertical), optional moment M
- Material/geometry set for a steel beam with square cross-section.
Outputs a PolyData .vtp polyline for ParaView with deflected shape.
"""

def main():
    parser = argparse.ArgumentParser(description="Cantilever beam (Euler-Bernoulli) deflection and ParaView export")
    parser.add_argument("--length", type=float, default=1.0, help="Beam length L [m]")
    parser.add_argument("--area", type=float, default=0.01, help="Square cross-section area A [m^2] (I=a^4/12, a=sqrt(A))")
    parser.add_argument("--E", type=float, default=200e9, help="Young's modulus [Pa]")
    parser.add_argument("--load", type=float, default=1000.0, help="Tip load P [N] downward")
    parser.add_argument("--elements", type=int, default=64, help="Number of elements")
    parser.add_argument("--scale", type=float, default=100.0, help="Visualization scale for deflection in geometry")
    parser.add_argument("--width_ratio", type=float, default=0.10, help="Ribbon width as fraction of length for VTP export")
    parser.add_argument("--solver", choices=["classical", "quantum", "quantum-cal"], default="classical", help="Reduced system: classical or quantum demo (quantum requires elements=1). 'quantum-cal' uses calibrated scaling for comparison.")
    args = parser.parse_args()

    # Geometry and material (steel)
    L = args.length
    A = args.area
    a = A ** 0.5  # side length [m]
    I = a**4 / 12.0  # second moment for square about bending axis [m^4]
    E = args.E
    P = args.load
    Ne = args.elements

    beam = Beam1DMesh(length=L, elements=Ne, E=E, I=I)
    K = beam.assemble_stiffness()
    f = beam.load_vector(tip_force=P)
    K_red, f_red, free = beam.apply_cantilever_bc(K, f)

    # Solve reduced system
    if args.solver in ("quantum", "quantum-cal"):
        if K_red.shape != (2, 2):
            print("[quantum] This demo supports only a 2x2 reduced system. Overriding elements=1.")
            Ne = 1
            beam = Beam1DMesh(length=L, elements=Ne, E=E, I=I)
            K = beam.assemble_stiffness()
            f = beam.load_vector(tip_force=P)
            K_red, f_red, free = beam.apply_cantilever_bc(K, f)
        # Scale system to improve success probability: A' = A/s, b' = b/s with s = max eigenvalue
        w_eig = np.linalg.eigvalsh(K_red)
        s = float(np.max(w_eig))
        A_scaled = K_red / s
        b_scaled = f_red / s
        if args.solver == "quantum":
            res = hhl_2x2_general(A_scaled, b_scaled)
        else:
            res = hhl_2x2_general_calibrated(A_scaled, b_scaled)
        u_red = res.x  # scaling preserves u
        tag = "[quantum]" if args.solver == "quantum" else "[quantum-cal]"
        print(f"{tag} scaled by s={s:.3e}, success_prob={res.success_prob:.4f} state_norm={res.state_norm:.4f}")
    else:
        u_red = classical_solve(K_red, f_red)
    u_full = beam.reconstruct_full(u_red, free)

    # Extract nodal deflection w and rotation theta
    w = u_full[0::2]
    theta = u_full[1::2]

    # Compare tip deflection with analytic: w(L) = P L^3 / (3 E I)
    w_tip_analytic = P * L**3 / (3.0 * E * I)
    w_tip_num = w[-1]
    print(f"Tip deflection (numeric): {w_tip_num:.6e}")
    print(f"Tip deflection (analytic): {w_tip_analytic:.6e}")
    if w_tip_analytic != 0:
        rel = abs(w_tip_num - w_tip_analytic) / abs(w_tip_analytic)
        print(f"Relative error: {100*rel:.2f}%")

    # Build undeformed and deformed coordinates along the beam centerline
    x = np.linspace(0.0, L, beam.Nn)
    # Visualization scaling
    scale = args.scale
    pts = np.column_stack([x, scale * w, np.zeros_like(x)])

    # Write thick ribbon representation with width = 10% of length
    width = args.width_ratio * L
    write_vtp_beam_thick("output/beam_cantilever.vtp", x, w, width=width, scale_deflection=scale, scalars=w, scalar_name="w")
    print(f"Wrote output/beam_cantilever.vtp (open in ParaView; ribbon width ~{args.width_ratio*100:.0f}% L; color by 'w')")

if __name__ == "__main__":
    main()
