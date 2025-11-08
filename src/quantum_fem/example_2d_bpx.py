import numpy as np
from quantum_fem.fem2d import PoissonQ1Mesh2D, classical_solve
from quantum_fem.preconditioner import BPXPreconditioner2D
from quantum_fem.io_vtk import write_vti_scalar

"""Demonstrate 2D BPX preconditioner on Poisson with Dirichlet boundary.
Build A (finest), compute M = B A, and report condition numbers.
"""

def cond(A: np.ndarray) -> float:
    w = np.linalg.eigvalsh(A)
    w = np.sort(w)
    # Filter tiny eigenvalues for numerical safety
    w = w[w > 1e-12]
    return float(w[-1] / w[0])


def main():
    # Finest grid n = 2^L intervals per side. Use 16 for nicer visualization.
    n = 16
    mesh = PoissonQ1Mesh2D(intervals=n)
    A = mesh.stiffness_matrix()
    b = mesh.load_vector()

    # BPX build
    L = int(np.log2(n))
    bpx = BPXPreconditioner2D(finest_intervals=n, levels=L)
    M, _ = bpx.apply(A, b)

    cA = cond(A)
    cM = cond(M)

    print(f"Grid: {n}x{n} intervals, interior unknowns: {(n-1)*(n-1)}")
    print(f"cond(A)  ≈ {cA:.2e}")
    print(f"cond(BA) ≈ {cM:.2e}  (expect significant reduction)")

    # Direct classical solve for the 2D field
    u = classical_solve(A, b)

    # Embed interior solution into a full (n+1)x(n+1) grid with zero Dirichlet boundary
    m = n - 1
    full = np.zeros((n + 1, n + 1), dtype=np.float32)
    u2d = u.reshape(m, m)
    full[1:n, 1:n] = u2d

    # Write VTK ImageData (.vti) for ParaView; spacing matches uniform grid
    write_vti_scalar("output/poisson2d_u.vti", full, spacing=(1.0/n, 1.0/n, 1.0), name="u")
    print("Wrote output/poisson2d_u.vti (open in ParaView).")

if __name__ == "__main__":
    main()
