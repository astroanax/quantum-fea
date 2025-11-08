import numpy as np
from quantum_fem.fem import OneDCantileverMesh, classical_solve
from quantum_fem.preconditioner import IdentityPreconditioner
from quantum_fem.hhl import hhl_2x2_A_eq_2I_minus_X, hhl_2x2_general

"""End-to-end example for a tiny cantilever using 2 elements (giving a 2x2 reduced system).

The stiffness matrix for 2 elements with coefficient c is:
K = c * [[2,-1],[-1,2]] after boundary reduction.
We solve K u = f with f = [0, F]^T (point load at free end), and compare to HHL routine
which is specialized to A = [[2,-1],[-1,2]]. We divide by c when passing to HHL to match its assumption.
"""

def main():
    mesh = OneDCantileverMesh(length=1.0, elements=2, young_modulus=1.0, area_moment=1.0)
    K = mesh.stiffness_matrix(bending=False)  # shape (2,2)
    f = mesh.load_vector(end_load=1.0)        # shape (2,)

    # Classical solution
    u_classical = classical_solve(K, f)

    # Preconditioner stub (identity)
    prec = IdentityPreconditioner()
    Kp, fp = prec.apply(K, f)

    # Attempt to detect proportionality to template for specialized solver
    template = np.array([[2,-1],[-1,2]])
    c = float(np.sum(Kp * template) / np.sum(template * template))
    rel_err = np.linalg.norm(Kp - c * template) / np.linalg.norm(Kp)
    use_specialized = rel_err < 1e-8

    # General routine directly on Kp and fp
    hhl_general = hhl_2x2_general(Kp, fp)
    u_quantum_general = hhl_general.x

    u_quantum_special = None
    hhl_res = None
    if use_specialized:
        b = fp / c
        hhl_res = hhl_2x2_A_eq_2I_minus_X(b)
        u_quantum_special = hhl_res.x / c

    print("Classical solution u:", u_classical)
    print("Quantum (general HHL) u:", u_quantum_general)
    print("Success prob (general):", hhl_general.success_prob)
    err_general = np.linalg.norm(u_classical - u_quantum_general)
    print("L2 error (general) =", err_general)
    if use_specialized:
        print("Quantum (specialized) u:", u_quantum_special)
        print("Success prob (specialized):", hhl_res.success_prob)
        err_special = np.linalg.norm(u_classical - u_quantum_special)
        print("L2 error (specialized) =", err_special)
    else:
        print("Specialized solver skipped (matrix not proportional to 2I - X template)")

if __name__ == "__main__":
    main()
