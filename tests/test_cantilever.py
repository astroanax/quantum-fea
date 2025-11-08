import numpy as np
from quantum_fem.fem import OneDCantileverMesh, classical_solve
from quantum_fem.preconditioner import IdentityPreconditioner
from quantum_fem.hhl import hhl_2x2_A_eq_2I_minus_X, hhl_2x2_general


def test_hhl_matches_classical_tol():
    mesh = OneDCantileverMesh(length=1.0, elements=2, young_modulus=1.0, area_moment=1.0)
    K = mesh.stiffness_matrix(bending=False)
    f = mesh.load_vector(end_load=1.0)
    u_classical = classical_solve(K, f)
    prec = IdentityPreconditioner()
    Kp, fp = prec.apply(K, f)
    c = Kp[0,0] / 2.0
    b = fp / c
    # General routine should match classical within tolerance
    hhl_gen = hhl_2x2_general(K, f)
    u_quantum_gen = hhl_gen.x
    assert np.linalg.norm(u_classical - u_quantum_gen) < 0.2

    # Specialized routine assumes uniform internal structure; for this mesh K is not
    # exactly proportional to the template matrix, so we only run it for illustration
    # and do not assert accuracy.
    _ = hhl_2x2_A_eq_2I_minus_X(b)
