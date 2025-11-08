import numpy as np
from quantum_fem.beam import Beam1DMesh, classical_solve


def test_cantilever_tip_deflection_accuracy():
    L = 1.0
    E = 1.0
    I = 1.0
    P = 1.0
    for Ne in [8, 16, 32]:
        beam = Beam1DMesh(length=L, elements=Ne, E=E, I=I)
        K = beam.assemble_stiffness()
        f = beam.load_vector(tip_force=P)
        K_red, f_red, free = beam.apply_cantilever_bc(K, f)
        u_red = classical_solve(K_red, f_red)
        u_full = beam.reconstruct_full(u_red, free)
        w = u_full[0::2]
        w_tip_num = w[-1]
        w_tip_analytic = P * L**3 / (3.0 * E * I)
        rel_err = abs(w_tip_num - w_tip_analytic) / w_tip_analytic
        assert rel_err < 0.05  # <=5% relative error for these mesh sizes
