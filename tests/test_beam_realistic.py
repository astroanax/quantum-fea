import numpy as np
from quantum_fem.beam import Beam1DMesh, classical_solve


def test_realistic_steel_beam_tip_deflection():
    L = 1.0
    A = 0.01  # m^2
    a = A ** 0.5
    I = a**4 / 12.0
    E = 200e9
    P = 1000.0
    for Ne in [16, 32, 64]:
        beam = Beam1DMesh(length=L, elements=Ne, E=E, I=I)
        K = beam.assemble_stiffness()
        f = beam.load_vector(tip_force=P)
        K_red, f_red, free = beam.apply_cantilever_bc(K, f)
        u_red = classical_solve(K_red, f_red)
        u_full = beam.reconstruct_full(u_red, free)
        w = u_full[0::2]
        w_tip = w[-1]
        w_analytic = P * L**3 / (3.0 * E * I)
        rel_err = abs(w_tip - w_analytic) / w_analytic
        assert rel_err < 0.05
