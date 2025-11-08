# Quantum FEM in Cirq

This project provides an initial experimental implementation inspired by the paper "Quantum Realization of the Finite Element Method" (arXiv:2403.19512). It demonstrates how to:

1. Assemble a tiny finite element stiffness matrix for a cantilever-like 1D beam (Euler-Bernoulli simplified to axial or bending proxy).
2. Precondition (placeholder BPX-style hook; currently identity / scaling) the linear system K u = f.
3. Solve a small linear system using a quantum linear solver routine (HHL-style) in Cirq for demonstration.
4. Compare quantum-simulated result with the classical solution.

> NOTE: This is a pedagogical starting point. The full algorithm in the paper involves a BPX preconditioner and multilevel structure; here we stub the preconditioning and focus on the linear solver pipeline.

## Structure

- `src/quantum_fem/fem.py` – Simple FEM mesh and stiffness assembly for a 1D cantilever discretized into linear elements.
- `src/quantum_fem/preconditioner.py` – Identity preconditioner and a 2D BPX-style multilevel preconditioner for Poisson on a Cartesian grid.
- `src/quantum_fem/hhl.py` – Minimal HHL implementation for 2^n dimension systems with normalization assumptions.
- `src/quantum_fem/block_encoding.py` – 4×4 HHL-like multi-eigenvalue demo (structured small-system block-encoding flavor).
- `src/quantum_fem/quantum_linear_functional.py` – Hadamard-test-based functional estimator ⟨g,x⟩.
- `src/quantum_fem/example_cantilever.py` – End-to-end example building the system and invoking the quantum solver.
- `src/quantum_fem/fem2d.py` – 2D Poisson Q1 mesh and stiffness/load assembly.
- `src/quantum_fem/example_2d_bpx.py` – Demonstrates BPX preconditioning effect on condition number and writes a `.vti` for ParaView.
- `src/quantum_fem/io_vtk.py` – Minimal VTK ImageData writer (ASCII .vti) for ParaView.
- `src/quantum_fem/beam.py` – Euler-Bernoulli cantilever beam FEM (Hermite cubic elements, bending stiffness).
- `src/quantum_fem/example_beam_cantilever.py` – Solves beam end-load deflection and exports a `.vtp` polyline for visualization.
 - `src/quantum_fem/example_quantum_4x4.py` – Runs the 4×4 demo and estimates a simple functional.
- `tests/test_cantilever.py` – Validates quantum vs classical solution within tolerance.
- `requirements.txt` – Python dependencies.

## Running

Install dependencies and run the example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/quantum_fem/example_cantilever.py
pytest -q
```

### Export 2D solution to ParaView

Run the 2D Poisson example to write `output/poisson2d_u.vti`:

```bash
python src/quantum_fem/example_2d_bpx.py
```

Open the resulting `.vti` file in ParaView.

### Quantum demos

- 2×2 HHL (toy): used in the beam example when `--solver quantum` and `--elements 1`.
- 4×4 small-system demo:

```bash
python src/quantum_fem/example_quantum_4x4.py
```

This prepares a 4×4 Laplacian system, applies an HHL-like multi-eigenvalue rotation, prints eigenvalues and a functional estimate via the Hadamard test.

### Cantilever beam deflection (1D bending)

Solve and visualize the Euler-Bernoulli beam with an end load:

```bash
python src/quantum_fem/example_beam_cantilever.py
```

Loads `output/beam_cantilever.vtp` in ParaView, containing a polyline with point data `w` (deflection) and geometry already displaced (scaled). Use a Tube filter for better visibility.

Current physical parameters (steel example):
- Length L = 1.0 m
- Cross-section area A = 0.01 m^2 (square side a = sqrt(A))
- Second moment I = a^4 / 12
- Young's modulus E = 200 GPa
- Tip load P = 1 kN downward

Analytic tip deflection w(L) = P L^3 / (3 E I) ≈ 2.0e-4 m (≈0.2 mm). The example scales deflection by 100× for visualization.

## Quantum solver notes

The current quantum HHL implementation for 2×2 systems is a **pedagogical hybrid** that:
- Uses quantum circuits to prepare the solution state via controlled rotations encoding eigenvalue reciprocals
- Postselects on an ancilla measurement to extract the solution subspace
- Uses classical eigendecomposition to calibrate the absolute scale of the solution

This approach achieves **0% error** for the cantilever beam demo because the quantum circuit correctly encodes the solution *direction*, and we use classical information to set the magnitude. This is appropriate for small educational demos but differs from a fully fault-tolerant quantum algorithm.

### Full quantum algorithm (per arXiv:2403.19512)

The paper describes a complete quantum FEM solver that achieves optimal complexity $\mathcal{O}(\mathrm{tol}^{-1})$ for 2D problems using:

1. **BPX preconditioning**: Transforms the FEM system $Ku=f$ into a well-conditioned system $BAu=Bf$ where $B$ is an additive multilevel operator. Our implementation includes a pedagogical BPX for 2D Poisson.

2. **Block encoding**: The preconditioned matrix is embedded into a unitary operator suitable for quantum computation. For small systems we demonstrate this via eigenvalue-controlled rotations.

3. **Phase estimation**: Extracts eigenvalues $\lambda_i$ without classical diagonalization, using $\log(n)$ ancilla qubits and controlled time-evolution $e^{iAt}$.

4. **Amplitude amplification**: Boosts the success probability of postselection from $\sim \kappa^{-2}$ to near-unity, where $\kappa$ is the condition number.

5. **Amplitude estimation**: Recovers the norm $\|A^{-1}b\|$ and functionals $\langle g, x \rangle$ via repeated measurements and Hadamard tests.

Our 2×2 demo implements steps 2 and partial step 4 (circuit structure for postselection). For a fully quantum implementation at scale, steps 1, 3, 4, and 5 would need to be integrated without classical eigendecomposition.

## Next Steps

1. Implement true BPX multilevel preconditioner (hierarchical bases, restriction/prolongation operators).
2. Extend to 2D Cartesian grid Poisson problem.
3. Use block-encoding of preconditioned matrix and quantum amplitude estimation for functionals of solution.
4. Integrate error mitigation strategies for running on actual hardware.
5. Resource estimation: gate counts, depth vs qubit count for larger meshes.

### BPX notes

The included BPX implementation is a simplified additive multilevel preconditioner over nested dyadic grids:

- Levels ℓ = 0..L with n_ℓ = 2^ℓ intervals per dimension.
- Interior Laplacians assembled via a 5-point stencil (equiv. to uniform Q1 FE).
- Prolongations are Kronecker products of 1D linear-interpolation operators.
- BPX operator B ≈ Σ_ℓ P_{ℓ→L} D_ℓ^{-1} P_{ℓ→L}^T.

This is sufficient to demonstrate cond(BA) ≪ cond(A) on small grids; further tuning (weights, smoothers, exact FE operators) can improve performance.

## Disclaimer

The current quantum part is illustrative (HHL on a simulator). The paper's optimal complexity claims depend on carefully engineered preconditioning and measurement strategies not yet included.
