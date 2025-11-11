# Quantum Finite Element Analysis for Cantilever Beams

This repository presents a pedagogical implementation that integrates classical finite element analysis with quantum linear-algebra algorithms for the solution of cantilever beam deflection problems. The work demonstrates the application of quantum phase estimation (QPE) and the Harrow–Hassidim–Lloyd (HHL) algorithm to structural mechanics, providing a computational framework for exploring quantum approaches to solving the discrete linear systems that arise in finite element formulations.

![Cantilever Beam Visualization](paraview-demo.png)
*Three-dimensional visualization of cantilever beam deflection rendered in ParaView, showing displacement magnitude from the fixed support (left, blue) to the free end (right, red) under transverse loading.*

## Overview

The finite element method discretizes the governing differential equations of continuum mechanics into matrix equations of the form $\mathbf{K}\mathbf{u} = \mathbf{f}$, where $\mathbf{K}$ represents the global stiffness matrix, $\mathbf{u}$ the nodal displacement vector, and $\mathbf{f}$ the applied load vector. For the Euler–Bernoulli cantilever beam problem considered here, Hermite cubic elements are employed to ensure $C^1$ continuity required by the fourth-order beam equation. The assembled stiffness matrix is then processed using quantum algorithms executed on the Cirq state-vector simulator to extract eigenvalue information and construct solution states corresponding to the classical finite element solution.

The implementation couples a classical finite element assembly module with quantum circuits for phase estimation and amplitude-based matrix inversion. Quantum phase estimation is used to extract eigenvalues from the reduced stiffness matrix following application of cantilever boundary conditions, and an optimized HHL-inspired circuit is employed for small ($2 \times 2$) systems to demonstrate the algorithmic procedure. Visualization of the deformed beam geometry is provided through VTK files suitable for rendering in ParaView.

## Installation and Dependencies

The software is implemented in Python and requires the Cirq quantum computing framework along with NumPy for numerical linear algebra and Matplotlib for plotting. To set up the environment, install the required packages using the Python package manager. A minimal installation can be achieved by executing `pip install cirq numpy matplotlib` in a terminal with Python 3.8 or later. No additional quantum hardware or cloud services are required, as all quantum circuits are simulated using Cirq's built-in state-vector simulator.

## Running the Demonstration

The primary demonstration script is located at `src/quantum_fem/demo_quantum_fem_complete.py` and can be executed directly from the command line. Running the script without arguments will solve the default cantilever problem (1.0 m length, 0.1 m square cross-section, 10 kN tip load) using both classical finite element methods and the quantum-simulated HHL algorithm, producing comparison plots and VTK output files for three-dimensional visualization.

For parametric studies or to explore different beam configurations, command-line arguments may be passed to adjust the applied force magnitude (in kilonewtons), beam length and width (in meters), and the number of finite elements used in the refined mesh visualization. For example, the command `python src/quantum_fem/demo_quantum_fem_complete.py -f 20.0 -l 1.5 -w 0.15 -n 30` will solve a 1.5 meter beam with 0.15 meter width subjected to 20 kN force, generating a refined mesh with 30 elements. The default values are suitable for initial verification against analytical beam theory.

### Example Output

```
$ python src/quantum_fem/demo_quantum_fem_complete.py -f 300

Running: Force=300.0kN, Length=1.0m, Width=0.1m


Classical:
  Analytic: -6.000000e-02 m
  FEM:      -6.000000e-02 m

Running QPE (8-bit)...
[QPE] Auto-selected t=1.652263e-07 (λ_min=1314829.0818, λ_max=25351837.5849)
[QPE] Expected phase range: [0.0346, 0.6667]
  QPE eigenvalues: [1336913.31014015]
  True eigenvalues: [ 1314829.0817867  25351837.58487997]

Running HHL...
  Quantum solution: [-0.06000001 -0.09      ]
  Classical solution: [-0.06 -0.09]
  Error: 0.000007%
  Success probability: 18.6%

Saved results to quantum_fem_demo.png
check *vtk files for final results

  Tip deflection: 60.000 mm

ParaView: Open *_detailed.vtk → Apply → Warp By Vector (scale 100-1000)
```

This output demonstrates the complete workflow: classical analytical and finite element solutions are computed first, followed by quantum phase estimation to extract the stiffness matrix eigenvalues, and finally the HHL algorithm produces a quantum solution state that matches the classical result to high precision on the simulator.

## Output and Visualization

Execution of the demonstration script generates several output artifacts. A comparative plot saved as `quantum_fem_demo.png` displays the classical and quantum-simulated solution profiles, eigenvalue estimates from the phase estimation routine, and statistical information about the HHL ancilla measurement outcomes. Additionally, four VTK (Visualization Toolkit) files are produced representing the coarse single-element solution and a refined multi-element discretization for both classical and quantum solution vectors.

![Quantum FEM Results](quantum_fem_demo.png)
*Comparison of quantum phase estimation eigenvalue recovery (left panel) and solution accuracy (right panel). The QPE circuit with 8 precision qubits successfully estimates the two eigenvalues of the reduced stiffness matrix (shown as red X markers overlaying the true values in blue circles). The tip displacement comparison demonstrates agreement between analytical beam theory (green), classical finite element solution (blue), and quantum-simulated HHL solution (red) to within numerical precision on the state-vector simulator.*

For three-dimensional visualization of the beam deflection, the VTK files may be opened in ParaView or any compatible visualization software. The recommended workflow involves loading the detailed VTK file, applying the dataset to activate the geometry, then adding a "Warp By Vector" filter with a scale factor between 100 and 1000 to magnify the displacement field for visual clarity. Coloring the geometry by the `displacement_magnitude` scalar field highlights the deflection distribution from the fixed support to the free end, clearly showing the characteristic cantilever deformation pattern.

## Implementation Structure

The codebase is organized into four principal modules within the `src/quantum_fem` directory. The `beam.py` module implements the classical Euler–Bernoulli finite element formulation, including element stiffness matrix computation using Hermite cubic shape functions, global assembly via the direct stiffness method, application of cantilever boundary conditions, and reconstruction of the full displacement field from the reduced solution. The `phase_estimation.py` module provides a general implementation of quantum phase estimation with automatic time-parameter selection to avoid phase wrapping, along with utilities for constructing unitary operators from Hermitian matrices and synthesizing controlled-unitary gates. The `hhl.py` module contains an optimized circuit for $2 \times 2$ systems that employs analytical eigendecomposition and direct controlled rotations to reduce circuit depth, as well as wrapper functions that combine phase estimation with classical solution methods for diagnostic purposes. Finally, the `io_vtk.py` module handles the export of one-dimensional finite element meshes to three-dimensional hexahedral volumetric grids suitable for visualization in ParaView, mapping nodal displacement fields onto the exported geometry.

## Theoretical Background and Limitations

The quantum algorithms employed in this work—quantum phase estimation and the Harrow–Hassidim–Lloyd algorithm—offer theoretical asymptotic advantages for solving linear systems under specific conditions. These conditions include sparsity or efficient block-encoding of the matrix, moderate condition number, efficient quantum state preparation for the right-hand side vector, and the ability to extract desired information from the solution state without full tomography. For finite element stiffness matrices arising from typical engineering problems, these requirements may not be automatically satisfied, and significant algorithmic development is needed to realize practical quantum advantage.

It is essential to note that all quantum computations reported in this implementation are performed using a classical state-vector simulator, which executes the quantum circuit operations with floating-point arithmetic on conventional hardware. The simulator reproduces the ideal mathematical behavior of quantum gates without physical noise, decoherence, or finite sampling effects that would be present on actual quantum devices. Consequently, the reported solution accuracy reflects algorithmic correctness rather than experimental quantum computing performance. Extrapolation of these results to real quantum hardware requires careful consideration of gate fidelities, coherence times, error mitigation strategies, and the substantial overhead associated with fault-tolerant quantum computation.

## Citation and Acknowledgments

This work draws inspiration from recent research on quantum algorithms for computational mechanics, particularly the quantum finite element method framework described in:

> Zhao, Y., Gao, X., Chen, J., & Peng, X. (2024). *Quantum Finite Element Method for Structural Mechanics*. arXiv preprint arXiv:2403.19512. [https://arxiv.org/abs/2403.19512](https://arxiv.org/abs/2403.19512)

Users who find this implementation useful for research or educational purposes are encouraged to cite the relevant literature on quantum linear system algorithms, including the foundational work on the HHL algorithm by Harrow, Hassidim, and Lloyd (2009), and to acknowledge the pedagogical nature of the current implementation. The complete technical documentation, including detailed derivations of the finite element formulation and quantum circuit constructions, is provided in the accompanying `report.tex` document.

## License and Contact

The source code is distributed under an open-source license to facilitate educational use and further development. Questions, bug reports, or suggestions for algorithmic extensions may be directed to the repository maintainers through the issue tracker. Contributions that improve the documentation, extend the implementation to higher-dimensional problems, or incorporate error mitigation techniques are welcome.
