# Quantum FEM: Steel Cantilever Beam Solver

Quantum finite element solver for 3D steel cantilever beam using HHL algorithm.

## Problem Setup

- **Material**: Steel (E = 200 GPa)
- **Geometry**: 1m × 0.1m × 0.1m (square cross-section)
- **Loading**: 10 kN downward force at free end
- **Deflection**: ~2 mm at tip

## Features

- **Classical FEM**: Hermite cubic beam elements (Euler-Bernoulli theory)
- **Quantum Phase Estimation**: Extract eigenvalues without classical diagonalization
- **HHL Solver**: Quantum linear solver with C=0.9*λ_min (unsaturated rotations)
- **3D Visualization**: VTK export for ParaView

## Usage

```bash
python src/quantum_fem/demo_quantum_fem_complete.py
```

Generates 4 VTK files for ParaView visualization:
- `classical_solution.vtk` - Basic classical solution
- `quantum_solution.vtk` - Basic quantum solution  
- `classical_solution_detailed.vtk` - 20 elements (smooth visualization)
- `quantum_solution_detailed.vtk` - 20 elements (smooth visualization)

## ParaView Visualization

1. Open: `paraview classical_solution_detailed.vtk`
2. Click "Apply"
3. Add filter: "Warp By Vector" (scale: 100-1000)
4. Color by: "displacement_magnitude"
5. See the beam bending downward!

## Files

- `beam.py` - Euler-Bernoulli beam FEM
- `hhl.py` - HHL quantum solver  
- `phase_estimation.py` - Quantum phase estimation
- `io_vtk.py` - 3D VTK export for ParaView
- `demo_quantum_fem_complete.py` - Complete demo

## Implementation

- Uses Cirq for quantum circuits
- Magnitude recovery: ||x|| = (||b||/C) * √p_success
- QPE with 8-bit precision (~7% eigenvalue error)
- Simulator achieves <0.001% solution error
- 3D volumetric mesh with hexahedral elements
