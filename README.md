# Quantum FEM: Cantilever Beam Solver

Quantum finite element method for cantilever beam analysis using QPE + HHL algorithm.

## Requirements

```bash
pip install cirq numpy matplotlib
```

## Usage

```bash
python src/quantum_fem/demo_quantum_fem_complete.py
```

### CLI Options

```bash
python src/quantum_fem/demo_quantum_fem_complete.py -f 20.0 -l 1.5 -w 0.15 -n 30
```

- `-f, --force`: Force in kN (default: 10.0)
- `-l, --length`: Beam length in meters (default: 1.0)
- `-w, --width`: Beam width in meters (default: 0.1)
- `-n, --elements`: Number of elements for detailed mesh (default: 20)

## Output

- `quantum_fem_demo.png`: Comparison plot
- `*.vtk`: ParaView visualization files

## ParaView Visualization

1. Open `*_detailed.vtk` in ParaView
2. Click "Apply"
3. Add "Warp By Vector" filter (scale: 100-1000)
4. Color by `displacement_magnitude`

## Components

- **beam.py**: Euler-Bernoulli FEM
- **phase_estimation.py**: Quantum Phase Estimation
- **hhl.py**: HHL algorithm (quantum linear solver)
- **io_vtk.py**: VTK export for visualization
