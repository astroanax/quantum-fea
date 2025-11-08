# CLI Usage Guide

## Easy Command Line Interface

Run the quantum FEM solver with custom parameters:

```bash
python src/quantum_fem/demo_quantum_fem_complete.py [OPTIONS]
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `-f, --force` | Applied force in kN | 10.0 |
| `-l, --length` | Beam length in meters | 1.0 |
| `-w, --width` | Beam width in meters (square section) | 0.1 |
| `-n, --elements` | Number of elements for detailed mesh | 20 |

## Examples

### Change the load:

```bash
# 5 kN force (half deflection)
python src/quantum_fem/demo_quantum_fem_complete.py -f 5.0

# 20 kN force (double deflection)
python src/quantum_fem/demo_quantum_fem_complete.py -f 20.0

# 50 kN force (large deflection)
python src/quantum_fem/demo_quantum_fem_complete.py -f 50.0
```

### Different beam geometry:

```bash
# Longer beam (2m) - more deflection
python src/quantum_fem/demo_quantum_fem_complete.py -l 2.0

# Thicker beam (0.2m) - less deflection
python src/quantum_fem/demo_quantum_fem_complete.py -w 0.2

# Combined: 2m long, 0.05m thick, 15kN force
python src/quantum_fem/demo_quantum_fem_complete.py -l 2.0 -w 0.05 -f 15.0
```

### Better visualization:

```bash
# More elements for smoother visualization
python src/quantum_fem/demo_quantum_fem_complete.py -n 50

# Combined with custom load
python src/quantum_fem/demo_quantum_fem_complete.py -f 25.0 -n 40
```

## Quick Tests

| Command | Deflection | Description |
|---------|-----------|-------------|
| `python ... -f 1.0` | 0.2 mm | Light load |
| `python ... -f 5.0` | 1.0 mm | Medium load |
| `python ... -f 10.0` | 2.0 mm | Default |
| `python ... -f 25.0` | 5.0 mm | Heavy load |
| `python ... -f 50.0` | 10.0 mm | Very heavy load |

## Output Files

After running, you get 4 VTK files:
- `classical_solution.vtk` (1 element)
- `quantum_solution.vtk` (1 element)
- `classical_solution_detailed.vtk` (n elements) ⭐ **Use this one**
- `quantum_solution_detailed.vtk` (n elements)

## View Results

```bash
paraview classical_solution_detailed.vtk
```

Then apply "Warp By Vector" filter to see the deflection!

## Tip Deflection Formula

For a cantilever beam with end load:

```
deflection = (F × L³) / (3 × E × I)
```

Where:
- F = Force (N)
- L = Length (m)
- E = Young's modulus (200 GPa for steel)
- I = Moment of inertia = w⁴/12 for square section

**Quick calculation**: Each 1 kN gives 0.2 mm deflection (for default 1m × 0.1m beam)
