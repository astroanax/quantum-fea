# Quantum FEM: Cantilever Beam Solver

Quantum finite element solver for cantilever beam problems using HHL algorithm.

## Features

- **Classical FEM**: Hermite cubic beam elements  
- **Quantum Phase Estimation**: Extract eigenvalues  
- **HHL Solver**: Quantum linear solver with C=0.9*λ_min  
- **2x2 System**: Tip displacement + rotation

## Usage

```bash
python src/quantum_fem/demo_quantum_fem_complete.py
```

## Files

- `beam.py` - Euler-Bernoulli beam FEM
- `hhl.py` - HHL quantum solver  
- `phase_estimation.py` - Quantum phase estimation
- `demo_quantum_fem_complete.py` - Complete demo

## Implementation

- Uses Cirq for quantum circuits
- Magnitude recovery: ||x|| = (||b||/C) * √p_success
- QPE with 8-bit precision (~1% eigenvalue error)
- Simulator achieves <0.01% solution error
