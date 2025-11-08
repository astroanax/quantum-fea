# ParaView Visualization Guide

## Generated Files

- **`classical_solution.vtk`** - Classical FEM solution
- **`quantum_solution.vtk`** - Quantum HHL solution

## Opening in ParaView

1. **Launch ParaView**
   ```bash
   paraview
   ```

2. **Open File**
   - File → Open
   - Select `classical_solution.vtk` or `quantum_solution.vtk`
   - Click "Apply" in Properties panel

3. **Basic Visualization**
   - In the toolbar, change coloring from "Solid Color" to:
     - `displacement` - Shows displacement vector field
     - `displacement_magnitude` - Shows scalar magnitude
     - `rotation` - Shows rotation vector field

4. **View Deformed Shape**
   - Select the loaded data in Pipeline Browser
   - Filters → Alphabetical → "Warp By Vector"
   - In Properties:
     - Vectors: `displacement`
     - Scale Factor: 1000 (increase to exaggerate deformation)
   - Click "Apply"

5. **Compare Solutions**
   - Open both VTK files
   - Use the "eye" icon in Pipeline Browser to toggle visibility
   - Apply different colors to distinguish them

## Data Fields

- **`displacement`**: [ux, uy, uz] displacement vector
  - ux: x-direction displacement
  - uy: y-direction displacement (beam deflection)
  - uz: z-direction displacement
  
- **`rotation`**: [rx, ry, rz] rotation vector
  - rx: rotation about x-axis
  - ry: rotation about y-axis
  - rz: rotation about z-axis (beam rotation)
  
- **`displacement_magnitude`**: Scalar norm of displacement

## Tips

- Use "Warp By Vector" with high scale factor (1000-10000) to see small deflections
- Compare classical vs quantum by loading both files
- The solutions should be nearly identical (<0.01% difference)
