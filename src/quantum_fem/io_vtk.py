"""VTK export for ParaView visualization."""

import numpy as np


def write_beam_3d_vtk(filename, length, width, n_elements, displacements_1d, rotations_1d):
    """Write 3D volumetric beam to VTK file for ParaView visualization.
    
    Creates a rectangular beam with proper 3D geometry from 1D FEM solution.
    
    Args:
        filename: Output .vtk file path
        length: Beam length (m)
        width: Beam cross-section width (m) - assumes square section
        n_elements: Number of elements along beam length
        displacements_1d: Array of nodal displacements [w0, w1, ...] in y-direction
        rotations_1d: Array of nodal rotations [θ0, θ1, ...] about z-axis
    """
    n_nodes_1d = n_elements + 1
    
    # Create 3D mesh: discretize cross-section into 2x2 grid
    n_cross = 2  # 2x2 points on cross-section
    total_nodes = n_nodes_1d * n_cross * n_cross
    
    # Generate node coordinates
    nodes = []
    node_map = {}  # (i_length, i_y, i_z) -> node_index
    
    x_coords = np.linspace(0, length, n_nodes_1d)
    y_coords = np.linspace(-width/2, width/2, n_cross)
    z_coords = np.linspace(-width/2, width/2, n_cross)
    
    node_idx = 0
    for i_x in range(n_nodes_1d):
        for i_y in range(n_cross):
            for i_z in range(n_cross):
                x = x_coords[i_x]
                y = y_coords[i_y]
                z = z_coords[i_z]
                nodes.append([x, y, z])
                node_map[(i_x, i_y, i_z)] = node_idx
                node_idx += 1
    
    nodes = np.array(nodes)
    
    # Create hexahedral cells (bricks)
    cells = []
    for i_x in range(n_nodes_1d - 1):
        for i_y in range(n_cross - 1):
            for i_z in range(n_cross - 1):
                # Hexahedron connectivity (VTK ordering)
                n0 = node_map[(i_x,   i_y,   i_z)]
                n1 = node_map[(i_x+1, i_y,   i_z)]
                n2 = node_map[(i_x+1, i_y+1, i_z)]
                n3 = node_map[(i_x,   i_y+1, i_z)]
                n4 = node_map[(i_x,   i_y,   i_z+1)]
                n5 = node_map[(i_x+1, i_y,   i_z+1)]
                n6 = node_map[(i_x+1, i_y+1, i_z+1)]
                n7 = node_map[(i_x,   i_y+1, i_z+1)]
                cells.append([n0, n1, n2, n3, n4, n5, n6, n7])
    
    cells = np.array(cells)
    n_cells = len(cells)
    
    # Compute displacements for each 3D node from 1D beam theory
    displacements_3d = np.zeros((total_nodes, 3))
    
    for i_x in range(n_nodes_1d):
        w = displacements_1d[i_x]  # Deflection at this x-position
        theta = rotations_1d[i_x]   # Rotation at this x-position
        
        for i_y in range(n_cross):
            for i_z in range(n_cross):
                node_idx = node_map[(i_x, i_y, i_z)]
                y_local = y_coords[i_y]
                z_local = z_coords[i_z]
                
                # Beam theory: all points on cross-section have same deflection w
                # but rotation θ causes additional displacement proportional to z-coordinate
                # u_y = w - θ * z  (for small angles)
                displacements_3d[node_idx, 0] = 0.0           # No axial displacement
                displacements_3d[node_idx, 1] = w - theta * z_local  # Deflection
                displacements_3d[node_idx, 2] = 0.0           # No z-displacement
    
    # Write VTK file
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("3D Cantilever Beam\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        
        f.write(f"POINTS {total_nodes} float\n")
        for node in nodes:
            f.write(f"{node[0]:.6e} {node[1]:.6e} {node[2]:.6e}\n")
        
        f.write(f"\nCELLS {n_cells} {n_cells * 9}\n")
        for cell in cells:
            f.write(f"8 {cell[0]} {cell[1]} {cell[2]} {cell[3]} {cell[4]} {cell[5]} {cell[6]} {cell[7]}\n")
        
        f.write(f"\nCELL_TYPES {n_cells}\n")
        for _ in range(n_cells):
            f.write("12\n")  # VTK_HEXAHEDRON
        
        f.write(f"\nPOINT_DATA {total_nodes}\n")
        f.write("VECTORS displacement float\n")
        for disp in displacements_3d:
            f.write(f"{disp[0]:.6e} {disp[1]:.6e} {disp[2]:.6e}\n")
        
        f.write("\nSCALARS displacement_magnitude float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for disp in displacements_3d:
            f.write(f"{np.linalg.norm(disp):.6e}\n")


def write_beam_vtk(filename, nodes, displacements, rotations=None):
    """Write beam solution to VTK file for ParaView.
    
    Args:
        filename: Output .vtk file path
        nodes: Nx3 array of node coordinates [x, y, z]
        displacements: Nx3 array of displacements [ux, uy, uz]
        rotations: Nx3 array of rotations [rx, ry, rz] (optional)
    """
    n_nodes = len(nodes)
    n_cells = n_nodes - 1
    
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Beam solution\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        
        f.write(f"POINTS {n_nodes} float\n")
        for node in nodes:
            f.write(f"{node[0]:.6e} {node[1]:.6e} {node[2]:.6e}\n")
        
        f.write(f"\nCELLS {n_cells} {n_cells * 3}\n")
        for i in range(n_cells):
            f.write(f"2 {i} {i+1}\n")
        
        f.write(f"\nCELL_TYPES {n_cells}\n")
        for _ in range(n_cells):
            f.write("3\n")
        
        f.write(f"\nPOINT_DATA {n_nodes}\n")
        f.write("VECTORS displacement float\n")
        for disp in displacements:
            f.write(f"{disp[0]:.6e} {disp[1]:.6e} {disp[2]:.6e}\n")
        
        if rotations is not None:
            f.write("\nVECTORS rotation float\n")
            for rot in rotations:
                f.write(f"{rot[0]:.6e} {rot[1]:.6e} {rot[2]:.6e}\n")
        
        f.write("\nSCALARS displacement_magnitude float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for disp in displacements:
            f.write(f"{np.linalg.norm(disp):.6e}\n")
