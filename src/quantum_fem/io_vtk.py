import numpy as np

def write_beam_3d_vtk(filename: str, length: float, width: float, n_elem: int,
                      displacements: np.ndarray, rotations: np.ndarray):
    """Export 3D beam to VTK with hexahedral elements"""
    n_nodes_1d = n_elem + 1
    n_cross = 3
    total_nodes = n_nodes_1d * n_cross * n_cross
    
    nodes = []
    x_coords = np.linspace(0, length, n_nodes_1d)
    y_coords = np.linspace(-width/2, width/2, n_cross)
    z_coords = np.linspace(-width/2, width/2, n_cross)
    
    for i, x in enumerate(x_coords):
        w = displacements[i]
        theta = rotations[i]
        for y in y_coords:
            for z in z_coords:
                y_rot = y * np.cos(theta) - w * np.sin(theta)
                z_pos = z
                w_pos = y * np.sin(theta) + w * np.cos(theta)
                nodes.append([x, y_rot, z_pos + w_pos])
    
    nodes = np.array(nodes)
    
    cells = []
    for i in range(n_elem):
        for j in range(n_cross - 1):
            for k in range(n_cross - 1):
                n0 = i * n_cross * n_cross + j * n_cross + k
                n1 = n0 + 1
                n2 = n0 + n_cross
                n3 = n2 + 1
                n4 = n0 + n_cross * n_cross
                n5 = n4 + 1
                n6 = n4 + n_cross
                n7 = n6 + 1
                cells.append([n0, n1, n3, n2, n4, n5, n7, n6])
    
    with open(filename, 'w') as f:
        f.write('# vtk DataFile Version 3.0\n')
        f.write('3D Beam Cantilever\n')
        f.write('ASCII\n')
        f.write('DATASET UNSTRUCTURED_GRID\n')
        f.write(f'POINTS {total_nodes} float\n')
        for node in nodes:
            f.write(f'{node[0]:.6e} {node[1]:.6e} {node[2]:.6e}\n')
        
        n_cells = len(cells)
        f.write(f'\nCELLS {n_cells} {n_cells * 9}\n')
        for cell in cells:
            f.write(f'8 {" ".join(map(str, cell))}\n')
        
        f.write(f'\nCELL_TYPES {n_cells}\n')
        for _ in cells:
            f.write('12\n')
        
        f.write(f'\nPOINT_DATA {total_nodes}\n')
        f.write('VECTORS displacement float\n')
        for i, node in enumerate(nodes):
            idx_1d = i // (n_cross * n_cross)
            w = displacements[idx_1d]
            f.write(f'0.0 0.0 {w:.6e}\n')
        
        f.write('SCALARS displacement_magnitude float 1\n')
        f.write('LOOKUP_TABLE default\n')
        for i in range(total_nodes):
            idx_1d = i // (n_cross * n_cross)
            w = abs(displacements[idx_1d])
            f.write(f'{w:.6e}\n')
