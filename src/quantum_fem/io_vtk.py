from __future__ import annotations
import os
import numpy as np
from typing import Tuple


def write_vti_scalar(
    file_path: str,
    data2d: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    name: str = "u",
    dtype: str = "Float32",
) -> None:
    """Write a 2D scalar field as VTK XML ImageData (.vti) for ParaView.

    - data2d: shape (Ny, Nx), y = rows, x = cols.
    - spacing: (dx, dy, dz)
    - origin: (ox, oy, oz)
    - name: scalar array name
    - dtype: VTK type string, e.g., Float32 or Float64
    """
    if data2d.ndim != 2:
        raise ValueError("data2d must be 2D array (Ny, Nx)")
    Ny, Nx = data2d.shape
    # VTK extents are inclusive indices over points; for Nx points, X1 = Nx-1
    extent = f"0 {Nx-1} 0 {Ny-1} 0 0"
    dx, dy, dz = spacing
    ox, oy, oz = origin

    # Flatten with x fastest, y slowest (row-major) matches VTK point ordering
    flat = data2d.astype(np.float32 if dtype == "Float32" else np.float64).ravel(order="C")

    # Chunk values into lines for readability
    def chunked(values, n):
        for i in range(0, len(values), n):
            yield values[i:i+n]

    lines = []
    for ch in chunked(flat, 12):
        lines.append(" ".join(f"{v:.7e}" for v in ch))
    data_str = "\n        ".join(lines)

    xml = f"""<?xml version="1.0"?>
<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">
  <ImageData WholeExtent="{extent}" Origin="{ox} {oy} {oz}" Spacing="{dx} {dy} {dz}">
    <Piece Extent="{extent}">
      <PointData Scalars="{name}">
        <DataArray type="{dtype}" Name="{name}" NumberOfComponents="1" format="ascii">
        {data_str}
        </DataArray>
      </PointData>
      <CellData>
      </CellData>
    </Piece>
  </ImageData>
</VTKFile>
"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(xml)


def write_vtp_polyline(
    file_path: str,
    points: np.ndarray,
    scalars: np.ndarray | None = None,
    scalar_name: str = "w",
    dtype: str = "Float32",
) -> None:
    """Write a PolyData (.vtp) with a single polyline and optional point scalar.

    - points: shape (N,3)
    - scalars: optional shape (N,) to attach as PointData
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be (N,3)")
    N = points.shape[0]

    # Build connectivity for one line with N points
    # VTK PolyLine cell: [N, 0, 1, 2, ..., N-1]
    connectivity = " ".join(str(i) for i in range(N))
    pts_lines = "\n        ".join(" ".join(f"{v:.7e}" for v in p) for p in points.astype(float))

    scalar_block = ""
    if scalars is not None:
        arr = scalars.astype(np.float32 if dtype == "Float32" else np.float64)
        vals = "\n          ".join(" ".join(f"{v:.7e}" for v in arr[i:i+12]) for i in range(0, len(arr), 12))
        scalar_block = f"""
      <PointData Scalars=\"{scalar_name}\">
        <DataArray type=\"{dtype}\" Name=\"{scalar_name}\" NumberOfComponents=\"1\" format=\"ascii\">
          {vals}
        </DataArray>
      </PointData>"""
    else:
        scalar_block = "\n      <PointData/>"

    xml = f"""<?xml version=\"1.0\"?>
<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">
  <PolyData>
    <Piece NumberOfPoints=\"{N}\" NumberOfVerts=\"0\" NumberOfLines=\"1\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">
      <Points>
        <DataArray type=\"{dtype}\" NumberOfComponents=\"3\" format=\"ascii\">
        {pts_lines}
        </DataArray>
      </Points>{scalar_block}
      <Lines>
        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">{connectivity}</DataArray>
        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">{N}</DataArray>
      </Lines>
      <Verts/>
      <Strips/>
      <Polys/>
    </Piece>
  </PolyData>
 </VTKFile>
"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(xml)


def write_vtp_beam_thick(
    file_path: str,
    x: np.ndarray,
    y: np.ndarray,
    width: float,
    scale_deflection: float = 1.0,
    scalars: np.ndarray | None = None,
    scalar_name: str = "w",
    dtype: str = "Float32",
) -> None:
    """Write a beam as a thick rectangular strip PolyData so it is visually wider in ParaView.

    Geometry: For each node i we create two points at z = ±width/2, sharing same (x_i, y_i*scale_deflection).
    Quads connect consecutive node pairs forming a ribbon. ParaView can then display it as a surface.
    """
    if x.shape != y.shape:
        raise ValueError("x and y must have same shape")
    N = x.size
    if N < 2:
        raise ValueError("Need at least 2 points")
    z_pos = width / 2.0
    z_neg = -width / 2.0
    pts_list = []
    for xi, yi in zip(x, y):
        Y = yi * scale_deflection
        pts_list.append((xi, Y, z_neg))
        pts_list.append((xi, Y, z_pos))
    pts_arr = np.array(pts_list, dtype=float)
    # Build quad connectivity
    connectivity = []
    offsets = []
    offset_acc = 0
    for i in range(N - 1):
        p0 = 2 * i
        p1 = 2 * i + 1
        p2 = 2 * (i + 1) + 1
        p3 = 2 * (i + 1)
        connectivity.extend([p0, p1, p2, p3])
        offset_acc += 4
        offsets.append(offset_acc)

    pts_lines = "\n        ".join(" ".join(f"{v:.7e}" for v in p) for p in pts_arr)
    conn_str = " ".join(str(c) for c in connectivity)
    offs_str = " ".join(str(o) for o in offsets)

    scalar_block = ""
    if scalars is not None:
        if scalars.shape != (N,):
            raise ValueError("scalars must match number of original nodes (N)")
        # Duplicate scalars for two points per node
        sdup = np.repeat(scalars, 2)
        arr = sdup.astype(np.float32 if dtype == "Float32" else np.float64)
        vals = "\n          ".join(" ".join(f"{v:.7e}" for v in arr[i:i+12]) for i in range(0, len(arr), 12))
        scalar_block = f"""
      <PointData Scalars=\"{scalar_name}\">
        <DataArray type=\"{dtype}\" Name=\"{scalar_name}\" NumberOfComponents=\"1\" format=\"ascii\">
          {vals}
        </DataArray>
      </PointData>"""
    else:
        scalar_block = "\n      <PointData/>"

    xml = f"""<?xml version=\"1.0\"?>
<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">
  <PolyData>
    <Piece NumberOfPoints=\"{2*N}\" NumberOfVerts=\"0\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"{N-1}\">{scalar_block}
      <Points>
        <DataArray type=\"{dtype}\" NumberOfComponents=\"3\" format=\"ascii\">
        {pts_lines}
        </DataArray>
      </Points>
      <Polys>
        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">{conn_str}</DataArray>
        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">{offs_str}</DataArray>
      </Polys>
      <Lines/>
      <Verts/>
      <Strips/>
    </Piece>
  </PolyData>
 </VTKFile>
"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(xml)
