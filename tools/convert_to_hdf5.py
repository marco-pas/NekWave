#!/usr/bin/env python3
"""
NekWave HDF5 Conversion Utility
Converts NekWave binary field archives (.bin) or CSV field exports into
standard HDF5 (.h5) datasets with companion XDMF (.xmf) for ParaView / VisIt.

Generates 3D micro-hexahedral cell connectivity matching NekWave's DG-SEM
GLL collocation points, allowing ParaView to render surfaces, volumes,
slices, streamlines, and vectors natively without blank screens.

Usage:
    python3 tools/convert_to_hdf5.py [output_directory]
    python3 tools/convert_to_hdf5.py output_3dboxpec/fields.bin -o output_3dboxpec/fields.h5
    python3 tools/convert_to_hdf5.py output_3dboxpec/fields.h5
"""

import sys
import os
import struct
import argparse
try:
    import numpy as np
except ImportError:
    np = None

try:
    import h5py
except ImportError:
    h5py = None

def generate_hex_connectivity(order_N, num_elements):
    """
    Builds micro-hexahedral cell connectivity table for GLL collocation points.
    For polynomial order N (N GLL points per direction, p = N - 1 sub-intervals),
    each element contains p^3 micro-hexahedral cells with 8 vertices each.
    """
    p = order_N - 1
    if p < 1 or num_elements <= 0:
        return None, 0
    cells_per_elem = p * p * p
    total_cells = num_elements * cells_per_elem
    nPtsPerElem = order_N * order_N * order_N

    conn = np.empty((total_cells, 8), dtype=np.int64)
    cell_idx = 0
    for e in range(num_elements):
        elemOffset = e * nPtsPerElem
        for ck in range(p):
            for cj in range(p):
                for ci in range(p):
                    conn[cell_idx] = [
                        elemOffset + ci     + order_N * (cj     + order_N * ck),
                        elemOffset + (ci+1) + order_N * (cj     + order_N * ck),
                        elemOffset + (ci+1) + order_N * (cj+1   + order_N * ck),
                        elemOffset + ci     + order_N * (cj+1   + order_N * ck),
                        elemOffset + ci     + order_N * (cj     + order_N * (ck+1)),
                        elemOffset + (ci+1) + order_N * (cj     + order_N * (ck+1)),
                        elemOffset + (ci+1) + order_N * (cj+1   + order_N * (ck+1)),
                        elemOffset + ci     + order_N * (cj+1   + order_N * (ck+1)),
                    ]
                    cell_idx += 1
    return conn, total_cells

def convert_bin_to_h5(bin_path, h5_path, xmf_path=None):
    if h5py is None or np is None:
        print("Error: numpy and h5py are required to generate .h5 files.")
        print("Install via: pip install numpy h5py (or 'module load cray-python' / conda)")
        sys.exit(1)

    print(f"[CONVERT] Reading binary archive: {bin_path}")
    with open(bin_path, "rb") as f:
        magic = f.read(16).decode("utf-8", errors="ignore").strip("\x00")
        if not magic.startswith("NEKWAVE_BIN"):
            raise ValueError(f"Invalid magic header in {bin_path}: {magic}")
        is_v2 = magic.startswith("NEKWAVE_BIN_V2")

        header_bytes = f.read(16)
        order_N, num_elements, npts, _ = struct.unpack("<4i", header_bytes)
        print(f"          Format={magic}, Order N={order_N}, Elements={num_elements}, Points={npts}")

        coord_bytes = f.read(npts * 3 * 8)
        coords = np.frombuffer(coord_bytes, dtype=np.float64).reshape((npts, 3))

        conn, total_cells = generate_hex_connectivity(order_N, num_elements)
        use_hex = (conn is not None and total_cells > 0)
        if use_hex:
            print(f"          Generated {total_cells} 3D micro-hexahedral cells for volumetric visualization.")

        has_any_derived = False

        with h5py.File(h5_path, "w") as h5:
            h5.attrs["nekwave_version"] = "2.0" if is_v2 else "1.0"
            h5.attrs["polynomial_order_N"] = order_N
            h5.attrs["num_elements"] = num_elements
            h5.attrs["total_points"] = npts
            h5.attrs["total_cells"] = total_cells if use_hex else npts

            mesh_grp = h5.create_group("mesh")
            mesh_grp.create_dataset("coordinates", data=coords, dtype="float64")
            if use_hex:
                mesh_grp.create_dataset("connectivity", data=conn, dtype="int64")

            ts_grp = h5.create_group("time_series")

            step_times = []
            step_count = 0

            while True:
                step_hdr = f.read(8)
                if not step_hdr or len(step_hdr) < 8:
                    break
                step_idx, _ = struct.unpack("<2i", step_hdr)
                time_val = struct.unpack("<d", f.read(8))[0]

                E_bytes = f.read(npts * 3 * 8)
                H_bytes = f.read(npts * 3 * 8)
                if len(E_bytes) < npts * 3 * 8 or len(H_bytes) < npts * 3 * 8:
                    break

                E = np.frombuffer(E_bytes, dtype=np.float64).reshape((npts, 3))
                H = np.frombuffer(H_bytes, dtype=np.float64).reshape((npts, 3))

                mag_E = np.sqrt(np.sum(E**2, axis=1))
                mag_H = np.sqrt(np.sum(H**2, axis=1))
                energy = 0.5 * (np.sum(E**2, axis=1) + np.sum(H**2, axis=1))

                step_grp = ts_grp.create_group(f"step_{step_idx}")
                step_grp.attrs["step"] = step_idx
                step_grp.attrs["time"] = time_val

                step_grp.create_dataset("E", data=E, dtype="float64")
                step_grp.create_dataset("H", data=H, dtype="float64")
                step_grp.create_dataset("magnitude_E", data=mag_E, dtype="float64")
                step_grp.create_dataset("magnitude_H", data=mag_H, dtype="float64")
                step_grp.create_dataset("energy_density", data=energy, dtype="float64")

                if is_v2:
                    has_derived_bytes = f.read(4)
                    if has_derived_bytes and len(has_derived_bytes) == 4:
                        has_derived = struct.unpack("<i", has_derived_bytes)[0]
                        if has_derived:
                            has_any_derived = True
                            div_E = np.frombuffer(f.read(npts * 8), dtype=np.float64)
                            div_H = np.frombuffer(f.read(npts * 8), dtype=np.float64)
                            curl_E = np.frombuffer(f.read(npts * 3 * 8), dtype=np.float64).reshape((npts, 3))
                            curl_H = np.frombuffer(f.read(npts * 3 * 8), dtype=np.float64).reshape((npts, 3))

                            step_grp.create_dataset("div_E", data=div_E, dtype="float64")
                            step_grp.create_dataset("div_H", data=div_H, dtype="float64")
                            step_grp.create_dataset("curl_E", data=curl_E, dtype="float64")
                            step_grp.create_dataset("curl_H", data=curl_H, dtype="float64")

                step_times.append((step_idx, time_val))
                step_count += 1

            print(f"[CONVERT] Successfully wrote {step_count} time steps to: {h5_path}")

    if xmf_path:
        write_xdmf(xmf_path, os.path.basename(h5_path), npts, step_times,
                   has_derived=has_any_derived, total_cells=total_cells, use_hex=use_hex)
        print(f"[CONVERT] Generated companion XDMF descriptor: {xmf_path}")

def update_existing_h5(h5_path, xmf_path=None):
    """
    Checks an existing .h5 file: adds /mesh/connectivity if missing,
    and updates/regenerates the companion .xmf file with Hexahedron topology.
    """
    if h5py is None or np is None:
        print("Error: numpy and h5py are required to inspect/update .h5 files.")
        sys.exit(1)

    print(f"[UPDATE] Inspecting existing HDF5 file: {h5_path}")
    with h5py.File(h5_path, "r+") as h5:
        order_N = int(h5.attrs.get("polynomial_order_N", 0))
        num_elements = int(h5.attrs.get("num_elements", 0))
        npts = int(h5.attrs.get("total_points", 0))

        if npts == 0 and "mesh/coordinates" in h5:
            npts = h5["mesh/coordinates"].shape[0]

        mesh_grp = h5["mesh"] if "mesh" in h5 else h5.create_group("mesh")

        conn, total_cells = generate_hex_connectivity(order_N, num_elements)
        use_hex = (conn is not None and total_cells > 0)

        if use_hex:
            if "connectivity" not in mesh_grp:
                print(f"[UPDATE] Adding missing /mesh/connectivity ({total_cells} cells)...")
                mesh_grp.create_dataset("connectivity", data=conn, dtype="int64")
            else:
                print(f"[UPDATE] Existing /mesh/connectivity found ({mesh_grp['connectivity'].shape[0]} cells).")
                total_cells = mesh_grp["connectivity"].shape[0]
            h5.attrs["total_cells"] = total_cells

        step_times = []
        has_derived = False
        if "time_series" in h5:
            ts_grp = h5["time_series"]
            for step_key in sorted(ts_grp.keys(), key=lambda k: int(k.replace("step_", "")) if k.startswith("step_") else 0):
                sgrp = ts_grp[step_key]
                step_idx = int(sgrp.attrs.get("step", int(step_key.replace("step_", "")) if step_key.startswith("step_") else 0))
                time_val = float(sgrp.attrs.get("time", 0.0))
                step_times.append((step_idx, time_val))
                if "div_E" in sgrp or "curl_E" in sgrp:
                    has_derived = True

    if not xmf_path:
        xmf_path = os.path.splitext(h5_path)[0] + ".xmf"

    write_xdmf(xmf_path, os.path.basename(h5_path), npts, step_times,
               has_derived=has_derived, total_cells=total_cells, use_hex=use_hex)
    print(f"[UPDATE] Successfully regenerated XDMF descriptor: {xmf_path}")

def write_xdmf(xmf_path, h5_basename, npts, step_times, has_derived=False, total_cells=0, use_hex=True):
    lines = [
        '<?xml version="1.0" ?>',
        '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>',
        '<Xdmf Version="3.0">',
        '  <Domain>',
        '    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">'
    ]

    for step_idx, time_val in step_times:
        lines.append(f'      <Grid Name="step_{step_idx}" GridType="Uniform">')
        lines.append(f'        <Time Value="{time_val:.8e}"/>')

        if use_hex and total_cells > 0:
            lines.extend([
                f'        <Topology TopologyType="Hexahedron" NumberOfElements="{total_cells}">',
                f'          <DataItem Dimensions="{total_cells} 8" NumberType="Int" Precision="8" Format="HDF">',
                f'            {h5_basename}:/mesh/connectivity',
                '          </DataItem>',
                '        </Topology>'
            ])
        else:
            lines.append(f'        <Topology TopologyType="Polyvertex" NumberOfElements="{npts}"/>')

        lines.extend([
            '        <Geometry GeometryType="XYZ">',
            f'          <DataItem Dimensions="{npts} 3" NumberType="Float" Precision="8" Format="HDF">',
            f'            {h5_basename}:/mesh/coordinates',
            '          </DataItem>',
            '        </Geometry>',
            '        <Attribute Name="E" AttributeType="Vector" Center="Node">',
            f'          <DataItem Dimensions="{npts} 3" NumberType="Float" Precision="8" Format="HDF">',
            f'            {h5_basename}:/time_series/step_{step_idx}/E',
            '          </DataItem>',
            '        </Attribute>',
            '        <Attribute Name="H" AttributeType="Vector" Center="Node">',
            f'          <DataItem Dimensions="{npts} 3" NumberType="Float" Precision="8" Format="HDF">',
            f'            {h5_basename}:/time_series/step_{step_idx}/H',
            '          </DataItem>',
            '        </Attribute>',
            '        <Attribute Name="magnitude_E" AttributeType="Scalar" Center="Node">',
            f'          <DataItem Dimensions="{npts}" NumberType="Float" Precision="8" Format="HDF">',
            f'            {h5_basename}:/time_series/step_{step_idx}/magnitude_E',
            '          </DataItem>',
            '        </Attribute>',
            '        <Attribute Name="magnitude_H" AttributeType="Scalar" Center="Node">',
            f'          <DataItem Dimensions="{npts}" NumberType="Float" Precision="8" Format="HDF">',
            f'            {h5_basename}:/time_series/step_{step_idx}/magnitude_H',
            '          </DataItem>',
            '        </Attribute>',
            '        <Attribute Name="energy_density" AttributeType="Scalar" Center="Node">',
            f'          <DataItem Dimensions="{npts}" NumberType="Float" Precision="8" Format="HDF">',
            f'            {h5_basename}:/time_series/step_{step_idx}/energy_density',
            '          </DataItem>',
            '        </Attribute>',
        ])

        if has_derived:
            lines.extend([
                '        <Attribute Name="div_E" AttributeType="Scalar" Center="Node">',
                f'          <DataItem Dimensions="{npts}" NumberType="Float" Precision="8" Format="HDF">',
                f'            {h5_basename}:/time_series/step_{step_idx}/div_E',
                '          </DataItem>',
                '        </Attribute>',
                '        <Attribute Name="div_H" AttributeType="Scalar" Center="Node">',
                f'          <DataItem Dimensions="{npts}" NumberType="Float" Precision="8" Format="HDF">',
                f'            {h5_basename}:/time_series/step_{step_idx}/div_H',
                '          </DataItem>',
                '        </Attribute>',
                '        <Attribute Name="curl_E" AttributeType="Vector" Center="Node">',
                f'          <DataItem Dimensions="{npts} 3" NumberType="Float" Precision="8" Format="HDF">',
                f'            {h5_basename}:/time_series/step_{step_idx}/curl_E',
                '          </DataItem>',
                '        </Attribute>',
                '        <Attribute Name="curl_H" AttributeType="Vector" Center="Node">',
                f'          <DataItem Dimensions="{npts} 3" NumberType="Float" Precision="8" Format="HDF">',
                f'            {h5_basename}:/time_series/step_{step_idx}/curl_H',
                '          </DataItem>',
                '        </Attribute>',
            ])

        lines.append('      </Grid>')

    lines.extend([
        '    </Grid>',
        '  </Domain>',
        '</Xdmf>'
    ])

    with open(xmf_path, "w") as f:
        f.write("\n".join(lines) + "\n")

def main():
    parser = argparse.ArgumentParser(description="NekWave HDF5 Output Converter")
    parser.add_argument("input_path", nargs="?", default="output", help="Output directory, fields.bin, or fields.h5")
    parser.add_argument("-o", "--output", help="Target .h5 output path")

    args = parser.parse_args()

    input_path = args.input_path
    if os.path.isdir(input_path):
        bin_path = os.path.join(input_path, "fields.bin")
        h5_path = args.output if args.output else os.path.join(input_path, "fields.h5")
        xmf_path = os.path.join(input_path, "fields.xmf")
        if os.path.exists(bin_path):
            convert_bin_to_h5(bin_path, h5_path, xmf_path)
        elif os.path.exists(h5_path):
            update_existing_h5(h5_path, xmf_path)
        else:
            print(f"Error: Neither fields.bin nor fields.h5 found in {input_path}")
            sys.exit(1)
    elif os.path.isfile(input_path):
        if input_path.endswith(".h5"):
            xmf_path = os.path.splitext(input_path)[0] + ".xmf"
            update_existing_h5(input_path, xmf_path)
        else:
            bin_path = input_path
            h5_path = args.output if args.output else os.path.splitext(input_path)[0] + ".h5"
            xmf_path = os.path.splitext(h5_path)[0] + ".xmf"
            convert_bin_to_h5(bin_path, h5_path, xmf_path)
    else:
        print(f"Error: Path {input_path} not found.")
        sys.exit(1)

if __name__ == "__main__":
    main()
