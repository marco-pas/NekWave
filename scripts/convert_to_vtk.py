#!/usr/bin/env python3
"""
NekWave VTK Conversion Utility
Converts NekWave binary field archives (.bin) into ParaView-native
VTK Collection (.pvd) and Unstructured Grid (.vtu) files with 3D
micro-hexahedral cells for volumetric slicing and cutting.

Usage:
    python3 tools/convert_to_vtk.py [output_dir]
    python3 tools/convert_to_vtk.py output/fields.bin -o output
"""

import sys
import os
import struct
import math
import argparse

def convert_bin_to_vtu(bin_path, out_dir):
    if not os.path.exists(bin_path):
        print(f"Error: Binary file not found: {bin_path}")
        return False

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(bin_path))[0]

    with open(bin_path, "rb") as f:
        magic = f.read(16).decode("utf-8", errors="ignore").strip("\x00")
        if not magic.startswith("NEKWAVE_BIN"):
            raise ValueError(f"Invalid magic identifier in {bin_path}: {magic}")

        header = struct.unpack("<4i", f.read(16))
        order_N, num_elements, npts, _ = header
        print(f"[VTK] Reading {bin_path}: Order N={order_N}, Elements={num_elements}, Points={npts}")

        p = order_N - 1
        use_hex = (p >= 1 and num_elements > 0)
        cells_per_elem = p * p * p
        total_cells = (num_elements * cells_per_elem) if use_hex else npts
        nPtsPerElem = order_N * order_N * order_N

        coords = struct.unpack(f"<{npts * 3}d", f.read(npts * 3 * 8))
        coord_lines = [f"          {coords[3*i]} {coords[3*i+1]} {coords[3*i+2]}" for i in range(npts)]
        coord_str = "\n".join(coord_lines)

        # Build connectivity, offsets, types
        conn_lines = []
        if use_hex:
            for e in range(num_elements):
                elemOffset = e * nPtsPerElem
                for ck in range(p):
                    for cj in range(p):
                        for ci in range(p):
                            p0 = elemOffset + ci     + order_N * (cj     + order_N * ck)
                            p1 = elemOffset + (ci+1) + order_N * (cj     + order_N * ck)
                            p2 = elemOffset + (ci+1) + order_N * (cj+1   + order_N * ck)
                            p3 = elemOffset + ci     + order_N * (cj+1   + order_N * ck)
                            p4 = elemOffset + ci     + order_N * (cj     + order_N * (ck+1))
                            p5 = elemOffset + (ci+1) + order_N * (cj     + order_N * (ck+1))
                            p6 = elemOffset + (ci+1) + order_N * (cj+1   + order_N * (ck+1))
                            p7 = elemOffset + ci     + order_N * (cj+1   + order_N * (ck+1))
                            conn_lines.append(f"          {p0} {p1} {p2} {p3} {p4} {p5} {p6} {p7}")
            conn_str = "\n".join(conn_lines)
            offset_str = " ".join(str(8 * (i + 1)) for i in range(total_cells))
            type_str = " ".join("12" for _ in range(total_cells)) # 12 = VTK_HEXAHEDRON
        else:
            conn_str = " ".join(str(i) for i in range(npts))
            offset_str = " ".join(str(i + 1) for i in range(npts))
            type_str = " ".join("1" for _ in range(npts))

        is_v2 = magic.startswith("NEKWAVE_BIN_V2")

        pvd_entries = []
        step_count = 0

        while True:
            step_header = f.read(8)
            if not step_header or len(step_header) < 8:
                break

            step, _ = struct.unpack("<2i", step_header)
            time = struct.unpack("<d", f.read(8))[0]
            bufE = struct.unpack(f"<{npts * 3}d", f.read(npts * 3 * 8))
            bufH = struct.unpack(f"<{npts * 3}d", f.read(npts * 3 * 8))

            bufDivE, bufDivH, bufCurlE, bufCurlH = None, None, None, None
            if is_v2:
                has_derived_bytes = f.read(4)
                if has_derived_bytes and len(has_derived_bytes) == 4:
                    has_derived = struct.unpack("<i", has_derived_bytes)[0]
                    if has_derived:
                        bufDivE = struct.unpack(f"<{npts}d", f.read(npts * 8))
                        bufDivH = struct.unpack(f"<{npts}d", f.read(npts * 8))
                        bufCurlE = struct.unpack(f"<{npts * 3}d", f.read(npts * 3 * 8))
                        bufCurlH = struct.unpack(f"<{npts * 3}d", f.read(npts * 3 * 8))

            vtu_filename = f"{stem}_step_{step}.vtu"
            vtu_path = os.path.join(out_dir, vtu_filename)

            e_lines = [f"          {bufE[3*i]} {bufE[3*i+1]} {bufE[3*i+2]}" for i in range(npts)]
            h_lines = [f"          {bufH[3*i]} {bufH[3*i+1]} {bufH[3*i+2]}" for i in range(npts)]
            magE_lines = [f"          {math.sqrt(bufE[3*i]**2 + bufE[3*i+1]**2 + bufE[3*i+2]**2)}" for i in range(npts)]
            magH_lines = [f"          {math.sqrt(bufH[3*i]**2 + bufH[3*i+1]**2 + bufH[3*i+2]**2)}" for i in range(npts)]
            energy_lines = [f"          {0.5*(bufE[3*i]**2 + bufE[3*i+1]**2 + bufE[3*i+2]**2 + bufH[3*i]**2 + bufH[3*i+1]**2 + bufH[3*i+2]**2)}" for i in range(npts)]

            with open(vtu_path, "w") as vtu:
                vtu.write('<?xml version="1.0"?>\n')
                vtu.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
                vtu.write('  <UnstructuredGrid>\n')
                vtu.write(f'    <Piece NumberOfPoints="{npts}" NumberOfCells="{total_cells}">\n')
                vtu.write('      <Points>\n')
                vtu.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
                vtu.write(coord_str + "\n")
                vtu.write('        </DataArray>\n')
                vtu.write('      </Points>\n')
                vtu.write('      <PointData Vectors="E" Scalars="magnitude_E">\n')
                vtu.write('        <DataArray type="Float64" Name="E" NumberOfComponents="3" format="ascii">\n')
                vtu.write("\n".join(e_lines) + "\n")
                vtu.write('        </DataArray>\n')
                vtu.write('        <DataArray type="Float64" Name="H" NumberOfComponents="3" format="ascii">\n')
                vtu.write("\n".join(h_lines) + "\n")
                vtu.write('        </DataArray>\n')
                if bufCurlE is not None:
                    ce_lines = [f"          {bufCurlE[3*i]} {bufCurlE[3*i+1]} {bufCurlE[3*i+2]}" for i in range(npts)]
                    magCE_lines = [f"          {math.sqrt(bufCurlE[3*i]**2 + bufCurlE[3*i+1]**2 + bufCurlE[3*i+2]**2)}" for i in range(npts)]
                    vtu.write('        <DataArray type="Float64" Name="curl_E" NumberOfComponents="3" format="ascii">\n')
                    vtu.write("\n".join(ce_lines) + "\n")
                    vtu.write('        </DataArray>\n')
                    vtu.write('        <DataArray type="Float64" Name="magnitude_curl_E" NumberOfComponents="1" format="ascii">\n')
                    vtu.write("\n".join(magCE_lines) + "\n")
                    vtu.write('        </DataArray>\n')
                if bufCurlH is not None:
                    ch_lines = [f"          {bufCurlH[3*i]} {bufCurlH[3*i+1]} {bufCurlH[3*i+2]}" for i in range(npts)]
                    magCH_lines = [f"          {math.sqrt(bufCurlH[3*i]**2 + bufCurlH[3*i+1]**2 + bufCurlH[3*i+2]**2)}" for i in range(npts)]
                    vtu.write('        <DataArray type="Float64" Name="curl_H" NumberOfComponents="3" format="ascii">\n')
                    vtu.write("\n".join(ch_lines) + "\n")
                    vtu.write('        </DataArray>\n')
                    vtu.write('        <DataArray type="Float64" Name="magnitude_curl_H" NumberOfComponents="1" format="ascii">\n')
                    vtu.write("\n".join(magCH_lines) + "\n")
                    vtu.write('        </DataArray>\n')
                if bufDivE is not None:
                    de_lines = [f"          {bufDivE[i]}" for i in range(npts)]
                    vtu.write('        <DataArray type="Float64" Name="div_E" NumberOfComponents="1" format="ascii">\n')
                    vtu.write("\n".join(de_lines) + "\n")
                    vtu.write('        </DataArray>\n')
                if bufDivH is not None:
                    dh_lines = [f"          {bufDivH[i]}" for i in range(npts)]
                    vtu.write('        <DataArray type="Float64" Name="div_H" NumberOfComponents="1" format="ascii">\n')
                    vtu.write("\n".join(dh_lines) + "\n")
                    vtu.write('        </DataArray>\n')
                vtu.write('        <DataArray type="Float64" Name="magnitude_E" NumberOfComponents="1" format="ascii">\n')
                vtu.write("\n".join(magE_lines) + "\n")
                vtu.write('        </DataArray>\n')
                vtu.write('        <DataArray type="Float64" Name="magnitude_H" NumberOfComponents="1" format="ascii">\n')
                vtu.write("\n".join(magH_lines) + "\n")
                vtu.write('        </DataArray>\n')
                vtu.write('        <DataArray type="Float64" Name="energy_density" NumberOfComponents="1" format="ascii">\n')
                vtu.write("\n".join(energy_lines) + "\n")
                vtu.write('        </DataArray>\n')
                vtu.write('      </PointData>\n')
                vtu.write('      <Cells>\n')
                vtu.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
                vtu.write(conn_str + "\n")
                vtu.write('        </DataArray>\n')
                vtu.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
                vtu.write(offset_str + "\n")
                vtu.write('        </DataArray>\n')
                vtu.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
                vtu.write(type_str + "\n")
                vtu.write('        </DataArray>\n')
                vtu.write('      </Cells>\n')
                vtu.write('    </Piece>\n')
                vtu.write('  </UnstructuredGrid>\n')
                vtu.write('</VTKFile>\n')

            pvd_entries.append((time, vtu_filename))
            step_count += 1

        pvd_path = os.path.join(out_dir, f"{stem}.pvd")
        with open(pvd_path, "w") as pvd:
            pvd.write('<?xml version="1.0"?>\n')
            pvd.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
            pvd.write('  <Collection>\n')
            for t, fname in pvd_entries:
                pvd.write(f'    <DataSet timestep="{t:.8e}" group="" part="0" file="{fname}"/>\n')
            pvd.write('  </Collection>\n')
            pvd.write('</VTKFile>\n')

        print(f"[VTK] Successfully created {step_count} 3D volumetric VTU time-steps ({total_cells} hex cells each):")
        print(f"      -> {pvd_path}")
        print("      Open this file in ParaView: File -> Open -> fields.pvd")
        print("      Then apply the Slice or Clip filter to cut through the 3D volume!")
        return True

def main():
    parser = argparse.ArgumentParser(description="Convert NekWave binary archive to ParaView VTK collection.")
    parser.add_argument("input", nargs="?", default="output/fields.bin", help="Path to input .bin file or output directory")
    parser.add_argument("-o", "--output", default=None, help="Target output directory for .vtu and .pvd files")
    args = parser.parse_args()

    bin_path = args.input
    if os.path.isdir(bin_path):
        out_dir = bin_path
        bin_path = os.path.join(out_dir, "fields.bin")
    else:
        out_dir = args.output or os.path.dirname(bin_path) or "."

    convert_bin_to_vtu(bin_path, out_dir)

if __name__ == "__main__":
    main()
