#!/usr/bin/env python3
"""
NekWave HDF5 Conversion Utility
Converts NekWave binary field archives (.bin) or CSV field exports into
standard HDF5 (.h5) datasets with companion XDMF (.xmf) for ParaView / VisIt.

Usage:
    python3 scripts/convert_to_hdf5.py [output_directory]
    python3 scripts/convert_to_hdf5.py output/fields.bin -o output/fields.h5
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

def convert_bin_to_h5(bin_path, h5_path, xmf_path=None):
    if h5py is None or np is None:
        print("Error: numpy and h5py are required to generate .h5 files. Please install via: pip install numpy h5py (or 'module load cray-python')")
        sys.exit(1)

    print(f"[CONVERT] Reading binary archive: {bin_path}")
    with open(bin_path, "rb") as f:
        # Read header: 16 bytes magic + 16 bytes (4 x int32)
        magic = f.read(16).decode("utf-8", errors="ignore").strip("\x00")
        if not magic.startswith("NEKWAVE_BIN"):
            raise ValueError(f"Invalid magic header in {bin_path}: {magic}")

        header_bytes = f.read(16)
        order_N, num_elements, npts, _ = struct.unpack("<4i", header_bytes)
        print(f"          Order N={order_N}, Elements={num_elements}, Points={npts}")

        # Read coordinates [npts, 3] (float64)
        coord_bytes = f.read(npts * 3 * 8)
        coords = np.frombuffer(coord_bytes, dtype=np.float64).reshape((npts, 3))

        # Open HDF5 output file
        with h5py.File(h5_path, "w") as h5:
            h5.attrs["nekwave_version"] = "1.0"
            h5.attrs["polynomial_order_N"] = order_N
            h5.attrs["num_elements"] = num_elements
            h5.attrs["total_points"] = npts

            mesh_grp = h5.create_group("mesh")
            mesh_grp.create_dataset("coordinates", data=coords, dtype="float64")

            ts_grp = h5.create_group("time_series")

            step_times = []
            step_count = 0

            # Read steps sequentially
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

                step_grp = ts_grp.create_group(f"step_{step_idx}")
                step_grp.attrs["step"] = step_idx
                step_grp.attrs["time"] = time_val

                step_grp.create_dataset("E", data=E, dtype="float64")
                step_grp.create_dataset("H", data=H, dtype="float64")

                step_times.append((step_idx, time_val))
                step_count += 1

            print(f"[CONVERT] Successfully wrote {step_count} time steps to: {h5_path}")

    # Generate companion XDMF descriptor
    if xmf_path:
        write_xdmf(xmf_path, os.path.basename(h5_path), npts, step_times)
        print(f"[CONVERT] Generated companion XDMF descriptor: {xmf_path}")

def write_xdmf(xmf_path, h5_basename, npts, step_times):
    lines = [
        '<?xml version="1.0" ?>',
        '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>',
        '<Xdmf Version="3.0">',
        '  <Domain>',
        '    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">'
    ]

    for step_idx, time_val in step_times:
        lines.extend([
            f'      <Grid Name="step_{step_idx}" GridType="Uniform">',
            f'        <Time Value="{time_val:.8e}"/>',
            f'        <Topology TopologyType="Polyvertex" NumberOfElements="{npts}"/>',
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
            '      </Grid>'
        ])

    lines.extend([
        '    </Grid>',
        '  </Domain>',
        '</Xdmf>'
    ])

    with open(xmf_path, "w") as f:
        f.write("\n".join(lines) + "\n")

def main():
    parser = argparse.ArgumentParser(description="NekWave HDF5 Output Converter")
    parser.add_argument("input_path", nargs="?", default="output", help="Output directory or input .bin file")
    parser.add_argument("-o", "--output", help="Target .h5 output path")

    args = parser.parse_args()

    input_path = args.input_path
    if os.path.isdir(input_path):
        bin_path = os.path.join(input_path, "fields.bin")
        h5_path = args.output if args.output else os.path.join(input_path, "fields.h5")
        xmf_path = os.path.join(input_path, "fields.xmf")
    elif os.path.isfile(input_path):
        bin_path = input_path
        h5_path = args.output if args.output else os.path.splitext(input_path)[0] + ".h5"
        xmf_path = os.path.splitext(h5_path)[0] + ".xmf"
    else:
        print(f"Error: Path {input_path} not found.")
        sys.exit(1)

    if os.path.exists(bin_path):
        convert_bin_to_h5(bin_path, h5_path, xmf_path)
    else:
        print(f"Error: Binary field file {bin_path} not found.")
        sys.exit(1)

if __name__ == "__main__":
    main()
