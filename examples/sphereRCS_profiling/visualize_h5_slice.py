#!/usr/bin/env python3
"""
2D Slice Field Visualizer & MP4 Video Generator for NekWave HDF5 Outputs.

Reads multi-rank (`fields_rank*.h5`) or single-rank (`fields.h5`) NekWave HDF5
archives, extracts a high-resolution 2D planar cut through the 3D GLL collocation
nodes (default: y = 0 E-plane slice through the PEC sphere), and generates:
  1. An MP4 animation (`sphere_wave_magE_xz.mp4`) of the electric field magnitude |E|
     (plus an animated GIF and AVI if ffmpeg is unavailable on the login node).
  2. A 6-panel snapshot montage (`sphere_wave_magE_xz_snapshots.png`) showing the
     incident wave hitting the PEC sphere, specular reflection, creeping waves,
     and the geometric shadow behind the sphere.

Usage:
  python3 examples/sphereRCS/visualize_h5_slice.py
  python3 examples/sphereRCS/visualize_h5_slice.py --input-dir build/output_sphereRCS --plane xz --fps 10
"""

import glob
import os
import sys

# ------------------------------------------------------------------------------
# Auto-detect Python environment with matplotlib/numpy on LUMI if needed
# ------------------------------------------------------------------------------
try:
    import numpy as np
    import matplotlib
except ImportError:
    if os.environ.get("NEKWAVE_PYTHON_REEXEC") != "1":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
        candidates = [
            os.path.join(repo_root, ".venv", "bin", "python3"),
            os.path.join(repo_root, "..", ".venv", "bin", "python3"),
            os.path.expanduser("~/.venv/bin/python3"),
        ]
        candidates.extend(sorted(glob.glob("/opt/cray/pe/python/*/bin/python3"), reverse=True))
        for cand in candidates:
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                os.environ["NEKWAVE_PYTHON_REEXEC"] = "1"
                os.execv(cand, [cand] + sys.argv)
    raise

import argparse
import ctypes
import io
import math
import re
import shutil
import struct

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

try:
    import h5py
    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False

try:
    from scipy.spatial import cKDTree
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


# ==============================================================================
# Fallback HDF5 Reader via Cray/System libhdf5.so (when h5py is not installed)
# ==============================================================================
class CtypesHdf5Reader:
    """Minimal read-only HDF5 wrapper using ctypes and libhdf5.so when h5py is absent."""

    _CACHED_LIB = None
    _H5T_NATIVE_DOUBLE = None
    _H5T_NATIVE_INT = None

    def __init__(self, filepath):
        self.filepath = filepath
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        self.lib = self._get_initialized_lib()
        self.H5T_NATIVE_DOUBLE = CtypesHdf5Reader._H5T_NATIVE_DOUBLE
        self.H5T_NATIVE_INT = CtypesHdf5Reader._H5T_NATIVE_INT
        self.file_id = self.lib.H5Fopen(filepath.encode("utf-8"), 0x0000, 0)  # H5F_ACC_RDONLY=0, H5P_DEFAULT=0
        if self.file_id < 0:
            raise IOError(f"Could not open HDF5 file via libhdf5: {filepath}")

    @classmethod
    def _get_initialized_lib(cls):
        if cls._CACHED_LIB is not None:
            return cls._CACHED_LIB

        search_paths = [
            "/opt/cray/pe/hdf5-parallel/1.12.2.11/crayclang/17.0/lib/libhdf5_parallel.so",
            "/opt/cray/pe/hdf5-parallel/1.12.2.11/AMD/5.0/lib/libhdf5_parallel.so",
        ]
        search_paths.extend(sorted(glob.glob("/opt/cray/pe/hdf5*/default/*/lib/libhdf5*.so"), reverse=True))
        search_paths.extend(sorted(glob.glob("/opt/cray/pe/hdf5*/*/*/lib/libhdf5*.so"), reverse=True))
        search_paths.extend(["libhdf5.so", "libhdf5_serial.so", "libhdf5_parallel.so"])

        lib = None
        for p in search_paths:
            try:
                lib = ctypes.CDLL(p)
                break
            except OSError:
                continue
        if lib is None:
            raise RuntimeError("Neither h5py nor libhdf5.so could be loaded.")

        lib.H5open.argtypes = []
        lib.H5open.restype = ctypes.c_int
        lib.H5open()  # Must call H5open() before reading H5T_NATIVE_*_g globals!

        lib.H5Fopen.argtypes = [ctypes.c_char_p, ctypes.c_uint, ctypes.c_int64]
        lib.H5Fopen.restype = ctypes.c_int64
        lib.H5Fclose.argtypes = [ctypes.c_int64]
        lib.H5Fclose.restype = ctypes.c_int

        lib.H5Dopen2.argtypes = [ctypes.c_int64, ctypes.c_char_p, ctypes.c_int64]
        lib.H5Dopen2.restype = ctypes.c_int64
        lib.H5Dclose.argtypes = [ctypes.c_int64]
        lib.H5Dclose.restype = ctypes.c_int
        lib.H5Dget_space.argtypes = [ctypes.c_int64]
        lib.H5Dget_space.restype = ctypes.c_int64
        lib.H5Dread.argtypes = [
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p
        ]
        lib.H5Dread.restype = ctypes.c_int

        lib.H5Sget_simple_extent_ndims.argtypes = [ctypes.c_int64]
        lib.H5Sget_simple_extent_ndims.restype = ctypes.c_int
        lib.H5Sget_simple_extent_dims.argtypes = [
            ctypes.c_int64, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)
        ]
        lib.H5Sget_simple_extent_dims.restype = ctypes.c_int
        lib.H5Sclose.argtypes = [ctypes.c_int64]
        lib.H5Sclose.restype = ctypes.c_int

        lib.H5Gopen2.argtypes = [ctypes.c_int64, ctypes.c_char_p, ctypes.c_int64]
        lib.H5Gopen2.restype = ctypes.c_int64
        lib.H5Gclose.argtypes = [ctypes.c_int64]
        lib.H5Gclose.restype = ctypes.c_int
        lib.H5Gget_num_objs.argtypes = [ctypes.c_int64, ctypes.POINTER(ctypes.c_uint64)]
        lib.H5Gget_num_objs.restype = ctypes.c_int
        lib.H5Gget_objname_by_idx.argtypes = [
            ctypes.c_int64, ctypes.c_uint64, ctypes.c_char_p, ctypes.c_size_t
        ]
        lib.H5Gget_objname_by_idx.restype = ctypes.c_ssize_t

        lib.H5Aopen.argtypes = [ctypes.c_int64, ctypes.c_char_p, ctypes.c_int64]
        lib.H5Aopen.restype = ctypes.c_int64
        lib.H5Aread.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        lib.H5Aread.restype = ctypes.c_int
        lib.H5Aclose.argtypes = [ctypes.c_int64]
        lib.H5Aclose.restype = ctypes.c_int

        lib.H5Lexists.argtypes = [ctypes.c_int64, ctypes.c_char_p, ctypes.c_int64]
        lib.H5Lexists.restype = ctypes.c_int

        cls._H5T_NATIVE_DOUBLE = ctypes.c_int64.in_dll(lib, "H5T_NATIVE_DOUBLE_g").value
        cls._H5T_NATIVE_INT = ctypes.c_int64.in_dll(lib, "H5T_NATIVE_INT_g").value
        cls._CACHED_LIB = lib
        return lib

    def has_path(self, path):
        return self.lib.H5Lexists(self.file_id, path.encode("utf-8"), 0) > 0

    def list_group(self, group_path):
        gid = self.lib.H5Gopen2(self.file_id, group_path.encode("utf-8"), 0)
        if gid < 0:
            return []
        n_objs = ctypes.c_uint64(0)
        self.lib.H5Gget_num_objs(gid, ctypes.byref(n_objs))
        names = []
        buf = ctypes.create_string_buffer(256)
        for i in range(n_objs.value):
            self.lib.H5Gget_objname_by_idx(gid, i, buf, 256)
            names.append(buf.value.decode("utf-8"))
        self.lib.H5Gclose(gid)
        return names

    def read_step_attrs(self, step_group_path):
        gid = self.lib.H5Gopen2(self.file_id, step_group_path.encode("utf-8"), 0)
        step_val = int(step_group_path.split("_")[-1])
        time_val = 0.0
        if gid >= 0:
            aid_t = self.lib.H5Aopen(gid, b"time", 0)
            if aid_t >= 0:
                t_c = ctypes.c_double(0.0)
                if self.lib.H5Aread(aid_t, self.H5T_NATIVE_DOUBLE, ctypes.byref(t_c)) >= 0:
                    time_val = float(t_c.value)
                self.lib.H5Aclose(aid_t)
            aid_s = self.lib.H5Aopen(gid, b"step", 0)
            if aid_s >= 0:
                s_c = ctypes.c_int(step_val)
                if self.lib.H5Aread(aid_s, self.H5T_NATIVE_INT, ctypes.byref(s_c)) >= 0:
                    step_val = int(s_c.value)
                self.lib.H5Aclose(aid_s)
            self.lib.H5Gclose(gid)
        return step_val, time_val

    def read_double_dataset(self, dset_path):
        did = self.lib.H5Dopen2(self.file_id, dset_path.encode("utf-8"), 0)
        if did < 0:
            raise KeyError(f"Dataset not found: {dset_path}")
        sid = self.lib.H5Dget_space(did)
        ndims = self.lib.H5Sget_simple_extent_ndims(sid)
        dims = (ctypes.c_uint64 * ndims)()
        self.lib.H5Sget_simple_extent_dims(sid, dims, None)
        shape = tuple(int(dims[i]) for i in range(ndims))
        arr = np.empty(shape, dtype=np.float64)
        self.lib.H5Dread(did, self.H5T_NATIVE_DOUBLE, 0, 0, 0, arr.ctypes.data_as(ctypes.c_void_p))
        self.lib.H5Sclose(sid)
        self.lib.H5Dclose(did)
        return arr

    def close(self):
        if self.file_id >= 0:
            self.lib.H5Fclose(self.file_id)
            self.file_id = -1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def ensure_ffmpeg_configured():
    """Finds ffmpeg in PATH or LUMI EasyBuild modules and configures LD_LIBRARY_PATH if needed."""
    if shutil.which("ffmpeg") is not None:
        return shutil.which("ffmpeg")

    fast_path = "/appl/lumi/SW/LUMI-24.03/L/EB/FFmpeg/6.1.1-cpeCray-24.03/bin/ffmpeg"
    if os.path.exists(fast_path):
        lumi_ffmpeg_candidates = [fast_path]
    else:
        lumi_ffmpeg_candidates = sorted(
            glob.glob("/appl/lumi/SW/LUMI-*/L/EB/FFmpeg/*/bin/ffmpeg") +
            glob.glob("/appl/lumi/SW/LUMI-*/C/EB/FFmpeg/*/bin/ffmpeg"),
            reverse=True
        )
    for ff_bin in lumi_ffmpeg_candidates:
        eb_root = os.path.abspath(os.path.join(os.path.dirname(ff_bin), "..", "..", ".."))
        lib_dirs = glob.glob(os.path.join(eb_root, "*", "*", "lib")) + glob.glob(os.path.join(eb_root, "*", "*", "lib64"))
        cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([cur_ld] if cur_ld else []))
        matplotlib.rcParams["animation.ffmpeg_path"] = ff_bin
        return ff_bin

    return None


def find_h5_files(input_dir):
    """Finds all rank HDF5 files (fields_rank*.h5) or single fields.h5 in input_dir."""
    rank_pattern = os.path.join(input_dir, "fields_rank*.h5")
    rank_files = sorted(
        glob.glob(rank_pattern),
        key=lambda p: int(re.search(r"rank(\d+)\.h5$", p).group(1))
        if re.search(r"rank(\d+)\.h5$", p) else 0
    )
    if rank_files:
        return rank_files

    single_file = os.path.join(input_dir, "fields.h5")
    if os.path.exists(single_file):
        return [single_file]

    return []


def collect_common_steps(h5_files):
    """Returns sorted list of (step_int, time_float, group_name) present across all rank files."""
    common_step_names = None
    step_time_map = {}

    for fpath in h5_files:
        if HAVE_H5PY:
            with h5py.File(fpath, "r") as f:
                if "time_series" not in f:
                    return []
                ts = f["time_series"]
                names = set(ts.keys())
                if common_step_names is None:
                    common_step_names = names
                    for name in names:
                        grp = ts[name]
                        step_val = int(grp.attrs.get("step", int(name.split("_")[-1])))
                        time_val = float(grp.attrs.get("time", 0.0))
                        step_time_map[name] = (step_val, time_val)
                else:
                    common_step_names &= names
        else:
            with CtypesHdf5Reader(fpath) as f:
                names = set(f.list_group("/time_series"))
                if common_step_names is None:
                    common_step_names = names
                    for name in names:
                        step_time_map[name] = f.read_step_attrs(f"/time_series/{name}")
                else:
                    common_step_names &= names

    if not common_step_names:
        return []

    steps = [
        (step_time_map[name][0], step_time_map[name][1], name)
        for name in common_step_names
    ]
    steps.sort(key=lambda item: item[0])
    return steps


def read_mesh_coordinates(fpath):
    if HAVE_H5PY:
        with h5py.File(fpath, "r") as f:
            return f["mesh/coordinates"][:]
    else:
        with CtypesHdf5Reader(fpath) as f:
            return f.read_double_dataset("/mesh/coordinates")


def read_order_N(fpath, npts):
    """Reads polynomial_order_N attribute from HDF5 root, or infers it from npts."""
    order_n = 0
    if HAVE_H5PY:
        with h5py.File(fpath, "r") as f:
            if "polynomial_order_N" in f.attrs:
                order_n = int(f.attrs["polynomial_order_N"])
    else:
        with CtypesHdf5Reader(fpath) as f:
            aid = f.lib.H5Aopen(f.file_id, b"polynomial_order_N", 0)
            if aid >= 0:
                val = ctypes.c_int(0)
                if f.lib.H5Aread(aid, f.H5T_NATIVE_INT, ctypes.byref(val)) >= 0:
                    order_n = int(val.value)
                f.lib.H5Aclose(aid)
    if order_n >= 2:
        return order_n
    for cand in (6, 5, 4, 7, 8, 3, 2):
        if (npts % (cand ** 3)) == 0:
            return cand
    return 1


def extract_element_gll_node_ids(fpath, order_n, npts):
    """
    Returns an integer array of shape (n_elem, N, N, N) mapping each element's
    tensor-product GLL nodes (k, j, i) to their index in /mesh/coordinates.
    Supports both discontinuous DG layout (npts == n_elem * N^3) and continuous
    CG layout (export_continuous: true) via /mesh/connectivity.
    """
    n3 = order_n ** 3
    if (npts % n3) == 0:
        n_elem = npts // n3
        return np.arange(npts, dtype=np.int64).reshape(n_elem, order_n, order_n, order_n)

    p = order_n - 1
    p3 = p ** 3
    conn = None
    if HAVE_H5PY:
        with h5py.File(fpath, "r") as f:
            if "/mesh/connectivity" in f:
                conn = np.asarray(f["/mesh/connectivity"][:], dtype=np.int64)
    else:
        with CtypesHdf5Reader(fpath) as f:
            if f.has_path("/mesh/connectivity"):
                conn = np.asarray(f.read_int_dataset("/mesh/connectivity"), dtype=np.int64)

    if conn is not None and conn.ndim == 2 and conn.shape[1] == 8 and (conn.shape[0] % p3) == 0:
        n_elem = conn.shape[0] // p3
        c = conn.reshape(n_elem, p, p, p, 8)
        node_ids = np.empty((n_elem, order_n, order_n, order_n), dtype=np.int64)
        node_ids[:, :p, :p, :p] = c[:, :, :, :, 0]
        node_ids[:, :p, :p, p]  = c[:, :, :, p - 1, 1]
        node_ids[:, :p, p, :p]  = c[:, :, p - 1, :, 3]
        node_ids[:, :p, p, p]   = c[:, :, p - 1, p - 1, 2]
        node_ids[:, p, :p, :p]  = c[:, p - 1, :, :, 4]
        node_ids[:, p, :p, p]   = c[:, p - 1, :, p - 1, 5]
        node_ids[:, p, p, :p]   = c[:, p - 1, p - 1, :, 7]
        node_ids[:, p, p, p]    = c[:, p - 1, p - 1, p - 1, 6]
        return node_ids
    return None


def compute_gll_lagrange_matrix(order_n, n_fine=24):
    """
    Computes the 1D Gauss-Lobatto-Legendre (GLL) nodes on [-1, 1] of order N
    and the exact degree-(N-1) Lagrange interpolation matrix L of shape (n_fine, order_n).
    """
    from numpy.polynomial.legendre import Legendre
    if order_n <= 1:
        return np.ones((1, 1), dtype=np.float64)
    if order_n == 2:
        xi_gll = np.array([-1.0, 1.0], dtype=np.float64)
    else:
        roots = np.sort(Legendre.basis(order_n - 1).deriv().roots())
        xi_gll = np.concatenate(([-1.0], roots, [1.0]))

    xi_fine = np.linspace(-1.0, 1.0, n_fine)
    L = np.ones((n_fine, order_n), dtype=np.float64)
    for j in range(order_n):
        for m in range(order_n):
            if m != j:
                L[:, j] *= (xi_fine - xi_gll[m]) / (xi_gll[j] - xi_gll[m])
    return L


def build_slice_interpolator(h5_files, plane="xz", cut_coord=0.0, bounds=(-1.75, 1.75),
                             grid_res=420, slab_half_thickness=0.12, n_fine=24, k_neighbors=4):
    """
    Extracts exact 2D GLL element slices lying on the cut plane (if available) and upsamples
    them using the exact degree-(N-1) GLL Lagrange polynomial basis to sub-pixel resolution,
    then precomputes sub-pixel KD-tree interpolation weights onto a regular 2D grid.
    """
    if plane.lower() == "xz":
        ax_u, ax_v, ax_n = 0, 2, 1  # x, z, cut along y = cut_coord
    elif plane.lower() == "xy":
        ax_u, ax_v, ax_n = 0, 1, 2  # x, y, cut along z = cut_coord
    elif plane.lower() == "yz":
        ax_u, ax_v, ax_n = 1, 2, 0  # y, z, cut along x = cut_coord
    else:
        raise ValueError(f"Unsupported plane '{plane}'. Choose from 'xz', 'xy', 'yz'.")

    first_coords = read_mesh_coordinates(h5_files[0])
    order_n = read_order_N(h5_files[0], first_coords.shape[0])

    rank_face_indices = []
    face_coords_list = []
    total_nodes = 0

    # Try extracting exact N x N spectral element slices on the cut plane
    if order_n >= 2:
        for fpath in h5_files:
            coords = read_mesh_coordinates(fpath)
            total_nodes += coords.shape[0]
            node_ids = extract_element_gll_node_ids(fpath, order_n, coords.shape[0])
            if node_ids is None:
                rank_face_indices.append(np.zeros((0, order_n, order_n), dtype=np.int64))
                continue
            c_elem = coords[node_ids]

            rank_faces = []
            # Check all 3 tensor-product directions for 2D slices lying on coord_n == cut_coord
            for d_axis in (0, 1, 2):
                for m_idx in range(order_n):
                    if d_axis == 0:
                        sl_c = c_elem[:, m_idx, :, :, :]
                        sl_i = node_ids[:, m_idx, :, :]
                    elif d_axis == 1:
                        sl_c = c_elem[:, :, m_idx, :, :]
                        sl_i = node_ids[:, :, m_idx, :]
                    else:
                        sl_c = c_elem[:, :, :, m_idx, :]
                        sl_i = node_ids[:, :, :, m_idx]

                    max_dn = np.max(np.abs(sl_c[:, :, :, ax_n] - cut_coord), axis=(1, 2))
                    hit = np.nonzero(max_dn < 1e-3)[0]
                    if hit.size > 0:
                        rank_faces.append(sl_i[hit])
                        face_coords_list.append(sl_c[hit])

            if rank_faces:
                rank_face_indices.append(np.concatenate(rank_faces, axis=0))
            else:
                rank_face_indices.append(np.zeros((0, order_n, order_n), dtype=np.int64))

    total_faces = sum(f.shape[0] for f in rank_face_indices)

    u_lin = np.linspace(bounds[0], bounds[1], grid_res)
    v_lin = np.linspace(bounds[0], bounds[1], grid_res)
    U, V = np.meshgrid(u_lin, v_lin)
    query_2d = np.column_stack([U.ravel(), V.ravel()])

    if total_faces > 0:
        L = compute_gll_lagrange_matrix(order_n, n_fine=n_fine)
        all_face_coords = np.concatenate(face_coords_list, axis=0)
        U_gll = all_face_coords[:, :, :, ax_u]
        V_gll = all_face_coords[:, :, :, ax_v]

        U_fine = np.einsum("ij,fjk,lk->fil", L, U_gll, L).ravel()
        V_fine = np.einsum("ij,fjk,lk->fil", L, V_gll, L).ravel()
        pts_2d = np.column_stack([U_fine, V_fine])

        print(f"[SLICE] Extracted {total_faces:,} exact {order_n}x{order_n} GLL element faces on plane "
              f"(spectrally upsampled via p={order_n - 1} Lagrange polynomials to {pts_2d.shape[0]:,} sub-element points).")

        tree = cKDTree(pts_2d)
        dists, knn_idx = tree.query(query_2d, k=k_neighbors)
        sigma = 0.006
        w_raw = np.exp(-0.5 * (dists / sigma) ** 2) + 1e-12
        weights = w_raw / np.sum(w_raw, axis=1, keepdims=True)
        outside_mask = (dists[:, 0] > 0.045).reshape(U.shape)

        interp_info = {
            "mode": "spectral_faces",
            "L": L,
            "x_gll": all_face_coords[:, :, :, 0],
            "knn_idx": knn_idx,
            "weights": weights,
            "outside_mask": outside_mask,
            "U": U,
            "V": V,
        }
        return U, V, rank_face_indices, interp_info

    # Fallback: thin 3D slab around cut_coord if cut_coord does not coincide with a GLL element plane
    slab_coords_3d = []
    rank_filter_indices = []
    for fpath in h5_files:
        coords = read_mesh_coordinates(fpath)
        mask = (
            (np.abs(coords[:, ax_n] - cut_coord) <= slab_half_thickness) &
            (coords[:, ax_u] >= bounds[0] - 0.15) & (coords[:, ax_u] <= bounds[1] + 0.15) &
            (coords[:, ax_v] >= bounds[0] - 0.15) & (coords[:, ax_v] <= bounds[1] + 0.15)
        )
        idx = np.nonzero(mask)[0]
        rank_filter_indices.append(idx)
        if idx.size > 0:
            slab_coords_3d.append(coords[idx])

    pts_3d = np.vstack(slab_coords_3d)
    pts_2d = np.column_stack([pts_3d[:, ax_u], pts_3d[:, ax_v]])
    tree = cKDTree(pts_2d)
    dists, knn_idx = tree.query(query_2d, k=k_neighbors)
    w_raw = 1.0 / (dists * dists + 1e-6)
    weights = w_raw / np.sum(w_raw, axis=1, keepdims=True)
    outside_mask = (dists[:, 0] > 0.08).reshape(U.shape)

    interp_info = {
        "mode": "slab",
        "x_gll": pts_3d[:, 0],
        "knn_idx": knn_idx,
        "weights": weights,
        "outside_mask": outside_mask,
        "U": U,
        "V": V,
    }
    return U, V, rank_filter_indices, interp_info


def read_all_steps_field_slabs(h5_files, steps, rank_filter_indices, field_mode="magE"):
    """
    Opens each rank HDF5 file ONCE and reads the signed electric field vector E(x, y, z)
    on the slice faces/nodes for all time steps so vector components can be spectrally
    interpolated BEFORE taking |E| (avoiding non-differentiable cusp averaging at zero crossings).
    """
    n_steps = len(steps)
    per_step_rank_chunks = [[] for _ in range(n_steps)]

    for r_idx, (fpath, idx) in enumerate(zip(h5_files, rank_filter_indices)):
        if idx.size == 0:
            continue
        if HAVE_H5PY:
            with h5py.File(fpath, "r") as f:
                ts = f["time_series"]
                for s_i, (_, _, step_name) in enumerate(steps):
                    grp = ts[step_name]
                    if "E" in grp:
                        per_step_rank_chunks[s_i].append(grp["E"][:][idx])
                    else:
                        per_step_rank_chunks[s_i].append(grp["magnitude_E"][:][idx][..., None])
        else:
            with CtypesHdf5Reader(fpath) as f:
                for s_i, (_, _, step_name) in enumerate(steps):
                    base = f"/time_series/{step_name}"
                    if f.has_path(f"{base}/E"):
                        E_full = f.read_double_dataset(f"{base}/E")
                        per_step_rank_chunks[s_i].append(E_full[idx])
                    else:
                        mag_full = f.read_double_dataset(f"{base}/magnitude_E")
                        per_step_rank_chunks[s_i].append(mag_full[idx][..., None])
        print(f"  Read {n_steps} time steps from rank file {r_idx + 1}/{len(h5_files)} ({os.path.basename(fpath)})")

    return [np.concatenate(chunks, axis=0) for chunks in per_step_rank_chunks]


def evaluate_slice_grid(step_data, interp_info, field_mode="magE",
                        scattered_only=False, time_val=0.0,
                        inc_e0=1.0, inc_x0=-0.90, inc_sigma_x=0.14, inc_lambda0=0.30):
    """
    Optionally subtracts the analytical incident plane wave E_inc(r, t) at GLL nodes to
    isolate E_scat(r, t), spectrally upsamples signed vector components (Ex, Ey, Ez) inside
    each element face via degree-(N-1) GLL Lagrange polynomials, interpolates onto the 2D
    pixel grid, and then computes |E| = sqrt(Ex^2 + Ey^2 + Ez^2) (or requested component).
    """
    knn_idx = interp_info["knn_idx"]
    weights = interp_info["weights"]
    outside_mask = interp_info["outside_mask"]
    U = interp_info["U"]

    if scattered_only and step_data.shape[-1] == 3:
        x_gll = interp_info["x_gll"]
        inc_k0 = 2.0 * np.pi / inc_lambda0
        s = x_gll - inc_x0 - time_val
        ez_inc = inc_e0 * np.cos(inc_k0 * s) * np.exp(-(s * s) / (2.0 * inc_sigma_x * inc_sigma_x))
        step_data = step_data.copy()
        step_data[..., 2] -= ez_inc

    if interp_info["mode"] == "spectral_faces":
        L = interp_info["L"]
        # step_data has shape (n_faces, N, N, C) where C=3 for vector E or C=1 for scalar
        fine_vals = np.einsum("ij,fjkc,lk->filc", L, step_data, L).reshape(-1, step_data.shape[-1])
    else:
        fine_vals = step_data

    # Interpolate each smooth signed component onto the 2D pixel grid
    grid_vec = np.sum(fine_vals[knn_idx] * weights[:, :, None], axis=1).reshape(U.shape + (fine_vals.shape[-1],))

    if fine_vals.shape[-1] == 1 or field_mode == "magE":
        if fine_vals.shape[-1] == 1:
            grid_out = grid_vec[..., 0]
        else:
            grid_out = np.sqrt(np.sum(grid_vec * grid_vec, axis=-1))
    elif field_mode == "Ez":
        grid_out = grid_vec[..., 2]
    elif field_mode == "Ey":
        grid_out = grid_vec[..., 1]
    elif field_mode == "Ex":
        grid_out = grid_vec[..., 0]
    else:
        grid_out = np.sqrt(np.sum(grid_vec * grid_vec, axis=-1))

    grid_out[outside_mask] = 0.0
    return grid_out


def main():
    parser = argparse.ArgumentParser(
        description="Visualize NekWave 3D HDF5 fields on a 2D cut plane and export MP4 video."
    )
    parser.add_argument(
        "-i", "--input-dir", default="build/output_sphereRCS",
        help="Directory containing fields_rank*.h5 or fields.h5 (default: build/output_sphereRCS)"
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output MP4 filename (default: <input-dir>/sphere_wave_[scat_]magE_xz.mp4)"
    )
    parser.add_argument(
        "--plane", default="xz", choices=["xz", "xy", "yz"],
        help="2D slice plane: 'xz' (y=0 E-plane), 'xy' (z=0 H-plane), or 'yz' (x=0) (default: xz)"
    )
    parser.add_argument(
        "--field", default="magE", choices=["magE", "Ez", "Ey", "Ex"],
        help="Field quantity to plot: 'magE' (|E| magnitude) or component 'Ez', 'Ey', 'Ex' (default: magE)"
    )
    parser.add_argument(
        "--scattered", action="store_true",
        help="Subtract analytical incident plane wave E_inc(r, t) to visualize only the scattered field E_scat(r, t)"
    )
    parser.add_argument(
        "--radius", type=float, default=0.30,
        help="PEC sphere radius a in meters (default: 0.30)"
    )
    parser.add_argument(
        "--domain-limit", type=float, default=1.75,
        help="Half-width of physical plot window in meters (default: 1.75)"
    )
    parser.add_argument(
        "--res", type=int, default=420,
        help="2D image grid resolution per axis (default: 420)"
    )
    parser.add_argument(
        "--fps", type=int, default=10,
        help="Video frames per second (default: 10)"
    )
    parser.add_argument(
        "--vmax", type=float, default=None,
        help="Colorbar maximum for |E| (default: 1.35 for total field, 0.65 for scattered field)"
    )
    args = parser.parse_args()
    if args.vmax is None:
        args.vmax = 0.65 if args.scattered else 1.35

    h5_files = find_h5_files(args.input_dir)
    if not h5_files:
        print(f"Error: No fields_rank*.h5 or fields.h5 found in '{args.input_dir}'.")
        sys.exit(1)

    steps = collect_common_steps(h5_files)
    if not steps:
        print(f"Error: No recorded time steps found in '{args.input_dir}'.")
        sys.exit(1)

    mode_tag = "Scattered Field Only E^scat" if args.scattered else "Total Field E^tot"
    print(f"[INFO] Found {len(h5_files)} HDF5 file(s) with {len(steps)} saved time steps "
          f"(step {steps[0][0]} at t = {steps[0][1]:.4f} s -> step {steps[-1][0]} at t = {steps[-1][1]:.4f} s) [{mode_tag}].")

    bounds = (-args.domain_limit, args.domain_limit)
    U, V, rank_idx, interp_info = build_slice_interpolator(
        h5_files, plane=args.plane, cut_coord=0.0, bounds=bounds, grid_res=args.res
    )

    R2 = U * U + V * V
    sphere_mask = R2 < (args.radius * args.radius)

    all_slab_vals = read_all_steps_field_slabs(h5_files, steps, rank_idx, field_mode=args.field)

    frames = []
    for idx_s, ((step_num, time_val, _), slab_vals) in enumerate(zip(steps, all_slab_vals)):
        grid_vals = evaluate_slice_grid(
            slab_vals, interp_info, field_mode=args.field,
            scattered_only=args.scattered, time_val=time_val
        )
        grid_vals[sphere_mask] = 0.0
        frames.append((step_num, time_val, grid_vals))

    axis_labels = {
        "xz": ("$x$ [m] (Wave Propagation $\\rightarrow$)", "$z$ [m] (Electric Polarization $\\mathbf{E}$)", "$y = 0$ E-Plane"),
        "xy": ("$x$ [m] (Wave Propagation $\\rightarrow$)", "$y$ [m] (Magnetic Polarization $\\mathbf{H}$)", "$z = 0$ H-Plane"),
        "yz": ("$y$ [m]", "$z$ [m]", "$x = 0$ Cross-Plane"),
    }[args.plane.lower()]

    sup_tag = "scat" if args.scattered else "tot"
    title_kind = "Scattered Electric Field" if args.scattered else "Total Electric Field"
    file_prefix = f"sphere_wave_scat_{args.field}_{args.plane}" if args.scattered else f"sphere_wave_{args.field}_{args.plane}"

    if args.field == "magE":
        cmap = "inferno"
        vmin, vmax = 0.0, args.vmax
        cbar_label = f"{title_kind} Magnitude $\\|\\mathbf{{E}}^{{\\mathrm{{{sup_tag}}}}}(\\mathbf{{r}}, t)\\|$"
    else:
        cmap = "RdBu_r"
        vmin, vmax = -args.vmax, args.vmax
        cbar_label = f"{title_kind} Component ${args.field}^{{\\mathrm{{{sup_tag}}}}}(\\mathbf{{r}}, t)$"

    # --------------------------------------------------------------------------
    # 1. Save 6-Panel Key Snapshot Montage PNG
    # --------------------------------------------------------------------------
    if len(frames) >= 6:
        target_times = [0.65, 0.97, 1.30, 1.62, 1.95, 2.38]
        snap_indices = []
        times_arr = np.array([f[1] for f in frames])
        for tt in target_times:
            best_i = int(np.argmin(np.abs(times_arr - tt)))
            if best_i not in snap_indices:
                snap_indices.append(best_i)
        while len(snap_indices) < 6:
            for cand in np.linspace(0, len(frames) - 1, 6, dtype=int):
                if int(cand) not in snap_indices:
                    snap_indices.append(int(cand))
                    break
        snap_indices = sorted(snap_indices[:6])
    else:
        snap_indices = list(range(len(frames)))

    fig_m, axes_m = plt.subplots(2, 3, figsize=(15, 9.5), dpi=180)
    axes_flat = axes_m.ravel()
    for ax_i, f_idx in enumerate(snap_indices):
        ax = axes_flat[ax_i]
        s_num, t_val, img = frames[f_idx]
        im = ax.imshow(
            img, origin="lower", extent=[bounds[0], bounds[1], bounds[0], bounds[1]],
            cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear"
        )
        sphere_patch = patches.Circle(
            (0.0, 0.0), args.radius, facecolor="#d9d9d9", edgecolor="white", lw=1.5, zorder=5
        )
        ax.add_patch(sphere_patch)
        ax.text(0.0, 0.0, "PEC", color="black", fontsize=8.5, fontweight="bold",
                ha="center", va="center", zorder=6)
        ax.set_title(f"Step {s_num} ($t = {t_val:.3f}$ s)", fontsize=11, fontweight="bold")
        ax.set_xlabel(axis_labels[0], fontsize=9.5)
        ax.set_ylabel(axis_labels[1], fontsize=9.5)

    for extra_ax in axes_flat[len(snap_indices):]:
        extra_ax.axis("off")

    fig_m.suptitle(
        f"NekWave 3D PEC Sphere Scattering ($a = {args.radius:.2f}$ m) — 2D {axis_labels[2]} Cut ({cbar_label})",
        fontsize=13.5, fontweight="bold", y=0.97
    )
    fig_m.subplots_adjust(left=0.06, right=0.90, bottom=0.07, top=0.90, wspace=0.24, hspace=0.28)
    cbar_ax = fig_m.add_axes([0.92, 0.12, 0.018, 0.75])
    fig_m.colorbar(im, cax=cbar_ax, label=cbar_label)

    montage_path = os.path.join(args.input_dir, f"{file_prefix}_snapshots.png")
    fig_m.savefig(montage_path, dpi=180)
    plt.close(fig_m)
    print(f"[OUTPUT] Saved 2D slice snapshot montage to: {montage_path}")

    # --------------------------------------------------------------------------
    # 2. Render Full Time-Series Animation (MP4 + GIF)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=160)
    s0, t0, img0 = frames[0]
    im_anim = ax.imshow(
        img0, origin="lower", extent=[bounds[0], bounds[1], bounds[0], bounds[1]],
        cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear"
    )
    sphere_circle = patches.Circle(
        (0.0, 0.0), args.radius, facecolor="#d9d9d9", edgecolor="white", lw=1.8, zorder=5
    )
    ax.add_patch(sphere_circle)
    ax.text(0.0, 0.0, "PEC\nSphere", color="black", fontsize=9.5, fontweight="bold",
            ha="center", va="center", zorder=6)

    ax.set_xlabel(axis_labels[0], fontsize=11)
    ax.set_ylabel(axis_labels[1], fontsize=11)
    title_obj = ax.set_title(
        f"NekWave 3D PEC Sphere — {title_kind} ({axis_labels[2]})\nStep {s0:5d} | Time $t = {t0:.4f}$ s",
        fontsize=12, fontweight="bold", pad=10
    )
    cbar = fig.colorbar(im_anim, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=10.5)
    fig.tight_layout()

    def update(frame_idx):
        s_num, t_val, img = frames[frame_idx]
        im_anim.set_data(img)
        title_obj.set_text(
            f"NekWave 3D PEC Sphere — {title_kind} ({axis_labels[2]})\nStep {s_num:5d} | Time $t = {t_val:.4f}$ s"
        )
        return [im_anim, title_obj]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 // args.fps, blit=False)

    mp4_path = args.output or os.path.join(args.input_dir, f"{file_prefix}.mp4")
    ffmpeg_bin = ensure_ffmpeg_configured()

    saved_mp4 = False
    if ffmpeg_bin is not None:
        try:
            writer = FFMpegWriter(fps=args.fps, codec="libx264", bitrate=3500, extra_args=["-pix_fmt", "yuv420p"])
            anim.save(mp4_path, writer=writer)
            saved_mp4 = True
            print(f"[OUTPUT] Saved MP4 video via ffmpeg ({ffmpeg_bin}) to: {mp4_path}")
        except Exception as e:
            print(f"[WARNING] libx264 encode failed ({e}), trying mpeg4 codec...")
            try:
                writer = FFMpegWriter(fps=args.fps, codec="mpeg4", bitrate=3500)
                anim.save(mp4_path, writer=writer)
                saved_mp4 = True
                print(f"[OUTPUT] Saved MP4 video (mpeg4) via ffmpeg to: {mp4_path}")
            except Exception as e2:
                print(f"[WARNING] ffmpeg fallback failed: {e2}")

    gif_path = os.path.splitext(mp4_path)[0] + ".gif"
    anim.save(gif_path, writer=PillowWriter(fps=args.fps))
    print(f"[OUTPUT] Saved animated GIF to: {gif_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()

