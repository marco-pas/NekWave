#!/usr/bin/env python3
"""
gmsh2nek.py - Vectorized Gmsh (.msh) to NekCEM / NekWave (.rea & .re2) Hexahedral Converter
============================================================================================

High-performance NumPy-vectorized converter for pure-hexahedral Gmsh meshes
(MSH 4.1 ASCII/Binary or MSH 2.2 ASCII) into NekCEM and NekWave mesh files:
  - Binary `.re2` (high-performance `#v002` binary mesh + boundary conditions)
  - ASCII  `.rea` (standalone ASCII mesh or lightweight `-NEL` companion header for `.re2`)

Performance Optimizations for Large Meshes (100k - 2M+ elements):
  1. Block-level C parsing (`np.fromstring` / `np.frombuffer`) for `$Nodes` and `$Elements`,
     avoiding millions of Python per-line string splits and tuple allocations.
  2. Vectorized spatial quantization & `np.unique` node deduplication.
  3. Vectorized BLAS (`@`) trilinear Jacobian evaluation across all hexahedra simultaneously
     at the element center and all 8 reference corners.
  4. Vectorized 6-face neighbor topology discovery via 16-byte structured face keys and
     `np.argsort` (zero Python dictionaries or element loops for interior faces).
  5. Vectorized `np.searchsorted` boundary quad tag matching and $O(N_\\text{PML})$ inward layer marching.
  6. Single-syscall `.tobytes()` binary `.re2` export and chunk-buffered `.rea` export.

Usage:
  # Fast binary .re2 + companion .rea (recommended for large meshes):
  python3 meshing/gmsh2nek.py meshing/sphereRCS_profiling/spherePEC.msh --format re2

  # Both full ASCII .rea and binary .re2:
  python3 meshing/gmsh2nek.py meshing/sphereRCS_dense/spherePEC.msh --format both
"""

import argparse
import math
import os
import struct
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ==============================================================================
# NekCEM / NekWave Hexahedron Face & Corner Conventions
# ==============================================================================
# Local 0-based corner indices (0..7) for each of the 6 NekCEM faces (0..5):
#   Face 0 (Nek face 1, s = -1): (0, 1, 5, 4)
#   Face 1 (Nek face 2, r = +1): (1, 2, 6, 5)
#   Face 2 (Nek face 3, s = +1): (2, 3, 7, 6)
#   Face 3 (Nek face 4, r = -1): (3, 0, 4, 7)
#   Face 4 (Nek face 5, t = -1): (0, 3, 2, 1)
#   Face 5 (Nek face 6, t = +1): (4, 5, 6, 7)
NEK_FACE_CORNERS: Tuple[Tuple[int, int, int, int], ...] = (
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
    (0, 3, 2, 1),
    (4, 5, 6, 7),
)
NEK_FACE_CORNERS_ARR = np.array(NEK_FACE_CORNERS, dtype=np.int32)

# Opposing local face index (0..5) for inward layer marching
NEK_OPP_FACE: Tuple[int, ...] = (2, 3, 0, 1, 5, 4)
NEK_OPP_FACE_ARR = np.array(NEK_OPP_FACE, dtype=np.int8)

# Reference coordinates (r, s, t) of the 8 hexahedron corners
REF_CORNERS: Tuple[Tuple[float, float, float], ...] = (
    (-1.0, -1.0, -1.0),
    (+1.0, -1.0, -1.0),
    (+1.0, +1.0, -1.0),
    (-1.0, +1.0, -1.0),
    (-1.0, -1.0, +1.0),
    (+1.0, -1.0, +1.0),
    (+1.0, +1.0, +1.0),
    (-1.0, +1.0, +1.0),
)

# Supported Gmsh element types:
#   2D Quads: 3 (4-node), 10 (9-node), 16 (8-node)
#   3D Hexes: 5 (8-node), 12 (27-node), 17 (20-node), 92 (64-node), 93 (125-node)
GMSH_QUAD_TYPES = {3, 10, 16}
GMSH_HEX_TYPES = {5, 12, 17, 92, 93}

# Number of nodes per Gmsh element type (for binary .msh parser)
GMSH_ELEM_NUM_NODES: Dict[int, int] = {
    1: 2,    # 2-node line
    2: 3,    # 3-node triangle
    3: 4,    # 4-node quad
    4: 4,    # 4-node tet
    5: 8,    # 8-node hex
    6: 6,    # 6-node prism
    7: 5,    # 5-node pyramid
    8: 3,    # 3-node 2nd order line
    9: 6,    # 6-node 2nd order triangle
    10: 9,   # 9-node 2nd order quad
    11: 10,  # 10-node 2nd order tet
    12: 27,  # 27-node 2nd order hex
    15: 1,   # 1-node point
    16: 8,   # 8-node serendipity quad
    17: 20,  # 20-node serendipity hex
    92: 64,  # 64-node 3rd order hex
    93: 125, # 125-node 4th order hex
}

# 16-byte structured dtype for 4 sorted int32 node IDs representing a canonical quad face
FACE_KEY_DTYPE = np.dtype([("n0", "<i4"), ("n1", "<i4"), ("n2", "<i4"), ("n3", "<i4")])


def _sort4_cols(f: np.ndarray) -> np.ndarray:
    """Sorts each row of an (N, 4) int32 array using a 5-comparator vectorized sorting network."""
    c0 = f[:, 0]
    c1 = f[:, 1]
    c2 = f[:, 2]
    c3 = f[:, 3]

    lo01, hi01 = np.minimum(c0, c1), np.maximum(c0, c1)
    lo23, hi23 = np.minimum(c2, c3), np.maximum(c2, c3)
    s0 = np.minimum(lo01, lo23)
    mid_a = np.maximum(lo01, lo23)
    mid_b = np.minimum(hi01, hi23)
    s3 = np.maximum(hi01, hi23)
    s1 = np.minimum(mid_a, mid_b)
    s2 = np.maximum(mid_a, mid_b)

    out = np.empty_like(f)
    out[:, 0] = s0
    out[:, 1] = s1
    out[:, 2] = s2
    out[:, 3] = s3
    return out


def _face_keys_uint64(sorted_faces: np.ndarray, max_nid: int) -> np.ndarray:
    """
    Packs 4 sorted int32 node IDs per quad face into a uint64 key for native SIMD radix sorting.
    For max_nid < 2^21 (2.09M nodes), packs (c0 << 42) | (c1 << 21) | c2 losslessly, since
    no two distinct quad faces in a conformal hexahedral mesh can share 3 corner vertices.
    For larger node IDs, uses a non-linear SplitMix64 cascade across all 4 node IDs.
    """
    u0 = sorted_faces[:, 0].astype(np.uint64)
    u1 = sorted_faces[:, 1].astype(np.uint64)
    u2 = sorted_faces[:, 2].astype(np.uint64)
    if max_nid < (1 << 21):
        return (u0 << np.uint64(42)) | (u1 << np.uint64(21)) | u2
    u3 = sorted_faces[:, 3].astype(np.uint64)
    h = (u0 + np.uint64(0x9E3779B97F4A7C15)) * np.uint64(0xBF58476D1CE4E5B9)
    h = (h ^ (h >> np.uint64(27))) ^ (u1 * np.uint64(0x94D049BB133111EB))
    h = (h ^ (h >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    h = (h ^ (h >> np.uint64(27))) ^ ((u2 << np.uint64(32)) | u3)
    h = (h ^ (h >> np.uint64(30))) * np.uint64(0x94D049BB133111EB)
    return h ^ (h >> np.uint64(31))




def _build_trilinear_deriv_matrix(r: float, s: float, t: float) -> np.ndarray:
    """Returns the (3, 8) matrix of trilinear shape function derivatives [dN/dr; dN/ds; dN/dt] at (r, s, t)."""
    rm, rp = 1.0 - r, 1.0 + r
    sm, sp = 1.0 - s, 1.0 + s
    tm, tp = 1.0 - t, 1.0 + t
    return np.array(
        [
            [
                -0.125 * sm * tm, +0.125 * sm * tm, +0.125 * sp * tm, -0.125 * sp * tm,
                -0.125 * sm * tp, +0.125 * sm * tp, +0.125 * sp * tp, -0.125 * sp * tp,
            ],
            [
                -0.125 * rm * tm, -0.125 * rp * tm, +0.125 * rp * tm, +0.125 * rm * tm,
                -0.125 * rm * tp, -0.125 * rp * tp, +0.125 * rp * tp, +0.125 * rm * tp,
            ],
            [
                -0.125 * rm * sm, -0.125 * rp * sm, -0.125 * rp * sp, -0.125 * rm * sp,
                +0.125 * rm * sm, +0.125 * rp * sm, +0.125 * rp * sp, +0.125 * rm * sp,
            ],
        ],
        dtype=np.float64,
    )


# Precomputed derivative matrices at element center (0,0,0) and all 8 corners
D_CENTER: np.ndarray = _build_trilinear_deriv_matrix(0.0, 0.0, 0.0)
D_CORNERS: np.ndarray = np.stack([_build_trilinear_deriv_matrix(r, s, t) for (r, s, t) in REF_CORNERS], axis=0)


# ==============================================================================
# Jacobian Evaluation & Right-Handed Hexahedron Orientation
# ==============================================================================
def eval_trilinear_det_j(
    corners: List[Tuple[float, float, float]], r: float, s: float, t: float
) -> float:
    """
    Evaluates the trilinear Jacobian determinant det(J) at reference point (r, s, t),
    matching Mesh::computeMetricsFromCorners() in src/mesh.cpp.
    """
    D = _build_trilinear_deriv_matrix(r, s, t)
    J = D @ np.asarray(corners, dtype=np.float64)
    return float(np.linalg.det(J))


def ensure_positive_jacobian(
    node_ids: List[int], nodes: Dict[int, Tuple[float, float, float]]
) -> Tuple[List[int], bool, float, float]:
    """
    Single-element helper retained for API compatibility.
    """
    coords = np.array([nodes[nid] for nid in node_ids], dtype=np.float64)
    J0 = D_CENTER @ coords
    det_center = (
        J0[0, 0] * (J0[1, 1] * J0[2, 2] - J0[2, 1] * J0[1, 2])
        - J0[1, 0] * (J0[0, 1] * J0[2, 2] - J0[2, 1] * J0[0, 2])
        + J0[2, 0] * (J0[0, 1] * J0[1, 2] - J0[1, 1] * J0[0, 2])
    )
    flipped = False
    if det_center < 0.0:
        node_ids = [
            node_ids[4], node_ids[5], node_ids[6], node_ids[7],
            node_ids[0], node_ids[1], node_ids[2], node_ids[3],
        ]
        coords = coords[[4, 5, 6, 7, 0, 1, 2, 3]]
        flipped = True

    Jc = D_CORNERS @ coords  # (8, 3, 3)
    dets = (
        Jc[:, 0, 0] * (Jc[:, 1, 1] * Jc[:, 2, 2] - Jc[:, 2, 1] * Jc[:, 1, 2])
        - Jc[:, 1, 0] * (Jc[:, 0, 1] * Jc[:, 2, 2] - Jc[:, 2, 1] * Jc[:, 0, 2])
        + Jc[:, 2, 0] * (Jc[:, 0, 1] * Jc[:, 1, 2] - Jc[:, 1, 1] * Jc[:, 0, 2])
    )
    return node_ids, flipped, float(dets.min()), float(dets.max())


def _det_3vec(r_vec: np.ndarray, s_vec: np.ndarray, t_vec: np.ndarray) -> np.ndarray:
    """Computes scalar triple product det([r_vec, s_vec, t_vec]) for arrays of shape (nel, 3)."""
    return (
        r_vec[:, 0] * (s_vec[:, 1] * t_vec[:, 2] - t_vec[:, 1] * s_vec[:, 2])
        - s_vec[:, 0] * (r_vec[:, 1] * t_vec[:, 2] - t_vec[:, 1] * r_vec[:, 2])
        + t_vec[:, 0] * (r_vec[:, 1] * s_vec[:, 2] - s_vec[:, 1] * r_vec[:, 2])
    )


def ensure_positive_jacobian_batch(
    hex_nodes: np.ndarray, node_coords: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, int, float, float]:
    """
    Vectorized orientation check and corner Jacobian evaluation for all `nel` hexahedra at once
    using the exact 12-edge trilinear factorization (zero BLAS thread-spawn overhead).
    """
    coords = node_coords[hex_nodes]  # (nel, 8, 3)

    # 12 edge derivative vectors (each = 0.5 * edge difference)
    er0 = 0.5 * (coords[:, 1] - coords[:, 0])
    er1 = 0.5 * (coords[:, 2] - coords[:, 3])
    er2 = 0.5 * (coords[:, 5] - coords[:, 4])
    er3 = 0.5 * (coords[:, 6] - coords[:, 7])

    es0 = 0.5 * (coords[:, 3] - coords[:, 0])
    es1 = 0.5 * (coords[:, 2] - coords[:, 1])
    es2 = 0.5 * (coords[:, 7] - coords[:, 4])
    es3 = 0.5 * (coords[:, 6] - coords[:, 5])

    et0 = 0.5 * (coords[:, 4] - coords[:, 0])
    et1 = 0.5 * (coords[:, 5] - coords[:, 1])
    et2 = 0.5 * (coords[:, 6] - coords[:, 2])
    et3 = 0.5 * (coords[:, 7] - coords[:, 3])

    # Center Jacobian (0, 0, 0) is the average of the 4 edge vectors in each direction
    r_c = 0.25 * (er0 + er1 + er2 + er3)
    s_c = 0.25 * (es0 + es1 + es2 + es3)
    t_c = 0.25 * (et0 + et1 + et2 + et3)
    det_center = _det_3vec(r_c, s_c, t_c)

    flip_mask = det_center < 0.0
    num_flipped = int(np.count_nonzero(flip_mask))
    if num_flipped > 0:
        perm = [4, 5, 6, 7, 0, 1, 2, 3]
        hex_nodes = hex_nodes.copy()
        hex_nodes[flip_mask] = hex_nodes[flip_mask][:, perm]
        coords[flip_mask] = coords[flip_mask][:, perm, :]
        er0 = 0.5 * (coords[:, 1] - coords[:, 0])
        er1 = 0.5 * (coords[:, 2] - coords[:, 3])
        er2 = 0.5 * (coords[:, 5] - coords[:, 4])
        er3 = 0.5 * (coords[:, 6] - coords[:, 7])
        es0 = 0.5 * (coords[:, 3] - coords[:, 0])
        es1 = 0.5 * (coords[:, 2] - coords[:, 1])
        es2 = 0.5 * (coords[:, 7] - coords[:, 4])
        es3 = 0.5 * (coords[:, 6] - coords[:, 5])
        et0 = 0.5 * (coords[:, 4] - coords[:, 0])
        et1 = 0.5 * (coords[:, 5] - coords[:, 1])
        et2 = 0.5 * (coords[:, 6] - coords[:, 2])
        et3 = 0.5 * (coords[:, 7] - coords[:, 3])

    # Evaluate det(J) at all 8 corners using their exact 3 incident edge vectors
    corner_triplets = (
        (er0, es0, et0),  # corner 0 (-1, -1, -1)
        (er0, es1, et1),  # corner 1 (+1, -1, -1)
        (er1, es1, et2),  # corner 2 (+1, +1, -1)
        (er1, es0, et3),  # corner 3 (-1, +1, -1)
        (er2, es2, et0),  # corner 4 (-1, -1, +1)
        (er2, es3, et1),  # corner 5 (+1, -1, +1)
        (er3, es3, et2),  # corner 6 (+1, +1, +1)
        (er3, es2, et3),  # corner 7 (-1, +1, +1)
    )
    min_j = float("inf")
    max_j = float("-inf")
    for rv, sv, tv in corner_triplets:
        det_c = _det_3vec(rv, sv, tv)
        c_min = float(det_c.min())
        c_max = float(det_c.max())
        if c_min < min_j:
            min_j = c_min
        if c_max > max_j:
            max_j = c_max

    return hex_nodes, coords, num_flipped, min_j, max_j


# ==============================================================================
# Physical Group Name -> Boundary Condition Tag Mapping
# ==============================================================================
def infer_bc_tag(phys_name: str, custom_map: Dict[str, str]) -> str:
    """
    Maps a Gmsh Physical Group name to a canonical NekCEM / NekWave BC tag:
    'PEC', 'PMC', 'PML', or 'P'.
    """
    if phys_name in custom_map:
        return custom_map[phys_name]
    upper = phys_name.upper()
    for k, v in custom_map.items():
        if k.upper() == upper:
            return v

    if "PML" in upper:
        return "PML"
    if "PMC" in upper or "SYM" in upper or "MAGNET" in upper or "NEUMANN" in upper:
        return "PMC"
    if "PER" in upper:
        return "P"
    if "PEC" in upper or "WALL" in upper or "CONDUCT" in upper or "METAL" in upper:
        return "PEC"
    return "PEC"


# ==============================================================================
# Gmsh .msh Parser (Vectorized MSH 4.1 ASCII/Binary and MSH 2.2 ASCII)
# ==============================================================================
class GmshMeshData:
    def __init__(self) -> None:
        self.version: str = "4.1"
        self.physical_names: Dict[Tuple[int, int], str] = {}
        self.entity_phys_tags: Dict[Tuple[int, int], List[int]] = {}
        # Fast contiguous storage for vectorized pipeline:
        self.node_coords: Optional[np.ndarray] = None      # (max_nid + 1, 3) float64
        self.valid_node_mask: Optional[np.ndarray] = None  # (max_nid + 1,) bool
        self.quad_blocks: List[Tuple[np.ndarray, List[int]]] = []  # list of ((N_q, 4) int32, phys_tags)
        self.hex_blocks: List[Tuple[np.ndarray, List[int]]] = []   # list of ((N_h, 8) int32, phys_tags)
        # Lazy dict/list properties for backward compatibility:
        self._nodes_dict: Optional[Dict[int, Tuple[float, float, float]]] = None
        self._quads_list: Optional[List[Tuple[List[int], List[int]]]] = None
        self._hexes_list: Optional[List[Tuple[List[int], List[int]]]] = None

    @property
    def nodes(self) -> Dict[int, Tuple[float, float, float]]:
        if self._nodes_dict is None:
            self._nodes_dict = {}
            if self.node_coords is not None and self.valid_node_mask is not None:
                nids = np.flatnonzero(self.valid_node_mask)
                coords = self.node_coords[nids]
                self._nodes_dict = {
                    int(nid): (float(c[0]), float(c[1]), float(c[2]))
                    for nid, c in zip(nids, coords)
                }
        return self._nodes_dict

    @property
    def quads(self) -> List[Tuple[List[int], List[int]]]:
        if self._quads_list is None:
            self._quads_list = []
            for q_arr, ptags in self.quad_blocks:
                pt_list = list(ptags)
                for row in q_arr.tolist():
                    self._quads_list.append((row, pt_list))
        return self._quads_list

    @property
    def hexes(self) -> List[Tuple[List[int], List[int]]]:
        if self._hexes_list is None:
            self._hexes_list = []
            for h_arr, ptags in self.hex_blocks:
                pt_list = list(ptags)
                for row in h_arr.tolist():
                    self._hexes_list.append((row, pt_list))
        return self._hexes_list


def _parse_msh4_binary(filepath: str, data: GmshMeshData) -> GmshMeshData:
    """Fast binary parser for Gmsh MSH 4.1 binary files."""
    with open(filepath, "rb") as f:
        buf = f.read()

    pos = 0
    n_bytes = len(buf)

    def read_line() -> str:
        nonlocal pos
        nl = buf.find(b"\n", pos)
        if nl < 0:
            s = buf[pos:].decode("utf-8", errors="replace").strip()
            pos = n_bytes
            return s
        s = buf[pos:nl].decode("utf-8", errors="replace").strip()
        pos = nl + 1
        return s

    while pos < n_bytes:
        line = read_line()
        if not line:
            continue

        if line == "$MeshFormat":
            fmt_line = read_line().split()
            data.version = fmt_line[0]
            data_size = int(fmt_line[2])
            # 4-byte endianness int (1) + newline
            pos += 4
            if pos < n_bytes and buf[pos : pos + 1] == b"\n":
                pos += 1
            while pos < n_bytes and read_line() != "$EndMeshFormat":
                pass

        elif line == "$PhysicalNames":
            num_phys = int(read_line())
            for _ in range(num_phys):
                parts = read_line().split(maxsplit=2)
                dim = int(parts[0])
                tag = int(parts[1])
                name = parts[2].strip().strip('"')
                data.physical_names[(dim, tag)] = name
            while pos < n_bytes and read_line() != "$EndPhysicalNames":
                pass

        elif line == "$Entities":
            counts = np.frombuffer(buf, dtype="<u8", count=4, offset=pos)
            pos += 32
            num_pts, num_curves, num_surfs, num_vols = (int(x) for x in counts)
            for _ in range(num_pts):
                pt_tag = struct.unpack_from("<i", buf, pos)[0]
                pos += 4 + 24  # tag (int32) + x,y,z (3 float64)
                num_phys = struct.unpack_from("<Q", buf, pos)[0]
                pos += 8
                phys_tags = list(np.frombuffer(buf, dtype="<i4", count=num_phys, offset=pos))
                pos += 4 * num_phys
                data.entity_phys_tags[(0, int(pt_tag))] = [int(t) for t in phys_tags]

            for dim, count in ((1, num_curves), (2, num_surfs), (3, num_vols)):
                for _ in range(count):
                    ent_tag = struct.unpack_from("<i", buf, pos)[0]
                    pos += 4 + 48  # tag (int32) + bbox (6 float64)
                    num_phys = struct.unpack_from("<Q", buf, pos)[0]
                    pos += 8
                    phys_tags = list(np.frombuffer(buf, dtype="<i4", count=num_phys, offset=pos))
                    pos += 4 * num_phys
                    num_bound = struct.unpack_from("<Q", buf, pos)[0]
                    pos += 8 + 4 * num_bound
                    data.entity_phys_tags[(dim, int(ent_tag))] = [int(t) for t in phys_tags]
            while pos < n_bytes and read_line() != "$EndEntities":
                pass

        elif line == "$Nodes":
            hdr = np.frombuffer(buf, dtype="<u8", count=4, offset=pos)
            pos += 32
            num_blocks, total_nodes, min_tag, max_tag = (int(x) for x in hdr)
            data.node_coords = np.zeros((max_tag + 1, 3), dtype=np.float64)
            data.valid_node_mask = np.zeros(max_tag + 1, dtype=bool)
            for _ in range(num_blocks):
                ent_dim, ent_tag, parametric = struct.unpack_from("<iii", buf, pos)
                pos += 12
                num_in_block = int(struct.unpack_from("<Q", buf, pos)[0])
                pos += 8
                if num_in_block > 0:
                    nids = np.frombuffer(buf, dtype="<u8", count=num_in_block, offset=pos)
                    pos += 8 * num_in_block
                    n_floats = 3 + (ent_dim if parametric else 0)
                    coords = np.frombuffer(
                        buf, dtype="<f8", count=num_in_block * n_floats, offset=pos
                    ).reshape(num_in_block, n_floats)[:, :3]
                    pos += 8 * num_in_block * n_floats
                    data.node_coords[nids] = coords
                    data.valid_node_mask[nids] = True
            while pos < n_bytes and read_line() != "$EndNodes":
                pass

        elif line == "$Elements":
            hdr = np.frombuffer(buf, dtype="<u8", count=4, offset=pos)
            pos += 32
            num_blocks = int(hdr[0])
            for _ in range(num_blocks):
                ent_dim, ent_tag, etype = struct.unpack_from("<iii", buf, pos)
                pos += 12
                num_in_block = int(struct.unpack_from("<Q", buf, pos)[0])
                pos += 8
                n_nodes = GMSH_ELEM_NUM_NODES.get(etype, 0)
                if n_nodes == 0:
                    raise ValueError(f"Unsupported binary Gmsh element type {etype}")
                row_len = 1 + n_nodes
                if num_in_block > 0:
                    raw = np.frombuffer(
                        buf, dtype="<u8", count=num_in_block * row_len, offset=pos
                    ).reshape(num_in_block, row_len)
                    pos += 8 * num_in_block * row_len
                    phys_tags = data.entity_phys_tags.get((ent_dim, ent_tag), [])
                    if etype in GMSH_QUAD_TYPES:
                        data.quad_blocks.append((raw[:, 1:5].astype(np.int32), phys_tags))
                    elif etype in GMSH_HEX_TYPES:
                        data.hex_blocks.append((raw[:, 1:9].astype(np.int32), phys_tags))
            while pos < n_bytes and read_line() != "$EndElements":
                pass

    return data


def parse_msh_file(filepath: str) -> GmshMeshData:
    """
    Parses a Gmsh .msh file (MSH 4.x ASCII/Binary or MSH 2.2 ASCII) using
    vectorized C block parsing (`np.fromstring` / `np.frombuffer`).
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Gmsh file not found: {filepath}")

    data = GmshMeshData()

    # Peek at $MeshFormat to check if binary MSH 4.x
    with open(filepath, "rb") as fb:
        head_chunk = fb.read(512)
    if b"$MeshFormat" in head_chunk:
        lines_head = head_chunk.splitlines()
        for i, l in enumerate(lines_head):
            if l.strip() == b"$MeshFormat" and i + 1 < len(lines_head):
                fmt_parts = lines_head[i + 1].split()
                if len(fmt_parts) >= 2 and int(fmt_parts[1]) != 0:
                    data.version = fmt_parts[0].decode("ascii", errors="replace")
                    if data.version.startswith("4."):
                        return _parse_msh4_binary(filepath, data)
                    raise ValueError(
                        f"Binary MSH {data.version} is not supported; please use MSH 4.1 or ASCII."
                    )
                break

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    n_lines = len(lines)
    idx = 0

    while idx < n_lines:
        line = lines[idx].strip()
        idx += 1
        if not line:
            continue

        if line == "$MeshFormat":
            fmt_parts = lines[idx].strip().split()
            data.version = fmt_parts[0]
            while idx < n_lines and lines[idx].strip() != "$EndMeshFormat":
                idx += 1
            idx += 1

        elif line == "$PhysicalNames":
            num_phys = int(lines[idx].strip())
            idx += 1
            for _ in range(num_phys):
                parts = lines[idx].strip().split(maxsplit=2)
                idx += 1
                dim = int(parts[0])
                tag = int(parts[1])
                name = parts[2].strip().strip('"')
                data.physical_names[(dim, tag)] = name
            while idx < n_lines and lines[idx].strip() != "$EndPhysicalNames":
                idx += 1
            idx += 1

        elif line == "$Entities":
            end_idx = idx
            while end_idx < n_lines and lines[end_idx].strip() != "$EndEntities":
                end_idx += 1
            tokens = " ".join(lines[idx:end_idx]).split()
            idx = end_idx + 1

            t_pos = 0
            num_pts = int(tokens[t_pos])
            num_curves = int(tokens[t_pos + 1])
            num_surfs = int(tokens[t_pos + 2])
            num_vols = int(tokens[t_pos + 3])
            t_pos += 4

            for _ in range(num_pts):
                pt_tag = int(tokens[t_pos])
                num_phys = int(tokens[t_pos + 4])
                t_pos += 5
                phys_tags = [int(tokens[t_pos + k]) for k in range(num_phys)]
                t_pos += num_phys
                data.entity_phys_tags[(0, pt_tag)] = phys_tags

            for dim, count in ((1, num_curves), (2, num_surfs), (3, num_vols)):
                for _ in range(count):
                    ent_tag = int(tokens[t_pos])
                    num_phys = int(tokens[t_pos + 7])
                    t_pos += 8
                    phys_tags = [int(tokens[t_pos + k]) for k in range(num_phys)]
                    t_pos += num_phys
                    num_bound = int(tokens[t_pos])
                    t_pos += 1 + num_bound
                    data.entity_phys_tags[(dim, ent_tag)] = phys_tags

        elif line == "$Nodes":
            if data.version.startswith("2."):
                num_nodes = int(lines[idx].strip())
                idx += 1
                raw = np.fromstring(
                    "".join(lines[idx : idx + num_nodes]), dtype=np.float64, sep=" "
                ).reshape(num_nodes, 4)
                idx += num_nodes
                nids = raw[:, 0].astype(np.int64)
                max_tag = int(nids.max()) if nids.size > 0 else 0
                data.node_coords = np.zeros((max_tag + 1, 3), dtype=np.float64)
                data.valid_node_mask = np.zeros(max_tag + 1, dtype=bool)
                data.node_coords[nids] = raw[:, 1:4]
                data.valid_node_mask[nids] = True
            else:
                # MSH 4.x format: numEntityBlocks totalNumNodes minNodeTag maxNodeTag
                hdr = lines[idx].strip().split()
                idx += 1
                num_blocks = int(hdr[0])
                max_tag = int(hdr[3])
                data.node_coords = np.zeros((max_tag + 1, 3), dtype=np.float64)
                data.valid_node_mask = np.zeros(max_tag + 1, dtype=bool)

                for _ in range(num_blocks):
                    bhdr = lines[idx].strip().split()
                    idx += 1
                    ent_dim = int(bhdr[0])
                    parametric = int(bhdr[2])
                    num_in_block = int(bhdr[3])
                    if num_in_block == 0:
                        continue

                    # Fast path: standard Gmsh output writes 1 node ID per line, then 1 coord per line
                    nids = np.fromstring(
                        "".join(lines[idx : idx + num_in_block]), dtype=np.int64, sep=" "
                    )
                    if nids.size == num_in_block:
                        idx += num_in_block
                    else:
                        # Fallback if multiple node IDs were packed onto single lines
                        nid_list: List[int] = []
                        while len(nid_list) < num_in_block:
                            nid_list.extend(int(x) for x in lines[idx].strip().split())
                            idx += 1
                        nids = np.array(nid_list, dtype=np.int64)

                    n_cols = 3 + (ent_dim if parametric != 0 else 0)
                    coords = np.fromstring(
                        "".join(lines[idx : idx + num_in_block]), dtype=np.float64, sep=" "
                    ).reshape(num_in_block, n_cols)[:, :3]
                    idx += num_in_block

                    data.node_coords[nids] = coords
                    data.valid_node_mask[nids] = True

            while idx < n_lines and lines[idx].strip() != "$EndNodes":
                idx += 1
            idx += 1

        elif line == "$Elements":
            if data.version.startswith("2."):
                num_elems = int(lines[idx].strip())
                idx += 1
                quad_by_tag: Dict[int, List[List[int]]] = defaultdict(list)
                hex_by_tag: Dict[int, List[List[int]]] = defaultdict(list)
                for _ in range(num_elems):
                    parts = [int(x) for x in lines[idx].strip().split()]
                    idx += 1
                    etype = parts[1]
                    num_tags = parts[2]
                    phys_tag = parts[3] if num_tags > 0 else 0
                    conn = parts[3 + num_tags :]
                    if etype in GMSH_QUAD_TYPES:
                        quad_by_tag[phys_tag].append(conn[:4])
                    elif etype in GMSH_HEX_TYPES:
                        hex_by_tag[phys_tag].append(conn[:8])
                for ptag, rows in quad_by_tag.items():
                    data.quad_blocks.append((np.array(rows, dtype=np.int32), [ptag] if ptag else []))
                for ptag, rows in hex_by_tag.items():
                    data.hex_blocks.append((np.array(rows, dtype=np.int32), [ptag] if ptag else []))
            else:
                # MSH 4.x format: numEntityBlocks totalNumElements minElemTag maxElemTag
                hdr = lines[idx].strip().split()
                idx += 1
                num_blocks = int(hdr[0])
                for _ in range(num_blocks):
                    bhdr = lines[idx].strip().split()
                    idx += 1
                    ent_dim = int(bhdr[0])
                    ent_tag = int(bhdr[1])
                    etype = int(bhdr[2])
                    num_in_block = int(bhdr[3])
                    if num_in_block == 0:
                        continue
                    phys_tags = data.entity_phys_tags.get((ent_dim, ent_tag), [])

                    if etype in GMSH_QUAD_TYPES:
                        raw = np.fromstring(
                            "".join(lines[idx : idx + num_in_block]), dtype=np.int64, sep=" "
                        ).reshape(num_in_block, -1)
                        idx += num_in_block
                        data.quad_blocks.append((raw[:, 1:5].astype(np.int32, copy=False), phys_tags))
                    elif etype in GMSH_HEX_TYPES:
                        raw = np.fromstring(
                            "".join(lines[idx : idx + num_in_block]), dtype=np.int64, sep=" "
                        ).reshape(num_in_block, -1)
                        idx += num_in_block
                        data.hex_blocks.append((raw[:, 1:9].astype(np.int32, copy=False), phys_tags))
                    else:
                        idx += num_in_block
            while idx < n_lines and lines[idx].strip() != "$EndElements":
                idx += 1
            idx += 1

    return data


# ==============================================================================
# Topology Builder & NekCEM / NekWave Connectivity Assembly (Vectorized)
# ==============================================================================
class ConvertedNekMesh:
    def __init__(self) -> None:
        # Contiguous NumPy arrays for high-speed I/O
        self.corners_arr: Optional[np.ndarray] = None       # (nel, 8, 3) float64
        self.groups_arr: Optional[np.ndarray] = None        # (nel,) int32
        self.face_bc_id: Optional[np.ndarray] = None        # (nel, 6) uint8 index into bc_tags_table
        self.face_nbr_elem: Optional[np.ndarray] = None     # (nel, 6) int32 (1-based neighbor element)
        self.face_nbr_face: Optional[np.ndarray] = None     # (nel, 6) int8  (1-based neighbor face)
        self.bc_tags_table: List[str] = ["E", "PEC", "PMC", "PML", "P"]

        self.pml_thickness: int = 0
        self.min_det_j: float = 0.0
        self.max_det_j: float = 0.0
        self.num_flipped: int = 0
        self.bc_counts: Dict[str, int] = defaultdict(int)

        # Lazy compatibility lists (populated only if accessed by external callers)
        self._element_corners_list: Optional[List[List[Tuple[float, float, float]]]] = None
        self._element_groups_list: Optional[List[int]] = None
        self._face_bcs_list: Optional[List[List[Tuple[str, int, int]]]] = None

    @property
    def num_elements(self) -> int:
        if self.corners_arr is not None:
            return int(self.corners_arr.shape[0])
        return len(self._element_corners_list) if self._element_corners_list else 0

    @property
    def element_corners(self) -> List[List[Tuple[float, float, float]]]:
        if self._element_corners_list is None:
            if self.corners_arr is None:
                self._element_corners_list = []
            else:
                self._element_corners_list = [
                    [(float(c[0]), float(c[1]), float(c[2])) for c in elem]
                    for elem in self.corners_arr
                ]
        return self._element_corners_list

    @element_corners.setter
    def element_corners(self, val: List[List[Tuple[float, float, float]]]) -> None:
        self._element_corners_list = val

    @property
    def element_groups(self) -> List[int]:
        if self._element_groups_list is None:
            self._element_groups_list = (
                self.groups_arr.tolist() if self.groups_arr is not None else []
            )
        return self._element_groups_list

    @element_groups.setter
    def element_groups(self, val: List[int]) -> None:
        self._element_groups_list = val

    @property
    def face_bcs(self) -> List[List[Tuple[str, int, int]]]:
        if self._face_bcs_list is None:
            if self.face_bc_id is None or self.face_nbr_elem is None or self.face_nbr_face is None:
                self._face_bcs_list = []
            else:
                tags = self.bc_tags_table
                bc_id = self.face_bc_id
                ne = self.face_nbr_elem
                nf = self.face_nbr_face
                nel = bc_id.shape[0]
                self._face_bcs_list = [
                    [(tags[int(bc_id[e, f])], int(ne[e, f]), int(nf[e, f])) for f in range(6)]
                    for e in range(nel)
                ]
        return self._face_bcs_list

    @face_bcs.setter
    def face_bcs(self, val: List[List[Tuple[str, int, int]]]) -> None:
        self._face_bcs_list = val


def build_nek_mesh(
    msh: GmshMeshData,
    custom_bc_map: Optional[Dict[str, str]] = None,
    default_bc: str = "PEC",
    exclude_volumes: Optional[Set[str]] = None,
    coord_tol: float = 1e-8,
) -> ConvertedNekMesh:
    """
    Constructs the NekCEM / NekWave element corner arrays, positive-Jacobian
    orientations, 6-face neighbor connectivity, periodic face pairing, and
    PML layer thickness using vectorized NumPy operations.
    """
    if custom_bc_map is None:
        custom_bc_map = {}
    if exclude_volumes is None:
        exclude_volumes = {"SPHERE_PEC", "PEC_INTERIOR", "PEC_VOLUME"}
    else:
        exclude_volumes = {v.upper() for v in exclude_volumes}

    # Support GmshMeshData populated either via fast blocks or legacy .nodes/.hexes
    if msh.node_coords is None:
        if not msh.nodes:
            raise ValueError("No nodes found in GmshMeshData!")
        max_nid = max(msh.nodes.keys())
        msh.node_coords = np.zeros((max_nid + 1, 3), dtype=np.float64)
        msh.valid_node_mask = np.zeros(max_nid + 1, dtype=bool)
        for nid, xyz in msh.nodes.items():
            msh.node_coords[nid] = xyz
            msh.valid_node_mask[nid] = True
    if not msh.hex_blocks and msh._hexes_list:
        msh.hex_blocks = [
            (np.array([h[0] for h in msh._hexes_list], dtype=np.int32), msh._hexes_list[0][1])
        ]

    if not msh.hex_blocks:
        raise ValueError("No 3D hexahedral elements found in the Gmsh mesh!")

    node_coords = msh.node_coords
    valid_mask = msh.valid_node_mask
    max_nid = node_coords.shape[0] - 1

    # 1. Fast spatial node deduplication via 64-bit spatial hash pre-check
    inv_tol = 1.0 / coord_tol
    valid_nids = np.flatnonzero(valid_mask).astype(np.int32)
    quant = np.rint(node_coords[valid_nids] * inv_tol).astype(np.int64)
    spatial_hash = (
        (quant[:, 0].astype(np.uint64) * np.uint64(73856093))
        ^ (quant[:, 1].astype(np.uint64) * np.uint64(19349663))
        ^ (quant[:, 2].astype(np.uint64) * np.uint64(83492791))
    )
    sh_sorted = np.sort(spatial_hash)
    canon_id = np.arange(max_nid + 1, dtype=np.int32)
    if sh_sorted.size > 1 and np.any(sh_sorted[:-1] == sh_sorted[1:]):
        q_dtype = np.dtype([("x", "<i8"), ("y", "<i8"), ("z", "<i8")])
        q_struct = np.ascontiguousarray(quant).view(q_dtype).ravel()
        _, first_idx, inv_idx = np.unique(q_struct, return_index=True, return_inverse=True)
        canon_id[valid_nids] = valid_nids[first_idx[inv_idx]]

    # Canonical BC tag lookup table
    bc_tags_table: List[str] = ["E", "PEC", "PMC", "PML", "P"]
    tag_to_id: Dict[str, int] = {t: i for i, t in enumerate(bc_tags_table)}

    def get_bc_id(tag: str) -> int:
        if tag not in tag_to_id:
            tag_to_id[tag] = len(bc_tags_table)
            bc_tags_table.append(tag)
        return tag_to_id[tag]

    default_bc_id = get_bc_id(default_bc)
    pec_bc_id = get_bc_id("PEC")
    per_bc_id = get_bc_id("P")
    pml_bc_id = get_bc_id("PML")

    # 2. Filter active 3D hexahedral blocks and excluded PEC volume blocks
    active_hex_parts: List[np.ndarray] = []
    active_grp_parts: List[np.ndarray] = []
    active_pml_parts: List[np.ndarray] = []
    excluded_hex_parts: List[np.ndarray] = []

    for h_arr, phys_tags in msh.hex_blocks:
        n_blk = h_arr.shape[0]
        if n_blk == 0:
            continue
        vol_names = [msh.physical_names.get((3, pt), "") for pt in phys_tags]
        is_excluded = any(vn.upper() in exclude_volumes for vn in vol_names if vn)
        if is_excluded:
            excluded_hex_parts.append(h_arr)
            continue

        is_pml_vol = any("PML" in vn.upper() for vn in vol_names if vn)
        grp_id = phys_tags[0] if phys_tags else 0
        active_hex_parts.append(h_arr)
        active_grp_parts.append(np.full(n_blk, grp_id, dtype=np.int32))
        active_pml_parts.append(np.full(n_blk, is_pml_vol, dtype=bool))

    if not active_hex_parts:
        raise ValueError("All 3D hexahedral elements were excluded!")

    hex_nodes = np.vstack(active_hex_parts)
    groups_arr = np.concatenate(active_grp_parts)
    is_pml_elem = np.concatenate(active_pml_parts)
    nel = hex_nodes.shape[0]

    # 3. Vectorized positive-Jacobian enforcement and corner det(J) bounds
    out = ConvertedNekMesh()
    out.bc_tags_table = bc_tags_table
    hex_nodes, corners_arr, num_flipped, min_j, max_j = ensure_positive_jacobian_batch(
        hex_nodes, node_coords
    )
    out.corners_arr = corners_arr
    out.groups_arr = groups_arr
    out.num_flipped = num_flipped
    out.min_det_j = min_j
    out.max_det_j = max_j

    # 4. Vectorized 6-face neighbor discovery via 5-comparator sort + uint64 SIMD radix sort
    canon_hex = canon_id[hex_nodes]  # (nel, 8) int32
    all_faces = _sort4_cols(canon_hex[:, NEK_FACE_CORNERS_ARR].reshape(nel * 6, 4))
    # Guard against coord_tol collapsing ultra-thin boundary layer elements (where edge < coord_tol)
    if np.any(all_faces[:, :-1] == all_faces[:, 1:]):
        canon_id = np.arange(max_nid + 1, dtype=np.int32)
        canon_hex = hex_nodes
        all_faces = _sort4_cols(canon_hex[:, NEK_FACE_CORNERS_ARR].reshape(nel * 6, 4))

    face_keys = _face_keys_uint64(all_faces, max_nid)
    order = np.argsort(face_keys)
    sorted_keys = face_keys[order]

    # If any uint64 hash bucket has >= 3 faces (possible only for >2.1M nodes), tie-break by exact 4-tuple
    if max_nid >= (1 << 21) and sorted_keys.size > 2:
        hash_eq = sorted_keys[:-1] == sorted_keys[1:]
        if np.any(hash_eq[:-1] & hash_eq[1:]):
            exact_struct = np.ascontiguousarray(all_faces).view(FACE_KEY_DTYPE).ravel()
            order = np.argsort(exact_struct, kind="mergesort")
            sorted_keys = face_keys[order]

    s_faces = all_faces[order]
    same_next = (
        (s_faces[:-1, 0] == s_faces[1:, 0])
        & (s_faces[:-1, 1] == s_faces[1:, 1])
        & (s_faces[:-1, 2] == s_faces[1:, 2])
        & (s_faces[:-1, 3] == s_faces[1:, 3])
    )
    if same_next.size > 1 and np.any(same_next[:-1] & same_next[1:]):
        raise ValueError("Non-manifold hexahedral face detected (shared by >= 3 elements)!")


    face_bc_id = np.full((nel, 6), default_bc_id, dtype=np.uint8)
    face_nbr_elem = np.zeros((nel, 6), dtype=np.int32)
    face_nbr_face = np.zeros((nel, 6), dtype=np.int8)

    # Interior paired faces ("E")
    pair_pos = np.flatnonzero(same_next)
    flat1 = order[pair_pos]
    flat2 = order[pair_pos + 1]
    e1, f1 = np.divmod(flat1, 6)
    e2, f2 = np.divmod(flat2, 6)

    face_bc_id[e1, f1] = 0  # 0 == "E"
    face_bc_id[e2, f2] = 0
    face_nbr_elem[e1, f1] = e2 + 1
    face_nbr_elem[e2, f2] = e1 + 1
    face_nbr_face[e1, f1] = f2 + 1
    face_nbr_face[e2, f2] = f1 + 1
    if pair_pos.size > 0:
        out.bc_counts["E"] = int(pair_pos.size * 2)

    # Unpaired boundary faces
    is_paired = np.zeros(nel * 6, dtype=bool)
    is_paired[pair_pos] = True
    is_paired[pair_pos + 1] = True
    bnd_flat = order[~is_paired]
    bnd_keys = sorted_keys[~is_paired]  # Already sorted because `sorted_keys` is sorted!
    bnd_e, bnd_f = np.divmod(bnd_flat, 6)
    bnd_bc_ids = np.full(bnd_flat.size, default_bc_id, dtype=np.uint8)

    # 5a. If any excluded volume hexes exist, mark exposed boundary faces as PEC
    if excluded_hex_parts:
        excl_hex = canon_id[np.vstack(excluded_hex_parts)]
        excl_faces = _sort4_cols(excl_hex[:, NEK_FACE_CORNERS_ARR].reshape(-1, 4))
        excl_keys = _face_keys_uint64(excl_faces, max_nid)
        in_excl = np.isin(bnd_keys, excl_keys)
        bnd_bc_ids[in_excl] = pec_bc_id

    # 5b. Match 2D boundary quads against boundary faces via vectorized searchsorted
    if msh.quad_blocks:
        quad_key_parts: List[np.ndarray] = []
        quad_id_parts: List[np.ndarray] = []
        for q_arr, phys_tags in msh.quad_blocks:
            if q_arr.shape[0] == 0:
                continue
            bc_tag = default_bc
            for ptag in phys_tags:
                pname = msh.physical_names.get((2, ptag), "")
                if pname:
                    bc_tag = infer_bc_tag(pname, custom_bc_map)
                    break
            bid = get_bc_id(bc_tag)
            q_canon = _sort4_cols(canon_id[q_arr])
            q_keys = _face_keys_uint64(q_canon, max_nid)
            quad_key_parts.append(q_keys)
            quad_id_parts.append(np.full(q_keys.size, bid, dtype=np.uint8))

        if quad_key_parts and bnd_keys.size > 0:
            all_q_keys = np.concatenate(quad_key_parts)
            all_q_ids = np.concatenate(quad_id_parts)
            # bnd_keys is already sorted! Find where each 2D quad key lands in bnd_keys:
            pos_in_bnd = np.searchsorted(bnd_keys, all_q_keys)
            valid_hit = pos_in_bnd < bnd_keys.size
            hit_indices = np.flatnonzero(valid_hit)
            matched = bnd_keys[pos_in_bnd[hit_indices]] == all_q_keys[hit_indices]
            matched_hits = hit_indices[matched]
            bnd_bc_ids[pos_in_bnd[matched_hits]] = all_q_ids[matched_hits]

    face_bc_id[bnd_e, bnd_f] = bnd_bc_ids

    # 6. Pair periodic boundary faces if any exist
    per_mask = bnd_bc_ids == per_bc_id
    if np.any(per_mask):
        per_e = bnd_e[per_mask]
        per_f = bnd_f[per_mask]
        # Compute centroids of periodic faces
        per_centroids = [
            (
                int(e),
                int(f),
                (
                    float(corners_arr[e, NEK_FACE_CORNERS_ARR[f], 0].mean()),
                    float(corners_arr[e, NEK_FACE_CORNERS_ARR[f], 1].mean()),
                    float(corners_arr[e, NEK_FACE_CORNERS_ARR[f], 2].mean()),
                ),
            )
            for e, f in zip(per_e, per_f)
        ]
        paired: Set[int] = set()
        for i, (pe1, pf1, (cx1, cy1, cz1)) in enumerate(per_centroids):
            if i in paired:
                continue
            best_j = -1
            best_dist = float("inf")
            for j in range(i + 1, len(per_centroids)):
                if j in paired:
                    continue
                pe2, pf2, (cx2, cy2, cz2) = per_centroids[j]
                dx, dy, dz = abs(cx1 - cx2), abs(cy1 - cy2), abs(cz1 - cz2)
                td = sorted([dx, dy, dz])
                if td[0] < 1e-5 and td[1] < 1e-5 and td[2] > 1e-5:
                    err = td[0] + td[1]
                    if err < best_dist:
                        best_dist = err
                        best_j = j
            if best_j >= 0:
                paired.add(i)
                paired.add(best_j)
                pe2, pf2, _ = per_centroids[best_j]
                face_nbr_elem[pe1, pf1] = pe2 + 1
                face_nbr_elem[pe2, pf2] = pe1 + 1
                face_nbr_face[pe1, pf1] = pf2 + 1
                face_nbr_face[pe2, pf2] = pf1 + 1
            else:
                raise ValueError(
                    f"Could not find matching periodic partner for element {pe1 + 1} face {pf1 + 1}"
                )

    # Count boundary condition occurrences
    unique_bids, bid_counts = np.unique(bnd_bc_ids, return_counts=True)
    for bid, cnt in zip(unique_bids, bid_counts):
        out.bc_counts[bc_tags_table[int(bid)]] += int(cnt)

    out.face_bc_id = face_bc_id
    out.face_nbr_elem = face_nbr_elem
    out.face_nbr_face = face_nbr_face

    # 7. Fast O(N_PML_faces) PML layer thickness detection
    if out.bc_counts.get("PML", 0) > 0:
        has_pml_vol = bool(np.any(is_pml_elem))
        pml_face_mask = bnd_bc_ids == pml_bc_id
        pml_start_e = bnd_e[pml_face_mask]
        pml_start_f = bnd_f[pml_face_mask]

        # Sample up to 64 outer PML faces (enough to cover all 6 box sides in microseconds)
        step_stride = max(1, pml_start_e.size // 64)
        layer_depths: List[int] = []
        for e0, f0 in zip(pml_start_e[::step_stride], pml_start_f[::step_stride]):
            depth = 0
            curr_e, curr_f = int(e0), int(f0)
            visited: Set[int] = set()
            while curr_e not in visited:
                visited.add(curr_e)
                if has_pml_vol and not is_pml_elem[curr_e]:
                    break
                depth += 1
                opp_f = int(NEK_OPP_FACE_ARR[curr_f])
                if face_bc_id[curr_e, opp_f] != 0:  # 0 == "E"
                    break
                nbr_e1 = int(face_nbr_elem[curr_e, opp_f])
                nbr_f1 = int(face_nbr_face[curr_e, opp_f])
                if nbr_e1 <= 0:
                    break
                curr_e = nbr_e1 - 1
                curr_f = nbr_f1 - 1
            if depth > 0:
                layer_depths.append(depth)
        if layer_depths:
            out.pml_thickness = min(layer_depths)

    return out


# ==============================================================================
# NekCEM / NekWave ASCII .rea Writer (Chunk-Buffered)
# ==============================================================================
def _elem_AlphaTag(e_1based: int) -> str:
    """Formats Nekton's classic [ 1a] element tag string."""
    cycle = (e_1based - 1) // 26 + 1
    letter = chr(ord("a") + ((e_1based - 1) % 26))
    return f"{cycle:4d}{letter}"


def write_rea_file(
    filepath: str,
    mesh: ConvertedNekMesh,
    order: int = 4,
    pml_order: float = 3.0,
    pml_reflect_err: float = 1.0e-6,
    use_re2_companion: bool = False,
) -> None:
    """
    Writes a NekCEM / NekWave compatible ASCII .rea file using buffered chunk I/O.
    If use_re2_companion=True, writes -NEL in the **MESH DATA** header so
    NekWave / NekCEM redirects to the binary .re2 file in the same directory.
    """
    nel = mesh.num_elements
    pml_thick = mesh.pml_thickness if mesh.pml_thickness > 0 else 2

    with open(filepath, "w", encoding="utf-8", buffering=8 * 1024 * 1024) as f:
        header_lines = [
            " ****** PARAMETERS *****\n",
            "    2.610000     NEKTON VERSION\n",
            " 3 DIMENSIONAL RUN\n",
            " 103 PARAMETERS FOLLOW\n",
        ]
        for p in range(1, 104):
            if p == 4:
                header_lines.append("   1                             4: ifte (1), iftm (2)\n")
            elif p == 10:
                header_lines.append("   1.0                           10: fintim\n")
            elif p == 11:
                header_lines.append("   1000                          11: nsteps\n")
            elif p == 12:
                header_lines.append("  -0.001                         12: dt\n")
            elif p == 13:
                header_lines.append("   10                            13: iocomm\n")
            elif p == 15:
                header_lines.append("   100                           15: iostep\n")
            elif p == 20:
                header_lines.append(f"   {float(order):.5f}                       20: ORDER (MESH)\n")
            elif p == 77:
                header_lines.append(
                    f"   {float(pml_thick):.6E}                  77: PMLTHICK : thickness of the PML in levels\n"
                )
            elif p == 78:
                header_lines.append(
                    f"   {float(pml_order):.6E}                  78: PMLORDER : polynomial order of the grading of sigma\n"
                )
            elif p == 79:
                header_lines.append(
                    f"   {float(pml_reflect_err):.6E}                  79: PMLREFERR : PML reflection error\n"
                )
            else:
                header_lines.append(f"   0                             {p}: ---\n")

        header_lines.extend(
            [
                "           4  Lines of passive scalar data follows2 CONDUCT; 2RHOCP\n",
                "   1.00000       1.00000       1.00000       1.00000       1.00000\n",
                "   1.00000       1.00000       1.00000       1.00000\n",
                "   1.00000       1.00000       1.00000       1.00000       1.00000\n",
                "   1.00000       1.00000       1.00000       1.00000\n",
                " 13 LOGICAL SWITCHES FOLLOW\n",
                "  F                             1: IFFLOW\n",
                "  T                             2: IFHEAT\n",
                "  T                             3: IFTRAN\n",
                "  F                             4: IFSRC\n",
                "  F                             5: IFCENTRAL\n",
                "  T                             6: IFUPWIND\n",
                "  F                             7: IFTM (2D only)\n",
                "  T                             8: IFTE (2D only)\n",
                "  F                             9: IFDEALIAS\n",
                "  F                             10: IFRK4\n",
                "  T                             11: IFEXP\n",
                "  F                             12: IFEIG\n",
                "  F                             13: IFNM\n",
                "   9.23077       9.23077      -4.38462      -5.30769     XFAC,YFAC,XZERO,YZERO\n",
                " **MESH DATA** 6 lines are X,Y,Z;X,Y,Z. Columns corners 1-4;5-8\n",
            ]
        )
        f.writelines(header_lines)

        if use_re2_companion:
            f.write(f"   {-nel:9d}  3   {nel:9d}           NEL,NDIM,NELV\n")
        else:
            f.write(f"   {nel:9d}  3   {nel:9d}           NEL,NDIM,NELV\n")

            if mesh.corners_arr is not None:
                corners_arr = mesh.corners_arr
                letters = [chr(ord("a") + k) for k in range(26)]
                chunk: List[str] = []
                for e_idx in range(nel):
                    e_1 = e_idx + 1
                    cycle = e_idx // 26 + 1
                    atag = f"{cycle:4d}{letters[e_idx % 26]}"
                    c = corners_arr[e_idx]
                    chunk.append(
                        f"            ELEMENT {e_1:11d} [{atag}]  GROUP  0\n"
                        f"{c[0,0]:14.6f}{c[1,0]:14.6f}{c[2,0]:14.6f}{c[3,0]:14.6f}\n"
                        f"{c[0,1]:14.6f}{c[1,1]:14.6f}{c[2,1]:14.6f}{c[3,1]:14.6f}\n"
                        f"{c[0,2]:14.6f}{c[1,2]:14.6f}{c[2,2]:14.6f}{c[3,2]:14.6f}\n"
                        f"{c[4,0]:14.6f}{c[5,0]:14.6f}{c[6,0]:14.6f}{c[7,0]:14.6f}\n"
                        f"{c[4,1]:14.6f}{c[5,1]:14.6f}{c[6,1]:14.6f}{c[7,1]:14.6f}\n"
                        f"{c[4,2]:14.6f}{c[5,2]:14.6f}{c[6,2]:14.6f}{c[7,2]:14.6f}\n"
                    )
                    if len(chunk) >= 4096:
                        f.write("".join(chunk))
                        chunk.clear()
                if chunk:
                    f.write("".join(chunk))
                    chunk.clear()
            else:
                for e_idx, corners in enumerate(mesh.element_corners):
                    e_1 = e_idx + 1
                    atag = _elem_AlphaTag(e_1)
                    f.write(f"            ELEMENT {e_1:11d} [{atag}]  GROUP  0\n")
                    f.write("".join(f"{corners[c][0]:14.6f}" for c in range(0, 4)) + "\n")
                    f.write("".join(f"{corners[c][1]:14.6f}" for c in range(0, 4)) + "\n")
                    f.write("".join(f"{corners[c][2]:14.6f}" for c in range(0, 4)) + "\n")
                    f.write("".join(f"{corners[c][0]:14.6f}" for c in range(4, 8)) + "\n")
                    f.write("".join(f"{corners[c][1]:14.6f}" for c in range(4, 8)) + "\n")
                    f.write("".join(f"{corners[c][2]:14.6f}" for c in range(4, 8)) + "\n")

            f.write("  ***** CURVED SIDE DATA *****\n")
            f.write("           0 Curved sides follow IEDGE,IEL,CURVE(I),I=1,5, CCURVE\n")
            f.write("  ***** BOUNDARY CONDITIONS *****\n")
            f.write("  ***** FLUID   BOUNDARY CONDITIONS *****\n")

            if (
                mesh.face_bc_id is not None
                and mesh.face_nbr_elem is not None
                and mesh.face_nbr_face is not None
            ):
                pad_tags = [f"{t:<3s}" for t in mesh.bc_tags_table]
                bc_id = mesh.face_bc_id
                ne_arr = mesh.face_nbr_elem
                nf_arr = mesh.face_nbr_face
                bc_chunk: List[str] = []
                tail = ".00000       0.00000       0.00000       0.00000\n"
                for e_idx in range(nel):
                    e_1 = e_idx + 1
                    b_row = bc_id[e_idx]
                    ne_row = ne_arr[e_idx]
                    nf_row = nf_arr[e_idx]
                    for f_idx in range(6):
                        bc_chunk.append(
                            f" {pad_tags[int(b_row[f_idx])]} {e_1:7d} {f_idx + 1:2d} "
                            f"{int(ne_row[f_idx]):7d}.00000 {int(nf_row[f_idx]):7d}{tail}"
                        )
                    if len(bc_chunk) >= 24576:
                        f.write("".join(bc_chunk))
                        bc_chunk.clear()
                if bc_chunk:
                    f.write("".join(bc_chunk))
                    bc_chunk.clear()
            else:
                for e_idx, faces in enumerate(mesh.face_bcs):
                    e_1 = e_idx + 1
                    for f_idx, (bc_tag, nbr_e, nbr_f) in enumerate(faces):
                        f_1 = f_idx + 1
                        f.write(
                            f" {bc_tag:<3s} {e_1:7d} {f_1:2d} {float(nbr_e):13.5f} {float(nbr_f):13.5f}"
                            "       0.00000       0.00000       0.00000\n"
                        )

        f.write(
            "  ***** NO THERMAL BOUNDARY CONDITIONS *****\n"
            "            0 PRESOLVE/RESTART OPTIONS  *****\n"
            "            7         INITIAL CONDITIONS *****\n"
            "C Default\nC Default\nC Default\nC Default\nC Default\nC Default\nC Default\n"
            "  ***** DRIVE FORCE DATA ***** BODY FORCE, FLOW, Q\n"
            "            4                 Lines of Drive force data follow\n"
            "C\nC\nC\nC\n"
            "  ***** Variable Property Data ***** Overrrides Parameter data.\n"
            "            1 Lines follow.\n"
            "            0 PACKETS OF DATA FOLLOW\n"
            "  ***** HISTORY AND INTEGRAL DATA *****\n"
            "            0   POINTS.  Hcode, I,J,H,IEL\n"
            "  ***** OUTPUT FIELD SPECIFICATION *****\n"
            "            6 SPECIFICATIONS FOLLOW\n"
            "  F      COORDINATES\n"
            "  F      VELOCITY\n"
            "  F      PRESSURE\n"
            "  T      TEMPERATURE\n"
            "  F      TEMPERATURE GRADIENT\n"
            "            0      PASSIVE SCALARS\n"
            "  ***** OBJECT SPECIFICATION *****\n"
            "       0 Surface Objects\n"
            "       0 Volume  Objects\n"
            "       0 Edge    Objects\n"
            "       0 Point   Objects\n"
        )


# ==============================================================================
# NekCEM / NekWave Binary .re2 Writer (#v002 format - Single-Syscall Vectorized)
# ==============================================================================
RE2_BC_DTYPE = np.dtype(
    [
        ("e", "<f8"),
        ("f", "<f8"),
        ("ne", "<f8"),
        ("nf", "<f8"),
        ("p1", "<f8"),
        ("p2", "<f8"),
        ("p3", "<f8"),
        ("tag", "S8"),
    ]
)


def write_re2_file(filepath: str, mesh: ConvertedNekMesh) -> None:
    """
    Writes a NekCEM / NekWave compatible binary .re2 file (#v002 format)
    using vectorized NumPy buffers, matching Mesh::loadFromRe2() in src/mesh.cpp.
    """
    nel = mesh.num_elements

    with open(filepath, "wb") as f:
        # 1. 80-byte ASCII header + 4-byte float endianness tag (6.54321)
        hdr_str = f"#v002{nel:9d}{3:3d}{nel:9d} this is the hdr".ljust(80)[:80]
        f.write(hdr_str.encode("ascii"))
        f.write(struct.pack("<f", 6.54321))

        # 2. Vectorized element block: (nel, 25) float64 -> [group, x0..x7, y0..y7, z0..z7]
        if mesh.corners_arr is not None and mesh.groups_arr is not None:
            elem_block = np.empty((nel, 25), dtype="<f8")
            elem_block[:, 0] = mesh.groups_arr
            elem_block[:, 1:9] = mesh.corners_arr[:, :, 0]
            elem_block[:, 9:17] = mesh.corners_arr[:, :, 1]
            elem_block[:, 17:25] = mesh.corners_arr[:, :, 2]
            f.write(elem_block.tobytes())
        else:
            for e_idx, corners in enumerate(mesh.element_corners):
                grp = float(mesh.element_groups[e_idx])
                xs = [corners[c][0] for c in range(8)]
                ys = [corners[c][1] for c in range(8)]
                zs = [corners[c][2] for c in range(8)]
                f.write(struct.pack("<25d", grp, *xs, *ys, *zs))

        # 3. Curved sides count (0.0)
        f.write(struct.pack("<d", 0.0))

        # 4. Vectorized boundary condition records (all non-"E" faces)
        if (
            mesh.face_bc_id is not None
            and mesh.face_nbr_elem is not None
            and mesh.face_nbr_face is not None
        ):
            bnd_e, bnd_f = np.nonzero(mesh.face_bc_id != 0)
            nbc = int(bnd_e.size)
            f.write(struct.pack("<d", float(nbc)))
            if nbc > 0:
                tag_bytes_lut = np.array(
                    [t.ljust(8)[:8].encode("ascii") for t in mesh.bc_tags_table], dtype="S8"
                )
                bc_buf = np.zeros(nbc, dtype=RE2_BC_DTYPE)
                bc_buf["e"] = bnd_e + 1
                bc_buf["f"] = bnd_f + 1
                bc_buf["ne"] = mesh.face_nbr_elem[bnd_e, bnd_f]
                bc_buf["nf"] = mesh.face_nbr_face[bnd_e, bnd_f]
                bc_buf["tag"] = tag_bytes_lut[mesh.face_bc_id[bnd_e, bnd_f]]
                f.write(bc_buf.tobytes())
        else:
            bc_records: List[Tuple[int, int, int, int, str]] = []
            for e_idx, faces in enumerate(mesh.face_bcs):
                e_1 = e_idx + 1
                for f_idx, (bc_tag, nbr_e, nbr_f) in enumerate(faces):
                    if bc_tag != "E":
                        bc_records.append((e_1, f_idx + 1, nbr_e, nbr_f, bc_tag))
            f.write(struct.pack("<d", float(len(bc_records))))
            for e_1, f_1, nbr_e, nbr_f, bc_tag in bc_records:
                tag_bytes = bc_tag.ljust(8)[:8].encode("ascii")
                f.write(
                    struct.pack(
                        "<7d8s",
                        float(e_1),
                        float(f_1),
                        float(nbr_e),
                        float(nbr_f),
                        0.0,
                        0.0,
                        0.0,
                        tag_bytes,
                    )
                )


# ==============================================================================
# Direct In-Memory Gmsh Python API Extractor & Converter (Zero .msh Disk I/O)
# ==============================================================================
def extract_from_gmsh_model(gmsh_module) -> GmshMeshData:
    """
    Extracts GmshMeshData directly from an active in-memory `gmsh.model`
    using C-level NumPy arrays (`getNodes`, `getElements`), bypassing
    any disk .msh serialization/parsing overhead.
    """
    data = GmshMeshData()
    data.version = "4.1 (in-memory)"

    for dim, ptag in gmsh_module.model.getPhysicalGroups():
        pname = gmsh_module.model.getPhysicalName(dim, ptag)
        data.physical_names[(int(dim), int(ptag))] = pname
        for ent_tag in gmsh_module.model.getEntitiesForPhysicalGroup(dim, ptag):
            data.entity_phys_tags.setdefault((int(dim), int(ent_tag)), []).append(int(ptag))

    node_tags, node_coords_flat, _ = gmsh_module.model.mesh.getNodes()
    nids = np.asarray(node_tags, dtype=np.int64)
    coords = np.asarray(node_coords_flat, dtype=np.float64).reshape(-1, 3)
    max_tag = int(nids.max()) if nids.size > 0 else 0
    data.node_coords = np.zeros((max_tag + 1, 3), dtype=np.float64)
    data.valid_node_mask = np.zeros(max_tag + 1, dtype=bool)
    data.node_coords[nids] = coords
    data.valid_node_mask[nids] = True

    for dim in (2, 3):
        for _, ent_tag in gmsh_module.model.getEntities(dim):
            phys_tags = data.entity_phys_tags.get((int(dim), int(ent_tag)), [])
            etypes, _, enodes_list = gmsh_module.model.mesh.getElements(dim, ent_tag)
            for etype, enodes in zip(etypes, enodes_list):
                et = int(etype)
                n_per_elem = GMSH_ELEM_NUM_NODES.get(et, 0)
                if n_per_elem == 0:
                    continue
                arr = np.asarray(enodes, dtype=np.int32).reshape(-1, n_per_elem)
                if et in GMSH_QUAD_TYPES:
                    data.quad_blocks.append((arr[:, :4], phys_tags))
                elif et in GMSH_HEX_TYPES:
                    data.hex_blocks.append((arr[:, :8], phys_tags))

    return data


def convert_gmsh_model_to_nek(
    gmsh_module,
    output_prefix: str,
    output_format: str = "re2",
    custom_bc_map: Optional[Dict[str, str]] = None,
    default_bc: str = "PEC",
    exclude_volumes: Optional[Set[str]] = None,
    order: int = 4,
    pml_order: float = 3.0,
    pml_reflect_err: float = 1.0e-6,
    verbose: bool = True,
) -> ConvertedNekMesh:
    """
    Directly converts the currently active `gmsh.model` in RAM into NekCEM / NekWave
    `.re2` and/or `.rea` files without writing or reading an intermediate `.msh` file.
    """
    t0 = time.perf_counter()
    msh_data = extract_from_gmsh_model(gmsh_module)
    t_parse = time.perf_counter()

    nek_mesh = build_nek_mesh(
        msh_data,
        custom_bc_map=custom_bc_map,
        default_bc=default_bc,
        exclude_volumes=exclude_volumes,
    )
    t_build = time.perf_counter()

    fmt = output_format.lower()
    rea_path = output_prefix + ".rea"
    re2_path = output_prefix + ".re2"

    if fmt in ("rea", "both"):
        write_rea_file(
            rea_path,
            nek_mesh,
            order=order,
            pml_order=pml_order,
            pml_reflect_err=pml_reflect_err,
            use_re2_companion=False,
        )
    if fmt in ("re2", "both"):
        write_re2_file(re2_path, nek_mesh)
        if fmt == "re2":
            write_rea_file(
                rea_path,
                nek_mesh,
                order=order,
                pml_order=pml_order,
                pml_reflect_err=pml_reflect_err,
                use_re2_companion=True,
            )
    t_write = time.perf_counter()

    if verbose:
        nel = nek_mesh.num_elements
        print("==========================================================")
        print("  In-Memory Gmsh -> NekCEM / NekWave Conversion Summary")
        print("==========================================================")
        print(f"  Total 3D Hex Elements:    {nel}")
        print(f"  Reoriented Hexes (J>0):   {nek_mesh.num_flipped}")
        print(f"  Corner det(J) range:      [{nek_mesh.min_det_j:.4e}, {nek_mesh.max_det_j:.4e}]")
        print(f"  Total Conversion Time:    {t_write - t0:.3f} s "
              f"(extract={t_parse - t0:.3f}s, build={t_build - t_parse:.3f}s, write={t_write - t_build:.3f}s)")
        print("==========================================================")

    return nek_mesh


# ==============================================================================
# High-Level Conversion Entry Point & CLI
# ==============================================================================
def convert_gmsh_to_nek(
    msh_path: str,
    output_prefix: Optional[str] = None,
    output_format: str = "both",
    custom_bc_map: Optional[Dict[str, str]] = None,
    default_bc: str = "PEC",
    exclude_volumes: Optional[Set[str]] = None,
    order: int = 4,
    pml_order: float = 3.0,
    pml_reflect_err: float = 1.0e-6,
    verbose: bool = True,
) -> ConvertedNekMesh:
    """
    Converts a Gmsh `.msh` file into NekCEM / NekWave `.rea` and/or `.re2` files.
    """
    if output_prefix is None:
        output_prefix = os.path.splitext(msh_path)[0]

    t0 = time.perf_counter()
    msh_data = parse_msh_file(msh_path)
    t_parse = time.perf_counter()

    nek_mesh = build_nek_mesh(
        msh_data,
        custom_bc_map=custom_bc_map,
        default_bc=default_bc,
        exclude_volumes=exclude_volumes,
    )
    t_build = time.perf_counter()

    fmt = output_format.lower()
    rea_path = output_prefix + ".rea"
    re2_path = output_prefix + ".re2"

    if fmt in ("rea", "both"):
        write_rea_file(
            rea_path,
            nek_mesh,
            order=order,
            pml_order=pml_order,
            pml_reflect_err=pml_reflect_err,
            use_re2_companion=False,
        )
    if fmt in ("re2", "both"):
        write_re2_file(re2_path, nek_mesh)
        if fmt == "re2":
            # Also write a lightweight companion .rea with -NEL pointing to .re2
            write_rea_file(
                rea_path,
                nek_mesh,
                order=order,
                pml_order=pml_order,
                pml_reflect_err=pml_reflect_err,
                use_re2_companion=True,
            )
    t_write = time.perf_counter()

    if verbose:
        nel = nek_mesh.num_elements
        print("==========================================================")
        print("  Gmsh -> NekCEM / NekWave Mesh Conversion Summary")
        print("==========================================================")
        print(f"  Input Gmsh file:          {msh_path} (MSH {msh_data.version})")
        print(f"  Total 3D Hex Elements:    {nel}")
        print(f"  Reoriented Hexes (J>0):   {nek_mesh.num_flipped}")
        print(f"  Corner det(J) range:      [{nek_mesh.min_det_j:.4e}, {nek_mesh.max_det_j:.4e}]")
        print("  Face Boundary Counts:")
        for tag, cnt in sorted(nek_mesh.bc_counts.items()):
            label = "Interior (E)" if tag == "E" else f"Boundary ({tag})"
            print(f"    - {label:<18s}: {cnt}")
        if nek_mesh.pml_thickness > 0:
            print(f"  Detected PML Thickness:   {nek_mesh.pml_thickness} element layers")
        if fmt in ("rea", "both", "re2"):
            mode_note = " (companion header -> .re2)" if fmt == "re2" else ""
            print(f"  Wrote ASCII .rea file:    {rea_path}{mode_note}")
        if fmt in ("re2", "both"):
            print(f"  Wrote Binary .re2 file:   {re2_path}")
        print("  Conversion Timings:")
        print(f"    - Parse .msh:           {t_parse - t0:.3f} s")
        print(f"    - Build & Jacobians:    {t_build - t_parse:.3f} s")
        print(f"    - Write output files:   {t_write - t_build:.3f} s")
        print(f"    - Total elapsed:        {t_write - t0:.3f} s")
        print("==========================================================")

    return nek_mesh


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Gmsh (.msh) hexahedral meshes to NekCEM / NekWave (.rea / .re2) format."
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_flag",
        default=None,
        help="Path to input Gmsh .msh file (e.g. --input sphereRCS/rcs_mesh.msh)",
    )
    parser.add_argument(
        "input_pos",
        nargs="?",
        default=None,
        help="Optional positional path to input Gmsh .msh file",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file prefix (default: same folder and basename as input .msh)",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["rea", "re2", "both"],
        default="both",
        help="Output mesh format: 'rea' (ASCII), 're2' (binary + companion .rea, fastest for big meshes), or 'both' (default)",
    )
    parser.add_argument(
        "--bc",
        action="append",
        default=[],
        metavar="PHYS_NAME=TAG",
        help="Override Physical Surface Name to BC tag mapping (e.g. --bc PML_Boundaries=PML --bc PEC_Boundary=PEC)",
    )
    parser.add_argument(
        "--default-bc",
        default="PEC",
        choices=["PEC", "PMC", "PML"],
        help="Default boundary condition for untagged exterior faces (default: PEC)",
    )
    parser.add_argument(
        "--exclude-volume",
        action="append",
        default=None,
        metavar="VOL_NAME",
        help="Exclude 3D Volume Physical Group (e.g. Sphere_PEC) and mark exposed faces as PEC",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=4,
        help="Polynomial order N written to .rea header (default: 4)",
    )
    parser.add_argument(
        "--pml-order",
        type=float,
        default=3.0,
        help="UPML conductivity polynomial order m (default: 3.0)",
    )
    parser.add_argument(
        "--pml-reflect-err",
        type=float,
        default=1.0e-6,
        help="UPML target normal reflection coefficient R(0) (default: 1.0e-6)",
    )

    args = parser.parse_args()
    msh_input = args.input_flag or args.input_pos
    if not msh_input:
        parser.error("Please provide an input .msh file via --input <path/to/mesh.msh>.")

    custom_map: Dict[str, str] = {}
    for item in args.bc:
        if "=" not in item:
            parser.error(f"Invalid --bc mapping '{item}'. Expected format: PhysicalName=TAG")
        k, v = item.split("=", 1)
        custom_map[k.strip()] = v.strip().upper()

    excl = set(args.exclude_volume) if args.exclude_volume else None

    convert_gmsh_to_nek(
        msh_path=msh_input,
        output_prefix=args.output,
        output_format=args.format,
        custom_bc_map=custom_map,
        default_bc=args.default_bc,
        exclude_volumes=excl,
        order=args.order,
        pml_order=args.pml_order,
        pml_reflect_err=args.pml_reflect_err,
        verbose=True,
    )


if __name__ == "__main__":
    main()
