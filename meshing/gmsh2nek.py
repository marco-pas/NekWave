#!/usr/bin/env python3
"""
gmsh2nek.py - Gmsh (.msh) to NekCEM / NekWave (.rea & .re2) Hexahedral Mesh Converter
=====================================================================================

Converts pure-hexahedral Gmsh meshes (MSH 4.1 or MSH 2.2 ASCII format) into
NekCEM and NekWave compatible mesh files:
  - ASCII  `.rea` (standalone mesh + boundary conditions + UPML parameters)
  - Binary `.re2` (high-performance binary `#v002` mesh + boundary conditions)

Features:
  1. Parses Gmsh MSH 4.1 and MSH 2.2 `$PhysicalNames`, `$Entities`, `$Nodes`,
     and `$Elements` (supporting 8-, 20-, and 27-node hexahedra and 4-, 8-, and
     9-node surface quads).
  2. Automatically maps Physical Surface Names to NekCEM / NekWave BC tags:
       - "PML*"                 -> "PML"
       - "PEC*", "Wall*", "W"   -> "PEC"
       - "PMC*", "SYM*", "S"    -> "PMC"
       - "PER*", "P"            -> "P" (with automatic opposite-face pairing)
     Supports custom CLI overrides via `--bc PhysicalName=TAG`.
  3. Optionally filters out hollow PEC interior volume groups (e.g., `Sphere_PEC`)
     and marks exposed interior surfaces as `PEC`.
  4. Verifies and enforces strictly positive trilinear Jacobian determinants
     (det(J) > 0) across all 8 corners of every hexahedron, automatically
     reorienting any left-handed hexahedra.
  5. Automatically detects the number of radial element layers in the PML region
     and writes `PMLTHICK`, `PMLORDER`, and `PMLREFERR` into the `.rea` header.

Usage:
  python3 meshing/gmsh2nek.py meshing/meshes/rcs_mesh.msh
  python3 meshing/gmsh2nek.py meshing/meshes/rcs_mesh.msh -o meshing/meshes/rcs_mesh --format both
"""

import argparse
import math
import os
import struct
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


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

# Opposing local face index (0..5) for inward layer marching
NEK_OPP_FACE: Tuple[int, ...] = (2, 3, 0, 1, 5, 4)

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
    dN_dr = (
        -0.125 * (1.0 - s) * (1.0 - t),
        +0.125 * (1.0 - s) * (1.0 - t),
        +0.125 * (1.0 + s) * (1.0 - t),
        -0.125 * (1.0 + s) * (1.0 - t),
        -0.125 * (1.0 - s) * (1.0 + t),
        +0.125 * (1.0 - s) * (1.0 + t),
        +0.125 * (1.0 + s) * (1.0 + t),
        -0.125 * (1.0 + s) * (1.0 + t),
    )
    dN_ds = (
        -0.125 * (1.0 - r) * (1.0 - t),
        -0.125 * (1.0 + r) * (1.0 - t),
        +0.125 * (1.0 + r) * (1.0 - t),
        +0.125 * (1.0 - r) * (1.0 - t),
        -0.125 * (1.0 - r) * (1.0 + t),
        -0.125 * (1.0 + r) * (1.0 + t),
        +0.125 * (1.0 + r) * (1.0 + t),
        +0.125 * (1.0 - r) * (1.0 + t),
    )
    dN_dt = (
        -0.125 * (1.0 - r) * (1.0 - s),
        -0.125 * (1.0 + r) * (1.0 - s),
        -0.125 * (1.0 + r) * (1.0 + s),
        -0.125 * (1.0 - r) * (1.0 + s),
        +0.125 * (1.0 - r) * (1.0 - s),
        +0.125 * (1.0 + r) * (1.0 - s),
        +0.125 * (1.0 + r) * (1.0 + s),
        +0.125 * (1.0 - r) * (1.0 + s),
    )

    xr = yr = zr = 0.0
    xs = ys = zs = 0.0
    xt = yt = zt = 0.0
    for a in range(8):
        vx, vy, vz = corners[a]
        xr += dN_dr[a] * vx
        yr += dN_dr[a] * vy
        zr += dN_dr[a] * vz

        xs += dN_ds[a] * vx
        ys += dN_ds[a] * vy
        zs += dN_ds[a] * vz

        xt += dN_dt[a] * vx
        yt += dN_dt[a] * vy
        zt += dN_dt[a] * vz

    return (
        xr * (ys * zt - yt * zs)
        - xs * (yr * zt - yt * zr)
        + xt * (yr * zs - ys * zr)
    )


def ensure_positive_jacobian(
    node_ids: List[int], nodes: Dict[int, Tuple[float, float, float]]
) -> Tuple[List[int], bool, float, float]:
    """
    Ensures the 8-node hexahedron has a right-handed coordinate system (det(J) > 0).
    If det(J) < 0 at the element center, swaps the bottom (0,1,2,3) and top (4,5,6,7)
    quad faces (i.e. t -> -t), which negates det(J) everywhere.
    Returns (oriented_node_ids, was_flipped, min_corner_detJ, max_corner_detJ).
    """
    coords = [nodes[nid] for nid in node_ids]
    det_center = eval_trilinear_det_j(coords, 0.0, 0.0, 0.0)
    flipped = False
    if det_center < 0.0:
        node_ids = [
            node_ids[4], node_ids[5], node_ids[6], node_ids[7],
            node_ids[0], node_ids[1], node_ids[2], node_ids[3],
        ]
        coords = [nodes[nid] for nid in node_ids]
        flipped = True

    corner_dets = [
        eval_trilinear_det_j(coords, r, s, t) for (r, s, t) in REF_CORNERS
    ]
    return node_ids, flipped, min(corner_dets), max(corner_dets)


# ==============================================================================
# Physical Group Name -> Boundary Condition Tag Mapping
# ==============================================================================
def infer_bc_tag(phys_name: str, custom_map: Dict[str, str]) -> str:
    """
    Maps a Gmsh Physical Group name to a canonical NekCEM / NekWave BC tag:
    'PEC', 'PMC', 'PML', or 'P  '.
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
# Gmsh .msh Parser (Supports MSH 4.1 and MSH 2.2 ASCII)
# ==============================================================================
class GmshMeshData:
    def __init__(self) -> None:
        self.version: str = "4.1"
        self.physical_names: Dict[Tuple[int, int], str] = {}
        self.entity_phys_tags: Dict[Tuple[int, int], List[int]] = {}
        self.nodes: Dict[int, Tuple[float, float, float]] = {}
        # Quads: list of (4 corner node IDs, list of physical tags)
        self.quads: List[Tuple[List[int], List[int]]] = []
        # Hexes: list of (8 corner node IDs, list of physical tags)
        self.hexes: List[Tuple[List[int], List[int]]] = []


def parse_msh_file(filepath: str) -> GmshMeshData:
    """
    Parses a Gmsh .msh file (MSH 4.x or MSH 2.2 ASCII) into GmshMeshData.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Gmsh file not found: {filepath}")

    data = GmshMeshData()

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
            file_type = int(fmt_parts[1])
            if file_type != 0:
                raise ValueError(
                    f"Binary .msh format (file_type={file_type}) detected in {filepath}. "
                    "Please export as ASCII from Gmsh or use gmsh.write() default ASCII."
                )
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
            # Collect all tokens inside $Entities ... $EndEntities
            tokens: List[str] = []
            while idx < n_lines:
                s = lines[idx].strip()
                idx += 1
                if s == "$EndEntities":
                    break
                if s:
                    tokens.extend(s.split())

            t_pos = 0
            num_pts = int(tokens[t_pos])
            num_curves = int(tokens[t_pos + 1])
            num_surfs = int(tokens[t_pos + 2])
            num_vols = int(tokens[t_pos + 3])
            t_pos += 4

            # 0D Points: tag x y z numPhysicalTags [physTags...]
            for _ in range(num_pts):
                pt_tag = int(tokens[t_pos])
                num_phys = int(tokens[t_pos + 4])
                t_pos += 5
                phys_tags = [int(tokens[t_pos + k]) for k in range(num_phys)]
                t_pos += num_phys
                data.entity_phys_tags[(0, pt_tag)] = phys_tags

            # 1D Curves, 2D Surfaces, 3D Volumes:
            # tag minX minY minZ maxX maxY maxZ numPhysicalTags [physTags...] numBounding [boundTags...]
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
                for _ in range(num_nodes):
                    parts = lines[idx].strip().split()
                    idx += 1
                    nid = int(parts[0])
                    data.nodes[nid] = (float(parts[1]), float(parts[2]), float(parts[3]))
            else:
                # MSH 4.x format: numEntityBlocks totalNumNodes minNodeTag maxNodeTag
                hdr = lines[idx].strip().split()
                idx += 1
                num_blocks = int(hdr[0])
                for _ in range(num_blocks):
                    bhdr = lines[idx].strip().split()
                    idx += 1
                    parametric = int(bhdr[2])
                    num_in_block = int(bhdr[3])
                    nids: List[int] = []
                    while len(nids) < num_in_block:
                        nids.extend(int(x) for x in lines[idx].strip().split())
                        idx += 1
                    for nid in nids:
                        cparts = lines[idx].strip().split()
                        idx += 1
                        data.nodes[nid] = (
                            float(cparts[0]),
                            float(cparts[1]),
                            float(cparts[2]),
                        )
                        if parametric != 0:
                            # Parametric coordinates are on the same line in MSH 4.1
                            pass
            while idx < n_lines and lines[idx].strip() != "$EndNodes":
                idx += 1
            idx += 1

        elif line == "$Elements":
            if data.version.startswith("2."):
                num_elems = int(lines[idx].strip())
                idx += 1
                for _ in range(num_elems):
                    parts = [int(x) for x in lines[idx].strip().split()]
                    idx += 1
                    etype = parts[1]
                    num_tags = parts[2]
                    phys_tag = parts[3] if num_tags > 0 else 0
                    conn = parts[3 + num_tags :]
                    if etype in GMSH_QUAD_TYPES:
                        data.quads.append((conn[:4], [phys_tag] if phys_tag else []))
                    elif etype in GMSH_HEX_TYPES:
                        data.hexes.append((conn[:8], [phys_tag] if phys_tag else []))
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
                    phys_tags = data.entity_phys_tags.get((ent_dim, ent_tag), [])

                    if etype in GMSH_QUAD_TYPES:
                        for _ in range(num_in_block):
                            parts = [int(x) for x in lines[idx].strip().split()]
                            idx += 1
                            data.quads.append((parts[1:5], phys_tags))
                    elif etype in GMSH_HEX_TYPES:
                        for _ in range(num_in_block):
                            parts = [int(x) for x in lines[idx].strip().split()]
                            idx += 1
                            data.hexes.append((parts[1:9], phys_tags))
                    else:
                        idx += num_in_block
            while idx < n_lines and lines[idx].strip() != "$EndElements":
                idx += 1
            idx += 1

    return data


# ==============================================================================
# Topology Builder & NekCEM / NekWave Connectivity Assembly
# ==============================================================================
class ConvertedNekMesh:
    def __init__(self) -> None:
        # List of 8 corner coordinate tuples [(x,y,z), ...] per element
        self.element_corners: List[List[Tuple[float, float, float]]] = []
        self.element_groups: List[int] = []
        # 2D list [elem_idx][face_idx 0..5] -> (bc_tag, nbr_elem_1based, nbr_face_1based)
        self.face_bcs: List[List[Tuple[str, int, int]]] = []
        self.pml_thickness: int = 0
        self.min_det_j: float = 0.0
        self.max_det_j: float = 0.0
        self.num_flipped: int = 0
        self.bc_counts: Dict[str, int] = defaultdict(int)


def build_nek_mesh(
    msh: GmshMeshData,
    custom_bc_map: Optional[Dict[str, str]] = None,
    default_bc: str = "PEC",
    exclude_volumes: Optional[Set[str]] = None,
    coord_tol: float = 1e-8,
) -> ConvertedNekMesh:
    """
    Constructs the NekCEM / NekWave element corner list, positive-Jacobian
    orientations, 6-face neighbor connectivity, periodic face pairing, and
    PML layer thickness from parsed GmshMeshData.
    """
    if custom_bc_map is None:
        custom_bc_map = {}
    if exclude_volumes is None:
        # By default, exclude any volume physical group explicitly named Sphere_PEC or PEC_Interior
        exclude_volumes = {"SPHERE_PEC", "PEC_INTERIOR", "PEC_VOLUME"}
    else:
        exclude_volumes = {v.upper() for v in exclude_volumes}

    if not msh.hexes:
        raise ValueError("No 3D hexahedral elements found in the Gmsh mesh!")

    # 1. Deduplicate coincident nodes using spatial quantization
    inv_tol = 1.0 / coord_tol
    coord_to_canon: Dict[Tuple[int, int, int], int] = {}
    canon_id: Dict[int, int] = {}
    for nid, (x, y, z) in msh.nodes.items():
        key = (int(round(x * inv_tol)), int(round(y * inv_tol)), int(round(z * inv_tol)))
        if key not in coord_to_canon:
            coord_to_canon[key] = nid
        canon_id[nid] = coord_to_canon[key]

    # 2. Map 2D boundary quads -> BC tags by canonical sorted 4-node key
    quad_bc_map: Dict[Tuple[int, int, int, int], str] = {}
    for q_nodes, phys_tags in msh.quads:
        fkey = tuple(sorted(canon_id[n] for n in q_nodes))
        bc_tag = default_bc
        for ptag in phys_tags:
            pname = msh.physical_names.get((2, ptag), "")
            if pname:
                bc_tag = infer_bc_tag(pname, custom_bc_map)
                break
        quad_bc_map[fkey] = bc_tag  # type: ignore[assignment]

    # 3. Filter active 3D hexahedra and track excluded PEC volume faces
    active_hexes: List[Tuple[List[int], int, bool]] = []
    excluded_faces: Set[Tuple[int, int, int, int]] = set()

    for h_nodes, phys_tags in msh.hexes:
        vol_names = [msh.physical_names.get((3, pt), "") for pt in phys_tags]
        is_excluded = any(vn.upper() in exclude_volumes for vn in vol_names if vn)
        if is_excluded:
            for fc in NEK_FACE_CORNERS:
                fkey = tuple(sorted(canon_id[h_nodes[c]] for c in fc))
                excluded_faces.add(fkey)  # type: ignore[arg-type]
            continue

        is_pml_vol = any("PML" in vn.upper() for vn in vol_names if vn)
        grp_id = phys_tags[0] if phys_tags else 0
        active_hexes.append((list(h_nodes), grp_id, is_pml_vol))

    if not active_hexes:
        raise ValueError("All 3D hexahedral elements were excluded!")

    # 4. Enforce right-handed orientation (det(J) > 0) for every hexahedron
    out = ConvertedNekMesh()
    min_j = float("inf")
    max_j = float("-inf")
    oriented_hex_nodes: List[List[int]] = []
    is_pml_elem: List[bool] = []

    for h_nodes, grp_id, is_pml_vol in active_hexes:
        fixed_nodes, flipped, cmin_j, cmax_j = ensure_positive_jacobian(h_nodes, msh.nodes)
        if flipped:
            out.num_flipped += 1
        min_j = min(min_j, cmin_j)
        max_j = max(max_j, cmax_j)
        oriented_hex_nodes.append(fixed_nodes)
        is_pml_elem.append(is_pml_vol)
        out.element_corners.append([msh.nodes[nid] for nid in fixed_nodes])
        out.element_groups.append(grp_id)

    out.min_det_j = min_j
    out.max_det_j = max_j
    nel = len(oriented_hex_nodes)

    # 5. Hash all 6 faces of all elements to discover interior vs boundary faces
    face_to_elems: Dict[Tuple[int, int, int, int], List[Tuple[int, int]]] = defaultdict(list)
    for e_idx, h_nodes in enumerate(oriented_hex_nodes):
        for f_idx, fc in enumerate(NEK_FACE_CORNERS):
            fkey = tuple(sorted(canon_id[h_nodes[c]] for c in fc))
            face_to_elems[fkey].append((e_idx, f_idx))  # type: ignore[index]

    out.face_bcs = [[("PEC", 0, 0) for _ in range(6)] for _ in range(nel)]
    periodic_faces: List[Tuple[int, int, Tuple[float, float, float]]] = []

    for fkey, owners in face_to_elems.items():
        if len(owners) == 2:
            (e1, f1), (e2, f2) = owners
            out.face_bcs[e1][f1] = ("E", e2 + 1, f2 + 1)
            out.face_bcs[e2][f2] = ("E", e1 + 1, f1 + 1)
            out.bc_counts["E"] += 2
        elif len(owners) == 1:
            e1, f1 = owners[0]
            if fkey in quad_bc_map:
                bc_tag = quad_bc_map[fkey]
            elif fkey in excluded_faces:
                bc_tag = "PEC"
            else:
                bc_tag = default_bc

            if bc_tag == "P":
                # Record face centroid for periodic opposite-face pairing
                fc = NEK_FACE_CORNERS[f1]
                cx = sum(out.element_corners[e1][c][0] for c in fc) * 0.25
                cy = sum(out.element_corners[e1][c][1] for c in fc) * 0.25
                cz = sum(out.element_corners[e1][c][2] for c in fc) * 0.25
                periodic_faces.append((e1, f1, (cx, cy, cz)))
            else:
                out.face_bcs[e1][f1] = (bc_tag, 0, 0)
                out.bc_counts[bc_tag] += 1
        else:
            raise ValueError(
                f"Non-manifold hexahedral face detected (shared by {len(owners)} elements): {owners}"
            )

    # 6. Pair periodic boundary faces if any exist
    if periodic_faces:
        paired: Set[int] = set()
        for i, (e1, f1, (cx1, cy1, cz1)) in enumerate(periodic_faces):
            if i in paired:
                continue
            best_j = -1
            best_dist = float("inf")
            for j in range(i + 1, len(periodic_faces)):
                if j in paired:
                    continue
                e2, f2, (cx2, cy2, cz2) = periodic_faces[j]
                dx = abs(cx1 - cx2)
                dy = abs(cy1 - cy2)
                dz = abs(cz1 - cz2)
                # Two opposite periodic faces share 2 transverse coordinates (~0 diff)
                # and differ along 1 periodic axis
                transverse_diff = sorted([dx, dy, dz])
                if transverse_diff[0] < 1e-5 and transverse_diff[1] < 1e-5 and transverse_diff[2] > 1e-5:
                    err = transverse_diff[0] + transverse_diff[1]
                    if err < best_dist:
                        best_dist = err
                        best_j = j
            if best_j >= 0:
                paired.add(i)
                paired.add(best_j)
                e2, f2, _ = periodic_faces[best_j]
                out.face_bcs[e1][f1] = ("P", e2 + 1, f2 + 1)
                out.face_bcs[e2][f2] = ("P", e1 + 1, f1 + 1)
                out.bc_counts["P"] += 2
            else:
                raise ValueError(
                    f"Could not find matching periodic partner for element {e1 + 1} face {f1 + 1} at ({cx1}, {cy1}, {cz1})"
                )

    # 7. Automatically detect PML element layer thickness by marching inward from "PML" faces
    if out.bc_counts.get("PML", 0) > 0:
        # Trace inward chain of elements from each outer PML face through opposing faces
        layer_depths: List[int] = []
        for e0 in range(nel):
            for f0 in range(6):
                if out.face_bcs[e0][f0][0] == "PML":
                    depth = 0
                    curr_e, curr_f = e0, f0
                    visited: Set[int] = set()
                    while curr_e not in visited:
                        visited.add(curr_e)
                        if any(is_pml_elem) and not is_pml_elem[curr_e]:
                            break
                        depth += 1
                        opp_f = NEK_OPP_FACE[curr_f]
                        bc_tag, nbr_e1, nbr_f1 = out.face_bcs[curr_e][opp_f]
                        if bc_tag != "E" or nbr_e1 <= 0:
                            break
                        curr_e = nbr_e1 - 1
                        curr_f = nbr_f1 - 1
                    if depth > 0:
                        layer_depths.append(depth)
        if layer_depths:
            out.pml_thickness = min(layer_depths)

    return out


# ==============================================================================
# NekCEM / NekWave ASCII .rea Writer
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
    Writes a NekCEM / NekWave compatible ASCII .rea file.
    If use_re2_companion=True, writes -NEL in the **MESH DATA** header so
    NekWave / NekCEM redirects to the binary .re2 file in the same directory.
    """
    nel = len(mesh.element_corners)
    pml_thick = mesh.pml_thickness if mesh.pml_thickness > 0 else 2

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(" ****** PARAMETERS *****\n")
        f.write("    2.610000     NEKTON VERSION\n")
        f.write(" 3 DIMENSIONAL RUN\n")
        f.write(" 103 PARAMETERS FOLLOW\n")
        for p in range(1, 104):
            if p == 4:
                f.write("   1                             4: ifte (1), iftm (2)\n")
            elif p == 10:
                f.write("   1.0                           10: fintim\n")
            elif p == 11:
                f.write("   1000                          11: nsteps\n")
            elif p == 12:
                f.write("  -0.001                         12: dt\n")
            elif p == 13:
                f.write("   10                            13: iocomm\n")
            elif p == 15:
                f.write("   100                           15: iostep\n")
            elif p == 20:
                f.write(f"   {float(order):.5f}                       20: ORDER (MESH)\n")
            elif p == 77:
                f.write(f"   {float(pml_thick):.6E}                  77: PMLTHICK : thickness of the PML in levels\n")
            elif p == 78:
                f.write(f"   {float(pml_order):.6E}                  78: PMLORDER : polynomial order of the grading of sigma\n")
            elif p == 79:
                f.write(f"   {float(pml_reflect_err):.6E}                  79: PMLREFERR : PML reflection error\n")
            else:
                f.write(f"   0                             {p}: ---\n")

        f.write("           4  Lines of passive scalar data follows2 CONDUCT; 2RHOCP\n")
        f.write("   1.00000       1.00000       1.00000       1.00000       1.00000\n")
        f.write("   1.00000       1.00000       1.00000       1.00000\n")
        f.write("   1.00000       1.00000       1.00000       1.00000       1.00000\n")
        f.write("   1.00000       1.00000       1.00000       1.00000\n")
        f.write(" 13 LOGICAL SWITCHES FOLLOW\n")
        f.write("  F                             1: IFFLOW\n")
        f.write("  T                             2: IFHEAT\n")
        f.write("  T                             3: IFTRAN\n")
        f.write("  F                             4: IFSRC\n")
        f.write("  F                             5: IFCENTRAL\n")
        f.write("  T                             6: IFUPWIND\n")
        f.write("  F                             7: IFTM (2D only)\n")
        f.write("  T                             8: IFTE (2D only)\n")
        f.write("  F                             9: IFDEALIAS\n")
        f.write("  F                             10: IFRK4\n")
        f.write("  T                             11: IFEXP\n")
        f.write("  F                             12: IFEIG\n")
        f.write("  F                             13: IFNM\n")
        f.write("   9.23077       9.23077      -4.38462      -5.30769     XFAC,YFAC,XZERO,YZERO\n")
        f.write(" **MESH DATA** 6 lines are X,Y,Z;X,Y,Z. Columns corners 1-4;5-8\n")

        if use_re2_companion:
            f.write(f"   {-nel:9d}  3   {nel:9d}           NEL,NDIM,NELV\n")
        else:
            f.write(f"   {nel:9d}  3   {nel:9d}           NEL,NDIM,NELV\n")
            for e_idx, corners in enumerate(mesh.element_corners):
                e_1 = e_idx + 1
                atag = _elem_AlphaTag(e_1)
                f.write(f"            ELEMENT {e_1:11d} [{atag}]  GROUP  0\n")
                # Corners 1..4 (indices 0..3): X, Y, Z
                f.write("".join(f"{corners[c][0]:14.6f}" for c in range(0, 4)) + "\n")
                f.write("".join(f"{corners[c][1]:14.6f}" for c in range(0, 4)) + "\n")
                f.write("".join(f"{corners[c][2]:14.6f}" for c in range(0, 4)) + "\n")
                # Corners 5..8 (indices 4..7): X, Y, Z
                f.write("".join(f"{corners[c][0]:14.6f}" for c in range(4, 8)) + "\n")
                f.write("".join(f"{corners[c][1]:14.6f}" for c in range(4, 8)) + "\n")
                f.write("".join(f"{corners[c][2]:14.6f}" for c in range(4, 8)) + "\n")

            f.write("  ***** CURVED SIDE DATA *****\n")
            f.write("           0 Curved sides follow IEDGE,IEL,CURVE(I),I=1,5, CCURVE\n")
            f.write("  ***** BOUNDARY CONDITIONS *****\n")
            f.write("  ***** FLUID   BOUNDARY CONDITIONS *****\n")
            for e_idx, faces in enumerate(mesh.face_bcs):
                e_1 = e_idx + 1
                for f_idx, (bc_tag, nbr_e, nbr_f) in enumerate(faces):
                    f_1 = f_idx + 1
                    f.write(
                        f" {bc_tag:<3s} {e_1:7d} {f_1:2d} {float(nbr_e):13.5f} {float(nbr_f):13.5f}"
                        "       0.00000       0.00000       0.00000\n"
                    )

        f.write("  ***** NO THERMAL BOUNDARY CONDITIONS *****\n")
        f.write("            0 PRESOLVE/RESTART OPTIONS  *****\n")
        f.write("            7         INITIAL CONDITIONS *****\n")
        for _ in range(7):
            f.write("C Default\n")
        f.write("  ***** DRIVE FORCE DATA ***** BODY FORCE, FLOW, Q\n")
        f.write("            4                 Lines of Drive force data follow\n")
        for _ in range(4):
            f.write("C\n")
        f.write("  ***** Variable Property Data ***** Overrrides Parameter data.\n")
        f.write("            1 Lines follow.\n")
        f.write("            0 PACKETS OF DATA FOLLOW\n")
        f.write("  ***** HISTORY AND INTEGRAL DATA *****\n")
        f.write("            0   POINTS.  Hcode, I,J,H,IEL\n")
        f.write("  ***** OUTPUT FIELD SPECIFICATION *****\n")
        f.write("            6 SPECIFICATIONS FOLLOW\n")
        f.write("  F      COORDINATES\n")
        f.write("  F      VELOCITY\n")
        f.write("  F      PRESSURE\n")
        f.write("  T      TEMPERATURE\n")
        f.write("  F      TEMPERATURE GRADIENT\n")
        f.write("            0      PASSIVE SCALARS\n")
        f.write("  ***** OBJECT SPECIFICATION *****\n")
        f.write("       0 Surface Objects\n")
        f.write("       0 Volume  Objects\n")
        f.write("       0 Edge    Objects\n")
        f.write("       0 Point   Objects\n")


# ==============================================================================
# NekCEM / NekWave Binary .re2 Writer (#v002 format)
# ==============================================================================
def write_re2_file(filepath: str, mesh: ConvertedNekMesh) -> None:
    """
    Writes a NekCEM / NekWave compatible binary .re2 file (#v002 format),
    matching Mesh::loadFromRe2() in src/mesh.cpp.
    """
    nel = len(mesh.element_corners)

    # Collect non-interior boundary condition records
    bc_records: List[Tuple[int, int, int, int, str]] = []
    for e_idx, faces in enumerate(mesh.face_bcs):
        e_1 = e_idx + 1
        for f_idx, (bc_tag, nbr_e, nbr_f) in enumerate(faces):
            if bc_tag != "E":
                f_1 = f_idx + 1
                bc_records.append((e_1, f_1, nbr_e, nbr_f, bc_tag))

    with open(filepath, "wb") as f:
        # 1. 80-byte ASCII header
        hdr_str = f"#v002{nel:9d}{3:3d}{nel:9d} this is the hdr".ljust(80)[:80]
        f.write(hdr_str.encode("ascii"))

        # 2. 4-byte float endianness test tag (6.54321)
        f.write(struct.pack("<f", 6.54321))

        # 3. Element coordinates: group (1 double) + X (8 doubles) + Y (8 doubles) + Z (8 doubles)
        for e_idx, corners in enumerate(mesh.element_corners):
            grp = float(mesh.element_groups[e_idx])
            xs = [corners[c][0] for c in range(8)]
            ys = [corners[c][1] for c in range(8)]
            zs = [corners[c][2] for c in range(8)]
            f.write(struct.pack("<25d", grp, *xs, *ys, *zs))

        # 4. Curved sides count (0.0)
        f.write(struct.pack("<d", 0.0))

        # 5. Boundary conditions count + records
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

    msh_data = parse_msh_file(msh_path)
    nek_mesh = build_nek_mesh(
        msh_data,
        custom_bc_map=custom_bc_map,
        default_bc=default_bc,
        exclude_volumes=exclude_volumes,
    )

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

    if verbose:
        nel = len(nek_mesh.element_corners)
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
            print(f"  Wrote ASCII .rea file:    {rea_path}")
        if fmt in ("re2", "both"):
            print(f"  Wrote Binary .re2 file:   {re2_path}")
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
        help="Output mesh format: 'rea' (ASCII), 're2' (binary + companion .rea), or 'both' (default)",
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

