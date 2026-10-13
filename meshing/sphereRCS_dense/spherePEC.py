#!/usr/bin/env python3
import os
import sys
import math
import gmsh

# ==============================================================================
# NekWave / NekCEM Style 100% Pure Hexahedral Mesh Generator for a PEC Sphere
# ==============================================================================
#
# Multi-Block Cubed-Sphere O-Grid Topology:
#   - Decomposes the sphere surface (r = r_sphere) into 6 non-periodic, 4-sided
#     spherical patches (`addSurfaceFilling` with `sphereCenterTag`).
#   - Connects the 6 spherical patches radially outward to the 6 faces of the
#     inner vacuum box [-L_vac/2, L_vac/2]^3 using 6 structured hexahedral blocks.
#   - Surrounds the vacuum box with 6 concentric hexahedral PML blocks extending
#     to [-box_L/2, box_L/2]^3 (matching NekCEM's absorbing PML box layer).
#   - Leaves the PEC sphere interior hollow (since E = H = 0 inside a PEC),
#     avoiding tiny core elements that would restrict the explicit CFL time step.
# ==============================================================================

gmsh.initialize()
gmsh.model.add("RCS_Hex_Mesh")

# --- 1. Geometric & Meshing Parameters ---
box_L = 4.0                  # Outer side length of the full domain including PML (m)
pml_thickness = 0.25         # Thickness of the outer PML absorbing layer (m) (set 0.0 for surface-only PML)
r_sphere = 0.3               # Radius of the PEC sphere (m)
mesh_size = 0.05            # Target characteristic element size (m)

# Hollow PEC Cavity (NekCEM standard):
include_sphere_interior = False

# Derived transfinite subdivision counts (number of hex elements along each direction)
vac_L = box_L - 2.0 * pml_thickness if pml_thickness > 0.0 else box_L
n_tangential = max(4, int(round((r_sphere * math.pi / 2.0) / mesh_size)))
n_radial_vac = max(4, int(round((0.5 * vac_L - r_sphere) / mesh_size)))
n_radial_pml = max(2, int(round(pml_thickness / mesh_size))) if pml_thickness > 0.0 else 0
n_radial_int = max(3, int(round((0.55 * r_sphere) / mesh_size))) if include_sphere_interior else 0
radial_grading = 1.08        # Geometric progression (>1 refines near the PEC sphere surface)

# --- 2. Helper Functions for Cubed-Sphere / Concentric Box Topology ---
# 8 canonical corner signs in (-/+ x, -/+ y, -/+ z)
CORNER_SIGNS = [
    (-1.0, -1.0, -1.0),  # 0
    (+1.0, -1.0, -1.0),  # 1
    (+1.0, +1.0, -1.0),  # 2
    (-1.0, +1.0, -1.0),  # 3
    (-1.0, -1.0, +1.0),  # 4
    (+1.0, -1.0, +1.0),  # 5
    (+1.0, +1.0, +1.0),  # 6
    (-1.0, +1.0, +1.0),  # 7
]

# 12 canonical undirected edges (u < v)
CANONICAL_EDGES = [
    (0, 1), (1, 2), (2, 3), (0, 3),  # z = -d
    (4, 5), (5, 6), (6, 7), (4, 7),  # z = +d
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
]

# 6 canonical quad faces (right-handed outward 4-cycles of corner indices)
FACE_CORNERS = [
    (0, 3, 2, 1),  # 0: -Z face
    (4, 5, 6, 7),  # 1: +Z face
    (0, 1, 5, 4),  # 2: -Y face
    (2, 3, 7, 6),  # 3: +Y face
    (3, 0, 4, 7),  # 4: -X face
    (1, 2, 6, 5),  # 5: +X face
]

center_pt = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)


def create_shell(half_coord, is_spherical, n_edge):
    """
    Creates an 8-corner, 12-edge, 6-face concentric shell in gmsh.model.geo.
    If is_spherical=True, edges are great-circle arcs and faces are spherical patches
    projected onto the sphere of radius r = half_coord * sqrt(3).
    If is_spherical=False, edges are straight lines and faces are flat box planes.
    """
    pts = []
    for sx, sy, sz in CORNER_SIGNS:
        pts.append(gmsh.model.geo.addPoint(sx * half_coord, sy * half_coord, sz * half_coord))

    edges = {}
    for u, v in CANONICAL_EDGES:
        if is_spherical:
            c = gmsh.model.geo.addCircleArc(pts[u], center_pt, pts[v])
        else:
            c = gmsh.model.geo.addLine(pts[u], pts[v])
        gmsh.model.geo.mesh.setTransfiniteCurve(c, n_edge + 1)
        edges[(u, v)] = c
        edges[(v, u)] = -c

    surfs = []
    for c0, c1, c2, c3 in FACE_CORNERS:
        loop = gmsh.model.geo.addCurveLoop([
            edges[(c0, c1)],
            edges[(c1, c2)],
            edges[(c2, c3)],
            edges[(c3, c0)],
        ])
        if is_spherical:
            s = gmsh.model.geo.addSurfaceFilling([loop], sphereCenterTag=center_pt)
        else:
            s = gmsh.model.geo.addPlaneSurface([loop])
        gmsh.model.geo.mesh.setTransfiniteSurface(s, cornerTags=[pts[c0], pts[c1], pts[c2], pts[c3]])
        gmsh.model.geo.mesh.setRecombine(2, s)
        surfs.append(s)

    return {"pts": pts, "edges": edges, "surfs": surfs}


def connect_shells(shell_in, shell_out, n_radial, progression=1.0):
    """
    Connects two concentric 6-face shells with 8 radial lines, 12 radial surfaces,
    and 6 transfinite hexahedral volume blocks.
    """
    rad_lines = []
    for c in range(8):
        l_tag = gmsh.model.geo.addLine(shell_in["pts"][c], shell_out["pts"][c])
        gmsh.model.geo.mesh.setTransfiniteCurve(l_tag, n_radial + 1, "Progression", progression)
        rad_lines.append(l_tag)

    rad_surfs = {}
    for u, v in CANONICAL_EDGES:
        p0 = shell_in["pts"][u]
        p1 = shell_in["pts"][v]
        p2 = shell_out["pts"][v]
        p3 = shell_out["pts"][u]
        loop = gmsh.model.geo.addCurveLoop([
            shell_in["edges"][(u, v)],
            rad_lines[v],
            shell_out["edges"][(v, u)],
            -rad_lines[u],
        ])
        s_rad = gmsh.model.geo.addSurfaceFilling([loop])
        gmsh.model.geo.mesh.setTransfiniteSurface(s_rad, cornerTags=[p0, p1, p2, p3])
        gmsh.model.geo.mesh.setRecombine(2, s_rad)
        rad_surfs[(u, v)] = s_rad
        rad_surfs[(v, u)] = s_rad

    vols = []
    for f, (c0, c1, c2, c3) in enumerate(FACE_CORNERS):
        sl = gmsh.model.geo.addSurfaceLoop([
            shell_in["surfs"][f],
            shell_out["surfs"][f],
            rad_surfs[(c0, c1)],
            rad_surfs[(c1, c2)],
            rad_surfs[(c2, c3)],
            rad_surfs[(c3, c0)],
        ])
        v = gmsh.model.geo.addVolume([sl])
        gmsh.model.geo.mesh.setTransfiniteVolume(
            v,
            [
                shell_in["pts"][c0], shell_in["pts"][c1], shell_in["pts"][c2], shell_in["pts"][c3],
                shell_out["pts"][c0], shell_out["pts"][c1], shell_out["pts"][c2], shell_out["pts"][c3],
            ],
        )
        gmsh.model.geo.mesh.setRecombine(3, v)
        vols.append(v)

    return vols


# --- 3. Build Concentric Multi-Block Shells ---
# 3a. PEC Sphere Surface Shell (at r = r_sphere => cube vertex coord = r_sphere / sqrt(3))
a_sphere = r_sphere / math.sqrt(3.0)
sphere_shell = create_shell(a_sphere, is_spherical=True, n_edge=n_tangential)

# 3b. Inner Sphere Volume (Optional 7-block pure-hex O-grid inside the PEC sphere)
sphere_vols = []
if include_sphere_interior:
    a_core = 0.45 * a_sphere
    core_shell = create_shell(a_core, is_spherical=False, n_edge=n_tangential)
    sl_core = gmsh.model.geo.addSurfaceLoop(core_shell["surfs"])
    v_core = gmsh.model.geo.addVolume([sl_core])
    gmsh.model.geo.mesh.setTransfiniteVolume(
        v_core,
        [
            core_shell["pts"][0], core_shell["pts"][1], core_shell["pts"][2], core_shell["pts"][3],
            core_shell["pts"][4], core_shell["pts"][5], core_shell["pts"][6], core_shell["pts"][7],
        ],
    )
    gmsh.model.geo.mesh.setRecombine(3, v_core)
    inner_pillow_vols = connect_shells(core_shell, sphere_shell, n_radial=n_radial_int, progression=1.0)
    sphere_vols = [v_core] + inner_pillow_vols

# 3c. Vacuum Region (6 hexahedral blocks from PEC sphere r = r_sphere to [-vac_L/2, vac_L/2]^3)
vac_shell = create_shell(0.5 * vac_L, is_spherical=False, n_edge=n_tangential)
air_vols = connect_shells(sphere_shell, vac_shell, n_radial=n_radial_vac, progression=radial_grading)

# 3d. Outer PML Absorbing Layer (6 hexahedral blocks from [-vac_L/2, vac_L/2]^3 to [-box_L/2, box_L/2]^3)
pml_vols = []
if pml_thickness > 0.0:
    pml_shell = create_shell(0.5 * box_L, is_spherical=False, n_edge=n_tangential)
    pml_vols = connect_shells(vac_shell, pml_shell, n_radial=n_radial_pml, progression=1.0)
    outer_pml_surfs = pml_shell["surfs"]
else:
    outer_pml_surfs = vac_shell["surfs"]

gmsh.model.geo.synchronize()

# --- 4. Assign Physical Groups (For the Solver) ---
# Tag Volumes
gmsh.model.addPhysicalGroup(3, air_vols, name="Air_Vacuum")
if pml_vols:
    gmsh.model.addPhysicalGroup(3, pml_vols, name="PML_Layer")
if sphere_vols:
    gmsh.model.addPhysicalGroup(3, sphere_vols, name="Sphere_PEC")

# Tag Surfaces
gmsh.model.addPhysicalGroup(2, outer_pml_surfs, name="PML_Boundaries")
gmsh.model.addPhysicalGroup(2, sphere_shell["surfs"], name="PEC_Boundary")

# --- 5. Generate 100% Hexahedral 3D Mesh ---
gmsh.model.mesh.generate(3)

# Verify element types (must be 100% hexahedra in 3D)
elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim=3)
total_hex = sum(len(tags) for etype, tags in zip(elem_types, elem_tags) if etype in (5, 12, 92, 93))
total_3d = sum(len(tags) for tags in elem_tags)
print("==========================================================")
print(f"  Generated 3D Mesh Summary: {total_hex} / {total_3d} Hexahedral Elements")
print(f"  - Tangential subdivisions per patch edge: {n_tangential}")
print(f"  - Radial subdivisions in Vacuum:          {n_radial_vac}")
if pml_vols:
    print(f"  - Radial subdivisions in PML Layer:       {n_radial_pml}")
if sphere_vols:
    print(f"  - Sphere PEC interior hex blocks:         7 ({n_radial_int} radial layers)")
print("==========================================================")

# --- 6. Save the Gmsh .msh File ---
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "spherePEC.msh")
gmsh.write(output_path)
print(f"Saved pure hexahedral Gmsh mesh to: {output_path}")

# Launch the GUI only when an interactive X11 display is available and -nopopup is not set
if "-nopopup" not in sys.argv and os.environ.get("DISPLAY"):
    gmsh.fltk.run()

gmsh.finalize()
