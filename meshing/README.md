# NekWave Meshing & Gmsh-to-NekCEM Converter (`gmsh2nek.py`)

This directory contains the mesh generation scripts and the standalone converter [`gmsh2nek.py`](file:///users/pasquale/develop/NekWave/meshing/gmsh2nek.py) that translates pure-hexahedral Gmsh (`.msh`) meshes into NekCEM / NekWave compatible `.rea` (ASCII) and `.re2` (binary `#v002`) formats.

---

## Directory Structure

```text
meshing/
├── gmsh2nek.py              # Standalone Gmsh (.msh) -> NekCEM/NekWave (.rea & .re2) converter
├── README.md                # This guide
└── sphereRCS/               # PEC Sphere in Vacuum with UPML Boundaries case
    ├── spherePEC.py         # Gmsh multi-block cubed-sphere hexahedral mesh generator
    ├── spherePEC.msh        # Generated Gmsh mesh (created by spherePEC.py)
    ├── spherePEC.rea        # Converted ASCII NekCEM/NekWave mesh (created by gmsh2nek.py)
    └── spherePEC.re2        # Converted binary NekCEM/NekWave mesh (created by gmsh2nek.py)
```

---

## Quick Start Workflow

### 1. Generate the Gmsh Mesh (`.msh`)
Run the Gmsh Python generator inside [`sphereRCS/`](file:///users/pasquale/develop/NekWave/meshing/sphereRCS/spherePEC.py) (pass `-nopopup` to skip the interactive Gmsh GUI):

```bash
cd meshing
python3 sphereRCS/spherePEC.py -nopopup
```

This creates `sphereRCS/spherePEC.msh` with a 100% hexahedral cubed-sphere O-grid mesh around a hollow PEC sphere surrounded by an outer UPML box.

### 2. Convert `.msh` to `.rea` and `.re2`
From the `meshing/` directory, run [`gmsh2nek.py`](file:///users/pasquale/develop/NekWave/meshing/gmsh2nek.py) with `--input`:

```bash
python3 gmsh2nek.py --input sphereRCS/spherePEC.msh
```

This automatically generates both:
- `sphereRCS/spherePEC.rea` (ASCII NekCEM / NekWave format)
- `sphereRCS/spherePEC.re2` (Binary `#v002` NekCEM / NekWave format)

---

## How `gmsh2nek.py` Works

1. **Physical Group Boundary Mapping**:
   `gmsh2nek.py` reads 2D surface Physical Groups from the `.msh` file and maps their names to NekCEM / NekWave boundary condition tags:
   - Names containing `PML` (e.g. `PML_Boundaries`) $\to$ `PML`
   - Names containing `PEC`, `WALL`, `CONDUCT`, or `METAL` (e.g. `PEC_Boundary`) $\to$ `PEC`
   - Names containing `PMC`, `SYM`, or `MAGNET` $\to$ `PMC`
   - Names containing `PER` $\to$ `P` (automatically pairs opposite periodic faces)
   - Custom overrides can be passed via `--bc SurfaceName=TAG` (e.g. `--bc OuterWall=PML`).

2. **Positive Jacobian Orientation ($\det J > 0$)**:
   Evaluates the 3D trilinear Jacobian determinant at all 8 corners of every hexahedron and automatically reorients any left-handed elements so that $\det J > 0$ everywhere.

3. **Automatic UPML Layer Thickness Detection**:
   Marches inward from outer `PML` boundary faces through opposing hexahedral faces to detect the number of radial PML element layers (`PMLTHICK`) and writes it into parameter `77` of the `.rea` file.

---

## Loading the Converted Mesh in NekWave (`.json`)

Point your case `.json` file to either the `.rea` or `.re2` file:

```json
{
    "mesh": {
        "type": "nekcem",
        "file": "meshing/sphereRCS/spherePEC.re2",
        "pml_thickness": 5,
        "pml_order": 3.0,
        "pml_reflect_err": 1.0e-6
    }
}
```
