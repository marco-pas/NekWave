#ifndef NW_SRC_BOUNDARY_CONDITIONS_HPP
#define NW_SRC_BOUNDARY_CONDITIONS_HPP

#include <array>
#include <cmath>
#include <string>
#include <vector>

class Mesh;

/**
 * @brief Strongly-typed enumeration of all supported boundary and interface conditions.
 *
 * Shared between host preprocessing (Mesh, BoundaryConditions, DgSolver) and
 * device surface flux kernels (gpu_compute_flux_kernel).
 */
enum class BcType : int {
    INTERIOR = 0, ///< Internal conforming element-to-element interface ("E")
    PEC      = 1, ///< Perfect Electric Conductor: n x E = 0 => [[E]] = -2 E^-, [[H]] = 0
    PMC      = 2, ///< Perfect Magnetic Conductor: n x H = 0 => [[E]] = 0, [[H]] = -2 H^-
    PML      = 3, ///< Perfectly Matched Layer outer wall (PEC mirror termination + volume UPML)
    PERIODIC = 4, ///< Periodic domain wrap-around interface ("P", "PERIODIC")
    MPI_CUT  = 5  ///< Inter-rank MPI halo partition boundary ("MPI")
};

/**
 * @brief Analytical incident plane-wave descriptor for Scattered-Field formulation (RCS).
 *
 * Represents a broadband Gaussian-modulated plane wave propagating in direction kHat
 * with electric polarization eHat and magnetic polarization hHat = kHat x eHat:
 *   s(r, t) = kHat . r - x0 - c * t   (with c = 1)
 *   A(r, t) = E0 * cos(k0 * s) * exp(-s^2 / (2 * sigmaX^2))
 *   E_inc(r, t) = A(r, t) * eHat
 *   H_inc(r, t) = A(r, t) * hHat
 */
struct IncidentPlaneWaveConfig {
    bool enabled = false;
    double E0 = 1.0;            ///< Peak electric field amplitude
    double k0 = 20.9439510239;  ///< Carrier wavenumber 2*pi / lambda0 (rad/m)
    double x0 = -0.90;          ///< Initial pulse center along kHat (m)
    double sigmaX = 0.14;       ///< Longitudinal Gaussian pulse width (m)
    double sphereRadius = 0.30; ///< Scatterer PEC sphere radius for interior null-field masking (m)

    double kx = 1.0, ky = 0.0, kz = 0.0;  ///< Unit propagation vector kHat
    double ex = 0.0, ey = 0.0, ez = 1.0;  ///< Unit electric polarization vector eHat
    double hx = 0.0, hy = -1.0, hz = 0.0; ///< Unit magnetic polarization vector hHat = kHat x eHat

    void updateMagneticPolarization() {
        hx = ky * ez - kz * ey;
        hy = kz * ex - kx * ez;
        hz = kx * ey - ky * ex;
    }

#if defined(__CUDACC__) || defined(__HIPCC__)
    __host__ __device__
#endif
    void evaluate(double x, double y, double z, double t,
                  double& Ex_inc, double& Ey_inc, double& Ez_inc,
                  double& Hx_inc, double& Hy_inc, double& Hz_inc) const {
        if (!enabled) {
            Ex_inc = 0.0; Ey_inc = 0.0; Ez_inc = 0.0;
            Hx_inc = 0.0; Hy_inc = 0.0; Hz_inc = 0.0;
            return;
        }
        double s = (kx * x + ky * y + kz * z) - x0 - t;
        double amp = E0 * std::cos(k0 * s) * std::exp(-(s * s) / (2.0 * sigmaX * sigmaX));
        Ex_inc = amp * ex;
        Ey_inc = amp * ey;
        Ez_inc = amp * ez;
        Hx_inc = amp * hx;
        Hy_inc = amp * hy;
        Hz_inc = amp * hz;
    }
};

/**
 * @brief Configuration parameters for NekCEM-style Uniaxial Perfectly Matched Layers (UPML).
 *
 * Matches NekCEM's /pmlparam/ common block in contrib/NekCEM/src/PML:
 *   - pmlthick   (thickness): Number of hexahedral element layers in the PML [1, 10]
 *   - pmlorder   (order):     Degree m of polynomial conductivity grading [1, 10]
 *   - pmlreferr  (reflectErr): Desired normal-incidence reflection error R(0) in (0, 1]
 */
struct PmlConfig {
    int thickness;      ///< Thickness of the PML in element layers (pmlthick)
    double order;       ///< Degree of the polynomial grading (pmlorder)
    double reflectErr;  ///< Permitted reflection error R(0) (pmlreferr)

    PmlConfig(int t = 2, double o = 3.0, double r = 1.0e-6)
        : thickness(t), order(o), reflectErr(r) {}
};

/**
 * @brief Host-side precomputed UPML topology and conductivity data.
 *
 * Mirrors NekCEM's /pml1/, /pml3/, and /pml4/ common blocks in contrib/NekCEM/src/PML:
 *   - pmlInner / pmlOuter: Inner and outer slab coordinates for (-X, +X, -Y, +Y, -Z, +Z)
 *   - pmlTag:Per-element 6-bit mask indicating membership in (-X, +X, -Y, +Y, -Z, +Z) PML slabs
 *   - pmlPtr: Compact list of 0-based element indices belonging to the PML region (size maxPml)
 *   - pmlSigma: Directional conductivity profile [sigma_x, sigma_y, sigma_z] at all GLL points
 *               stored in SoA layout of size 3 * totalPoints.
 */
struct PmlData {
    bool enabled;
    int maxPml;       ///< Local number of PML elements on this MPI rank
    int maxPmlGlobal; ///< Global number of PML elements across all MPI ranks

    std::array<double, 6> pmlInner;
    std::array<double, 6> pmlOuter;

    std::vector<int> pmlTag;      ///< Bitmask per element (size numElements)
    std::vector<int> pmlPtr;      ///< Compact list of PML element IDs (size maxPml)
    std::vector<double> pmlSigma; ///< Conductivity [sigX, sigY, sigZ] (size 3 * totalPoints)

    PmlData()
        : enabled(false),
          maxPml(0),
          maxPmlGlobal(0),
          pmlInner({{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}),
          pmlOuter({{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}) {}
};

/**
 * @brief Central coordinator for Boundary Conditions (PEC, PMC, PML, Periodic) in NekWave.
 *
 * Provides:
 *   1. Tag normalization and parsing for .rea, .re2, and JSON configurations.
 *   2. Face-point BcType classification for GPU surface Riemann flux kernels.
 *   3. 1-to-1 port of NekCEM's time-domain Maxwell UPML setup (contrib/NekCEM/src/cem_maxwell_pml.F):
 *        - pml_errchk          -> validatePmlConfig()
 *        - pml_fill_faceary    -> pmlFillFaceArray()
 *        - march_faces         -> marchFaces()
 *        - dir_local_to_global -> dirLocalToGlobal()
 *        - pml_extent_and_tags -> pmlExtentAndTags()
 *        - pml_calc_sigma      -> pmlCalcSigma()
 */
class BoundaryConditions {
public:
    /**
     * @brief Parses a raw boundary condition string tag into a canonical BcType enum.
     *
     * @param rawTag            Raw string from .rea, .re2, or JSON config.
     * @param neighborElementId Neighbor element index (>= 0 if connected, -1 if exterior).
     * @return Canonical BcType value.
     */
    static BcType parseBcTag(const std::string& rawTag, int neighborElementId = -1);

    /**
     * @brief Converts a BcType enum back to its canonical string label ("PEC", "PMC", "PML", "PERIODIC", "E", "MPI").
     */
    static std::string toString(BcType bc);

    /**
     * @brief Populates the flattened face-quadrature-point boundary condition array for GPU flux evaluation.
     *
     * @param mesh     Initialized Mesh object.
     * @param h_bcType Output vector of size totalFacePoints containing static_cast<int>(BcType).
     */
    static void buildFacePointBcTypes(const Mesh& mesh, std::vector<int>& h_bcType);

    /**
     * @brief Returns true if the mesh contains any boundary face tagged with BcType::PML.
     */
    static bool hasPmlFaces(const Mesh& mesh);

    /**
     * @brief Validates UPML configuration parameters (matching NekCEM's pml_errchk).
     */
    static bool validatePmlConfig(const PmlConfig& cfg);

    /**
     * @brief Runs NekCEM's full UPML topological face-marching and polynomial sigma grading pipeline.
     *
     * Ports pml_fill_faceary, march_faces, pml_extent_and_tags, and pml_calc_sigma
     * from contrib/NekCEM/src/cem_maxwell_pml.F.
     *
     * @param mesh Reference to the Mesh (prior to or after partitioning).
     * @param cfg  UPML grading parameters (thickness, order, reflectErr).
     * @return Populated PmlData structure.
     */
    static PmlData initializePml(const Mesh& mesh, const PmlConfig& cfg);

    /**
     * @brief Translates a local reference direction (0..5 in symmetric convention: -r,+r,-s,+s,-t,+t)
     *        of element `elt` to a global Cartesian direction (0..5: -X,+X,-Y,+Y,-Z,+Z).
     *
     * Matches dir_local_to_global in contrib/NekCEM/src/cem_maxwell_pml.F.
     */
    static int dirLocalToGlobal(const Mesh& mesh, int elt, int symDir);

private:
    static void pmlFillFaceArray(const Mesh& mesh, std::vector<double>& faceary, int thick);
    static void marchFaces(const Mesh& mesh, std::vector<double>& faceary);
    static void pmlExtentAndTags(const Mesh& mesh, const std::vector<double>& faceary, PmlData& pml);
    static void pmlCalcSigma(const Mesh& mesh, const PmlConfig& cfg, PmlData& pml);
};

#endif // NW_SRC_BOUNDARY_CONDITIONS_HPP
