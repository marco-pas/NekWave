// @@ Note
// The .rea mesh parsing logic in this file is taken 1-to-1 from the NekCEM / Nek5000
// format to enable rapid testing with validated electromagnetic benchmark cases.
// In future iterations, this legacy format will have to be replaced or augmented
// by a modern meshing architecture (such as Neko's .nmsh format or native Gmsh).

#ifndef NW_SRC_MESH_HPP
#define NW_SRC_MESH_HPP

#include <vector>
#include <string>
#include <array>

// @@ FaceInfo struct to hold boundary and neighbor connectivity
struct FaceInfo {
    int elementId;          // Element index (0-indexed)
    int faceId;             // Face index (0 to 5, corresponding to Nek faces 1 to 6)
    std::string bcType;     // Boundary condition type ("PEC", "E", "PML", etc.)
    int neighborElementId;  // Neighbor element index (-1 if domain boundary)
    int neighborFaceId;     // Neighbor face index (-1 if domain boundary)
};

// @@ FacePointData struct to hold precomputed face node geometry and connectivity
// Face node mappings and Nanson outward unit normals
struct FacePointData {
    int volIdxMinus;        // Volume index on interior element (minus side)
    int volIdxPlus;         // Volume index on exterior neighbor element (plus side, -1 for PEC)
    double nx, ny, nz;      // Physical outward unit normal vector
    double dA;              // Surface area metric element (J_face * w_p * w_q via Nanson formula)
};

// @@ ElementFaceData struct to store all quadrature points on a hex face
struct ElementFaceData {
    int elementId;
    int faceId;
    std::string bcType;
    std::vector<FacePointData> points;
};

// @@ Mesh class which handles spatial domain, geometry, and .rea parsing
// Precomputation & Geometric Discretization
// Maps reference coordinates (r, s, t) in [-1, 1]^3 to physical hex elements
class Mesh {
public:
    // @@ init mesh with polynomial order N and element count
    Mesh(int N = 3, int numElements = 1);
    
    // @@ cleaning up mesh resources
    ~Mesh();
    
    // @@ init mesh geometry, operators, and default metrics
    // Setup GLL quadrature nodes, weights, and differentiation matrices
    void initialize();

    // @@ load mesh geometry and connectivity directly from a NekCEM .rea file
    // Adapted 1-to-1 from NekCEM format; to be replaced by Neko-style reader in the future
    bool loadFromRea(const std::string& filename);

    // @@ create a structured Cartesian box mesh with arbitrary elements in each direction
    // Automatic multi-element Cartesian hex mesh generation
    bool createBoxMesh(int nelx, int nely, int nelz,
                       double xmin = -1.0, double xmax = 1.0,
                       double ymin = -1.0, double ymax = 1.0,
                       double zmin = -1.0, double zmax = 1.0,
                       bool periodicX = false, bool periodicY = false, bool periodicZ = false);

    // @@ config an element with an affine bounding box transformation
    // Computes constant Jacobian J and metric factors J^{-T}
    void setAffineBox(int elementIdx, double xmin, double xmax, double ymin, double ymax, double zmin, double zmax);

    // @@ number of 1D points N
    int getN() const { return m_N; }

    // @@ polynomial order N
    void setN(int N);

    // @@ number of points per element N_p = N * N * N
    int getNumPointsPerElement() const { return m_numPointsPerElement; }

    // @@ number of elements in the mesh
    int getNumElements() const { return m_numElements; }

    // @@ total number of points across all elements
    int getTotalPoints() const { return m_totalPoints; }

    // @@ minimum physical distance between adjacent collocation nodes across all elements
    // Computes dxmin matching NekCEM's get_dxmin in nek5_courant.F
    double computeMinNodeDistance() const;

    // @@ automatic stable CFL Courant number
    // CFL from dimension d=3, polynomial order N, and LSRK45 stability bound
    double computeAutomaticCFL(double safetyFactor = 0.4) const;
    // dt = CFL_auto * (dxmin / c)
    double computeAutomaticDt(double waveSpeed = 1.0, double safetyFactor = 0.4) const;

    // @@ 1D derivative matrix D
    const std::vector<double>& getD() const { return m_D; }

    // @@ transpose of 1D derivative matrix Dt
    const std::vector<double>& getDt() const { return m_Dt; }

    // @@ 3D quadrature weights w3
    const std::vector<double>& getW3() const { return m_w3; }

    // @@ 1D GLL quadrature nodes and weights
    const std::vector<double>& getGllZ() const { return m_gll_z; }
    const std::vector<double>& getGllW() const { return m_gll_w; }

    // @@ face connectivity and boundary condition records
    const std::vector<FaceInfo>& getFaces() const { return m_faces; }

    // @@ precomputed face quadrature and connectivity data
    // Face data list for interface flux evaluation
    const std::vector<ElementFaceData>& getFaceData() const { return m_faceData; }

    // @@ computing the volume index for a node (p, q) on a given element face
    // Maps face (p, q) collocation node to 3D volume node index
    int getFaceNodeVolIndex(int elemId, int faceId, int p, int q) const;

    // @@ geometric metric factors (entries of J^{-T} or cofactor matrix)
    const std::vector<double>& getRx() const { return m_rx; }
    const std::vector<double>& getSx() const { return m_sx; }
    const std::vector<double>& getTx() const { return m_tx; }
    const std::vector<double>& getRy() const { return m_ry; }
    const std::vector<double>& getSy() const { return m_sy; }
    const std::vector<double>& getTy() const { return m_ty; }
    const std::vector<double>& getRz() const { return m_rz; }
    const std::vector<double>& getSz() const { return m_sz; }
    const std::vector<double>& getTz() const { return m_tz; }
    const std::vector<double>& getJac() const { return m_jac; }

    // @@ physical coordinates of all collocation nodes
    const std::vector<double>& getCoordX() const { return m_coord_x; }
    const std::vector<double>& getCoordY() const { return m_coord_y; }
    const std::vector<double>& getCoordZ() const { return m_coord_z; }

    // @@ element corners in physical space
    const std::vector<std::array<std::array<double, 3>, 8>>& getElementCorners() const { return m_elementCorners; }

private:
    // @@ store polynomial degree parameters
    int m_N;
    int m_numPointsPerElement;
    int m_numElements;
    int m_totalPoints;

    // @@ store 1D reference coordinates and weights
    std::vector<double> m_gll_z;
    std::vector<double> m_gll_w;

    // @@ store 1D differentiation matrices D and Dt
    std::vector<double> m_D;
    std::vector<double> m_Dt;

    // @@ store 3D element quadrature weights w3
    std::vector<double> m_w3;

    // @@ store physical node coordinates
    std::vector<double> m_coord_x;
    std::vector<double> m_coord_y;
    std::vector<double> m_coord_z;

    // @@ store geometric metric factor arrays of length m_totalPoints
    std::vector<double> m_rx;
    std::vector<double> m_sx;
    std::vector<double> m_tx;
    std::vector<double> m_ry;
    std::vector<double> m_sy;
    std::vector<double> m_ty;
    std::vector<double> m_rz;
    std::vector<double> m_sz;
    std::vector<double> m_tz;
    std::vector<double> m_jac;

    // @@ store the 8 corner coordinates for each element
    std::vector<std::array<std::array<double, 3>, 8>> m_elementCorners;

    // @@ store face connectivity and boundary conditions
    std::vector<FaceInfo> m_faces;

    // @@ store the precomputed geometric and connectivity data for each face
    std::vector<ElementFaceData> m_faceData;

    // @@ helper to setup 1D GLL points, weights, and derivative matrix D
    void setupGLL();

    // @@ helper to recompute physical metrics from element corners
    void computeMetricsFromCorners();

    // @@ helper to compute face normals, area metrics, and neighbor node matching
    // NANSON RELATIONS & PIOLA: Precomputes n * dA = J * J^{-T} * n_ref * dA_ref and trace indices
    void setupFaceData();
};

#endif // NW_SRC_MESH_HPP
