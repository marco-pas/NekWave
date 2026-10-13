// @@ Note
// The .rea mesh parsing logic in this file is taken 1-to-1 from the NekCEM / Nek5000
// In future Neko's .nmsh format or native Gmsh
// this is also where the operator gets built as these olny depend on the mesh topology

#include "mesh.hpp"
#include "device/gpu_runtime.hpp"
#include "device/dg_preproc_kernels.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <cstdint>
#include <tuple>

// @@ Helper to evaluate the Legendre polynomial of degree n at x
// Use Bonnet's 3-term recurrence relation (https://en.wikipedia.org/wiki/Legendre_polynomials)
static double legendrePoly(int n, double x) {
    if (n == 0) return 1.0;
    if (n == 1) return x;
    double p0 = 1.0;
    double p1 = x;
    double p2 = 0.0;
    for (int k = 1; k < n; ++k) {
        p2 = ((2.0 * k + 1.0) * x * p1 - k * p0) / (k + 1.0);
        p0 = p1;
        p1 = p2;
    }
    return p2;
}

// @@ Helper to evaluate the derivative of Legendre polynomial of degree n at x
// Analytic derivative of Legendre polynomial for root finding
static double legendrePolyDeriv(int n, double x) {
    if (n == 0) return 0.0;
    if (n == 1) return 1.0;
    return (n / (x * x - 1.0)) * (x * legendrePoly(n, x) - legendrePoly(n - 1, x));
}

// @@ Mesh constructor
Mesh::Mesh(int N, int numElements)
    : m_N(N),
      m_numElements(numElements),
      m_bIsPartitioned(false),
      m_numHaloPoints(0) {
    m_numPointsPerElement = m_N * m_N * m_N;
    m_totalPoints = m_numElements * m_numPointsPerElement;
}

// @@ Mesh destructor
Mesh::~Mesh() = default;

// @@ Polynomial order N
void Mesh::setN(int N) {
    m_N = N;
    m_numPointsPerElement = m_N * m_N * m_N;
    m_totalPoints = m_numElements * m_numPointsPerElement;
    invalidateGeometryCache();
    setupGLL();
}

void Mesh::invalidateGeometryCache() {
    m_coord_x.clear();
    m_coord_y.clear();
    m_coord_z.clear();
    m_jac.clear();
    m_faceData.clear();
    m_cachedMinNodeDist = -1.0;
}

// @@ GLL collocation points, weights, and derivative matrix setup
void Mesh::setupGLL() {
    using Clock = std::chrono::high_resolution_clock;
    auto tStart = Clock::now();
    NW_PROFILER_PUSH("Mesh::setupGLL");
    int N = m_N;
    int p = N - 1; // polynomial degree

    m_gll_z.resize(N);
    m_gll_w.resize(N);
    m_D.assign(N * N, 0.0);
    m_Dt.assign(N * N, 0.0);

    // @@ compute 1D Gauss-Lobatto-Legendre (GLL) points and quadrature weights
    // Collocation on [-1, 1] using roots of (1 - x^2) L'_p(x) = 0
    if (N == 1) {
        m_gll_z[0] = 0.0;
        m_gll_w[0] = 2.0;
        m_D[0] = 0.0;
        m_Dt[0] = 0.0;
    } else {
        m_gll_z[0] = -1.0;
        m_gll_z[N - 1] = 1.0;

        // @@ compute interior GLL roots via Newton-Raphson iteration
        // %% Newton-Raphson root finding with Chebyshev initial guess
        for (int i = 1; i < N - 1; ++i) {
            double xi = -std::cos((M_PI * i) / p);
            for (int it = 0; it < 20; ++it) {
                double L = legendrePoly(p, xi);
                double Ld = legendrePolyDeriv(p, xi);
                double Ldd = (2.0 * xi * Ld - p * (p + 1.0) * L) / (1.0 - xi * xi);
                double delta = Ld / Ldd;
                xi -= delta;
                if (std::abs(delta) < 1e-15) break;
            }
            m_gll_z[i] = xi;
        }

        // @@ compute 1D GLL integration weights
        // GLL weights: w_i = 2 / (p * (p + 1) * [L_p(x_i)]^2)
        for (int i = 0; i < N; ++i) {
            double L = legendrePoly(p, m_gll_z[i]);
            m_gll_w[i] = 2.0 / (p * (p + 1.0) * L * L);
        }

        // --
        // IMPORTANT: this is where the stiffness matrix gets built!
        // This is done just once while building the mesh.
        // The matrix gets called in "gpu_volume_curl_kernel"
        // --
        // @@ build 1D derivative matrix D and its transpose Dt (NekCEM DGLL)
        // D_ij = L_p(x_i) / ( L_p(x_j) * (x_i - x_j) ) 
        double d0 = p * (p + 1.0) / 4.0;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                int idx = i + j * N; // column-major: row i, column j
                if (i != j) {
                    double Li = legendrePoly(p, m_gll_z[i]);
                    double Lj = legendrePoly(p, m_gll_z[j]);
                    m_D[idx] = Li / (Lj * (m_gll_z[i] - m_gll_z[j]));
                } else if (i == 0 && j == 0) {
                    m_D[idx] = -d0;
                } else if (i == p && j == p) {
                    m_D[idx] = d0;
                } else {
                    m_D[idx] = 0.0;
                }
                // @@ recording the transpose Dt
                m_Dt[j + i * N] = m_D[idx];
            }
        }
    }

    // @@ generating 3D quadrature weights w3 = wi * wj * wk
    // 3D Tensor-Product Quadrature Weights w3_ijk = w_i * w_j * w_k
    m_w3.resize(m_numPointsPerElement);
    for (int k = 0; k < N; ++k) {
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < N; ++i) {
                int idx = i + N * (j + N * k);
                m_w3[idx] = m_gll_w[i] * m_gll_w[j] * m_gll_w[k];
            }
        }
    }
    NW_PROFILER_POP();
    m_subTimings.gllSetupSec += std::chrono::duration<double>(Clock::now() - tStart).count();
}

// @@ configuring an element with an affine bounding box transformation
void Mesh::setAffineBox(int elementIdx, double xmin, double xmax,
                        double ymin, double ymax, double zmin, double zmax) {
    if (elementIdx < 0 || elementIdx >= m_numElements) return;

    if (m_elementCorners.size() != static_cast<size_t>(m_numElements)) {
        m_elementCorners.resize(m_numElements);
    }

    m_elementCorners[elementIdx][0] = {{xmin, ymin, zmin}};
    m_elementCorners[elementIdx][1] = {{xmax, ymin, zmin}};
    m_elementCorners[elementIdx][2] = {{xmax, ymax, zmin}};
    m_elementCorners[elementIdx][3] = {{xmin, ymax, zmin}};
    m_elementCorners[elementIdx][4] = {{xmin, ymin, zmax}};
    m_elementCorners[elementIdx][5] = {{xmax, ymin, zmax}};
    m_elementCorners[elementIdx][6] = {{xmax, ymax, zmax}};
    m_elementCorners[elementIdx][7] = {{xmin, ymax, zmax}};

    if (m_faces.empty()) {
        for (int e = 0; e < m_numElements; ++e) {
            for (int f = 0; f < 6; ++f) {
                FaceInfo info;
                info.elementId = e;
                info.faceId = f;
                info.bcType = "PEC";
                info.neighborElementId = -1;
                info.neighborFaceId = -1;
                m_faces.push_back(info);
            }
        }
    }
    invalidateGeometryCache();
}

// @@ Helper to resolve relative mesh file paths across working directories and NEKWAVE_SOURCE_DIR
static std::string resolveMeshFilePath(const std::string& filename, std::ios_base::openmode mode = std::ios_base::in) {
    if (filename.empty()) return "";
    std::vector<std::string> candidates = {
        filename,
        "../" + filename,
        "../../" + filename,
        "../../../" + filename
#ifdef NEKWAVE_SOURCE_DIR
        , std::string(NEKWAVE_SOURCE_DIR) + "/" + filename
#endif
    };
    for (const auto& cand : candidates) {
        std::ifstream testFile(cand, mode);
        if (testFile.is_open()) {
            testFile.close();
            return cand;
        }
    }
    return "";
}

// @@ .rea file parser taken 1-to-1 from NekCEM / Nek5000
bool Mesh::loadFromRea(const std::string& filename) {
    using Clock = std::chrono::high_resolution_clock;
    m_subTimings = PreprocessSubTimings{};
    auto tParseStart = Clock::now();
    NW_PROFILER_PUSH("Mesh::parseReaFile");
    std::string resolvedPath = resolveMeshFilePath(filename, std::ios_base::in);
    if (resolvedPath.empty()) {
        std::cerr << "[ERROR] Could not open .rea mesh file: '" << filename << "'" << std::endl;
        NW_PROFILER_POP();
        return false;
    }
    std::ifstream file(resolvedPath);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not open .rea mesh file: '" << resolvedPath << "'" << std::endl;
        NW_PROFILER_POP();
        return false;
    }

    std::string line;
    bool foundMesh = false;
    int nel = 0, ndim = 3;

    // @@ scanning the file for the **MESH DATA** section (and optional PMLTHICK param 77)
    while (std::getline(file, line)) {
        if (line.find("PMLTHICK") != std::string::npos) {
            std::stringstream pss(line);
            double pmlThickVal = 0.0;
            if (pss >> pmlThickVal && pmlThickVal >= 1.0) {
                m_pmlConfig.thickness = static_cast<int>(std::round(pmlThickVal));
            }
        }
        if (line.find("**MESH DATA**") != std::string::npos) {
            foundMesh = true;
            break;
        }
    }

    if (!foundMesh) {
        std::cerr << "Error: Could not find **MESH DATA** section in " << resolvedPath << std::endl;
        NW_PROFILER_POP();
        return false;
    }

    // @@ reading the line with NEL, NDIM
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        ss >> nel >> ndim;
    }

    if (nel < 0) {
        NW_PROFILER_POP();
        std::cout << "[MESH] Negative NEL = " << nel << " indicates binary .re2 mesh. Redirecting to loadFromRe2..." << std::endl;
        std::string re2name = resolvedPath;
        size_t dotPos = re2name.rfind('.');
        if (dotPos != std::string::npos) {
            re2name = re2name.substr(0, dotPos) + ".re2";
        } else {
            re2name += ".re2";
        }
        return loadFromRe2(re2name, std::abs(nel));
    }

    if (nel == 0) {
        std::cerr << "Error: Invalid number of elements (NEL = " << nel << ") in " << resolvedPath << std::endl;
        NW_PROFILER_POP();
        return false;
    }

    m_numElements = nel;
    m_totalPoints = m_numElements * m_numPointsPerElement;
    m_elementCorners.resize(m_numElements);

    // @@ reading 8 corner coordinates for each hex element
    for (int e = 0; e < nel; ++e) {
        // Read header line: "ELEMENT ... [ ... ] GROUP ..."
        std::getline(file, line);

        double x[8], y[8], z[8];
        // Line 1: x1, x2, x3, x4
        std::getline(file, line);
        { std::stringstream ss(line); ss >> x[0] >> x[1] >> x[2] >> x[3]; }
        // Line 2: y1, y2, y3, y4
        std::getline(file, line);
        { std::stringstream ss(line); ss >> y[0] >> y[1] >> y[2] >> y[3]; }
        // Line 3: z1, z2, z3, z4
        std::getline(file, line);
        { std::stringstream ss(line); ss >> z[0] >> z[1] >> z[2] >> z[3]; }

        // Line 4: x5, x6, x7, x8
        std::getline(file, line);
        { std::stringstream ss(line); ss >> x[4] >> x[5] >> x[6] >> x[7]; }
        // Line 5: y5, y6, y7, y8
        std::getline(file, line);
        { std::stringstream ss(line); ss >> y[4] >> y[5] >> y[6] >> y[7]; }
        // Line 6: z5, z6, z7, z8
        std::getline(file, line);
        { std::stringstream ss(line); ss >> z[4] >> z[5] >> z[6] >> z[7]; }

        for (int a = 0; a < 8; ++a) {
            m_elementCorners[e][a] = {x[a], y[a], z[a]};
        }
    }

    // @@ scanning for boundary conditions section
    bool foundBC = false;
    while (std::getline(file, line)) {
        if (line.find("FLUID") != std::string::npos && line.find("BOUNDARY CONDITIONS") != std::string::npos) {
            foundBC = true;
            break;
        }
    }

    m_faces.clear();
    if (foundBC) {
        int totalFaces = nel * 6;
        for (int f = 0; f < totalFaces; ++f) {
            if (!std::getline(file, line)) break;
            if (line.empty()) continue;

            std::stringstream ss(line);
            std::string bcType;
            int elemId = 0, faceId = 0;
            double nbrElem = 0.0, nbrFace = 0.0;

            ss >> bcType >> elemId >> faceId >> nbrElem >> nbrFace;

            FaceInfo info;
            info.elementId = elemId - 1; // Convert 1-based to 0-based
            info.faceId = faceId - 1;    // Convert 1-based to 0-based
            info.neighborElementId = (nbrElem > 0.0) ? static_cast<int>(nbrElem) - 1 : -1;
            info.neighborFaceId = (nbrFace > 0.0) ? static_cast<int>(nbrFace) - 1 : -1;
            info.bcType = BoundaryConditions::toString(
                BoundaryConditions::parseBcTag(bcType, info.neighborElementId)
            );

            m_faces.push_back(info);
        }
    }
    NW_PROFILER_POP();
    m_subTimings.fileReadAndConnectivitySec += std::chrono::duration<double>(Clock::now() - tParseStart).count();

    setupGLL();
    invalidateGeometryCache();
    setupPml();

    return true;
}

// @@ load mesh directly from a NekCEM binary .re2 file
bool Mesh::loadFromRe2(const std::string& filename, int nel) {
    using Clock = std::chrono::high_resolution_clock;
    m_subTimings = PreprocessSubTimings{};
    auto tParseStart = Clock::now();
    NW_PROFILER_PUSH("Mesh::parseRe2AndPairFaces");
    std::string resolvedPath = resolveMeshFilePath(filename, std::ios::binary);
    if (resolvedPath.empty()) {
        std::cerr << "[ERROR] Could not open .re2 mesh file: '" << filename << "'" << std::endl;
        NW_PROFILER_POP();
        return false;
    }
    std::ifstream file(resolvedPath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not open .re2 mesh file: '" << resolvedPath << "'" << std::endl;
        NW_PROFILER_POP();
        return false;
    }

    // Read 80-byte header
    char header[81] = {0};
    file.read(header, 80);
    if (!file) {
        std::cerr << "Error: Failed to read header from " << filename << std::endl;
        NW_PROFILER_POP();
        return false;
    }

    // Read 4-byte endianness test tag (6.54321)
    float tag = 0.0f;
    file.read(reinterpret_cast<char*>(&tag), sizeof(float));
    if (!file) {
        std::cerr << "Error: Failed to read tag from " << filename << std::endl;
        NW_PROFILER_POP();
        return false;
    }

    // If nel wasn't provided, parse from header: "#v002 <nel> <ndim> <nelv> ..."
    if (nel <= 0) {
        std::string hdrStr(header, 80);
        size_t p = hdrStr.find_first_not_of(" \t", 5);
        if (p != std::string::npos) {
            std::stringstream ss(hdrStr.substr(p));
            ss >> nel;
        }
    }

    if (nel <= 0) {
        std::cerr << "Error: Invalid number of elements (NEL = " << nel << ") in " << filename << std::endl;
        NW_PROFILER_POP();
        return false;
    }

    m_numElements = nel;
    m_totalPoints = m_numElements * m_numPointsPerElement;
    m_elementCorners.resize(m_numElements);

    // Bulk-read all elements: each element is 25 doubles (group + 8 x + 8 y + 8 z)
    std::vector<double> elemBuf(static_cast<size_t>(nel) * 25);
    file.read(reinterpret_cast<char*>(elemBuf.data()), static_cast<std::streamsize>(elemBuf.size() * sizeof(double)));
    if (!file) {
        std::cerr << "[ERROR] Failed to read element coordinate block from " << resolvedPath << std::endl;
        NW_PROFILER_POP();
        return false;
    }

    for (int e = 0; e < nel; ++e) {
        const double* rec = &elemBuf[static_cast<size_t>(e) * 25];
        const double* x = rec + 1;
        const double* y = rec + 9;
        const double* z = rec + 17;
        for (int a = 0; a < 8; ++a) {
            m_elementCorners[e][a] = {x[a], y[a], z[a]};
        }
    }
    elemBuf.clear();
    elemBuf.shrink_to_fit();

    // Read curved faces count
    double ncurvD = 0.0;
    file.read(reinterpret_cast<char*>(&ncurvD), sizeof(double));

    // Read boundary conditions count
    double nbcD = 0.0;
    file.read(reinterpret_cast<char*>(&nbcD), sizeof(double));
    int nbc = static_cast<int>(nbcD);

    m_faces.clear();
    m_faces.resize(static_cast<size_t>(nel) * 6);
    std::vector<uint8_t> hasBc(static_cast<size_t>(nel) * 6, 0);

    for (int i = 0; i < nbc; ++i) {
        double bcDoubles[7] = {0.0};
        char tagStr[9] = {0};

        file.read(reinterpret_cast<char*>(bcDoubles), 7 * sizeof(double));
        file.read(tagStr, 8);

        int e = static_cast<int>(bcDoubles[0]) - 1;
        int f = static_cast<int>(bcDoubles[1]) - 1;
        int nbrE = (bcDoubles[2] > 0.0) ? (static_cast<int>(bcDoubles[2]) - 1) : -1;
        int nbrF = (bcDoubles[3] > 0.0) ? (static_cast<int>(bcDoubles[3]) - 1) : -1;

        if (e < 0 || e >= nel || f < 0 || f >= 6) continue;

        std::string rawTag(tagStr);
        BcType parsedBc = BoundaryConditions::parseBcTag(rawTag, nbrE);
        if (parsedBc == BcType::INTERIOR && nbrE >= 0 && rawTag.find('E') == std::string::npos) {
            parsedBc = BcType::PERIODIC;
        }

        size_t idx = static_cast<size_t>(e) * 6 + static_cast<size_t>(f);
        m_faces[idx].elementId = e;
        m_faces[idx].faceId = f;
        m_faces[idx].bcType = BoundaryConditions::toString(parsedBc);
        m_faces[idx].neighborElementId = nbrE;
        m_faces[idx].neighborFaceId = nbrF;
        hasBc[idx] = 1;
    }

    const int faceCorners[6][4] = {
        {0, 1, 5, 4},
        {1, 2, 6, 5},
        {2, 3, 7, 6},
        {3, 0, 4, 7},
        {0, 3, 2, 1},
        {4, 5, 6, 7}
    };

    // O(1) spatial hash table for pairing interior conforming faces by centroid
    std::unordered_map<uint64_t, int> interiorCenters;
    interiorCenters.reserve(static_cast<size_t>(nel) * 3);
    auto makeHashKey = [](double cx, double cy, double cz) -> uint64_t {
        uint64_t ux = static_cast<uint64_t>(static_cast<int64_t>(std::llround(cx * 1000000.0)));
        uint64_t uy = static_cast<uint64_t>(static_cast<int64_t>(std::llround(cy * 1000000.0)));
        uint64_t uz = static_cast<uint64_t>(static_cast<int64_t>(std::llround(cz * 1000000.0)));
        uint64_t h = ux * 0x9E3779B97F4A7C15ULL;
        h ^= uy * 0xBF58476D1CE4E5B9ULL + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
        h ^= uz * 0x94D049BB133111EBULL + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
        return h;
    };

    for (int e = 0; e < nel; ++e) {
        for (int f = 0; f < 6; ++f) {
            int idx = e * 6 + f;
            if (hasBc[idx]) continue;

            double cx = 0.0, cy = 0.0, cz = 0.0;
            for (int v = 0; v < 4; ++v) {
                int cIdx = faceCorners[f][v];
                cx += m_elementCorners[e][cIdx][0] * 0.25;
                cy += m_elementCorners[e][cIdx][1] * 0.25;
                cz += m_elementCorners[e][cIdx][2] * 0.25;
            }
            uint64_t key = makeHashKey(cx, cy, cz);
            auto cIt = interiorCenters.find(key);
            if (cIt != interiorCenters.end()) {
                int otherIdx = cIt->second;
                int otherE = otherIdx / 6;
                int otherF = otherIdx % 6;

                m_faces[idx].elementId = e;
                m_faces[idx].faceId = f;
                m_faces[idx].bcType = "E";
                m_faces[idx].neighborElementId = otherE;
                m_faces[idx].neighborFaceId = otherF;

                m_faces[otherIdx].elementId = otherE;
                m_faces[otherIdx].faceId = otherF;
                m_faces[otherIdx].bcType = "E";
                m_faces[otherIdx].neighborElementId = e;
                m_faces[otherIdx].neighborFaceId = f;
                interiorCenters.erase(cIt);
            } else {
                m_faces[idx].elementId = e;
                m_faces[idx].faceId = f;
                m_faces[idx].bcType = "PEC";
                m_faces[idx].neighborElementId = -1;
                m_faces[idx].neighborFaceId = -1;
                interiorCenters.emplace(key, idx);
            }
        }
    }
    NW_PROFILER_POP();
    m_subTimings.fileReadAndConnectivitySec += std::chrono::duration<double>(Clock::now() - tParseStart).count();

    setupGLL();
    invalidateGeometryCache();
    setupPml();
    return true;
}

// @@ rescale mesh bounding box to specified physical coordinates [xmin, xmax] x [ymin, ymax] x [zmin, zmax]
void Mesh::rescale(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax) {
    if (m_numElements <= 0 || m_elementCorners.empty()) return;

    double curXmin = 1e30, curXmax = -1e30;
    double curYmin = 1e30, curYmax = -1e30;
    double curZmin = 1e30, curZmax = -1e30;

    for (int e = 0; e < m_numElements; ++e) {
        for (int a = 0; a < 8; ++a) {
            curXmin = std::min(curXmin, m_elementCorners[e][a][0]);
            curXmax = std::max(curXmax, m_elementCorners[e][a][0]);
            curYmin = std::min(curYmin, m_elementCorners[e][a][1]);
            curYmax = std::max(curYmax, m_elementCorners[e][a][1]);
            curZmin = std::min(curZmin, m_elementCorners[e][a][2]);
            curZmax = std::max(curZmax, m_elementCorners[e][a][2]);
        }
    }

    double dxCur = curXmax - curXmin;
    double dyCur = curYmax - curYmin;
    double dzCur = curZmax - curZmin;

    if (dxCur < 1e-14) dxCur = 1.0;
    if (dyCur < 1e-14) dyCur = 1.0;
    if (dzCur < 1e-14) dzCur = 1.0;

    for (int e = 0; e < m_numElements; ++e) {
        for (int a = 0; a < 8; ++a) {
            double rx = (m_elementCorners[e][a][0] - curXmin) / dxCur;
            double ry = (m_elementCorners[e][a][1] - curYmin) / dyCur;
            double rz = (m_elementCorners[e][a][2] - curZmin) / dzCur;

            m_elementCorners[e][a][0] = xmin + rx * (xmax - xmin);
            m_elementCorners[e][a][1] = ymin + ry * (ymax - ymin);
            m_elementCorners[e][a][2] = zmin + rz * (zmax - zmin);
        }
    }

    invalidateGeometryCache();
    setupPml();
}

// @@ generate structured Cartesian box mesh with nelx * nely * nelz hexahedral elements
// Exact element corner calculation and interface connectivity
bool Mesh::createBoxMesh(int nelx, int nely, int nelz,
                         double xmin, double xmax,
                         double ymin, double ymax,
                         double zmin, double zmax,
                         bool periodicX, bool periodicY, bool periodicZ,
                         const std::string& outerBc) {
    if (nelx < 1 || nely < 1 || nelz < 1) return false;

    using Clock = std::chrono::high_resolution_clock;
    m_subTimings = PreprocessSubTimings{};
    NW_PROFILER_PUSH("Mesh::createBoxMesh");
    m_numElements = nelx * nely * nelz;
    m_totalPoints = m_numElements * m_numPointsPerElement;
    setupGLL();

    auto tBoxStart = Clock::now();
    std::string extBc = BoundaryConditions::toString(BoundaryConditions::parseBcTag(outerBc, -1));
    if (extBc == "E" || extBc == "PERIODIC" || extBc == "MPI") {
        extBc = "PEC";
    }

    double dx = (xmax - xmin) / nelx;
    double dy = (ymax - ymin) / nely;
    double dz = (zmax - zmin) / nelz;

    m_elementCorners.resize(m_numElements);
    m_faces.clear();

    // Element index lambda: e = ex + nelx * (ey + nely * ez)
    auto elemIdx = [=](int ex, int ey, int ez) {
        return ex + nelx * (ey + nely * ez);
    };

    for (int ez = 0; ez < nelz; ++ez) {
        double z0 = zmin + ez * dz;
        double z1 = z0 + dz;
        for (int ey = 0; ey < nely; ++ey) {
            double y0 = ymin + ey * dy;
            double y1 = y0 + dy;
            for (int ex = 0; ex < nelx; ++ex) {
                double x0 = xmin + ex * dx;
                double x1 = x0 + dx;
                int e = elemIdx(ex, ey, ez);

                // 8 corners for hexahedron in standard right-handed ordering
                m_elementCorners[e][0] = {x0, y0, z0};
                m_elementCorners[e][1] = {x1, y0, z0};
                m_elementCorners[e][2] = {x1, y1, z0};
                m_elementCorners[e][3] = {x0, y1, z0};
                m_elementCorners[e][4] = {x0, y0, z1};
                m_elementCorners[e][5] = {x1, y0, z1};
                m_elementCorners[e][6] = {x1, y1, z1};
                m_elementCorners[e][7] = {x0, y1, z1};

                // Face 0: s = -1 (y min) -> neighbor is (ex, ey - 1, ez) with its face 2 (s = +1)
                FaceInfo f0;
                f0.elementId = e; f0.faceId = 0;
                if (ey > 0) {
                    f0.bcType = "E";
                    f0.neighborElementId = elemIdx(ex, ey - 1, ez);
                    f0.neighborFaceId = 2;
                } else if (periodicY) {
                    f0.bcType = "PERIODIC";
                    f0.neighborElementId = elemIdx(ex, nely - 1, ez);
                    f0.neighborFaceId = 2;
                } else {
                    f0.bcType = extBc;
                    f0.neighborElementId = -1; f0.neighborFaceId = -1;
                }
                m_faces.push_back(f0);

                // Face 1: r = +1 (x max) -> neighbor is (ex + 1, ey, ez) with its face 3 (r = -1)
                FaceInfo f1;
                f1.elementId = e; f1.faceId = 1;
                if (ex < nelx - 1) {
                    f1.bcType = "E";
                    f1.neighborElementId = elemIdx(ex + 1, ey, ez);
                    f1.neighborFaceId = 3;
                } else if (periodicX) {
                    f1.bcType = "PERIODIC";
                    f1.neighborElementId = elemIdx(0, ey, ez);
                    f1.neighborFaceId = 3;
                } else {
                    f1.bcType = extBc;
                    f1.neighborElementId = -1; f1.neighborFaceId = -1;
                }
                m_faces.push_back(f1);

                // Face 2: s = +1 (y max) -> neighbor is (ex, ey + 1, ez) with its face 0 (s = -1)
                FaceInfo f2;
                f2.elementId = e; f2.faceId = 2;
                if (ey < nely - 1) {
                    f2.bcType = "E";
                    f2.neighborElementId = elemIdx(ex, ey + 1, ez);
                    f2.neighborFaceId = 0;
                } else if (periodicY) {
                    f2.bcType = "PERIODIC";
                    f2.neighborElementId = elemIdx(ex, 0, ez);
                    f2.neighborFaceId = 0;
                } else {
                    f2.bcType = extBc;
                    f2.neighborElementId = -1; f2.neighborFaceId = -1;
                }
                m_faces.push_back(f2);

                // Face 3: r = -1 (x min) -> neighbor is (ex - 1, ey, ez) with its face 1 (r = +1)
                FaceInfo f3;
                f3.elementId = e; f3.faceId = 3;
                if (ex > 0) {
                    f3.bcType = "E";
                    f3.neighborElementId = elemIdx(ex - 1, ey, ez);
                    f3.neighborFaceId = 1;
                } else if (periodicX) {
                    f3.bcType = "PERIODIC";
                    f3.neighborElementId = elemIdx(nelx - 1, ey, ez);
                    f3.neighborFaceId = 1;
                } else {
                    f3.bcType = extBc;
                    f3.neighborElementId = -1; f3.neighborFaceId = -1;
                }
                m_faces.push_back(f3);

                // Face 4: t = -1 (z min) -> neighbor is (ex, ey, ez - 1) with its face 5 (t = +1)
                FaceInfo f4;
                f4.elementId = e; f4.faceId = 4;
                if (ez > 0) {
                    f4.bcType = "E";
                    f4.neighborElementId = elemIdx(ex, ey, ez - 1);
                    f4.neighborFaceId = 5;
                } else if (periodicZ) {
                    f4.bcType = "PERIODIC";
                    f4.neighborElementId = elemIdx(ex, ey, nelz - 1);
                    f4.neighborFaceId = 5;
                } else {
                    f4.bcType = extBc;
                    f4.neighborElementId = -1; f4.neighborFaceId = -1;
                }
                m_faces.push_back(f4);

                // Face 5: t = +1 (z max) -> neighbor is (ex, ey, ez + 1) with its face 4 (t = -1)
                FaceInfo f5;
                f5.elementId = e; f5.faceId = 5;
                if (ez < nelz - 1) {
                    f5.bcType = "E";
                    f5.neighborElementId = elemIdx(ex, ey, ez + 1);
                    f5.neighborFaceId = 4;
                } else if (periodicZ) {
                    f5.bcType = "PERIODIC";
                    f5.neighborElementId = elemIdx(ex, ey, 0);
                    f5.neighborFaceId = 4;
                } else {
                    f5.bcType = extBc;
                    f5.neighborElementId = -1; f5.neighborFaceId = -1;
                }
                m_faces.push_back(f5);
            }
        }
    }
    m_subTimings.fileReadAndConnectivitySec += std::chrono::duration<double>(Clock::now() - tBoxStart).count();

    invalidateGeometryCache();
    setupPml();

    NW_PROFILER_POP();
    return true;
}

// @@ the default mesh initialization routine
void Mesh::initialize() {
    setupGLL();

    // Default reference cube [-1, 1]^3
    m_elementCorners.resize(m_numElements);
    for (int e = 0; e < m_numElements; ++e) {
        m_elementCorners[e][0] = {{-1.0, -1.0, -1.0}};
        m_elementCorners[e][1] = {{ 1.0, -1.0, -1.0}};
        m_elementCorners[e][2] = {{ 1.0,  1.0, -1.0}};
        m_elementCorners[e][3] = {{-1.0,  1.0, -1.0}};
        m_elementCorners[e][4] = {{-1.0, -1.0,  1.0}};
        m_elementCorners[e][5] = {{ 1.0, -1.0,  1.0}};
        m_elementCorners[e][6] = {{ 1.0,  1.0,  1.0}};
        m_elementCorners[e][7] = {{-1.0,  1.0,  1.0}};
    }

    if (m_faces.empty()) {
        for (int e = 0; e < m_numElements; ++e) {
            for (int f = 0; f < 6; ++f) {
                FaceInfo info;
                info.elementId = e;
                info.faceId = f;
                info.bcType = "PEC";
                info.neighborElementId = -1;
                info.neighborFaceId = -1;
                m_faces.push_back(info);
            }
        }
    }
    invalidateGeometryCache();
}

void Mesh::getCornerBoundingBox(double& xmin, double& xmax,
                                double& ymin, double& ymax,
                                double& zmin, double& zmax) const {
    xmin = 1.0e30; xmax = -1.0e30;
    ymin = 1.0e30; ymax = -1.0e30;
    zmin = 1.0e30; zmax = -1.0e30;
    for (int e = 0; e < m_numElements; ++e) {
        for (int a = 0; a < 8; ++a) {
            xmin = std::min(xmin, m_elementCorners[e][a][0]);
            xmax = std::max(xmax, m_elementCorners[e][a][0]);
            ymin = std::min(ymin, m_elementCorners[e][a][1]);
            ymax = std::max(ymax, m_elementCorners[e][a][1]);
            zmin = std::min(zmin, m_elementCorners[e][a][2]);
            zmax = std::max(zmax, m_elementCorners[e][a][2]);
        }
    }
}

void Mesh::setHostCoordsAndJac(std::vector<double>&& cx,
                               std::vector<double>&& cy,
                               std::vector<double>&& cz,
                               std::vector<double>&& jac) const {
    m_coord_x = std::move(cx);
    m_coord_y = std::move(cy);
    m_coord_z = std::move(cz);
    m_jac     = std::move(jac);
}

size_t Mesh::getAllocatedHostBytes() const {
    return m_elementCorners.size() * 24ULL * sizeof(double) +
           m_faces.size() * sizeof(FaceInfo) +
           (m_coord_x.size() + m_coord_y.size() + m_coord_z.size() + m_jac.size()) * sizeof(double) +
           m_faceData.size() * 64ULL;
}

void Mesh::preprocessOnGpu(bool needCoordsAndJac, bool needFaceData) const {
    if (m_numElements <= 0 || m_totalPoints <= 0 || m_elementCorners.empty()) return;

    NW_PROFILER_PUSH("Mesh::preprocessOnGpu");
    const int N = m_N;
    const int nelt = m_numElements;
    const int npts = m_totalPoints;
    const size_t volDoublesBytes = static_cast<size_t>(npts) * sizeof(double);

    std::vector<double> h_corners(static_cast<size_t>(nelt) * 24);
    for (int e = 0; e < nelt; ++e) {
        for (int a = 0; a < 8; ++a) {
            h_corners[static_cast<size_t>(e) * 24 + a * 3 + 0] = m_elementCorners[e][a][0];
            h_corners[static_cast<size_t>(e) * 24 + a * 3 + 1] = m_elementCorners[e][a][1];
            h_corners[static_cast<size_t>(e) * 24 + a * 3 + 2] = m_elementCorners[e][a][2];
        }
    }

    double* d_corners = nullptr;
    double* d_gll_z   = nullptr;
    double* d_coordX  = nullptr;
    double* d_coordY  = nullptr;
    double* d_coordZ  = nullptr;
    double* d_jac     = nullptr;
    double* d_rx      = nullptr;
    double* d_sx      = nullptr;
    double* d_tx      = nullptr;
    double* d_ry      = nullptr;
    double* d_sy      = nullptr;
    double* d_ty      = nullptr;
    double* d_rz      = nullptr;
    double* d_sz      = nullptr;
    double* d_tz      = nullptr;

    NW_GPU_CHECK(cudaMalloc(&d_corners, h_corners.size() * sizeof(double)));
    NW_GPU_CHECK(cudaMalloc(&d_gll_z, static_cast<size_t>(N) * sizeof(double)));
    NW_GPU_CHECK(cudaMalloc(&d_coordX, volDoublesBytes));
    NW_GPU_CHECK(cudaMalloc(&d_coordY, volDoublesBytes));
    NW_GPU_CHECK(cudaMalloc(&d_coordZ, volDoublesBytes));
    NW_GPU_CHECK(cudaMalloc(&d_jac,    volDoublesBytes));
    NW_GPU_CHECK(cudaMalloc(&d_rx,     volDoublesBytes));
    NW_GPU_CHECK(cudaMalloc(&d_sx,     volDoublesBytes));
    NW_GPU_CHECK(cudaMalloc(&d_tx,     volDoublesBytes));
    NW_GPU_CHECK(cudaMalloc(&d_ry,     volDoublesBytes));
    NW_GPU_CHECK(cudaMalloc(&d_sy,     volDoublesBytes));
    NW_GPU_CHECK(cudaMalloc(&d_ty,     volDoublesBytes));
    NW_GPU_CHECK(cudaMalloc(&d_rz,     volDoublesBytes));
    NW_GPU_CHECK(cudaMalloc(&d_sz,     volDoublesBytes));
    NW_GPU_CHECK(cudaMalloc(&d_tz,     volDoublesBytes));

    NW_GPU_CHECK(cudaMemcpy(d_corners, h_corners.data(), h_corners.size() * sizeof(double), cudaMemcpyHostToDevice));
    NW_GPU_CHECK(cudaMemcpy(d_gll_z, m_gll_z.data(), static_cast<size_t>(N) * sizeof(double), cudaMemcpyHostToDevice));

    nekwave::device::launch_compute_volume_metrics(
        d_corners, d_gll_z,
        d_coordX, d_coordY, d_coordZ,
        d_jac,
        d_rx, d_sx, d_tx,
        d_ry, d_sy, d_ty,
        d_rz, d_sz, d_tz,
        N, nelt, npts
    );

    if (m_cachedMinNodeDist <= 0.0) {
        double dxiMin = (N > 1) ? (m_gll_z[1] - m_gll_z[0]) : 2.0;
        m_cachedMinNodeDist = nekwave::device::launch_compute_min_node_distance(
            d_coordX, d_coordY, d_coordZ,
            d_rx, d_sx, d_tx,
            d_ry, d_sy, d_ty,
            d_rz, d_sz, d_tz,
            dxiMin, N, nelt, npts
        );
    }

    if (needCoordsAndJac && m_coord_x.empty()) {
        m_coord_x.resize(npts);
        m_coord_y.resize(npts);
        m_coord_z.resize(npts);
        m_jac.resize(npts);
        NW_GPU_CHECK(cudaMemcpy(m_coord_x.data(), d_coordX, volDoublesBytes, cudaMemcpyDeviceToHost));
        NW_GPU_CHECK(cudaMemcpy(m_coord_y.data(), d_coordY, volDoublesBytes, cudaMemcpyDeviceToHost));
        NW_GPU_CHECK(cudaMemcpy(m_coord_z.data(), d_coordZ, volDoublesBytes, cudaMemcpyDeviceToHost));
        NW_GPU_CHECK(cudaMemcpy(m_jac.data(),     d_jac,    volDoublesBytes, cudaMemcpyDeviceToHost));
    }

    if (needFaceData && m_faceData.empty() && !m_faces.empty()) {
        const int numFaces = static_cast<int>(m_faces.size());
        const int N2 = N * N;
        const int totalFacePoints = numFaces * N2;
        const size_t faceIntsBytes    = static_cast<size_t>(totalFacePoints) * sizeof(int);
        const size_t faceDoublesBytes = static_cast<size_t>(totalFacePoints) * sizeof(double);

        std::vector<int> h_faceElem(numFaces);
        std::vector<int> h_faceId(numFaces);
        std::vector<int> h_neighborElem(numFaces);
        std::vector<int> h_neighborFace(numFaces);
        std::vector<int> h_faceBcCode(numFaces);
        for (int f = 0; f < numFaces; ++f) {
            h_faceElem[f]     = m_faces[f].elementId;
            h_faceId[f]       = m_faces[f].faceId;
            h_neighborElem[f] = m_faces[f].neighborElementId;
            h_neighborFace[f] = m_faces[f].neighborFaceId;
            h_faceBcCode[f]   = static_cast<int>(
                BoundaryConditions::parseBcTag(m_faces[f].bcType, m_faces[f].neighborElementId)
            );
        }

        int* d_faceElem        = nullptr;
        int* d_faceId          = nullptr;
        int* d_neighborElem    = nullptr;
        int* d_neighborFace    = nullptr;
        int* d_faceBcCode      = nullptr;
        int* d_mpiCutGhostIdx  = nullptr;
        double* d_gll_w        = nullptr;
        int* d_volIdxMinus     = nullptr;
        int* d_volIdxPlus      = nullptr;
        int* d_bcType          = nullptr;
        double* d_nx           = nullptr;
        double* d_ny           = nullptr;
        double* d_nz           = nullptr;
        double* d_dA           = nullptr;
        double* d_fx           = nullptr;
        double* d_fy           = nullptr;
        double* d_fz           = nullptr;

        const size_t topoIntsBytes = static_cast<size_t>(numFaces) * sizeof(int);
        NW_GPU_CHECK(cudaMalloc(&d_faceElem,     topoIntsBytes));
        NW_GPU_CHECK(cudaMalloc(&d_faceId,       topoIntsBytes));
        NW_GPU_CHECK(cudaMalloc(&d_neighborElem, topoIntsBytes));
        NW_GPU_CHECK(cudaMalloc(&d_neighborFace, topoIntsBytes));
        NW_GPU_CHECK(cudaMalloc(&d_faceBcCode,   topoIntsBytes));
        NW_GPU_CHECK(cudaMalloc(&d_gll_w,        static_cast<size_t>(N) * sizeof(double)));
        NW_GPU_CHECK(cudaMalloc(&d_volIdxMinus,  faceIntsBytes));
        NW_GPU_CHECK(cudaMalloc(&d_volIdxPlus,   faceIntsBytes));
        NW_GPU_CHECK(cudaMalloc(&d_bcType,       faceIntsBytes));
        NW_GPU_CHECK(cudaMalloc(&d_nx,           faceDoublesBytes));
        NW_GPU_CHECK(cudaMalloc(&d_ny,           faceDoublesBytes));
        NW_GPU_CHECK(cudaMalloc(&d_nz,           faceDoublesBytes));
        NW_GPU_CHECK(cudaMalloc(&d_dA,           faceDoublesBytes));
        NW_GPU_CHECK(cudaMalloc(&d_fx,           faceDoublesBytes));
        NW_GPU_CHECK(cudaMalloc(&d_fy,           faceDoublesBytes));
        NW_GPU_CHECK(cudaMalloc(&d_fz,           faceDoublesBytes));

        if (!m_mpiCutGhostIndices.empty()) {
            NW_GPU_CHECK(cudaMalloc(&d_mpiCutGhostIdx, faceIntsBytes));
            NW_GPU_CHECK(cudaMemcpy(d_mpiCutGhostIdx, m_mpiCutGhostIndices.data(), faceIntsBytes, cudaMemcpyHostToDevice));
        }

        NW_GPU_CHECK(cudaMemcpy(d_faceElem,     h_faceElem.data(),     topoIntsBytes, cudaMemcpyHostToDevice));
        NW_GPU_CHECK(cudaMemcpy(d_faceId,       h_faceId.data(),       topoIntsBytes, cudaMemcpyHostToDevice));
        NW_GPU_CHECK(cudaMemcpy(d_neighborElem, h_neighborElem.data(), topoIntsBytes, cudaMemcpyHostToDevice));
        NW_GPU_CHECK(cudaMemcpy(d_neighborFace, h_neighborFace.data(), topoIntsBytes, cudaMemcpyHostToDevice));
        NW_GPU_CHECK(cudaMemcpy(d_faceBcCode,   h_faceBcCode.data(),   topoIntsBytes, cudaMemcpyHostToDevice));
        NW_GPU_CHECK(cudaMemcpy(d_gll_w,        m_gll_w.data(),        static_cast<size_t>(N) * sizeof(double), cudaMemcpyHostToDevice));

        nekwave::device::launch_setup_face_data(
            d_faceElem, d_faceId, d_neighborElem, d_neighborFace, d_faceBcCode,
            d_gll_w,
            d_coordX, d_coordY, d_coordZ,
            d_jac,
            d_rx, d_sx, d_tx,
            d_ry, d_sy, d_ty,
            d_rz, d_sz, d_tz,
            d_volIdxMinus, d_volIdxPlus, d_bcType,
            d_nx, d_ny, d_nz, d_dA,
            d_fx, d_fy, d_fz,
            N, numFaces, totalFacePoints,
            nullptr, d_mpiCutGhostIdx
        );

        std::vector<int> h_volIdxMinus(totalFacePoints);
        std::vector<int> h_volIdxPlus(totalFacePoints);
        std::vector<double> h_nx(totalFacePoints);
        std::vector<double> h_ny(totalFacePoints);
        std::vector<double> h_nz(totalFacePoints);
        std::vector<double> h_dA(totalFacePoints);

        NW_GPU_CHECK(cudaMemcpy(h_volIdxMinus.data(), d_volIdxMinus, faceIntsBytes,    cudaMemcpyDeviceToHost));
        NW_GPU_CHECK(cudaMemcpy(h_volIdxPlus.data(),  d_volIdxPlus,  faceIntsBytes,    cudaMemcpyDeviceToHost));
        NW_GPU_CHECK(cudaMemcpy(h_nx.data(),          d_nx,          faceDoublesBytes, cudaMemcpyDeviceToHost));
        NW_GPU_CHECK(cudaMemcpy(h_ny.data(),          d_ny,          faceDoublesBytes, cudaMemcpyDeviceToHost));
        NW_GPU_CHECK(cudaMemcpy(h_nz.data(),          d_nz,          faceDoublesBytes, cudaMemcpyDeviceToHost));
        NW_GPU_CHECK(cudaMemcpy(h_dA.data(),          d_dA,          faceDoublesBytes, cudaMemcpyDeviceToHost));

        m_faceData.resize(numFaces);
        for (int f = 0; f < numFaces; ++f) {
            ElementFaceData& efd = m_faceData[f];
            efd.elementId = m_faces[f].elementId;
            efd.faceId    = m_faces[f].faceId;
            efd.bcType    = m_faces[f].bcType;
            efd.points.resize(N2);
            for (int p = 0; p < N2; ++p) {
                int idx = f * N2 + p;
                efd.points[p].volIdxMinus = h_volIdxMinus[idx];
                efd.points[p].volIdxPlus  = h_volIdxPlus[idx];
                efd.points[p].nx          = h_nx[idx];
                efd.points[p].ny          = h_ny[idx];
                efd.points[p].nz          = h_nz[idx];
                efd.points[p].dA          = h_dA[idx];
            }
        }

        if (d_mpiCutGhostIdx) NW_GPU_CHECK(cudaFree(d_mpiCutGhostIdx));
        NW_GPU_CHECK(cudaFree(d_faceElem));
        NW_GPU_CHECK(cudaFree(d_faceId));
        NW_GPU_CHECK(cudaFree(d_neighborElem));
        NW_GPU_CHECK(cudaFree(d_neighborFace));
        NW_GPU_CHECK(cudaFree(d_faceBcCode));
        NW_GPU_CHECK(cudaFree(d_gll_w));
        NW_GPU_CHECK(cudaFree(d_volIdxMinus));
        NW_GPU_CHECK(cudaFree(d_volIdxPlus));
        NW_GPU_CHECK(cudaFree(d_bcType));
        NW_GPU_CHECK(cudaFree(d_nx));
        NW_GPU_CHECK(cudaFree(d_ny));
        NW_GPU_CHECK(cudaFree(d_nz));
        NW_GPU_CHECK(cudaFree(d_dA));
        NW_GPU_CHECK(cudaFree(d_fx));
        NW_GPU_CHECK(cudaFree(d_fy));
        NW_GPU_CHECK(cudaFree(d_fz));
    }

    NW_GPU_CHECK(cudaFree(d_corners));
    NW_GPU_CHECK(cudaFree(d_gll_z));
    NW_GPU_CHECK(cudaFree(d_coordX));
    NW_GPU_CHECK(cudaFree(d_coordY));
    NW_GPU_CHECK(cudaFree(d_coordZ));
    NW_GPU_CHECK(cudaFree(d_jac));
    NW_GPU_CHECK(cudaFree(d_rx));
    NW_GPU_CHECK(cudaFree(d_sx));
    NW_GPU_CHECK(cudaFree(d_tx));
    NW_GPU_CHECK(cudaFree(d_ry));
    NW_GPU_CHECK(cudaFree(d_sy));
    NW_GPU_CHECK(cudaFree(d_ty));
    NW_GPU_CHECK(cudaFree(d_rz));
    NW_GPU_CHECK(cudaFree(d_sz));
    NW_GPU_CHECK(cudaFree(d_tz));
    NW_PROFILER_POP();
}

const std::vector<double>& Mesh::getJac() const {
    if (m_jac.empty() && !m_elementCorners.empty()) {
        preprocessOnGpu(true, false);
    }
    return m_jac;
}

const std::vector<double>& Mesh::getCoordX() const {
    if (m_coord_x.empty() && !m_elementCorners.empty()) {
        preprocessOnGpu(true, false);
    }
    return m_coord_x;
}

const std::vector<double>& Mesh::getCoordY() const {
    if (m_coord_y.empty() && !m_elementCorners.empty()) {
        preprocessOnGpu(true, false);
    }
    return m_coord_y;
}

const std::vector<double>& Mesh::getCoordZ() const {
    if (m_coord_z.empty() && !m_elementCorners.empty()) {
        preprocessOnGpu(true, false);
    }
    return m_coord_z;
}

const std::vector<ElementFaceData>& Mesh::getFaceData() const {
    if (m_faceData.empty() && !m_faces.empty()) {
        preprocessOnGpu(false, true);
    }
    return m_faceData;
}

// @@ compute minimum node distance dxmin on the GPU matching NekCEM's get_dxmin
double Mesh::computeMinNodeDistance() const {
    if (m_cachedMinNodeDist <= 0.0 && !m_elementCorners.empty()) {
        preprocessOnGpu(false, false);
    }
    return (m_cachedMinNodeDist > 0.0) ? m_cachedMinNodeDist : 1.0;
}

// @@ automatic stable CFL Courant number based on polynomial order and 3D stability bound
double Mesh::computeAutomaticCFL(double safetyFactor) const {
    const double alpha_RK = 1.75;           // Conservative stability limit with numerical penalty fluxes
    const double sqrt_dim = std::sqrt(3.0);  // 3D spatial dimension
    double cfl = (safetyFactor * alpha_RK) / sqrt_dim;
    return cfl;
}

// @@ time step dt directly from inputs, order N, and mesh size dxmin
// dt = CFL_auto * (dxmin / c)
double Mesh::computeAutomaticDt(double waveSpeed, double safetyFactor) const {
    double dxmin = computeMinNodeDistance();
    double cfl = computeAutomaticCFL(safetyFactor);
    return (cfl * dxmin) / waveSpeed;
}

// @@ volume node index for a point (p, q) on a given face
// Face-to-volume index mapping matching NekCEM's cemface
int Mesh::getFaceNodeVolIndex(int elemId, int faceId, int p, int q) const {
    int i = 0, j = 0, k = 0;
    int N = m_N;
    switch (faceId) {
        case 0: // Face 1 in Nek: s = -1 (y min)
            i = p; j = 0; k = q;
            break;
        case 1: // Face 2 in Nek: r = +1 (x max)
            i = N - 1; j = p; k = q;
            break;
        case 2: // Face 3 in Nek: s = +1 (y max)
            i = p; j = N - 1; k = q;
            break;
        case 3: // Face 4 in Nek: r = -1 (x min)
            i = 0; j = p; k = q;
            break;
        case 4: // Face 5 in Nek: t = -1 (z min)
            i = p; j = q; k = 0;
            break;
        case 5: // Face 6 in Nek: t = +1 (z max)
            i = p; j = q; k = N - 1;
            break;
        default:
            break;
    }
    return elemId * m_numPointsPerElement + i + N * (j + N * k);
}

// @@ initialize UPML topology and conductivity profiles if PML boundary faces exist
void Mesh::setupPml() {
    using Clock = std::chrono::high_resolution_clock;
    auto tStart = Clock::now();
    NW_PROFILER_PUSH("Mesh::setupPml");
    m_pmlData = BoundaryConditions::initializePml(*this, m_pmlConfig);
    NW_PROFILER_POP();
    m_subTimings.pmlInitSec += std::chrono::duration<double>(Clock::now() - tStart).count();
}

// @@ multi-GPU MPI domain decomposition (Recursive Coordinate Bisection)
bool Mesh::partition(int rank, int numRanks) {
    if (numRanks <= 1) {
        m_bIsPartitioned = false;
        m_numHaloPoints = 0;
        m_mpiHalos.clear();
        m_mpiCutGhostIndices.clear();
        return true;
    }

    // Populate global face matching on the GPU so cut-face node pairings are known
    const auto& globalFaceData = getFaceData();

    NW_PROFILER_PUSH("Mesh::partition");
    int totalElems = m_numElements;
    if (totalElems < numRanks) {
        std::cerr << "[MPI ERROR] Total elements (" << totalElems 
                  << ") is less than number of ranks (" << numRanks << ")!" << std::endl;
        NW_PROFILER_POP();
        return false;
    }

    // 1. Calculate centroid for each element
    struct ElemCentroid {
        int elemId;
        double cx, cy, cz;
    };
    std::vector<ElemCentroid> centroids(totalElems);
    double minX = 1e30, maxX = -1e30;
    double minY = 1e30, maxY = -1e30;
    double minZ = 1e30, maxZ = -1e30;

    for (int e = 0; e < totalElems; ++e) {
        double cx = 0.0, cy = 0.0, cz = 0.0;
        for (int v = 0; v < 8; ++v) {
            cx += m_elementCorners[e][v][0];
            cy += m_elementCorners[e][v][1];
            cz += m_elementCorners[e][v][2];
        }
        cx *= 0.125; cy *= 0.125; cz *= 0.125;
        centroids[e] = {e, cx, cy, cz};
        minX = std::min(minX, cx); maxX = std::max(maxX, cx);
        minY = std::min(minY, cy); maxY = std::max(maxY, cy);
        minZ = std::min(minZ, cz); maxZ = std::max(maxZ, cz);
    }

    // Determine principal axis for partitioning (longest domain extent)
    double spanX = maxX - minX;
    double spanY = maxY - minY;
    double spanZ = maxZ - minZ;

    if (spanX >= spanY && spanX >= spanZ) {
        std::sort(centroids.begin(), centroids.end(), [](const ElemCentroid& a, const ElemCentroid& b) {
            return a.cx < b.cx;
        });
    } else if (spanY >= spanX && spanY >= spanZ) {
        std::sort(centroids.begin(), centroids.end(), [](const ElemCentroid& a, const ElemCentroid& b) {
            return a.cy < b.cy;
        });
    } else {
        std::sort(centroids.begin(), centroids.end(), [](const ElemCentroid& a, const ElemCentroid& b) {
            return a.cz < b.cz;
        });
    }

    // 2. Assign elements to ranks
    std::vector<int> elemOwner(totalElems, -1);
    for (int i = 0; i < totalElems; ++i) {
        int assignedRank = (i * numRanks) / totalElems;
        elemOwner[centroids[i].elemId] = assignedRank;
    }

    // Identify local elements for this rank
    std::vector<int> localElemIds;
    std::vector<int> globalToLocalElem(totalElems, -1);
    for (int e = 0; e < totalElems; ++e) {
        if (elemOwner[e] == rank) {
            globalToLocalElem[e] = static_cast<int>(localElemIds.size());
            localElemIds.push_back(e);
        }
    }

    int localNumElems = static_cast<int>(localElemIds.size());
    int nxyz = m_numPointsPerElement;
    int localTotalPoints = localNumElems * nxyz;

    // 3. Extract local element corners
    std::vector<std::array<std::array<double, 3>, 8>> new_corners(localNumElems);
    for (int locE = 0; locE < localNumElems; ++locE) {
        new_corners[locE] = m_elementCorners[localElemIds[locE]];
    }

    // 4. Identify cut faces and build halo exchange structures
    struct CutFaceRef {
        int globE;
        int faceId;
        int nbrGlobE;
        int nbrFaceId;
        int nbrRank;
        size_t faceDataIdx;
    };

    std::map<int, std::vector<CutFaceRef>> rankCutFaces;

    for (size_t fIdx = 0; fIdx < m_faces.size(); ++fIdx) {
        const auto& f = m_faces[fIdx];
        int globE = f.elementId;
        int nbrGlobE = f.neighborElementId;
        if (nbrGlobE >= 0 && elemOwner[globE] != elemOwner[nbrGlobE]) {
            int ownerA = elemOwner[globE];
            int ownerB = elemOwner[nbrGlobE];

            if (ownerA == rank) {
                int otherRank = ownerB;
                rankCutFaces[otherRank].push_back({globE, f.faceId, nbrGlobE, f.neighborFaceId, otherRank, fIdx});
            }
        }
    }

    // Sort cut faces for each neighbor rank canonically using exact integer element & face IDs
    for (auto& kv : rankCutFaces) {
        std::sort(kv.second.begin(), kv.second.end(), [](const CutFaceRef& a, const CutFaceRef& b) {
            int minEA = std::min(a.globE, a.nbrGlobE);
            int minEB = std::min(b.globE, b.nbrGlobE);
            if (minEA != minEB) return minEA < minEB;
            int maxEA = std::max(a.globE, a.nbrGlobE);
            int maxEB = std::max(b.globE, b.nbrGlobE);
            if (maxEA != maxEB) return maxEA < maxEB;
            int canonFA = (a.globE < a.nbrGlobE) ? a.faceId : a.nbrFaceId;
            int canonFB = (b.globE < b.nbrGlobE) ? b.faceId : b.nbrFaceId;
            return canonFA < canonFB;
        });
    }

    // 5. Build MpiHaloExchangeInfo and assign ghost node offsets
    m_mpiHalos.clear();
    int currentGhostOffset = 0;
    std::map<std::pair<size_t, size_t>, int> facePtToGhost;

    for (const auto& kv : rankCutFaces) {
        int nbrRank = kv.first;
        const auto& cutList = kv.second;

        MpiHaloExchangeInfo halo;
        halo.neighborRank = nbrRank;
        halo.ghostOffset = currentGhostOffset;

        for (const auto& cut : cutList) {
            const auto& origPts = globalFaceData[cut.faceDataIdx].points;
            int locE = globalToLocalElem[cut.globE];

            struct PtSort {
                size_t ptIdx;
                int locVolMinus;
                int canonicalVolKey;
            };
            std::vector<PtSort> sortedPts(origPts.size());
            for (size_t p = 0; p < origPts.size(); ++p) {
                int globVM = origPts[p].volIdxMinus;
                int globVP = origPts[p].volIdxPlus;
                int canonKey = (cut.globE < cut.nbrGlobE) ? globVM : globVP;
                int locVM = locE * nxyz + (globVM % nxyz);
                sortedPts[p] = {p, locVM, canonKey};
            }
            std::sort(sortedPts.begin(), sortedPts.end(), [](const PtSort& a, const PtSort& b) {
                if (a.canonicalVolKey != b.canonicalVolKey) return a.canonicalVolKey < b.canonicalVolKey;
                return a.ptIdx < b.ptIdx;
            });

            for (size_t sp = 0; sp < sortedPts.size(); ++sp) {
                halo.sendVolIndices.push_back(sortedPts[sp].locVolMinus);
                int ghostIdx = localTotalPoints + currentGhostOffset;
                facePtToGhost[{cut.faceDataIdx, sortedPts[sp].ptIdx}] = ghostIdx;
                currentGhostOffset++;
            }
        }
        halo.numPoints = static_cast<int>(halo.sendVolIndices.size());
        m_mpiHalos.push_back(halo);
    }
    m_numHaloPoints = currentGhostOffset;

    // 6. Build partitioned m_faces and m_mpiCutGhostIndices for this rank
    std::vector<FaceInfo> new_faces;
    m_mpiCutGhostIndices.clear();
    const int N2 = m_N * m_N;

    for (size_t fIdx = 0; fIdx < m_faces.size(); ++fIdx) {
        const auto& f = m_faces[fIdx];
        int globE = f.elementId;
        if (elemOwner[globE] != rank) continue;

        int locE = globalToLocalElem[globE];
        int globNbr = f.neighborElementId;

        FaceInfo newF = f;
        newF.elementId = locE;

        if (globNbr < 0) {
            newF.neighborElementId = -1;
            newF.neighborFaceId = -1;
            for (int p = 0; p < N2; ++p) m_mpiCutGhostIndices.push_back(-1);
        } else if (elemOwner[globNbr] == rank) {
            newF.neighborElementId = globalToLocalElem[globNbr];
            for (int p = 0; p < N2; ++p) m_mpiCutGhostIndices.push_back(-1);
        } else {
            // MPI Cut Face
            newF.neighborElementId = -1;
            newF.bcType = "MPI";
            for (int p = 0; p < N2; ++p) {
                m_mpiCutGhostIndices.push_back(facePtToGhost[{fIdx, static_cast<size_t>(p)}]);
            }
        }

        new_faces.push_back(newF);
    }

    // 6b. Partition UPML topological data if enabled
    if (m_pmlData.enabled) {
        std::vector<int> new_pmlTag(localNumElems, 0);
        std::vector<int> new_pmlPtr;

        for (int locE = 0; locE < localNumElems; ++locE) {
            int globE = localElemIds[locE];
            int tag = (globE < static_cast<int>(m_pmlData.pmlTag.size())) ? m_pmlData.pmlTag[globE] : 0;
            new_pmlTag[locE] = tag;
            if (tag != 0) {
                new_pmlPtr.push_back(locE);
            }
        }
        m_pmlData.pmlTag = std::move(new_pmlTag);
        m_pmlData.pmlPtr = std::move(new_pmlPtr);
        m_pmlData.maxPml = static_cast<int>(m_pmlData.pmlPtr.size());
    }

    // 7. Commit partitioned mesh state (metrics/coords will be evaluated on GPU for the local partition)
    m_numElements = localNumElems;
    m_totalPoints = localTotalPoints;
    m_elementCorners = std::move(new_corners);
    m_faces = std::move(new_faces);
    invalidateGeometryCache();

    m_bIsPartitioned = true;

    std::cout << "[MPI] Rank " << rank << " mesh partitioned: " 
              << m_numElements << " of " << totalElems << " elements, "
              << m_totalPoints << " local nodes, "
              << m_numHaloPoints << " halo ghost nodes across "
              << m_mpiHalos.size() << " neighbor rank(s)." << std::endl;

    NW_PROFILER_POP();
    return true;
}
