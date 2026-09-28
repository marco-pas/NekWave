// @@ Note
// The .rea mesh parsing logic in this file is taken 1-to-1 from the NekCEM / Nek5000
// In future Neko's .nmsh format or native Gmsh)

#include "mesh.hpp"
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <map>
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
      m_numElements(numElements) {
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
    setupGLL();
}

// @@ GLL collocation points, weights, and derivative matrix setup
void Mesh::setupGLL() {
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

        // @@ build 1D derivative matrix D and its transpose Dt (NekCEM DGLL)
        // D_ij = L_p(x_i) / (L_p(x_j) * (x_i - x_j))
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
}

// @@ configuring an element with an affine bounding box transformation
// AFFINE transformation here (!)
// J_aff = diag(dx/2, dy/2, dz/2), det(J_aff) = (dx * dy * dz) / 8, rx = 2/dx, sy = 2/dy, tz = 2/dz
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

    computeMetricsFromCorners();

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
    setupFaceData();
}

// @@ computing general trilinear hex metrics and physical coordinates from element corners
// General Trilinear Coordinate Transformation
// x(r,s,t) = sum_{a=0}^7 N_a(r,s,t) * v_a
void Mesh::computeMetricsFromCorners() {
    m_coord_x.assign(m_totalPoints, 0.0);
    m_coord_y.assign(m_totalPoints, 0.0);
    m_coord_z.assign(m_totalPoints, 0.0);

    m_rx.assign(m_totalPoints, 0.0);
    m_sx.assign(m_totalPoints, 0.0);
    m_tx.assign(m_totalPoints, 0.0);
    m_ry.assign(m_totalPoints, 0.0);
    m_sy.assign(m_totalPoints, 0.0);
    m_ty.assign(m_totalPoints, 0.0);
    m_rz.assign(m_totalPoints, 0.0);
    m_sz.assign(m_totalPoints, 0.0);
    m_tz.assign(m_totalPoints, 0.0);
    m_jac.assign(m_totalPoints, 1.0);

    int N = m_N;

    for (int e = 0; e < m_numElements; ++e) {
        const auto& corners = m_elementCorners[e];
        int elemOffset = e * m_numPointsPerElement;

        for (int k = 0; k < N; ++k) {
            double t = m_gll_z[k];
            for (int j = 0; j < N; ++j) {
                double s = m_gll_z[j];
                for (int i = 0; i < N; ++i) {
                    double r = m_gll_z[i];
                    int ptIdx = elemOffset + i + N * (j + N * k);

                    // Trilinear shape functions N_a(r, s, t)
                    double N_fn[8];
                    N_fn[0] = 0.125 * (1.0 - r) * (1.0 - s) * (1.0 - t);
                    N_fn[1] = 0.125 * (1.0 + r) * (1.0 - s) * (1.0 - t);
                    N_fn[2] = 0.125 * (1.0 + r) * (1.0 + s) * (1.0 - t);
                    N_fn[3] = 0.125 * (1.0 - r) * (1.0 + s) * (1.0 - t);
                    N_fn[4] = 0.125 * (1.0 - r) * (1.0 - s) * (1.0 + t);
                    N_fn[5] = 0.125 * (1.0 + r) * (1.0 - s) * (1.0 + t);
                    N_fn[6] = 0.125 * (1.0 + r) * (1.0 + s) * (1.0 + t);
                    N_fn[7] = 0.125 * (1.0 - r) * (1.0 + s) * (1.0 + t);

                    // Derivatives of shape functions w.r.t r, s, t
                    double dN_dr[8], dN_ds[8], dN_dt[8];
                    dN_dr[0] = -0.125 * (1.0 - s) * (1.0 - t);
                    dN_dr[1] =  0.125 * (1.0 - s) * (1.0 - t);
                    dN_dr[2] =  0.125 * (1.0 + s) * (1.0 - t);
                    dN_dr[3] = -0.125 * (1.0 + s) * (1.0 - t);
                    dN_dr[4] = -0.125 * (1.0 - s) * (1.0 + t);
                    dN_dr[5] =  0.125 * (1.0 - s) * (1.0 + t);
                    dN_dr[6] =  0.125 * (1.0 + s) * (1.0 + t);
                    dN_dr[7] = -0.125 * (1.0 + s) * (1.0 + t);

                    dN_ds[0] = -0.125 * (1.0 - r) * (1.0 - t);
                    dN_ds[1] = -0.125 * (1.0 + r) * (1.0 - t);
                    dN_ds[2] =  0.125 * (1.0 + r) * (1.0 - t);
                    dN_ds[3] =  0.125 * (1.0 - r) * (1.0 - t);
                    dN_ds[4] = -0.125 * (1.0 - r) * (1.0 + t);
                    dN_ds[5] = -0.125 * (1.0 + r) * (1.0 + t);
                    dN_ds[6] =  0.125 * (1.0 + r) * (1.0 + t);
                    dN_ds[7] =  0.125 * (1.0 - r) * (1.0 + t);

                    dN_dt[0] = -0.125 * (1.0 - r) * (1.0 - s);
                    dN_dt[1] = -0.125 * (1.0 + r) * (1.0 - s);
                    dN_dt[2] = -0.125 * (1.0 + r) * (1.0 + s);
                    dN_dt[3] = -0.125 * (1.0 - r) * (1.0 + s);
                    dN_dt[4] =  0.125 * (1.0 - r) * (1.0 - s);
                    dN_dt[5] =  0.125 * (1.0 + r) * (1.0 - s);
                    dN_dt[6] =  0.125 * (1.0 + r) * (1.0 + s);
                    dN_dt[7] =  0.125 * (1.0 - r) * (1.0 + s);

                    // Compute physical position and Jacobian entries
                    double x = 0.0, y = 0.0, z = 0.0;
                    double xr = 0.0, yr = 0.0, zr = 0.0;
                    double xs = 0.0, ys = 0.0, zs = 0.0;
                    double xt = 0.0, yt = 0.0, zt = 0.0;

                    for (int a = 0; a < 8; ++a) {
                        double vx = corners[a][0];
                        double vy = corners[a][1];
                        double vz = corners[a][2];

                        x += N_fn[a] * vx;
                        y += N_fn[a] * vy;
                        z += N_fn[a] * vz;

                        xr += dN_dr[a] * vx;
                        yr += dN_dr[a] * vy;
                        zr += dN_dr[a] * vz;

                        xs += dN_ds[a] * vx;
                        ys += dN_ds[a] * vy;
                        zs += dN_ds[a] * vz;

                        xt += dN_dt[a] * vx;
                        yt += dN_dt[a] * vy;
                        zt += dN_dt[a] * vz;
                    }

                    m_coord_x[ptIdx] = x;
                    m_coord_y[ptIdx] = y;
                    m_coord_z[ptIdx] = z;

                    // Jacobian determinant J = det(J_aff)
                    double detJ = xr * (ys * zt - yt * zs)
                                - xs * (yr * zt - yt * zr)
                                + xt * (yr * zs - ys * zr);

                    m_jac[ptIdx] = detJ;

                    // Inverse transpose metric tensor J^{-T}
                    double invJ = 1.0 / detJ;
                    m_rx[ptIdx] = invJ * (ys * zt - yt * zs);
                    m_sx[ptIdx] = invJ * (yt * zr - yr * zt);
                    m_tx[ptIdx] = invJ * (yr * zs - ys * zr);

                    m_ry[ptIdx] = invJ * (xt * zs - xs * zt);
                    m_sy[ptIdx] = invJ * (xr * zt - xt * zr);
                    m_ty[ptIdx] = invJ * (xs * zr - xr * zs);

                    m_rz[ptIdx] = invJ * (xs * yt - xt * ys);
                    m_sz[ptIdx] = invJ * (xt * yr - xr * yt);
                    m_tz[ptIdx] = invJ * (xr * ys - xs * yr);
                }
            }
        }
    }
}

// @@ .rea file parser taken 1-to-1 from NekCEM / Nek5000
bool Mesh::loadFromRea(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        // Fallback: check ../<filename> if executed from build/ subdirectory
        std::ifstream fileFallback("../" + filename);
        if (fileFallback.is_open()) {
            fileFallback.close();
            return loadFromRea("../" + filename);
        }
        std::cerr << "Error: Could not open .rea file: " << filename << std::endl;
        return false;
    }

    std::string line;
    bool foundMesh = false;
    int nel = 0, ndim = 3;

    // @@ scanning the file for the **MESH DATA** section
    while (std::getline(file, line)) {
        if (line.find("**MESH DATA**") != std::string::npos) {
            foundMesh = true;
            break;
        }
    }

    if (!foundMesh) {
        std::cerr << "Error: Could not find **MESH DATA** section in " << filename << std::endl;
        return false;
    }

    // @@ reading the line with NEL, NDIM
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        ss >> nel >> ndim;
    }

    if (nel < 0) {
        std::cout << "[MESH] Negative NEL = " << nel << " indicates binary .re2 mesh. Redirecting to loadFromRe2..." << std::endl;
        std::string re2name = filename;
        size_t dotPos = re2name.rfind('.');
        if (dotPos != std::string::npos) {
            re2name = re2name.substr(0, dotPos) + ".re2";
        } else {
            re2name += ".re2";
        }
        return loadFromRe2(re2name, std::abs(nel));
    }

    if (nel == 0) {
        std::cerr << "Error: Invalid number of elements (NEL = " << nel << ") in " << filename << std::endl;
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
            info.bcType = bcType;
            info.neighborElementId = (nbrElem > 0.0) ? static_cast<int>(nbrElem) - 1 : -1;
            info.neighborFaceId = (nbrFace > 0.0) ? static_cast<int>(nbrFace) - 1 : -1;

            m_faces.push_back(info);
        }
    }

    // @@ setting up GLL basis and evaluating metrics
    setupGLL();
    computeMetricsFromCorners();
    setupFaceData();

    return true;
}

// @@ load mesh directly from a NekCEM binary .re2 file
bool Mesh::loadFromRe2(const std::string& filename, int nel) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::ifstream fileFallback("../" + filename, std::ios::binary);
        if (fileFallback.is_open()) {
            fileFallback.close();
            return loadFromRe2("../" + filename, nel);
        }
        std::cerr << "Error: Could not open .re2 file: " << filename << std::endl;
        return false;
    }

    // Read 80-byte header
    char header[81] = {0};
    file.read(header, 80);
    if (!file) {
        std::cerr << "Error: Failed to read header from " << filename << std::endl;
        return false;
    }

    // Read 4-byte endianness test tag (6.54321)
    float tag = 0.0f;
    file.read(reinterpret_cast<char*>(&tag), sizeof(float));
    if (!file) {
        std::cerr << "Error: Failed to read tag from " << filename << std::endl;
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
        return false;
    }

    m_numElements = nel;
    m_totalPoints = m_numElements * m_numPointsPerElement;
    m_elementCorners.resize(m_numElements);

    // Read elements: group (double) + 8 x (double) + 8 y (double) + 8 z (double)
    for (int e = 0; e < nel; ++e) {
        double group = 0.0;
        double x[8], y[8], z[8];
        file.read(reinterpret_cast<char*>(&group), sizeof(double));
        file.read(reinterpret_cast<char*>(x), 8 * sizeof(double));
        file.read(reinterpret_cast<char*>(y), 8 * sizeof(double));
        file.read(reinterpret_cast<char*>(z), 8 * sizeof(double));

        for (int a = 0; a < 8; ++a) {
            m_elementCorners[e][a] = {x[a], y[a], z[a]};
        }
    }

    // Read curved faces count
    double ncurvD = 0.0;
    file.read(reinterpret_cast<char*>(&ncurvD), sizeof(double));

    // Read boundary conditions count
    double nbcD = 0.0;
    file.read(reinterpret_cast<char*>(&nbcD), sizeof(double));
    int nbc = static_cast<int>(nbcD);

    // Map: (elemId, faceId) -> FaceInfo
    std::map<std::pair<int, int>, FaceInfo> bcMap;
    for (int i = 0; i < nbc; ++i) {
        double elemD = 0.0, faceD = 0.0, nbrElemD = 0.0, nbrFaceD = 0.0;
        double p1 = 0.0, p2 = 0.0, p3 = 0.0;
        char tagStr[9] = {0};

        file.read(reinterpret_cast<char*>(&elemD), sizeof(double));
        file.read(reinterpret_cast<char*>(&faceD), sizeof(double));
        file.read(reinterpret_cast<char*>(&nbrElemD), sizeof(double));
        file.read(reinterpret_cast<char*>(&nbrFaceD), sizeof(double));
        file.read(reinterpret_cast<char*>(&p1), sizeof(double));
        file.read(reinterpret_cast<char*>(&p2), sizeof(double));
        file.read(reinterpret_cast<char*>(&p3), sizeof(double));
        file.read(tagStr, 8);

        int e = static_cast<int>(elemD) - 1;
        int f = static_cast<int>(faceD) - 1;
        int nbrE = (nbrElemD > 0.0) ? (static_cast<int>(nbrElemD) - 1) : -1;
        int nbrF = (nbrFaceD > 0.0) ? (static_cast<int>(nbrFaceD) - 1) : -1;

        std::string rawTag(tagStr);
        std::string bcType = "PEC";
        if (rawTag.find('P') != std::string::npos) {
            bcType = "PERIODIC";
        } else if (rawTag.find('E') != std::string::npos) {
            bcType = "E";
        } else if (rawTag.find('W') != std::string::npos || rawTag.find("PEC") != std::string::npos) {
            bcType = "PEC";
        } else if (nbrE >= 0) {
            bcType = "PERIODIC";
        }

        FaceInfo info;
        info.elementId = e;
        info.faceId = f;
        info.bcType = bcType;
        info.neighborElementId = nbrE;
        info.neighborFaceId = nbrF;
        bcMap[{e, f}] = info;
    }

    // Hex face corner definitions matching NekWave / NekCEM:
    // Face 0 (s=-1): 0, 1, 5, 4
    // Face 1 (r=+1): 1, 2, 6, 5
    // Face 2 (s=+1): 2, 3, 7, 6
    // Face 3 (r=-1): 3, 0, 4, 7
    // Face 4 (t=-1): 0, 3, 2, 1
    // Face 5 (t=+1): 4, 5, 6, 7
    const int faceCorners[6][4] = {
        {0, 1, 5, 4},
        {1, 2, 6, 5},
        {2, 3, 7, 6},
        {3, 0, 4, 7},
        {0, 3, 2, 1},
        {4, 5, 6, 7}
    };

    // Index face centroids for interior conforming faces
    std::map<std::tuple<long long, long long, long long>, std::pair<int, int>> interiorCenters;
    auto makeKey = [](double cx, double cy, double cz) {
        long long ix = static_cast<long long>(std::round(cx * 100000.0));
        long long iy = static_cast<long long>(std::round(cy * 100000.0));
        long long iz = static_cast<long long>(std::round(cz * 100000.0));
        return std::make_tuple(ix, iy, iz);
    };

    m_faces.clear();
    m_faces.resize(nel * 6);

    for (int e = 0; e < nel; ++e) {
        for (int f = 0; f < 6; ++f) {
            int idx = e * 6 + f;
            auto it = bcMap.find({e, f});
            if (it != bcMap.end()) {
                m_faces[idx] = it->second;
            } else {
                // Interior face: match neighbor via centroid
                double cx = 0.0, cy = 0.0, cz = 0.0;
                for (int v = 0; v < 4; ++v) {
                    int cIdx = faceCorners[f][v];
                    cx += m_elementCorners[e][cIdx][0] * 0.25;
                    cy += m_elementCorners[e][cIdx][1] * 0.25;
                    cz += m_elementCorners[e][cIdx][2] * 0.25;
                }
                auto key = makeKey(cx, cy, cz);
                auto cIt = interiorCenters.find(key);
                if (cIt != interiorCenters.end()) {
                    int otherE = cIt->second.first;
                    int otherF = cIt->second.second;
                    int otherIdx = otherE * 6 + otherF;

                    FaceInfo infoThis;
                    infoThis.elementId = e;
                    infoThis.faceId = f;
                    infoThis.bcType = "E";
                    infoThis.neighborElementId = otherE;
                    infoThis.neighborFaceId = otherF;
                    m_faces[idx] = infoThis;

                    FaceInfo infoOther;
                    infoOther.elementId = otherE;
                    infoOther.faceId = otherF;
                    infoOther.bcType = "E";
                    infoOther.neighborElementId = e;
                    infoOther.neighborFaceId = f;
                    m_faces[otherIdx] = infoOther;
                } else {
                    interiorCenters[key] = {e, f};
                }
            }
        }
    }

    setupGLL();
    computeMetricsFromCorners();
    setupFaceData();
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

    computeMetricsFromCorners();
    setupFaceData();
}

// @@ generate structured Cartesian box mesh with nelx * nely * nelz hexahedral elements
// Exact element corner calculation and interface connectivity
bool Mesh::createBoxMesh(int nelx, int nely, int nelz,
                         double xmin, double xmax,
                         double ymin, double ymax,
                         double zmin, double zmax,
                         bool periodicX, bool periodicY, bool periodicZ) {
    if (nelx < 1 || nely < 1 || nelz < 1) return false;

    m_numElements = nelx * nely * nelz;
    m_totalPoints = m_numElements * m_numPointsPerElement;
    setupGLL();

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
                    f0.bcType = "PEC";
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
                    f1.bcType = "PEC";
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
                    f2.bcType = "PEC";
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
                    f3.bcType = "PEC";
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
                    f4.bcType = "PEC";
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
                    f5.bcType = "PEC";
                    f5.neighborElementId = -1; f5.neighborFaceId = -1;
                }
                m_faces.push_back(f5);
            }
        }
    }

    // Evaluate metric factor tensors and face trace connections
    computeMetricsFromCorners();
    setupFaceData();

    return true;
}

// @@ the default mesh initialization routine
// %% METHOD STEP 1D: Global Mesh Initialization (GLL Setup + Default Metric Factors)
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

    computeMetricsFromCorners();

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
    setupFaceData();
}

// @@ compute minimum node distance dxmin matching NekCEM's get_dxmin
// Finds smallest spacing between adjacent GLL nodes across all hex elements --> CFL
double Mesh::computeMinNodeDistance() const {
    double minD2 = 1.0e20;
    int N = m_N;

    for (int e = 0; e < m_numElements; ++e) {
        int offset = e * m_numPointsPerElement;

        for (int k = 0; k < N; ++k) {
            for (int j = 0; j < N; ++j) {
                for (int i = 0; i < N; ++i) {
                    int idx = offset + i + N * (j + N * k);
                    double x0 = m_coord_x[idx];
                    double y0 = m_coord_y[idx];
                    double z0 = m_coord_z[idx];

                    // Check distance along r (i + 1)
                    if (i + 1 < N) {
                        int idx_r = offset + (i + 1) + N * (j + N * k);
                        double dx = m_coord_x[idx_r] - x0;
                        double dy = m_coord_y[idx_r] - y0;
                        double dz = m_coord_z[idx_r] - z0;
                        minD2 = std::min(minD2, dx * dx + dy * dy + dz * dz);
                    }

                    // Check distance along s (j + 1)
                    if (j + 1 < N) {
                        int idx_s = offset + i + N * ((j + 1) + N * k);
                        double dx = m_coord_x[idx_s] - x0;
                        double dy = m_coord_y[idx_s] - y0;
                        double dz = m_coord_z[idx_s] - z0;
                        minD2 = std::min(minD2, dx * dx + dy * dy + dz * dz);
                    }

                    // Check distance along t (k + 1)
                    if (k + 1 < N) {
                        int idx_t = offset + i + N * (j + N * (k + 1));
                        double dx = m_coord_x[idx_t] - x0;
                        double dy = m_coord_y[idx_t] - y0;
                        double dz = m_coord_z[idx_t] - z0;
                        minD2 = std::min(minD2, dx * dx + dy * dy + dz * dz);
                    }
                }
            }
        }
    }

    return std::sqrt(minD2);
}

// @@ automatic stable CFL Courant number based on polynomial order and 3D stability bound
// In 3D DG-SEM with LSRK45, the spatial operator eigenvalue bound is:
// lambda_max <= sqrt(d) * c / dxmin.
// For Carpenter-Kennedy LSRK45, the stability limit on the imaginary axis is alpha_RK45 ~ 1.75 to 2.82.
// With safety factor S (default 0.35 - 0.4 for DG with boundary penalty terms):
//      CFL_auto = S * (alpha_RK45 / sqrt(3)) ~ 0.20 - 0.25
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

// @@ computing outward physical normal vectors, face quadrature weights, and neighbor node connectivity
// PIOLA transform & NANSON formula: n * dA = J * J^{-T} * n_ref * dA_ref
void Mesh::setupFaceData() {
    m_faceData.clear();
    m_faceData.reserve(m_faces.size());

    int N = m_N;
    for (const auto& face : m_faces) {
        ElementFaceData efd;
        efd.elementId = face.elementId;
        efd.faceId = face.faceId;
        efd.bcType = face.bcType;
        efd.points.reserve(N * N);

        int eA = face.elementId;
        int fA = face.faceId;

        for (int qA = 0; qA < N; ++qA) {
            for (int pA = 0; pA < N; ++pA) {
                FacePointData pt;
                pt.volIdxMinus = getFaceNodeVolIndex(eA, fA, pA, qA);
                int vM = pt.volIdxMinus;
                double J = m_jac[vM];
                double w2 = m_gll_w[pA] * m_gll_w[qA];

                // Outward normal in reference coordinates n_ref transformed via cofactor matrix J * J^{-T}
                double Nx = 0.0, Ny = 0.0, Nz = 0.0;
                switch (fA) {
                    case 0: // s = -1: n_ref = (0, -1, 0)
                        Nx = -J * m_sx[vM]; Ny = -J * m_sy[vM]; Nz = -J * m_sz[vM];
                        break;
                    case 1: // r = +1: n_ref = (1, 0, 0)
                        Nx = +J * m_rx[vM]; Ny = +J * m_ry[vM]; Nz = +J * m_rz[vM];
                        break;
                    case 2: // s = +1: n_ref = (0, 1, 0)
                        Nx = +J * m_sx[vM]; Ny = +J * m_sy[vM]; Nz = +J * m_sz[vM];
                        break;
                    case 3: // r = -1: n_ref = (-1, 0, 0)
                        Nx = -J * m_rx[vM]; Ny = -J * m_ry[vM]; Nz = -J * m_rz[vM];
                        break;
                    case 4: // t = -1: n_ref = (0, 0, -1)
                        Nx = -J * m_tx[vM]; Ny = -J * m_ty[vM]; Nz = -J * m_tz[vM];
                        break;
                    case 5: // t = +1: n_ref = (0, 0, 1)
                        Nx = +J * m_tx[vM]; Ny = +J * m_ty[vM]; Nz = +J * m_tz[vM];
                        break;
                }

                double L = std::sqrt(Nx * Nx + Ny * Ny + Nz * Nz);
                pt.nx = (L > 1e-14) ? (Nx / L) : 0.0;
                pt.ny = (L > 1e-14) ? (Ny / L) : 0.0;
                pt.nz = (L > 1e-14) ? (Nz / L) : 0.0;
                pt.dA = L * w2;

                // Determine exterior neighbor node index
                if (face.bcType == "PEC" || face.neighborElementId < 0) {
                    pt.volIdxPlus = -1;
                } else if (face.bcType == "PERIODIC") {
                    // Periodic boundary: aligned opposite Cartesian faces map directly by (pA, qA)
                    int eB = face.neighborElementId;
                    int fB = face.neighborFaceId;
                    pt.volIdxPlus = getFaceNodeVolIndex(eB, fB, pA, qA);
                } else {
                    int eB = face.neighborElementId;
                    int fB = face.neighborFaceId;
                    double xA = m_coord_x[vM];
                    double yA = m_coord_y[vM];
                    double zA = m_coord_z[vM];

                    int matchedVolB = -1;
                    for (int qB = 0; qB < N; ++qB) {
                        for (int pB = 0; pB < N; ++pB) {
                            int vB = getFaceNodeVolIndex(eB, fB, pB, qB);
                            double dx = xA - m_coord_x[vB];
                            double dy = yA - m_coord_y[vB];
                            double dz = zA - m_coord_z[vB];
                            if (dx * dx + dy * dy + dz * dz < 1e-10) {
                                matchedVolB = vB;
                                break;
                            }
                        }
                        if (matchedVolB >= 0) break;
                    }
                    pt.volIdxPlus = matchedVolB;
                }

                efd.points.push_back(pt);
            }
        }
        m_faceData.push_back(efd);
    }
}
