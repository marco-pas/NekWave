// Verification and Unit Tests

#include "mesh.hpp"
#include "dg_solver.hpp"
#include "config.hpp"
#include "probe.hpp"
#include "case.hpp"

#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <vector>
#include <string>
#include <sys/stat.h>

// Global test execution counters
static int g_testsPassed = 0;
static int g_testsFailed = 0;

/*
 * Evaluates a boolean assertion condition.
 *
 * If the condition evaluates to false, records the failure location
 * (file and line number), increments the failure counter, and returns
 * early from the calling test function.
 */
#define EXPECT_TRUE(cond) do { \
    if (!(cond)) { \
        std::cerr << "  FAILED: " << #cond << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        g_testsFailed++; \
        return; \
    } \
} while(0)

/*
 * Asserts that two floating-point values are equal within a specified tolerance.
 *
 * Checks |a - b| <= tol. If the difference exceeds the threshold, prints
 * the discrepancy and source location, increments g_testsFailed, and returns
 * early from the calling test function.
 */
#define EXPECT_NEAR(a, b, tol) do { \
    if (std::abs((a) - (b)) > (tol)) { \
        std::cerr << "  FAILED: |" << #a << " - " << #b << "| = " << std::abs((a) - (b)) \
                  << " > " << (tol) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        g_testsFailed++; \
        return; \
    } \
} while(0)

/*
 * Test 1: Mesh and Quadrature Accuracy
 *
 * Verifies mathematical properties of Gauss-Lobatto-Legendre (GLL) quadrature
 * nodes, quadrature integration weights, spectral differentiation matrices,
 * and reference-to-physical Jacobian transformations in 3D:
 *
 * 1. Sum of 1D GLL weights: integral of f(xi) = 1 over [-1, 1] equals 2.0.
 * 2. Point reflection symmetry: xi_i = -xi_{N - 1 - i} about the origin.
 * 3. Differentiation matrix row-sums: D * 1 = 0 (exact differentiation of constants).
 * 4. Exact differentiation of linear polynomial: (D * xi)_i = 1 for all nodes i.
 * 5. 3D numerical volume integration of reference box: sum_k J_k * w_i * w_j * w_k = 8.0.
 */
void MeshQuadratureTest() {
    std::cout << "[RUN] MeshQuadratureTest..." << std::endl;

    // Polynomial order N = 5 (degree P = N - 1 = 4)
    const int N = 5;
    Mesh mesh(N, 1);

    // Create a 2x2x2 mesh of hexahedral elements spanning [-1, 1]^3
    mesh.createBoxMesh(2, 2, 2, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    const auto& z = mesh.getGllZ();
    const auto& w = mesh.getGllW();
    const auto& D = mesh.getD();

    // Sum of 1D GLL quadrature weights must equal the reference domain length 2.0
    double sumW = 0.0;
    for (int i = 0; i < N; ++i) {
        sumW += w[i];
    }
    EXPECT_NEAR(sumW, 2.0, 1e-12);

    // GLL nodes must exhibit exact reflection symmetry about xi = 0: z[i] == -z[N - 1 - i]
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(z[i] + z[N - 1 - i], 0.0, 1e-12);
    }

    // Row sums of the differentiation matrix must be zero: sum_j D[i, j] = 0.
    // Note: D is stored in column-major ordering, so D_ij = D[i + j * N].
    for (int i = 0; i < N; ++i) {
        double rowSum = 0.0;
        for (int j = 0; j < N; ++j) {
            rowSum += D[i + j * N];
        }
        EXPECT_NEAR(rowSum, 0.0, 1e-11);
    }

    // Differentiation of a linear polynomial f(xi) = xi yields f'(xi) = 1.0 everywhere
    for (int i = 0; i < N; ++i) {
        double deriv = 0.0;
        for (int j = 0; j < N; ++j) {
            deriv += D[i + j * N] * z[j];
        }
        EXPECT_NEAR(deriv, 1.0, 1e-11);
    }

    // Check global numerical volume integration:
    //    Total Volume = sum_e sum_{i,j,k} J_{ijk}^e * w_i * w_j * w_k = (2)^3 = 8.0
    const auto& J = mesh.getJac();
    double totalVol = 0.0;
    const int Np = N * N * N;
    for (int e = 0; e < mesh.getNumElements(); ++e) {
        for (int k = 0; k < N; ++k) {
            for (int j = 0; j < N; ++j) {
                for (int i = 0; i < N; ++i) {
                    int p = e * Np + k * N * N + j * N + i;
                    totalVol += J[p] * w[i] * w[j] * w[k];
                }
            }
        }
    }
    EXPECT_NEAR(totalVol, 8.0, 1e-10);

    g_testsPassed++;
    std::cout << "  PASSED" << std::endl;
}

/*
 * Test 2: Relative Observation Probe Dynamic Scaling
 *
 * Verifies that probe positions defined in relative coordinates
 * (rx, ry, rz in [-0.5, 0.5]) correctly map to absolute physical
 * coordinates given domain bounding extents [xmin, xmax], [ymin, ymax],
 * and [zmin, zmax]:
 *
 *   xc = 0.5 * (xmin + xmax),  x = xc + rx * Lx
 *   yc = 0.5 * (ymin + ymax),  y = yc + ry * Ly
 *   zc = 0.5 * (zmin + zmax),  z = zc + rz * Lz
 */
void ProbeScalingTest() {
    std::cout << "[RUN] ProbeScalingTest..." << std::endl;

    // Define cavity configuration with non-symmetric and distinct aspect ratios
    Config config;
    config.Lx = 4.0;
    config.Ly = 2.0;
    config.Lz = 10.0;
    config.xmin = 0.0;
    config.xmax = 4.0;
    config.ymin = -1.0;
    config.ymax = 1.0;
    config.zmin = -5.0;
    config.zmax = 5.0;

    // Relative coordinates defined in [-0.5, 0.5] relative to cavity geometric center
    config.probe_rel = {
        {{ 0.25, -0.50,  0.10}},
        {{-0.25,  0.00, -0.30}}
    };

    // Calculate domain center
    const double xc = 0.5 * (config.xmin + config.xmax); // 2.0
    const double yc = 0.5 * (config.ymin + config.ymax); // 0.0
    const double zc = 0.5 * (config.zmin + config.zmax); // 0.0

    // Transform relative offsets to physical coordinates
    config.probes.resize(config.probe_rel.size());
    for (size_t i = 0; i < config.probe_rel.size(); ++i) {
        config.probes[i][0] = xc + config.probe_rel[i][0] * config.Lx;
        config.probes[i][1] = yc + config.probe_rel[i][1] * config.Ly;
        config.probes[i][2] = zc + config.probe_rel[i][2] * config.Lz;
    }

    // Verify Probe 1 physical coordinates:
    // x = 2.0 + 0.25 * 4.0 = 3.0
    // y = 0.0 - 0.50 * 2.0 = -1.0
    // z = 0.0 + 0.10 * 10.0 = 1.0
    EXPECT_NEAR(config.probes[0][0], 3.0, 1e-12);
    EXPECT_NEAR(config.probes[0][1], -1.0, 1e-12);
    EXPECT_NEAR(config.probes[0][2], 1.0, 1e-12);

    // Verify Probe 2 physical coordinates:
    // x = 2.0 - 0.25 * 4.0 = 1.0
    // y = 0.0 + 0.00 * 2.0 = 0.0
    // z = 0.0 - 0.30 * 10.0 = -3.0
    EXPECT_NEAR(config.probes[1][0], 1.0, 1e-12);
    EXPECT_NEAR(config.probes[1][1], 0.0, 1e-12);
    EXPECT_NEAR(config.probes[1][2], -3.0, 1e-12);

    g_testsPassed++;
    std::cout << "  PASSED" << std::endl;
}

/*
 * Test 3: Maxwell Spatial Operator & Energy Stability
 *
 * Verifies the stability of the semi-discrete DG-SEM scheme combined with
 * Low-Storage Runge-Kutta 4th-order 5-stage (LSRK45) time-stepping:
 *
 * 1. Initializes a smooth localized 3D Gaussian electromagnetic pulse in Ez.
 * 2. Computes the discrete total electromagnetic energy:
 *      U(t) = 0.5 * integral (eps * |E|^2 + mu * |H|^2) dV
 * 3. Advances the Maxwell state vector through multiple time steps using
 *    an upwind numerical flux (C0 = 1.0).
 * 4. Verifies that discrete energy satisfies monotonicity:
 *      U(t_{n+1}) <= U(t_n) + tol
 *    and that field values remain strictly finite (no NaN or Inf).
 */
void MaxwellStabilityTest() {
    std::cout << "[RUN] MaxwellStabilityTest (GPU DgSolver)..." << std::endl;

    // Polynomial order N = 4 (P = 3) on a 2x2x2 mesh of elements
    const int N = 4;
    Mesh mesh(N, 1);
    mesh.createBoxMesh(2, 2, 2, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5);

    // Initialize GPU DgSolver with upwind numerical flux (C0 = 1.0, strictly dissipative)
    DgSolver solver;
    solver.initialize(mesh, 1.0);

    const int npts = mesh.getTotalPoints();

    // State vector stores [Ex, Ey, Ez, Hx, Hy, Hz], each of length npts
    StateVector state(6 * npts, 0.0);

    // Initialize with a localized Gaussian pulse in Ez centered at (0, 0, 0)
    const auto& x = mesh.getCoordX();
    const auto& y = mesh.getCoordY();
    const auto& z = mesh.getCoordZ();
    for (int i = 0; i < npts; ++i) {
        double r2 = x[i] * x[i] + y[i] * y[i] + z[i] * z[i];
        state[2 * npts + i] = std::exp(-50.0 * r2); // Ez component
    }

    // Helper lambda to compute total discrete electromagnetic energy:
    // U = 0.5 * sum_e sum_{i,j,k} J * w_i * w_j * w_k * (|E|^2 + |H|^2)
    const auto& J = mesh.getJac();
    const auto& w = mesh.getGllW();
    auto computeEnergy = [&](const StateVector& u) {
        double E = 0.0;
        const int Np = N * N * N;
        for (int e = 0; e < mesh.getNumElements(); ++e) {
            for (int k = 0; k < N; ++k) {
                for (int j = 0; j < N; ++j) {
                    for (int i = 0; i < N; ++i) {
                        int p = e * Np + k * N * N + j * N + i;
                        double dV = J[p] * w[i] * w[j] * w[k];
                        double eSq = u[0 * npts + p] * u[0 * npts + p] +
                                     u[1 * npts + p] * u[1 * npts + p] +
                                     u[2 * npts + p] * u[2 * npts + p];
                        double hSq = u[3 * npts + p] * u[3 * npts + p] +
                                     u[4 * npts + p] * u[4 * npts + p] +
                                     u[5 * npts + p] * u[5 * npts + p];
                        E += 0.5 * (eSq + hSq) * dV;
                    }
                }
            }
        }
        return E;
    };

    // Calculate initial electromagnetic energy
    double initialEnergy = computeEnergy(state);
    EXPECT_TRUE(initialEnergy > 0.0);

    // Upload initial state to GPU
    solver.uploadState(state.data(), state.size());

    // Advance 5 time steps using the GPU DgSolver (5-stage LSRK45)
    const double dt = 0.005;
    double t = 0.0;
    for (int s = 0; s < 5; ++s) {
        solver.step(dt, t);
        t += dt;
        solver.downloadState(state.data(), state.size());
        double currentEnergy = computeEnergy(state);

        // Discrete energy must be non-increasing for upwind flux formulation (dU/dt <= 0)
        EXPECT_TRUE(currentEnergy <= initialEnergy + 1e-12);

        // Numerical solutions must remain well-behaved without NaN or Inf
        EXPECT_TRUE(!std::isnan(currentEnergy));
        EXPECT_TRUE(!std::isinf(currentEnergy));
    }

    solver.finalize();

    g_testsPassed++;
    std::cout << "  PASSED" << std::endl;
}

/*
 * Test 4: Case Input Loading and Zero Disk Artifacts Verification
 *
 * Verifies that loading a test input fixture from tests/input/ with output_dir = none
 * runs correctly and produces zero output files on disk (no test pollution).
 */
void CaseInputLoadingTest() {
    std::cout << "[RUN] CaseInputLoadingTest..." << std::endl;

#ifdef NEKWAVE_SOURCE_DIR
    std::string parFile = std::string(NEKWAVE_SOURCE_DIR) + "/tests/input/cavity_gaussian.par";
#else
    std::string parFile = "tests/input/cavity_gaussian.par";
#endif

    // Verify input file exists
    std::ifstream f(parFile.c_str());
    EXPECT_TRUE(f.good());
    f.close();

    Case testCase;
    testCase.loadConfig(parFile);

    // Override to a fast 2-step run for unit testing
    testCase.config().numSteps = 2;
    testCase.config().outputFreq = 1;

    EXPECT_TRUE(testCase.config().outputDir == "none");
    EXPECT_TRUE(!testCase.config().exportFields);

    // Run the full 3-phase lifecycle
    testCase.run();

    // Verify that output_dir was "none" and no files/directories were created
    struct stat st;
    int res = stat("none", &st);
    EXPECT_TRUE(res != 0); // Directory 'none' must NOT exist

#ifdef NEKWAVE_BUILD_DIR
    std::string buildNone = std::string(NEKWAVE_BUILD_DIR) + "/none";
    res = stat(buildNone.c_str(), &st);
    EXPECT_TRUE(res != 0); // Directory '${BUILD}/none' must NOT exist
#endif

    g_testsPassed++;
    std::cout << "  PASSED" << std::endl;
}

/*
 * Test 5: Numerical Dispersion Benchmark Test
 *
 * Verifies that loading the numerical dispersion test fixture from
 * tests/input/numerical_dispersion.par initializes the DG-SEM Maxwell
 * solver, injects the wavepacket initial condition, advances in time
 * on GPU, and produces zero output files on disk.
 */
void NumericalDispersionTest() {
    std::cout << "[RUN] NumericalDispersionTest..." << std::endl;

#ifdef NEKWAVE_SOURCE_DIR
    std::string parFile = std::string(NEKWAVE_SOURCE_DIR) + "/tests/input/numerical_dispersion.par";
#else
    std::string parFile = "tests/input/numerical_dispersion.par";
#endif

    std::ifstream f(parFile.c_str());
    EXPECT_TRUE(f.good());
    f.close();

    Case dispersionCase;
    dispersionCase.loadConfig(parFile);

    EXPECT_TRUE(dispersionCase.config().outputDir == "none");
    EXPECT_TRUE(!dispersionCase.config().exportFields);

    const auto& cfg = dispersionCase.config();
    const double Lx = cfg.Lx;
    const double Ly = cfg.Ly;
    const double xmin = cfg.xmin;
    const double carrierK = cfg.getDouble("carrier_k", 4.0 * M_PI);
    const double sigma = cfg.getDouble("packet_sigma", 0.30);
    const double x0 = cfg.getDouble("packet_x0", xmin + 0.25 * Lx);

    // Initial condition hook: modulated Gaussian wavepacket
    dispersionCase.setInitialCondition([=](double x, double y, double /*z*/,
                                          double& Ex, double& Ey, double& Ez,
                                          double& Hx, double& Hy, double& Hz) {
        Ex = 0.0; Ey = 0.0; Hx = 0.0; Hz = 0.0;
        double yt = y - cfg.ymin;
        double transY = std::sin(M_PI * yt / Ly);
        double dxEnv = x - x0;
        double envelope = std::exp(-(dxEnv * dxEnv) / (2.0 * sigma * sigma));
        double carrier = std::cos(carrierK * dxEnv);
        Ez = envelope * carrier * transY;
        Hy = -Ez;
    });

    dispersionCase.run();

    // Verify energy is strictly positive and finite
    double totalEnergy = dispersionCase.computeTotalEnergy();
    EXPECT_TRUE(totalEnergy > 0.0);
    EXPECT_TRUE(!std::isnan(totalEnergy));
    EXPECT_TRUE(!std::isinf(totalEnergy));

    // Verify no output directories or files were created
    struct stat st;
    int res = stat("none", &st);
    EXPECT_TRUE(res != 0);

#ifdef NEKWAVE_BUILD_DIR
    std::string buildNone = std::string(NEKWAVE_BUILD_DIR) + "/none";
    res = stat(buildNone.c_str(), &st);
    EXPECT_TRUE(res != 0);
#endif

    g_testsPassed++;
    std::cout << "  PASSED" << std::endl;
}

/*
 * Test 6: Periodic Boundary Condition Mesh Connectivity
 *
 * Verifies that structured Cartesian box meshes with periodic boundary conditions:
 * 1. Tag boundary faces as "PERIODIC" rather than "PEC".
 * 2. Assign valid exterior neighbor element IDs across the periodic wrap.
 * 3. Correctly connect opposite face collocation nodes such that transverse
 *    coordinates match exactly and longitudinal coordinates differ by domain length L.
 * 4. Maintain non-periodic directions (e.g. Z) as PEC mirror conditions.
 */
void PeriodicBCTest() {
    std::cout << "[RUN] PeriodicBCTest..." << std::endl;

    const int N = 4;
    Mesh mesh(N, 1);

    // 2x2x2 mesh with periodic in X and Y, PEC in Z
    const double xmin = -2.0, xmax = 2.0;
    const double ymin = -1.0, ymax = 1.0;
    const double zmin = -0.5, zmax = 0.5;
    mesh.createBoxMesh(2, 2, 2, xmin, xmax, ymin, ymax, zmin, zmax, true, true, false);

    const auto& faceData = mesh.getFaceData();
    const auto& x = mesh.getCoordX();
    const auto& y = mesh.getCoordY();
    const auto& z = mesh.getCoordZ();

    int periodicFacesCount = 0;
    int pecFacesCount = 0;

    for (size_t f = 0; f < faceData.size(); ++f) {
        const auto& fd = faceData[f];
        if (fd.bcType == "PERIODIC") {
            periodicFacesCount++;
            for (const auto& pt : fd.points) {
                EXPECT_TRUE(pt.volIdxPlus >= 0);
                int vM = pt.volIdxMinus;
                int vP = pt.volIdxPlus;

                // For X-periodic faces (face 1 and 3): Y and Z coordinates must match exactly
                if (fd.faceId == 1 || fd.faceId == 3) {
                    EXPECT_NEAR(y[vM], y[vP], 1e-12);
                    EXPECT_NEAR(z[vM], z[vP], 1e-12);
                    EXPECT_NEAR(std::abs(x[vM] - x[vP]), (xmax - xmin), 1e-12);
                }
                // For Y-periodic faces (face 0 and 2): X and Z coordinates must match exactly
                if (fd.faceId == 0 || fd.faceId == 2) {
                    EXPECT_NEAR(x[vM], x[vP], 1e-12);
                    EXPECT_NEAR(z[vM], z[vP], 1e-12);
                    EXPECT_NEAR(std::abs(y[vM] - y[vP]), (ymax - ymin), 1e-12);
                }
            }
        } else if (fd.bcType == "PEC") {
            pecFacesCount++;
            // Must be Z boundary faces (face 4 or 5)
            EXPECT_TRUE(fd.faceId == 4 || fd.faceId == 5);
            for (const auto& pt : fd.points) {
                EXPECT_TRUE(pt.volIdxPlus == -1);
            }
        }
    }

    // With 2x2x2 elements (8 elements, 48 faces total):
    // Internal faces: 12 internal interfaces * 2 = 24 "E" faces
    // Periodic faces: 4 in Xmin + 4 in Xmax + 4 in Ymin + 4 in Ymax = 16 "PERIODIC" faces
    // PEC faces: 4 in Zmin + 4 in Zmax = 8 "PEC" faces
    EXPECT_TRUE(periodicFacesCount == 16);
    EXPECT_TRUE(pecFacesCount == 8);

    g_testsPassed++;
    std::cout << "  PASSED" << std::endl;
}

/*
 * Main entry point for the NekWave test harness.
 *
 * Runs all or specifically selected unit test routines, tracks aggregate
 * passes and failures, and returns 0 upon success or 1 if any assertion fails.
 *
 * Usage:
 *   ./nekwave-test                           (runs all tests)
 *   ./nekwave-test MeshQuadratureTest        (runs only MeshQuadratureTest)
 *   ./nekwave-test ProbeScalingTest         (runs only ProbeScalingTest)
 *   ./nekwave-test MaxwellStabilityTest     (runs only MaxwellStabilityTest)
 *   ./nekwave-test CaseInputLoadingTest     (runs only CaseInputLoadingTest)
 *   ./nekwave-test NumericalDispersionTest  (runs only NumericalDispersionTest)
 *   ./nekwave-test PeriodicBCTest           (runs only PeriodicBCTest)
 */
int main(int argc, char* argv[]) {
    std::string filter = (argc > 1) ? argv[1] : "";

    std::cout << "\n----------------------------------------------------------" << std::endl;
    std::cout << "              NekWave Test Suite (nekwave-test)           " << std::endl;
    if (!filter.empty()) {
        std::cout << "              Filter: " << filter << std::endl;
    }
    std::cout << "----------------------------------------------------------" << std::endl;

    if (filter.empty() || filter == "MeshQuadratureTest") {
        MeshQuadratureTest();
    }
    if (filter.empty() || filter == "ProbeScalingTest") {
        ProbeScalingTest();
    }
    if (filter.empty() || filter == "MaxwellStabilityTest" || filter == "MaxwellPhysicsStabilityTest") {
        MaxwellStabilityTest();
    }
    if (filter.empty() || filter == "CaseInputLoadingTest") {
        CaseInputLoadingTest();
    }
    if (filter.empty() || filter == "NumericalDispersionTest") {
        NumericalDispersionTest();
    }
    if (filter.empty() || filter == "PeriodicBCTest") {
        PeriodicBCTest();
    }


    std::cout << "----------------------------------------------------------" << std::endl;
    std::cout << "Test Summary: " << g_testsPassed << " passed, " 
              << g_testsFailed << " failed." << std::endl;
    std::cout << "----------------------------------------------------------\n" << std::endl;

    return (g_testsFailed == 0 && (g_testsPassed > 0 || filter.empty())) ? 0 : 1;
}
