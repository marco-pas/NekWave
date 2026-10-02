// ==============================================================================
// NekWave: High-Order Discontinuous Galerkin Convergence Benchmark Suite
// ==============================================================================
//
// Verifies spatial accuracy of the 3D DG-SEM Maxwell solver over 10 wave periods:
//   1. h-refinement study for TWO polynomial degrees:
//        - P = 3 (N = 4)
//        - P = 6 (N = 7)
//      across meshes: 2x2x1, 4x4x1, 8x8x1, 16x16x1.
//
//   2. p-refinement study for TWO mesh resolutions:
//        - Coarse mesh: 2x2x1 elements
//        - Refined mesh: 8x8x1 elements
//      across polynomial orders: N in {2, 3, 4, 5, 6, 7, 8} (degrees P = 1..7).
//
//   3. Scans across numerical flux penalty parameters:
//        - C0 = 0.0 (Energy-conserving Central Flux)
//        - C0 = 0.5 (Intermediate Dissipative Flux)
//        - C0 = 1.0 (Strictly Dissipative Upwind Flux)
//
// Analytical Benchmark:
//   Transverse electromagnetic plane wave in 3D periodic domain [-1, 1]^3:
//     Ez(x, y, z, t) = cos(kx * x + ky * y - omega * t)
//     Hx(x, y, z, t) = (1 / sqrt(2)) * cos(kx * x + ky * y - omega * t)
//     Hy(x, y, z, t) = -(1 / sqrt(2)) * cos(kx * x + ky * y - omega * t)
//   with kx = pi, ky = pi, kz = 0, omega = sqrt(2) * pi (c = 1).
//   One period T_period = 2 * pi / omega = sqrt(2).
//   Simulated for 10 periods: T_final = 10 * sqrt(2).
//   At t = T_final, omega * T_final = 20 * pi, so the exact solution returns
//   identically to the initial state t = 0.
//
// ==============================================================================

#include "mesh.hpp"
#include "dg_solver.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <sys/stat.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct TrialResult {
    int N;             // Polynomial order (N points, degree P = N - 1)
    int nelx;          // Number of elements in X
    int nely;          // Number of elements in Y
    int nelz;          // Number of elements in Z
    int totalElements; // Total hex elements
    int totalDOFs;     // Total spatial degrees of freedom (6 * totalPoints)
    double h;          // Representative mesh spacing (dx)
    double dt;         // Time step size
    int numSteps;      // Total time steps to reach t = 10 * T_period
    double l2Error;    // Discrete L2 error of electromagnetic state
    double linfError;  // Discrete Linf (max) error
    double eoc;        // Empirical Order of Convergence: log(e1/e2) / log(h1/h2)
    double elapsedSec; // Wall-clock execution time
};

// Evaluates the exact analytical traveling wave solution at (x, y, z, t)
static void getExactSolution(double x, double y, double /*z*/, double t,
                             double& Ez, double& Hx, double& Hy) {
    const double pi = M_PI;
    const double kx = pi;
    const double ky = pi;
    const double omega = std::sqrt(2.0) * pi;
    const double phase = kx * x + ky * y - omega * t;

    Ez = std::cos(phase);
    Hx = (1.0 / std::sqrt(2.0)) * std::cos(phase);
    Hy = -(1.0 / std::sqrt(2.0)) * std::cos(phase);
}

// Executes a single simulation run for 10 periods (T_final = 10 * sqrt(2))
static TrialResult runConvergenceTrial(int N, int nelx, int nely, int nelz,
                                      double C0, int numPeriods = 10,
                                      double cflSafety = 0.20) {
    auto startTime = std::chrono::high_resolution_clock::now();

    TrialResult res;
    res.N = N;
    res.nelx = nelx;
    res.nely = nely;
    res.nelz = nelz;
    res.totalElements = nelx * nely * nelz;
    res.h = 2.0 / nelx; // Domain [-1, 1], Lx = 2.0
    res.eoc = 0.0;

    // 1. Setup periodic Cartesian box mesh on [-1, 1] x [-1, 1] x [-0.5, 0.5]
    Mesh mesh(N, 1);
    mesh.createBoxMesh(nelx, nely, nelz,
                       -1.0, 1.0,
                       -1.0, 1.0,
                       -0.5, 0.5,
                       true, true, true); // All periodic

    const int npts = mesh.getTotalPoints();
    res.totalDOFs = 6 * npts;

    // 2. Initialize GPU DgSolver
    DgSolver solver;
    solver.initialize(mesh, C0);

    // 3. Set Initial Condition at t = 0
    StateVector state(6 * npts, 0.0);
    const auto& xCoord = mesh.getCoordX();
    const auto& yCoord = mesh.getCoordY();
    const auto& zCoord = mesh.getCoordZ();

    for (int i = 0; i < npts; ++i) {
        double Ez, Hx, Hy;
        getExactSolution(xCoord[i], yCoord[i], zCoord[i], 0.0, Ez, Hx, Hy);
        state[0 * npts + i] = 0.0;  // Ex
        state[1 * npts + i] = 0.0;  // Ey
        state[2 * npts + i] = Ez;   // Ez
        state[3 * npts + i] = Hx;   // Hx
        state[4 * npts + i] = Hy;   // Hy
        state[5 * npts + i] = 0.0;  // Hz
    }
    solver.uploadState(state.data(), state.size());

    // 4. Exact final time for numPeriods periods: T_final = numPeriods * sqrt(2)
    const double T_period = std::sqrt(2.0);
    const double T_final = numPeriods * T_period;

    double dt = mesh.computeAutomaticDt(1.0, cflSafety);
    int numSteps = std::max(1, static_cast<int>(std::ceil(T_final / dt)));
    dt = T_final / numSteps; // Harmonize dt to hit T_final with machine precision

    res.dt = dt;
    res.numSteps = numSteps;

    // 5. Time integration over 10 complete periods
    double simTime = 0.0;
    for (int step = 0; step < numSteps; ++step) {
        solver.step(dt, simTime);
        simTime += dt;
    }

    // 6. Download final solution and evaluate error norms against analytical solution at t = T_final
    solver.downloadState(state.data(), state.size());

    const auto& J = mesh.getJac();
    const auto& w1 = mesh.getGllW();
    const int Np = N * N * N;

    double errorL2Sq = 0.0;
    double errorLinf = 0.0;

    for (int e = 0; e < mesh.getNumElements(); ++e) {
        int elemOffset = e * Np;
        for (int k = 0; k < N; ++k) {
            for (int j = 0; j < N; ++j) {
                for (int i = 0; i < N; ++i) {
                    int p = elemOffset + i + N * (j + N * k);
                    double Ez_ex, Hx_ex, Hy_ex;
                    getExactSolution(xCoord[p], yCoord[p], zCoord[p], T_final, Ez_ex, Hx_ex, Hy_ex);

                    double diffEz = state[2 * npts + p] - Ez_ex;
                    double diffHx = state[3 * npts + p] - Hx_ex;
                    double diffHy = state[4 * npts + p] - Hy_ex;

                    double ptDiffSq = diffEz * diffEz + diffHx * diffHx + diffHy * diffHy;
                    double dV = J[p] * w1[i] * w1[j] * w1[k];

                    errorL2Sq += ptDiffSq * dV;

                    double maxPtDiff = std::max({std::abs(diffEz), std::abs(diffHx), std::abs(diffHy)});
                    errorLinf = std::max(errorLinf, maxPtDiff);
                }
            }
        }
    }

    res.l2Error = std::sqrt(errorL2Sq);
    res.linfError = errorLinf;

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;
    res.elapsedSec = elapsed.count();

    return res;
}

// Compute Empirical Order of Convergence (EOC)
static void computeEOC(std::vector<TrialResult>& results) {
    for (size_t i = 1; i < results.size(); ++i) {
        double ePrev = results[i - 1].l2Error;
        double eCurr = results[i].l2Error;
        double hPrev = results[i - 1].h;
        double hCurr = results[i].h;

        if (eCurr > 1e-15 && ePrev > 1e-15 && hPrev > hCurr) {
            results[i].eoc = std::log(ePrev / eCurr) / std::log(hPrev / hCurr);
        } else {
            results[i].eoc = 0.0;
        }
    }
}

// Print formatted table header
static void printTableHeader() {
    std::cout << "------------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << std::setw(6)  << "N (P)"
              << std::setw(16) << "Mesh"
              << std::setw(10) << "h"
              << std::setw(10) << "DOFs"
              << std::setw(14) << "dt"
              << std::setw(15) << "L2 Error"
              << std::setw(15) << "Linf Error"
              << std::setw(10) << "EOC"
              << std::setw(10) << "Time (s)" << std::endl;
    std::cout << "------------------------------------------------------------------------------------------------------" << std::endl;
}

// Print formatted table row
static void printTableRow(const TrialResult& r, bool showEOC = true) {
    std::string nPStr = std::to_string(r.N) + " (P=" + std::to_string(r.N - 1) + ")";
    std::string meshStr = std::to_string(r.nelx) + "x" + std::to_string(r.nely) + "x" + std::to_string(r.nelz);
    std::cout << std::setw(6)  << nPStr
              << std::setw(16) << meshStr
              << std::setw(10) << std::fixed << std::setprecision(4) << r.h
              << std::setw(10) << r.totalDOFs
              << std::setw(14) << std::scientific << std::setprecision(3) << r.dt
              << std::setw(15) << std::scientific << std::setprecision(5) << r.l2Error
              << std::setw(15) << std::scientific << std::setprecision(5) << r.linfError;
    if (showEOC) {
        if (r.eoc > 0.0) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(2) << r.eoc;
        } else {
            std::cout << std::setw(10) << "-";
        }
    } else {
        std::cout << std::setw(10) << "-";
    }
    std::cout << std::setw(10) << std::fixed << std::setprecision(2) << r.elapsedSec << std::endl;
}

// Export multi-curve results to JSON
static void exportResultsToJson(const std::string& filename,
                                double C0, int numPeriods,
                                const std::vector<TrialResult>& hResultsN3,
                                const std::vector<TrialResult>& hResultsN5,
                                const std::vector<TrialResult>& hResultsN8,
                                const std::vector<TrialResult>& pResultsMesh4,
                                const std::vector<TrialResult>& pResultsMesh16) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Warning: Could not open " << filename << " for writing JSON results." << std::endl;
        return;
    }

    auto dumpTrialList = [&](const std::vector<TrialResult>& list) {
        out << "[\n";
        for (size_t i = 0; i < list.size(); ++i) {
            const auto& r = list[i];
            out << "        {\n";
            out << "          \"N\": " << r.N << ",\n";
            out << "          \"degree_P\": " << r.N - 1 << ",\n";
            out << "          \"nelx\": " << r.nelx << ",\n";
            out << "          \"nely\": " << r.nely << ",\n";
            out << "          \"nelz\": " << r.nelz << ",\n";
            out << "          \"h\": " << r.h << ",\n";
            out << "          \"DOFs\": " << r.totalDOFs << ",\n";
            out << "          \"dt\": " << r.dt << ",\n";
            out << "          \"num_steps\": " << r.numSteps << ",\n";
            out << "          \"l2_error\": " << r.l2Error << ",\n";
            out << "          \"linf_error\": " << r.linfError << ",\n";
            out << "          \"eoc\": " << r.eoc << ",\n";
            out << "          \"elapsed_sec\": " << r.elapsedSec << "\n";
            out << "        }" << (i + 1 < list.size() ? ",\n" : "\n");
        }
        out << "      ]";
    };

    out << "{\n";
    out << "  \"C0\": " << C0 << ",\n";
    out << "  \"num_periods\": " << numPeriods << ",\n";
    out << "  \"h_refinement_N3\": "; dumpTrialList(hResultsN3); out << ",\n";
    out << "  \"h_refinement_N5\": "; dumpTrialList(hResultsN5); out << ",\n";
    out << "  \"h_refinement_N8\": "; dumpTrialList(hResultsN8); out << ",\n";
    out << "  \"p_refinement_mesh4\": "; dumpTrialList(pResultsMesh4); out << ",\n";
    out << "  \"p_refinement_mesh16\": "; dumpTrialList(pResultsMesh16); out << "\n";
    out << "}\n";

    std::cout << "[INFO] Results exported to: " << filename << std::endl;
}

// Executes full suite for a given C0 value
static void runStudyForC0(double C0, const std::string& outJsonPath, int numPeriods = 10) {
    std::cout << "======================================================================================================" << std::endl;
    std::cout << "               NekWave DG-SEM Convergence Study (10 Wave Periods, C0 = " << C0 << ")                " << std::endl;
    std::cout << "======================================================================================================" << std::endl;
    std::cout << "Simulation Duration: " << numPeriods << " periods (T_final = " << numPeriods << " * sqrt(2) ~ " 
              << numPeriods * std::sqrt(2.0) << ")" << std::endl;
    std::cout << "Flux Parameter C0  : " << C0 << " (" 
              << (C0 == 0.0 ? "Central" : (C0 == 1.0 ? "Upwind" : "Intermediate")) << ")" << std::endl << std::endl;

    const std::vector<int> hMeshSizes = {2, 4, 8, 16, 32, 64};

    // 1. h-refinement: N = 3 (Degree P = 2)
    std::cout << ">>> PART 1A: h-Refinement for Polynomial Order N = 3 (Degree P = 2)" << std::endl;
    printTableHeader();
    std::vector<TrialResult> hResultsN3;
    for (int nel : hMeshSizes) {
        TrialResult r = runConvergenceTrial(3, nel, nel, 1, C0, numPeriods);
        hResultsN3.push_back(r);
        computeEOC(hResultsN3);
        printTableRow(hResultsN3.back(), true);
    }
    std::cout << "------------------------------------------------------------------------------------------------------" << std::endl << std::endl;

    // 2. h-refinement: N = 5 (Degree P = 4)
    std::cout << ">>> PART 1B: h-Refinement for Polynomial Order N = 5 (Degree P = 4)" << std::endl;
    printTableHeader();
    std::vector<TrialResult> hResultsN5;
    for (int nel : hMeshSizes) {
        TrialResult r = runConvergenceTrial(5, nel, nel, 1, C0, numPeriods);
        hResultsN5.push_back(r);
        computeEOC(hResultsN5);
        printTableRow(hResultsN5.back(), true);
    }
    std::cout << "------------------------------------------------------------------------------------------------------" << std::endl << std::endl;

    // 3. h-refinement: N = 8 (Degree P = 7)
    std::cout << ">>> PART 1C: h-Refinement for Polynomial Order N = 8 (Degree P = 7)" << std::endl;
    printTableHeader();
    std::vector<TrialResult> hResultsN8;
    for (int nel : hMeshSizes) {
        TrialResult r = runConvergenceTrial(8, nel, nel, 1, C0, numPeriods);
        hResultsN8.push_back(r);
        computeEOC(hResultsN8);
        printTableRow(hResultsN8.back(), true);
    }
    std::cout << "------------------------------------------------------------------------------------------------------" << std::endl << std::endl;

    const std::vector<int> pOrders = {2, 3, 4, 5, 6, 7, 8};

    // 4. p-refinement: Mesh 4x4x1 hex elements
    std::cout << ">>> PART 2A: p-Refinement for Mesh 4x4x1 hex elements" << std::endl;
    printTableHeader();
    std::vector<TrialResult> pResultsMesh4;
    for (int N : pOrders) {
        TrialResult r = runConvergenceTrial(N, 4, 4, 1, C0, numPeriods);
        pResultsMesh4.push_back(r);
        printTableRow(pResultsMesh4.back(), false);
    }
    std::cout << "------------------------------------------------------------------------------------------------------" << std::endl << std::endl;

    // 5. p-refinement: Refined Mesh 16x16x1 hex elements
    std::cout << ">>> PART 2B: p-Refinement for Refined Mesh 16x16x1 hex elements" << std::endl;
    printTableHeader();
    std::vector<TrialResult> pResultsMesh16;
    for (int N : pOrders) {
        TrialResult r = runConvergenceTrial(N, 16, 16, 1, C0, numPeriods);
        pResultsMesh16.push_back(r);
        printTableRow(pResultsMesh16.back(), false);
    }
    std::cout << "------------------------------------------------------------------------------------------------------" << std::endl << std::endl;

    exportResultsToJson(outJsonPath, C0, numPeriods, hResultsN3, hResultsN5, hResultsN8, pResultsMesh4, pResultsMesh16);
}

int main(int argc, char* argv[]) {
    // Ensure output directories exist
    std::string outDir = "examples/numerical_analysis/convergence_test/output";
    struct stat st;
    if (stat("examples/numerical_analysis/convergence_test", &st) == 0) {
        mkdir("examples/numerical_analysis/convergence_test/output", 0755);
    } else {
        outDir = "output";
        mkdir("output", 0755);
    }
    // Also ensure local output/ exists
    mkdir("output", 0755);

    const int numPeriods = 10;

    if (argc > 1 && std::string(argv[1]) != "--all") {
        // Run single C0 specified by user
        double C0 = std::stod(argv[1]);
        std::string tag = (C0 == 0.0) ? "0.0" : (C0 == 0.5 ? "0.5" : (C0 == 1.0 ? "1.0" : std::to_string(C0).substr(0, 3)));
        std::string outJson = (argc > 2) ? argv[2] : (outDir + "/convergence_results_C0_" + tag + ".json");
        runStudyForC0(C0, outJson, numPeriods);
    } else {
        // Run full scan across C0 = 0.0, 0.5, 1.0
        std::vector<double> c0Scan = {0.0, 0.5, 1.0};
        for (double c0 : c0Scan) {
            std::string tag = (c0 == 0.0) ? "0.0" : (c0 == 0.5 ? "0.5" : "1.0");
            std::string outJson = outDir + "/convergence_results_C0_" + tag + ".json";
            runStudyForC0(c0, outJson, numPeriods);
        }
    }


    std::cout << "======================================================================================================" << std::endl;
    std::cout << "All 10-period convergence trials completed successfully." << std::endl;
    std::cout << "======================================================================================================" << std::endl;

    return 0;
}
