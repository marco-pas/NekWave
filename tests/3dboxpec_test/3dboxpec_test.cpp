// NekWave Regression Test: NekCEM Canonical 3D Box PEC Verification
// Validates Maxwell DGTD solver on legacy NekCEM .rea mesh geometry and PEC boundaries

#include "case.hpp"
#include "comm.hpp"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>

int main(int argc, char* argv[]) {
    Comm::init(&argc, &argv);
    if (Comm::isRoot()) {
        std::cout << "==========================================================" << std::endl;
        std::cout << "      NekWave Test: NekCEM 3D Box PEC Cavity Case         " << std::endl;
        std::cout << "==========================================================" << std::endl;
    }

    std::string parFile = "tests/3dboxpec_test/3dboxpec_test.par";
    if (argc > 1) {
        parFile = argv[1];
    }

    Case simulationCase;
    simulationCase.loadConfig(parFile);

    // 1. Explicit initial condition hook
    simulationCase.setInitialConditionHook([](double x, double y, double z,
                                              double& ex, double& ey, double& ez,
                                              double& hx, double& hy, double& hz) {
        const double invSqrt6 = 1.0 / std::sqrt(6.0);
        double sx = std::sin(M_PI * x);
        double cx = std::cos(M_PI * x);
        double sy = std::sin(M_PI * y);
        double cy = std::cos(M_PI * y);
        double sz = std::sin(M_PI * z);
        double cz = std::cos(M_PI * z);

        ex = 0.0;
        ey = 0.0;
        ez = 0.0;

        hx = -sx * cy * cz * invSqrt6;
        hy = -cx * sy * cz * invSqrt6;
        hz =  2.0 * cx * cy * sz * invSqrt6;
    });

    double finalRelativeL2 = 0.0;
    double finalLinfError = 0.0;

    // 2. Define postprocessing verification hook comparing with 3dboxpec exact solution
    simulationCase.setPostprocessingHook([&](Case& c) {
        std::cout << "\n==========================================================" << std::endl;
        std::cout << "        3D Box PEC Exact Analytical Verification          " << std::endl;
        std::cout << "==========================================================" << std::endl;

        const double simTime = c.currentTime();
        const auto& mesh = c.mesh();
        const auto& state = c.state();
        const int npts = mesh.getTotalPoints();
        const auto& x = mesh.getCoordX();
        const auto& y = mesh.getCoordY();
        const auto& z = mesh.getCoordZ();
        const auto& w3 = mesh.getW3();
        const auto& jac = mesh.getJac();
        const int Np = mesh.getNumPointsPerElement();

        const double sqrt6 = std::sqrt(6.0);
        const double sqrt2 = std::sqrt(2.0);
        const double omega = M_PI * std::sqrt(3.0);

        const double tmph = std::cos(omega * simTime) / sqrt6;
        const double tmpe = std::sin(omega * simTime) / sqrt2;

        double l2ErrorNum = 0.0;
        double l2ExactDen = 0.0;
        double maxErr = 0.0;

        for (int e = 0; e < mesh.getNumElements(); ++e) {
            int elemOffset = e * Np;
            for (int k = 0; k < Np; ++k) {
                int idx = elemOffset + k;
                double dV = jac[idx] * w3[k];

                double exactEx = -std::cos(M_PI * x[idx]) * std::sin(M_PI * y[idx]) * std::sin(M_PI * z[idx]) * tmpe;
                double exactEy =  std::sin(M_PI * x[idx]) * std::cos(M_PI * y[idx]) * std::sin(M_PI * z[idx]) * tmpe;
                double exactEz = 0.0;

                double exactHx = -std::sin(M_PI * x[idx]) * std::cos(M_PI * y[idx]) * std::cos(M_PI * z[idx]) * tmph;
                double exactHy = -std::cos(M_PI * x[idx]) * std::sin(M_PI * y[idx]) * std::cos(M_PI * z[idx]) * tmph;
                double exactHz =  2.0 * std::cos(M_PI * x[idx]) * std::cos(M_PI * y[idx]) * std::sin(M_PI * z[idx]) * tmph;

                double dEx = state[0 * npts + idx] - exactEx;
                double dEy = state[1 * npts + idx] - exactEy;
                double dEz = state[2 * npts + idx] - exactEz;
                double dHx = state[3 * npts + idx] - exactHx;
                double dHy = state[4 * npts + idx] - exactHy;
                double dHz = state[5 * npts + idx] - exactHz;

                double errSq = dEx*dEx + dEy*dEy + dEz*dEz + dHx*dHx + dHy*dHy + dHz*dHz;
                double exSq  = exactEx*exactEx + exactEy*exactEy + exactEz*exactEz +
                               exactHx*exactHx + exactHy*exactHy + exactHz*exactHz;

                l2ErrorNum += errSq * dV;
                l2ExactDen += exSq * dV;

                maxErr = std::max(maxErr, std::abs(dEx));
                maxErr = std::max(maxErr, std::abs(dEy));
                maxErr = std::max(maxErr, std::abs(dEz));
                maxErr = std::max(maxErr, std::abs(dHx));
                maxErr = std::max(maxErr, std::abs(dHy));
                maxErr = std::max(maxErr, std::abs(dHz));
            }
        }

        double globalL2Num = Comm::allreduceSum(l2ErrorNum);
        double globalL2Den = Comm::allreduceSum(l2ExactDen);
        finalRelativeL2 = std::sqrt(globalL2Num / (globalL2Den > 1e-14 ? globalL2Den : 1.0));
        finalLinfError = Comm::allreduceMax(maxErr);

        if (Comm::isRoot()) {
            std::cout << "  Simulation Time (t):     " << std::scientific << std::setprecision(5) << simTime << std::endl;
            std::cout << "  Theoretical Frequency w: " << omega << std::endl;
            std::cout << "  Relative L2 Error:       " << finalRelativeL2 << std::endl;
            std::cout << "  Discrete Linf Error:     " << finalLinfError << std::endl;
            std::cout << "==========================================================" << std::endl;
        }
    });

    // 3. Execution phases
    simulationCase.preprocess();
    simulationCase.simulate();
    simulationCase.postprocess();

    const double tolerance = 0.05;
    bool passed = (finalRelativeL2 <= tolerance);
    if (!passed) {
        if (Comm::isRoot()) {
            std::cerr << "Verification FAILED: Relative L2 error " << finalRelativeL2 
                      << " exceeds threshold " << tolerance << std::endl;
        }
    } else {
        if (Comm::isRoot()) {
            std::cout << "Verification PASSED: Solution matches NekCEM analytical eigenmode.\n" << std::endl;
        }
    }

    Comm::finalize();
    return passed ? 0 : 1;
}
