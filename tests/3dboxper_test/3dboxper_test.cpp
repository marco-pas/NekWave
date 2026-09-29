// NekWave Regression Test: NekCEM Canonical 3D Box Periodic Verification
// Validates Maxwell DGTD solver on legacy NekCEM .rea mesh geometry and periodic boundaries

#include "case.hpp"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>

int main(int argc, char* argv[]) {
    std::cout << "==========================================================" << std::endl;
    std::cout << "    NekWave Test: NekCEM 3D Box Periodic Cavity Case      " << std::endl;
    std::cout << "==========================================================" << std::endl;

    std::string parFile = "tests/3dboxper_test/3dboxper_test.par";
    if (argc > 1) {
        parFile = argv[1];
    }

    Case simulationCase;
    simulationCase.loadConfig(parFile);

    // 1. Explicit initial condition hook
    simulationCase.setInitialConditionHook([](double x, double y, double z,
                                              double& ex, double& ey, double& ez,
                                              double& hx, double& hy, double& hz) {
        ex = 0.0;
        ey = std::cos(x) * std::sin(y) * std::sin(z);
        ez = std::cos(x) * std::cos(y) * std::cos(z);

        hx = 0.0;
        hy = 0.0;
        hz = 0.0;
    });

    double finalRelativeL2 = 0.0;
    double finalLinfError = 0.0;

    // 2. Define postprocessing verification hook comparing with 3dboxper exact solution
    simulationCase.setPostprocessingHook([&](Case& c) {
        std::cout << "\n==========================================================" << std::endl;
        std::cout << "     3D Box Periodic Exact Analytical Verification        " << std::endl;
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

        const double omega = std::sqrt(3.0);
        const double tmph = (std::abs(omega) > 1e-14) ? (std::sin(omega * simTime) / omega) : simTime;
        const double tmpe = std::cos(omega * simTime);

        double l2ErrorNum = 0.0;
        double l2ExactDen = 0.0;
        double maxErr = 0.0;

        for (int e = 0; e < mesh.getNumElements(); ++e) {
            int elemOffset = e * Np;
            for (int k = 0; k < Np; ++k) {
                int idx = elemOffset + k;
                double dV = jac[idx] * w3[k];

                double exactEx = 0.0;
                double exactEy =  std::cos(x[idx]) * std::sin(y[idx]) * std::sin(z[idx]) * tmpe;
                double exactEz =  std::cos(x[idx]) * std::cos(y[idx]) * std::cos(z[idx]) * tmpe;

                double exactHx =  2.0 * std::cos(x[idx]) * std::sin(y[idx]) * std::cos(z[idx]) * tmph;
                double exactHy = -std::sin(x[idx]) * std::cos(y[idx]) * std::cos(z[idx]) * tmph;
                double exactHz =  std::sin(x[idx]) * std::sin(y[idx]) * std::sin(z[idx]) * tmph;

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

        finalRelativeL2 = std::sqrt(l2ErrorNum / (l2ExactDen > 1e-14 ? l2ExactDen : 1.0));
        finalLinfError = maxErr;

        std::cout << "  Simulation Time (t):     " << std::scientific << std::setprecision(5) << simTime << std::endl;
        std::cout << "  Theoretical Frequency w: " << omega << std::endl;
        std::cout << "  Relative L2 Error:       " << finalRelativeL2 << std::endl;
        std::cout << "  Discrete Linf Error:     " << finalLinfError << std::endl;
        std::cout << "==========================================================" << std::endl;
    });

    // 3. Execution phases
    simulationCase.preprocess();
    simulationCase.simulate();
    simulationCase.postprocess();

    const double tolerance = 0.05;
    if (finalRelativeL2 > tolerance) {
        std::cerr << "Verification FAILED: Relative L2 error " << finalRelativeL2 
                  << " exceeds threshold " << tolerance << std::endl;
        return 1;
    }

    std::cout << "Verification PASSED: Solution matches NekCEM periodic eigenmode.\n" << std::endl;
    return 0;
}
