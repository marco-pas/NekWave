#include "case.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(int argc, char* argv[]) {
    std::string parFile = (argc > 1) ? argv[1] : "cavity_eigenmode.par";

    std::cout << "==========================================================" << std::endl;
    std::cout << "      NekWave Example: Analytical Cavity TM_110 Eigenmode " << std::endl;
    std::cout << "==========================================================" << std::endl;

    Case cavityCase;
    cavityCase.loadConfig(parFile);

    const auto& cfg = cavityCase.config();
    const double Lx = cfg.Lx;
    const double Ly = cfg.Ly;
    const double xmin = cfg.xmin;
    const double ymin = cfg.ymin;

    const int m = cfg.getInt("mode_m", 1);
    const int n = cfg.getInt("mode_n", 1);
    const double E0 = cfg.getDouble("amplitude", 1.0);

    const double kx = m * M_PI / Lx;
    const double ky = n * M_PI / Ly;
    const double omega = std::sqrt(kx * kx + ky * ky);

    // Register analytical initial condition hook: TM_110 mode at t = 0
    cavityCase.setInitialCondition([=](double x, double y, double /*z*/,
                                       double& Ex, double& Ey, double& Ez,
                                       double& Hx, double& Hy, double& Hz) {
        double xt = x - xmin;
        double yt = y - ymin;
        Ex = 0.0;
        Ey = 0.0;
        Ez = E0 * std::sin(kx * xt) * std::sin(ky * yt);
        Hx = 0.0;
        Hy = 0.0;
        Hz = 0.0;
    });

    double finalL2Error = 0.0;
    double finalLinfError = 0.0;

    // Register user postprocessing hook: Compare numerical solution with exact analytical solution
    cavityCase.setPostprocessingHook([&](Case& c) {
        const double tFinal = c.currentTime();
        const auto& mesh = c.mesh();
        const auto& state = c.state();
        const auto& x = mesh.getCoordX();
        const auto& y = mesh.getCoordY();
        const auto& J = mesh.getJac();
        const auto& w = mesh.getGllW();
        const int N = mesh.getN();
        const int Np = N * N * N;
        const int npts = mesh.getTotalPoints();

        const double* EzNum = &state[2 * npts];

        double errSqSum = 0.0;
        double volSum = 0.0;
        double maxErr = 0.0;

        for (int e = 0; e < mesh.getNumElements(); ++e) {
            for (int k = 0; k < N; ++k) {
                for (int j = 0; j < N; ++j) {
                    for (int i = 0; i < N; ++i) {
                        int p = e * Np + k * N * N + j * N + i;
                        double dV = J[p] * w[i] * w[j] * w[k];
                        double xt = x[p] - xmin;
                        double yt = y[p] - ymin;
                        double ezExact = E0 * std::sin(kx * xt) * std::sin(ky * yt) * std::cos(omega * tFinal);
                        double diff = std::abs(EzNum[p] - ezExact);

                        errSqSum += diff * diff * dV;
                        volSum += dV;
                        if (diff > maxErr) maxErr = diff;
                    }
                }
            }
        }

        finalL2Error = std::sqrt(errSqSum / volSum);
        finalLinfError = maxErr;

        std::cout << "\n==========================================================" << std::endl;
        std::cout << "          Analytical Eigenmode Verification Results       " << std::endl;
        std::cout << "==========================================================" << std::endl;
        std::cout << "  Simulation Time (t):     " << std::scientific << std::setprecision(5) << tFinal << std::endl;
        std::cout << "  Theoretical Frequency w: " << std::scientific << std::setprecision(5) << omega << std::endl;
        std::cout << "  Discrete L2 Error:       " << std::scientific << std::setprecision(6) << finalL2Error << std::endl;
        std::cout << "  Discrete Linf Error:     " << std::scientific << std::setprecision(6) << finalLinfError << std::endl;
        std::cout << "==========================================================" << std::endl;
    });

    // 1. Preprocessing phase
    cavityCase.preprocess();

    // 2. Simulation phase
    cavityCase.simulate();

    // 3. Postprocessing phase (triggers the error calculation hook)
    cavityCase.postprocess();

    // Verify error against accuracy threshold (spectral DG accuracy)
    const double tolerance = 5.0e-3;
    if (finalL2Error > tolerance) {
        std::cerr << "Verification FAILED: L2 error " << finalL2Error 
                  << " exceeds threshold " << tolerance << std::endl;
        return 1;
    }

    std::cout << "Verification PASSED: Solution matches analytical eigenmode.\n" << std::endl;
    return 0;
}

