// NekWave Example: NekCEM Canonical 3D Box PEC Benchmark Case
// Runs 3D Maxwell simulation on NekCEM 3dboxpec.rea mesh with PEC boundary conditions

#include "case.hpp"
#include "comm.hpp"
#include <iostream>
#include <cmath>

int main(int argc, char* argv[]) {
    Comm::init(&argc, &argv);

    if (Comm::isRoot()) {
        std::cout << "==========================================================" << std::endl;
        std::cout << "    NekWave Simulation: NekCEM 3D Box PEC Cavity Case     " << std::endl;
        std::cout << "==========================================================" << std::endl;
    }

    std::string parFile = "examples/3dboxpec/3dboxpec.par";
    if (argc > 1) {
        parFile = argv[1];
    }

    Case simulationCase;
    simulationCase.loadConfig(parFile);

    // Explicit field initialization hook defined directly in example driver
    simulationCase.setInitialConditionHook([](double x, double y, double z,
                                              double& ex, double& ey, double& ez,
                                              double& hx, double& hy, double& hz) {
        // const double invSqrt6 = 1.0 / std::sqrt(6.0);
        // double sx = std::sin(M_PI * x);
        // double cx = std::cos(M_PI * x);
        // double sy = std::sin(M_PI * y);
        // double cy = std::cos(M_PI * y);
        // double sz = std::sin(M_PI * z);
        // double cz = std::cos(M_PI * z);

        // ex = 0.0; 
        // ey = 0.0;
        // ez = 0.0;

        // hx = -sx * cy * cz * invSqrt6;
        // hy = -cx * sy * cz * invSqrt6;
        // hz =  2.0 * cx * cy * sz * invSqrt6;

        // Pre-compute basic arguments
        double pi_x = M_PI * x;
        double pi_y = M_PI * y;
        double pi_z = M_PI * z;

        // Base frequencies (m, n, p = 1)
        double s1x = std::sin(pi_x), c1x = std::cos(pi_x);
        double s1y = std::sin(pi_y), c1y = std::cos(pi_y);
        double s1z = std::sin(pi_z), c1z = std::cos(pi_z);

        // First harmonic frequencies (m, n, p = 2)
        double s2x = std::sin(2.0 * pi_x), c2x = std::cos(2.0 * pi_x);
        double s2y = std::sin(2.0 * pi_y), c2y = std::cos(2.0 * pi_y);
        double s2z = std::sin(2.0 * pi_z), c2z = std::cos(2.0 * pi_z);

        // Electric field starts at 0
        ex = 0.0; 
        ey = 0.0; 
        ez = 0.0;

        // Mode 1: Frequencies (2, 1, 1). Amplitudes (1, -1, -1). 
        // Div constraint: 1(2) + (-1)(1) + (-1)(1) = 0
        double hx1 =  1.0 * s2x * c1y * c1z;
        double hy1 = -1.0 * c2x * s1y * c1z;
        double hz1 = -1.0 * c2x * c1y * s1z;

        // Mode 2: Frequencies (1, 2, 1). Amplitudes (-1, 1, -1).
        // Div constraint: (-1)(1) + 1(2) + (-1)(1) = 0
        double hx2 = -1.0 * s1x * c2y * c1z;
        double hy2 =  1.0 * c1x * s2y * c1z;
        double hz2 = -1.0 * c1x * c2y * s1z;

        // Mode 3: Frequencies (1, 1, 2). Amplitudes (-1, -1, 1).
        // Div constraint: (-1)(1) + (-1)(1) + 1(2) = 0
        double hx3 = -1.0 * s1x * c1y * c2z;
        double hy3 = -1.0 * c1x * s1y * c2z;
        double hz3 =  1.0 * c1x * c1y * s2z;

        // Superposition 
        // You can multiply this by a normalization constant like your invSqrt6 if needed
        hx = hx1 + hx2 + hx3;
        hy = hy1 + hy2 + hy3;
        hz = hz1 + hz2 + hz3;
    });

    // Configure data saving fields directly in example driver
    Case::SaveOptions saveOptions;
    // Standard data saving: E and H fields
    saveOptions.saveE = true;
    saveOptions.saveH = true;
    // Extra data savings:
    saveOptions.saveCurlE = false;
    saveOptions.saveCurlH = false;
    saveOptions.saveDivE = true;
    saveOptions.saveDivH = true;
    saveOptions.saveMagnitudeE = false;
    saveOptions.saveMagnitudeH = false;
    saveOptions.saveMagnitudeCurlE = false;
    saveOptions.saveMagnitudeCurlH = false;
    saveOptions.saveEnergyDensity = true;
    simulationCase.setSaveOptions(saveOptions);

    // Run simulation workflow
    simulationCase.preprocess();
    simulationCase.simulate();
    simulationCase.postprocess();

    if (Comm::isRoot()) {
        std::cout << "Simulation completed successfully.\n" << std::endl;
    }
    Comm::finalize();
    return 0;
}
