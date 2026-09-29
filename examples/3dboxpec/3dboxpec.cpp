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

    // Configure data saving fields directly in example driver
    Case::SaveOptions saveOptions;
    // Standard data saving: E and H fields
    saveOptions.saveE = true;
    saveOptions.saveH = true;
    // Extra data savings:
    saveOptions.saveCurlE = true;
    saveOptions.saveCurlH = true;
    saveOptions.saveDivE = true;
    saveOptions.saveDivH = true;
    saveOptions.saveMagnitudeE = true;
    saveOptions.saveMagnitudeH = true;
    saveOptions.saveMagnitudeCurlE = true;
    saveOptions.saveMagnitudeCurlH = true;
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
