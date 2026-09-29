// NekWave Example: NekCEM Canonical 3D Box Periodic Benchmark Case
// Runs 3D Maxwell simulation on NekCEM 3dboxper.rea mesh with periodic boundary conditions

#include "case.hpp"
#include "comm.hpp"
#include <iostream>
#include <cmath>

int main(int argc, char* argv[]) {
    Comm::init(&argc, &argv);

    if (Comm::isRoot()) {
        std::cout << "==========================================================" << std::endl;
        std::cout << "  NekWave Simulation: NekCEM 3D Box Periodic Cavity Case  " << std::endl;
        std::cout << "==========================================================" << std::endl;
    }

    std::string parFile = "examples/3dboxper/3dboxper.par";
    if (argc > 1) {
        parFile = argv[1];
    }

    Case simulationCase;
    simulationCase.loadConfig(parFile);

    // Explicit field initialization hook defined directly in example driver
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
    saveOptions.saveMagnitudeE = true;
    saveOptions.saveMagnitudeH = true;
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
