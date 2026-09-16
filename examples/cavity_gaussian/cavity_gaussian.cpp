#include "case.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    std::string parFile = (argc > 1) ? argv[1] : "cavity_gaussian.par";

    std::cout << "==========================================================" << std::endl;
    std::cout << "      NekWave Example: Cavity Gaussian Pulse Benchmark    " << std::endl;
    std::cout << "==========================================================" << std::endl;

    Case cavityCase;
    cavityCase.loadConfig(parFile);

    // 1. Preprocessing phase (mesh, quadrature, metrics, state allocation, probes)
    cavityCase.preprocess();

    // 2. Simulation phase (LSRK45 time stepping and streaming diagnostics)
    cavityCase.simulate();

    // 3. Postprocessing phase (export final fields, finalize probe histories)
    cavityCase.postprocess();

    return 0;
}

