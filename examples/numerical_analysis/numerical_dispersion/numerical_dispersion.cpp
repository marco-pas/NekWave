#include "case.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    std::string configFile = (argc > 1) ? argv[1] : "numerical_dispersion.json";

    std::cout << "==========================================================" << std::endl;
    std::cout << "      NekWave Example: Numerical Dispersion Benchmark     " << std::endl;
    std::cout << "==========================================================" << std::endl;

    Case dispersionCase;
    dispersionCase.loadConfig(configFile);
    dispersionCase.setWaveType("bloch");

    dispersionCase.preprocess();
    dispersionCase.simulate();
    dispersionCase.postprocess();

    return 0;
}
