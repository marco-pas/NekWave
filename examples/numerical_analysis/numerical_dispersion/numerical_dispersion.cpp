#include "case.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    std::string parFile = (argc > 1) ? argv[1] : "numerical_dispersion.par";

    std::cout << "==========================================================" << std::endl;
    std::cout << "      NekWave Example: Numerical Dispersion Benchmark     " << std::endl;
    std::cout << "==========================================================" << std::endl;

    Case dispersionCase;
    dispersionCase.loadConfig(parFile);

    dispersionCase.preprocess();
    dispersionCase.simulate();
    dispersionCase.postprocess();

    return 0;
}
