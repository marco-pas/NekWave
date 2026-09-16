#include "case.hpp"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    std::cout << "\n----------------------------------------------------------" << std::endl;
    std::cout << "             NekWave: C++11 DG Maxwell Solver             " << std::endl;
    std::cout << "----------------------------------------------------------\n" << std::endl;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path-to-case.par>\n\n"
                  << "Available example cases:\n"
                  << "  " << argv[0] << " examples/cavity_gaussian/cavity_gaussian.par\n"
                  << "  " << argv[0] << " examples/cavity_eigenmode/cavity_eigenmode.par\n" << std::endl;
        return 1;
    }

    std::string configFile = argv[1];
    Case simCase;

    // Check command line arguments
    if (configFile.find(".rea") != std::string::npos) {
        Config cfg;
        cfg.meshFile = configFile;
        std::cout << "Using mesh file from CLI: " << configFile << std::endl;
        simCase.setConfig(cfg);
    } else {
        std::cout << "Loading configuration: " << configFile << std::endl;
        if (!simCase.config().loadFromFile(configFile)) {
            std::cerr << "Error: Failed to load configuration from " << configFile << std::endl;
            return 1;
        }
    }

    // Explicit 3-phase simulation pipeline
    simCase.preprocess();
    simCase.simulate();
    simCase.postprocess();

    return 0;
}
