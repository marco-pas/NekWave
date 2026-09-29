#include "case.hpp"
#include "comm.hpp"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    Comm::init(&argc, &argv);

    if (Comm::isRoot()) {
        std::cout << "\n----------------------------------------------------------" << std::endl;
        std::cout << "             NekWave: C++11 DG Maxwell Solver             " << std::endl;
        std::cout << "----------------------------------------------------------\n" << std::endl;
    }

    if (argc < 2) {
        if (Comm::isRoot()) {
            std::cerr << "Usage: " << argv[0] << " <path-to-case.par>\n\n"
                      << "Available example cases:\n"
                      << "  " << argv[0] << " examples/3dboxpec/3dboxpec.par\n"
                      << "  " << argv[0] << " examples/3dboxper/3dboxper.par\n" << std::endl;
        }
        Comm::finalize();
        return 1;
    }

    std::string configFile = argv[1];
    Case simCase;

    // Check command line arguments
    if (configFile.find(".rea") != std::string::npos) {
        Config cfg;
        cfg.meshFile = configFile;
        if (Comm::isRoot()) {
            std::cout << "Using mesh file from CLI: " << configFile << std::endl;
        }
        simCase.setConfig(cfg);
    } else {
        if (Comm::isRoot()) {
            std::cout << "Loading configuration: " << configFile << std::endl;
        }
        if (!simCase.config().loadFromFile(configFile)) {
            if (Comm::isRoot()) {
                std::cerr << "Error: Failed to load configuration from " << configFile << std::endl;
            }
            Comm::finalize();
            return 1;
        }
    }

    // Explicit 3-phase simulation pipeline
    simCase.preprocess();
    simCase.simulate();
    simCase.postprocess();

    Comm::finalize();
    return 0;
}
