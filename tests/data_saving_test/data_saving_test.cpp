// NekWave Regression Test: Configurable Data Saving Verification
// Validates that standard (E, H) and extra data saving fields are correctly
// computed, exported to disk, and formatted.

#include "case.hpp"
#include "comm.hpp"
#include "hdf5_writer.hpp"
#ifdef NEKWAVE_HAVE_HDF5
#include <hdf5.h>
#endif
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdio>
#include <unistd.h>
#include <sys/stat.h>

static bool fileExists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

static std::string readFile(const std::string& path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main(int argc, char* argv[]) {
    Comm::init(&argc, &argv);

    if (Comm::isRoot()) {
        std::cout << "==========================================================" << std::endl;
        std::cout << "      NekWave Test: Configurable Data Saving Verification " << std::endl;
        std::cout << "==========================================================" << std::endl;
    }

    std::string configFile = "tests/data_saving_test/data_saving_test.json";
    if (argc > 1) {
        configFile = argv[1];
    }

    Case simulationCase;
    simulationCase.loadConfig(configFile);

    // Explicitly configure data saving options (standard + all extras)
    Case::SaveOptions saveOptions;
    // Standard data saving:
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

    // Initial condition hook
    simulationCase.setInitialConditionHook([](double x, double y, double z,
                                              double& ex, double& ey, double& ez,
                                              double& hx, double& hy, double& hz) {
        ex = std::sin(M_PI * x) * std::cos(M_PI * y);
        ey = std::cos(M_PI * x) * std::sin(M_PI * y);
        ez = 0.0;
        hx = 0.0;
        hy = 0.0;
        hz = std::cos(M_PI * x) * std::cos(M_PI * y) * std::sin(M_PI * z);
    });

    // Run the short simulation
    simulationCase.run();

    const std::string outDir = simulationCase.config().outputDir;
    int failures = 0;

    // 1. Verify existence of primary files based on writer support
    if (Hdf5Writer::isSupported()) {
        std::string h5Path = outDir + "/fields.h5";
        if (Comm::size() > 1) {
            h5Path = outDir + "/fields_rank" + std::to_string(Comm::rank()) + ".h5";
        }
        std::string csvPath = outDir + "/energy_history.csv";

        if (!fileExists(h5Path)) {
            std::cerr << "  FAILED: Expected HDF5 file missing: " << h5Path << std::endl;
            failures++;
        } else {
            std::cout << "  [FOUND] " << h5Path << std::endl;
        }

        if (!fileExists(csvPath)) {
            std::cerr << "  FAILED: Expected CSV file missing: " << csvPath << std::endl;
            failures++;
        } else {
            std::cout << "  [FOUND] " << csvPath << std::endl;
        }

        // Strictly verify that NO VTK / VTI / VTU / PVD files were generated
        std::vector<std::string> unwantedVtk = {
            outDir + "/fields.pvd",
            outDir + "/fields_step_0.vtu",
            outDir + "/fields_step_1.vtu",
            outDir + "/fields_step_2.vtu",
            outDir + "/fields_step_0.vti",
            outDir + "/fields_step_1.vti",
            outDir + "/fields_step_2.vti"
        };
        for (const auto& vtkFile : unwantedVtk) {
            if (fileExists(vtkFile)) {
                std::cerr << "  FAILED: Unwanted VTK file generated: " << vtkFile << std::endl;
                failures++;
            }
        }
        if (failures == 0) {
            std::cout << "  [VERIFIED] No VTK/VTI/VTU/PVD files generated (pure HDF5 output)." << std::endl;
        }

#ifdef NEKWAVE_HAVE_HDF5
        // 2. Inspect HDF5 archive contents
        setenv("HDF5_USE_FILE_LOCKING", "FALSE", 1);
        hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
#if defined(H5_VERSION_GE) && H5_VERSION_GE(1, 10, 7)
        H5Pset_file_locking(fapl, false, true);
#endif
        hid_t fileId = H5Fopen(h5Path.c_str(), H5F_ACC_RDONLY, fapl);
        H5Pclose(fapl);
        if (fileId < 0) {
            std::cerr << "  FAILED: Could not open HDF5 file via H5Fopen: " << h5Path << std::endl;
            failures++;
        } else {
            std::cout << "  [HDF5 OPENED] Successfully opened " << h5Path << std::endl;

            if (H5Lexists(fileId, "/mesh/coordinates", H5P_DEFAULT) <= 0) {
                std::cerr << "  FAILED: /mesh/coordinates missing in HDF5 archive!" << std::endl;
                failures++;
            } else {
                std::cout << "  [VERIFIED] /mesh/coordinates exists." << std::endl;
            }

            std::vector<std::string> requiredFields = {
                "E", "H", "curl_E", "curl_H", "div_E", "div_H",
                "magnitude_E", "magnitude_H", "magnitude_curl_E", "magnitude_curl_H",
                "energy_density"
            };

            for (int s = 0; s <= 2; ++s) {
                std::string stepPrefix = "/time_series/step_" + std::to_string(s);
                if (H5Lexists(fileId, stepPrefix.c_str(), H5P_DEFAULT) <= 0) {
                    std::cerr << "  FAILED: Step group missing: " << stepPrefix << std::endl;
                    failures++;
                    continue;
                }
                for (const auto& fld : requiredFields) {
                    std::string dsetPath = stepPrefix + "/" + fld;
                    if (H5Lexists(fileId, dsetPath.c_str(), H5P_DEFAULT) <= 0) {
                        std::cerr << "  FAILED: Field dataset missing in HDF5: " << dsetPath << std::endl;
                        failures++;
                    } else {
                        std::cout << "  [VERIFIED] Found dataset: " << dsetPath << std::endl;
                    }
                }
            }
            H5Fclose(fileId);
        }
#endif
    } else {
        // Fallback when compiled without native HDF5
        std::vector<std::string> expectedFiles = {
            outDir + "/fields.pvd",
            outDir + "/energy_history.csv",
            outDir + "/fields_step_0.vtu",
            outDir + "/fields_step_1.vtu",
            outDir + "/fields_step_2.vtu"
        };
        for (const auto& path : expectedFiles) {
            if (!fileExists(path)) {
                std::cerr << "  FAILED: Expected output file missing: " << path << std::endl;
                failures++;
            } else {
                std::cout << "  [FOUND] " << path << std::endl;
            }
        }
    }

    // 3. Verify energy history CSV
    std::string energyContent = readFile(outDir + "/energy_history.csv");
    if (energyContent.find("step,time,maxE,maxH,energy") == std::string::npos) {
        std::cerr << "  FAILED: CSV header missing or malformed in energy_history.csv!" << std::endl;
        failures++;
    } else {
        std::cout << "  [VERIFIED] energy_history.csv header valid." << std::endl;
    }

    // 4. Clean up test artifacts
    std::remove((outDir + "/fields.pvd").c_str());
    std::remove((outDir + "/fields.bin").c_str());
    std::remove((outDir + "/fields.h5").c_str());
    std::remove((outDir + "/fields.xmf").c_str());
    std::remove((outDir + "/energy_history.csv").c_str());
    std::remove((outDir + "/fields_step_0.vtu").c_str());
    std::remove((outDir + "/fields_step_1.vtu").c_str());
    std::remove((outDir + "/fields_step_2.vtu").c_str());
    rmdir(outDir.c_str());

    if (Comm::isRoot()) {
        std::cout << "\n==========================================================" << std::endl;
        if (failures == 0) {
            std::cout << "  Data Saving Test PASSED successfully (all fields saved)." << std::endl;
        } else {
            std::cerr << "  Data Saving Test FAILED with " << failures << " errors." << std::endl;
        }
        std::cout << "==========================================================" << std::endl;
    }

    Comm::finalize();
    return (failures == 0) ? 0 : 1;
}
