#ifndef NW_SRC_CONFIG_HPP
#define NW_SRC_CONFIG_HPP

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <array>
#include <unordered_map>

/*
 * Extensible simulation configuration container.
 *
 * Parses key-value parameter files, maps standard DG-SEM settings to
 * typed member fields, and stores arbitrary case-specific parameters
 * in a general key-value registry accessible via typed getters.
 */
struct Config {
    // Standard DG-SEM simulation settings
    std::string meshFile = "";            // Path to mesh file (.rea or custom)
    int order = 3;                        // Polynomial order N (collocation points per direction)
    double cfl = -1.0;                    // CFL number (negative: auto-calculated)
    double dt = -1.0;                     // Time step size (negative: auto-calculated)
    int numSteps = 100;                   // Total simulation steps
    int outputFreq = 10;                  // Diagnostic and logging frequency
    double pulseSigma = 20.0;             // Gaussian pulse spatial width
    double c0 = 1.0;                      // Flux penalty: 0.0 (central), 1.0 (upwind)
    int nelx = 0;                         // Cartesian box element count in X
    int nely = 0;                         // Cartesian box element count in Y
    int nelz = 0;                         // Cartesian box element count in Z
    double Lx = 2.0, Ly = 2.0, Lz = 2.0;  // Box physical dimensions
    double xmin = -1.0, xmax = 1.0;       // Domain bounding box
    double ymin = -1.0, ymax = 1.0;
    double zmin = -1.0, zmax = 1.0;
    bool hasExplicitBoundsX = false;
    bool hasExplicitBoundsY = false;
    bool hasExplicitBoundsZ = false;
    std::string outputDir = "output";     // Output directory for CSVs and plots

    // Observation probe coordinates
    std::vector<std::array<double, 3>> probe_rel; // Relative coordinates in [-0.5, 0.5]^3
    std::vector<std::array<double, 3>> probes;    // Absolute physical coordinates

    // Generic parameter registry for user-defined case parameters
    std::unordered_map<std::string, std::string> params;

    /*
     * Checks if a parameter exists in the registry.
     */
    bool has(const std::string& key) const {
        std::string lk = toLower(key);
        return params.find(lk) != params.end();
    }

    /*
     * Retrieves a string parameter.
     */
    std::string getString(const std::string& key, const std::string& defaultVal = "") const {
        std::string lk = toLower(key);
        auto it = params.find(lk);
        return (it != params.end()) ? it->second : defaultVal;
    }

    /*
     * Retrieves a floating-point parameter.
     */
    double getDouble(const std::string& key, double defaultVal = 0.0) const {
        std::string lk = toLower(key);
        auto it = params.find(lk);
        if (it != params.end()) {
            try {
                return std::stod(it->second);
            } catch (...) {
                return defaultVal;
            }
        }
        return defaultVal;
    }

    /*
     * Retrieves an integer parameter.
     */
    int getInt(const std::string& key, int defaultVal = 0) const {
        std::string lk = toLower(key);
        auto it = params.find(lk);
        if (it != params.end()) {
            try {
                return std::stoi(it->second);
            } catch (...) {
                return defaultVal;
            }
        }
        return defaultVal;
    }

    /*
     * Retrieves a boolean parameter.
     */
    bool getBool(const std::string& key, bool defaultVal = false) const {
        std::string lk = toLower(key);
        auto it = params.find(lk);
        if (it != params.end()) {
            std::string val = toLower(it->second);
            return (val == "true" || val == "1" || val == "yes" || val == "on");
        }
        return defaultVal;
    }

    /*
     * Sets or updates a parameter in the registry.
     */
    void set(const std::string& key, const std::string& val) {
        params[toLower(key)] = val;
    }

    /*
     * Loads and parses a case parameter file.
     */
    bool loadFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            // Check parent directory if running from a build/ or example/ directory
            std::string parentPath = "../" + filename;
            file.open(parentPath);
        }
        if (!file.is_open()) {
            std::string grandParentPath = "../../" + filename;
            file.open(grandParentPath);
        }
        if (!file.is_open()) {
            std::cerr << "Warning: Could not open config file '" << filename 
                      << "'. Using default parameters." << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(file, line)) {
            // Strip comments starting with #
            auto commentPos = line.find('#');
            if (commentPos != std::string::npos) {
                line = line.substr(0, commentPos);
            }

            // Find key = value or key: value
            auto sepPos = line.find('=');
            if (sepPos == std::string::npos) {
                sepPos = line.find(':');
            }
            if (sepPos == std::string::npos) continue;

            std::string key = line.substr(0, sepPos);
            std::string val = line.substr(sepPos + 1);

            trim(key);
            trim(val);

            if (key.empty() || val.empty()) continue;

            std::string lkey = toLower(key);
            params[lkey] = val;

            // Handle multi-value dynamic probe lines: probe = x y z or probe_rel = rx ry rz
            if (lkey == "probe") {
                std::stringstream ss(val);
                double px = 0.0, py = 0.0, pz = 0.0;
                if (ss >> px >> py >> pz) {
                    probes.push_back({{px, py, pz}});
                }
                continue;
            }
            if (lkey == "probe_rel" || lkey == "relative_probe") {
                std::stringstream ss(val);
                double rx = 0.0, ry = 0.0, rz = 0.0;
                if (ss >> rx >> ry >> rz) {
                    probe_rel.push_back({{rx, ry, rz}});
                }
                continue;
            }

            // Map standard keys
            if (lkey == "mesh_file" || lkey == "mesh") meshFile = val;
            else if (lkey == "order" || lkey == "n") order = std::stoi(val);
            else if (lkey == "cfl") {
                if (toLower(val) == "auto") cfl = -1.0;
                else cfl = std::stod(val);
            }
            else if (lkey == "dt") {
                if (toLower(val) == "auto") dt = -1.0;
                else dt = std::stod(val);
            }
            else if (lkey == "num_steps" || lkey == "steps") numSteps = std::stoi(val);
            else if (lkey == "output_freq" || lkey == "freq") outputFreq = std::stoi(val);
            else if (lkey == "pulse_sigma" || lkey == "sigma") pulseSigma = std::stod(val);
            else if (lkey == "c0") c0 = std::stod(val);
            else if (lkey == "flux_type") {
                if (toLower(val) == "central") c0 = 0.0;
                else if (toLower(val) == "upwind") c0 = 1.0;
            }
            else if (lkey == "elements_x" || lkey == "nelx") nelx = std::stoi(val);
            else if (lkey == "elements_y" || lkey == "nely") nely = std::stoi(val);
            else if (lkey == "elements_z" || lkey == "nelz") nelz = std::stoi(val);
            else if (lkey == "l_x" || lkey == "lx" || lkey == "length_x") Lx = std::stod(val);
            else if (lkey == "l_y" || lkey == "ly" || lkey == "length_y") Ly = std::stod(val);
            else if (lkey == "l_z" || lkey == "lz" || lkey == "length_z") Lz = std::stod(val);
            else if (lkey == "xmin") { xmin = std::stod(val); hasExplicitBoundsX = true; }
            else if (lkey == "xmax") { xmax = std::stod(val); hasExplicitBoundsX = true; }
            else if (lkey == "ymin") { ymin = std::stod(val); hasExplicitBoundsY = true; }
            else if (lkey == "ymax") { ymax = std::stod(val); hasExplicitBoundsY = true; }
            else if (lkey == "zmin") { zmin = std::stod(val); hasExplicitBoundsZ = true; }
            else if (lkey == "zmax") { zmax = std::stod(val); hasExplicitBoundsZ = true; }
            else if (lkey == "output_dir") outputDir = val;
            // Indexed relative probe specifications
            else if (lkey == "probe1_rx" || lkey == "probe1_rel_x") ensureProbeRel(0, 0, std::stod(val));
            else if (lkey == "probe1_ry" || lkey == "probe1_rel_y") ensureProbeRel(0, 1, std::stod(val));
            else if (lkey == "probe1_rz" || lkey == "probe1_rel_z") ensureProbeRel(0, 2, std::stod(val));
            else if (lkey == "probe2_rx" || lkey == "probe2_rel_x") ensureProbeRel(1, 0, std::stod(val));
            else if (lkey == "probe2_ry" || lkey == "probe2_rel_y") ensureProbeRel(1, 1, std::stod(val));
            else if (lkey == "probe2_rz" || lkey == "probe2_rel_z") ensureProbeRel(1, 2, std::stod(val));
            else if (lkey == "probe3_rx" || lkey == "probe3_rel_x") ensureProbeRel(2, 0, std::stod(val));
            else if (lkey == "probe3_ry" || lkey == "probe3_rel_y") ensureProbeRel(2, 1, std::stod(val));
            else if (lkey == "probe3_rz" || lkey == "probe3_rel_z") ensureProbeRel(2, 2, std::stod(val));
        }

        // Synchronize bounding coordinates and physical box dimensions
        if (hasExplicitBoundsX) {
            Lx = xmax - xmin;
        } else {
            xmin = -Lx / 2.0;
            xmax =  Lx / 2.0;
        }

        if (hasExplicitBoundsY) {
            Ly = ymax - ymin;
        } else {
            ymin = -Ly / 2.0;
            ymax =  Ly / 2.0;
        }

        if (hasExplicitBoundsZ) {
            Lz = zmax - zmin;
        } else {
            zmin = -Lz / 2.0;
            zmax =  Lz / 2.0;
        }

        // Convert relative probe coordinates to absolute physical positions
        double xc = 0.5 * (xmin + xmax);
        double yc = 0.5 * (ymin + ymax);
        double zc = 0.5 * (zmin + zmax);

        for (const auto& r : probe_rel) {
            double px = xc + r[0] * Lx;
            double py = yc + r[1] * Ly;
            double pz = zc + r[2] * Lz;
            probes.push_back({{px, py, pz}});
        }

        return true;
    }

private:
    static std::string toLower(const std::string& s) {
        std::string res = s;
        std::transform(res.begin(), res.end(), res.begin(), [](unsigned char c) {
            return std::tolower(c);
        });
        return res;
    }

    static void trim(std::string& s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
            return !std::isspace(ch);
        }));
        s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(), s.end());
    }

    void ensureProbeRel(size_t index, int component, double val) {
        if (probe_rel.size() <= index) {
            probe_rel.resize(index + 1, {{0.0, 0.0, 0.0}});
        }
        probe_rel[index][component] = val;
    }
};

#endif // NW_SRC_CONFIG_HPP
