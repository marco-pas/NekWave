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
#include "json.hpp"

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
    int numSteps = 100;                   // Total simulation steps (or max_steps)
    int maxSteps = 100;                   // Maximum simulation steps (-1 if unlimited / finalTime governed)
    bool hasExplicitMaxSteps = false;     // True if max_steps or num_steps was explicitly set
    double finalTime = -1.0;              // Simulation termination time (< 0 if unlimited / step governed)
    int outputFreq = 10;                  // Diagnostic and logging frequency (stdout + energy history)
    int saveFreq = -1;                    // Field snapshot export frequency (-1: defaults to outputFreq)
    std::string exportFormat = "hdf5";    // Snapshot export format ("hdf5" or "csv" or "vtk")
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
    bool exportFields = false;            // Whether to export 3D volume nodal fields (field_initial.csv, field_final.csv)
    bool exportContinuousVtk = false;    // Whether to average DG interface nodes into a continuous CG mesh for ParaView
    bool scatteredFieldMode = false;      // True to solve in Scattered-Field formulation (analytical incident wave on PEC, reconstruct total field on save)

    // Boundary condition settings (default outer BC: "PEC"; options: "PEC", "PMC", "PML", "PERIODIC")
    std::string defaultBc = "PEC";
    bool periodicX = false;
    bool periodicY = false;
    bool periodicZ = false;

    // UPML (Perfectly Matched Layer) parameters matching NekCEM's /pmlparam/
    int pmlThickness = 2;                 // Number of element layers in the PML (pmlthick)
    double pmlOrder = 3.0;                // Degree of polynomial conductivity grading (pmlorder)
    double pmlReflectErr = 1.0e-6;        // Desired normal reflection error R(0) (pmlreferr)

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
    /*
     * Applies a key-value parameter setting to standard fields and the general registry.
     */
    void applyKeyValue(const std::string& key, const std::string& val) {
        std::string lkey = toLower(key);
        trim(lkey);
        std::string trimVal = val;
        trim(trimVal);

        if (lkey.empty() || trimVal.empty()) return;

        // Wave type must NOT be specified in configuration files; it must be specified directly in C++ code
        if (lkey == "wave_type") {
            std::cout << "[CONFIG] Notice: 'wave_type' in configuration file is ignored (wave type must be specified directly in C++ code)." << std::endl;
            return;
        }

        params[lkey] = trimVal;

        // Handle multi-value dynamic probe lines: probe = x y z or probe_rel = rx ry rz
        if (lkey == "probe") {
            std::stringstream ss(trimVal);
            double px = 0.0, py = 0.0, pz = 0.0;
            if (ss >> px >> py >> pz) {
                probes.push_back({{px, py, pz}});
            }
            return;
        }
        if (lkey == "probe_rel" || lkey == "relative_probe") {
            std::stringstream ss(trimVal);
            double rx = 0.0, ry = 0.0, rz = 0.0;
            if (ss >> rx >> ry >> rz) {
                probe_rel.push_back({{rx, ry, rz}});
            }
            return;
        }

        // Map standard keys
        if (lkey == "mesh_file" || lkey == "mesh") meshFile = trimVal;
        else if (lkey == "order" || lkey == "n") order = std::stoi(trimVal);
        else if (lkey == "cfl") {
            if (toLower(trimVal) == "auto") cfl = -1.0;
            else cfl = std::stod(trimVal);
        }
        else if (lkey == "dt") {
            if (toLower(trimVal) == "auto") dt = -1.0;
            else dt = std::stod(trimVal);
        }
        else if (lkey == "max_steps" || lkey == "num_steps" || lkey == "steps") {
            maxSteps = std::stoi(trimVal);
            numSteps = maxSteps;
            hasExplicitMaxSteps = true;
        }
        else if (lkey == "final_time" || lkey == "t_final" || lkey == "tfinal" || lkey == "time_final") {
            finalTime = std::stod(trimVal);
        }
        else if (lkey == "output_frequency" || lkey == "output_freq" || lkey == "freq") {
            outputFreq = std::stoi(trimVal);
        }
        else if (lkey == "save_frequency" || lkey == "save_freq" || lkey == "export_freq") {
            saveFreq = std::stoi(trimVal);
        }
        else if (lkey == "export_format" || lkey == "save_format") {
            exportFormat = toLower(trimVal);
        }
        else if (lkey == "pulse_sigma" || lkey == "sigma") pulseSigma = std::stod(trimVal);
        else if (lkey == "c0") c0 = std::stod(trimVal);
        else if (lkey == "flux_type") {
            if (toLower(trimVal) == "central") c0 = 0.0;
            else if (toLower(trimVal) == "upwind") c0 = 1.0;
        }
        else if (lkey == "elements_x" || lkey == "nelx") nelx = std::stoi(trimVal);
        else if (lkey == "elements_y" || lkey == "nely") nely = std::stoi(trimVal);
        else if (lkey == "elements_z" || lkey == "nelz") nelz = std::stoi(trimVal);
        else if (lkey == "l_x" || lkey == "lx" || lkey == "length_x") Lx = std::stod(trimVal);
        else if (lkey == "l_y" || lkey == "ly" || lkey == "length_y") Ly = std::stod(trimVal);
        else if (lkey == "l_z" || lkey == "lz" || lkey == "length_z") Lz = std::stod(trimVal);
        else if (lkey == "xmin") { xmin = std::stod(trimVal); hasExplicitBoundsX = true; }
        else if (lkey == "xmax") { xmax = std::stod(trimVal); hasExplicitBoundsX = true; }
        else if (lkey == "ymin") { ymin = std::stod(trimVal); hasExplicitBoundsY = true; }
        else if (lkey == "ymax") { ymax = std::stod(trimVal); hasExplicitBoundsY = true; }
        else if (lkey == "zmin") { zmin = std::stod(trimVal); hasExplicitBoundsZ = true; }
        else if (lkey == "zmax") { zmax = std::stod(trimVal); hasExplicitBoundsZ = true; }
        else if (lkey == "output_dir") outputDir = trimVal;
        else if (lkey == "export_fields" || lkey == "save_fields") {
            std::string lv = toLower(trimVal);
            exportFields = (lv == "true" || lv == "1" || lv == "yes" || lv == "on");
        }
        else if (lkey == "export_continuous_vtk" || lkey == "continuous_vtk" || lkey == "export_continuous") {
            std::string lv = toLower(trimVal);
            exportContinuousVtk = (lv == "true" || lv == "1" || lv == "yes" || lv == "on");
        }
        else if (lkey == "scattered_field_mode" || lkey == "scatteredfieldmode" || lkey == "scattered_field") {
            std::string lv = toLower(trimVal);
            scatteredFieldMode = (lv == "true" || lv == "1" || lv == "yes" || lv == "on");
        }
        else if (lkey == "periodic_x") {
            std::string lv = toLower(trimVal);
            periodicX = (lv == "true" || lv == "1" || lv == "yes" || lv == "on");
        }
        else if (lkey == "periodic_y") {
            std::string lv = toLower(trimVal);
            periodicY = (lv == "true" || lv == "1" || lv == "yes" || lv == "on");
        }
        else if (lkey == "periodic_z") {
            std::string lv = toLower(trimVal);
            periodicZ = (lv == "true" || lv == "1" || lv == "yes" || lv == "on");
        }
        else if (lkey == "periodic" || lkey == "bc" || lkey == "bc_type" || lkey == "boundary_condition") {
            std::string lv = toLower(trimVal);
            if (lv == "periodic" || lv == "true" || lv == "1" || lv == "yes" || lv == "all" || lv == "p") {
                periodicX = periodicY = periodicZ = true;
                defaultBc = "PERIODIC";
            } else if (lv == "pmc" || lv == "sym" || lv == "s") {
                defaultBc = "PMC";
            } else if (lv == "pml") {
                defaultBc = "PML";
            } else if (lv == "pec" || lv == "w" || lv == "v") {
                defaultBc = "PEC";
            }
        }
        else if (lkey == "pml_thickness" || lkey == "pmlthick" || lkey == "pml_layers") {
            pmlThickness = std::stoi(trimVal);
        }
        else if (lkey == "pml_order" || lkey == "pmlorder") {
            pmlOrder = std::stod(trimVal);
        }
        else if (lkey == "pml_reflect_err" || lkey == "pmlreferr" || lkey == "pml_r0") {
            pmlReflectErr = std::stod(trimVal);
        }
        // Indexed relative probe specifications
        else if (lkey == "probe1_rx" || lkey == "probe1_rel_x") ensureProbeRel(0, 0, std::stod(trimVal));
        else if (lkey == "probe1_ry" || lkey == "probe1_rel_y") ensureProbeRel(0, 1, std::stod(trimVal));
        else if (lkey == "probe1_rz" || lkey == "probe1_rel_z") ensureProbeRel(0, 2, std::stod(trimVal));
        else if (lkey == "probe2_rx" || lkey == "probe2_rel_x") ensureProbeRel(1, 0, std::stod(trimVal));
        else if (lkey == "probe2_ry" || lkey == "probe2_rel_y") ensureProbeRel(1, 1, std::stod(trimVal));
        else if (lkey == "probe2_rz" || lkey == "probe2_rel_z") ensureProbeRel(1, 2, std::stod(trimVal));
        else if (lkey == "probe3_rx" || lkey == "probe3_rel_x") ensureProbeRel(2, 0, std::stod(trimVal));
        else if (lkey == "probe3_ry" || lkey == "probe3_rel_y") ensureProbeRel(2, 1, std::stod(trimVal));
        else if (lkey == "probe3_rz" || lkey == "probe3_rel_z") ensureProbeRel(2, 2, std::stod(trimVal));
    }

    /*
     * Loads parameters from a parsed JSON root object.
     * Supports both structured nested sections (mesh, numerics, time, output, probes)
     * and flat JSON representations.
     */
    void loadFromJson(const nekwave::JsonValue& root) {
        if (!root.isObject()) return;

        // 1. Process mesh section (box mesh or NekCEM .rea mesh)
        if (root.has("mesh")) {
            const auto& m = root["mesh"];
            if (m.isString()) {
                applyKeyValue("mesh", m.asString());
            } else if (m.isObject()) {
                if (m.has("type") && toLower(m["type"].asString()) == "box") {
                    applyKeyValue("mesh", "box");
                }
                if (m.has("file")) {
                    applyKeyValue("mesh", m["file"].asString());
                } else if (m.has("mesh_file")) {
                    applyKeyValue("mesh", m["mesh_file"].asString());
                } else if (m.has("path")) {
                    applyKeyValue("mesh", m["path"].asString());
                }
                if (m.has("nelx") || m.has("elements_x")) {
                    applyKeyValue("mesh", "box");
                }
                for (const auto& k : m.keys()) {
                    if (k != "file" && k != "mesh_file" && k != "path" && k != "type") {
                        applyKeyValue(k, m[k].asString());
                    }
                }
            }
        }

        // 2. Process absolute observation probes
        if (root.has("probes") || root.has("probe")) {
            const auto& p = root.has("probes") ? root["probes"] : root["probe"];
            if (p.isArray()) {
                for (size_t i = 0; i < p.size(); ++i) {
                    const auto& item = p[i];
                    if (item.isArray() && item.size() >= 3) {
                        probes.push_back({{item[0].asDouble(), item[1].asDouble(), item[2].asDouble()}});
                    } else if (item.isString()) {
                        applyKeyValue("probe", item.asString());
                    }
                }
            } else if (p.isString()) {
                applyKeyValue("probe", p.asString());
            }
        }

        // 3. Process relative observation probes
        if (root.has("relative_probes") || root.has("probe_rel")) {
            const auto& pr = root.has("relative_probes") ? root["relative_probes"] : root["probe_rel"];
            if (pr.isArray()) {
                for (size_t i = 0; i < pr.size(); ++i) {
                    const auto& item = pr[i];
                    if (item.isArray() && item.size() >= 3) {
                        probe_rel.push_back({{item[0].asDouble(), item[1].asDouble(), item[2].asDouble()}});
                    } else if (item.isString()) {
                        applyKeyValue("probe_rel", item.asString());
                    }
                }
            } else if (pr.isString()) {
                applyKeyValue("probe_rel", pr.asString());
            }
        }

        // 4. Process nested sections (numerics, time, output, etc.) and top-level keys
        for (const auto& k : root.keys()) {
            if (k == "_comment" || k == "comment") continue;
            if (k == "mesh" || k == "probes" || k == "probe" || k == "relative_probes" || k == "probe_rel") continue;

            const auto& v = root[k];
            if (v.isObject()) {
                for (const auto& subk : v.keys()) {
                    if (subk == "_comment" || subk == "comment") continue;
                    applyKeyValue(subk, v[subk].asString());
                }
            } else if (!v.isArray()) {
                applyKeyValue(k, v.asString());
            }
        }
    }

    /*
     * Loads and parses a case JSON configuration file.
     */
    bool loadFromFile(const std::string& filename) {
        std::vector<std::string> candidates = { filename };

        if (filename.size() > 4 && filename.substr(filename.size() - 4) == ".par") {
            candidates.push_back(filename.substr(0, filename.size() - 4) + ".json");
        } else if (filename.size() > 5 && filename.substr(filename.size() - 5) == ".json") {
            candidates.push_back(filename.substr(0, filename.size() - 5) + ".par");
        } else {
            candidates.push_back(filename + ".json");
            candidates.push_back(filename + ".par");
        }

        std::ifstream file;
        std::string resolvedPath = "";
        for (const auto& cand : candidates) {
            file.open(cand);
            if (file.is_open()) { resolvedPath = cand; break; }
            std::string parentPath = "../" + cand;
            file.open(parentPath);
            if (file.is_open()) { resolvedPath = parentPath; break; }
            std::string grandParentPath = "../../" + cand;
            file.open(grandParentPath);
            if (file.is_open()) { resolvedPath = grandParentPath; break; }
        }

        if (!file.is_open()) {
            std::cerr << "Warning: Could not open config file '" << filename 
                      << "'. Using default parameters." << std::endl;
            return false;
        }

        std::string content((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
        file.close();

        // Check if content begins with JSON structure (skipping whitespace/comments)
        size_t p = 0;
        size_t n = content.length();
        bool isJson = false;
        while (p < n) {
            char c = content[p];
            if (std::isspace(static_cast<unsigned char>(c))) {
                p++;
                continue;
            }
            if (c == '/' && p + 1 < n && content[p + 1] == '/') {
                p += 2;
                while (p < n && content[p] != '\n' && content[p] != '\r') p++;
                continue;
            }
            if (c == '/' && p + 1 < n && content[p + 1] == '*') {
                p += 2;
                while (p + 1 < n && !(content[p] == '*' && content[p + 1] == '/')) p++;
                if (p + 1 < n) p += 2;
                continue;
            }
            if (c == '#') {
                p++;
                while (p < n && content[p] != '\n' && content[p] != '\r') p++;
                continue;
            }
            if (c == '{' || c == '[') {
                isJson = true;
            }
            break;
        }

        if (isJson || filename.find(".json") != std::string::npos) {
            std::string errMsg;
            nekwave::JsonValue root = nekwave::JsonValue::parse(content, &errMsg);
            if (!errMsg.empty()) {
                std::cerr << "Warning: JSON parse error in '" << filename << "': " << errMsg << std::endl;
            }
            if (root.isObject()) {
                loadFromJson(root);
            }
        } else {
            // Legacy line-by-line key = value parsing
            std::istringstream ss(content);
            std::string line;
            while (std::getline(ss, line)) {
                auto commentPos = line.find('#');
                if (commentPos != std::string::npos) {
                    line = line.substr(0, commentPos);
                }
                auto sepPos = line.find('=');
                if (sepPos == std::string::npos) {
                    sepPos = line.find(':');
                }
                if (sepPos == std::string::npos) continue;

                std::string key = line.substr(0, sepPos);
                std::string val = line.substr(sepPos + 1);
                trim(key);
                trim(val);
                if (!key.empty() && !val.empty()) {
                    applyKeyValue(key, val);
                }
            }
        }

        // If final_time was explicitly specified without an explicit max_steps,
        // let final_time drive simulation termination
        if (finalTime > 0.0 && !hasExplicitMaxSteps) {
            maxSteps = -1;
            numSteps = -1;
        }

        // Default saveFreq to outputFreq if not set
        if (saveFreq <= 0) {
            saveFreq = outputFreq;
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
