#ifndef NW_SRC_PROBE_HPP
#define NW_SRC_PROBE_HPP

#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>

class Mesh;

// @@ data structure for a single spatial observation probe
// Stores target coordinate, mapped GLL node index, and actual physical coordinate
struct ProbePoint {
    int id = 0;
    double targetX = 0.0;
    double targetY = 0.0;
    double targetZ = 0.0;
    int nodeIdx = -1;
    double actualX = 0.0;
    double actualY = 0.0;
    double actualZ = 0.0;
    double dist = 0.0;

    // Spectral element Lagrange polynomial interpolation
    int elemOffset = 0;
    std::vector<double> interpWeights;
};

// @@ ProbeManager class to orchestrate multi-point field observation and logging
// Encapsulates nearest-node mapping and high-frequency CSV time-series streaming
class ProbeManager {
public:
    ProbeManager() = default;
    ~ProbeManager();

    void init(const Mesh& mesh, 
              const std::vector<std::array<double, 3>>& targetCoords, 
              const std::string& outputDir = "output",
              int numModes = 1,
              double domainLx = 1.0,
              int elementsX = 1,
              const std::vector<double>& modeK = {});

    // Record electromagnetic field values at all probe nodes for the current time step
    void record(int step, double time, const std::vector<double>& state, int npts);

    // Close output streams and flush data
    void finalize();

    // Accessors
    const std::vector<ProbePoint>& getProbes() const { return m_probes; }
    size_t getNumProbes() const { return m_probes.size(); }

private:
    std::vector<ProbePoint> m_probes;
    std::string m_outputDir;
    std::ofstream m_combinedFile;
    bool m_initialized = false;
};

#endif // NW_SRC_PROBE_HPP

