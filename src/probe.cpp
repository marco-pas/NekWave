#include "probe.hpp"
#include "mesh.hpp"
#include <sys/stat.h>
#include <sys/types.h>

// @@ defining the destructor to guarantee all output file streams are cleanly flushed and closed
ProbeManager::~ProbeManager() {
    finalize();
}

// @@ multi-point observation probes and searching for closest mesh GLL collocation nodes
void ProbeManager::init(const Mesh& mesh, 
                        const std::vector<std::array<double, 3>>& targetCoords, 
                        const std::string& outputDir) {
    m_outputDir = outputDir;
    m_probes.clear();

    // Ensure output directory exists (POSIX mkdir)
    mkdir(m_outputDir.c_str(), 0755);

    int npts = mesh.getTotalPoints();
    const auto& x = mesh.getCoordX();
    const auto& y = mesh.getCoordY();
    const auto& z = mesh.getCoordZ();

    // Find closest collocation node for each probe target
    for (size_t i = 0; i < targetCoords.size(); ++i) {
        ProbePoint p;
        p.id = static_cast<int>(i + 1);
        p.targetX = targetCoords[i][0];
        p.targetY = targetCoords[i][1];
        p.targetZ = targetCoords[i][2];

        double minDist2 = 1.0e20;
        int bestIdx = 0;
        for (int k = 0; k < npts; ++k) {
            double dx = x[k] - p.targetX;
            double dy = y[k] - p.targetY;
            double dz = z[k] - p.targetZ;
            double d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < minDist2) {
                minDist2 = d2;
                bestIdx = k;
            }
        }

        p.nodeIdx = bestIdx;
        p.actualX = x[bestIdx];
        p.actualY = y[bestIdx];
        p.actualZ = z[bestIdx];
        p.dist = std::sqrt(minDist2);

        m_probes.push_back(p);
    }

    // Open combined multi-probe time history CSV
    std::string filename = m_outputDir + "/probe_history.csv";
    m_combinedFile.open(filename);
    if (!m_combinedFile.is_open()) {
        std::cerr << "Warning: Could not open probe output file: " << filename << std::endl;
        return;
    }

    // Write header: step, time, then Ex_k, Ey_k, Ez_k, Hx_k, Hy_k, Hz_k for each probe k
    m_combinedFile << "step,time";
    for (const auto& pr : m_probes) {
        m_combinedFile << ",Ex_" << pr.id << ",Ey_" << pr.id << ",Ez_" << pr.id
                       << ",Hx_" << pr.id << ",Hy_" << pr.id << ",Hz_" << pr.id;
    }
    m_combinedFile << "\n";
    m_combinedFile << std::scientific << std::setprecision(8);

    m_initialized = true;

    // Print probe locations summary
    std::cout << "\nObservation Probes Configured (" << m_probes.size() << " locations):" << std::endl;
    for (const auto& pr : m_probes) {
        std::cout << "  Probe " << pr.id 
                  << ": Target (" << pr.targetX << ", " << pr.targetY << ", " << pr.targetZ << ") -> "
                  << "Node #" << pr.nodeIdx << " (" << pr.actualX << ", " << pr.actualY << ", " << pr.actualZ << ")"
                  << " [offset: " << std::scientific << std::setprecision(2) << pr.dist << "]" << std::endl;
    }
    std::cout << "Streaming probe history to: " << filename << std::endl;
}

// @@ recording field values at all observation probes for the current time step
void ProbeManager::record(int step, double time, const std::vector<double>& state, int npts) {
    if (!m_combinedFile.is_open()) return;

    m_combinedFile << step << "," << time;

    for (const auto& pr : m_probes) {
        int p = pr.nodeIdx;
        double Ex = state[0 * npts + p];
        double Ey = state[1 * npts + p];
        double Ez = state[2 * npts + p];
        double Hx = state[3 * npts + p];
        double Hy = state[4 * npts + p];
        double Hz = state[5 * npts + p];

        m_combinedFile << "," << Ex << "," << Ey << "," << Ez
                       << "," << Hx << "," << Hy << "," << Hz;
    }
    m_combinedFile << "\n";
}

// @@ finalizing probe streams and closing file handles
void ProbeManager::finalize() {
    if (m_combinedFile.is_open()) {
        m_combinedFile.close();
        std::cout << "Exported multi-probe time history to " << m_outputDir << "/probe_history.csv" << std::endl;
    }
    m_initialized = false;
}

