#include "probe.hpp"
#include "mesh.hpp"
#include <sys/stat.h>
#include <sys/types.h>

// @@ defining the destructor to guarantee all output file streams are cleanly flushed and closed
ProbeManager::~ProbeManager() {
    finalize();
}

// 1D Lagrange polynomial basis evaluation on GLL nodes
static std::vector<double> evalLagrange1D(const std::vector<double>& gllZ, double xi) {
    int N = static_cast<int>(gllZ.size());
    std::vector<double> L(N, 1.0);
    for (int i = 0; i < N; ++i) {
        if (std::abs(xi - gllZ[i]) < 1e-12) {
            std::fill(L.begin(), L.end(), 0.0);
            L[i] = 1.0;
            return L;
        }
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i != j) {
                L[i] *= (xi - gllZ[j]) / (gllZ[i] - gllZ[j]);
            }
        }
    }
    return L;
}

// @@ multi-point observation probes with spectral polynomial interpolation
void ProbeManager::init(const Mesh& mesh, 
                        const std::vector<std::array<double, 3>>& targetCoords, 
                        const std::string& outputDir,
                        int numModes,
                        double domainLx,
                        int elementsX,
                        const std::vector<double>& modeK) {
    m_outputDir = outputDir;
    m_probes.clear();

    bool bSaveToFile = (!m_outputDir.empty() && m_outputDir != "none" && m_outputDir != "None" && m_outputDir != "NONE");

    // Ensure output directory exists (POSIX mkdir) if output is enabled
    if (bSaveToFile) {
        mkdir(m_outputDir.c_str(), 0755);
    }

    int N = mesh.getN();
    const auto& gllZ = mesh.getGllZ();
    const auto& elementCorners = mesh.getElementCorners();
    int numElements = mesh.getNumElements();
    int ptsPerElem = N * N * N;

    // Build spectral Lagrange interpolation weights for each probe target
    for (size_t i = 0; i < targetCoords.size(); ++i) {
        ProbePoint p;
        p.id = static_cast<int>(i + 1);
        p.targetX = targetCoords[i][0];
        p.targetY = targetCoords[i][1];
        p.targetZ = targetCoords[i][2];

        // Find element containing this target point
        int bestElem = 0;
        double refR = 0.0, refS = 0.0, refT = 0.0;
        double bestDistToElem = 1e20;

        for (int e = 0; e < numElements; ++e) {
            const auto& c = elementCorners[e];
            double x0 = c[0][0], x1 = c[1][0];
            double y0 = c[0][1], y1 = c[2][1];
            double z0 = c[0][2], z1 = c[4][2];

            double r = (std::abs(x1 - x0) > 1e-14) ? 2.0 * (p.targetX - x0) / (x1 - x0) - 1.0 : 0.0;
            double s = (std::abs(y1 - y0) > 1e-14) ? 2.0 * (p.targetY - y0) / (y1 - y0) - 1.0 : 0.0;
            double t = (std::abs(z1 - z0) > 1e-14) ? 2.0 * (p.targetZ - z0) / (z1 - z0) - 1.0 : 0.0;

            double dr = std::max(0.0, std::max(-1.0 - r, r - 1.0));
            double ds = std::max(0.0, std::max(-1.0 - s, s - 1.0));
            double dt = std::max(0.0, std::max(-1.0 - t, t - 1.0));
            double boxDist2 = dr * dr + ds * ds + dt * dt;

            if (boxDist2 < bestDistToElem) {
                bestDistToElem = boxDist2;
                bestElem = e;
                refR = std::max(-1.0, std::min(1.0, r));
                refS = std::max(-1.0, std::min(1.0, s));
                refT = std::max(-1.0, std::min(1.0, t));
            }
        }

        p.elemOffset = bestElem * ptsPerElem;
        std::vector<double> Lr = evalLagrange1D(gllZ, refR);
        std::vector<double> Ls = evalLagrange1D(gllZ, refS);
        std::vector<double> Lt = evalLagrange1D(gllZ, refT);

        p.interpWeights.resize(ptsPerElem);
        for (int k = 0; k < N; ++k) {
            for (int j = 0; j < N; ++j) {
                for (int ii = 0; ii < N; ++ii) {
                    p.interpWeights[ii + N * (j + N * k)] = Lr[ii] * Ls[j] * Lt[k];
                }
            }
        }

        // Exact physical coordinates via spectral element interpolation
        p.actualX = p.targetX;
        p.actualY = p.targetY;
        p.actualZ = p.targetZ;
        p.dist = 0.0;
        p.nodeIdx = -1;

        m_probes.push_back(p);
    }

    // Open combined multi-probe time history CSV if output is enabled
    if (bSaveToFile && !m_probes.empty()) {
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

        // Export probe coordinates and exact separation distance to probes.json
        std::string jsonPath = m_outputDir + "/probes.json";
        std::ofstream jsonFile(jsonPath);
        if (jsonFile.is_open()) {
            jsonFile << "{\n  \"order\": " << N << ",\n";
            jsonFile << "  \"num_modes\": " << numModes << ",\n";
            jsonFile << "  \"num_probes\": " << m_probes.size() << ",\n";
            if (m_probes.size() >= 2) {
                double d_x = m_probes[1].actualX - m_probes[0].actualX;
                double d_y = m_probes[1].actualY - m_probes[0].actualY;
                double d_z = m_probes[1].actualZ - m_probes[0].actualZ;
                double dist = std::sqrt(d_x * d_x + d_y * d_y + d_z * d_z);
                jsonFile << "  \"separation_distance\": " << std::scientific << std::setprecision(8) << dist << ",\n";
                jsonFile << "  \"delta_x\": " << d_x << ",\n";
                jsonFile << "  \"delta_y\": " << d_y << ",\n";
                jsonFile << "  \"delta_z\": " << d_z << ",\n";
                jsonFile << "  \"domain_Lx\": " << std::scientific << std::setprecision(8) << domainLx << ",\n";
                jsonFile << "  \"elements_x\": " << elementsX << ",\n";
                if (!modeK.empty()) {
                    jsonFile << "  \"k_modes\": [";
                    for (size_t mi = 0; mi < modeK.size(); ++mi) {
                        jsonFile << std::scientific << std::setprecision(8) << modeK[mi] << (mi + 1 < modeK.size() ? ", " : "");
                    }
                    jsonFile << "],\n";
                }
            }
            jsonFile << "  \"probes\": [\n";
            for (size_t i = 0; i < m_probes.size(); ++i) {
                const auto& pr = m_probes[i];
                jsonFile << "    {\n"
                         << "      \"id\": " << pr.id << ",\n"
                         << "      \"nodeIdx\": " << pr.nodeIdx << ",\n"
                         << "      \"target\": [" << pr.targetX << ", " << pr.targetY << ", " << pr.targetZ << "],\n"
                         << "      \"actual\": [" << pr.actualX << ", " << pr.actualY << ", " << pr.actualZ << "],\n"
                         << "      \"offset\": " << pr.dist << "\n"
                         << "    }" << (i + 1 < m_probes.size() ? "," : "") << "\n";
            }
            jsonFile << "  ]\n}\n";
            jsonFile.close();
        }
    }

    m_initialized = true;

    // Print probe locations summary
    std::cout << "\nObservation Probes Configured (" << m_probes.size() << " locations):" << std::endl;
    for (const auto& pr : m_probes) {
        std::cout << "  Probe " << pr.id 
                  << ": Exact Position (" << pr.actualX << ", " << pr.actualY << ", " << pr.actualZ << ")"
                  << " [Spectral Element GLL Interpolated]" << std::endl;
    }
    if (bSaveToFile && !m_probes.empty()) {
        std::cout << "Streaming probe history to: " << (m_outputDir + "/probe_history.csv") << std::endl;
    }
}

// @@ recording field values at all observation probes via spectral interpolation
void ProbeManager::record(int step, double time, const std::vector<double>& state, int npts) {
    if (!m_combinedFile.is_open()) return;

    m_combinedFile << step << "," << time;

    for (const auto& pr : m_probes) {
        double Ex = 0.0, Ey = 0.0, Ez = 0.0, Hx = 0.0, Hy = 0.0, Hz = 0.0;
        int offset = pr.elemOffset;
        int nWeights = static_cast<int>(pr.interpWeights.size());
        for (int p = 0; p < nWeights; ++p) {
            double w = pr.interpWeights[p];
            int idx = offset + p;
            Ex += w * state[0 * npts + idx];
            Ey += w * state[1 * npts + idx];
            Ez += w * state[2 * npts + idx];
            Hx += w * state[3 * npts + idx];
            Hy += w * state[4 * npts + idx];
            Hz += w * state[5 * npts + idx];
        }

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

