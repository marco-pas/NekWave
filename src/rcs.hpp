#ifndef NW_SRC_RCS_HPP
#define NW_SRC_RCS_HPP

#include <array>
#include <functional>
#include <string>
#include <vector>
#include <fstream>
#include "boundary_conditions.hpp"

class Mesh;

/**
 * @brief Surface quadrature point on the scatterer boundary (PEC) for Near-to-Far-Field (NTFF) integration.
 */
struct RcsSurfacePoint {
    int volIdx = -1;       ///< Local volume collocation node index (minus side)
    double x = 0.0;        ///< Physical x-coordinate on scatterer surface
    double y = 0.0;        ///< Physical y-coordinate on scatterer surface
    double z = 0.0;        ///< Physical z-coordinate on scatterer surface
    double nsx = 0.0;      ///< Outward unit normal x-component (pointing from scatterer into vacuum)
    double nsy = 0.0;      ///< Outward unit normal y-component (pointing from scatterer into vacuum)
    double nsz = 0.0;      ///< Outward unit normal z-component (pointing from scatterer into vacuum)
    double dA = 0.0;       ///< 2D GLL surface quadrature area weight (J_face * w_p * w_q)
};

/**
 * @brief Configuration for Near-to-Far-Field (NTFF) Radar Cross Section (RCS) extraction.
 */
struct RcsConfig {
    bool enabled = false;

    /// Target wavenumbers k = omega / c = 2 * pi / lambda (rad/m) for running DFT accumulation.
    /// The first entry wavenumbers[0] is the primary carrier wavenumber k0 used for the angular pattern.
    std::vector<double> wavenumbers;

    /// Reference scatterer sphere radius a (m) for normalizing RCS by (pi * a^2).
    /// If <= 0.0, automatically determined from the PEC surface corner radius.
    double sphereRadius = 0.30;

    /// Number of polar angles theta in [0, 180] degrees for bistatic RCS pattern.
    int numTheta = 361;

    /// Upwind flux penalty parameter c0 (1.0 for upwind, 0.0 for central).
    double c0 = 1.0;

    /// Unit vector in the direction of incident wave propagation k_inc (defines theta = 0 forward scatter).
    std::array<double, 3> kHatInc = {{1.0, 0.0, 0.0}};

    /// Unit vector in the direction of incident electric field polarization E_inc (defines phi = 0 E-plane).
    std::array<double, 3> eHatInc = {{0.0, 0.0, 1.0}};

    /// Analytical incident plane wave configuration (active when scatteredFieldMode == true).
    IncidentPlaneWaveConfig incidentPlaneWave;

    /// Callback returning the incident electric field amplitude E_inc(0, t) at the origin r = (0,0,0) at time t.
    std::function<double(double t)> incidentWaveFn;
};

/**
 * @brief Near-to-Far-Field (NTFF) and Radar Cross Section (RCS) monitor.
 *
 * Implements Love's Surface Equivalence Principle (Stratton-Chu integral) on the
 * spectral element GLL boundary quadrature nodes of a PEC scatterer:
 *
 *   1. During preprocess():
 *      Extracts all local PEC boundary quadrature points, their physical coordinates r',
 *      scatterer outward unit normals n_s = -n_elem, and exact Nanson GLL area weights dA.
 *
 *   2. During simulate():
 *      At every time step t_n, evaluates the equivalent surface electric and magnetic currents
 *          J_s(r', t_n) =  n_s(r') x H^-(r', t_n)
 *          M_s(r', t_n) = -n_s(r') x E^-(r', t_n)
 *      and accumulates their running discrete Fourier transforms (DFT) at all target wavenumbers k:
 *          J_s_tilde(r', k) += J_s(r', t_n) * exp(-i * k * c * t_n) * dt
 *          M_s_tilde(r', k) += M_s(r', t_n) * exp(-i * k * c * t_n) * dt
 *      alongside the incident pulse spectrum E_inc_tilde(0, k) at the scatterer center.
 *
 *   3. During postprocess():
 *      Integrates the complex radiation vectors N(theta, phi) and L(theta, phi) over the
 *      closed scatterer surface using GLL quadrature and MPI reduction across all GPUs/ranks:
 *          N(r_hat, k) = \oint_S J_s_tilde(r', k) * exp(+i * k * r_hat . r') dA
 *          L(r_hat, k) = \oint_S M_s_tilde(r', k) * exp(+i * k * r_hat . r') dA
 *      and exports the bistatic E-plane and H-plane RCS patterns, frequency sweep, and
 *      surface phasor archive.
 */
class RcsMonitor {
public:
    RcsMonitor() = default;
    ~RcsMonitor();

    void setConfig(const RcsConfig& cfg) { m_config = cfg; }
    const RcsConfig& config() const { return m_config; }
    RcsConfig& config() { return m_config; }

    /**
     * @brief Extracts PEC scatterer surface GLL quadrature points and allocates DFT buffers.
     */
    void init(const Mesh& mesh, const std::string& outputDir);

    /**
     * @brief Accumulates running DFT of surface currents J_s, M_s and incident field E_inc(0, t).
     *
     * @param step  Current time step index (0 for initial state, >= 1 for time-advanced steps).
     * @param time  Current physical simulation time t.
     * @param dt    Quadrature time weight (e.g., 0.5 * dt at t = 0, dt for interior steps).
     * @param state Host state vector [Ex, Ey, Ez, Hx, Hy, Hz] of length 6 * npts.
     * @param npts  Number of local volume collocation points on this MPI rank.
     */
    void record(int step, double time, double dt, const std::vector<double>& state, int npts);

    /**
     * @brief Evaluates and writes current NTFF bistatic RCS and frequency sweep CSVs without closing streams.
     */
    void writeSpectra(bool verbose = false);

    /**
     * @brief Performs global MPI surface quadrature, evaluates NTFF bistatic RCS, and writes CSV outputs.
     */
    void finalize();

    bool isEnabled() const { return m_config.enabled && m_initialized; }

private:
    RcsConfig m_config;
    std::string m_outputDir;
    bool m_initialized = false;
    bool m_finalized = false;

    std::vector<RcsSurfacePoint> m_surfPoints;
    double m_totalAreaGlobal = 0.0;
    double m_effectiveRadius = 0.0;
    int m_totalPointsGlobal = 0;

    // Running DFT accumulators indexed by [freqIdx][3 * ptIdx + comp]
    std::vector<std::vector<double>> m_JsRe;
    std::vector<std::vector<double>> m_JsIm;
    std::vector<std::vector<double>> m_MsRe;
    std::vector<std::vector<double>> m_MsIm;

    // Running DFT accumulator for the incident electric field E_inc(0, t) at the scatterer center
    std::vector<double> m_EincRe;
    std::vector<double> m_EincIm;

    // Time-series log of RMS surface current on the scatterer
    std::ofstream m_historyFile;
};

#endif // NW_SRC_RCS_HPP
