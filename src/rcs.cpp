#include "rcs.hpp"
#include "mesh.hpp"
#include "boundary_conditions.hpp"
#include "comm.hpp"

#ifdef NEKWAVE_ENABLE_MPI
#include <mpi.h>
#endif

#include <cmath>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <sys/stat.h>
#include <sys/types.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

RcsMonitor::~RcsMonitor() {
    if (m_initialized && !m_finalized) {
        finalize();
    }
}

void RcsMonitor::init(const Mesh& mesh, const std::string& outputDir) {
    m_outputDir = outputDir;
    m_surfPoints.clear();
    m_initialized = false;
    m_finalized = false;

    if (!m_config.enabled || m_config.wavenumbers.empty()) {
        return;
    }

    bool bSaveToFile = (!m_outputDir.empty() && m_outputDir != "none" &&
                        m_outputDir != "None" && m_outputDir != "NONE");
    if (bSaveToFile && Comm::isRoot()) {
        mkdir(m_outputDir.c_str(), 0755);
    }

    const auto& faces = mesh.getFaces();
    const auto& faceData = mesh.getFaceData();
    const auto& coordX = mesh.getCoordX();
    const auto& coordY = mesh.getCoordY();
    const auto& coordZ = mesh.getCoordZ();

    double localArea = 0.0;
    double localWeightedRadius = 0.0;
    double localMaxRadius = 0.0;

    for (size_t fIdx = 0; fIdx < faceData.size(); ++fIdx) {
        const auto& fd = faceData[fIdx];
        int neighborId = (fIdx < faces.size()) ? faces[fIdx].neighborElementId : -1;
        BcType bc = BoundaryConditions::parseBcTag(fd.bcType, neighborId);

        if (bc != BcType::PEC) {
            continue;
        }

        for (const auto& pt : fd.points) {
            RcsSurfacePoint sp;
            sp.volIdx = pt.volIdxMinus;
            sp.x = coordX[sp.volIdx];
            sp.y = coordY[sp.volIdx];
            sp.z = coordZ[sp.volIdx];

            // pt.nx, pt.ny, pt.nz is the outward normal of the vacuum hex element,
            // which points INTO the hollow PEC scatterer. The scatterer outward unit
            // normal n_s (pointing from the scatterer INTO the vacuum exterior) is -n_elem.
            double nsx = -pt.nx;
            double nsy = -pt.ny;
            double nsz = -pt.nz;

            // Verify outward radial orientation relative to origin (0, 0, 0)
            double rDotN = sp.x * nsx + sp.y * nsy + sp.z * nsz;
            if (rDotN < 0.0) {
                nsx = -nsx;
                nsy = -nsy;
                nsz = -nsz;
            }

            sp.nsx = nsx;
            sp.nsy = nsy;
            sp.nsz = nsz;
            sp.dA = pt.dA;

            double r = std::sqrt(sp.x * sp.x + sp.y * sp.y + sp.z * sp.z);
            localArea += sp.dA;
            localWeightedRadius += r * sp.dA;
            if (r > localMaxRadius) {
                localMaxRadius = r;
            }

            m_surfPoints.push_back(sp);
        }
    }

    int localPts = static_cast<int>(m_surfPoints.size());
    m_totalPointsGlobal = static_cast<int>(Comm::allreduceSum(static_cast<double>(localPts)));
    m_totalAreaGlobal = Comm::allreduceSum(localArea);
    double globalWeightedRadius = Comm::allreduceSum(localWeightedRadius);
    double globalMaxRadius = Comm::allreduceMax(localMaxRadius);

    m_effectiveRadius = (m_totalAreaGlobal > 1e-14) ? (globalWeightedRadius / m_totalAreaGlobal) : globalMaxRadius;
    if (m_config.sphereRadius <= 0.0) {
        m_config.sphereRadius = globalMaxRadius;
    }

    const size_t numFreqs = m_config.wavenumbers.size();
    m_JsRe.assign(numFreqs, std::vector<double>(3 * localPts, 0.0));
    m_JsIm.assign(numFreqs, std::vector<double>(3 * localPts, 0.0));
    m_MsRe.assign(numFreqs, std::vector<double>(3 * localPts, 0.0));
    m_MsIm.assign(numFreqs, std::vector<double>(3 * localPts, 0.0));

    m_EincRe.assign(numFreqs, 0.0);
    m_EincIm.assign(numFreqs, 0.0);

    if (bSaveToFile && Comm::isRoot()) {
        std::string histPath = m_outputDir + "/rcs_surface_history.csv";
        m_historyFile.open(histPath);
        if (m_historyFile.is_open()) {
            m_historyFile << "step,time,Einc_origin,rms_Js,rms_Ms\n";
            m_historyFile << std::scientific << std::setprecision(8);
        }
    }

    if (Comm::isRoot()) {
        double k0 = m_config.wavenumbers[0];
        double a = m_config.sphereRadius;
        std::cout << "[RCS/NTFF] Initialized PEC Surface Equivalence Monitor:" << std::endl;
        std::cout << "  Surface GLL Points: " << m_totalPointsGlobal << " across " << Comm::size() << " MPI rank(s)" << std::endl;
        std::cout << "  Surface Area:       " << std::fixed << std::setprecision(6) << m_totalAreaGlobal
                  << " m^2 (4*pi*a^2 = " << (4.0 * M_PI * a * a) << " m^2, a_eff = " << m_effectiveRadius << " m)" << std::endl;
        std::cout << "  Primary Carrier:    k0 = " << k0 << " rad/m (k0*a = " << (k0 * a)
                  << ", frequency channels = " << numFreqs << ")" << std::endl;
    }

    m_initialized = true;
}

void RcsMonitor::record(int step, double time, double dt, const std::vector<double>& state, int npts) {
    if (!m_initialized || !m_config.enabled || dt <= 0.0) {
        return;
    }

    const size_t numFreqs = m_config.wavenumbers.size();
    const size_t numLocalPts = m_surfPoints.size();

    // 1. Evaluate incident wave E_inc(0, t) at scatterer center r = (0, 0, 0)
    double eInc = 0.0;
    if (m_config.incidentWaveFn) {
        eInc = m_config.incidentWaveFn(time);
    }

    // Precompute trigonometric factors cos(k*t)*dt and -sin(k*t)*dt for each frequency channel
    std::vector<double> cosWtDt(numFreqs);
    std::vector<double> negSinWtDt(numFreqs);
    for (size_t m = 0; m < numFreqs; ++m) {
        double phase = m_config.wavenumbers[m] * time; // omega * t = k * c * t with c = 1
        cosWtDt[m]    =  std::cos(phase) * dt;
        negSinWtDt[m] = -std::sin(phase) * dt;

        m_EincRe[m] += eInc * cosWtDt[m];
        m_EincIm[m] += eInc * negSinWtDt[m];
    }

    const double* Ex = &state[0 * npts];
    const double* Ey = &state[1 * npts];
    const double* Ez = &state[2 * npts];
    const double* Hx = &state[3 * npts];
    const double* Hy = &state[4 * npts];
    const double* Hz = &state[5 * npts];

    double localIntJs2 = 0.0;
    double localIntMs2 = 0.0;

    // 2. Accumulate surface current phasors at all local PEC GLL nodes
    for (size_t q = 0; q < numLocalPts; ++q) {
        const auto& sp = m_surfPoints[q];
        int v = sp.volIdx;

        double ex = Ex[v], ey = Ey[v], ez = Ez[v];
        double hx = Hx[v], hy = Hy[v], hz = Hz[v];

        // In Scattered-Field mode, reconstruct the total boundary trace:
        //   E_tot^- = E_scat^- + E_inc(r', t)
        //   H_tot^- = H_scat^- + H_inc(r', t)
        if (m_config.incidentPlaneWave.enabled) {
            double ex_inc = 0.0, ey_inc = 0.0, ez_inc = 0.0;
            double hx_inc = 0.0, hy_inc = 0.0, hz_inc = 0.0;
            m_config.incidentPlaneWave.evaluate(sp.x, sp.y, sp.z, time,
                                                ex_inc, ey_inc, ez_inc,
                                                hx_inc, hy_inc, hz_inc);
            ex += ex_inc;
            ey += ey_inc;
            ez += ez_inc;
            hx += hx_inc;
            hy += hy_inc;
            hz += hz_inc;
        }

        // Electric surface current: J_s = n_s x H^-
        double Jx = sp.nsy * hz - sp.nsz * hy;
        double Jy = sp.nsz * hx - sp.nsx * hz;
        double Jz = sp.nsx * hy - sp.nsy * hx;

        // Magnetic surface current (weak DG trace): M_s = -n_s x E^-
        double Mx = -(sp.nsy * ez - sp.nsz * ey);
        double My = -(sp.nsz * ex - sp.nsx * ez);
        double Mz = -(sp.nsx * ey - sp.nsy * ex);

        for (size_t m = 0; m < numFreqs; ++m) {
            double cDt = cosWtDt[m];
            double sDt = negSinWtDt[m];

            m_JsRe[m][3 * q + 0] += Jx * cDt;
            m_JsRe[m][3 * q + 1] += Jy * cDt;
            m_JsRe[m][3 * q + 2] += Jz * cDt;

            m_JsIm[m][3 * q + 0] += Jx * sDt;
            m_JsIm[m][3 * q + 1] += Jy * sDt;
            m_JsIm[m][3 * q + 2] += Jz * sDt;

            m_MsRe[m][3 * q + 0] += Mx * cDt;
            m_MsRe[m][3 * q + 1] += My * cDt;
            m_MsRe[m][3 * q + 2] += Mz * cDt;

            m_MsIm[m][3 * q + 0] += Mx * sDt;
            m_MsIm[m][3 * q + 1] += My * sDt;
            m_MsIm[m][3 * q + 2] += Mz * sDt;
        }

        if (step % 10 == 0) {
            localIntJs2 += (Jx * Jx + Jy * Jy + Jz * Jz) * sp.dA;
            localIntMs2 += (Mx * Mx + My * My + Mz * Mz) * sp.dA;
        }
    }

    if (step % 10 == 0 && m_totalAreaGlobal > 1e-14) {
        double globalIntJs2 = Comm::allreduceSum(localIntJs2);
        double globalIntMs2 = Comm::allreduceSum(localIntMs2);
        if (Comm::isRoot() && m_historyFile.is_open()) {
            double rmsJs = std::sqrt(globalIntJs2 / m_totalAreaGlobal);
            double rmsMs = std::sqrt(globalIntMs2 / m_totalAreaGlobal);
            m_historyFile << step << "," << time << "," << eInc << "," << rmsJs << "," << rmsMs << "\n";
            m_historyFile.flush();
        }
    }
}

void RcsMonitor::writeSpectra(bool verbose) {
    if (!m_initialized || !m_config.enabled) {
        return;
    }

    bool bSaveToFile = (!m_outputDir.empty() && m_outputDir != "none" &&
                        m_outputDir != "None" && m_outputDir != "NONE");
    if (!bSaveToFile) {
        return;
    }

    // Construct incident-wave-aligned right-handed orthonormal triad (e1, e2, e3):
    //   e3 = kHatInc (forward propagation direction, theta_s = 0)
    //   e1 = eHatInc (electric polarization direction, phi_s = 0 E-plane)
    //   e2 = e3 x e1 (magnetic polarization direction, phi_s = 90 deg H-plane)
    const double e3x = m_config.kHatInc[0], e3y = m_config.kHatInc[1], e3z = m_config.kHatInc[2];
    const double e1x = m_config.eHatInc[0], e1y = m_config.eHatInc[1], e1z = m_config.eHatInc[2];
    const double e2x = e3y * e1z - e3z * e1y;
    const double e2y = e3z * e1x - e3x * e1z;
    const double e2z = e3x * e1y - e3y * e1x;

    const double a = m_config.sphereRadius;
    const double geomCrossSection = M_PI * a * a;
    const double c0 = m_config.c0;

    // Helper lambda to evaluate local radiation vectors (N, L, N_upwind) for a given wavenumber index m and direction r_hat
    auto integrateDirectionLocal = [&](size_t m, double rx, double ry, double rz,
                                       double N_re[3], double N_im[3],
                                       double L_re[3], double L_im[3],
                                       double Nup_re[3], double Nup_im[3]) {
        for (int c = 0; c < 3; ++c) {
            N_re[c] = 0.0; N_im[c] = 0.0;
            L_re[c] = 0.0; L_im[c] = 0.0;
            Nup_re[c] = 0.0; Nup_im[c] = 0.0;
        }

        const double k = m_config.wavenumbers[m];
        const auto& JsRe = m_JsRe[m];
        const auto& JsIm = m_JsIm[m];
        const auto& MsRe = m_MsRe[m];
        const auto& MsIm = m_MsIm[m];

        for (size_t q = 0; q < m_surfPoints.size(); ++q) {
            const auto& sp = m_surfPoints[q];
            // Spatial phase factor exp(+i * k * (r_hat . r'))
            double kr = k * (rx * sp.x + ry * sp.y + rz * sp.z);
            double cosKr = std::cos(kr) * sp.dA;
            double sinKr = std::sin(kr) * sp.dA;

            double jx_r = JsRe[3 * q + 0], jx_i = JsIm[3 * q + 0];
            double jy_r = JsRe[3 * q + 1], jy_i = JsIm[3 * q + 1];
            double jz_r = JsRe[3 * q + 2], jz_i = JsIm[3 * q + 2];

            double mx_r = MsRe[3 * q + 0], mx_i = MsIm[3 * q + 0];
            double my_r = MsRe[3 * q + 1], my_i = MsIm[3 * q + 1];
            double mz_r = MsRe[3 * q + 2], mz_i = MsIm[3 * q + 2];

            // Complex multiplication: (J_r + i J_i) * (cosKr + i sinKr)
            N_re[0] += jx_r * cosKr - jx_i * sinKr;
            N_im[0] += jx_r * sinKr + jx_i * cosKr;
            N_re[1] += jy_r * cosKr - jy_i * sinKr;
            N_im[1] += jy_r * sinKr + jy_i * cosKr;
            N_re[2] += jz_r * cosKr - jz_i * sinKr;
            N_im[2] += jz_r * sinKr + jz_i * cosKr;

            L_re[0] += mx_r * cosKr - mx_i * sinKr;
            L_im[0] += mx_r * sinKr + mx_i * cosKr;
            L_re[1] += my_r * cosKr - my_i * sinKr;
            L_im[1] += my_r * sinKr + my_i * cosKr;
            L_re[2] += mz_r * cosKr - mz_i * sinKr;
            L_im[2] += mz_r * sinKr + mz_i * cosKr;

            // Upwind Riemann surface current on PEC: J_s^* = J_s + c0 * (n_s x M_s)
            double nxm_x_r = sp.nsy * mz_r - sp.nsz * my_r;
            double nxm_y_r = sp.nsz * mx_r - sp.nsx * mz_r;
            double nxm_z_r = sp.nsx * my_r - sp.nsy * mx_r;

            double nxm_x_i = sp.nsy * mz_i - sp.nsz * my_i;
            double nxm_y_i = sp.nsz * mx_i - sp.nsx * mz_i;
            double nxm_z_i = sp.nsx * my_i - sp.nsy * mx_i;

            double jux_r = jx_r + c0 * nxm_x_r, jux_i = jx_i + c0 * nxm_x_i;
            double juy_r = jy_r + c0 * nxm_y_r, juy_i = jy_i + c0 * nxm_y_i;
            double juz_r = jz_r + c0 * nxm_z_r, juz_i = jz_i + c0 * nxm_z_i;

            Nup_re[0] += jux_r * cosKr - jux_i * sinKr;
            Nup_im[0] += jux_r * sinKr + jux_i * cosKr;
            Nup_re[1] += juy_r * cosKr - juy_i * sinKr;
            Nup_im[1] += juy_r * sinKr + juy_i * cosKr;
            Nup_re[2] += juz_r * cosKr - juz_i * sinKr;
            Nup_im[2] += juz_r * sinKr + juz_i * cosKr;
        }
    };

    // --------------------------------------------------------------------------
    // 1. Bistatic Angular Sweep (E-Plane phi = 0 and H-Plane phi = 90 deg) at k0
    // --------------------------------------------------------------------------
    const int nTheta = std::max(2, m_config.numTheta);
    // 18 doubles per (theta, plane): N_re[3], N_im[3], L_re[3], L_im[3], Nup_re[3], Nup_im[3]
    // 2 planes (0 = E-plane, 1 = H-plane) => 36 doubles per theta
    std::vector<double> localBuf(nTheta * 36, 0.0);
    std::vector<double> globalBuf(nTheta * 36, 0.0);

    for (int i = 0; i < nTheta; ++i) {
        double theta = M_PI * static_cast<double>(i) / static_cast<double>(nTheta - 1);
        double cosT = std::cos(theta);
        double sinT = std::sin(theta);

        for (int plane = 0; plane < 2; ++plane) {
            double cosP = (plane == 0) ? 1.0 : 0.0;
            double sinP = (plane == 0) ? 0.0 : 1.0;

            double rx = cosT * e3x + sinT * (cosP * e1x + sinP * e2x);
            double ry = cosT * e3y + sinT * (cosP * e1y + sinP * e2y);
            double rz = cosT * e3z + sinT * (cosP * e1z + sinP * e2z);

            double N_re[3], N_im[3], L_re[3], L_im[3], Nup_re[3], Nup_im[3];
            integrateDirectionLocal(0, rx, ry, rz, N_re, N_im, L_re, L_im, Nup_re, Nup_im);

            double* dst = &localBuf[(i * 2 + plane) * 18];
            for (int c = 0; c < 3; ++c) {
                dst[0 + c]  = N_re[c];
                dst[3 + c]  = N_im[c];
                dst[6 + c]  = L_re[c];
                dst[9 + c]  = L_im[c];
                dst[12 + c] = Nup_re[c];
                dst[15 + c] = Nup_im[c];
            }
        }
    }

#ifdef NEKWAVE_ENABLE_MPI
    MPI_Allreduce(localBuf.data(), globalBuf.data(), static_cast<int>(localBuf.size()),
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
    globalBuf = localBuf;
#endif

    const double k0 = m_config.wavenumbers[0];
    const double eIncMagSq0 = m_EincRe[0] * m_EincRe[0] + m_EincIm[0] * m_EincIm[0];
    const double prefactor0 = (eIncMagSq0 > 1e-30) ? ((k0 * k0) / (4.0 * M_PI * eIncMagSq0)) : 0.0;

    if (Comm::isRoot()) {
        std::string bistaticPath = m_outputDir + "/rcs_bistatic.csv";
        std::string tmpBistaticPath = bistaticPath + ".tmp";
        std::ofstream out(tmpBistaticPath);
        if (out.is_open()) {
            out << "theta_deg,theta_rad,"
                << "rcs_e_plane_m2,rcs_h_plane_m2,"
                << "rcs_e_plane_norm,rcs_h_plane_norm,"
                << "rcs_e_plane_dbsm,rcs_h_plane_dbsm,"
                << "rcs_e_upwind_norm,rcs_h_upwind_norm,"
                << "rcs_e_jsonly_norm,rcs_h_jsonly_norm\n";
            out << std::scientific << std::setprecision(8);

            double backRcsNormE = 0.0, fwdRcsNormE = 0.0;

            for (int i = 0; i < nTheta; ++i) {
                double thetaDeg = 180.0 * static_cast<double>(i) / static_cast<double>(nTheta - 1);
                double theta = M_PI * static_cast<double>(i) / static_cast<double>(nTheta - 1);
                double cosT = std::cos(theta);
                double sinT = std::sin(theta);

                double sigmaStratton[2] = {0.0, 0.0};
                double sigmaUpwind[2]   = {0.0, 0.0};
                double sigmaJsOnly[2]   = {0.0, 0.0};

                for (int plane = 0; plane < 2; ++plane) {
                    double cosP = (plane == 0) ? 1.0 : 0.0;
                    double sinP = (plane == 0) ? 0.0 : 1.0;

                    // Spherical basis vectors theta_hat and phi_hat
                    double thx = -sinT * e3x + cosT * (cosP * e1x + sinP * e2x);
                    double thy = -sinT * e3y + cosT * (cosP * e1y + sinP * e2y);
                    double thz = -sinT * e3z + cosT * (cosP * e1z + sinP * e2z);

                    double phx = -sinP * e1x + cosP * e2x;
                    double phy = -sinP * e1y + cosP * e2y;
                    double phz = -sinP * e1z + cosP * e2z;

                    const double* src = &globalBuf[(i * 2 + plane) * 18];
                    const double* N_re   = &src[0];
                    const double* N_im   = &src[3];
                    const double* L_re   = &src[6];
                    const double* L_im   = &src[9];
                    const double* Nup_re = &src[12];
                    const double* Nup_im = &src[15];

                    double Nth_re = N_re[0] * thx + N_re[1] * thy + N_re[2] * thz;
                    double Nth_im = N_im[0] * thx + N_im[1] * thy + N_im[2] * thz;
                    double Nph_re = N_re[0] * phx + N_re[1] * phy + N_re[2] * phz;
                    double Nph_im = N_im[0] * phx + N_im[1] * phy + N_im[2] * phz;

                    double Lth_re = L_re[0] * thx + L_re[1] * thy + L_re[2] * thz;
                    double Lth_im = L_im[0] * thx + L_im[1] * thy + L_im[2] * thz;
                    double Lph_re = L_re[0] * phx + L_re[1] * phy + L_re[2] * phz;
                    double Lph_im = L_im[0] * phx + L_im[1] * phy + L_im[2] * phz;

                    double Nup_th_re = Nup_re[0] * thx + Nup_re[1] * thy + Nup_re[2] * thz;
                    double Nup_th_im = Nup_im[0] * thx + Nup_im[1] * thy + Nup_im[2] * thz;
                    double Nup_ph_re = Nup_re[0] * phx + Nup_re[1] * phy + Nup_re[2] * phz;
                    double Nup_ph_im = Nup_im[0] * phx + Nup_im[1] * phy + Nup_im[2] * phz;

                    // 1) Full Stratton-Chu (both J_s and M_s from DG interior trace):
                    //    E_theta ~ (N_theta + L_phi), E_phi ~ (L_theta - N_phi)
                    double Eth_re = Nth_re + Lph_re;
                    double Eth_im = Nth_im + Lph_im;
                    double Eph_re = Lth_re - Nph_re;
                    double Eph_im = Lth_im - Nph_im;
                    sigmaStratton[plane] = prefactor0 * (Eth_re * Eth_re + Eth_im * Eth_im +
                                                         Eph_re * Eph_re + Eph_im * Eph_im);

                    // 2) Upwind Riemann boundary state (J_s^* = J_s + c0*(n_s x M_s), M_s^* = 0):
                    sigmaUpwind[plane] = prefactor0 * (Nup_th_re * Nup_th_re + Nup_th_im * Nup_th_im +
                                                       Nup_ph_re * Nup_ph_re + Nup_ph_im * Nup_ph_im);

                    // 3) Electric surface current only (J_s only, L = 0):
                    sigmaJsOnly[plane] = prefactor0 * (Nth_re * Nth_re + Nth_im * Nth_im +
                                                       Nph_re * Nph_re + Nph_im * Nph_im);
                }

                double normE = sigmaStratton[0] / geomCrossSection;
                double normH = sigmaStratton[1] / geomCrossSection;
                double dbsmE = 10.0 * std::log10(std::max(sigmaStratton[0], 1e-30));
                double dbsmH = 10.0 * std::log10(std::max(sigmaStratton[1], 1e-30));

                if (i == 0) fwdRcsNormE = normE;
                if (i == nTheta - 1) backRcsNormE = normE;

                out << thetaDeg << "," << theta << ","
                    << sigmaStratton[0] << "," << sigmaStratton[1] << ","
                    << normE << "," << normH << ","
                    << dbsmE << "," << dbsmH << ","
                    << (sigmaUpwind[0] / geomCrossSection) << "," << (sigmaUpwind[1] / geomCrossSection) << ","
                    << (sigmaJsOnly[0] / geomCrossSection) << "," << (sigmaJsOnly[1] / geomCrossSection) << "\n";
            }
            out.flush();
            out.close();
            std::rename(tmpBistaticPath.c_str(), bistaticPath.c_str());
            if (verbose) {
                std::cout << "[POSTPROCESS] Exported bistatic RCS pattern (" << nTheta
                          << " angles) to " << bistaticPath << std::endl;
                std::cout << "  -> Forward Scatter RCS (theta =   0 deg): sigma / (pi*a^2) = "
                          << std::fixed << std::setprecision(4) << fwdRcsNormE << std::endl;
                std::cout << "  -> Monostatic Backscatter (theta = 180 deg): sigma / (pi*a^2) = "
                          << std::fixed << std::setprecision(4) << backRcsNormE << std::endl;
            }
        }
    }

    // --------------------------------------------------------------------------
    // 2. Broadband Frequency Sweep (Backscatter & Forward Scatter vs. k*a)
    // --------------------------------------------------------------------------
    const size_t numFreqs = m_config.wavenumbers.size();
    if (numFreqs > 1) {
        std::vector<double> localFreqBuf(numFreqs * 36, 0.0);
        std::vector<double> globalFreqBuf(numFreqs * 36, 0.0);

        for (size_t m = 0; m < numFreqs; ++m) {
            // Direction 0: Forward scatter (+e3, theta = 0)
            // Direction 1: Monostatic backscatter (-e3, theta = pi)
            for (int dir = 0; dir < 2; ++dir) {
                double sgn = (dir == 0) ? 1.0 : -1.0;
                double rx = sgn * e3x, ry = sgn * e3y, rz = sgn * e3z;
                double N_re[3], N_im[3], L_re[3], L_im[3], Nup_re[3], Nup_im[3];
                integrateDirectionLocal(m, rx, ry, rz, N_re, N_im, L_re, L_im, Nup_re, Nup_im);

                double* dst = &localFreqBuf[(m * 2 + dir) * 18];
                for (int c = 0; c < 3; ++c) {
                    dst[0 + c]  = N_re[c];
                    dst[3 + c]  = N_im[c];
                    dst[6 + c]  = L_re[c];
                    dst[9 + c]  = L_im[c];
                    dst[12 + c] = Nup_re[c];
                    dst[15 + c] = Nup_im[c];
                }
            }
        }

#ifdef NEKWAVE_ENABLE_MPI
        MPI_Allreduce(localFreqBuf.data(), globalFreqBuf.data(), static_cast<int>(localFreqBuf.size()),
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
        globalFreqBuf = localFreqBuf;
#endif

        if (Comm::isRoot()) {
            std::string freqPath = m_outputDir + "/rcs_frequency_sweep.csv";
            std::string tmpFreqPath = freqPath + ".tmp";
            std::ofstream out(tmpFreqPath);
            if (out.is_open()) {
                out << "k,ka,wavelength,Einc_mag,backscatter_rcs_norm,forward_rcs_norm,backscatter_upwind_norm\n";
                out << std::scientific << std::setprecision(8);

                for (size_t m = 0; m < numFreqs; ++m) {
                    double k = m_config.wavenumbers[m];
                    double ka = k * a;
                    double wavelength = (k > 1e-14) ? (2.0 * M_PI / k) : 0.0;
                    double eIncMagSq = m_EincRe[m] * m_EincRe[m] + m_EincIm[m] * m_EincIm[m];
                    double prefactor = (eIncMagSq > 1e-30) ? ((k * k) / (4.0 * M_PI * eIncMagSq)) : 0.0;

                    double rcsNorm[2] = {0.0, 0.0};
                    double rcsUpwindNorm[2] = {0.0, 0.0};

                    for (int dir = 0; dir < 2; ++dir) {
                        double cosT = (dir == 0) ? 1.0 : -1.0;
                        double thx = cosT * e1x, thy = cosT * e1y, thz = cosT * e1z;
                        double phx = e2x, phy = e2y, phz = e2z;

                        const double* src = &globalFreqBuf[(m * 2 + dir) * 18];
                        const double* N_re   = &src[0];
                        const double* N_im   = &src[3];
                        const double* L_re   = &src[6];
                        const double* L_im   = &src[9];
                        const double* Nup_re = &src[12];
                        const double* Nup_im = &src[15];

                        double Nth_re = N_re[0] * thx + N_re[1] * thy + N_re[2] * thz;
                        double Nth_im = N_im[0] * thx + N_im[1] * thy + N_im[2] * thz;
                        double Nph_re = N_re[0] * phx + N_re[1] * phy + N_re[2] * phz;
                        double Nph_im = N_im[0] * phx + N_im[1] * phy + N_im[2] * phz;

                        double Lth_re = L_re[0] * thx + L_re[1] * thy + L_re[2] * thz;
                        double Lth_im = L_im[0] * thx + L_im[1] * thy + L_im[2] * thz;
                        double Lph_re = L_re[0] * phx + L_re[1] * phy + L_re[2] * phz;
                        double Lph_im = L_im[0] * phx + L_im[1] * phy + L_im[2] * phz;

                        double Nup_th_re = Nup_re[0] * thx + Nup_re[1] * thy + Nup_re[2] * thz;
                        double Nup_th_im = Nup_im[0] * thx + Nup_im[1] * thy + Nup_im[2] * thz;
                        double Nup_ph_re = Nup_re[0] * phx + Nup_re[1] * phy + Nup_re[2] * phz;
                        double Nup_ph_im = Nup_im[0] * phx + Nup_im[1] * phy + Nup_im[2] * phz;

                        double Eth_re = Nth_re + Lph_re, Eth_im = Nth_im + Lph_im;
                        double Eph_re = Lth_re - Nph_re, Eph_im = Lth_im - Nph_im;

                        double sigma = prefactor * (Eth_re * Eth_re + Eth_im * Eth_im +
                                                    Eph_re * Eph_re + Eph_im * Eph_im);
                        double sigmaUp = prefactor * (Nup_th_re * Nup_th_re + Nup_th_im * Nup_th_im +
                                                      Nup_ph_re * Nup_ph_re + Nup_ph_im * Nup_ph_im);
                        rcsNorm[dir] = sigma / geomCrossSection;
                        rcsUpwindNorm[dir] = sigmaUp / geomCrossSection;
                    }

                    out << k << "," << ka << "," << wavelength << ","
                        << std::sqrt(eIncMagSq) << ","
                        << rcsNorm[1] << "," << rcsNorm[0] << "," << rcsUpwindNorm[1] << "\n";
                }
                out.flush();
                out.close();
                std::rename(tmpFreqPath.c_str(), freqPath.c_str());
                if (verbose) {
                    std::cout << "[POSTPROCESS] Exported broadband RCS frequency sweep (" << numFreqs
                              << " wavenumbers) to " << freqPath << std::endl;
                }
            }
        }
    }
}

void RcsMonitor::finalize() {
    if (!m_initialized || m_finalized || !m_config.enabled) {
        return;
    }
    m_finalized = true;

    if (m_historyFile.is_open()) {
        m_historyFile.flush();
        m_historyFile.close();
    }

    writeSpectra(true);
}
