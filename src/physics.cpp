#include "physics.hpp"

#ifdef NEKWAVE_ENABLE_CUDA
#include "cuda/mxm_cuda.hpp"
static bool s_useCudaMxm = true;
#else
static bool s_useCudaMxm = false;
#endif

// @@ the Physics constructor
Physics::Physics() : m_c0(1.0) {}

// @@ the Physics destructor
Physics::~Physics() = default;

void Physics::setUseCudaMxm(bool enable) {
#ifdef NEKWAVE_ENABLE_CUDA
    s_useCudaMxm = enable;
#else
    (void)enable;
    s_useCudaMxm = false;
#endif
}

bool Physics::getUseCudaMxm() {
    return s_useCudaMxm;
}

// --------------------- (!) ---------------------

// @@ matrix-matrix multiplication C = A * B in column-major layout
// Fast tensor contraction operator matching NekCEM's mxm
void Physics::mxm(const double* A, int n1, const double* B, int n2, double* C, int n3) {
#ifdef NEKWAVE_ENABLE_CUDA
    if (s_useCudaMxm) {
        nw_cuda_mxm(A, n1, B, n2, C, n3);
        return;
    }
#endif
    // @@ looping over the columns of C
    for (int j = 0; j < n3; ++j) {
        // @@ looping over the rows of C
        for (int i = 0; i < n1; ++i) {
            double sum = 0.0;
            // @@ contracting along the inner dimension n2
            for (int l = 0; l < n2; ++l) {
                sum += A[i + l * n1] * B[l + j * n2];
            }
            C[i + j * n1] = sum;
        }
    }
    // this is the algorithmic bottleneck.
}

// The issue here is that tensor cores need a specific data format, so padding is often needed.
// p = 5 --> 5x5 matrix --> tensor core needs 8x8 or 16x16
// the higher p, the more naturally we can approach this

// adding GLM probably can make sure that we increment the number of computations needed, given that we also add div

// --------------------- (!) ---------------------

// @@ local element differentiation using tensor product contractions
// Evaluates reference derivatives (ur, us, ut) using 1D derivative matrix D
//      Contraction 1: ur = (I (x) I (x) D) u
//      Contraction 2: us = (I (x) D (x) I) u
//      Contraction 3: ut = (D (x) I (x) I) u
void Physics::local_grad3(const double* u, int N, const std::vector<double>& D, const std::vector<double>& Dt,
                         double* ur, double* us, double* ut) {
    int m1 = N;
    int m2 = N * N;

    // Note: every mxm is a triple loop O(N^4), extacly 2N^4

    // @@ computing ur = (I (x) I (x) D) u using a single contraction
    // D is N x N, u is viewed as N x N^2
    mxm(D.data(), m1, u, m1, ur, m2); // (N x N) x (N x N^2)

    // @@ computing us = (I (x) D (x) I) u across each 2D k-slice
    // For each k in [0, N-1], u_slice is N x N, multiplied by D^T
    for (int k = 0; k < N; ++k) {
        mxm(u + k * m2, m1, Dt.data(), m1, us + k * m2, m1); // // N x ((N x N) x (N x N))
    }

    // @@ computing ut = (D (x) I (x) I) u using a single contraction
    // u is viewed as N^2 x N, multiplied by D^T
    mxm(u, m2, Dt.data(), m1, ut, m1); // (N^2 x N) x (N x N)
}

/* not used in NekCEM (used for edge elements tho)

// @@ covariant Piola transform for H(curl) fields
// 1-forms (tangential fields like E and H) transform via:
//      u_phys = J^{-T} * u_ref
// Line integrals are invariant: int_C u_phys . dx = int_{C_ref} u_ref . dxi

void Physics::covariantPiola(double J_rx, double J_sx, double J_tx,
                             double J_ry, double J_sy, double J_ty,
                             double J_rz, double J_sz, double J_tz,
                             double ur_ref, double us_ref, double ut_ref,
                             double& ux_phys, double& uy_phys, double& uz_phys) {
    ux_phys = J_rx * ur_ref + J_sx * us_ref + J_tx * ut_ref;
    uy_phys = J_ry * ur_ref + J_sy * us_ref + J_ty * ut_ref;
    uz_phys = J_rz * ur_ref + J_sz * us_ref + J_tz * ut_ref;
}

// @@ contravariant Piola transform for H(div) fields
// 2-forms (flux densities like B, D, and curls) transform via:
//      w_phys = (1 / det(J)) * J_aff * w_ref
// Surface fluxes are invariant: int_S w_phys . dA = int_{S_ref} w_ref . dA_ref

void Physics::contravariantPiola(double dx_dr, double dx_ds, double dx_dt,
                                 double dy_dr, double dy_ds, double dy_dt,
                                 double dz_dr, double dz_ds, double dz_dt,
                                 double detJ,
                                 double wr_ref, double ws_ref, double wt_ref,
                                 double& wx_phys, double& wy_phys, double& wz_phys) {
    double invDet = 1.0 / detJ;
    wx_phys = invDet * (dx_dr * wr_ref + dx_ds * ws_ref + dx_dt * wt_ref);
    wy_phys = invDet * (dy_dr * wr_ref + dy_ds * ws_ref + dy_dt * wt_ref);
    wz_phys = invDet * (dz_dr * wr_ref + dz_ds * ws_ref + dz_dt * wt_ref);
}

*/

// @@ weighted curl in physical space
// Volume weighted curl matching NekCEM's maxwell_wght_curl
// (nabla x u)_phys = (1 / J) * [ (grad_ref u) x (grad_ref x) ]
void Physics::compute_weighted_curl(const Mesh& mesh,
                                   const double* u1, const double* u2, const double* u3,
                                   double* w1, double* w2, double* w3) {
    int N = mesh.getN();
    int nxyz = mesh.getNumPointsPerElement();
    int nelt = mesh.getNumElements();

    const std::vector<double>& D = mesh.getD();
    const std::vector<double>& Dt = mesh.getDt();
    const std::vector<double>& w3mn = mesh.getW3();
    const std::vector<double>& jac = mesh.getJac();

    const std::vector<double>& rx = mesh.getRx();
    const std::vector<double>& sx = mesh.getSx();
    const std::vector<double>& tx = mesh.getTx();

    const std::vector<double>& ry = mesh.getRy();
    const std::vector<double>& sy = mesh.getSy();
    const std::vector<double>& ty = mesh.getTy();

    const std::vector<double>& rz = mesh.getRz();
    const std::vector<double>& sz = mesh.getSz();
    const std::vector<double>& tz = mesh.getTz();

    // @@ allocating temporary reference gradient components for the element
    std::vector<double> u1r(nxyz), u1s(nxyz), u1t(nxyz);
    std::vector<double> u2r(nxyz), u2s(nxyz), u2t(nxyz);
    std::vector<double> u3r(nxyz), u3s(nxyz), u3t(nxyz);

    // @@ looping over each element in the mesh
    for (int e = 0; e < nelt; ++e) {
        int offset = e * nxyz;

        // @@ differentiating each component of the vector field locally
        // Local gradient evaluation via 1D derivative matrices
        local_grad3(u1 + offset, N, D, Dt, u1r.data(), u1s.data(), u1t.data());
        local_grad3(u2 + offset, N, D, Dt, u2r.data(), u2s.data(), u2t.data());
        local_grad3(u3 + offset, N, D, Dt, u3r.data(), u3s.data(), u3t.data());

        // @@ looping over each point in the element to assemble the weighted physical curl
        for (int i = 0; i < nxyz; ++i) {
            int k = offset + i;
            double w = w3mn[i] * jac[k];

            // @@ applying quadrature weights to reference derivatives
            // Scaling with 3D GLL weights and Jacobian: J_k * W_ijk
            double u1rw = u1r[i] * w;
            double u1sw = u1s[i] * w;
            double u1tw = u1t[i] * w;

            double u2rw = u2r[i] * w;
            double u2sw = u2s[i] * w;
            double u2tw = u2t[i] * w;

            double u3rw = u3r[i] * w;
            double u3sw = u3s[i] * w;
            double u3tw = u3t[i] * w;

            // Metric factors from coordinate transformation (J^{-T})
            double rx_k = rx[k];
            double sx_k = sx[k];
            double tx_k = tx[k];

            double ry_k = ry[k];
            double sy_k = sy[k];
            double ty_k = ty[k];

            double rz_k = rz[k];
            double sz_k = sz[k];
            double tz_k = tz[k];

            // @@ computing physical weighted curl x-component: dw3/dy - dw2/dz
            //      (nabla x u)_x = (u3_r * ry + u3_s * sy + u3_t * ty) - (u2_r * rz + u2_s * sz + u2_t * tz)
            w1[k] = (u3rw * ry_k + u3sw * sy_k + u3tw * ty_k)
                  - (u2rw * rz_k + u2sw * sz_k + u2tw * tz_k);

            // @@ computing physical weighted curl y-component: dw1/dz - dw3/dx
            //      (nabla x u)_y = (u1_r * rz + u1_s * sz + u1_t * tz) - (u3_r * rx + u3_s * sx + u3_t * tx)
            w2[k] = (u1rw * rz_k + u1sw * sz_k + u1tw * tz_k)
                  - (u3rw * rx_k + u3sw * sx_k + u3tw * tx_k);

            // @@ computing physical weighted curl z-component: dw2/dx - dw1/dy
            //      (nabla x u)_z = (u2_r * rx + u2_s * sx + u2_t * tx) - (u1_r * ry + u1_s * sy + u1_t * ty)
            w3[k] = (u2rw * rx_k + u2sw * sx_k + u2tw * tx_k)
                  - (u1rw * ry_k + u1sw * sy_k + u1tw * ty_k);
        }
    }
}

// @@ evaluating discrete curl operator for 3D Maxwell fields
// Dispatches curl(H) and -curl(E) evaluations
//      dE/dt = (1/eps) * (nabla x H)
//      dH/dt = -(1/mu) * (nabla x E)
void Physics::computeCurl(const Mesh& mesh, const StateVector& state, StateVector& rhs) {
    int npts = mesh.getTotalPoints();
    if (state.size() < static_cast<size_t>(6 * npts) || rhs.size() < static_cast<size_t>(6 * npts)) {
        return;
    }

    // @@ decomposing state into field pointers: E = (u1, u2, u3), H = (u4, u5, u6)
    const double* Ex = &state[0 * npts];
    const double* Ey = &state[1 * npts];
    const double* Ez = &state[2 * npts];
    const double* Hx = &state[3 * npts];
    const double* Hy = &state[4 * npts];
    const double* Hz = &state[5 * npts];

    double* resEx = &rhs[0 * npts];
    double* resEy = &rhs[1 * npts];
    double* resEz = &rhs[2 * npts];
    double* resHx = &rhs[3 * npts];
    double* resHy = &rhs[4 * npts];
    double* resHz = &rhs[5 * npts];

    // @@ computing weighted curl of H for electric field residual: curl(H)
    compute_weighted_curl(mesh, Hx, Hy, Hz, resEx, resEy, resEz);

    // @@ computing weighted curl of E for magnetic field residual: curl(E)
    compute_weighted_curl(mesh, Ex, Ey, Ez, resHx, resHy, resHz);

    // @@ applying the negative sign for magnetic field update: -curl(E)
    for (int i = 0; i < npts; ++i) {
        resHx[i] = -resHx[i];
        resHy[i] = -resHy[i];
        resHz[i] = -resHz[i];
    }
}

// @@ the main Maxwell RHS evaluation based on NekCEM's cem_maxwell_op
// Full Spatial Operator Pipeline
// RHS = M^{-1} [ Volume_Curl + Surface_Flux ] (no sources for now)
void Physics::evaluateRHS(const Mesh& mesh, const StateVector& state, StateVector& rhs, double rkTime) {
    (void)rkTime; // Silence unused parameter warning in stub

    // @@ ensuring the RHS vector is the correct size and initialized to zero
    rhs.assign(state.size(), 0.0);
    StateVector flux(state.size(), 0.0);

    // @@ calling the routine to compute the curl of the E and H fields
    // Volume term evaluation via weighted curl
    computeCurl(mesh, state, rhs);

    // @@ extracting the trace of the fields on the element faces
    // Restrict interior field traces to element faces (cem_maxwell_restrict_to_face)
    restrictToFace(mesh, state);

    // @@ computing the numerical fluxes across the faces
    // Numerical flux evaluation (cem_maxwell_flux) using Nanson's formula for face normals
    computeFlux(mesh, state, flux);

    // @@ adding the surface flux integrals to the volume residual
    // Surface flux lifting into volume residuals (cem_maxwell_add_flux_to_res)
    addFluxToRHS(mesh, flux, rhs);

    // @@ multiplying the residual by the inverse mass matrix
    // Diagonal mass matrix scaling (cem_maxwell_invqmass)
    applyInverseMassMatrix(mesh, rhs);
}

// @@ the restriction of volume fields to the face quadrature points
// Extracts traces E^-, H^- onto the quadrilateral faces of each hex element
// Corresponds to cem_maxwell_restrict_to_face in cem_maxwell.F
void Physics::restrictToFace(const Mesh& mesh, const StateVector& state) {
    int npts = mesh.getTotalPoints();
    const auto& faceData = mesh.getFaceData();

    size_t totalFacePoints = 0;
    for (const auto& fd : faceData) {
        totalFacePoints += fd.points.size();
    }

    if (m_fEN.size() != 3 * totalFacePoints) {
        m_fEN.resize(3 * totalFacePoints);
        m_fHN.resize(3 * totalFacePoints);
    }

    const double* Ex = &state[0 * npts];
    const double* Ey = &state[1 * npts];
    const double* Ez = &state[2 * npts];
    const double* Hx = &state[3 * npts];
    const double* Hy = &state[4 * npts];
    const double* Hz = &state[5 * npts];

    size_t ptIdx = 0;
    for (const auto& fd : faceData) {
        for (const auto& pt : fd.points) {
            int vM = pt.volIdxMinus;
            m_fEN[0 * totalFacePoints + ptIdx] = Ex[vM];
            m_fEN[1 * totalFacePoints + ptIdx] = Ey[vM];
            m_fEN[2 * totalFacePoints + ptIdx] = Ez[vM];

            m_fHN[0 * totalFacePoints + ptIdx] = Hx[vM];
            m_fHN[1 * totalFacePoints + ptIdx] = Hy[vM];
            m_fHN[2 * totalFacePoints + ptIdx] = Hz[vM];

            ptIdx++;
        }
    }
}

// @@ the calculation of the numerical flux (Upwind or Central)
// Evaluates jumps [[E]] = E^+ - E^-, [[H]] = H^+ - H^-
// PIOLA TRANSFORM FOR SURFACE FLUXES (!)
// Uses Nanson's formula: n * dA = J * J^{-T} * n_ref * dA_ref to transform reference normals to physical space
// Upwind flux: 
//              F^*_H = -0.5 * (n x [[E]]) - 0.5 * C0 * (n x (n x [[H]]))
//              F^*_E = +0.5 * (n x [[H]]) - 0.5 * C0 * (n x (n x [[E]]))
// Matches cem_maxwell_flux3d and cem_maxwell_flux_pec in cem_maxwell.F
void Physics::computeFlux(const Mesh& mesh, const StateVector& state, StateVector& flux) {
    int npts = mesh.getTotalPoints();
    const auto& faceData = mesh.getFaceData();

    size_t totalFacePoints = 0;
    for (const auto& fd : faceData) {
        totalFacePoints += fd.points.size();
    }

    flux.assign(6 * totalFacePoints, 0.0);

    const double* Ex = &state[0 * npts];
    const double* Ey = &state[1 * npts];
    const double* Ez = &state[2 * npts];
    const double* Hx = &state[3 * npts];
    const double* Hy = &state[4 * npts];
    const double* Hz = &state[5 * npts];

    size_t ptIdx = 0;
    for (const auto& fd : faceData) {
        for (const auto& pt : fd.points) {
            double Ex_m = m_fEN[0 * totalFacePoints + ptIdx];
            double Ey_m = m_fEN[1 * totalFacePoints + ptIdx];
            double Ez_m = m_fEN[2 * totalFacePoints + ptIdx];

            double Hx_m = m_fHN[0 * totalFacePoints + ptIdx];
            double Hy_m = m_fHN[1 * totalFacePoints + ptIdx];
            double Hz_m = m_fHN[2 * totalFacePoints + ptIdx];

            double dEx = 0.0, dEy = 0.0, dEz = 0.0;
            double dHx = 0.0, dHy = 0.0, dHz = 0.0;

            if (fd.bcType == "PEC" || pt.volIdxPlus < 0) {
                // PEC mirror conditions:
                // n x E^+ = -n x E^- => [[E]] = -2 E^-
                // n x H^+ = +n x H^- => [[H]] = 0
                dEx = -2.0 * Ex_m;
                dEy = -2.0 * Ey_m;
                dEz = -2.0 * Ez_m;
                dHx = 0.0;
                dHy = 0.0;
                dHz = 0.0;
            } else {
                // Internal connection face: evaluate difference with neighbor trace
                int vP = pt.volIdxPlus;
                dEx = Ex[vP] - Ex_m;
                dEy = Ey[vP] - Ey_m;
                dEz = Ez[vP] - Ez_m;
                dHx = Hx[vP] - Hx_m;
                dHy = Hy[vP] - Hy_m;
                dHz = Hz[vP] - Hz_m;
            }

            // Cross product: n x [[E]]
            double nxE_x = pt.ny * dEz - pt.nz * dEy;
            double nxE_y = pt.nz * dEx - pt.nx * dEz;
            double nxE_z = pt.nx * dEy - pt.ny * dEx;

            // Cross product: n x [[H]]
            double nxH_x = pt.ny * dHz - pt.nz * dHy;
            double nxH_y = pt.nz * dHx - pt.nx * dHz;
            double nxH_z = pt.nx * dHy - pt.ny * dHx;

            // Double cross product: n x (n x [[E]])
            double nxnxE_x = pt.ny * nxE_z - pt.nz * nxE_y;
            double nxnxE_y = pt.nz * nxE_x - pt.nx * nxE_z;
            double nxnxE_z = pt.nx * nxE_y - pt.ny * nxE_x;

            // Double cross product: n x (n x [[H]])
            double nxnxH_x = pt.ny * nxH_z - pt.nz * nxH_y;
            double nxnxH_y = pt.nz * nxH_x - pt.nx * nxH_z;
            double nxnxH_z = pt.nx * nxH_y - pt.ny * nxH_x;

            // Upwind / Central Numerical Flux formula matching NekCEM lines 991-996
            double flxHx = -0.5 * nxE_x - 0.5 * m_c0 * nxnxH_x;
            double flxHy = -0.5 * nxE_y - 0.5 * m_c0 * nxnxH_y;
            double flxHz = -0.5 * nxE_z - 0.5 * m_c0 * nxnxH_z;

            double flxEx = +0.5 * nxH_x - 0.5 * m_c0 * nxnxE_x;
            double flxEy = +0.5 * nxH_y - 0.5 * m_c0 * nxnxE_y;
            double flxEz = +0.5 * nxH_z - 0.5 * m_c0 * nxnxE_z;

            flux[0 * totalFacePoints + ptIdx] = flxHx;
            flux[1 * totalFacePoints + ptIdx] = flxHy;
            flux[2 * totalFacePoints + ptIdx] = flxHz;
            flux[3 * totalFacePoints + ptIdx] = flxEx;
            flux[4 * totalFacePoints + ptIdx] = flxEy;
            flux[5 * totalFacePoints + ptIdx] = flxEz;

            ptIdx++;
        }
    }
}

// @@ the addition of the numerical flux to the right hand side
// Lifts face numerical fluxes into the volume residual using face quadrature areas
// Matches cem_maxwell_add_flux_to_res in cem_maxwell.F
void Physics::addFluxToRHS(const Mesh& mesh, const StateVector& flux, StateVector& rhs) {
    int npts = mesh.getTotalPoints();
    const auto& faceData = mesh.getFaceData();

    size_t totalFacePoints = 0;
    for (const auto& fd : faceData) {
        totalFacePoints += fd.points.size();
    }

    double* resEx = &rhs[0 * npts];
    double* resEy = &rhs[1 * npts];
    double* resEz = &rhs[2 * npts];
    double* resHx = &rhs[3 * npts];
    double* resHy = &rhs[4 * npts];
    double* resHz = &rhs[5 * npts];

    size_t ptIdx = 0;
    for (const auto& fd : faceData) {
        for (const auto& pt : fd.points) {
            int vM = pt.volIdxMinus;
            double a = pt.dA; // Face quadrature metric area

            double flxHx = flux[0 * totalFacePoints + ptIdx];
            double flxHy = flux[1 * totalFacePoints + ptIdx];
            double flxHz = flux[2 * totalFacePoints + ptIdx];
            double flxEx = flux[3 * totalFacePoints + ptIdx];
            double flxEy = flux[4 * totalFacePoints + ptIdx];
            double flxEz = flux[5 * totalFacePoints + ptIdx];

            resHx[vM] += a * flxHx;
            resHy[vM] += a * flxHy;
            resHz[vM] += a * flxHz;

            resEx[vM] += a * flxEx;
            resEy[vM] += a * flxEy;
            resEz[vM] += a * flxEz;

            ptIdx++;
        }
    }
}

// @@ the application of the inverse mass matrix
// Exact diagonal mass matrix inversion (GLL quadrature mass lumping)
//      res_E = res_E / (eps * J * W),  res_H = res_H / (mu * J * W)
// Matches cem_maxwell_invqmass in cem_maxwell.F
void Physics::applyInverseMassMatrix(const Mesh& mesh, StateVector& rhs) {
    int npts = mesh.getTotalPoints();
    int nxyz = mesh.getNumPointsPerElement();
    int nelt = mesh.getNumElements();

    const std::vector<double>& jac = mesh.getJac();
    const std::vector<double>& w3 = mesh.getW3();

    // Material parameters (vacuum: eps = 1.0, mu = 1.0)
    const double eps = 1.0;
    const double mu = 1.0;

    for (int e = 0; e < nelt; ++e) {
        int offset = e * nxyz;
        for (int i = 0; i < nxyz; ++i) {
            int k = offset + i;
            double invMass_E = 1.0 / (eps * jac[k] * w3[i]);
            double invMass_H = 1.0 / (mu * jac[k] * w3[i]);

            // Electric field components: resEx, resEy, resEz
            rhs[0 * npts + k] *= invMass_E;
            rhs[1 * npts + k] *= invMass_E;
            rhs[2 * npts + k] *= invMass_E;

            // Magnetic field components: resHx, resHy, resHz
            rhs[3 * npts + k] *= invMass_H;
            rhs[4 * npts + k] *= invMass_H;
            rhs[5 * npts + k] *= invMass_H;
        }
    }
}
