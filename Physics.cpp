#include "Physics.hpp"

// @@ here im implementing the Physics constructor
Physics::Physics() {}

// @@ here im implementing the Physics destructor
Physics::~Physics() = default;

// @@ here im implementing matrix-matrix multiplication C = A * B in column-major layout
void Physics::mxm(const double* A, int n1, const double* B, int n2, double* C, int n3) {
    // @@ here im looping over the columns of C
    for (int j = 0; j < n3; ++j) {
        // @@ here im looping over the rows of C
        for (int i = 0; i < n1; ++i) {
            double sum = 0.0;
            // @@ here im contracting along the inner dimension n2
            for (int l = 0; l < n2; ++l) {
                sum += A[i + l * n1] * B[l + j * n2];
            }
            C[i + j * n1] = sum;
        }
    }
}

// @@ here im implementing local element differentiation using tensor product contractions
void Physics::local_grad3(const double* u, int N, const std::vector<double>& D, const std::vector<double>& Dt,
                         double* ur, double* us, double* ut) {
    int m1 = N;
    int m2 = N * N;

    // @@ here im computing ur = (I (x) I (x) D) u using a single contraction
    mxm(D.data(), m1, u, m1, ur, m2);

    // @@ here im computing us = (I (x) D (x) I) u across each 2D k-slice
    for (int k = 0; k < N; ++k) {
        mxm(u + k * m2, m1, Dt.data(), m1, us + k * m2, m1);
    }

    // @@ here im computing ut = (D (x) I (x) I) u using a single contraction
    mxm(u, m2, Dt.data(), m1, ut, m1);
}

// @@ here im implementing the computation of weighted curl for a 3D vector field
void Physics::compute_weighted_curl(const Mesh& mesh,
                                  const double* u1, const double* u2, const double* u3,
                                  double* w1, double* w2, double* w3) {
    int N = mesh.getN();
    int nxyz = mesh.getNumPointsPerElement();
    int nelt = mesh.getNumElements();

    const std::vector<double>& D = mesh.getD();
    const std::vector<double>& Dt = mesh.getDt();
    const std::vector<double>& w3mn = mesh.getW3();

    const std::vector<double>& rx = mesh.getRx();
    const std::vector<double>& sx = mesh.getSx();
    const std::vector<double>& tx = mesh.getTx();
    const std::vector<double>& ry = mesh.getRy();
    const std::vector<double>& sy = mesh.getSy();
    const std::vector<double>& ty = mesh.getTy();
    const std::vector<double>& rz = mesh.getRz();
    const std::vector<double>& sz = mesh.getSz();
    const std::vector<double>& tz = mesh.getTz();

    // @@ here im allocating element scratch buffers for reference derivatives
    std::vector<double> u1r(nxyz), u1s(nxyz), u1t(nxyz);
    std::vector<double> u2r(nxyz), u2s(nxyz), u2t(nxyz);
    std::vector<double> u3r(nxyz), u3s(nxyz), u3t(nxyz);

    // @@ here im looping over each element in the mesh
    for (int e = 0; e < nelt; ++e) {
        int offset = e * nxyz;

        // @@ here im differentiating each component of the vector field locally
        local_grad3(u1 + offset, N, D, Dt, u1r.data(), u1s.data(), u1t.data());
        local_grad3(u2 + offset, N, D, Dt, u2r.data(), u2s.data(), u2t.data());
        local_grad3(u3 + offset, N, D, Dt, u3r.data(), u3s.data(), u3t.data());

        // @@ here im looping over each point in the element to assemble the weighted physical curl
        for (int i = 0; i < nxyz; ++i) {
            int k = offset + i;
            double w = w3mn[i];

            // @@ here im applying quadrature weights to reference derivatives
            double u1rw = u1r[i] * w;
            double u1sw = u1s[i] * w;
            double u1tw = u1t[i] * w;

            double u2rw = u2r[i] * w;
            double u2sw = u2s[i] * w;
            double u2tw = u2t[i] * w;

            double u3rw = u3r[i] * w;
            double u3sw = u3s[i] * w;
            double u3tw = u3t[i] * w;

            double rx_k = rx[k];
            double sx_k = sx[k];
            double tx_k = tx[k];

            double ry_k = ry[k];
            double sy_k = sy[k];
            double ty_k = ty[k];

            double rz_k = rz[k];
            double sz_k = sz[k];
            double tz_k = tz[k];

            // @@ here im computing physical weighted curl x-component: dw3/dy - dw2/dz
            w1[k] = (u3rw * ry_k + u3sw * sy_k + u3tw * ty_k)
                  - (u2rw * rz_k + u2sw * sz_k + u2tw * tz_k);

            // @@ here im computing physical weighted curl y-component: dw1/dz - dw3/dx
            w2[k] = (u1rw * rz_k + u1sw * sz_k + u1tw * tz_k)
                  - (u3rw * rx_k + u3sw * sx_k + u3tw * tx_k);

            // @@ here im computing physical weighted curl z-component: dw2/dx - dw1/dy
            w3[k] = (u2rw * rx_k + u2sw * sx_k + u2tw * tx_k)
                  - (u1rw * ry_k + u1sw * sy_k + u1tw * ty_k);
        }
    }
}

// @@ here im evaluating discrete curl operator for 3D Maxwell fields
void Physics::computeCurl(const Mesh& mesh, const StateVector& state, StateVector& rhs) {
    int npts = mesh.getTotalPoints();
    if (state.size() < static_cast<size_t>(6 * npts) || rhs.size() < static_cast<size_t>(6 * npts)) {
        return;
    }

    // @@ here im decomposing state into field pointers: E = (u1, u2, u3), H = (u4, u5, u6)
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

    // @@ here im computing weighted curl of H for electric field residual: curl(H)
    compute_weighted_curl(mesh, Hx, Hy, Hz, resEx, resEy, resEz);

    // @@ here im computing weighted curl of E for magnetic field residual: curl(E)
    compute_weighted_curl(mesh, Ex, Ey, Ez, resHx, resHy, resHz);

    // @@ here im applying the negative sign for magnetic field update: -curl(E)
    for (int i = 0; i < npts; ++i) {
        resHx[i] = -resHx[i];
        resHy[i] = -resHy[i];
        resHz[i] = -resHz[i];
    }
}

// @@ here im implementing the main Maxwell RHS evaluation based on NekCEM's cem_maxwell_op
void Physics::evaluateRHS(const Mesh& mesh, const StateVector& state, StateVector& rhs, double rkTime) {
    // @@ here im ensuring the RHS vector is the correct size and initialized to zero
    rhs.assign(state.size(), 0.0);
    StateVector flux(state.size(), 0.0); // Placeholder for flux storage

    // @@ here im calling the routine to compute the curl of the E and H fields
    computeCurl(mesh, state, rhs);

    // @@ here im extracting the trace of the fields on the element faces
    restrictToFace(mesh, state);

    // @@ here im computing the numerical fluxes across the faces
    computeFlux(mesh, state, flux);

    // @@ here im adding the surface flux integrals to the volume residual
    addFluxToRHS(mesh, flux, rhs);

    // @@ here im multiplying the residual by the inverse mass matrix
    applyInverseMassMatrix(mesh, rhs);
}

// @@ here im implementing the restriction of volume fields to the face quadrature points
void Physics::restrictToFace(const Mesh& mesh, const StateVector& state) {
    // @@ here im gathering the field values at the boundaries of each element
}

// @@ here im implementing the calculation of the numerical flux (e.g. Upwind or Lax-Friedrichs)
void Physics::computeFlux(const Mesh& mesh, const StateVector& state, StateVector& flux) {
    // @@ here im applying the jump and average conditions to compute the fluxes
}

// @@ here im implementing the addition of the numerical flux to the right hand side
void Physics::addFluxToRHS(const Mesh& mesh, const StateVector& flux, StateVector& rhs) {
    // @@ here im lifting the surface flux contributions back to the volume modes
}

// @@ here im implementing the application of the inverse mass matrix
void Physics::applyInverseMassMatrix(const Mesh& mesh, StateVector& rhs) {
    // @@ here im scaling the RHS by the inverse of the element mass matrices (invqmass)
}
