#ifndef NW_SRC_DG_SOLVER_HPP
#define NW_SRC_DG_SOLVER_HPP

#include <memory>
#include <vector>
#include <cstddef>

/**
 * @brief State vector type alias representing 6 electromagnetic field components:
 *        [Ex, Ey, Ez, Hx, Hy, Hz], each of length totalPoints (N_elt * N_p).
 *
 * Stored in Structure-of-Arrays (SoA) layout for coalesced GPU global memory transactions.
 */
using StateVector = std::vector<double>;

class Mesh;

/**
 * @brief Fully GPU-resident Discontinuous Galerkin Spectral Element (DG-SEM) Maxwell solver.
 *
 * Implements the explicit spatial Discontinuous Galerkin operator pipeline for 3D
 * time-domain Maxwell curl equations:
 *      du/dt = M^{-1} [ K * u + F^* ]
 * where:
 *   - u = [E, H]^T is the 6-component electromagnetic state vector
 *   - K * u is the volume weak/strong curl operator evaluated via tensor-product contractions
 *   - F^* is the numerical surface flux (central or upwind with PEC mirror conditions)
 *   - M^{-1} is the exact diagonal inverse mass matrix on GLL collocation points
 *
 * Keeps state vectors, mesh metrics, face quadrature data, and Runge-Kutta
 * auxiliary vectors in GPU VRAM across all time steps. Zero host-to-device
 * or device-to-host transfers occur during the simulation time-stepping loop,
 * except when observation probes or field snapshots are exported.
 */
class DgSolver {
public:
    // Initializing DgSolver host coordinator
    DgSolver();
    // Cleaning up GPU resources and buffers
    ~DgSolver();

    DgSolver(const DgSolver&) = delete;
    DgSolver& operator=(const DgSolver&) = delete;

    /**
     * @brief Allocates GPU memory and uploads static mesh metrics and face data.
     *
     * Uploads:
     *   - 1D Gauss-Lobatto-Legendre (GLL) differentiation matrix D and weights w3
     *   - Geometric metric terms: Jacobian J, coordinate derivatives (rx, sx, tx, ry, sy, ty, rz, sz, tz)
     *   - Face connectivity, outward normal vectors (nx, ny, nz), and surface metric elements dA
     *
     * @param mesh Reference to initialized Mesh object.
     * @param c0   Numerical flux penalty parameter (0.0 for energy-conserving central flux,
     *             1.0 for strictly dissipative upwind flux).
     * @return true on success, false otherwise.
     */
    bool initialize(const Mesh& mesh, double c0);

    /**
     * @brief Copies initial condition state vector from host RAM to GPU VRAM.
     *
     * @param hostState Pointer to contiguous host state buffer [Ex, Ey, Ez, Hx, Hy, Hz].
     * @param size      Total number of entries (6 * totalPoints).
     */
    void uploadState(const double* hostState, size_t size);

    /**
     * @brief Copies current state vector from GPU VRAM back to host RAM.
     *
     * Used for observation probes, continuous energy history, and field exports.
     *
     * @param hostState Pointer to destination host state buffer.
     * @param size      Total number of entries (6 * totalPoints).
     */
    void downloadState(double* hostState, size_t size) const;

    /**
     * @brief Advances Maxwell state by one complete time step using 5-stage LSRK45 on GPU.
     *
     * For each of the 5 Low-Storage Runge-Kutta stages:
     *   1. Evaluates volume curl: resE += +curl(H) / eps, resH += -curl(E) / mu
     *   2. Restricts interior volume fields to boundary and interface element faces
     *   3. Computes upwind / central numerical Riemann fluxes (with PEC mirror conditions)
     *   4. Lifts numerical surface fluxes into volume residuals
     *   5. Applies diagonal inverse mass matrix M^{-1} = 1 / (J * w_i * w_j * w_k)
     *   6. Updates state vector and auxiliary vector 'k' via Carpenter-Kennedy LSRK45
     *
     * @param dt   Time step size.
     * @param time Current simulation time.
     */
    void step(double dt, double time);

    /**
     * @brief Releases all allocated GPU device buffers and streams.
     */
    void finalize();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

namespace nekwave {
using DgSolver = ::DgSolver;
}

#endif // NW_SRC_DG_SOLVER_HPP
