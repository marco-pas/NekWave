#ifndef NW_CUDA_GPU_SOLVER_HPP
#define NW_CUDA_GPU_SOLVER_HPP

#include <memory>
#include <cstddef>

class Mesh;

/**
 * @brief Fully GPU-resident time-domain Maxwell solver.
 *
 * Keeps state vectors, mesh metrics, face quadrature data, and Runge-Kutta
 * auxiliary vectors in GPU VRAM across all time steps. Zero host-to-device
 * or device-to-host transfers occur during the simulation time-stepping loop,
 * except when diagnostics or field snapshots are exported.
 */
class GpuSolver {
public:
    GpuSolver();
    ~GpuSolver();

    GpuSolver(const GpuSolver&) = delete;
    GpuSolver& operator=(const GpuSolver&) = delete;

    /**
     * @brief Allocates GPU memory and uploads static mesh metrics and face data.
     *
     * @param mesh Reference to initialized Mesh object.
     * @param c0   Numerical flux penalty parameter (0.0 for central, 1.0 for upwind).
     * @return true on success, false otherwise.
     */
    bool initialize(const Mesh& mesh, double c0);

    /**
     * @brief Copies initial condition state vector from host RAM to GPU VRAM.
     */
    void uploadState(const double* hostState, size_t size);

    /**
     * @brief Copies current state vector from GPU VRAM back to host RAM.
     */
    void downloadState(double* hostState, size_t size) const;

    /**
     * @brief Advances Maxwell state by one complete time step using 5-stage LSRK45 on GPU.
     *
     * @param dt   Time step size.
     * @param time Current simulation time.
     */
    void step(double dt, double time);

    /**
     * @brief Releases all allocated GPU device buffers.
     */
    void finalize();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

#endif // NW_CUDA_GPU_SOLVER_HPP
