#ifndef NW_SRC_COMM_HPP
#define NW_SRC_COMM_HPP

#include <string>
#include <vector>
#include <iostream>

#ifdef NEKWAVE_ENABLE_MPI
#include <mpi.h>
#endif

/**
 * @brief Global communicator and multi-GPU hardware manager.
 *
 * Provides a unified MPI interface for rank topology, GPU device binding,
 * and global collective reductions. Transparently falls back to single-GPU
 * serial execution when built without MPI.
 */
class Comm {
public:
    /**
     * @brief Initializes MPI and assigns local GPU via cudaSetDevice(rank % numGpus).
     */
    static void init(int* argc = nullptr, char*** argv = nullptr);

    /**
     * @brief Finalizes MPI environment.
     */
    static void finalize();

    /**
     * @brief Returns global MPI rank ID (0 if serial).
     */
    static int rank();

    /**
     * @brief Returns total number of MPI ranks (1 if serial).
     */
    static int size();

    /**
     * @brief Returns true if current rank is root (rank 0).
     */
    static bool isRoot();

    /**
     * @brief Blocks until all ranks reach this barrier.
     */
    static void barrier();

    /**
     * @brief Global collective reduction: Sum of doubles.
     */
    static double allreduceSum(double val);

    /**
     * @brief Global collective reduction: Maximum of doubles.
     */
    static double allreduceMax(double val);

    /**
     * @brief Global collective reduction: Minimum of doubles.
     */
    static double allreduceMin(double val);

    /**
     * @brief Global collective reduction: Sum of integers.
     */
    static int allreduceSum(int val);

    /**
     * @brief Returns active CUDA GPU device index bound to this rank.
     */
    static int getGpuId();

#ifdef NEKWAVE_ENABLE_MPI
    /**
     * @brief Native MPI communicator.
     */
    static MPI_Comm world();
#endif

private:
    static bool s_bInitialized;
    static int s_rank;
    static int s_size;
    static int s_gpuId;
};

#endif // NW_SRC_COMM_HPP
