#include "comm.hpp"
#include "device/gpu_runtime.hpp"
#include <iostream>

bool Comm::s_bInitialized = false;
int Comm::s_rank = 0;
int Comm::s_size = 1;
int Comm::s_gpuId = 0;

void Comm::init(int* argc, char*** argv) {
    if (s_bInitialized) return;

#ifdef NEKWAVE_ENABLE_MPI
    int mpiInit = 0;
    MPI_Initialized(&mpiInit);
    if (!mpiInit) {
        if (argc && argv) {
            MPI_Init(argc, argv);
        } else {
            int fakeArgc = 0;
            char** fakeArgv = nullptr;
            MPI_Init(&fakeArgc, &fakeArgv);
        }
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &s_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &s_size);
#else
    (void)argc;
    (void)argv;
    s_rank = 0;
    s_size = 1;
#endif

    // Bind MPI rank to local GPU
    int numGpus = 0;
    cudaError_t err = cudaGetDeviceCount(&numGpus);
    if (err == cudaSuccess && numGpus > 0) {
        s_gpuId = s_rank % numGpus;
        cudaSetDevice(s_gpuId);
        if (s_size > 1) {
            std::cout << "[MPI] Rank " << s_rank << " of " << s_size 
                      << " successfully bound to GPU " << s_gpuId 
                      << " (" << numGpus << " GPUs available)" << std::endl;
        }
    } else {
        s_gpuId = 0;
    }

    s_bInitialized = true;
}

void Comm::finalize() {
    if (!s_bInitialized) return;

#ifdef NEKWAVE_ENABLE_MPI
    int mpiFinal = 0;
    MPI_Finalized(&mpiFinal);
    if (!mpiFinal) {
        MPI_Finalize();
    }
#endif

    s_bInitialized = false;
}

int Comm::rank() {
    return s_rank;
}

int Comm::size() {
    return s_size;
}

bool Comm::isRoot() {
    return s_rank == 0;
}

int Comm::getGpuId() {
    return s_gpuId;
}

void Comm::barrier() {
#ifdef NEKWAVE_ENABLE_MPI
    if (s_size > 1) {
        MPI_Barrier(MPI_COMM_WORLD);
    }
#endif
}

double Comm::allreduceSum(double val) {
#ifdef NEKWAVE_ENABLE_MPI
    if (s_size > 1) {
        double out = 0.0;
        MPI_Allreduce(&val, &out, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return out;
    }
#endif
    return val;
}

double Comm::allreduceMax(double val) {
#ifdef NEKWAVE_ENABLE_MPI
    if (s_size > 1) {
        double out = 0.0;
        MPI_Allreduce(&val, &out, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        return out;
    }
#endif
    return val;
}

double Comm::allreduceMin(double val) {
#ifdef NEKWAVE_ENABLE_MPI
    if (s_size > 1) {
        double out = 0.0;
        MPI_Allreduce(&val, &out, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        return out;
    }
#endif
    return val;
}

int Comm::allreduceSum(int val) {
#ifdef NEKWAVE_ENABLE_MPI
    if (s_size > 1) {
        int out = 0;
        MPI_Allreduce(&val, &out, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        return out;
    }
#endif
    return val;
}

#ifdef NEKWAVE_ENABLE_MPI
MPI_Comm Comm::world() {
    return MPI_COMM_WORLD;
}
#endif
