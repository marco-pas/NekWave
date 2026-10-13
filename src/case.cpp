// Three phases:
//     1) Preprocessing
//     2) Simulation
//     3) Postprocessing
// The case gets taken from the /examples folder

#include "case.hpp"
#include "comm.hpp"
#include "device/gpu_runtime.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <csignal>
#include <chrono>
#include <sys/stat.h>
#include <sys/resource.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {
volatile sig_atomic_t g_terminationSignalReceived = 0;
void nekwaveSignalHandler(int /*signum*/) {
    g_terminationSignalReceived = 1;
}

size_t getHostPeakRssBytes() {
    struct rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return static_cast<size_t>(usage.ru_maxrss) * 1024ULL;
    }
    return 0;
}
} // namespace

Case::Case()
    : dt_(0.0)
    , cfl_(0.0)
    , currentTime_(0.0)
    , currentStep_(0)
    , bIsPreprocessed_(false)
    , bIsSimulated_(false)
    , bIsPostprocessed_(false)
{
    mesh_ = std::unique_ptr<Mesh>(new Mesh());
    dgSolver_ = std::unique_ptr<DgSolver>(new DgSolver());
}

Case::Case(const std::string& configFile)
    : Case()
{
    loadConfig(configFile);
}

Case::~Case() {
    if (energyFile_.is_open()) {
        energyFile_.close();
    }
}

void Case::loadConfig(const std::string& configFile) {
    if (!configFile.empty()) {
        if (!config_.loadFromFile(configFile)) {
            if (Comm::isRoot()) {
                std::cerr << "[FATAL ERROR] Failed to load configuration file '" << configFile
                          << "'. Terminating simulation." << std::endl;
            }
            Comm::finalize();
            std::exit(EXIT_FAILURE);
        }
        // If waveType_ was already specified programmatically in C++, ensure it takes precedence
        if (!waveType_.empty()) {
            config_.set("wave_type", waveType_);
        }
    }
}

void Case::setWaveType(const std::string& waveType) {
    waveType_ = waveType;
    config_.set("wave_type", waveType);
}

const std::string& Case::waveType() const {
    return waveType_;
}

void Case::setConfig(const Config& config) {
    config_ = config;
    if (!waveType_.empty()) {
        config_.set("wave_type", waveType_);
    }
}

void Case::setInitialCondition(InitialConditionFn fn) {
    initialConditionHook_ = fn;
}

void Case::setPostprocessingHook(PostprocessingFn fn) {
    postprocessingHook_ = fn;
}

void Case::addProbe(double x, double y, double z) {
    customProbes_.push_back({{x, y, z}});
}

void Case::addRelativeProbe(double rx, double ry, double rz) {
    customRelativeProbes_.push_back({{rx, ry, rz}});
}

// -------------------------
// @@ Phase 1: Preprocessing
// -------------------------
void Case::preprocess() {
    preprocess(config_);
}

void Case::preprocess(const Config& cfg) {
    using Clock = std::chrono::high_resolution_clock;
    auto tPreStart = Clock::now();
    NW_PROFILER_PUSH("Case::preprocess");

    Comm::init();
    config_ = cfg;
    timings_ = PhaseTimings{};

    bool bSaveOutput = (!config_.outputDir.empty() && 
                        config_.outputDir != "none" && 
                        config_.outputDir != "None" && 
                        config_.outputDir != "NONE");

#ifdef NEKWAVE_BUILD_DIR
    // If outputDir is relative, always direct it inside the CMake build folder
    if (bSaveOutput && config_.outputDir[0] != '/') {
        config_.outputDir = std::string(NEKWAVE_BUILD_DIR) + "/" + config_.outputDir;
    }
#endif

    const std::string& outDir = config_.outputDir;
    if (bSaveOutput) {
        mkdir(outDir.c_str(), 0755);
    }

    mesh_->setN(config_.order);
    mesh_->setPmlConfig(PmlConfig(config_.pmlThickness, config_.pmlOrder, config_.pmlReflectErr));
    bSolverInitializedInPreprocess_ = false;
    hasNonZeroInitialCondition_ = false;

    auto tMeshStart = Clock::now();
    NW_PROFILER_PUSH("Case::loadMesh");
    // Mesh setup: Built-in box generator or external .rea mesh file
    if (config_.nelx > 0 && config_.nely > 0 && config_.nelz > 0) {
        if (Comm::isRoot()) {
            std::cout << "[PREPROCESS] Generating Cartesian box mesh: "
                      << config_.nelx << " x " << config_.nely << " x " << config_.nelz
                      << " = " << config_.nelx * config_.nely * config_.nelz 
                      << " elements (poly_order p = " << (config_.order - 1) << ", N = " << config_.order << ")" << std::endl;
            std::cout << "             Domain: [" << config_.xmin << ", " << config_.xmax << "] x ["
                      << config_.ymin << ", " << config_.ymax << "] x ["
                      << config_.zmin << ", " << config_.zmax << "]" << std::endl;
            if (config_.periodicX || config_.periodicY || config_.periodicZ) {
                std::cout << "             Boundary Conditions: Periodic in ["
                          << (config_.periodicX ? "X " : "")
                          << (config_.periodicY ? "Y " : "")
                          << (config_.periodicZ ? "Z " : "")
                          << "]" << std::endl;
            } else {
                std::cout << "             Boundary Conditions: " << config_.defaultBc << std::endl;
            }
        }
        mesh_->createBoxMesh(config_.nelx, config_.nely, config_.nelz,
                             config_.xmin, config_.xmax,
                             config_.ymin, config_.ymax,
                             config_.zmin, config_.zmax,
                             config_.periodicX, config_.periodicY, config_.periodicZ,
                             config_.defaultBc);
    } else if (!config_.meshFile.empty() && config_.meshFile != "box") {
        if (Comm::isRoot()) {
            std::cout << "[PREPROCESS] Loading mesh file: " << config_.meshFile 
                      << " (poly_order p = " << (config_.order - 1) << ", N = " << config_.order << ")" << std::endl;
        }
        bool bOk = false;
        if (config_.meshFile.find(".re2") != std::string::npos) {
            bOk = mesh_->loadFromRe2(config_.meshFile);
        } else {
            bOk = mesh_->loadFromRea(config_.meshFile);
        }
        if (!bOk) {
            if (Comm::isRoot()) {
                std::cerr << "[FATAL ERROR] Specified mesh file '" << config_.meshFile
                          << "' could not be found or loaded. Terminating simulation." << std::endl;
            }
            NW_PROFILER_POP();
            NW_PROFILER_POP();
            Comm::finalize();
            std::exit(EXIT_FAILURE);
        }
        if (Comm::isRoot()) {
            std::cout << "[PREPROCESS] Successfully loaded mesh '" << config_.meshFile
                      << "': " << mesh_->getNumElements() << " elements ("
                      << mesh_->getTotalPoints() << " GLL collocation points)." << std::endl;
        }
    } else {
        if (Comm::isRoot()) {
            std::cout << "[PREPROCESS] Using default 3x3x3 Cartesian box mesh (27 elements, Order N = " 
                      << config_.order << ")" << std::endl;
        }
        mesh_->createBoxMesh(3, 3, 3, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
                             config_.periodicX, config_.periodicY, config_.periodicZ,
                             config_.defaultBc);
    }

    std::string currentWave = !waveType_.empty() ? waveType_ : config_.getString("wave_type");
    {
        double minX = 1e30, maxX = -1e30;
        double minY = 1e30, maxY = -1e30;
        double minZ = 1e30, maxZ = -1e30;
        mesh_->getCornerBoundingBox(minX, maxX, minY, maxY, minZ, maxZ);
        if (currentWave == "3dboxper" && std::abs(minX - (-1.0)) < 1e-3 && std::abs(maxX - 1.0) < 1e-3) {
            std::cout << "[PREPROCESS] Rescaling mesh domain from [-1, 1]^3 to [0, 2*pi]^3 matching NekCEM usrdat2..." << std::endl;
            mesh_->rescale(0.0, 2.0 * M_PI, 0.0, 2.0 * M_PI, 0.0, 2.0 * M_PI);
            minX = 0.0; maxX = 2.0 * M_PI;
            minY = 0.0; maxY = 2.0 * M_PI;
            minZ = 0.0; maxZ = 2.0 * M_PI;
        }
        if (minX < maxX) {
            config_.xmin = minX; config_.xmax = maxX; config_.Lx = maxX - minX;
            config_.ymin = minY; config_.ymax = maxY; config_.Ly = maxY - minY;
            config_.zmin = minZ; config_.zmax = maxZ; config_.Lz = maxZ - minZ;
        }
    }
    NW_PROFILER_POP();
    timings_.loadMeshSec += std::chrono::duration<double>(Clock::now() - tMeshStart).count();

    // Partition mesh across MPI ranks if running in parallel
    if (Comm::size() > 1) {
        NW_PROFILER_PUSH("Case::partitionMesh");
        mesh_->partition(Comm::rank(), Comm::size());
        NW_PROFILER_POP();
    }

    const auto& meshSub = mesh_->getSubTimings();
    timings_.meshParseAndTopoSec = meshSub.fileReadAndConnectivitySec;
    timings_.pmlSetupSec         = meshSub.pmlInitSec;

    // Assemble probe coordinates early so we know whether any host coordinates are needed
    std::vector<std::array<double, 3>> allProbes = config_.probes;
    for (const auto& p : customProbes_) {
        allProbes.push_back(p);
    }
    const double xc = 0.5 * (config_.xmin + config_.xmax);
    const double yc = 0.5 * (config_.ymin + config_.ymax);
    const double zc = 0.5 * (config_.zmin + config_.zmax);
    for (const auto& r : customRelativeProbes_) {
        allProbes.push_back({{xc + r[0] * config_.Lx, yc + r[1] * config_.Ly, zc + r[2] * config_.Lz}});
    }
    if (allProbes.empty() && config_.has("num_probes")) {
        int numProbes = config_.getInt("num_probes", 0);
        if (numProbes > 0) {
            if (config_.nelx == 1 && config_.nely == 1 && config_.nelz == 1) {
                std::cout << "[PREPROCESS] Generating " << numProbes 
                          << " observation probes placed at GLL collocation nodes..." << std::endl;
                const auto& gllZ = mesh_->getGllZ();
                int N = mesh_->getN();
                int actualProbes = std::min(numProbes, N);
                for (int p = 0; p < actualProbes; ++p) {
                    double xp = config_.xmin + 0.5 * (1.0 + gllZ[p]) * config_.Lx;
                    allProbes.push_back({{xp, yc, zc}});
                }
            } else {
                std::cout << "[PREPROCESS] Generating " << numProbes 
                          << " collinear observation probes along centerline x-axis..." << std::endl;
                const double xStart = config_.xmin + 0.05 * config_.Lx;
                const double xEnd   = config_.xmax - 0.05 * config_.Lx;
                const double dxP    = (numProbes > 1) ? (xEnd - xStart) / (numProbes - 1) : 0.0;
                for (int p = 0; p < numProbes; ++p) {
                    allProbes.push_back({{xStart + p * dxP, yc, zc}});
                }
            }
        }
    }

    hasNonZeroInitialCondition_ = (initialConditionHook_ != nullptr ||
                                   currentWave == "bloch" ||
                                   currentWave == "periodic" ||
                                   currentWave == "multimode" ||
                                   currentWave == "packet" ||
                                   currentWave == "gaussian");
    const bool needHostCoordsAndJac = hasNonZeroInitialCondition_ ||
                                      (bSaveOutput && config_.exportFields) ||
                                      !allProbes.empty() ||
                                      rcsMonitor_.config().enabled ||
                                      config_.outputEnergy ||
                                      postprocessingHook_ != nullptr;
    const bool needHostState = needHostCoordsAndJac || config_.outputMaxFields;

    const int npts = mesh_->getTotalPoints();
    double dxminLocal = 0.0;

    if (Comm::isRoot()) {
        std::cout << "[PREPROCESS] Executing GPU-native mesh preprocessing (metrics, face normals, UPML sigma, CFL dxmin)..." << std::endl;
    }
    dgSolver_.reset(new DgSolver());
    dgSolver_->preprocessAndInitialize(*mesh_, config_.c0, &dxminLocal, needHostCoordsAndJac);
    dgSolver_->setIncidentWave(rcsMonitor_.config().incidentPlaneWave);
    bSolverInitializedInPreprocess_ = true;

    const auto gpuPre = dgSolver_->getGpuPreprocessTimings();
    timings_.usedGpuPreprocess = gpuPre.usedGpuPreprocess;
    timings_.gpuTopoUploadSec  = gpuPre.topoUploadSec;
    timings_.volumeMetricsSec  = gpuPre.volumeMetricsKernelSec;
    timings_.faceSetupSec      = gpuPre.faceSetupKernelSec;
    timings_.gpuPmlSigmaSec    = gpuPre.pmlSigmaKernelSec;
    timings_.cflAndDtSec       = gpuPre.minDistKernelSec;
    timings_.h2dTopologyBytes  = gpuPre.h2dTopologyBytes;

    if (needHostState) {
        state_.assign(6 * static_cast<size_t>(npts), 0.0);
    } else {
        state_.clear();
        state_.shrink_to_fit();
    }

    const double dxmin = Comm::allreduceMin(dxminLocal);
    const double waveSpeed = 1.0;

    if (config_.dt > 0.0) {
        dt_ = config_.dt;
        cfl_ = (dt_ * waveSpeed) / dxmin;
    } else if (config_.cfl > 0.0) {
        cfl_ = config_.cfl;
        dt_ = (cfl_ * dxmin) / waveSpeed;
    } else {
        cfl_ = mesh_->computeAutomaticCFL(0.3);
        dt_ = (cfl_ * dxmin) / waveSpeed;
    }

    currentTime_ = 0.0;
    currentStep_ = 0;

    int actualNumModes = config_.getInt("num_modes", 1);
    std::vector<double> actualModeK;

    auto tInitFieldsStart = Clock::now();
    NW_PROFILER_PUSH("Case::initFields");
    // Field initialization!
    if (initialConditionHook_) {
        std::cout << "[PREPROCESS] Evaluating user initial condition hook..." << std::endl;
        const auto& x = mesh_->getCoordX();
        const auto& y = mesh_->getCoordY();
        const auto& z = mesh_->getCoordZ();
        double* Ex = &state_[0 * npts];
        double* Ey = &state_[1 * npts];
        double* Ez = &state_[2 * npts];
        double* Hx = &state_[3 * npts];
        double* Hy = &state_[4 * npts];
        double* Hz = &state_[5 * npts];

        for (int k = 0; k < npts; ++k) {
            double ex = 0.0, ey = 0.0, ez = 0.0;
            double hx = 0.0, hy = 0.0, hz = 0.0;
            initialConditionHook_(x[k], y[k], z[k], ex, ey, ez, hx, hy, hz);
            Ex[k] = ex;
            Ey[k] = ey;
            Ez[k] = ez;
            Hx[k] = hx;
            Hy[k] = hy;
            Hz[k] = hz;
        }
    } else if (currentWave == "bloch" || currentWave == "periodic") {
        std::cout << "[PREPROCESS] Evaluating periodic Bloch electromagnetic plane wave in (Ez, Hx, Hy)..." << std::endl;
        const auto& x = mesh_->getCoordX();
        const auto& y = mesh_->getCoordY();
        double* Ez = &state_[2 * npts];
        double* Hx = &state_[3 * npts];
        double* Hy = &state_[4 * npts];
        const double Lx = config_.Lx;
        const double Ly = config_.Ly;
        const double xmin = config_.xmin;
        const double ymin = config_.ymin;
        const double carrierK = config_.getDouble("carrier_k", 2.0 * M_PI / Lx);
        const double angleDeg = config_.getDouble("angle_deg", 0.0);
        const double angleRad = angleDeg * M_PI / 180.0;

        // Reciprocal lattice vector (mx, my) to guarantee exact periodic continuity across periodic boundaries
        double kx = 0.0, ky = 0.0;
        if (std::abs(angleDeg - 0.0) < 1e-4) {
            kx = 2.0 * M_PI / Lx; ky = 0.0;
        } else if (std::abs(angleDeg - 45.0) < 1e-4) {
            kx = 2.0 * M_PI / Lx; ky = 2.0 * M_PI / Ly;
        } else if (std::abs(angleDeg - 90.0) < 1e-4) {
            kx = 0.0; ky = 2.0 * M_PI / Ly;
        } else if (std::abs(angleDeg - 26.6) < 1.0 || std::abs(angleDeg - 30.0) < 5.0) {
            kx = 4.0 * M_PI / Lx; ky = 2.0 * M_PI / Ly;
        } else if (std::abs(angleDeg - 18.4) < 1.0 || std::abs(angleDeg - 15.0) < 5.0) {
            kx = 6.0 * M_PI / Lx; ky = 2.0 * M_PI / Ly;
        } else {
            int mx = std::max(1, (int)std::round(std::cos(angleRad) * 4.0));
            int my = (int)std::round(std::sin(angleRad) * 4.0);
            kx = 2.0 * M_PI * mx / Lx;
            ky = 2.0 * M_PI * my / Ly;
        }
        const double kMag = std::sqrt(kx * kx + ky * ky);
        const double cosA = kx / kMag;
        const double sinA = ky / kMag;
        int pDegree = mesh_->getN() - 1;
        int totalDofX = config_.nelx * pDegree;
        int maxModes = std::max(1, totalDofX / 2);
        int requestedModes = config_.getInt("num_modes", 1);
        int numModes = std::min(requestedModes, maxModes);
        if (requestedModes > maxModes) {
            std::cout << "[PREPROCESS] Notice: num_modes clamped from " << requestedModes 
                      << " to " << numModes << " (grid Nyquist limit = " 
                      << maxModes << " modes for " << config_.nelx << " elements of order N=" 
                      << mesh_->getN() << ")." << std::endl;
        }
        actualNumModes = numModes;

        double kStart = config_.getDouble("k_start", 0.0);
        double kEnd   = config_.getDouble("k_end", 0.0);
        if (kStart > 0.0 && kEnd > 0.0) {
            std::cout << "[PREPROCESS] Generating " << numModes 
                      << " linspace modes from k = " << kStart 
                      << " to " << kEnd << " rad/m" << std::endl;
            for (int m = 0; m < numModes; ++m) {
                double km = (numModes > 1) ? kStart + m * (kEnd - kStart) / (numModes - 1) : kStart;
                actualModeK.push_back(km);
            }
        } else {
            for (int m = 1; m <= numModes; ++m) {
                actualModeK.push_back(m * kMag);
            }
        }

        for (int k = 0; k < npts; ++k) {
            double xt = x[k] - xmin;
            double yt = y[k] - ymin;
            double sumE = 0.0;
            if (numModes > 1) {
                for (size_t m = 0; m < actualModeK.size(); ++m) {
                    double phase = actualModeK[m] * (cosA * xt + sinA * yt);
                    sumE += std::cos(phase);
                }
                sumE /= std::sqrt((double)numModes);
            } else {
                double phase = (config_.has("carrier_k") && std::abs(angleDeg) < 1e-4)
                                ? carrierK * xt
                                : (kx * xt + ky * yt);
                sumE = std::cos(phase);
            }
            Ez[k] = sumE;
            Hx[k] =  sinA * sumE;
            Hy[k] = -cosA * sumE;
        }
    } else if (currentWave == "multimode") {
        std::cout << "[PREPROCESS] Evaluating multi-harmonic sinusoidal standing wave packet in Ez..." << std::endl;
        const auto& x = mesh_->getCoordX();
        const auto& y = mesh_->getCoordY();
        double* Ez = &state_[2 * npts];
        const double Lx = config_.Lx;
        const double Ly = config_.Ly;
        const double xmin = config_.xmin;
        const double ymin = config_.ymin;
        for (int k = 0; k < npts; ++k) {
            double xt = x[k] - xmin;
            double yt = y[k] - ymin;
            double transY = std::sin(M_PI * yt / Ly);
            double sumE = 0.0;
            for (int m = 1; m <= 8; ++m) {
                double km = m * M_PI / Lx;
                sumE += (1.0 / m) * std::sin(km * xt);
            }
            Ez[k] = sumE * transY;
        }
    } else if (currentWave == "packet") {
        std::cout << "[PREPROCESS] Evaluating modulated Gaussian wavepacket in Ez..." << std::endl;
        const auto& x = mesh_->getCoordX();
        const auto& y = mesh_->getCoordY();
        const double carrierK = config_.getDouble("carrier_k", 4.0 * M_PI);
        const double sigma    = config_.getDouble("packet_sigma", 0.30);
        const double x0       = config_.getDouble("packet_x0", config_.xmin + 0.25 * config_.Lx);
        const double xmin     = config_.xmin;
        const double ymin     = config_.ymin;
        const double Ly       = config_.Ly;
        double* Ez = &state_[2 * npts];
        for (int k = 0; k < npts; ++k) {
            double yt = y[k] - ymin;
            double transY = std::sin(M_PI * yt / Ly);
            double dxEnv = x[k] - x0;
            double envelope = std::exp(-(dxEnv * dxEnv) / (2.0 * sigma * sigma));
            double carrier  = std::sin(carrierK * (x[k] - xmin));
            Ez[k] = envelope * carrier * transY;
        }
    } else if (currentWave == "gaussian") {
        std::cout << "[PREPROCESS] Evaluating Gaussian pulse in Ez..." << std::endl;
        const auto& x = mesh_->getCoordX();
        const auto& y = mesh_->getCoordY();
        const auto& z = mesh_->getCoordZ();
        const double sigma = config_.pulseSigma;
        double* Ez = &state_[2 * npts];
        for (int k = 0; k < npts; ++k) {
            double r2 = x[k] * x[k] + y[k] * y[k] + z[k] * z[k];
            Ez[k] = std::exp(-sigma * r2);
        }
    } else {
        std::cout << "[PREPROCESS] No initial condition hook provided; all electromagnetic fields initialized to 0.0 on GPU." << std::endl;
    }
    NW_PROFILER_POP();

    NW_PROFILER_PUSH("Case::initProbes");
    probes_.init(*mesh_, allProbes, outDir, actualNumModes, config_.Lx, config_.nelx, actualModeK);
    rcsMonitor_.config().c0 = config_.c0;
    rcsMonitor_.init(*mesh_, outDir);
    NW_PROFILER_POP();
    timings_.initFieldsAndProbesSec += std::chrono::duration<double>(Clock::now() - tInitFieldsStart).count();

    // Synchronize maxSteps and numSteps
    int maxSteps = -1;
    if (config_.maxSteps != 100) maxSteps = config_.maxSteps;
    else if (config_.numSteps != 100) maxSteps = config_.numSteps;
    else if (config_.hasExplicitMaxSteps) maxSteps = 100;
    else if (config_.finalTime <= 0.0) maxSteps = 100;
    config_.maxSteps = maxSteps;
    config_.numSteps = maxSteps;

    if (config_.saveFreq <= 0) {
        config_.saveFreq = config_.outputFreq;
    }

    // Export initial fields (t = 0) if enabled
    StateVector exportState;
    if (bSaveOutput && config_.exportFields) {
        auto tIoStart = Clock::now();
        NW_PROFILER_PUSH("Case::initialExport");
        reconstructTotalField(0.0, exportState);
        if (config_.exportFormat == "hdf5" || config_.exportFormat == "vtk") {
            std::string prec = config_.exportPrecision;
            if (prec == "float32" || prec == "fp32" || prec == "single" || prec == "f32" || prec == "4") {
                saveOptions_.useFloat32 = true;
            } else if (prec == "float64" || prec == "fp64" || prec == "double" || prec == "f64" || prec == "8") {
                saveOptions_.useFloat32 = false;
            }
            saveOptions_.compressionLevel = std::max(0, std::min(9, config_.hdf5Compression));
            saveOptions_.enableShuffle = config_.hdf5Shuffle;

            hdf5Writer_.reset(new Hdf5Writer());
            hdf5Writer_->setFieldSaveOptions(saveOptions_);
            hdf5Writer_->initialize(outDir + "/fields.h5", *mesh_, true, config_.exportContinuous);
            bool needDerivatives = saveOptions_.saveCurlE || saveOptions_.saveCurlH ||
                                   saveOptions_.saveDivE || saveOptions_.saveDivH ||
                                   saveOptions_.saveMagnitudeCurlE || saveOptions_.saveMagnitudeCurlH;
            std::vector<double> divE, divH, curlE, curlH;
            if (needDerivatives) {
                computeFieldDerivatives(exportState.data(), divE, divH, curlE, curlH);
            }
            Hdf5Writer::DerivedFields derived{
                divE.empty() ? nullptr : divE.data(),
                divH.empty() ? nullptr : divH.data(),
                curlE.empty() ? nullptr : curlE.data(),
                curlH.empty() ? nullptr : curlH.data()
            };
            hdf5Writer_->writeStep(0, 0.0, exportState.data(), npts, &derived);
        } else {
            saveFields(outDir + "/field_initial.csv", 0.0);
        }
        NW_PROFILER_POP();
        timings_.hdf5IoSec += std::chrono::duration<double>(Clock::now() - tIoStart).count();
    }

    // Record t = 0 on probes and RCS surface monitor only if active
    if (probes_.getNumProbes() > 0) {
        if (rcsMonitor_.config().incidentPlaneWave.enabled) {
            if (exportState.empty()) {
                reconstructTotalField(0.0, exportState);
            }
            probes_.record(0, 0.0, exportState, npts);
        } else {
            probes_.record(0, 0.0, state_, npts);
        }
    }
    if (rcsMonitor_.isEnabled()) {
        rcsMonitor_.record(0, 0.0, 0.5 * dt_, state_, npts);
    }

    // Evaluate initial diagnostics across all ranks (only if enabled in config)
    double initEnergy = config_.outputEnergy ? computeTotalEnergy() : 0.0;
    double initMaxE = config_.outputMaxFields ? getMaxE() : 0.0;
    double initMaxH = config_.outputMaxFields ? getMaxH() : 0.0;

    // Initialize continuous energy log if output and at least one diagnostic are enabled
    if (Comm::isRoot() && bSaveOutput && (config_.outputMaxFields || config_.outputEnergy)) {
        energyFile_.open(outDir + "/energy_history.csv");
        if (energyFile_.is_open()) {
            energyFile_ << "step,time,maxE,maxH,energy\n";
            energyFile_ << std::scientific << std::setprecision(8);
            energyFile_ << 0 << "," << 0.0 << ",";
            if (config_.outputMaxFields) energyFile_ << initMaxE << "," << initMaxH << ",";
            else energyFile_ << "-,-,";
            if (config_.outputEnergy) energyFile_ << initEnergy << "\n";
            else energyFile_ << "-\n";
            energyFile_.flush();
        }
    }

    if (Comm::isRoot()) {
        std::cout << "\n----------------------------------------------------------" << std::endl;
        std::cout << "               NekWave Solver Setup Summary               " << std::endl;
        std::cout << "----------------------------------------------------------\n" << std::endl;
        std::cout << "  Elements:          " << mesh_->getNumElements() << std::endl;
        std::cout << "  Polynomial Order:  p = " << (mesh_->getN() - 1) << " (N = " << mesh_->getN() << " GLL pts/dir, Np = " << mesh_->getNumPointsPerElement() << " nodes/elem)" << std::endl;
        std::cout << "  Total Collocation: " << npts << " points" << std::endl;
        std::cout << "  Domain Dimensions: Lx = " << config_.Lx << ", Ly = " << config_.Ly << ", Lz = " << config_.Lz << std::endl;
        std::cout << "  Bounding Box:      [" << config_.xmin << ", " << config_.xmax << "] x [" 
                  << config_.ymin << ", " << config_.ymax << "] x [" 
                  << config_.zmin << ", " << config_.zmax << "]" << std::endl;
        std::cout << "  Faces Total:       " << (mesh_->hasHostFaceData() ? mesh_->getFaceData().size() : mesh_->getFaces().size()) << std::endl;
        std::cout << "  Flux Formulation:  " << ((config_.c0 == 0.0) ? "Central (C0 = 0.0, energy-conserving)" : "Upwind (C0 = 1.0, dissipative)") << std::endl;
        std::cout << "  Field Formulation: " << (config_.scatteredFieldMode ? "Scattered-Field (analytical E_inc on PEC, total field on save)" : "Total-Field") << std::endl;
        std::cout << "  Probes Active:     " << allProbes.size() << " locations in " << outDir << "/" << std::endl;
        std::cout << "  Minimum Spacing:   dxmin = " << std::scientific << std::setprecision(2) << dxmin << std::endl;
        std::cout << "  CFL Number:        " << std::scientific << std::setprecision(2) << cfl_ << std::endl;
        std::cout << "  Time Step:         dt = " << std::scientific << std::setprecision(2) << dt_ << std::endl;
        if (config_.maxSteps > 0) {
            std::cout << "  Max Steps:         " << config_.maxSteps << std::endl;
        }
        if (config_.finalTime > 0.0) {
            std::cout << "  Final Time:        " << std::scientific << std::setprecision(2) << config_.finalTime << std::endl;
        }
        std::cout << "  Output Frequency:  " << config_.outputFreq << " steps" << std::endl;
        std::cout << "  Save Frequency:    " << config_.saveFreq << " steps" << std::endl;
        if (bSaveOutput && config_.exportFields) {
            std::cout << "  Export Format:     " << config_.exportFormat
                      << " (" << (saveOptions_.useFloat32 ? "float32" : "float64")
                      << ", deflate=" << saveOptions_.compressionLevel
                      << (saveOptions_.compressionLevel > 0 && saveOptions_.enableShuffle ? "+shuffle" : "")
                      << ", " << (config_.exportContinuous ? "continuous CG" : "discontinuous DG") << ")" << std::endl;
        }
        if (config_.outputEnergy) {
            std::cout << "  Initial Energy:    " << std::scientific << std::setprecision(2) << initEnergy << std::endl;
        } else {
            std::cout << "  Initial Energy:    -" << std::endl;
        }
        if (config_.outputMaxFields) {
            std::cout << "  Initial max|E|:    " << std::scientific << std::setprecision(2) << initMaxE 
                      << " | max|H|: " << std::scientific << std::setprecision(2) << initMaxH << std::endl;
        } else {
            std::cout << "  Initial max|E|:    - | max|H|: -" << std::endl;
        }
        std::cout << "\n----------------------------------------------------------\n" << std::endl;
    }

    bIsPreprocessed_ = true;
    NW_PROFILER_POP();
    timings_.preprocessSec += std::chrono::duration<double>(Clock::now() - tPreStart).count();
}

// ----------------------
// @@ Phase 2: Simulation
// ----------------------
void Case::simulate() {
    assert(bIsPreprocessed_ && "Must call preprocess() before simulate()");
    using Clock = std::chrono::high_resolution_clock;
    auto tSimStart = Clock::now();
    NW_PROFILER_PUSH("Case::simulate");

    const int npts = mesh_->getTotalPoints();
    const int maxSteps = config_.maxSteps;
    const double finalTime = config_.finalTime;
    const int outFreq = std::max(1, config_.outputFreq);
    const int saveFreq = std::max(1, config_.saveFreq);

    if (Comm::isRoot()) {
        std::cout << "[SIMULATE] Initializing full GPU data residency (DgSolver)..." << std::endl;
    }
    auto tInitStart = Clock::now();
    NW_PROFILER_PUSH("Case::initSolverState");
    if (!bSolverInitializedInPreprocess_ || !dgSolver_) {
        dgSolver_.reset(new DgSolver());
        dgSolver_->initialize(*mesh_, config_.c0);
    }
    dgSolver_->setIncidentWave(rcsMonitor_.config().incidentPlaneWave);
    if (!state_.empty() && (hasNonZeroInitialCondition_ || !bSolverInitializedInPreprocess_)) {
        dgSolver_->uploadState(state_.data(), state_.size());
    }
    lastMemHwStats_ = dgSolver_->getMemoryAndHardwareStats();
    NW_PROFILER_POP();
    timings_.initSolverSec += std::chrono::duration<double>(Clock::now() - tInitStart).count();

    if (Comm::isRoot()) {
        const double mb = 1.0 / (1024.0 * 1024.0);
        const double gb = 1.0 / (1024.0 * 1024.0 * 1024.0);
        size_t hostRss = getHostPeakRssBytes();
        std::cout << "\n----------------------------------------------------------" << std::endl;
        std::cout << "             Hardware & Memory Footprint (Rank 0)         " << std::endl;
        std::cout << "----------------------------------------------------------" << std::endl;
        std::cout << "  GPU Device:        " << lastMemHwStats_.deviceName
                  << " (" << lastMemHwStats_.computeUnits << " CUs/SMs, "
                  << std::fixed << std::setprecision(0) << (lastMemHwStats_.clockRateKHz / 1000.0) << " MHz)" << std::endl;
        std::cout << "  GPU VRAM Solver:   " << std::fixed << std::setprecision(2)
                  << (lastMemHwStats_.totalNekWaveVramBytes * mb) << " MiB total allocated by DgSolver" << std::endl;
        std::cout << "    - State & LSRK:  " << (lastMemHwStats_.stateBuffersBytes * mb) << " MiB" << std::endl;
        std::cout << "    - Vol Metrics:   " << (lastMemHwStats_.metricBuffersBytes * mb) << " MiB" << std::endl;
        std::cout << "    - Face Topo:     " << (lastMemHwStats_.faceTopologyBytes * mb) << " MiB" << std::endl;
        std::cout << "    - Face Scratch:  " << (lastMemHwStats_.faceScratchBytes * mb) << " MiB" << std::endl;
        if (lastMemHwStats_.pmlBuffersBytes > 0) {
            std::cout << "    - UPML Aux:      " << (lastMemHwStats_.pmlBuffersBytes * mb) << " MiB" << std::endl;
        }
        if (lastMemHwStats_.mpiHaloBuffersBytes > 0) {
            std::cout << "    - MPI Halos:     " << (lastMemHwStats_.mpiHaloBuffersBytes * mb) << " MiB" << std::endl;
        }
        if (lastMemHwStats_.totalGlobalMemBytes > 0) {
            std::cout << "  GPU Driver VRAM:   " << (lastMemHwStats_.driverUsedVramBytes * mb) << " MiB used / "
                      << (lastMemHwStats_.totalGlobalMemBytes * gb) << " GiB total" << std::endl;
        }
        std::cout << "  Host CPU Peak RSS: " << (hostRss * mb) << " MiB" << std::endl;
        std::cout << "----------------------------------------------------------" << std::endl;

        std::cout << "\n[SIMULATE] Advancing time integration on GPU (LSRK45)..." << std::endl;
        std::cout << std::setw(8) << "Step" 
                  << std::setw(14) << "Time" 
                  << std::setw(14) << "max|E|" 
                  << std::setw(14) << "max|H|" 
                  << std::setw(16) << "Total Energy" << std::endl;
        std::cout << "------------------------------------------------------------------" << std::endl;
    }

    const double initEnergy = config_.outputEnergy ? computeTotalEnergy() : 0.0;
    double peakScatteredEnergy = 0.0;
    const bool isScatteredMode = rcsMonitor_.config().incidentPlaneWave.enabled;
    const auto& incWave = rcsMonitor_.config().incidentPlaneWave;
    const double tPulsePassedSphere = std::abs(incWave.x0) + incWave.sphereRadius + 4.5 * incWave.sigmaX;
    StateVector exportState;

    const bool hasActiveProbes   = (probes_.getNumProbes() > 0);
    const bool hasActiveRcs      = rcsMonitor_.isEnabled();
    const bool needDiagReduction = (config_.outputMaxFields || config_.outputEnergy);
    bool lastStepStateDownloaded = false;

    g_terminationSignalReceived = 0;
    std::signal(SIGTERM, nekwaveSignalHandler);
    std::signal(SIGINT, nekwaveSignalHandler);

    int step = 0;
    while (true) {
        // Termination condition: either step count reached or final time reached
        if (maxSteps > 0 && step >= maxSteps) {
            break;
        }
        if (finalTime > 0.0 && currentTime_ >= finalTime - 1e-13) {
            break;
        }

        // Sub-step clamping to hit finalTime exactly if needed
        double currentDt = dt_;
        if (finalTime > 0.0 && currentTime_ + currentDt > finalTime) {
            currentDt = finalTime - currentTime_;
        }

        NW_PROFILER_PUSH("Case::step");
        step++;
        auto tStepStart = Clock::now();
        dgSolver_->step(currentDt, currentTime_);
        timings_.solverStepSec += std::chrono::duration<double>(Clock::now() - tStepStart).count();
        currentTime_ += currentDt;
        currentStep_ = step;

        bool isFinal = (maxSteps > 0 && step >= maxSteps) || (finalTime > 0.0 && currentTime_ >= finalTime - 1e-13);
        bool willSave = (step % saveFreq == 0 || isFinal) && (hdf5Writer_ != nullptr || hasActiveRcs);
        bool willDiag = (step % outFreq == 0 || isFinal);

        bool stateDownloaded = false;
        if (hasActiveProbes || hasActiveRcs || willSave || (willDiag && needDiagReduction)) {
            auto tDlStart = Clock::now();
            NW_PROFILER_PUSH("Case::downloadState");
            dgSolver_->downloadState(state_.data(), state_.size());
            NW_PROFILER_POP();
            timings_.downloadStateSec += std::chrono::duration<double>(Clock::now() - tDlStart).count();
            stateDownloaded = true;
        }

        bool exportStateReconstructed = false;
        if (hasActiveProbes || hasActiveRcs) {
            auto tProbesStart = Clock::now();
            NW_PROFILER_PUSH("Case::probesRecord");
            if (hasActiveProbes) {
                if (isScatteredMode) {
                    reconstructTotalField(currentTime_, exportState);
                    exportStateReconstructed = true;
                    probes_.record(step, currentTime_, exportState, npts);
                } else {
                    probes_.record(step, currentTime_, state_, npts);
                }
            }
            if (hasActiveRcs) {
                rcsMonitor_.record(step, currentTime_, currentDt, state_, npts);
            }
            NW_PROFILER_POP();
            timings_.probesAndRcsSec += std::chrono::duration<double>(Clock::now() - tProbesStart).count();
        }

        // Save HDF5/VTK snapshots and checkpoint RCS spectra according to saveFreq or on final step
        if (step % saveFreq == 0 || isFinal) {
            auto tIoStart = Clock::now();
            if (hdf5Writer_) {
                NW_PROFILER_PUSH("Case::hdf5WriteStep");
                if (isScatteredMode && !exportStateReconstructed) {
                    reconstructTotalField(currentTime_, exportState);
                    exportStateReconstructed = true;
                }
                const double* fieldPtr = isScatteredMode ? exportState.data() : state_.data();
                bool needDerivatives = saveOptions_.saveCurlE || saveOptions_.saveCurlH ||
                                       saveOptions_.saveDivE || saveOptions_.saveDivH ||
                                       saveOptions_.saveMagnitudeCurlE || saveOptions_.saveMagnitudeCurlH;
                std::vector<double> divE, divH, curlE, curlH;
                if (needDerivatives) {
                    computeFieldDerivatives(fieldPtr, divE, divH, curlE, curlH);
                }
                Hdf5Writer::DerivedFields derived{
                    divE.empty() ? nullptr : divE.data(),
                    divH.empty() ? nullptr : divH.data(),
                    curlE.empty() ? nullptr : curlE.data(),
                    curlH.empty() ? nullptr : curlH.data()
                };
                hdf5Writer_->writeStep(step, currentTime_, fieldPtr, npts, &derived);
                NW_PROFILER_POP();
            }
            if (hasActiveRcs) {
                rcsMonitor_.writeSpectra(false);
            }
            timings_.hdf5IoSec += std::chrono::duration<double>(Clock::now() - tIoStart).count();
        }

        // Output diagnostics according to outFreq or on final step
        bool stopSimulation = false;
        if (step % outFreq == 0 || isFinal) {
            double sigFlag = (g_terminationSignalReceived != 0) ? 1.0 : 0.0;
            if (Comm::allreduceMax(sigFlag) > 0.0) {
                if (Comm::isRoot()) {
                    std::cout << "\n[STOP] Termination signal (SIGTERM/SIGINT) received at step " << step
                              << " (t = " << std::fixed << std::setprecision(5) << currentTime_
                              << "). Saving state and finalizing cleanly..." << std::endl;
                }
                if (hdf5Writer_ && !(step % saveFreq == 0 || isFinal)) {
                    auto tIoStart = Clock::now();
                    if (!stateDownloaded) {
                        dgSolver_->downloadState(state_.data(), state_.size());
                        stateDownloaded = true;
                    }
                    if (isScatteredMode && !exportStateReconstructed) {
                        reconstructTotalField(currentTime_, exportState);
                        exportStateReconstructed = true;
                    }
                    const double* fieldPtr = isScatteredMode ? exportState.data() : state_.data();
                    bool needDerivatives = saveOptions_.saveCurlE || saveOptions_.saveCurlH ||
                                           saveOptions_.saveDivE || saveOptions_.saveDivH ||
                                           saveOptions_.saveMagnitudeCurlE || saveOptions_.saveMagnitudeCurlH;
                    std::vector<double> divE, divH, curlE, curlH;
                    if (needDerivatives) {
                        computeFieldDerivatives(fieldPtr, divE, divH, curlE, curlH);
                    }
                    Hdf5Writer::DerivedFields derived{
                        divE.empty() ? nullptr : divE.data(),
                        divH.empty() ? nullptr : divH.data(),
                        curlE.empty() ? nullptr : curlE.data(),
                        curlH.empty() ? nullptr : curlH.data()
                    };
                    hdf5Writer_->writeStep(step, currentTime_, fieldPtr, npts, &derived);
                    timings_.hdf5IoSec += std::chrono::duration<double>(Clock::now() - tIoStart).count();
                }
                stopSimulation = true;
            }

            double maxE = 0.0;
            double maxH = 0.0;
            double energy = 0.0;
            if (needDiagReduction) {
                auto tDiagStart = Clock::now();
                NW_PROFILER_PUSH("Case::computeDiagnostics");
                if (config_.outputMaxFields) {
                    maxE = getMaxE();
                    maxH = getMaxH();
                }
                if (config_.outputEnergy) {
                    energy = computeTotalEnergy();
                    if (energy > peakScatteredEnergy) {
                        peakScatteredEnergy = energy;
                    }
                }
                NW_PROFILER_POP();
                timings_.diagnosticsSec += std::chrono::duration<double>(Clock::now() - tDiagStart).count();
            }

            if (Comm::isRoot()) {
                if (energyFile_.is_open()) {
                    energyFile_ << step << "," << currentTime_ << ",";
                    if (config_.outputMaxFields) energyFile_ << maxE << "," << maxH << ",";
                    else energyFile_ << "-,-,";
                    if (config_.outputEnergy) energyFile_ << energy << "\n";
                    else energyFile_ << "-\n";
                    energyFile_.flush();
                }

                std::cout << std::setw(8) << step 
                          << std::setw(14) << std::fixed << std::setprecision(5) << currentTime_;
                if (config_.outputMaxFields) {
                    std::cout << std::setw(14) << std::fixed << std::setprecision(5) << maxE 
                              << std::setw(14) << std::fixed << std::setprecision(5) << maxH;
                } else {
                    std::cout << std::setw(14) << "-"
                              << std::setw(14) << "-";
                }
                if (config_.outputEnergy) {
                    std::cout << std::setw(16) << std::scientific << std::setprecision(6) << energy << std::endl;
                } else {
                    std::cout << std::setw(16) << "-" << std::endl;
                }
            }

            // Automatic energy-based stopping criteria (only evaluated when output_energy is enabled):
            if (config_.outputEnergy && initEnergy > 0.0) {
                // Total-Field mode:
                //  1) Stop if energy exceeds +5% of initial energy (numerical instability guard)
                //  2) Stop if energy drops to <= 50% of initial energy (wave energy absorbed / exited domain)
                if (std::isnan(energy) || std::isinf(energy) || energy > 1.05 * initEnergy) {
                    if (Comm::isRoot()) {
                        std::cout << "\n[STOP] Simulation terminated at step " << step
                                  << " (t = " << std::fixed << std::setprecision(5) << currentTime_ << "): "
                                  << "Total energy (" << std::scientific << std::setprecision(6) << energy
                                  << ") exceeded +5% of initial energy (" << initEnergy
                                  << "). Stopping to prevent numerical instability." << std::endl;
                    }
                    stopSimulation = true;
                } else if (energy <= 0.50 * initEnergy) {
                    if (Comm::isRoot()) {
                        std::cout << "\n[STOP] Simulation completed at step " << step
                                  << " (t = " << std::fixed << std::setprecision(5) << currentTime_ << "): "
                                  << "Total energy (" << std::scientific << std::setprecision(6) << energy
                                  << ") dropped to <= 50% of initial energy (" << initEnergy
                                  << ") as wave energy exited the domain." << std::endl;
                    }
                    if (hdf5Writer_ && !(step % saveFreq == 0 || isFinal)) {
                        auto tIoStart = Clock::now();
                        const double* fieldPtr = isScatteredMode ? exportState.data() : state_.data();
                        bool needDerivatives = saveOptions_.saveCurlE || saveOptions_.saveCurlH ||
                                               saveOptions_.saveDivE || saveOptions_.saveDivH ||
                                               saveOptions_.saveMagnitudeCurlE || saveOptions_.saveMagnitudeCurlH;
                        std::vector<double> divE, divH, curlE, curlH;
                        if (needDerivatives) {
                            computeFieldDerivatives(fieldPtr, divE, divH, curlE, curlH);
                        }
                        Hdf5Writer::DerivedFields derived{
                            divE.empty() ? nullptr : divE.data(),
                            divH.empty() ? nullptr : divH.data(),
                            curlE.empty() ? nullptr : curlE.data(),
                            curlH.empty() ? nullptr : curlH.data()
                        };
                        hdf5Writer_->writeStep(step, currentTime_, fieldPtr, npts, &derived);
                        timings_.hdf5IoSec += std::chrono::duration<double>(Clock::now() - tIoStart).count();
                    }
                    stopSimulation = true;
                }
            } else if (config_.outputEnergy && isScatteredMode) {
                // Scattered-Field mode (initial energy at t=0 is 0):
                //  1) Stop if NaN/Inf or runaway energy > 1e3
                //  2) Stop once the incident pulse has passed the sphere and scattered energy drops to <= 0.5% of peak
                if (std::isnan(energy) || std::isinf(energy) || energy > 1.0e3) {
                    if (Comm::isRoot()) {
                        std::cout << "\n[STOP] Simulation terminated at step " << step
                                  << " (t = " << std::fixed << std::setprecision(5) << currentTime_ << "): "
                                  << "Scattered energy (" << std::scientific << std::setprecision(6) << energy
                                  << ") exceeded stability limit." << std::endl;
                    }
                    stopSimulation = true;
                } else if (currentTime_ > tPulsePassedSphere && peakScatteredEnergy > 1e-12 &&
                           energy <= 0.005 * peakScatteredEnergy) {
                    if (Comm::isRoot()) {
                        std::cout << "\n[STOP] Simulation completed at step " << step
                                  << " (t = " << std::fixed << std::setprecision(5) << currentTime_ << "): "
                                  << "Scattered energy (" << std::scientific << std::setprecision(6) << energy
                                  << ") decayed to <= 0.5% of peak scattered energy (" << peakScatteredEnergy
                                  << ") as scattered waves exited into the UPML." << std::endl;
                    }
                    if (hdf5Writer_ && !(step % saveFreq == 0 || isFinal)) {
                        auto tIoStart = Clock::now();
                        const double* fieldPtr = exportState.data();
                        bool needDerivatives = saveOptions_.saveCurlE || saveOptions_.saveCurlH ||
                                               saveOptions_.saveDivE || saveOptions_.saveDivH ||
                                               saveOptions_.saveMagnitudeCurlE || saveOptions_.saveMagnitudeCurlH;
                        std::vector<double> divE, divH, curlE, curlH;
                        if (needDerivatives) {
                            computeFieldDerivatives(fieldPtr, divE, divH, curlE, curlH);
                        }
                        Hdf5Writer::DerivedFields derived{
                            divE.empty() ? nullptr : divE.data(),
                            divH.empty() ? nullptr : divH.data(),
                            curlE.empty() ? nullptr : curlE.data(),
                            curlH.empty() ? nullptr : curlH.data()
                        };
                        hdf5Writer_->writeStep(step, currentTime_, fieldPtr, npts, &derived);
                        timings_.hdf5IoSec += std::chrono::duration<double>(Clock::now() - tIoStart).count();
                    }
                    stopSimulation = true;
                }
            }
        }
        lastStepStateDownloaded = stateDownloaded;
        NW_PROFILER_POP();
        if (stopSimulation) {
            break;
        }
    }

    if (!lastStepStateDownloaded &&
        (needDiagReduction || config_.exportFields || postprocessingHook_ != nullptr)) {
        auto tDlEndStart = Clock::now();
        dgSolver_->downloadState(state_.data(), state_.size());
        timings_.downloadStateSec += std::chrono::duration<double>(Clock::now() - tDlEndStart).count();
    }

    bIsSimulated_ = true;
    NW_PROFILER_POP();
    timings_.simulateTotalSec += std::chrono::duration<double>(Clock::now() - tSimStart).count();
}

// --------------------------
// @@ Phase 3: Postprocessing
// --------------------------
void Case::postprocess() {
    assert(bIsSimulated_ && "Must call simulate() before postprocess()");
    using Clock = std::chrono::high_resolution_clock;
    auto tPostStart = Clock::now();
    NW_PROFILER_PUSH("Case::postprocess");

    const std::string& outDir = config_.outputDir;
    bool bSaveOutput = (!outDir.empty() && outDir != "none" && outDir != "None" && outDir != "NONE");

    // Finalize probe observations and Near-to-Far-Field (NTFF) RCS calculation
    NW_PROFILER_PUSH("Case::probesFinalize");
    probes_.finalize();
    rcsMonitor_.finalize();
    NW_PROFILER_POP();

    // Close energy history log
    if (energyFile_.is_open()) {
        energyFile_.close();
        if (Comm::isRoot()) {
            std::cout << "[POSTPROCESS] Exported energy history to " << outDir << "/energy_history.csv" << std::endl;
        }
    }

    // Finalize HDF5 time series writer if active
    if (hdf5Writer_) {
        auto tIoStart = Clock::now();
        NW_PROFILER_PUSH("Case::hdf5Close");
        hdf5Writer_->close();
        NW_PROFILER_POP();
        timings_.hdf5IoSec += std::chrono::duration<double>(Clock::now() - tIoStart).count();
    }

    // Export final field distributions if enabled and format is not HDF5/VTK
    if (bSaveOutput && config_.exportFields && config_.exportFormat != "hdf5" && config_.exportFormat != "vtk") {
        saveFields(outDir + "/field_final.csv", currentTime_);
    }

    if (Comm::isRoot()) {
        std::cout << "------------------------------------------------------------------" << std::endl;
        std::cout << "Simulation completed successfully at t = " << std::scientific << std::setprecision(6) << currentTime_ << std::endl;
    }

    // Execute user-defined postprocessing hook if registered
    if (postprocessingHook_) {
        if (Comm::isRoot()) {
            std::cout << "[POSTPROCESS] Executing custom user postprocessing hook..." << std::endl;
        }
        postprocessingHook_(*this);
    }

    // Refresh peak memory stats before releasing GPU memory
    if (dgSolver_) {
        lastMemHwStats_ = dgSolver_->getMemoryAndHardwareStats();
        dgSolver_->finalize();
    }

    NW_PROFILER_POP();
    timings_.postprocessSec += std::chrono::duration<double>(Clock::now() - tPostStart).count();

    if (Comm::isRoot()) {
        const double totalWallSec = std::max(1e-9, timings_.preprocessSec + timings_.simulateTotalSec + timings_.postprocessSec);
        const int stepsDone = std::max(1, currentStep_);
        const double nptsTotal = static_cast<double>(mesh_->getTotalPoints());
        const double mdofPerSec = (nptsTotal * 6.0 * stepsDone / 1.0e6) / std::max(1e-9, timings_.solverStepSec);
        size_t hostPeakRss = getHostPeakRssBytes();

        std::cout << "\n----------------------------------------------------------" << std::endl;
        std::cout << "              HPC Performance & Trace Summary             " << std::endl;
        std::cout << "----------------------------------------------------------" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Total Wall Time:       " << std::setw(8) << totalWallSec << " s (100.0%)" << std::endl;
        std::cout << "    - Preprocess:        " << std::setw(8) << timings_.preprocessSec << " s ("
                  << std::setprecision(1) << (100.0 * timings_.preprocessSec / totalWallSec) << "%)" << std::setprecision(3) << std::endl;
        std::cout << "      * Mesh I/O & Topo: " << std::setw(8) << timings_.meshParseAndTopoSec << " s (CPU)" << std::endl;
        std::cout << "      * PML Topology:    " << std::setw(8) << timings_.pmlSetupSec << " s (CPU)" << std::endl;
        std::cout << "      * GPU Topo H2D:    " << std::setw(8) << timings_.gpuTopoUploadSec << " s ("
                  << std::setprecision(1) << (timings_.h2dTopologyBytes / (1024.0 * 1024.0)) << " MiB)" << std::setprecision(3) << std::endl;
        std::cout << "      * GPU Vol Metrics: " << std::setw(8) << timings_.volumeMetricsSec << " s (GPU kernel)" << std::endl;
        std::cout << "      * GPU Face Setup:  " << std::setw(8) << timings_.faceSetupSec << " s (GPU kernel)" << std::endl;
        std::cout << "      * GPU PML Sigma:   " << std::setw(8) << timings_.gpuPmlSigmaSec << " s (GPU kernel)" << std::endl;
        std::cout << "      * GPU CFL dxmin:   " << std::setw(8) << timings_.cflAndDtSec << " s (GPU kernel)" << std::endl;
        std::cout << "      * Init Fields/Prb: " << std::setw(8) << timings_.initFieldsAndProbesSec << " s" << std::endl;
        std::cout << "    - GPU Solver Init:   " << std::setw(8) << timings_.initSolverSec << " s ("
                  << std::setprecision(1) << (100.0 * timings_.initSolverSec / totalWallSec) << "%)" << std::setprecision(3) << std::endl;
        std::cout << "    - GPU LSRK45 Steps:  " << std::setw(8) << timings_.solverStepSec << " s ("
                  << std::setprecision(1) << (100.0 * timings_.solverStepSec / totalWallSec) << "%)" << std::setprecision(3) << std::endl;
        std::cout << "    - D2H State Copy:    " << std::setw(8) << timings_.downloadStateSec << " s ("
                  << std::setprecision(1) << (100.0 * timings_.downloadStateSec / totalWallSec) << "%)" << std::setprecision(3) << std::endl;
        std::cout << "    - Probes & NTFF RCS: " << std::setw(8) << timings_.probesAndRcsSec << " s ("
                  << std::setprecision(1) << (100.0 * timings_.probesAndRcsSec / totalWallSec) << "%)" << std::setprecision(3) << std::endl;
        std::cout << "    - HDF5 / Field I/O:  " << std::setw(8) << timings_.hdf5IoSec << " s ("
                  << std::setprecision(1) << (100.0 * timings_.hdf5IoSec / totalWallSec) << "%)" << std::setprecision(3) << std::endl;
        std::cout << "    - Diagnostics:       " << std::setw(8) << timings_.diagnosticsSec << " s ("
                  << std::setprecision(1) << (100.0 * timings_.diagnosticsSec / totalWallSec) << "%)" << std::setprecision(3) << std::endl;
        std::cout << "    - Postprocess:       " << std::setw(8) << timings_.postprocessSec << " s ("
                  << std::setprecision(1) << (100.0 * timings_.postprocessSec / totalWallSec) << "%)" << std::setprecision(3) << std::endl;
        std::cout << "  GPU Step Throughput:   " << std::setprecision(2) << (stepsDone / std::max(1e-9, timings_.solverStepSec))
                  << " steps/s (" << mdofPerSec << " MDOF/s)" << std::endl;
        std::cout << "----------------------------------------------------------\n" << std::endl;

        if (bSaveOutput) {
            std::ofstream tel(outDir + "/hpc_telemetry.json");
            if (tel.is_open()) {
                size_t hostStateBytes = state_.size() * sizeof(double);
                size_t hostMeshBytes = mesh_->getAllocatedHostBytes();
                tel << "{\n"
                    << "  \"hardware\": {\n"
                    << "    \"device_name\": \"" << lastMemHwStats_.deviceName << "\",\n"
                    << "    \"compute_units\": " << lastMemHwStats_.computeUnits << ",\n"
                    << "    \"clock_rate_khz\": " << lastMemHwStats_.clockRateKHz << ",\n"
                    << "    \"memory_clock_rate_khz\": " << lastMemHwStats_.memoryClockRateKHz << ",\n"
                    << "    \"memory_bus_width_bits\": " << lastMemHwStats_.memoryBusWidthBits << ",\n"
                    << "    \"total_global_mem_bytes\": " << lastMemHwStats_.totalGlobalMemBytes << ",\n"
                    << "    \"mpi_ranks\": " << Comm::size() << "\n"
                    << "  },\n"
                    << "  \"problem\": {\n"
                    << "    \"num_elements\": " << lastMemHwStats_.numElements << ",\n"
                    << "    \"poly_order\": " << (lastMemHwStats_.orderN - 1) << ",\n"
                    << "    \"order_n\": " << lastMemHwStats_.orderN << ",\n"
                    << "    \"npts\": " << lastMemHwStats_.npts << ",\n"
                    << "    \"total_face_points\": " << lastMemHwStats_.totalFacePoints << ",\n"
                    << "    \"max_pml_elements\": " << lastMemHwStats_.maxPmlElements << ",\n"
                    << "    \"npts_pml\": " << lastMemHwStats_.nptsPml << ",\n"
                    << "    \"num_halo_points\": " << lastMemHwStats_.numHaloPoints << ",\n"
                    << "    \"steps_executed\": " << currentStep_ << ",\n"
                    << "    \"dt\": " << std::scientific << std::setprecision(8) << dt_ << ",\n"
                    << "    \"cfl\": " << cfl_ << "\n"
                    << "  },\n"
                    << "  \"memory_bytes\": {\n"
                    << "    \"gpu_state_and_rk\": " << lastMemHwStats_.stateBuffersBytes << ",\n"
                    << "    \"gpu_volume_metrics\": " << lastMemHwStats_.metricBuffersBytes << ",\n"
                    << "    \"gpu_face_topology\": " << lastMemHwStats_.faceTopologyBytes << ",\n"
                    << "    \"gpu_face_scratch\": " << lastMemHwStats_.faceScratchBytes << ",\n"
                    << "    \"gpu_pml_auxiliary\": " << lastMemHwStats_.pmlBuffersBytes << ",\n"
                    << "    \"gpu_mpi_halo\": " << lastMemHwStats_.mpiHaloBuffersBytes << ",\n"
                    << "    \"gpu_total_solver\": " << lastMemHwStats_.totalNekWaveVramBytes << ",\n"
                    << "    \"gpu_driver_used\": " << lastMemHwStats_.driverUsedVramBytes << ",\n"
                    << "    \"gpu_driver_free\": " << lastMemHwStats_.driverFreeVramBytes << ",\n"
                    << "    \"gpu_driver_total\": " << lastMemHwStats_.totalGlobalMemBytes << ",\n"
                    << "    \"h2d_preprocess_topology\": " << timings_.h2dTopologyBytes << ",\n"
                    << "    \"host_peak_rss\": " << hostPeakRss << ",\n"
                    << "    \"host_state_vector\": " << hostStateBytes << ",\n"
                    << "    \"host_mesh_metrics\": " << hostMeshBytes << "\n"
                    << "  },\n"
                    << "  \"phase_timings_sec\": {\n"
                    << std::fixed << std::setprecision(6)
                    << "    \"used_gpu_preprocess\": " << (timings_.usedGpuPreprocess ? "true" : "false") << ",\n"
                    << "    \"preprocess\": " << timings_.preprocessSec << ",\n"
                    << "    \"load_mesh\": " << timings_.loadMeshSec << ",\n"
                    << "    \"preprocess_mesh_parse_and_topo\": " << timings_.meshParseAndTopoSec << ",\n"
                    << "    \"preprocess_gpu_topo_upload\": " << timings_.gpuTopoUploadSec << ",\n"
                    << "    \"preprocess_volume_metrics\": " << timings_.volumeMetricsSec << ",\n"
                    << "    \"preprocess_face_setup\": " << timings_.faceSetupSec << ",\n"
                    << "    \"preprocess_pml_setup\": " << timings_.pmlSetupSec << ",\n"
                    << "    \"preprocess_gpu_pml_sigma\": " << timings_.gpuPmlSigmaSec << ",\n"
                    << "    \"preprocess_cfl_and_dt\": " << timings_.cflAndDtSec << ",\n"
                    << "    \"preprocess_init_fields_and_probes\": " << timings_.initFieldsAndProbesSec << ",\n"
                    << "    \"init_solver\": " << timings_.initSolverSec << ",\n"
                    << "    \"gpu_solver_step\": " << timings_.solverStepSec << ",\n"
                    << "    \"device_to_host_copy\": " << timings_.downloadStateSec << ",\n"
                    << "    \"probes_and_rcs\": " << timings_.probesAndRcsSec << ",\n"
                    << "    \"hdf5_io\": " << timings_.hdf5IoSec << ",\n"
                    << "    \"diagnostics\": " << timings_.diagnosticsSec << ",\n"
                    << "    \"postprocess\": " << timings_.postprocessSec << ",\n"
                    << "    \"total_simulate\": " << timings_.simulateTotalSec << ",\n"
                    << "    \"total_wall\": " << totalWallSec << "\n"
                    << "  }\n"
                    << "}\n";
                tel.close();
            }
        }
    }

    bIsPostprocessed_ = true;
}

void Case::run() {
    preprocess();
    simulate();
    postprocess();
}


// Diagnostic and Export Helpers

double Case::computeTotalEnergy() const {
    if (!mesh_) return 0.0;

    const auto& J = mesh_->getJac();
    const auto& w = mesh_->getGllW();
    const int N = mesh_->getN();
    const int Np = N * N * N;
    const int npts = mesh_->getTotalPoints();

    const double* Ex = &state_[0 * npts];
    const double* Ey = &state_[1 * npts];
    const double* Ez = &state_[2 * npts];
    const double* Hx = &state_[3 * npts];
    const double* Hy = &state_[4 * npts];
    const double* Hz = &state_[5 * npts];

    double energy = 0.0;

    for (int e = 0; e < mesh_->getNumElements(); ++e) {
        for (int k = 0; k < N; ++k) {
            for (int j = 0; j < N; ++j) {
                for (int i = 0; i < N; ++i) {
                    int p = e * Np + k * N * N + j * N + i;
                    double dV = J[p] * w[i] * w[j] * w[k];
                    double eSq = Ex[p] * Ex[p] + Ey[p] * Ey[p] + Ez[p] * Ez[p];
                    double hSq = Hx[p] * Hx[p] + Hy[p] * Hy[p] + Hz[p] * Hz[p];
                    energy += 0.5 * (eSq + hSq) * dV;
                }
            }
        }
    }

    return Comm::allreduceSum(energy);
}

double Case::getMaxE() const {
    const int npts = mesh_->getTotalPoints();
    const double* Ex = &state_[0 * npts];
    const double* Ey = &state_[1 * npts];
    const double* Ez = &state_[2 * npts];

    double maxVal = 0.0;
    for (int i = 0; i < npts; ++i) {
        double mag = std::sqrt(Ex[i] * Ex[i] + Ey[i] * Ey[i] + Ez[i] * Ez[i]);
        if (mag > maxVal) maxVal = mag;
    }
    return Comm::allreduceMax(maxVal);
}

double Case::getMaxH() const {
    const int npts = mesh_->getTotalPoints();
    const double* Hx = &state_[3 * npts];
    const double* Hy = &state_[4 * npts];
    const double* Hz = &state_[5 * npts];

    double maxVal = 0.0;
    for (int i = 0; i < npts; ++i) {
        double mag = std::sqrt(Hx[i] * Hx[i] + Hy[i] * Hy[i] + Hz[i] * Hz[i]);
        if (mag > maxVal) maxVal = mag;
    }
    return Comm::allreduceMax(maxVal);
}

void Case::reconstructTotalField(double time, StateVector& outTotalState) const {
    outTotalState = state_;
    if (!mesh_ || !rcsMonitor_.config().incidentPlaneWave.enabled) {
        return;
    }

    const auto& incWave = rcsMonitor_.config().incidentPlaneWave;
    const auto& x = mesh_->getCoordX();
    const auto& y = mesh_->getCoordY();
    const auto& z = mesh_->getCoordZ();
    const int npts = mesh_->getTotalPoints();
    const double rInnerMask = std::max(0.0, incWave.sphereRadius - 1.0e-4);
    const double rInnerMaskSq = rInnerMask * rInnerMask;

    double* Ex = &outTotalState[0 * npts];
    double* Ey = &outTotalState[1 * npts];
    double* Ez = &outTotalState[2 * npts];
    double* Hx = &outTotalState[3 * npts];
    double* Hy = &outTotalState[4 * npts];
    double* Hz = &outTotalState[5 * npts];

    for (int i = 0; i < npts; ++i) {
        double rSq = x[i] * x[i] + y[i] * y[i] + z[i] * z[i];
        if (rSq < rInnerMaskSq) {
            // Strictly inside the PEC sphere: guarantee null total field
            Ex[i] = 0.0; Ey[i] = 0.0; Ez[i] = 0.0;
            Hx[i] = 0.0; Hy[i] = 0.0; Hz[i] = 0.0;
        } else {
            double ex_inc = 0.0, ey_inc = 0.0, ez_inc = 0.0;
            double hx_inc = 0.0, hy_inc = 0.0, hz_inc = 0.0;
            incWave.evaluate(x[i], y[i], z[i], time,
                             ex_inc, ey_inc, ez_inc,
                             hx_inc, hy_inc, hz_inc);
            Ex[i] += ex_inc;
            Ey[i] += ey_inc;
            Ez[i] += ez_inc;
            Hx[i] += hx_inc;
            Hy[i] += hy_inc;
            Hz[i] += hz_inc;
        }
    }

    // On PEC scatterer surface nodes, enforce exact tangential electric field cancellation
    // (n x E_tot = 0 => E_tot = (n . E_tot) n) and zero normal magnetic field (n . H_tot = 0)
    const auto& faces = mesh_->getFaces();
    const auto& faceData = mesh_->getFaceData();
    for (size_t fIdx = 0; fIdx < faceData.size(); ++fIdx) {
        const auto& fd = faceData[fIdx];
        int neighborId = (fIdx < faces.size()) ? faces[fIdx].neighborElementId : -1;
        BcType bc = BoundaryConditions::parseBcTag(fd.bcType, neighborId);
        if (bc != BcType::PEC) continue;

        for (const auto& pt : fd.points) {
            int v = pt.volIdxMinus;
            double nx = pt.nx, ny = pt.ny, nz = pt.nz;
            double eDotN = Ex[v] * nx + Ey[v] * ny + Ez[v] * nz;
            Ex[v] = eDotN * nx;
            Ey[v] = eDotN * ny;
            Ez[v] = eDotN * nz;

            double hDotN = Hx[v] * nx + Hy[v] * ny + Hz[v] * nz;
            Hx[v] -= hDotN * nx;
            Hy[v] -= hDotN * ny;
            Hz[v] -= hDotN * nz;
        }
    }
}

void Case::saveFields(const std::string& filename, double time) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for field export." << std::endl;
        return;
    }

    StateVector exportState;
    reconstructTotalField(time, exportState);

    const auto& x = mesh_->getCoordX();
    const auto& y = mesh_->getCoordY();
    const auto& z = mesh_->getCoordZ();
    const int npts = mesh_->getTotalPoints();

    const double* Ex = &exportState[0 * npts];
    const double* Ey = &exportState[1 * npts];
    const double* Ez = &exportState[2 * npts];
    const double* Hx = &exportState[3 * npts];
    const double* Hy = &exportState[4 * npts];
    const double* Hz = &exportState[5 * npts];

    file << "# NekWave Field Export (t = " << std::scientific << std::setprecision(6) << time << ")\n";
    file << "x,y,z,Ex,Ey,Ez,Hx,Hy,Hz\n";
    file << std::scientific << std::setprecision(8);

    for (int i = 0; i < npts; ++i) {
        file << x[i] << "," << y[i] << "," << z[i] << ","
             << Ex[i] << "," << Ey[i] << "," << Ez[i] << ","
             << Hx[i] << "," << Hy[i] << "," << Hz[i] << "\n";
    }

    file.close();
    std::cout << "Exported field data to " << filename << " (" << npts 
              << " points, t = " << std::scientific << std::setprecision(2) << time << ")" << std::endl;
}

void Case::computeFieldDerivatives(const double* state,
                                   std::vector<double>& divE,
                                   std::vector<double>& divH,
                                   std::vector<double>& curlE,
                                   std::vector<double>& curlH) const {
    if (!mesh_) return;

    int N = mesh_->getN();
    int N3 = N * N * N;
    int nelt = mesh_->getNumElements();
    int npts = mesh_->getTotalPoints();

    divE.assign(npts, 0.0);
    divH.assign(npts, 0.0);
    curlE.assign(3 * npts, 0.0);
    curlH.assign(3 * npts, 0.0);

    const auto& D = mesh_->getD();
    const auto& z = mesh_->getGllZ();
    const auto& corners = mesh_->getElementCorners();

    const double* Ex = &state[0 * npts];
    const double* Ey = &state[1 * npts];
    const double* Ez = &state[2 * npts];
    const double* Hx = &state[3 * npts];
    const double* Hy = &state[4 * npts];
    const double* Hz = &state[5 * npts];

    double* cEx = &curlE[0 * npts];
    double* cEy = &curlE[1 * npts];
    double* cEz = &curlE[2 * npts];
    double* cHx = &curlH[0 * npts];
    double* cHy = &curlH[1 * npts];
    double* cHz = &curlH[2 * npts];

    for (int e = 0; e < nelt; ++e) {
        int elemOffset = e * N3;
        const auto& c = corners[e];
        for (int k = 0; k < N; ++k) {
            double tk = z[k];
            double tm = 0.5 * (1.0 - tk), tp = 0.5 * (1.0 + tk);
            for (int j = 0; j < N; ++j) {
                double sj = z[j];
                double sm = 0.5 * (1.0 - sj), sp = 0.5 * (1.0 + sj);
                for (int i = 0; i < N; ++i) {
                    double ri = z[i];
                    double rm = 0.5 * (1.0 - ri), rp = 0.5 * (1.0 + ri);

                    int m = i + N * (j + N * k);
                    int p = elemOffset + m;

                    double dNdr[8] = {
                        -0.5 * sm * tm,  0.5 * sm * tm,  0.5 * sp * tm, -0.5 * sp * tm,
                        -0.5 * sm * tp,  0.5 * sm * tp,  0.5 * sp * tp, -0.5 * sp * tp
                    };
                    double dNds[8] = {
                        -0.5 * rm * tm, -0.5 * rp * tm,  0.5 * rp * tm,  0.5 * rm * tm,
                        -0.5 * rm * tp, -0.5 * rp * tp,  0.5 * rp * tp,  0.5 * rm * tp
                    };
                    double dNdt[8] = {
                        -0.5 * rm * sm, -0.5 * rp * sm, -0.5 * rp * sp, -0.5 * rm * sp,
                         0.5 * rm * sm,  0.5 * rp * sm,  0.5 * rp * sp,  0.5 * rm * sp
                    };

                    double xr = 0.0, xs = 0.0, xt = 0.0;
                    double yr = 0.0, ys = 0.0, yt = 0.0;
                    double zr = 0.0, zs = 0.0, zt = 0.0;
                    for (int v = 0; v < 8; ++v) {
                        xr += dNdr[v] * c[v][0]; xs += dNds[v] * c[v][0]; xt += dNdt[v] * c[v][0];
                        yr += dNdr[v] * c[v][1]; ys += dNds[v] * c[v][1]; yt += dNdt[v] * c[v][1];
                        zr += dNdr[v] * c[v][2]; zs += dNds[v] * c[v][2]; zt += dNdt[v] * c[v][2];
                    }
                    double detJ = xr * (ys * zt - yt * zs)
                                - xs * (yr * zt - yt * zr)
                                + xt * (yr * zs - ys * zr);
                    double invJ = 1.0 / detJ;
                    double rx_k =  (ys * zt - yt * zs) * invJ;
                    double ry_k = -(xs * zt - xt * zs) * invJ;
                    double rz_k =  (xs * yt - xt * ys) * invJ;
                    double sx_k = -(yr * zt - yt * zr) * invJ;
                    double sy_k =  (xr * zt - xt * zr) * invJ;
                    double sz_k = -(xr * yt - xt * yr) * invJ;
                    double tx_k =  (yr * zs - ys * zr) * invJ;
                    double ty_k = -(xr * zs - xs * zr) * invJ;
                    double tz_k =  (xr * ys - xs * yr) * invJ;

                    // Reference derivatives for E
                    double Ex_r = 0.0, Ex_s = 0.0, Ex_t = 0.0;
                    double Ey_r = 0.0, Ey_s = 0.0, Ey_t = 0.0;
                    double Ez_r = 0.0, Ez_s = 0.0, Ez_t = 0.0;

                    // Reference derivatives for H
                    double Hx_r = 0.0, Hx_s = 0.0, Hx_t = 0.0;
                    double Hy_r = 0.0, Hy_s = 0.0, Hy_t = 0.0;
                    double Hz_r = 0.0, Hz_s = 0.0, Hz_t = 0.0;

                    for (int l = 0; l < N; ++l) {
                        double d_il = D[i + l * N];
                        double d_jl = D[j + l * N];
                        double d_kl = D[k + l * N];

                        int idx_r = elemOffset + l + N * (j + N * k);
                        int idx_s = elemOffset + i + N * (l + N * k);
                        int idx_t = elemOffset + i + N * (j + N * l);

                        Ex_r += d_il * Ex[idx_r];
                        Ey_r += d_il * Ey[idx_r];
                        Ez_r += d_il * Ez[idx_r];

                        Hx_r += d_il * Hx[idx_r];
                        Hy_r += d_il * Hy[idx_r];
                        Hz_r += d_il * Hz[idx_r];

                        Ex_s += d_jl * Ex[idx_s];
                        Ey_s += d_jl * Ey[idx_s];
                        Ez_s += d_jl * Ez[idx_s];

                        Hx_s += d_jl * Hx[idx_s];
                        Hy_s += d_jl * Hy[idx_s];
                        Hz_s += d_jl * Hz[idx_s];

                        Ex_t += d_kl * Ex[idx_t];
                        Ey_t += d_kl * Ey[idx_t];
                        Ez_t += d_kl * Ez[idx_t];

                        Hx_t += d_kl * Hx[idx_t];
                        Hy_t += d_kl * Hy[idx_t];
                        Hz_t += d_kl * Hz[idx_t];
                    }

                    // Physical derivatives for E: dEx/dx, dEy/dy, dEz/dz, etc.
                    double dEx_dx = Ex_r * rx_k + Ex_s * sx_k + Ex_t * tx_k;
                    double dEx_dy = Ex_r * ry_k + Ex_s * sy_k + Ex_t * ty_k;
                    double dEx_dz = Ex_r * rz_k + Ex_s * sz_k + Ex_t * tz_k;

                    double dEy_dx = Ey_r * rx_k + Ey_s * sx_k + Ey_t * tx_k;
                    double dEy_dy = Ey_r * ry_k + Ey_s * sy_k + Ey_t * ty_k;
                    double dEy_dz = Ey_r * rz_k + Ey_s * sz_k + Ey_t * tz_k;

                    double dEz_dx = Ez_r * rx_k + Ez_s * sx_k + Ez_t * tx_k;
                    double dEz_dy = Ez_r * ry_k + Ez_s * sy_k + Ez_t * ty_k;
                    double dEz_dz = Ez_r * rz_k + Ez_s * sz_k + Ez_t * tz_k;

                    // Physical derivatives for H
                    double dHx_dx = Hx_r * rx_k + Hx_s * sx_k + Hx_t * tx_k;
                    double dHx_dy = Hx_r * ry_k + Hx_s * sy_k + Hx_t * ty_k;
                    double dHx_dz = Hx_r * rz_k + Hx_s * sz_k + Hx_t * tz_k;

                    double dHy_dx = Hy_r * rx_k + Hy_s * sx_k + Hy_t * tx_k;
                    double dHy_dy = Hy_r * ry_k + Hy_s * sy_k + Hy_t * ty_k;
                    double dHy_dz = Hy_r * rz_k + Hy_s * sz_k + Hy_t * tz_k;

                    double dHz_dx = Hz_r * rx_k + Hz_s * sx_k + Hz_t * tx_k;
                    double dHz_dy = Hz_r * ry_k + Hz_s * sy_k + Hz_t * ty_k;
                    double dHz_dz = Hz_r * rz_k + Hz_s * sz_k + Hz_t * tz_k;

                    // Divergence: div(E) = dEx/dx + dEy/dy + dEz/dz
                    divE[p] = dEx_dx + dEy_dy + dEz_dz;
                    divH[p] = dHx_dx + dHy_dy + dHz_dz;

                    // Curl: curl(E) = (dEz/dy - dEy/dz, dEx/dz - dEz/dx, dEy/dx - dEx/dy)
                    cEx[p] = dEz_dy - dEy_dz;
                    cEy[p] = dEx_dz - dEz_dx;
                    cEz[p] = dEy_dx - dEx_dy;

                    // Curl: curl(H) = (dHz/dy - dHy/dz, dHx/dz - dHz/dx, dHy/dx - dHx/dy)
                    cHx[p] = dHz_dy - dHy_dz;
                    cHy[p] = dHx_dz - dHz_dx;
                    cHz[p] = dHy_dx - dHx_dy;
                }
            }
        }
    }
}

