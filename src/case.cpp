// Three phases:
//     1) Preprocessing
//     2) Simulation
//     3) Postprocessing
// The case gets taken from the /examples folder

#include "case.hpp"
#include "comm.hpp"

#ifndef USE_HIP
#include <nvtx3/nvToolsExt.h>
#endif

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <sys/stat.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
        config_.loadFromFile(configFile);
    }
}

void Case::setConfig(const Config& config) {
    config_ = config;
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
    Comm::init();
    config_ = cfg;

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

#ifndef USE_HIP
    nvtxRangePushA("Case::loadMesh");
#endif
    // Mesh setup: Built-in box generator or external .rea mesh file
    if (config_.nelx > 0 && config_.nely > 0 && config_.nelz > 0) {
        std::cout << "[PREPROCESS] Generating Cartesian box mesh: "
                  << config_.nelx << " x " << config_.nely << " x " << config_.nelz
                  << " = " << config_.nelx * config_.nely * config_.nelz 
                  << " elements (Order N = " << config_.order << ")" << std::endl;
        std::cout << "             Domain: [" << config_.xmin << ", " << config_.xmax << "] x ["
                  << config_.ymin << ", " << config_.ymax << "] x ["
                  << config_.zmin << ", " << config_.zmax << "]" << std::endl;
        if (config_.periodicX || config_.periodicY || config_.periodicZ) {
            std::cout << "             Boundary Conditions: Periodic in ["
                      << (config_.periodicX ? "X " : "")
                      << (config_.periodicY ? "Y " : "")
                      << (config_.periodicZ ? "Z " : "")
                      << "]" << std::endl;
        }
        mesh_->createBoxMesh(config_.nelx, config_.nely, config_.nelz,
                             config_.xmin, config_.xmax,
                             config_.ymin, config_.ymax,
                             config_.zmin, config_.zmax,
                             config_.periodicX, config_.periodicY, config_.periodicZ);
    } else if (!config_.meshFile.empty() && config_.meshFile != "box") {
        std::cout << "[PREPROCESS] Loading mesh file: " << config_.meshFile 
                  << " (Order N = " << config_.order << ")" << std::endl;
        bool bOk = false;
        if (config_.meshFile.find(".re2") != std::string::npos) {
            bOk = mesh_->loadFromRe2(config_.meshFile);
        } else {
            bOk = mesh_->loadFromRea(config_.meshFile);
        }
        if (!bOk) {
            std::cerr << "Warning: Could not load mesh. Falling back to default 3x3x3 mesh." << std::endl;
            mesh_->createBoxMesh(3, 3, 3, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
        }
    } else {
        std::cout << "[PREPROCESS] Using default 3x3x3 Cartesian box mesh (27 elements, Order N = " 
                  << config_.order << ")" << std::endl;
        mesh_->createBoxMesh(3, 3, 3, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
    }

    if (config_.has("wave_type") && config_.getString("wave_type") == "3dboxper") {
        const auto& cx = mesh_->getCoordX();
        double minX = 1e30, maxX = -1e30;
        for (size_t k = 0; k < cx.size(); ++k) {
            minX = std::min(minX, cx[k]);
            maxX = std::max(maxX, cx[k]);
        }
        if (std::abs(minX - (-1.0)) < 1e-3 && std::abs(maxX - 1.0) < 1e-3) {
            std::cout << "[PREPROCESS] Rescaling mesh domain from [-1, 1]^3 to [0, 2*pi]^3 matching NekCEM usrdat2..." << std::endl;
            mesh_->rescale(0.0, 2.0 * M_PI, 0.0, 2.0 * M_PI, 0.0, 2.0 * M_PI);
        }
    }
#ifndef USE_HIP
    nvtxRangePop();
#endif

    // Partition mesh across MPI ranks if running in parallel
    if (Comm::size() > 1) {
#ifndef USE_HIP
        nvtxRangePushA("Case::partitionMesh");
#endif
        mesh_->partition(Comm::rank(), Comm::size());
#ifndef USE_HIP
        nvtxRangePop();
#endif
    }

    // Allocate the state vector with 6 variables (local points + halo ghost nodes)
    const int npts = mesh_->getTotalPoints();
    const int totalAlloc = npts + mesh_->getNumHaloPoints();
    state_.assign(6 * totalAlloc, 0.0); // Everything set to 0

    // Calculate stable time-stepping metrics (synchronized across all ranks)
    const double dxminLocal = mesh_->computeMinNodeDistance();
    const double dxmin = Comm::allreduceMin(dxminLocal);
    const double waveSpeed = 1.0;

    if (config_.dt > 0.0) {
        dt_ = config_.dt;
        cfl_ = (dt_ * waveSpeed) / dxmin;
    } else if (config_.cfl > 0.0) {
        cfl_ = config_.cfl;
        dt_ = (cfl_ * dxmin) / waveSpeed;
    } else {
        cfl_ = mesh_->computeAutomaticCFL(0.4);
        dt_ = mesh_->computeAutomaticDt(waveSpeed, 0.4);
    }

    currentTime_ = 0.0;
    currentStep_ = 0;

    // Evaluate initial conditions
    const auto& x = mesh_->getCoordX();
    const auto& y = mesh_->getCoordY();
    const auto& z = mesh_->getCoordZ();

    int actualNumModes = config_.getInt("num_modes", 1);
    std::vector<double> actualModeK;

#ifndef USE_HIP
    nvtxRangePushA("Case::initFields");
#endif
    // Field initialization!
    if (initialConditionHook_) {
        std::cout << "[PREPROCESS] Evaluating user initial condition hook..." << std::endl;
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
    } else if (config_.has("wave_type") && (config_.getString("wave_type") == "bloch" || config_.getString("wave_type") == "periodic")) {
        std::cout << "[PREPROCESS] Evaluating periodic Bloch electromagnetic plane wave in (Ez, Hx, Hy)..." << std::endl;
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
    } else if (config_.has("wave_type") && config_.getString("wave_type") == "multimode") {
        std::cout << "[PREPROCESS] Evaluating multi-harmonic sinusoidal standing wave packet in Ez..." << std::endl;
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
    } else if (config_.has("wave_type") && config_.getString("wave_type") == "packet") {
        std::cout << "[PREPROCESS] Evaluating modulated Gaussian wavepacket in Ez..." << std::endl;
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
    } else if (config_.has("wave_type") && config_.getString("wave_type") == "gaussian") {
        std::cout << "[PREPROCESS] Evaluating Gaussian pulse in Ez..." << std::endl;
        const double sigma = config_.pulseSigma;
        double* Ez = &state_[2 * npts];
        for (int k = 0; k < npts; ++k) {
            double r2 = x[k] * x[k] + y[k] * y[k] + z[k] * z[k];
            Ez[k] = std::exp(-sigma * r2);
        }
    } else {
        std::cout << "[PREPROCESS] No initial condition hook provided; all electromagnetic fields initialized to 0.0." << std::endl;
    }
#ifndef USE_HIP
    nvtxRangePop();
#endif

#ifndef USE_HIP
    nvtxRangePushA("Case::initProbes");
#endif
    // Assemble probe coordinates
    std::vector<std::array<double, 3>> allProbes = config_.probes;
    for (const auto& p : customProbes_) {
        allProbes.push_back(p);
    }

    // Convert any custom relative probes to physical coordinates
    const double xc = 0.5 * (config_.xmin + config_.xmax);
    const double yc = 0.5 * (config_.ymin + config_.ymax);
    const double zc = 0.5 * (config_.zmin + config_.zmax);
    for (const auto& r : customRelativeProbes_) {
        allProbes.push_back({{xc + r[0] * config_.Lx, yc + r[1] * config_.Ly, zc + r[2] * config_.Lz}});
    }

    // If no explicit probes configured, check for num_probes parameter
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

    probes_.init(*mesh_, allProbes, outDir, actualNumModes, config_.Lx, config_.nelx, actualModeK);
#ifndef USE_HIP
    nvtxRangePop();
#endif

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
    if (bSaveOutput && config_.exportFields) {
#ifndef USE_HIP
        nvtxRangePushA("Case::initialExport");
#endif
        if (config_.exportFormat == "hdf5" || config_.exportFormat == "vtk") {
            hdf5Writer_.reset(new Hdf5Writer());
            hdf5Writer_->setFieldSaveOptions(saveOptions_);
            hdf5Writer_->initialize(outDir + "/fields.h5", *mesh_, true, config_.exportContinuousVtk);
            bool needDerivatives = saveOptions_.saveCurlE || saveOptions_.saveCurlH ||
                                   saveOptions_.saveDivE || saveOptions_.saveDivH ||
                                   saveOptions_.saveMagnitudeCurlE || saveOptions_.saveMagnitudeCurlH;
            std::vector<double> divE, divH, curlE, curlH;
            if (needDerivatives) {
                computeFieldDerivatives(state_.data(), divE, divH, curlE, curlH);
            }
            Hdf5Writer::DerivedFields derived{
                divE.empty() ? nullptr : divE.data(),
                divH.empty() ? nullptr : divH.data(),
                curlE.empty() ? nullptr : curlE.data(),
                curlH.empty() ? nullptr : curlH.data()
            };
            hdf5Writer_->writeStep(0, 0.0, state_.data(), npts, &derived);
        } else {
            saveFields(outDir + "/field_initial.csv", 0.0);
        }
#ifndef USE_HIP
        nvtxRangePop();
#endif
    }

    // Record t = 0 on probes
    probes_.record(0, 0.0, state_, npts);

    // Evaluate initial diagnostics across all ranks
    double initEnergy = computeTotalEnergy();
    double initMaxE = getMaxE();
    double initMaxH = getMaxH();

    // Initialize continuous energy log if output is enabled
    if (Comm::isRoot() && bSaveOutput) {
        energyFile_.open(outDir + "/energy_history.csv");
        if (energyFile_.is_open()) {
            energyFile_ << "step,time,maxE,maxH,energy\n";
            energyFile_ << std::scientific << std::setprecision(8);
            energyFile_ << 0 << "," << 0.0 << "," << initMaxE << "," << initMaxH << "," << initEnergy << "\n";
            energyFile_.flush();
        }
    }

    if (Comm::isRoot()) {
        std::cout << "\n----------------------------------------------------------" << std::endl;
        std::cout << "               NekWave Solver Setup Summary               " << std::endl;
        std::cout << "----------------------------------------------------------\n" << std::endl;
        std::cout << "  Elements:          " << mesh_->getNumElements() << std::endl;
        std::cout << "  Polynomial Order:  " << mesh_->getN() << " (Np = " << mesh_->getNumPointsPerElement() << " nodes/elem)" << std::endl;
        std::cout << "  Total Collocation: " << npts << " points" << std::endl;
        std::cout << "  Domain Dimensions: Lx = " << config_.Lx << ", Ly = " << config_.Ly << ", Lz = " << config_.Lz << std::endl;
        std::cout << "  Bounding Box:      [" << config_.xmin << ", " << config_.xmax << "] x [" 
                  << config_.ymin << ", " << config_.ymax << "] x [" 
                  << config_.zmin << ", " << config_.zmax << "]" << std::endl;
        std::cout << "  Faces Total:       " << mesh_->getFaceData().size() << std::endl;
        std::cout << "  Flux Formulation:  " << ((config_.c0 == 0.0) ? "Central (C0 = 0.0, energy-conserving)" : "Upwind (C0 = 1.0, dissipative)") << std::endl;
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
        std::cout << "  Initial Energy:    " << std::scientific << std::setprecision(2) << initEnergy << std::endl;
        std::cout << "  Initial max|E|:    " << std::scientific << std::setprecision(2) << initMaxE 
                  << " | max|H|: " << std::scientific << std::setprecision(2) << initMaxH << std::endl;
        std::cout << "\n----------------------------------------------------------\n" << std::endl;
    }

    bIsPreprocessed_ = true;
}

// ----------------------
// @@ Phase 2: Simulation
// ----------------------
void Case::simulate() {
    assert(bIsPreprocessed_ && "Must call preprocess() before simulate()");

    const int npts = mesh_->getTotalPoints();
    const int maxSteps = config_.maxSteps;
    const double finalTime = config_.finalTime;
    const int outFreq = std::max(1, config_.outputFreq);
    const int saveFreq = std::max(1, config_.saveFreq);

    if (Comm::isRoot()) {
        std::cout << "[SIMULATE] Initializing full GPU data residency (DgSolver)..." << std::endl;
    }
#ifndef USE_HIP
    nvtxRangePushA("Case::initSolverState");
#endif
    dgSolver_.reset(new DgSolver());
    dgSolver_->initialize(*mesh_, config_.c0);
    dgSolver_->uploadState(state_.data(), state_.size());
#ifndef USE_HIP
    nvtxRangePop();
#endif

    if (Comm::isRoot()) {
        std::cout << "\n[SIMULATE] Advancing time integration on GPU (LSRK45)..." << std::endl;
        std::cout << std::setw(8) << "Step" 
                  << std::setw(14) << "Time" 
                  << std::setw(14) << "max|E|" 
                  << std::setw(14) << "max|H|" 
                  << std::setw(16) << "Total Energy" << std::endl;
        std::cout << "------------------------------------------------------------------" << std::endl;
    }

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

#ifndef USE_HIP
        nvtxRangePushA("Case::step");
#endif
        step++;
        dgSolver_->step(currentDt, currentTime_);
        currentTime_ += currentDt;
        currentStep_ = step;

#ifndef USE_HIP
        nvtxRangePushA("Case::downloadState");
#endif
        dgSolver_->downloadState(state_.data(), state_.size());
#ifndef USE_HIP
        nvtxRangePop();
#endif

#ifndef USE_HIP
        nvtxRangePushA("Case::probesRecord");
#endif
        probes_.record(step, currentTime_, state_, npts);
#ifndef USE_HIP
        nvtxRangePop();
#endif

        bool isFinal = (maxSteps > 0 && step >= maxSteps) || (finalTime > 0.0 && currentTime_ >= finalTime - 1e-13);

        // Save HDF5/VTK snapshots according to saveFreq or on final step
        if (hdf5Writer_ && (step % saveFreq == 0 || isFinal)) {
#ifndef USE_HIP
            nvtxRangePushA("Case::hdf5WriteStep");
#endif
            bool needDerivatives = saveOptions_.saveCurlE || saveOptions_.saveCurlH ||
                                   saveOptions_.saveDivE || saveOptions_.saveDivH ||
                                   saveOptions_.saveMagnitudeCurlE || saveOptions_.saveMagnitudeCurlH;
            std::vector<double> divE, divH, curlE, curlH;
            if (needDerivatives) {
                computeFieldDerivatives(state_.data(), divE, divH, curlE, curlH);
            }
            Hdf5Writer::DerivedFields derived{
                divE.empty() ? nullptr : divE.data(),
                divH.empty() ? nullptr : divH.data(),
                curlE.empty() ? nullptr : curlE.data(),
                curlH.empty() ? nullptr : curlH.data()
            };
            hdf5Writer_->writeStep(step, currentTime_, state_.data(), npts, &derived);
#ifndef USE_HIP
            nvtxRangePop();
#endif
        }

        // Output diagnostics according to outFreq or on final step
        if (step % outFreq == 0 || isFinal) {
#ifndef USE_HIP
            nvtxRangePushA("Case::computeDiagnostics");
#endif
            double maxE = getMaxE();
            double maxH = getMaxH();
            double energy = computeTotalEnergy();
#ifndef USE_HIP
            nvtxRangePop();
#endif

            if (Comm::isRoot()) {
                if (energyFile_.is_open()) {
                    energyFile_ << step << "," << currentTime_ << "," << maxE << "," << maxH << "," << energy << "\n";
                    energyFile_.flush();
                }

                std::cout << std::setw(8) << step 
                          << std::setw(14) << std::fixed << std::setprecision(5) << currentTime_ 
                          << std::setw(14) << std::fixed << std::setprecision(5) << maxE 
                          << std::setw(14) << std::fixed << std::setprecision(5) << maxH 
                          << std::setw(16) << std::scientific << std::setprecision(6) << energy << std::endl;
            }
        }
#ifndef USE_HIP
        nvtxRangePop();
#endif
    }

    dgSolver_->downloadState(state_.data(), state_.size());

    bIsSimulated_ = true;
}

// --------------------------
// @@ Phase 3: Postprocessing
// --------------------------
void Case::postprocess() {
    assert(bIsSimulated_ && "Must call simulate() before postprocess()");

    const std::string& outDir = config_.outputDir;
    bool bSaveOutput = (!outDir.empty() && outDir != "none" && outDir != "None" && outDir != "NONE");

    // Finalize probe observations
#ifndef USE_HIP
    nvtxRangePushA("Case::probesFinalize");
#endif
    probes_.finalize();
#ifndef USE_HIP
    nvtxRangePop();
#endif

    // Close energy history log
    if (energyFile_.is_open()) {
        energyFile_.close();
        if (Comm::isRoot()) {
            std::cout << "[POSTPROCESS] Exported energy history to " << outDir << "/energy_history.csv" << std::endl;
        }
    }

    // Finalize HDF5 time series writer if active
    if (hdf5Writer_) {
#ifndef USE_HIP
        nvtxRangePushA("Case::hdf5Close");
#endif
        hdf5Writer_->close();
#ifndef USE_HIP
        nvtxRangePop();
#endif
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

    if (dgSolver_) {
        dgSolver_->finalize();
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

void Case::saveFields(const std::string& filename, double time) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for field export." << std::endl;
        return;
    }

    const auto& x = mesh_->getCoordX();
    const auto& y = mesh_->getCoordY();
    const auto& z = mesh_->getCoordZ();
    const int npts = mesh_->getTotalPoints();

    const double* Ex = &state_[0 * npts];
    const double* Ey = &state_[1 * npts];
    const double* Ez = &state_[2 * npts];
    const double* Hx = &state_[3 * npts];
    const double* Hy = &state_[4 * npts];
    const double* Hz = &state_[5 * npts];

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
    const auto& rx = mesh_->getRx();
    const auto& sx = mesh_->getSx();
    const auto& tx = mesh_->getTx();
    const auto& ry = mesh_->getRy();
    const auto& sy = mesh_->getSy();
    const auto& ty = mesh_->getTy();
    const auto& rz = mesh_->getRz();
    const auto& sz = mesh_->getSz();
    const auto& tz = mesh_->getTz();

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
        for (int k = 0; k < N; ++k) {
            for (int j = 0; j < N; ++j) {
                for (int i = 0; i < N; ++i) {
                    int m = i + N * (j + N * k);
                    int p = elemOffset + m;

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

                    double rx_k = rx[p], sx_k = sx[p], tx_k = tx[p];
                    double ry_k = ry[p], sy_k = sy[p], ty_k = ty[p];
                    double rz_k = rz[p], sz_k = sz[p], tz_k = tz[p];

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

