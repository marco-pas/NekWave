// Three phases:
//     1) Preprocessing
//     2) Simulation
//     3) Postprocessing
// The case gets taken from the /examples folder

#include "case.hpp"

#ifdef NEKWAVE_ENABLE_CUDA
#include "cuda/gpu_solver.hpp"
#endif

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <sys/stat.h>

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
    physics_ = std::unique_ptr<Physics>(new Physics());
    timeStepper_ = std::unique_ptr<TimeStepperRK45>(new TimeStepperRK45());
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
    config_ = cfg;

    const std::string& outDir = config_.outputDir;
    mkdir(outDir.c_str(), 0755);

    mesh_->setN(config_.order);

    // Mesh setup: Built-in box generator or external .rea mesh file
    if (config_.nelx > 0 && config_.nely > 0 && config_.nelz > 0) {
        std::cout << "[PREPROCESS] Generating Cartesian box mesh: "
                  << config_.nelx << " x " << config_.nely << " x " << config_.nelz
                  << " = " << config_.nelx * config_.nely * config_.nelz 
                  << " elements (Order N = " << config_.order << ")" << std::endl;
        std::cout << "             Domain: [" << config_.xmin << ", " << config_.xmax << "] x ["
                  << config_.ymin << ", " << config_.ymax << "] x ["
                  << config_.zmin << ", " << config_.zmax << "]" << std::endl;
        mesh_->createBoxMesh(config_.nelx, config_.nely, config_.nelz,
                             config_.xmin, config_.xmax,
                             config_.ymin, config_.ymax,
                             config_.zmin, config_.zmax);
    } else if (!config_.meshFile.empty() && config_.meshFile != "box") {
        std::cout << "[PREPROCESS] Loading mesh file: " << config_.meshFile 
                  << " (Order N = " << config_.order << ")" << std::endl;
        bool bOk = mesh_->loadFromRea(config_.meshFile);
        if (!bOk) {
            std::cerr << "Warning: Could not load mesh. Falling back to default 3x3x3 mesh." << std::endl;
            mesh_->createBoxMesh(3, 3, 3, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
        }
    } else {
        std::cout << "[PREPROCESS] Using default 3x3x3 Cartesian box mesh (27 elements, Order N = " 
                  << config_.order << ")" << std::endl;
        mesh_->createBoxMesh(3, 3, 3, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
    }

    const int npts = mesh_->getTotalPoints();
    state_.assign(6 * npts, 0.0);

    // Calculate stable time-stepping metrics
    const double dxmin = mesh_->computeMinNodeDistance();
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
    } else {
        std::cout << "[PREPROCESS] Evaluating default Gaussian pulse in Ez..." << std::endl;
        const double sigma = config_.pulseSigma;
        double* Ez = &state_[2 * npts];
        for (int k = 0; k < npts; ++k) {
            double r2 = x[k] * x[k] + y[k] * y[k] + z[k] * z[k];
            Ez[k] = std::exp(-sigma * r2);
        }
    }

    // Configure numerical flux parameter C0
    physics_->setC0(config_.c0);

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

    probes_.init(*mesh_, allProbes, outDir);

    // Export initial fields (t = 0)
    saveFields(outDir + "/field_initial.csv", 0.0);

    // Record t = 0 on probes
    probes_.record(0, 0.0, state_, npts);

    // Initialize continuous energy log
    energyFile_.open(outDir + "/energy_history.csv");
    if (energyFile_.is_open()) {
        energyFile_ << "step,time,maxE,maxH,energy\n";
        energyFile_ << std::scientific << std::setprecision(8);
        energyFile_ << 0 << "," << 0.0 << "," << getMaxE() << "," << getMaxH() << "," << computeTotalEnergy() << "\n";
        energyFile_.flush();
    }

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
    std::cout << "  Total Steps:       " << config_.numSteps << " (Final Time: " << std::scientific << std::setprecision(2) << config_.numSteps * dt_ << ")" << std::endl;
    std::cout << "  Initial Energy:    " << std::scientific << std::setprecision(2) << computeTotalEnergy() << std::endl;
    std::cout << "  Initial max|E|:    " << std::scientific << std::setprecision(2) << getMaxE() 
              << " | max|H|: " << std::scientific << std::setprecision(2) << getMaxH() << std::endl;
    std::cout << "\n----------------------------------------------------------\n" << std::endl;

    bIsPreprocessed_ = true;
}

// ----------------------
// @@ Phase 2: Simulation
// ----------------------
void Case::simulate() {
    assert(bIsPreprocessed_ && "Must call preprocess() before simulate()");

    const int npts = mesh_->getTotalPoints();
    const int numSteps = config_.numSteps;
    const int freq = std::max(1, config_.outputFreq);

#ifdef NEKWAVE_ENABLE_CUDA
    std::cout << "[SIMULATE] Initializing full GPU data residency (GpuSolver)..." << std::endl;
    gpuSolver_.reset(new GpuSolver());
    gpuSolver_->initialize(*mesh_, config_.c0);
    gpuSolver_->uploadState(state_.data(), state_.size());
#endif

    std::cout << "\n[SIMULATE] Advancing time integration (LSRK45)..." << std::endl;
    std::cout << std::setw(8) << "Step" 
              << std::setw(14) << "Time" 
              << std::setw(14) << "max|E|" 
              << std::setw(14) << "max|H|" 
              << std::setw(16) << "Total Energy" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;

    for (int step = 1; step <= numSteps; ++step) {
#ifdef NEKWAVE_ENABLE_CUDA
        gpuSolver_->step(dt_, currentTime_);
        currentTime_ += dt_;
        currentStep_ = step;

        gpuSolver_->downloadState(state_.data(), state_.size());
#else
        timeStepper_->step(*mesh_, *physics_, state_, dt_, currentTime_);
        currentTime_ += dt_;
        currentStep_ = step;
#endif

        probes_.record(step, currentTime_, state_, npts);

        if (step % freq == 0 || step == numSteps) {
            double maxE = getMaxE();
            double maxH = getMaxH();
            double energy = computeTotalEnergy();

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

#ifdef NEKWAVE_ENABLE_CUDA
    gpuSolver_->downloadState(state_.data(), state_.size());
#endif

    bIsSimulated_ = true;
}

// --------------------------
// @@ Phase 3: Postprocessing
// --------------------------
void Case::postprocess() {
    assert(bIsSimulated_ && "Must call simulate() before postprocess()");

    const std::string& outDir = config_.outputDir;

    // Finalize probe observations
    probes_.finalize();

    // Close energy history log
    if (energyFile_.is_open()) {
        energyFile_.close();
        std::cout << "[POSTPROCESS] Exported energy history to " << outDir << "/energy_history.csv" << std::endl;
    }

    // Export final field distributions
    saveFields(outDir + "/field_final.csv", currentTime_);

    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "Simulation completed successfully at t = " << std::scientific << std::setprecision(6) << currentTime_ << std::endl;

    // Execute user-defined postprocessing hook if registered
    if (postprocessingHook_) {
        std::cout << "[POSTPROCESS] Executing custom user postprocessing hook..." << std::endl;
        postprocessingHook_(*this);
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

    return energy;
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
    return maxVal;
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
    return maxVal;
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
