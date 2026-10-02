#ifndef NW_SRC_CASE_HPP
#define NW_SRC_CASE_HPP

#include "mesh.hpp"
#include "dg_solver.hpp"
#include "config.hpp"
#include "probe.hpp"

#include <memory>
#include <string>
#include <vector>
#include <array>
#include <functional>
#include <fstream>

/**
 * @brief User-defined initial condition callback signature.
 *
 * Evaluated at each collocation point (x, y, z) during the preprocessing phase.
 */
using InitialConditionFn = std::function<void(double x, double y, double z,
                                              double& Ex, double& Ey, double& Ez,
                                              double& Hx, double& Hy, double& Hz)>;

#include "hdf5_writer.hpp"

class Case;

/**
 * @brief User-defined postprocessing callback signature.
 *
 * Invoked during the postprocessing phase after time integration concludes.
 */
using PostprocessingFn = std::function<void(Case& c)>;

/**
 * @brief Primary simulation coordinator defining a modular 3-phase lifecycle.
 *
 * Conforming to Neko's case architecture (case.f90), every simulation
 * separates cleanly into three explicit phases:
 *
 *   1. preprocess(): Mesh loading, quadrature metrics, state allocation,
 *                    initial conditions, and observation probe registration.
 *   2. simulate():   LSRK45 GPU time-stepping, continuous energy tracking,
 *                    and probe time-series recording.
 *   3. postprocess(): Final field export, file closure, and user postprocessing.
 */
class Case {
public:
    Case();
    explicit Case(const std::string& configFile);
    ~Case();

    // Configuration
    void loadConfig(const std::string& configFile);
    void setConfig(const Config& config);
    const Config& config() const { return config_; }
    Config& config() { return config_; }

    // Wave type (must be specified programmatically in C++ code)
    void setWaveType(const std::string& waveType);
    const std::string& waveType() const;

    // User customization hooks
    void setInitialCondition(InitialConditionFn fn);
    void setInitialConditionHook(InitialConditionFn fn) { setInitialCondition(fn); }
    void setPostprocessingHook(PostprocessingFn fn);
    void addProbe(double x, double y, double z);
    void addRelativeProbe(double rx, double ry, double rz);

    // Field save customization
    using SaveOptions = Hdf5Writer::FieldSaveOptions;
    void setSaveOptions(const SaveOptions& opts) { saveOptions_ = opts; }
    const SaveOptions& saveOptions() const { return saveOptions_; }
    SaveOptions& saveOptions() { return saveOptions_; }

    // 3-Phase Simulation Lifecycle
    void preprocess();
    void preprocess(const Config& cfg);
    void simulate();
    void postprocess();

    // Full pipeline execution
    void run();

    // Field and energy diagnostics
    double computeTotalEnergy() const;
    double getMaxE() const;
    double getMaxH() const;
    void saveFields(const std::string& filename, double time) const;

    // Component accessors
    const Mesh& mesh() const { return *mesh_; }
    Mesh& mesh() { return *mesh_; }

    const DgSolver& solver() const { return *dgSolver_; }
    DgSolver& solver() { return *dgSolver_; }
    const DgSolver& dgSolver() const { return *dgSolver_; }
    DgSolver& dgSolver() { return *dgSolver_; }

    const StateVector& state() const { return state_; }
    StateVector& state() { return state_; }

    double dt() const { return dt_; }
    double cfl() const { return cfl_; }
    double currentTime() const { return currentTime_; }
    int currentStep() const { return currentStep_; }
    int numSteps() const { return config_.numSteps; }
    int maxSteps() const { return config_.maxSteps; }
    double finalTime() const { return config_.finalTime; }
    int outputFreq() const { return config_.outputFreq; }
    int saveFreq() const { return config_.saveFreq; }
    bool exportContinuousVtk() const { return config_.exportContinuousVtk; }

    /**
     * @brief Computes exact DG spectral spatial derivatives (divergence and curl) on GLL nodes.
     *
     * Evaluates div(E), div(H) and curl(E), curl(H) using element differentiation matrix D
     * and metric transformation factors without inter-element finite differencing.
     */
    void computeFieldDerivatives(const double* state,
                                 std::vector<double>& divE,
                                 std::vector<double>& divH,
                                 std::vector<double>& curlE,
                                 std::vector<double>& curlH) const;

private:
    Config config_;

    std::unique_ptr<Mesh> mesh_;
    std::unique_ptr<DgSolver> dgSolver_;
    std::unique_ptr<Hdf5Writer> hdf5Writer_;
    ProbeManager probes_;
    StateVector state_;

    double dt_;
    double cfl_;
    double currentTime_;
    int currentStep_;

    std::ofstream energyFile_;

    std::string waveType_;
    InitialConditionFn initialConditionHook_;
    PostprocessingFn postprocessingHook_;
    std::vector<std::array<double, 3>> customProbes_;
    std::vector<std::array<double, 3>> customRelativeProbes_;

    bool bIsPreprocessed_;
    bool bIsSimulated_;
    bool bIsPostprocessed_;
    SaveOptions saveOptions_;
};

#endif // NW_SRC_CASE_HPP
