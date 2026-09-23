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

    // User customization hooks
    void setInitialCondition(InitialConditionFn fn);
    void setPostprocessingHook(PostprocessingFn fn);
    void addProbe(double x, double y, double z);
    void addRelativeProbe(double rx, double ry, double rz);

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

private:
    Config config_;

    std::unique_ptr<Mesh> mesh_;
    std::unique_ptr<DgSolver> dgSolver_;
    ProbeManager probes_;
    StateVector state_;

    double dt_;
    double cfl_;
    double currentTime_;
    int currentStep_;

    std::ofstream energyFile_;

    InitialConditionFn initialConditionHook_;
    PostprocessingFn postprocessingHook_;
    std::vector<std::array<double, 3>> customProbes_;
    std::vector<std::array<double, 3>> customRelativeProbes_;

    bool bIsPreprocessed_;
    bool bIsSimulated_;
    bool bIsPostprocessed_;
};

#endif // NW_SRC_CASE_HPP
