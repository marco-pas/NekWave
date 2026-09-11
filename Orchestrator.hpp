#pragma once
#include "Mesh.hpp"
#include "Physics.hpp"
#include "TimeStepperRK45.hpp"
#include <memory>

// @@ here im defining the main Orchestrator class to coordinate the DG solver
class Orchestrator {
public:
    // @@ here im initializing the Orchestrator and its components
    Orchestrator();
    // @@ here im cleaning up the Orchestrator
    ~Orchestrator();

    // @@ here im declaring the method to setup the simulation
    void setup();
    
    // @@ here im declaring the main execution loop
    void run();

private:
    // @@ here im holding the mesh component
    std::unique_ptr<Mesh> m_mesh;
    
    // @@ here im holding the physics component
    std::unique_ptr<Physics> m_physics;
    
    // @@ here im holding the time stepper component
    std::unique_ptr<TimeStepperRK45> m_timeStepper;

    // @@ here im storing the global state vector of the simulation
    StateVector m_state;
};

