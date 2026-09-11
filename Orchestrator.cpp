#include "Orchestrator.hpp"
#include <iostream>

// @@ here im implementing the constructor to allocate components using C++11 syntax
Orchestrator::Orchestrator() {
    m_mesh = std::unique_ptr<Mesh>(new Mesh(3, 1));
    m_physics = std::unique_ptr<Physics>(new Physics());
    m_timeStepper = std::unique_ptr<TimeStepperRK45>(new TimeStepperRK45());
}

// @@ here im implementing the destructor
Orchestrator::~Orchestrator() = default;

// @@ here im implementing the setup phase for the solver
void Orchestrator::setup() {
    // @@ here im initializing the mesh geometry and connectivity
    m_mesh->initialize();
    
    // @@ here im resizing the state vector based on the mesh size. 
    // In 3D Maxwell, this holds Ex, Ey, Ez, Hx, Hy, Hz for all degrees of freedom.
    m_state.resize(6 * m_mesh->getTotalPoints(), 0.0);
}

// @@ here im implementing the main simulation loop
void Orchestrator::run() {
    // @@ here im setting a placeholder time step size
    double dt = 0.01;
    // @@ here im setting a placeholder number of steps
    int numSteps = 100;
    // @@ here im keeping track of the simulation time
    double time = 0.0;

    std::cout << "Starting NekWave Maxwell Solver..." << std::endl;

    // @@ here im looping over the specified number of time steps
    for (int step = 0; step < numSteps; ++step) {
        // @@ here im calling the time stepper to advance the solution by one full dt using the LSRK45 method
        m_timeStepper->step(*m_mesh, *m_physics, m_state, dt, time);
        
        // @@ here im updating the global simulation time
        time += dt;

        if ((step + 1) % 10 == 0) {
            std::cout << "Step: " << (step + 1) << " | Time: " << time << std::endl;
        }
    }
    
    std::cout << "Simulation Finished." << std::endl;
}
