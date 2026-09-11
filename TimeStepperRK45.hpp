#pragma once
#include "Physics.hpp"
#include "Mesh.hpp"
#include <vector>

// @@ here im defining the RK45 time stepper class using a low-storage formulation
class TimeStepperRK45 {
public:
    // @@ here im initializing the time stepper and its coefficients
    TimeStepperRK45();
    // @@ here im cleaning up time stepper resources
    ~TimeStepperRK45();

    // @@ here im declaring the step method to advance the solution in time explicitly
    void step(Mesh& mesh, Physics& physics, StateVector& state, double dt, double time);

private:
    // @@ here im declaring the arrays for LSRK45 coefficients
    std::vector<double> m_rk4a;
    std::vector<double> m_rk4b;
    std::vector<double> m_rk4c;

    // @@ here im declaring the auxiliary state vector 'k' for the low-storage RK
    StateVector m_k;
};
