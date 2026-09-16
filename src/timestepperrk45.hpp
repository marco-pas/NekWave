#ifndef NW_SRC_TIMESTEPPERRK45_HPP
#define NW_SRC_TIMESTEPPERRK45_HPP

#include "physics.hpp"
#include "mesh.hpp"
#include <vector>

// @@ def RK45 time stepper class using a low-storage formulation
class TimeStepperRK45 {
public:
    // @@ init time stepper and coefficients
    TimeStepperRK45();
    // @@ clean up time stepper resources
    ~TimeStepperRK45();

    // @@ declare the step method to advance the solution in time explicitly
    void step(Mesh& mesh, Physics& physics, StateVector& state, double dt, double time);

private:
    // @@ declare the arrays for LSRK45 coefficients
    std::vector<double> m_rk4a;
    std::vector<double> m_rk4b;
    std::vector<double> m_rk4c;

    // @@ declare the auxiliary state vector 'k' for the low-storage RK
    StateVector m_k;
};

#endif // NW_SRC_TIMESTEPPERRK45_HPP
