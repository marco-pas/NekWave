#include "TimeStepperRK45.hpp"

// @@ here im implementing the TimeStepperRK45 constructor and setting up the LSRK45 coefficients
TimeStepperRK45::TimeStepperRK45() {
    // @@ here im allocating and initializing the RK45 coefficients based on NekCEM's rk_storage
    m_rk4a = {0.0,
              -567301805773.0 / 1357537059087.0,
              -2404267990393.0 / 2016746695238.0,
              -3550918686646.0 / 2091501179385.0,
              -1275806237668.0 / 842570457699.0};

    m_rk4b = {1432997174477.0 / 9575080441755.0,
              5161836677717.0 / 13612068292357.0,
              1720146321549.0 / 2090206949498.0,
              3134564353537.0 / 4481467310338.0,
              2277821191437.0 / 14882151754819.0};

    m_rk4c = {0.0,
              1432997174477.0 / 9575080441755.0,
              2526269341429.0 / 6820363962896.0,
              2006345519317.0 / 3224310063776.0,
              2802321613138.0 / 2924317926251.0,
              1.0};
}

// @@ here im implementing the TimeStepperRK45 destructor
TimeStepperRK45::~TimeStepperRK45() = default;

// @@ here im implementing the 5-stage Low-Storage RK45 step method
void TimeStepperRK45::step(Mesh& mesh, Physics& physics, StateVector& state, double dt, double time) {
    // @@ here im resizing the auxiliary vector if it hasn't been initialized to match the state
    if (m_k.size() != state.size()) {
        m_k.assign(state.size(), 0.0);
    }

    StateVector rhs(state.size(), 0.0);

    // @@ here im looping over the 5 stages of the RK45 method
    for (int stage = 0; stage < 5; ++stage) {
        // @@ here im computing the current sub-step time
        double rkTime = time + dt * m_rk4c[stage];
        
        // @@ here im evaluating the spatial operator (RHS) for the Maxwell equations
        physics.evaluateRHS(mesh, state, rhs, rkTime);

        // @@ here im updating the intermediate vector 'k' and the main 'state' using LSRK45 formula
        for (size_t i = 0; i < state.size(); ++i) {
            m_k[i] = m_rk4a[stage] * m_k[i] + dt * rhs[i];
            state[i] = state[i] + m_rk4b[stage] * m_k[i];
        }
    }
}
