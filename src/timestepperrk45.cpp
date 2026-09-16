#include "timestepperrk45.hpp"

// @@ implementing the TimeStepperRK45 constructor and setting up the LSRK45 coefficients
// init Low-Storage Runge-Kutta 4th-Order 5-Stage Scheme (LSRK45)
// from "Carpenter, Kennedy (1994)" (rk_storage in NekCEM cem_common.F)
TimeStepperRK45::TimeStepperRK45() {
    // @@ allocating and initializing the RK45 coefficients based on NekCEM's rk_storage
    // %% 5-stage low-storage a-coefficients
    m_rk4a = {0.0,
              -567301805773.0 / 1357537059087.0,
              -2404267990393.0 / 2016746695238.0,
              -3550918686646.0 / 2091501179385.0,
              -1275806237668.0 / 842570457699.0};

    // %% 5-stage low-storage b-coefficients
    m_rk4b = {1432997174477.0 / 9575080441755.0,
              5161836677717.0 / 13612068292357.0,
              1720146321549.0 / 2090206949498.0,
              3134564353537.0 / 4481467310338.0,
              2277821191437.0 / 14882151754819.0};

    // %% 5-stage low-storage c-coefficients (time offsets)
    m_rk4c = {0.0,
              1432997174477.0 / 9575080441755.0,
              2526269341429.0 / 6820363962896.0,
              2006345519317.0 / 3224310063776.0,
              2802321613138.0 / 2924317926251.0,
              1.0};
}

// @@ TimeStepperRK45 destructor
TimeStepperRK45::~TimeStepperRK45() = default;

// @@ 5-stage Low-Storage RK45 step method (!)
// For each stage s in [1, 5]:
//      1) Compute sub-stage time: t_s = t + c_s * dt (rk_c)
//      2) Evaluate spatial operator: rhs = evaluateRHS(mesh, state, t_s) (cem_maxwell_op)
//      3) Low-storage state update (rk4_upd):
//              k = a_s * k + dt * rhs
//              state = state + b_s * k
void TimeStepperRK45::step(Mesh& mesh, Physics& physics, StateVector& state, double dt, double time) {
    // @@ resizing the auxiliary vector if it hasn't been initialized to match the state
    // vector 'k' provides low memory footprint
    if (m_k.size() != state.size()) {
        m_k.assign(state.size(), 0.0);
    }

    StateVector rhs(state.size(), 0.0);

    // @@ looping over the 5 stages of the RK45 method (cem_maxwell_op_rk)
    for (int stage = 0; stage < 5; ++stage) {
        // @@ computing the current sub-step time (cem_common.F)
        double rkTime = time + dt * m_rk4c[stage];
        
        // @@ evaluating the spatial operator (RHS) for Maxwell (cem_maxwell_op)
        physics.evaluateRHS(mesh, state, rhs, rkTime);

        // @@ updating vector 'k' and main 'state' using LSRK45 formula (cem_common.F)
        for (size_t i = 0; i < state.size(); ++i) {
            m_k[i] = m_rk4a[stage] * m_k[i] + dt * rhs[i];
            state[i] = state[i] + m_rk4b[stage] * m_k[i];
        }
    }
}
