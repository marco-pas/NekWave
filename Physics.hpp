#pragma once
#include <vector>
#include "Mesh.hpp"

// @@ here im defining the State vector type alias for clarity
using StateVector = std::vector<double>;

// @@ here im defining the Physics class to evaluate spatial terms (RHS) for Maxwell's equations
class Physics {
public:
    // @@ here im initializing the Physics component
    Physics();
    // @@ here im cleaning up Physics resources
    ~Physics();

    // @@ here im declaring the fully explicit DG spatial operator evaluation for Maxwell
    void evaluateRHS(const Mesh& mesh, const StateVector& state, StateVector& rhs, double rkTime);

    // @@ here im declaring the basic matrix-matrix multiplication routine for tensor contractions
    static void mxm(const double* A, int n1, const double* B, int n2, double* C, int n3);

    // @@ here im declaring the local element differentiation function local_grad3
    static void local_grad3(const double* u, int N, const std::vector<double>& D, const std::vector<double>& Dt,
                           double* ur, double* us, double* ut);

    // @@ here im declaring the routine to compute weighted curl of a 3D vector field
    void compute_weighted_curl(const Mesh& mesh,
                              const double* u1, const double* u2, const double* u3,
                              double* w1, double* w2, double* w3);

private:
    // @@ here im declaring the method to compute the curl of the fields
    void computeCurl(const Mesh& mesh, const StateVector& state, StateVector& rhs);
    
    // @@ here im declaring the method to restrict fields to faces for flux computation
    void restrictToFace(const Mesh& mesh, const StateVector& state);
    
    // @@ here im declaring the method to compute the numerical flux
    void computeFlux(const Mesh& mesh, const StateVector& state, StateVector& flux);
    
    // @@ here im declaring the method to add the computed flux to the RHS
    void addFluxToRHS(const Mesh& mesh, const StateVector& flux, StateVector& rhs);
    
    // @@ here im declaring the method to apply the inverse mass matrix
    void applyInverseMassMatrix(const Mesh& mesh, StateVector& rhs);
};
