#ifndef NW_SRC_PHYSICS_HPP
#define NW_SRC_PHYSICS_HPP

#include <vector>
#include "mesh.hpp"

// @@ defining the State vector type alias for clarity
using StateVector = std::vector<double>;

// @@ defining the Physics class to evaluate spatial terms (RHS) for Maxwell's equations
// Spatial Operator Pipeline (RHS = M^{-1} [ K * u + F^* ])
class Physics {
public:
    // @@ initializing the Physics component
    Physics();
    // @@ cleaning up Physics resources
    ~Physics();

    // @@ declaring the fully explicit DG spatial operator evaluation for Maxwell
    void evaluateRHS(const Mesh& mesh, const StateVector& state, StateVector& rhs, double rkTime);

    // @@ declaring the basic matrix-matrix multiplication routine for tensor contractions
    static void mxm(const double* A, int n1, const double* B, int n2, double* C, int n3);

    // @@ declaring the local element differentiation function local_grad3
    // Use tensor-product contractions:
    //      u_r = (I (x) I (x) D) u,  u_s = (I (x) D (x) I) u,  u_t = (D (x) I (x) I) u
    static void local_grad3(const double* u, int N, const std::vector<double>& D, const std::vector<double>& Dt,
                           double* ur, double* us, double* ut);

    // @@ declaring the routine to compute weighted curl of a 3D vector field
    // PIOLA TRANSFORM IN VOLUME:
    // Reference derivatives are converted to physical curl via metric terms (cofactor matrix of J_aff)
    void compute_weighted_curl(const Mesh& mesh,
                              const double* u1, const double* u2, const double* u3,
                              double* w1, double* w2, double* w3);

    // @@ setting the numerical flux penalty parameter C0 (1.0 for Upwind, 0.0 for Central)
    // C0 = 1.0 gives upwind flux, C0 = 0.0 gives energy-conserving central flux
    void setC0(double c0) { m_c0 = c0; }
    double getC0() const { return m_c0; }

private:
    // @@ storing the numerical flux penalty parameter C0
    double m_c0;

    // @@ storing face traces of electric and magnetic fields matching NekCEM's fEN and fHN
    std::vector<double> m_fEN;
    std::vector<double> m_fHN;

    // @@ storing the computed surface numerical fluxes matching NekCEM's srflx
    // Surface fluxes srflx (0..2 for H-residual, 3..5 for E-residual)
    std::vector<double> m_srflx;

    // @@ declaring the method to compute the curl of the fields
    // curl(H) and -curl(E) evaluations
    void computeCurl(const Mesh& mesh, const StateVector& state, StateVector& rhs);
    
    // @@ declaring the method to restrict fields to faces for flux computation
    // Face restriction corresponding to cem_maxwell_restrict_to_face
    void restrictToFace(const Mesh& mesh, const StateVector& state);
    
    // @@ declaring the method to compute the numerical flux
    // Upwind/Central numerical flux corresponding to cem_maxwell_flux
    // PIOLA TRANSFORM IN FLUX: Nanson's relation n * dA = J * J^{-T} * n_ref * dA_ref
    void computeFlux(const Mesh& mesh, const StateVector& state, StateVector& flux);
    
    // @@ declaring the method to add the computed flux to the RHS
    // Surface flux lifting corresponding to cem_maxwell_add_flux_to_res
    void addFluxToRHS(const Mesh& mesh, const StateVector& flux, StateVector& rhs);
    
    // @@ declaring the method to apply the inverse mass matrix
    // Mass matrix inversion corresponding to cem_maxwell_invqmass
    void applyInverseMassMatrix(const Mesh& mesh, StateVector& rhs);
};

#endif // NW_SRC_PHYSICS_HPP
