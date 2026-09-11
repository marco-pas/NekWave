#pragma once
#include <vector>

// @@ here im defining the Mesh class which handles the spatial domain and geometry
class Mesh {
public:
    // @@ here im initializing the mesh with default polynomial order N and element count
    Mesh(int N = 3, int numElements = 1);
    
    // @@ here im cleaning up mesh resources
    ~Mesh();
    
    // @@ here im initializing the mesh geometry, operators, and metrics
    void initialize();

    // @@ here im returning the number of 1D points N
    int getN() const { return m_N; }

    // @@ here im returning the number of points per element N_p = N * N * N
    int getNumPointsPerElement() const { return m_numPointsPerElement; }

    // @@ here im returning the number of elements in the mesh
    int getNumElements() const { return m_numElements; }

    // @@ here im returning the total number of points across all elements
    int getTotalPoints() const { return m_totalPoints; }

    // @@ here im returning the 1D derivative matrix D
    const std::vector<double>& getD() const { return m_D; }

    // @@ here im returning the transpose of 1D derivative matrix Dt
    const std::vector<double>& getDt() const { return m_Dt; }

    // @@ here im returning the 3D quadrature weights w3
    const std::vector<double>& getW3() const { return m_w3; }

    // @@ here im returning geometric metric factors
    const std::vector<double>& getRx() const { return m_rx; }
    const std::vector<double>& getSx() const { return m_sx; }
    const std::vector<double>& getTx() const { return m_tx; }
    const std::vector<double>& getRy() const { return m_ry; }
    const std::vector<double>& getSy() const { return m_sy; }
    const std::vector<double>& getTy() const { return m_ty; }
    const std::vector<double>& getRz() const { return m_rz; }
    const std::vector<double>& getSz() const { return m_sz; }
    const std::vector<double>& getTz() const { return m_tz; }
    const std::vector<double>& getJac() const { return m_jac; }

private:
    // @@ here im storing polynomial degree parameters
    int m_N;
    int m_numPointsPerElement;
    int m_numElements;
    int m_totalPoints;

    // @@ here im storing 1D reference coordinates and weights
    std::vector<double> m_z;
    std::vector<double> m_w;

    // @@ here im storing 1D differentiation matrices D and Dt
    std::vector<double> m_D;
    std::vector<double> m_Dt;

    // @@ here im storing 3D element quadrature weights w3
    std::vector<double> m_w3;

    // @@ here im storing geometric metric factor arrays of length m_totalPoints
    std::vector<double> m_rx;
    std::vector<double> m_sx;
    std::vector<double> m_tx;
    std::vector<double> m_ry;
    std::vector<double> m_sy;
    std::vector<double> m_ty;
    std::vector<double> m_rz;
    std::vector<double> m_sz;
    std::vector<double> m_tz;
    std::vector<double> m_jac;

    // @@ here im declaring a helper to setup 1D GLL points, weights, and derivative matrix D
    void setupGLL();
};
