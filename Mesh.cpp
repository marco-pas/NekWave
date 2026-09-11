#include "Mesh.hpp"
#include <cmath>

// @@ here im defining a helper to evaluate the Legendre polynomial of degree n at x
static double legendrePoly(int n, double x) {
    if (n == 0) return 1.0;
    if (n == 1) return x;
    double p0 = 1.0;
    double p1 = x;
    double p2 = 0.0;
    for (int k = 1; k < n; ++k) {
        p2 = ((2.0 * k + 1.0) * x * p1 - k * p0) / (k + 1.0);
        p0 = p1;
        p1 = p2;
    }
    return p2;
}

// @@ here im defining a helper to evaluate the derivative of Legendre polynomial of degree n at x
static double legendrePolyDeriv(int n, double x) {
    if (n == 0) return 0.0;
    if (n == 1) return 1.0;
    return (n / (x * x - 1.0)) * (x * legendrePoly(n, x) - legendrePoly(n - 1, x));
}

// @@ here im implementing the Mesh constructor
Mesh::Mesh(int N, int numElements)
    : m_N(N),
      m_numElements(numElements) {
    m_numPointsPerElement = m_N * m_N * m_N;
    m_totalPoints = m_numElements * m_numPointsPerElement;
}

// @@ here im implementing the Mesh destructor
Mesh::~Mesh() = default;

// @@ here im implementing the GLL collocation points, weights, and derivative matrix setup
void Mesh::setupGLL() {
    int N = m_N;
    int p = N - 1; // polynomial degree

    m_z.resize(N);
    m_w.resize(N);
    m_D.assign(N * N, 0.0);
    m_Dt.assign(N * N, 0.0);

    // @@ here im computing 1D Gauss-Lobatto-Legendre (GLL) points and quadrature weights
    if (N == 1) {
        m_z[0] = 0.0;
        m_w[0] = 2.0;
        m_D[0] = 0.0;
        m_Dt[0] = 0.0;
    } else {
        m_z[0] = -1.0;
        m_z[N - 1] = 1.0;

        // @@ here im computing interior GLL roots via Newton-Raphson iteration
        for (int i = 1; i < N - 1; ++i) {
            // initial Chebyshev-Gauss-Lobatto estimate
            double xi = -std::cos((M_PI * i) / p);
            for (int it = 0; it < 20; ++it) {
                double L = legendrePoly(p, xi);
                double Ld = legendrePolyDeriv(p, xi);
                // second derivative from Legendre differential equation
                double Ldd = (2.0 * xi * Ld - p * (p + 1.0) * L) / (1.0 - xi * xi);
                double delta = Ld / Ldd;
                xi -= delta;
                if (std::abs(delta) < 1e-15) break;
            }
            m_z[i] = xi;
        }

        // @@ here im computing the 1D GLL integration weights
        for (int i = 0; i < N; ++i) {
            double L = legendrePoly(p, m_z[i]);
            m_w[i] = 2.0 / (p * (p + 1.0) * L * L);
        }

        // @@ here im constructing the 1D derivative matrix D and its transpose Dt matching NekCEM's DGLL
        double d0 = p * (p + 1.0) / 4.0;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                int idx = i + j * N; // column-major: row i, column j
                if (i != j) {
                    double Li = legendrePoly(p, m_z[i]);
                    double Lj = legendrePoly(p, m_z[j]);
                    m_D[idx] = Li / (Lj * (m_z[i] - m_z[j]));
                } else if (i == 0 && j == 0) {
                    m_D[idx] = -d0;
                } else if (i == p && j == p) {
                    m_D[idx] = d0;
                } else {
                    m_D[idx] = 0.0;
                }
                // @@ here im recording the transpose Dt
                m_Dt[j + i * N] = m_D[idx];
            }
        }
    }

    // @@ here im generating 3D quadrature weights w3 = wi * wj * wk
    m_w3.resize(m_numPointsPerElement);
    for (int k = 0; k < N; ++k) {
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < N; ++i) {
                int idx = i + N * (j + N * k);
                m_w3[idx] = m_w[i] * m_w[j] * m_w[k];
            }
        }
    }
}

// @@ here im implementing the mesh initialization routine
void Mesh::initialize() {
    // @@ here im setting up GLL operators
    setupGLL();

    // @@ here im allocating arrays for geometric metric factors across all elements
    m_rx.assign(m_totalPoints, 1.0);
    m_sx.assign(m_totalPoints, 0.0);
    m_tx.assign(m_totalPoints, 0.0);

    m_ry.assign(m_totalPoints, 0.0);
    m_sy.assign(m_totalPoints, 1.0);
    m_ty.assign(m_totalPoints, 0.0);

    m_rz.assign(m_totalPoints, 0.0);
    m_sz.assign(m_totalPoints, 0.0);
    m_tz.assign(m_totalPoints, 1.0);

    m_jac.assign(m_totalPoints, 1.0);
}
