// ==============================================================================
// NekWave: Reference Element DGTD Bloch-Floquet Matrix Exporter
// ==============================================================================
//
// Extracts exact reference element operators (Mass matrix M, Stiffness matrix S,
// and Interface Flux matrices F_k) for the 2D Maxwell TM system (Ez, Hx, Hy)
// directly from NekWave's core mesh quadrature, spectral differentiation, and
// face flux routines matching dg_solver.cpp and dg_kernels.cu.
//
// Outputs matrices in JSON format to be analyzed by analytical Floquet solvers.
//
// Usage:
//   ./analytical_export <order> <C0> [output_file.json]
// ==============================================================================

#include "mesh.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>

int main(int argc, char* argv[]) {
    int N = 4; // Default polynomial order N = 4 (degree P = 3)
    double C0 = 0.0; // Default central flux
    if (argc > 1) {
        N = std::stoi(argv[1]);
    }
    if (argc > 2) {
        C0 = std::stod(argv[2]);
    }
    std::string outJson = (argc > 3) ? argv[3] : "analytical_matrices_order" + std::to_string(N) + ".json";

    std::cout << "==========================================================" << std::endl;
    std::cout << "    NekWave Reference Element Analytical Matrix Exporter  " << std::endl;
    std::cout << "           (Maxwell TM System: Ez, Hx, Hy)                " << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << "Polynomial Order N = " << N << " (Polynomial degree P = " << N - 1 << ")" << std::endl;
    std::cout << "Flux Formulation C0 = " << C0 << " (0.0 = Central, 1.0 = Upwind)" << std::endl;

    // Instantiate a single reference element of size dx = 1, dy = 1, dz = 1
    const double dx = 1.0, dy = 1.0, dz = 1.0;
    Mesh mesh(N, 1);
    mesh.createBoxMesh(1, 1, 1, -0.5 * dx, 0.5 * dx, -0.5 * dy, 0.5 * dy, -0.5 * dz, 0.5 * dz, true, true, true);

    const auto& w1 = mesh.getGllW();
    const auto& D1 = mesh.getD(); 

    // 2D reference element has N^2 spatial nodes
    const int np2D = N * N;
    // Maxwell TM system has 3 field components: Ez (0), Hx (1), Hy (2) -> total DOFs = 3 * N^2
    const int dim = 3 * np2D;

    auto idx_Ez = [np2D](int p) { return 0 * np2D + p; };
    auto idx_Hx = [np2D](int p) { return 1 * np2D + p; };
    auto idx_Hy = [np2D](int p) { return 2 * np2D + p; };

    std::vector<std::vector<double>> M2D(np2D, std::vector<double>(np2D, 0.0));
    std::vector<std::vector<double>> S2D_x(np2D, std::vector<double>(np2D, 0.0));
    std::vector<std::vector<double>> S2D_y(np2D, std::vector<double>(np2D, 0.0));

    // Diagonal 2D Mass Matrix: M_{(i,j),(i,j)} = (dx / 2) * (dy / 2) * w_i * w_j
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            int p = i + N * j;
            M2D[p][p] = (dx / 2.0) * (dy / 2.0) * w1[i] * w1[j];
        }
    }

    // 2D Mass-weighted derivative matrices:
    // S2D_x = M2D * D_x,  S2D_y = M2D * D_y
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            int row = i + N * j;
            for (int k = 0; k < N; ++k) {
                // S_x derivative in xi (i index)
                int col_x = k + N * j;
                double d_ik = D1[i + k * N];
                S2D_x[row][col_x] += (dy / 2.0) * w1[j] * w1[i] * d_ik;
                
                // S_y derivative in eta (j index)
                int col_y = i + N * k;
                double d_jk = D1[j + k * N];
                S2D_y[row][col_y] += (dx / 2.0) * w1[i] * w1[j] * d_jk;
            }
        }
    }

    // 3D Maxwell Block Mass Matrix (dim x dim)
    std::vector<std::vector<double>> M(dim, std::vector<double>(dim, 0.0));
    for (int p = 0; p < np2D; ++p) {
        M[idx_Ez(p)][idx_Ez(p)] = M2D[p][p];
        M[idx_Hx(p)][idx_Hx(p)] = M2D[p][p];
        M[idx_Hy(p)][idx_Hy(p)] = M2D[p][p];
    }

    // Full Stiffness Matrix S_total (dim x dim): includes volume curl and face self-flux penalties
    // Maxwell TM system:
    //   dEz/dt = +(dHy/dx - dHx/dy) + flux_Ez
    //   dHx/dt = -dEz/dy            + flux_Hx
    //   dHy/dt = +dEz/dx            + flux_Hy
    std::vector<std::vector<double>> S_total(dim, std::vector<double>(dim, 0.0));

    // Volume curl couplings
    for (int p = 0; p < np2D; ++p) {
        int i = p % N;
        int j = p / N;
        for (int k = 0; k < N; ++k) {
            int col_x = k + N * j;
            int col_y = i + N * k;
            
            // Ez equation: +dHy/dx - dHx/dy
            S_total[idx_Ez(p)][idx_Hy(col_x)] += S2D_x[p][col_x];
            S_total[idx_Ez(p)][idx_Hx(col_y)] -= S2D_y[p][col_y];

            // Hx equation: -dEz/dy
            S_total[idx_Hx(p)][idx_Ez(col_y)] -= S2D_y[p][col_y];

            // Hy equation: +dEz/dx
            S_total[idx_Hy(p)][idx_Ez(col_x)] += S2D_x[p][col_x];
        }
    }

    // Interface neighbor matrices (dim x dim)
    std::vector<std::vector<double>> F_left(dim, std::vector<double>(dim, 0.0));
    std::vector<std::vector<double>> F_right(dim, std::vector<double>(dim, 0.0));
    std::vector<std::vector<double>> F_bottom(dim, std::vector<double>(dim, 0.0));
    std::vector<std::vector<double>> F_top(dim, std::vector<double>(dim, 0.0));

    // --------------------------------------------------------------------------
    // Face 1: Left neighbor (-x face, i = 0, n = (-1, 0))
    // --------------------------------------------------------------------------
    for (int j = 0; j < N; ++j) {
        int p_ref = 0 + N * j;
        int p_neigh = (N - 1) + N * j;
        double ds = (dy / 2.0) * w1[j];

        // Self terms added to S_total
        S_total[idx_Ez(p_ref)][idx_Hy(p_ref)] += 0.5 * ds;
        S_total[idx_Ez(p_ref)][idx_Ez(p_ref)] -= 0.5 * C0 * ds;
        S_total[idx_Hy(p_ref)][idx_Ez(p_ref)] += 0.5 * ds;
        S_total[idx_Hy(p_ref)][idx_Hy(p_ref)] -= 0.5 * C0 * ds;

        // Neighbor terms
        F_left[idx_Ez(p_ref)][idx_Hy(p_neigh)] -= 0.5 * ds;
        F_left[idx_Ez(p_ref)][idx_Ez(p_neigh)] += 0.5 * C0 * ds;
        F_left[idx_Hy(p_ref)][idx_Ez(p_neigh)] -= 0.5 * ds;
        F_left[idx_Hy(p_ref)][idx_Hy(p_neigh)] += 0.5 * C0 * ds;
    }

    // --------------------------------------------------------------------------
    // Face 2: Right neighbor (+x face, i = N - 1, n = (+1, 0))
    // --------------------------------------------------------------------------
    for (int j = 0; j < N; ++j) {
        int p_ref = (N - 1) + N * j;
        int p_neigh = 0 + N * j;
        double ds = (dy / 2.0) * w1[j];

        // Self terms added to S_total
        S_total[idx_Ez(p_ref)][idx_Hy(p_ref)] -= 0.5 * ds;
        S_total[idx_Ez(p_ref)][idx_Ez(p_ref)] -= 0.5 * C0 * ds;
        S_total[idx_Hy(p_ref)][idx_Ez(p_ref)] -= 0.5 * ds;
        S_total[idx_Hy(p_ref)][idx_Hy(p_ref)] -= 0.5 * C0 * ds;

        // Neighbor terms
        F_right[idx_Ez(p_ref)][idx_Hy(p_neigh)] += 0.5 * ds;
        F_right[idx_Ez(p_ref)][idx_Ez(p_neigh)] += 0.5 * C0 * ds;
        F_right[idx_Hy(p_ref)][idx_Ez(p_neigh)] += 0.5 * ds;
        F_right[idx_Hy(p_ref)][idx_Hy(p_neigh)] += 0.5 * C0 * ds;
    }

    // --------------------------------------------------------------------------
    // Face 3: Bottom neighbor (-y face, j = 0, n = (0, -1))
    // --------------------------------------------------------------------------
    for (int i = 0; i < N; ++i) {
        int p_ref = i + N * 0;
        int p_neigh = i + N * (N - 1);
        double ds = (dx / 2.0) * w1[i];

        // Self terms added to S_total
        S_total[idx_Ez(p_ref)][idx_Hx(p_ref)] -= 0.5 * ds;
        S_total[idx_Ez(p_ref)][idx_Ez(p_ref)] -= 0.5 * C0 * ds;
        S_total[idx_Hx(p_ref)][idx_Ez(p_ref)] -= 0.5 * ds;
        S_total[idx_Hx(p_ref)][idx_Hx(p_ref)] -= 0.5 * C0 * ds;

        // Neighbor terms
        F_bottom[idx_Ez(p_ref)][idx_Hx(p_neigh)] += 0.5 * ds;
        F_bottom[idx_Ez(p_ref)][idx_Ez(p_neigh)] += 0.5 * C0 * ds;
        F_bottom[idx_Hx(p_ref)][idx_Ez(p_neigh)] += 0.5 * ds;
        F_bottom[idx_Hx(p_ref)][idx_Hx(p_neigh)] += 0.5 * C0 * ds;
    }

    // --------------------------------------------------------------------------
    // Face 4: Top neighbor (+y face, j = N - 1, n = (0, +1))
    // --------------------------------------------------------------------------
    for (int i = 0; i < N; ++i) {
        int p_ref = i + N * (N - 1);
        int p_neigh = i + N * 0;
        double ds = (dx / 2.0) * w1[i];

        // Self terms added to S_total
        S_total[idx_Ez(p_ref)][idx_Hx(p_ref)] += 0.5 * ds;
        S_total[idx_Ez(p_ref)][idx_Ez(p_ref)] -= 0.5 * C0 * ds;
        S_total[idx_Hx(p_ref)][idx_Ez(p_ref)] += 0.5 * ds;
        S_total[idx_Hx(p_ref)][idx_Hx(p_ref)] -= 0.5 * C0 * ds;

        // Neighbor terms
        F_top[idx_Ez(p_ref)][idx_Hx(p_neigh)] -= 0.5 * ds;
        F_top[idx_Ez(p_ref)][idx_Ez(p_neigh)] += 0.5 * C0 * ds;
        F_top[idx_Hx(p_ref)][idx_Ez(p_neigh)] -= 0.5 * ds;
        F_top[idx_Hx(p_ref)][idx_Hx(p_neigh)] += 0.5 * C0 * ds;
    }

    // Export to JSON
    std::ofstream out(outJson);
    out << "{\n";
    out << "  \"order\": " << N << ",\n";
    out << "  \"dx\": " << dx << ",\n";
    out << "  \"dy\": " << dy << ",\n";
    out << "  \"dxmin\": " << mesh.computeMinNodeDistance() << ",\n";
    out << "  \"C0\": " << C0 << ",\n";

    // M
    out << "  \"M\": [\n";
    for (int i = 0; i < dim; ++i) {
        out << "    [";
        for (int j = 0; j < dim; ++j) out << std::scientific << std::setprecision(12) << M[i][j] << (j + 1 < dim ? ", " : "");
        out << "]" << (i + 1 < dim ? ",\n" : "\n");
    }
    out << "  ],\n";

    // S (Total with self-flux penalty)
    out << "  \"S\": [\n";
    for (int i = 0; i < dim; ++i) {
        out << "    [";
        for (int j = 0; j < dim; ++j) out << std::scientific << std::setprecision(12) << S_total[i][j] << (j + 1 < dim ? ", " : "");
        out << "]" << (i + 1 < dim ? ",\n" : "\n");
    }
    out << "  ],\n";

    out << "  \"neighbors\": [\n";
    out << "    { \"name\": \"left\", \"dr\": [" << -dx << ", 0.0], \"F\": [\n";
    for (int i=0; i<dim; ++i) { out << "      ["; for(int j=0; j<dim; ++j) out << F_left[i][j] << (j+1<dim?", ":""); out << "]" << (i+1<dim?",\n":"\n"); }
    out << "    ]},\n";
    out << "    { \"name\": \"right\", \"dr\": [" << dx << ", 0.0], \"F\": [\n";
    for (int i=0; i<dim; ++i) { out << "      ["; for(int j=0; j<dim; ++j) out << F_right[i][j] << (j+1<dim?", ":""); out << "]" << (i+1<dim?",\n":"\n"); }
    out << "    ]},\n";
    out << "    { \"name\": \"bottom\", \"dr\": [0.0, " << -dy << "], \"F\": [\n";
    for (int i=0; i<dim; ++i) { out << "      ["; for(int j=0; j<dim; ++j) out << F_bottom[i][j] << (j+1<dim?", ":""); out << "]" << (i+1<dim?",\n":"\n"); }
    out << "    ]},\n";
    out << "    { \"name\": \"top\", \"dr\": [0.0, " << dy << "], \"F\": [\n";
    for (int i=0; i<dim; ++i) { out << "      ["; for(int j=0; j<dim; ++j) out << F_top[i][j] << (j+1<dim?", ":""); out << "]" << (i+1<dim?",\n":"\n"); }
    out << "    ]}\n";
    out << "  ]\n";
    out << "}\n";
    out.close();

    std::cout << "Successfully exported matrices to " << outJson << std::endl;
    return 0;
}
