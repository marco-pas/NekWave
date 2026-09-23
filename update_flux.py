import re

with open("examples/analytical_dispersion/analytical_export.cpp", "r") as f:
    content = f.read()

# Replace argument parsing
new_args = """    int N = 4; // Default polynomial order N = 4 (degree P = 3)
    double C0 = 0.0; // Default central flux
    if (argc > 1) N = std::stoi(argv[1]);
    if (argc > 2) C0 = std::stod(argv[2]);
    std::string outJson = (argc > 3) ? argv[3] : "analytical_matrices_order" + std::to_string(N) + ".json";"""

content = re.sub(r'    int N = 4;.*?std::to_string\(N\) \+ "\.json";', new_args, content, flags=re.DOTALL)

# Add C0 to header
content = content.replace('<< " (Polynomial degree P = " << N - 1 << ")" << std::endl;',
                          '<< " (Polynomial degree P = " << N - 1 << "), Flux C0 = " << C0 << std::endl;')

# Replace flux logic
new_flux = """    // Vector velocity for 2D advection equation
    double cx = 1.0;
    double cy = 1.0;

    // We must build the full Stiffness matrix including the SELF flux penalty
    // M * U_dot = S_volume * U - integral( F_self ) - sum integral( F_neigh )
    std::vector<std::vector<double>> S_total(np2D, std::vector<double>(np2D, 0.0));
    for (int i = 0; i < np2D; ++i) {
        for (int j = 0; j < np2D; ++j) {
            S_total[i][j] = cx * S2D_x[i][j] + cy * S2D_y[i][j];
        }
    }

    std::vector<std::vector<double>> F_left(np2D, std::vector<double>(np2D, 0.0));
    std::vector<std::vector<double>> F_right(np2D, std::vector<double>(np2D, 0.0));
    std::vector<std::vector<double>> F_bottom(np2D, std::vector<double>(np2D, 0.0));
    std::vector<std::vector<double>> F_top(np2D, std::vector<double>(np2D, 0.0));

    // Left neighbor (-x face, i=0, n_x = -1)
    double cn_left = cx * (-1.0);
    double f_self_left = 0.5 * (cn_left + C0 * std::abs(cn_left));
    double f_neigh_left = 0.5 * (cn_left - C0 * std::abs(cn_left));
    for (int j = 0; j < N; ++j) {
        int p_ref = 0 + N * j;
        int p_neigh = (N - 1) + N * j;
        double ds = (dy / 2.0) * w1[j];
        S_total[p_ref][p_ref] -= f_self_left * ds;
        F_left[p_ref][p_neigh] = -f_neigh_left * ds;
    }

    // Right neighbor (+x face, i=N-1, n_x = 1)
    double cn_right = cx * (1.0);
    double f_self_right = 0.5 * (cn_right + C0 * std::abs(cn_right));
    double f_neigh_right = 0.5 * (cn_right - C0 * std::abs(cn_right));
    for (int j = 0; j < N; ++j) {
        int p_ref = (N - 1) + N * j;
        int p_neigh = 0 + N * j;
        double ds = (dy / 2.0) * w1[j];
        S_total[p_ref][p_ref] -= f_self_right * ds;
        F_right[p_ref][p_neigh] = -f_neigh_right * ds;
    }

    // Bottom neighbor (-y face, j=0, n_y = -1)
    double cn_bottom = cy * (-1.0);
    double f_self_bottom = 0.5 * (cn_bottom + C0 * std::abs(cn_bottom));
    double f_neigh_bottom = 0.5 * (cn_bottom - C0 * std::abs(cn_bottom));
    for (int i = 0; i < N; ++i) {
        int p_ref = i + N * 0;
        int p_neigh = i + N * (N - 1);
        double ds = (dx / 2.0) * w1[i];
        S_total[p_ref][p_ref] -= f_self_bottom * ds;
        F_bottom[p_ref][p_neigh] = -f_neigh_bottom * ds;
    }

    // Top neighbor (+y face, j=N-1, n_y = 1)
    double cn_top = cy * (1.0);
    double f_self_top = 0.5 * (cn_top + C0 * std::abs(cn_top));
    double f_neigh_top = 0.5 * (cn_top - C0 * std::abs(cn_top));
    for (int i = 0; i < N; ++i) {
        int p_ref = i + N * (N - 1);
        int p_neigh = i + N * 0;
        double ds = (dx / 2.0) * w1[i];
        S_total[p_ref][p_ref] -= f_self_top * ds;
        F_top[p_ref][p_neigh] = -f_neigh_top * ds;
    }"""

old_flux_pattern = r'    // Combine into a single combined stiffness matrix representing advection c=1 along an arbitrary vector.*?(?=    // Export to JSON)'
content = re.sub(old_flux_pattern, new_flux + "\n\n", content, flags=re.DOTALL)

# Add C0 to JSON output
content = content.replace('"order": " << N << ",\\n";', '"order": " << N << ",\\n";\n    out << "  \\"C0\\": " << C0 << ",\\n";')

# Replace S output
content = re.sub(r'    // S.*?out << "  \],\\n";', """    // S (Total with self-flux penalty)
    out << "  \\"S\\": [\\n";
    for (int i = 0; i < np2D; ++i) {
        out << "    [";
        for (int j = 0; j < np2D; ++j) out << std::scientific << std::setprecision(12) << S_total[i][j] << (j + 1 < np2D ? ", " : "");
        out << "]" << (i + 1 < np2D ? ",\\n" : "\\n");
    }
    out << "  ],\\n";""", content, flags=re.DOTALL)

with open("examples/analytical_dispersion/analytical_export.cpp", "w") as f:
    f.write(content)

