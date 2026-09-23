import re

with open("examples/numerical_analysis/neumann_stability/plot_neumann.py", "r") as f:
    content = f.read()

# 1. Update load_matrices to fetch dxmin
old_loader = r"dx, order, c0 = asd.load_matrices\(matrices_json\)"
new_loader = """M, S, neighbors, dx, order, c0 = asd.load_matrices(matrices_json)
    
    with open(matrices_json, 'r') as f:
        data = json.load(f)
        dxmin = data.get("dxmin", dx / order)"""
content = re.sub(r"M, S, neighbors, dx, order, c0 = asd.load_matrices\(matrices_json\)", new_loader, content)

old_return = r"return np.array\(all_eigenvalues\), dx, order, c0"
new_return = r"return np.array(all_eigenvalues), dx, order, c0, dxmin"
content = re.sub(old_return, new_return, content)

# 2. Update plot_neumann_stability to use safety_factor instead of cfl, and compute dt
old_plot_sig = r"def plot_neumann_stability\(matrices_json=None, cfl=0.5\):"
new_plot_sig = r"def plot_neumann_stability(matrices_json=None, safety_factor=0.4):"
content = re.sub(old_plot_sig, new_plot_sig, content)

old_overlay = r'''    if matrices_json:
        print\(f"Overlaying spatial eigenvalues from \{matrices_json\} at CFL = \{cfl\}"\)
        eigs, dx, order, c0 = compute_spatial_eigenvalues\(matrices_json\)
        
        # Calculate Delta t based on CFL
        c_physical = 1.0
        dt = cfl \* dx / c_physical
        
        # Scaled eigenvalues: z = lambda_H \* dt
        z_vals = eigs \* dt
        
        plt.scatter\(np.real\(z_vals\), np.imag\(z_vals\), s=5, color='red', alpha=0.6, label=f'Spatial Eigenvalues \(CFL=\{cfl\}\)'\)
        title_str \+= f'\\nOrder \$N=\{order\}\$, Flux \$C_0=\{c0\}\$, \$CFL=\{cfl\}\$''''

new_overlay = r'''    if matrices_json:
        eigs, dx, order, c0, dxmin = compute_spatial_eigenvalues(matrices_json)
        
        # Automatic dt calculation based on dg_solver.cpp
        alpha_RK = 1.75
        sqrt_dim = np.sqrt(3.0)
        cfl_auto = (safety_factor * alpha_RK) / sqrt_dim
        
        c_physical = 1.0
        dt = (cfl_auto * dxmin) / c_physical
        
        print(f"Overlaying spatial eigenvalues from {matrices_json} at safetyFactor = {safety_factor}")
        print(f"  -> dxmin = {dxmin:.6f}, CFL_auto = {cfl_auto:.4f}, dt = {dt:.6e}")
        
        # Scaled eigenvalues: z = lambda_H * dt
        z_vals = eigs * dt
        
        plt.scatter(np.real(z_vals), np.imag(z_vals), s=5, color='red', alpha=0.6, label=f'Spatial Eigenvalues\n(dt = {dt:.2e})')
        title_str += f'\nOrder $N={order}$, Flux $C_0={c0}$, Safety = {safety_factor} (CFL $\\approx$ {cfl_auto:.3f})'
'''
content = re.sub(old_overlay, new_overlay, content)

# 3. Update main
old_main = r'''if __name__ == "__main__":
    if len\(sys.argv\) > 1:
        json_file = sys.argv\[1\]
        cfl = float\(sys.argv\[2\]\) if len\(sys.argv\) > 2 else 0.5
        plot_neumann_stability\(json_file, cfl\)
    else:
        print\("Usage: python3 plot_neumann.py \[matrices.json\] \[CFL\]"\)'''

new_main = r'''if __name__ == "__main__":
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
        sf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
        plot_neumann_stability(json_file, sf)
    else:
        print("Usage: python3 plot_neumann.py [matrices.json] [safetyFactor]")'''
content = re.sub(old_main, new_main, content)

with open("examples/numerical_analysis/neumann_stability/plot_neumann.py", "w") as f:
    f.write(content)
