import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Import the spatial tools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../tools')))
import analytical_spatial_dispersion as asd

def stability_polynomial(z):
    # Carpenter & Kennedy LSRK45 (1994)
    rk4a = [0.0,
            -567301805773.0 / 1357537059087.0,
            -2404267990393.0 / 2016746695238.0,
            -3550918686646.0 / 2091501179385.0,
            -1275806237668.0 / 842570457699.0]
            
    rk4b = [1432997174477.0 / 9575080441755.0,
            5161836677717.0 / 13612068292357.0,
            1720146321549.0 / 2090206949498.0,
            3134564353537.0 / 4481467310338.0,
            2277821191437.0 / 14882151754819.0]

    u = np.ones_like(z, dtype=np.complex128)
    k = np.zeros_like(z, dtype=np.complex128)
    
    for a, b in zip(rk4a, rk4b):
        k = a * k + z * u
        u = u + b * k
        
    return u

def solve_full_dispersion(matrices_json, safety_factor=0.4):
    """
    Computes spatio-temporal dispersion.
    """
    M, S, neighbors, dx, order, c0 = asd.load_matrices(matrices_json)
    M_inv = np.linalg.inv(M)
    
    with open(matrices_json, 'r') as f:
        data = json.load(f)
        dxmin = data.get("dxmin", dx / order)
        
    alpha_RK = 1.75
    sqrt_dim = np.sqrt(3.0)
    cfl_auto = (safety_factor * alpha_RK) / sqrt_dim
    c_physical = 1.0
    dt = (cfl_auto * dxmin) / c_physical
    
    print(f"\n[Solver] Full Spatio-Temporal Floquet sweep...")
    print(f"  -> Order: {order}, Flux: {c0}, dxmin: {dxmin:.5f}")
    print(f"  -> CFL: {cfl_auto:.4f}, dt: {dt:.5e}")
    
    thetas_deg = [0, 15, 30, 45]
    k_mags = np.linspace(0.01, np.pi / dx, 100) # this limits the total number of PPW to 628!
    
    results = {theta: [] for theta in thetas_deg}
    
    for theta_deg in thetas_deg:
        theta_rad = np.radians(theta_deg)
        print(f"  -> Sweeping angle: {theta_deg} deg")
        
        prev_eigenvector = None
        
        for k_idx, k_mag in enumerate(k_mags):
            k_x = k_mag * np.cos(theta_rad)
            k_y = k_mag * np.sin(theta_rad)
            
            # Construct Spatial Amplification Matrix
            H = asd.build_H(M_inv, S, neighbors, k_x, k_y)
            
            # Spatial Eigenvalues and Eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(H)
            
            # Scale to complex z = lambda_H * dt
            z = eigenvalues * dt
            
            # Pass through LSRK45 temporal stability polynomial
            G = stability_polynomial(z)
            
            # Extract discrete numerical frequencies from phase: omega = arg(G(z)) / dt
            omega_signed = np.angle(G) / dt
            omega_num_all = np.abs(omega_signed)
            
            # Normalize all new eigenvectors (columns)
            col_norms = np.linalg.norm(eigenvectors, axis=0, keepdims=True)
            col_norms[col_norms == 0.0] = 1.0
            norm_eigenvectors = eigenvectors / col_norms
            
            # Mode Filtering
            if k_idx == 0:
                # Initialization at lowest k:
                # Strictly isolate the forward-propagating physical mode (+omega)
                # This prevents hopping between forward and backward waves.
                omega_exact = c_physical * k_mag
                freq_diff = np.abs(omega_signed - omega_exact)
                damping = np.abs(np.real(eigenvalues))
                best_idx = np.argmin(freq_diff + damping)
            else:
                # Tracking loop:
                # Calculate absolute inner product with previous physical eigenvector using conjugate transpose
                correlations = np.array([
                    np.abs(np.vdot(prev_eigenvector, norm_eigenvectors[:, i]))
                    for i in range(len(eigenvalues))
                ])
                best_idx = np.argmax(correlations)
            
            # Update prev_eigenvector for the next k step
            prev_eigenvector = norm_eigenvectors[:, best_idx]
            
            # 1. Full discrete mode (coupled space + time)
            omega_num_full = omega_num_all[best_idx]
            v_p_full = omega_num_full / (k_mag * c_physical)
            
            # 2. Pure spatial mode (Delta t -> 0)
            lambda_phys = eigenvalues[best_idx]
            omega_num_space = np.abs(np.imag(lambda_phys))
            v_p_space = omega_num_space / (k_mag * c_physical)
            
            # 3. Pure temporal mode (Delta x -> 0, continuous wave z = j * c * k * dt)
            z_exact = 1j * (c_physical * k_mag) * dt
            G_exact = stability_polynomial(z_exact)
            omega_num_time = np.abs(np.angle(G_exact)) / dt
            v_p_time = omega_num_time / (k_mag * c_physical)
            
            ppw = (2.0 * np.pi) / (k_mag * dx)
            
            results[theta_deg].append({
                "ppw": ppw,
                "k_mag": k_mag,
                "v_p_full": v_p_full,
                "v_p_space": v_p_space,
                "v_p_time": v_p_time,
                "w_full": omega_num_full,
                "w_space": omega_num_space,
                "w_time": omega_num_time
            })
            
    return results, order, c0, safety_factor

def plot_full_dispersion(results, order, c0, safety_factor):
    """
    Plots the dual-panel dispersion results with decoupled space and time curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
    fig.suptitle(
        f"DGTD Dispersion Decomposition: Space vs. Time (Order N = {order}, Flux C0 = {c0:.1f}, Safety = {safety_factor})\n"
        r"[Solid: Full (Space+Time) | Dashed: Pure Spatial ($\Delta t \to 0$) | Dotted: Pure Temporal ($\Delta x \to 0$)]",
        fontsize=12, fontweight='bold'
    )
    
    hex_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']

    # --- Left Plot: v_p / c vs PPW ---
    ax1.axhline(1.0, color='gray', linestyle='--', linewidth=1.2, label=r'Exact: $v_p / c = 1.0$')
    
    # Plot pure time curve first (isotropic, angle-independent)
    first_angle_data = list(results.values())[0]
    ppws_t = [d["ppw"] for d in first_angle_data]
    vps_t = [d["v_p_time"] for d in first_angle_data]
    ax1.plot(ppws_t, vps_t, ':', color='black', linewidth=2.2, alpha=0.9, label='Pure Time (RK45)')

    for idx, (theta_deg, data) in enumerate(results.items()):
        color = hex_colors[idx % len(hex_colors)]
        ppws = [d["ppw"] for d in data]
        vps_full = [d["v_p_full"] for d in data]
        vps_space = [d["v_p_space"] for d in data]
        
        # Coupled Space+Time
        ax1.plot(ppws, vps_full, 'o-', color=color, linewidth=1.8, markersize=3, alpha=0.85,
                 label=f"$\\theta = {theta_deg}^\\circ$ (Full)")
        # Pure Spatial
        ax1.plot(ppws, vps_space, '--', color=color, linewidth=1.4, alpha=0.75,
                 label=f"$\\theta = {theta_deg}^\\circ$ (Spatial)")

    ax1.set_xlabel(r'Points Per Wavelength (PPW = $\lambda / \Delta x$)', fontsize=11, fontweight='bold')
    ax1.set_ylabel(r'Normalized Phase Velocity $v_p / c$', fontsize=11, fontweight='bold')
    
    all_ppws = [d["ppw"] for data in results.values() for d in data]
    if all_ppws:
        ax1.set_xlim(max(all_ppws), 2.0)
    
    ax1.set_xscale("log")
    ax1.set_ylim([0.00, 2.0])
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left', frameon=True, framealpha=0.9, fontsize=8, ncol=2)

    # --- Right Plot: omega_num vs k ---
    k_line = [0.0, 4.0]
    
    ax2.plot(k_line, k_line, color='gray', linestyle='--', linewidth=1.2, label=r'Exact: $\omega = c k$')
    
    # Pure time curve
    ks_t = [d["k_mag"] for d in first_angle_data]
    ws_t = [d["w_time"] for d in first_angle_data]
    ax2.plot(ks_t, ws_t, ':', color='black', linewidth=2.2, alpha=0.9, label='Pure Time (RK45)')

    for idx, (theta_deg, data) in enumerate(results.items()):
        color = hex_colors[idx % len(hex_colors)]
        k_vals = [d["k_mag"] for d in data]
        w_fulls = [d["w_full"] for d in data]
        w_spaces = [d["w_space"] for d in data]
        
        ax2.plot(k_vals, w_fulls, 'o-', color=color, linewidth=1.8, markersize=3, alpha=0.85,
                 label=f"$\\theta = {theta_deg}^\\circ$ (Full)")
        ax2.plot(k_vals, w_spaces, '--', color=color, linewidth=1.4, alpha=0.75,
                 label=f"$\\theta = {theta_deg}^\\circ$ (Spatial)")

    ax2.set_xlabel(r'Wavenumber Magnitude $|k|$ [rad]', fontsize=11, fontweight='bold')
    ax2.set_ylabel(r'Numerical Frequency $\omega_{\mathrm{num}}$ [rad/s]', fontsize=11, fontweight='bold')
    ax2.set_xlim(0, 4.0)
    ax2.set_ylim(0, 4.0)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=8, ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    os.makedirs('results', exist_ok=True)
    out_file = f"results/full_dispersion_N{order}_C{c0:.1f}.png"
    plt.savefig(out_file, dpi=300)
    print(f"\n[Plotter] Full Dispersion plot saved to {out_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 plot_full_dispersion.py <matrices.json> [safetyFactor]")
        sys.exit(1)
        
    json_file = sys.argv[1]
    sf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
    
    results, order, c0, safety = solve_full_dispersion(json_file, sf)
    plot_full_dispersion(results, order, c0, safety)
