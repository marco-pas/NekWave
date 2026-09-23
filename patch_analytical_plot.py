import re

with open("tools/analytical_dispersion.py", "r") as f:
    content = f.read()

# 1. Update the results append
old_append = "results[theta_deg].append((ppw, v_p_normalized))"
new_append = "results[theta_deg].append((ppw, v_p_normalized, k_mag, omega_num))"
content = content.replace(old_append, new_append)

# 2. Rewrite plot_dispersion
old_plot = r'''def plot_dispersion(results, order, c0):
    """
    Plots normalized phase velocity vs Points Per Wavelength.
    """
    plt.figure(figsize=(10, 6))
    
    for theta_deg, data in results.items():
        ppws = [d[0] for d in data]
        vp_norms = [d[1] for d in data]
        plt.plot(ppws, vp_norms, label=f"$\\theta = {theta_deg}^\\circ$", marker='o', markersize=4, alpha=0.8)

    plt.axhline(1.0, color='k', linestyle='--', label='Exact ($v_p = c$)')
    
    plt.title(f"Analytical DGTD Dispersion (Bloch-Floquet)\nOrder $N={order}$, Flux $C_0={c0}$")
    plt.xlabel("Points Per Wavelength (PPW = $\\lambda / dx$)")
    plt.ylabel("Normalized Phase Velocity ($v_p / c$)")
    plt.xlim(max(max(d[0] for d in data) for data in results.values()), 2.0) # From infinity down to Nyquist (2 PPW)
    plt.ylim(0.5, 1.5)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    out_file = "analytical_dispersion.png"
    plt.savefig(out_file, dpi=300)
    print(f"\n[Plotter] Dispersion plot saved to {out_file}")'''

new_plot = r'''def plot_dispersion(results, order, c0):
    """
    Plots normalized phase velocity vs Points Per Wavelength on the left,
    and k vs omega_num on the right.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    fig.suptitle(
        f"Analytical DGTD Dispersion Analysis (1 Element, Order N = {order}, Flux C0 = {c0})",
        fontsize=13, fontweight='bold'
    )
    
    hex_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # --- Left Plot: v_p / c vs PPW ---
    ax1.axhline(1.0, color='gray', linestyle='--', linewidth=1.2, label=r'Exact: $v_p / c = 1.0$')
    
    for idx, (theta_deg, data) in enumerate(results.items()):
        ppws = [d[0] for d in data]
        vp_norms = [d[1] for d in data]
        ax1.plot(ppws, vp_norms, 'o-', color=hex_colors[idx % len(hex_colors)],
                 linewidth=2.0, markersize=4, alpha=0.8, label=f"$\\theta = {theta_deg}^\\circ$")

    ax1.set_xlabel(r'Points Per Wavelength (PPW = $\lambda / \Delta x$)', fontsize=11, fontweight='bold')
    ax1.set_ylabel(r'Normalized Phase Velocity $v_p / c$', fontsize=11, fontweight='bold')
    
    # Calculate limits for PPW axis
    all_ppws = [d[0] for data in results.values() for d in data]
    ax1.set_xlim(max(all_ppws), 2.0) # High PPW to low PPW (Nyquist)
    
    ax1.set_ylim([0.30, 1.6])
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left', frameon=True, framealpha=0.9)

    # --- Right Plot: omega_num vs k ---
    all_ks = [d[2] for data in results.values() for d in data]
    max_k = max(all_ks) if all_ks else 10.0
    k_line = [0.0, max_k * 1.1]
    
    # Exact omega = c * k (c=1.0)
    ax2.plot(k_line, k_line, color='gray', linestyle='--', linewidth=1.2, label=r'Exact: $\omega = c k$')
    
    for idx, (theta_deg, data) in enumerate(results.items()):
        k_vals = [d[2] for d in data]
        w_nums = [d[3] for d in data]
        ax2.plot(k_vals, w_nums, 'o-', color=hex_colors[idx % len(hex_colors)],
                 linewidth=2.0, markersize=4, alpha=0.8, label=f"$\\theta = {theta_deg}^\\circ$")

    ax2.set_xlabel(r'Wavenumber Magnitude $|k|$ [rad]', fontsize=11, fontweight='bold')
    ax2.set_ylabel(r'Numerical Frequency $\omega_{\mathrm{num}}$ [rad/s]', fontsize=11, fontweight='bold')
    ax2.set_xlim(0, max_k * 1.05)
    ax2.set_ylim(0, max_k * 1.05)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper left', frameon=True, framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    out_file = "analytical_dispersion.png"
    plt.savefig(out_file, dpi=300)
    print(f"\n[Plotter] Dispersion plot saved to {out_file}")'''

# Regex to safely replace the old plot definition.
content = re.sub(r'def plot_dispersion\(results, order, c0\):.*?print\(f"\\n\[Plotter\] Dispersion plot saved to \{out_file\}"\)', new_plot, content, flags=re.DOTALL)

with open("tools/analytical_dispersion.py", "w") as f:
    f.write(content)

