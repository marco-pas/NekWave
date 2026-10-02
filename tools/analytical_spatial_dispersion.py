#!/usr/bin/env python3
"""
================================================================================
NekWave: Analytical Bloch-Floquet Dispersion Solver
================================================================================

This script ingests the reference element matrices (Mass M, Stiffness S, Flux F_k)
exported by the C++ engine. It constructs the spatial amplification matrix H(k_vec)
using the Bloch-Floquet plane wave ansatz, solves the eigenvalue problem, and 
filters spurious modes to compute the numerical dispersion relation.

Theoretical Formulation:
  U_k = U_0 * exp(-j * (k_vec . dr_k))
  d(U_0)/dt = H(k_vec) * U_0
  H(k_vec) = M^(-1) * [ S + sum( F_k * exp(-j * (k_vec . dr_k)) ) ]

Usage:
  python3 tools/analytical_spatial_dispersion.py <matrices.json>
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt

def load_matrices(filename):
    """
    Loads M, S, and neighbors (F_k, dr_k) from the C++ exported JSON.
    Returns: M, S, neighbors, dx
    """
    print(f"\n[Loader] Reading matrices from {filename}...")
    with open(filename, 'r') as f:
        data = json.load(f)
        
    M = np.array(data["M"], dtype=np.complex128)
    S = np.array(data["S"], dtype=np.complex128)
    dx = data["dx"]
    order = data.get("order", "Unknown")
    c0 = data.get("C0", "Unknown")
    
    neighbors = []
    for nb in data["neighbors"]:
        neighbors.append({
            "name": nb["name"],
            "dr": np.array(nb["dr"], dtype=np.float64),
            "F": np.array(nb["F"], dtype=np.complex128)
        })
        
    print(f"\n[Loader] Loaded {len(M)}x{len(M)} matrices. Neighbors found: {len(neighbors)}")
    return M, S, neighbors, dx, order, c0

def build_H(M_inv, S, neighbors, k_x, k_y):
    """
    Constructs the spatial amplification matrix H(k_vec).
    H(k) = M_inv @ [ S + sum_k ( F_k * exp(-j * k_vec . dr_k) ) ]
    """
    H = np.copy(S)
    k_vec = np.array([k_x, k_y])
    
    for nb in neighbors:
        dr_k = nb["dr"]
        F_k = nb["F"]
        # Scalar phase shift for the periodic neighbor
        phi_k = np.exp(-1j * np.dot(k_vec, dr_k))
        H += F_k * phi_k
        
    H = M_inv @ H
    return H

def solve_dispersion(M, S, neighbors, dx, thetas_deg, c=1.0):
    """
    Performs the parameter sweep over angles (theta) and wavenumbers (k_mag),
    solves the eigenvalue problem, and extracts the physical mode.
    """
    M_inv = np.linalg.inv(M)
    
    # Define wavenumber array from near 0 up to Nyquist limit (pi / dx)
    k_mags = np.linspace(0.01, np.pi / dx, 100)
    
    results = {theta: [] for theta in thetas_deg}
    
    print(f"\n[Solver] Beginning Floquet eigenvalue sweep...")
    for theta_deg in thetas_deg:
        theta_rad = np.radians(theta_deg)
        print(f"  -> Sweeping angle: {theta_deg} deg")
        
        prev_eigenvector = None
        
        for k_idx, k_mag in enumerate(k_mags):
            k_x = k_mag * np.cos(theta_rad)
            k_y = k_mag * np.sin(theta_rad)
            
            # Construct Amplification Matrix
            H = build_H(M_inv, S, neighbors, k_x, k_y)
            
            # Solve Eigenvalue Problem
            eigenvalues, eigenvectors = np.linalg.eig(H)
            
            # Convert to numerical frequencies: omega_num = |Im(lambda)|
            omega_signed = np.imag(eigenvalues)
            omega_numerical_all = np.abs(omega_signed)
            
            # Normalize all new eigenvectors (columns)
            col_norms = np.linalg.norm(eigenvectors, axis=0, keepdims=True)
            col_norms[col_norms == 0.0] = 1.0
            norm_eigenvectors = eigenvectors / col_norms
            
            # Mode Filtering
            if k_idx == 0:
                # Initialization at lowest k:
                # Strictly isolate the forward-propagating physical mode (+omega)
                omega_exact = c * k_mag
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
            
            omega_num = omega_numerical_all[best_idx]
            
            # Normalized phase velocity: v_p / c
            v_p_normalized = omega_num / (k_mag * c)
            
            # Points Per Wavelength (PPW)
            ppw = (2.0 * np.pi) / (k_mag * dx)
            
            results[theta_deg].append((ppw, v_p_normalized, k_mag, omega_num))
            
    return results

def plot_dispersion(results, order, c0):
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
    
    ax1.set_xscale("log")
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
    
    out_file = "analytical_spatial_dispersion.png"
    plt.savefig(out_file, dpi=300)
    print(f"\n[Plotter] Dispersion plot saved to {out_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 tools/analytical_spatial_dispersion.py <matrices.json>")
        sys.exit(1)
        
    filename = sys.argv[1]
    M, S, neighbors, dx, order, c0 = load_matrices(filename)
    
    thetas_deg = [0, 15, 30, 45]
    c_physical = 1.0 # Wave speed
    
    results = solve_dispersion(M, S, neighbors, dx, thetas_deg, c=c_physical)
    plot_dispersion(results, order, c0)

if __name__ == "__main__":
    main()
