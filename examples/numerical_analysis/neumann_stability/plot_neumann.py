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

def compute_spatial_eigenvalues(matrices_json):
    M, S, neighbors, dx, order, c0 = asd.load_matrices(matrices_json)
    M_inv = np.linalg.inv(M)
    
    with open(matrices_json, 'r') as f:
        data = json.load(f)
        dxmin = data.get("dxmin", dx / order)
    
    # Sweep over angles and wavenumbers to collect all spatial eigenvalues
    thetas_deg = [0, 15, 30, 45]
    k_mags = np.linspace(0.01, np.pi / dx, 50)
    
    all_eigenvalues = []
    
    for theta_deg in thetas_deg:
        theta_rad = np.radians(theta_deg)
        for k_mag in k_mags:
            k_x = k_mag * np.cos(theta_rad)
            k_y = k_mag * np.sin(theta_rad)
            
            H = asd.build_H(M_inv, S, neighbors, k_x, k_y)
            eigenvalues, _ = np.linalg.eig(H)
            all_eigenvalues.extend(eigenvalues)
            
    return np.array(all_eigenvalues), dx, order, c0, dxmin

def plot_neumann_stability(matrices_json=None, safety_factor=0.4):
    # Grid of complex plane
    x = np.linspace(-5, 1.0, 500)
    y = np.linspace(-4, 4, 500)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    G_mag = np.abs(stability_polynomial(Z))
    
    plt.figure(figsize=(10, 8), dpi=150)
    
    # Fill stability region
    plt.contourf(X, Y, G_mag, levels=[0.0, 1.0], colors=['#a1d99b'], alpha=0.5)
    plt.contour(X, Y, G_mag, levels=[1.0], colors='k', linewidths=1.5, linestyles='solid')
    
    title_str = 'Neumann Stability Analysis (LSRK45 Carpenter & Kennedy)'
    
    if matrices_json:
        eigs, dx, order, c0, dxmin = compute_spatial_eigenvalues(matrices_json)
        
        # Automatic dt calculation strictly mapped from dg_solver.cpp
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
        plt.legend(loc='upper left')
    
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.xlabel('Re(z) = Re($\lambda \Delta t$)', fontsize=12)
    plt.ylabel('Im(z) = Im($\lambda \Delta t$)', fontsize=12)
    plt.title(title_str, fontsize=13, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    os.makedirs('results', exist_ok=True)
    out_file = 'results/neumann_stability.png'
    plt.savefig(out_file, dpi=300)
    print(f"Saved neumann stability plot to {out_file}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
        sf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
        plot_neumann_stability(json_file, sf)
    else:
        print("Usage: python3 plot_neumann.py [matrices.json] [safetyFactor]")
        print("Plotting bare stability region without spatial eigenvalues...")
        plot_neumann_stability()
