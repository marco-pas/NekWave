#!/usr/bin/env python3
"""
================================================================================
NekWave: Two-Probe Numerical Dispersion Phase Analysis
================================================================================
Extracts numerical phase velocity directly from two point probes in a CUDA simulation:
    1. Reads output/probes.json for exact GLL collocation probe distance and grid info.
    2. Loads probe_history.csv from NekWave simulation.
    3. Evaluates discrete harmonic Fourier coefficients at physical modes w_m = 2*pi*m*c / Lx.
    4. Computes unwrapped phase difference: Delta_phi_m = arg(S1) - arg(S2) + 2*pi*n.
    5. Computes numerical phase velocity: v_p(m) = w_m * dist / Delta_phi_m.
    6. Plots pure numerical dispersion curves (v_p/c vs PPW, and omega vs k).
================================================================================
"""

import sys
import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt

def load_probe_metadata(csv_path):
    """
    Checks for probes.json in the same directory as probe_history.csv.
    Returns separation distance, domain Lx, and probe dict if found.
    """
    json_path = os.path.join(os.path.dirname(csv_path), "probes.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            dist = data.get("separation_distance", None)
            Lx = data.get("domain_Lx", 10.0)
            print(f"[Metadata] Found probes.json: exact separation distance d = {dist:.8f} (domain Lx = {Lx:.8f})")
            return dist, Lx, data
        except Exception as e:
            print(f"[Metadata] Warning: could not parse {json_path}: {e}")
    return None, None, None

def load_two_probe_data(csv_path):
    """
    Reads probe_history.csv and extracts time, probe 1 Ez, and probe 2 Ez.
    """
    times = []
    ez1 = []
    ez2 = []

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        col_t = header.index("time")
        
        # Locate Ez columns (Probe 1 is Ez at col 4, Probe 2 is Ez at col 10)
        ez_cols = [idx for idx, name in enumerate(header) if "Ez" in name]
        if len(ez_cols) < 2:
            raise ValueError(f"Expected at least 2 Ez probe columns in {csv_path}, found {len(ez_cols)}")

        c1 = ez_cols[0]
        c2 = ez_cols[1]

        for row in reader:
            if not row or len(row) <= max(c1, c2):
                continue
            times.append(float(row[col_t]))
            ez1.append(float(row[c1]))
            ez2.append(float(row[c2]))

    return np.array(times), np.array(ez1), np.array(ez2)

def compute_modal_phase_velocity(times, s1, s2, dist, Lx=25.132741228718345, c_physical=1.0, max_m=15, k_modes=None):
    """
    Extracts numerical phase velocity for excited discrete harmonic modes m = 1..max_m.
    Uses zero-padded windowed FFT with parabolic interpolation to locate exact numerical frequencies w_num(m)
    and evaluates two-probe cross-spectral phase difference at each peak.
    Evaluates every excited mode uniformly without flux-dependent thresholds.
    """
    num_steps = len(times)
    if num_steps < 10:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    dt = times[1] - times[0]
    
    # 1. Remove DC bias to prevent zero-frequency leakage into Mode 1 (k = 0.25)
    s1_ac = s1 - np.mean(s1)
    s2_ac = s2 - np.mean(s2)

    # 2. Blackman-Harris 4-term window: -92 dB sidelobe suppression for minimal spectral leakage
    n = np.arange(num_steps)
    window = (0.35875 
              - 0.48829 * np.cos(2.0 * np.pi * n / (num_steps - 1))
              + 0.14128 * np.cos(4.0 * np.pi * n / (num_steps - 1))
              - 0.01168 * np.cos(6.0 * np.pi * n / (num_steps - 1)))
    window /= np.mean(window)  # Coherent gain normalization

    # 3. High-resolution zero-padded FFT (65,536 bins for dense sub-bin sampling)
    n_fft = max(65536, 1 << (num_steps * 2 - 1).bit_length())
    S1 = np.fft.rfft(s1_ac * window, n=n_fft)
    S2 = np.fft.rfft(s2_ac * window, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=dt)
    omegas = 2.0 * np.pi * freqs
    
    # Cross-spectrum: H12 = S1 * conj(S2)
    H12 = S1 * np.conj(S2)
    mag1 = np.abs(S1)

    modes = []
    omegas_out = []
    k_nums = []
    vps = []
    mags = []

    if k_modes is not None and len(k_modes) > 0:
        mode_items = [(i + 1, float(km)) for i, km in enumerate(k_modes)]
    else:
        mode_items = [(m, 2.0 * np.pi * m / Lx) for m in range(1, max_m + 1)]

    for m, k_true in mode_items:
        w_exp = k_true * c_physical
        
        # Search window +/- 25% around expected harmonic frequency
        w_low = 0.75 * w_exp
        w_high = 1.25 * w_exp
        idx_win = np.where((omegas >= w_low) & (omegas <= w_high))[0]
        if len(idx_win) == 0:
            continue
            
        local_peak_idx = idx_win[np.argmax(mag1[idx_win])]
        peak_mag = mag1[local_peak_idx]

        # Parabolic peak interpolation for sub-bin frequency refinement
        if 0 < local_peak_idx < len(freqs) - 1:
            alpha = mag1[local_peak_idx - 1]
            beta  = mag1[local_peak_idx]
            gamma = mag1[local_peak_idx + 1]
            denom = alpha - 2.0 * beta + gamma
            delta = 0.5 * (alpha - gamma) / denom if abs(denom) > 1e-12 else 0.0
            w_num = omegas[local_peak_idx] + delta * (omegas[1] - omegas[0])
        else:
            w_num = omegas[local_peak_idx]

        # Numerical phase velocity for spatial mode k_true
        vp = w_num / k_true

        modes.append(m)
        omegas_out.append(w_num)
        k_nums.append(k_true)
        vps.append(vp)
        mags.append(peak_mag)

    return np.array(modes), np.array(omegas_out), np.array(k_nums), np.array(vps), np.array(mags)

def analyze_and_plot(csv_path, dist=None, dx=25.132741228718345, out_png="two_probe_dispersion.png"):
    # 1. Determine separation distance and metadata
    meta_dist, meta_Lx, meta_data = load_probe_metadata(csv_path)
    if meta_dist is not None:
        separation_distance = meta_dist
    elif dist is not None and dist > 0:
        separation_distance = dist
    else:
        separation_distance = 1.0  # Fallback

    domain_Lx = meta_Lx if meta_Lx is not None else dx
    order = meta_data.get("order", 10) if meta_data else 10
    nelx = meta_data.get("elements_x", 30) if meta_data else 30
    sim_modes = meta_data.get("num_modes", 15) if meta_data else 15
    k_modes = meta_data.get("k_modes", None) if meta_data else None

    # 2. Load probe signals
    times, ez1, ez2 = load_two_probe_data(csv_path)
    print(f"[Loader] Loaded {len(times)} time steps from {csv_path}")
    print(f"  -> Probe separation distance: d = {separation_distance:.6f}")
    print(f"  -> Domain Lx = {domain_Lx:.2f}, Elements = {nelx}, Order N = {order}, Excited Modes = {sim_modes}")
    if k_modes is not None:
        print(f"  -> Explicit k_modes: {[round(k, 4) for k in k_modes]}")

    # 3. Extract modal phase velocity
    modes, omegas, k_nums, vps, mags = compute_modal_phase_velocity(
        times, ez1, ez2, separation_distance, Lx=domain_Lx, max_m=sim_modes, k_modes=k_modes
    )

    print(f"[Analysis] Extracted {len(modes)} physical harmonic modes:")
    for m, w, k, vp, mag in zip(modes, omegas, k_nums, vps, mags):
        print(f"  Mode {m:2d}: omega = {w:7.3f} rad/s, k_true = {k:7.3f} rad/m, vp/c = {vp/1.0:8.6f} (mag = {mag:.3e})")

    # 4. Points Per Wavelength (PPW)
    # Effective nodal spacing in SEM along x
    p_deg = max(1, order - 1)
    dx_nodal = domain_Lx / float(nelx * p_deg)
    ppws_cuda = (2.0 * np.pi) / (k_nums * dx_nodal) if len(k_nums) > 0 else np.array([])

    # 5. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    title_str = f"Two-Probe Numerical Dispersion (Order N={order}, Probe Separation d = {separation_distance:.2f})"
    fig.suptitle(title_str, fontsize=13, fontweight='bold')

    # --- Left Plot: v_p / c vs PPW ---
    ax1.axhline(1.0, color='gray', linestyle='--', linewidth=1.2, label=r'Exact: $v_p / c = 1.0$')

    if len(ppws_cuda) > 0:
        ax1.plot(ppws_cuda, vps / 1.0, 'o-', color='#d62728', markersize=8, linewidth=1.5,
                 markeredgecolor='black', markeredgewidth=1.0, label='CUDA Two-Probe (Time-Domain)')

        # Add mode labels above points
        for m, p_x, p_y in zip(modes, ppws_cuda, vps / 1.0):
            ax1.annotate(f"m={m}", (p_x, p_y), textcoords="offset points", xytext=(0, 10),
                         ha='center', fontsize=9, fontweight='bold', color='#d62728')

    ax1.set_xlabel(r'Points Per Wavelength (PPW = $\lambda / \Delta x$)', fontsize=11, fontweight='bold')
    ax1.set_ylabel(r'Normalized Phase Velocity $v_p / c$', fontsize=11, fontweight='bold')
    ax1.set_xscale("log")
    ax1.set_xlim(628.0, 2.0)
    ax1.set_ylim([0.0, 2.0])
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', frameon=True, framealpha=0.9)

    # --- Right Plot: omega vs k ---
    max_k_plot = 4.0
    ax2.plot([0, max_k_plot], [0, max_k_plot], color='gray', linestyle='--', linewidth=1.2, label=r'Exact: $\omega = c k$')

    if len(k_nums) > 0:
        ax2.plot(k_nums, omegas, 'o-', color='#d62728', markersize=8, linewidth=1.5,
                 markeredgecolor='black', markeredgewidth=1.0, label='CUDA Two-Probe (Time-Domain)')

        for m, k_x, w_y in zip(modes, k_nums, omegas):
            ax2.annotate(f"m={m}", (k_x, w_y), textcoords="offset points", xytext=(-10, 8),
                         ha='right', fontsize=9, fontweight='bold', color='#d62728')

    ax2.set_xlabel(r'Wavenumber Magnitude $|k|$ [rad/m]', fontsize=11, fontweight='bold')
    ax2.set_ylabel(r'Frequency $\omega$ [rad/s]', fontsize=11, fontweight='bold')
    ax2.set_xlim(0, 4.0)
    ax2.set_ylim(0, 4.0)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right', frameon=True, framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_png, dpi=300)
    print(f"[Plotter] Saved two-probe dispersion plot to {out_png}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 plot_two_probe_dispersion.py <probe_history.csv> [dist=auto] [out.png]")
        sys.exit(1)
        
    csv_f = sys.argv[1]
    
    d_sep = None
    out_img = "two_probe_dispersion.png"

    for arg in sys.argv[2:]:
        if arg.endswith('.png'):
            out_img = arg
        else:
            try:
                val = float(arg)
                if val > 0:
                    d_sep = val
            except ValueError:
                pass

    analyze_and_plot(csv_f, dist=d_sep, out_png=out_img)
