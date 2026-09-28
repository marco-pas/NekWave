#!/usr/bin/env python3
"""
NekWave: High-Fidelity Time-Domain Maxwell Solver Postprocessing

Generates a unified, publication-quality 3-panel analysis dashboard:
  1. Top-Left:  Relative discrete energy drift (U(t) - U0) / U0
  2. Top-Right: Centerline spatial field cut Ez(x) at t = 0 vs t = t_final
  3. Bottom:    Wide 1D multi-probe Fourier spectra with theoretical cavity
                eigenmode frequency markers.

Usage:
  python3 postprocess.py
  python3 postprocess.py --input-folder build/output --output-folder build/output
  python3 postprocess.py -i output -o output --case examples/cavity_gaussian/cavity_gaussian.par
"""

import sys
import os
import shutil
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Apply publication-quality typography and styling
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset":   "stix",
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "axes.unicode_minus": False,
    "font.size":          10,
    "axes.labelsize":     11,
    "axes.titlesize":     11,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.7,
    "ytick.major.width":  0.7,
    "xtick.minor.width":  0.4,
    "ytick.minor.width":  0.4,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.top":          True,
    "ytick.right":        True,
})

def parse_args():
    """Parses command-line arguments for input and output locations."""
    parser = argparse.ArgumentParser(
        description="NekWave postprocessing and modal analysis visualization utility."
    )
    parser.add_argument(
        "-i", "--input-folder", "--input-dir",
        dest="input_folder",
        type=str,
        default=None,
        help="Directory containing simulation CSV files (energy_history.csv, field_final.csv, etc.)."
    )
    parser.add_argument(
        "-o", "--output-folder", "--output-dir",
        dest="output_folder",
        type=str,
        default=None,
        help="Directory where the generated plot will be saved."
    )
    parser.add_argument(
        "-c", "--case", "--case-file", "--par",
        dest="case_file",
        type=str,
        default=None,
        help="Optional path to parameter file (.par) for cavity dimensions and wave speed."
    )
    parser.add_argument(
        "-f", "--filename",
        dest="filename",
        type=str,
        default="nekwave_dashboard.png",
        help="Name of the output figure file (default: nekwave_dashboard.png)."
    )
    return parser.parse_args()

def find_file(filename, input_dir=None):
    """Searches for file in the specified input directory, default locations, or parent directories."""
    candidates = []
    if input_dir:
        candidates.append(os.path.join(input_dir, filename))

    candidates.extend([
        os.path.join("output", filename),
        os.path.join("build", "output", filename),
        os.path.join("..", "output", filename),
        os.path.join("..", "build", "output", filename),
        os.path.join("examples", "cavity_gaussian", "output", filename),
        os.path.join("..", "examples", "cavity_gaussian", "output", filename),
        filename,
        os.path.join("..", filename),
    ])

    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def resolve_output_dir(output_dir=None, input_dir=None):
    """Determines the target directory for saving generated plots."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    if input_dir and os.path.isdir(input_dir):
        return input_dir

    candidates = [
        "output",
        os.path.join("build", "output"),
        os.path.join("..", "output"),
        os.path.join("..", "build", "output"),
        os.path.join("examples", "cavity_gaussian", "output"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    os.makedirs("output", exist_ok=True)
    return "output"

def load_case_params(case_file=None, input_dir=None):
    """Parses simulation parameters from case parameter file."""
    par_file = None
    if case_file and os.path.exists(case_file):
        par_file = case_file
    elif input_dir:
        for fname in os.listdir(input_dir):
            if fname.endswith(".par"):
                par_file = os.path.join(input_dir, fname)
                break

    if not par_file:
        for candidate in ["case.par", "examples/cavity_gaussian/cavity_gaussian.par",
                          "../examples/cavity_gaussian/cavity_gaussian.par", "cavity_gaussian.par"]:
            found = find_file(candidate, input_dir) or (candidate if os.path.exists(candidate) else None)
            if found:
                par_file = found
                break

    params = {
        'Lx': 2.0, 'Ly': 2.0, 'Lz': 2.0,
        'c_wave': 1.0,
        'C0_flux': 0.0,
    }

    if not par_file:
        params['xmin'] = -params['Lx'] / 2.0
        params['xmax'] =  params['Lx'] / 2.0
        params['ymin'] = -params['Ly'] / 2.0
        params['ymax'] =  params['Ly'] / 2.0
        params['zmin'] = -params['Lz'] / 2.0
        params['zmax'] =  params['Lz'] / 2.0
        params['xc'] = 0.0
        params['yc'] = 0.0
        params['zc'] = 0.0
        return params

    print(f"[NekWave Postprocess] Using parameter configuration: {par_file}")
    try:
        with open(par_file, 'r') as f:
            for line in f:
                if '#' in line:
                    line = line.split('#', 1)[0]
                line = line.strip()
                if not line or ('=' not in line and ':' not in line):
                    continue
                sep = '=' if '=' in line else ':'
                k, v = line.split(sep, 1)
                k = k.strip().lower()
                v = v.strip()
                if k in ['l_x', 'lx', 'length_x']: params['Lx'] = float(v)
                elif k in ['l_y', 'ly', 'length_y']: params['Ly'] = float(v)
                elif k in ['l_z', 'lz', 'length_z']: params['Lz'] = float(v)
                elif k == 'xmin': params['xmin'] = float(v)
                elif k == 'xmax': params['xmax'] = float(v)
                elif k == 'ymin': params['ymin'] = float(v)
                elif k == 'ymax': params['ymax'] = float(v)
                elif k == 'zmin': params['zmin'] = float(v)
                elif k == 'zmax': params['zmax'] = float(v)
                elif k in ['wavespeed', 'wave_speed', 'c']: params['c_wave'] = float(v)
                elif k in ['c0', 'flux_c0', 'flux_penalty']:
                    try:
                        params['C0_flux'] = float(v)
                    except ValueError:
                        if v.lower() == 'central': params['C0_flux'] = 0.0
                        elif v.lower() == 'upwind': params['C0_flux'] = 1.0
    except Exception as e:
        print(f"[NekWave Postprocess] Warning reading parameter file: {e}")

    if 'xmin' not in params or 'xmax' not in params:
        params['xmin'] = -params['Lx'] / 2.0
        params['xmax'] =  params['Lx'] / 2.0
    else:
        params['Lx'] = params['xmax'] - params['xmin']

    if 'ymin' not in params or 'ymax' not in params:
        params['ymin'] = -params['Ly'] / 2.0
        params['ymax'] =  params['Ly'] / 2.0
    else:
        params['Ly'] = params['ymax'] - params['ymin']

    if 'zmin' not in params or 'zmax' not in params:
        params['zmin'] = -params['Lz'] / 2.0
        params['zmax'] =  params['Lz'] / 2.0
    else:
        params['Lz'] = params['zmax'] - params['zmin']

    params['xc'] = 0.5 * (params['xmin'] + params['xmax'])
    params['yc'] = 0.5 * (params['ymin'] + params['ymax'])
    params['zc'] = 0.5 * (params['zmin'] + params['zmax'])

    return params

def load_csv_data(filepath):
    """Loads CSV file while properly skipping comments and headers."""
    if not filepath or not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        print(f"[NekWave Postprocess] Warning: {filepath} is empty or not found.")
        return None

    print(f"[NekWave Postprocess] Loading: {filepath}")
    col_names = None
    header_line_idx = -1
    with open(filepath, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            col_names = [col.strip() for col in line.split(',')]
            header_line_idx = idx
            break

    if not col_names:
        print(f"[NekWave Postprocess] Warning: Could not find valid CSV header in {filepath}")
        return None

    data = np.genfromtxt(filepath, delimiter=',', comments='#', skip_header=header_line_idx + 1)
    if data.size == 0:
        return None
    if data.ndim == 1:
        data = data.reshape(1, -1)
    data_dict = {}
    for idx, name in enumerate(col_names):
        data_dict[name] = data[:, idx]
    return data_dict

def generate_dashboard(args=None):
    """Generates a single unified analysis figure containing Energy, Centerline cut, and 1D FFT modes."""
    input_dir = args.input_folder if args else None
    output_dir = args.output_folder if args else None
    case_file = args.case_file if args else None
    filename = args.filename if args and args.filename else "nekwave_dashboard.png"

    energy_file = find_file("energy_history.csv", input_dir)
    field_init_file = find_file("field_initial.csv", input_dir)
    field_final_file = find_file("field_final.csv", input_dir)
    probe_file = find_file("probe_history.csv", input_dir)

    if not energy_file:
        print("[Error] Required energy_history.csv not found.")
        if input_dir:
            print(f"  Looked inside: {input_dir}")
        return

    energy_data = load_csv_data(energy_file)
    if not energy_data:
        print(f"[Error] Failed to read energy data from {energy_file}")
        return

    f_init = load_csv_data(field_init_file) if field_init_file else None
    f_final = load_csv_data(field_final_file) if field_final_file else None

    time_hist = energy_data['time']
    energy_hist = energy_data['energy']
    params = load_case_params(case_file, input_dir)
    Lx, Ly, Lz = params['Lx'], params['Ly'], params['Lz']
    xmin, xmax = params['xmin'], params['xmax']
    ymin, ymax = params['ymin'], params['ymax']
    zmin, zmax = params['zmin'], params['zmax']
    xc, yc, zc = params['xc'], params['yc'], params['zc']
    c_wave = params.get('c_wave', 1.0)

    e0 = energy_hist[0]
    drift = (energy_hist - e0) / e0 if e0 != 0 else np.zeros_like(energy_hist)

    has_fields = (f_init is not None and f_final is not None and 
                  'x' in f_init and 'x' in f_final and 'Ez' in f_init and 'Ez' in f_final)

    if has_fields:
        fig = plt.figure(figsize=(11, 8), dpi=150)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.38, wspace=0.25)
        ax_energy = fig.add_subplot(gs[0, 0])
        ax_cut = fig.add_subplot(gs[0, 1])
        ax_fft = fig.add_subplot(gs[1, :])
    else:
        # Clean 2-panel layout focused on Energy Conservation and Cavity Resonant Modes
        fig = plt.figure(figsize=(10, 7.5), dpi=150)
        gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.3], hspace=0.35)
        ax_energy = fig.add_subplot(gs[0])
        ax_cut = None
        ax_fft = fig.add_subplot(gs[1])

    # -------------------------------------------------------------------------
    # Panel 1: Relative Energy Drift
    # -------------------------------------------------------------------------
    ax_energy.axhline(0.0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax_energy.plot(time_hist, drift, '-', color='#1f77b4', linewidth=1.5, label='$(U(t)-U_0)/U_0$')
    ax_energy.set_xlabel('Time $t$')
    ax_energy.set_ylabel('$(U(t) - U_0) / U_0$')
    ax_energy.set_title('Relative Energy Conservation Drift')
    ax_energy.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax_energy.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
    ax_energy.grid(True, linestyle=':', alpha=0.5)

    max_abs_drift = float(np.max(np.abs(drift))) if len(drift) > 0 else 0.0
    final_drift = float(drift[-1]) if len(drift) > 0 else 0.0
    drift_text = f"$U_0 = {e0:.6e}$\nMax $|\\Delta U/U_0| = {max_abs_drift:.2e}$\nFinal $\\Delta U/U_0 = {final_drift:+.2e}$"
    ax_energy.text(0.02, 0.95, drift_text, transform=ax_energy.transAxes, verticalalignment='top',
                   fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='#cccccc'))
    ax_energy.legend(loc='best', frameon=False)

    # -------------------------------------------------------------------------
    # Panel 2: Centerline Spatial Field Cut Ez(x) (if field data present)
    # -------------------------------------------------------------------------
    if ax_cut is not None:
        dist_init = np.hypot(f_init['y'] - yc, f_init['z'] - zc)
        tol_r = 0.08 * min(Ly, Lz)
        mask_init = dist_init <= tol_r
        if np.sum(mask_init) < 20:
            idx_c = np.argsort(dist_init)[:max(50, int(0.02 * len(dist_init)))]
            mask_init = np.zeros(len(dist_init), dtype=bool)
            mask_init[idx_c] = True

        dist_final = np.hypot(f_final['y'] - yc, f_final['z'] - zc)
        mask_final = dist_final <= tol_r
        if np.sum(mask_final) < 20:
            idx_c = np.argsort(dist_final)[:max(50, int(0.02 * len(dist_final)))]
            mask_final = np.zeros(len(dist_final), dtype=bool)
            mask_final[idx_c] = True

        x_cut_init = f_init['x'][mask_init]
        Ez_cut_init = f_init['Ez'][mask_init]
        order_init = np.argsort(x_cut_init)

        x_cut_final = f_final['x'][mask_final]
        Ez_cut_final = f_final['Ez'][mask_final]
        order_final = np.argsort(x_cut_final)

        t_final = time_hist[-1] if len(time_hist) > 0 else 0.0
        ax_cut.plot(x_cut_init[order_init], Ez_cut_init[order_init], 'k--', linewidth=1.2, label='Initial ($t = 0$)')
        ax_cut.plot(x_cut_final[order_final], Ez_cut_final[order_final], 'r-', linewidth=1.2, label=f'Final ($t = {t_final:.2f}$)')
        dx_margin = 0.04 * Lx
        ax_cut.set_xlim([xmin - dx_margin, xmax + dx_margin])
        ax_cut.legend(loc='best', frameon=False)
        ax_cut.set_xlabel('Spatial coordinate $x$')
        ax_cut.set_ylabel('Electric field $E_z$')
        ax_cut.set_title('Centerline Field Cut ($y = y_c, z = z_c$)')
        ax_cut.grid(True, linestyle=':', alpha=0.5)

    # -------------------------------------------------------------------------
    # Panel 3: Wide 1D Multi-Probe Fourier Spectrum

    if probe_file:
        probe_data = load_csv_data(probe_file)
        time = probe_data['time']

        probe_indices = []
        for k in range(1, 10):
            if f'Ez_{k}' in probe_data:
                probe_indices.append(k)
        if not probe_indices and 'Ez' in probe_data:
            probe_indices = [1]
            probe_data['Ez_1'] = probe_data['Ez']

        N = len(time)
        if N >= 4:
            dt_step = (time[-1] - time[0]) / (N - 1)
            freqs = np.fft.rfftfreq(N, d=dt_step)

            # Calculate theoretical cavity eigenmodes: f = (c/2) * sqrt((m/Lx)^2 + (n/Ly)^2 + (p/Lz)^2)
            f_fund = (c_wave / 2.0) * np.sqrt((1.0 / Lx)**2 + (1.0 / Ly)**2)
            mode_dict = {}
            for m in range(1, 6):
                for n in range(1, 6):
                    for p in range(0, 4):
                        f_m = (c_wave / 2.0) * np.sqrt((m / Lx)**2 + (n / Ly)**2 + (p / Lz)**2)
                        f_key = round(float(f_m), 5)
                        if f_key not in mode_dict:
                            mode_dict[f_key] = []
                        mode_dict[f_key].append((m, n, p))

            sorted_modes = sorted(mode_dict.items(), key=lambda x: x[0])
            distinct_modes = []
            theory_freqs = []
            for f_val, trips in sorted_modes:
                if len(trips) == 1:
                    lbl = f"({trips[0][0]},{trips[0][1]},{trips[0][2]})"
                else:
                    first_two = [f"({t[0]},{t[1]},{t[2]})" for t in trips[:2]]
                    lbl = "/".join(first_two)
                distinct_modes.append(([trips[0][0], trips[0][1], trips[0][2]], lbl))
                theory_freqs.append(f_val)

            theory_freqs = np.array(theory_freqs)
            probe_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

            all_spec = np.zeros_like(freqs)
            for i, k in enumerate(probe_indices):
                Ez_k = probe_data[f'Ez_{k}']
                Ez_ac = Ez_k - np.mean(Ez_k)
                fft_k = np.fft.rfft(Ez_ac)
                spec_k = np.abs(fft_k)
                max_s = np.max(spec_k)
                if max_s > 0:
                    spec_k /= max_s
                all_spec = np.maximum(all_spec, spec_k)

            # Determine adaptive frequency plotting range
            active_mask = (all_spec >= 0.04) & (freqs > 0)
            if np.any(active_mask):
                f_active_max = np.max(freqs[active_mask])
                max_plot_freq = max(f_active_max * 1.18, 3.0 * f_fund)
            else:
                max_plot_freq = 4.0 * f_fund
            max_plot_freq = min(freqs[-1], max_plot_freq)
            freq_mask = freqs <= max_plot_freq

            # Plot spectrum for each observation probe
            for i, k in enumerate(probe_indices):
                Ez_k = probe_data[f'Ez_{k}']
                Ez_ac = Ez_k - np.mean(Ez_k)
                fft_k = np.fft.rfft(Ez_ac)
                spec_k = np.abs(fft_k)
                max_s = np.max(spec_k)
                if max_s > 0:
                    spec_k /= max_s

                col = probe_colors[i % len(probe_colors)]
                ax_fft.plot(freqs[freq_mask], spec_k[freq_mask], color=col, linewidth=1.2, alpha=0.85, label=f'Probe {k}')
                if i == 0:
                    ax_fft.fill_between(freqs[freq_mask], spec_k[freq_mask], color=col, alpha=0.10)

            # Filter adjacent modes to avoid label overlap
            df_min = 0.035 * max_plot_freq
            filtered_modes = []
            for m_tuple, f_th in zip(distinct_modes, theory_freqs):
                if f_th > max_plot_freq:
                    continue
                if not filtered_modes:
                    filtered_modes.append((m_tuple, f_th))
                elif (f_th - filtered_modes[-1][1]) >= df_min:
                    filtered_modes.append((m_tuple, f_th))

            theory_colors = ['#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#e6550d', '#31a354']
            for idx, ((_, label_str), f_th) in enumerate(filtered_modes):
                col = theory_colors[idx % len(theory_colors)]
                ax_fft.axvline(f_th, color=col, linestyle='--', linewidth=1.0, alpha=0.75)
                y_text = 0.32 + 0.22 * (idx % 3)
                ax_fft.text(f_th, y_text, f' {label_str}', color=col, fontsize=8, rotation=90,
                            verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.75, edgecolor='none'))

            ax_fft.set_xlim([0.0, max_plot_freq])
            ax_fft.set_ylim([0.0, 1.25])
            ax_fft.set_xlabel('Frequency $f$ [cycles / time]')
            ax_fft.set_ylabel('Normalized Amplitude $|\\hat{E}_z(f)|$')
            ax_fft.set_title('Cavity Resonant Mode Frequency Spectrum (1D Multi-Probe FFT)')
            ax_fft.legend(loc='best', frameon=False, ncol=len(probe_indices))
            ax_fft.grid(True, linestyle=':', alpha=0.5)
        else:
            ax_fft.text(0.5, 0.5, 'Insufficient probe time samples for FFT', ha='center', va='center')
    else:
        ax_fft.text(0.5, 0.5, 'No probe_history.csv found', ha='center', va='center')

    target_dir = resolve_output_dir(output_dir, input_dir)
    out_file = os.path.join(target_dir, filename)
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"[NekWave Postprocess] Master dashboard saved: {out_file}")

    # -------------------------------------------------------------------------
    # Terminal Summary Report: Energy Drift & Cavity Resonant Modes
    # -------------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("                    NekWave Simulation Analysis Report")
    print("=" * 76)
    print(f" Initial Energy U_0:      {e0:.8e}")
    print(f" Final Energy U_f:        {energy_hist[-1]:.8e}")
    print(f" Maximum Relative Drift:  {max_abs_drift:.4e}")
    print(f" Final Relative Drift:    {final_drift:+.4e}")
    print("-" * 76)
    print(f"{'Step':>8} | {'Time':>10} | {'Total Energy':>18} | {'Relative Drift':>18}")
    print("-" * 76)
    step_hist = energy_data.get('step', np.arange(len(time_hist)))
    num_pts = len(time_hist)
    # Print up to 15 evenly spaced points plus initial and final
    stride = max(1, num_pts // 12)
    printed_indices = set(range(0, num_pts, stride))
    printed_indices.add(num_pts - 1)
    for idx in sorted(printed_indices):
        s_val = int(step_hist[idx]) if idx < len(step_hist) else idx
        print(f"{s_val:8d} | {time_hist[idx]:10.4f} | {energy_hist[idx]:18.8e} | {drift[idx]:+18.4e}")
    print("-" * 76)

    # Cavity Resonant Modes
    if probe_file and 'distinct_modes' in locals() and len(theory_freqs) > 0:
        print("\n Cavity Resonant Mode Spectrum (Detected FFT Peaks vs. Theory):")
        print(f"{'Peak Freq (f)':>14} | {'Theory Freq':>12} | {'Cavity Mode (m,n,p)':>20} | {'Error (%)':>10}")
        print("-" * 76)

        # Detect local peaks in combined spectrum
        peaks = []
        for p_i in range(1, len(freqs) - 1):
            if freq_mask[p_i] and freqs[p_i] > 0.05 and all_spec[p_i] > 0.05:
                if all_spec[p_i] > all_spec[p_i - 1] and all_spec[p_i] > all_spec[p_i + 1]:
                    peaks.append((freqs[p_i], all_spec[p_i]))
        peaks.sort(key=lambda x: x[1], reverse=True)

        for f_pk, amp in peaks[:8]:
            dists = np.abs(theory_freqs - f_pk)
            b_idx = np.argmin(dists)
            f_th = theory_freqs[b_idx]
            m_lbl = distinct_modes[b_idx][1]
            err = 100.0 * abs(f_pk - f_th) / f_th if f_th > 0 else 0.0
            print(f"{f_pk:14.5f} | {f_th:12.5f} | {m_lbl:>20} | {err:9.2f}%")
        print("=" * 76 + "\n")

    # Copy to agent artifact directory if available
    artifact_dirs = [
        "/afs/pdc.kth.se/home/m/marcopas/.gemini/antigravity/brain/e9a93cc1-19fc-435d-be1d-a448fc998657",
        os.environ.get("ANTIGRAVITY_ARTIFACT_DIR", "")
    ]
    for ad in artifact_dirs:
        if ad and os.path.isdir(ad):
            dest = os.path.join(ad, filename)
            shutil.copyfile(out_file, dest)
            print(f"[NekWave Postprocess] Synced to agent artifact: {dest}")

if __name__ == "__main__":
    cli_args = parse_args()
    generate_dashboard(cli_args)