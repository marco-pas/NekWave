#!/usr/bin/env python3
"""
NekWave: Resonant Cavity Modal & Energy Drift Analysis Tool

Computes:
  1. Discrete relative energy drift: (U(t) - U_0) / U_0
  2. Multi-probe FFT frequency spectrum vs. analytical 3D PEC cavity eigenmodes:
       f_{mnp} = (c / 2) * sqrt((m / L_x)^2 + (n / L_y)^2 + (p / L_z)^2)
  3. Publication-quality multi-panel visualization dashboard (PNG).

Usage examples:
  python3 analyze_cavity.py --input-folder ./output --output-folder ./output
  python3 ../tools/analyze_cavity.py -i build_cuda/output -o build_cuda/output
  python3 ../tools/analyze_cavity.py output
"""

import sys
import os
import io
import csv
import math
import cmath
import shutil
import argparse

# Check for numpy and matplotlib availability
HAVE_NUMPY = False
HAVE_MATPLOTLIB = False

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    pass

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAVE_MATPLOTLIB = True
except ImportError:
    pass


# -----------------------------------------------------------------------------
# CLI Arguments Parser
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="NekWave cavity resonant mode frequency and energy drift analysis tool."
    )
    parser.add_argument(
        "pos_input",
        nargs="?",
        default=None,
        help="Optional positional argument for input folder (e.g. ./output)."
    )
    parser.add_argument(
        "-i", "--input-folder", "--input-dir",
        dest="input_folder",
        type=str,
        default=None,
        help="Directory containing simulation CSV files (energy_history.csv, probe_history.csv)."
    )
    parser.add_argument(
        "-o", "--output-folder", "--output-dir",
        dest="output_folder",
        type=str,
        default=None,
        help="Directory where the analysis dashboard plot will be saved."
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
    args = parser.parse_args()
    if not args.input_folder and args.pos_input:
        args.input_folder = args.pos_input
    if not args.output_folder and args.input_folder:
        args.output_folder = args.input_folder
    return args


# -----------------------------------------------------------------------------
# File Resolution and Parameter Loading
# -----------------------------------------------------------------------------
def find_file(filename, input_dir=None):
    candidates = []
    if input_dir:
        candidates.append(os.path.join(input_dir, filename))
    candidates.extend([
        os.path.join("output", filename),
        os.path.join("build_cuda", "output", filename),
        os.path.join("build", "output", filename),
        os.path.join("..", "output", filename),
        os.path.join("..", "build_cuda", "output", filename),
        os.path.join("..", "build", "output", filename),
        os.path.join("examples", "cavity_gaussian", "output", filename),
        filename,
        os.path.join("..", filename),
    ])
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def resolve_output_dir(output_dir=None, input_dir=None):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    if input_dir and os.path.isdir(input_dir):
        return input_dir
    for c in ["output", "build_cuda/output", "build/output"]:
        if os.path.isdir(c):
            return c
    os.makedirs("output", exist_ok=True)
    return "output"


def load_case_params(case_file=None, input_dir=None):
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

    print(f"[NekWave Analyze] Using parameter configuration: {par_file}")
    try:
        with open(par_file, 'r', encoding='utf-8', errors='ignore') as f:
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
        print(f"[NekWave Analyze] Warning reading parameter file: {e}")

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


# -----------------------------------------------------------------------------
# Robust CSV Reader (Sanitizes Null Bytes and Incomplete Lines)
# -----------------------------------------------------------------------------
def load_csv_data(filepath):
    if not filepath or not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return None

    print(f"[NekWave Analyze] Loading: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            clean_text = f.read().replace('\x00', '')
    except Exception as e:
        print(f"[NekWave Analyze] Error reading {filepath}: {e}")
        return None

    lines = clean_text.splitlines()
    header_idx = -1
    col_names = None
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        col_names = [col.strip() for col in line.split(',')]
        header_idx = idx
        break

    if not col_names:
        return None

    data_cols = {col: [] for col in col_names}
    for line in lines[header_idx + 1:]:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split(',')
        if len(parts) != len(col_names):
            continue
        try:
            for col, val_str in zip(col_names, parts):
                data_cols[col].append(float(val_str.strip()))
        except ValueError:
            continue

    if HAVE_NUMPY:
        for k in data_cols:
            data_cols[k] = np.array(data_cols[k])

    return data_cols


# -----------------------------------------------------------------------------
# Radix-2 Cooley-Tukey FFT (Pure Python Fallback)
# -----------------------------------------------------------------------------
def pure_fft(x):
    n = len(x)
    if n <= 1:
        return x
    even = pure_fft(x[0::2])
    odd = pure_fft(x[1::2])
    factor = [cmath.exp(-2j * math.pi * k / n) * odd[k] for k in range(n // 2)]
    return [even[k] + factor[k] for k in range(n // 2)] + [even[k] - factor[k] for k in range(n // 2)]


# -----------------------------------------------------------------------------
# Main Analysis & Dashboard Generator
# -----------------------------------------------------------------------------
def analyze(args=None):
    if args is None:
        args = parse_args()

    input_dir = args.input_folder
    output_dir = args.output_folder
    case_file = args.case_file
    filename = args.filename if args.filename else "nekwave_dashboard.png"

    energy_file = find_file("energy_history.csv", input_dir)
    field_init_file = find_file("field_initial.csv", input_dir)
    field_final_file = find_file("field_final.csv", input_dir)
    probe_file = find_file("probe_history.csv", input_dir)

    if not energy_file:
        print("[Error] Required energy_history.csv not found.")
        if input_dir:
            print(f"  Searched inside: {input_dir}")
        return

    energy_data = load_csv_data(energy_file)
    if not energy_data or len(energy_data.get('time', [])) == 0:
        print(f"[Error] Failed to read valid energy data from {energy_file}")
        return

    f_init = load_csv_data(field_init_file) if field_init_file else None
    f_final = load_csv_data(field_final_file) if field_final_file else None

    time_hist = list(energy_data['time'])
    energy_hist = list(energy_data['energy'])
    step_hist = list(energy_data.get('step', range(len(time_hist))))

    params = load_case_params(case_file, input_dir)
    Lx, Ly, Lz = params['Lx'], params['Ly'], params['Lz']
    xmin, xmax = params['xmin'], params['xmax']
    ymin, ymax = params['ymin'], params['ymax']
    zmin, zmax = params['zmin'], params['zmax']
    xc, yc, zc = params['xc'], params['yc'], params['zc']
    c_wave = params.get('c_wave', 1.0)

    e0 = energy_hist[0]
    drift = [(e - e0) / e0 if e0 != 0 else 0.0 for e in energy_hist]
    max_abs_drift = max(abs(d) for d in drift) if drift else 0.0
    final_drift = drift[-1] if drift else 0.0

    # -------------------------------------------------------------------------
    # Terminal Summary Report: Energy Drift
    # -------------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("           NekWave Resonant Cavity & Energy Drift Analysis")
    print("=" * 76)
    print(f" Initial Total Energy U_0:  {e0:.8e}")
    print(f" Final Total Energy U_f:    {energy_hist[-1]:.8e}")
    print(f" Maximum Relative Drift:    {max_abs_drift:.4e}")
    print(f" Final Relative Drift:      {final_drift:+.4e}")
    print("-" * 76)
    print(f"{'Step':>8} | {'Time t':>10} | {'Total Energy U(t)':>20} | {'Relative Drift':>20}")
    print("-" * 76)
    num_pts = len(time_hist)
    stride = max(1, num_pts // 12)
    sample_indices = sorted(list(set(list(range(0, num_pts, stride)) + [num_pts - 1])))
    for idx in sample_indices:
        s_val = int(step_hist[idx]) if idx < len(step_hist) else idx
        print(f"{s_val:8d} | {time_hist[idx]:10.4f} | {energy_hist[idx]:20.8e} | {drift[idx]:+20.4e}")
    print("-" * 76)

    # -------------------------------------------------------------------------
    # Theoretical Cavity Resonant Modes
    # -------------------------------------------------------------------------
    theoretical_modes = {}
    distinct_modes = []
    theory_freqs = []
    for m in range(1, 6):
        for n in range(1, 6):
            for p in range(0, 4):
                f_m = (c_wave / 2.0) * math.sqrt((m / Lx)**2 + (n / Ly)**2 + (p / Lz)**2)
                f_key = round(float(f_m), 5)
                if f_key not in theoretical_modes:
                    theoretical_modes[f_key] = []
                theoretical_modes[f_key].append((m, n, p))

    for f_val in sorted(theoretical_modes.keys()):
        trips = theoretical_modes[f_val]
        if len(trips) == 1:
            lbl = f"({trips[0][0]},{trips[0][1]},{trips[0][2]})"
        else:
            first_two = [f"({t[0]},{t[1]},{t[2]})" for t in trips[:2]]
            lbl = "/".join(first_two)
        distinct_modes.append(([trips[0][0], trips[0][1], trips[0][2]], lbl))
        theory_freqs.append(f_val)

    # -------------------------------------------------------------------------
    # Multi-Probe Fourier Spectrum Analysis
    # -------------------------------------------------------------------------
    probe_data = load_csv_data(probe_file) if probe_file else None
    probe_indices = []
    freqs = []
    all_spec = []
    peak_report = []

    if probe_data and 'time' in probe_data and len(probe_data['time']) >= 4:
        p_times = list(probe_data['time'])
        for k in range(1, 10):
            if f'Ez_{k}' in probe_data:
                probe_indices.append(k)
        if not probe_indices and 'Ez' in probe_data:
            probe_indices = [1]
            probe_data['Ez_1'] = probe_data['Ez']

        n_samples = len(p_times)
        if HAVE_NUMPY:
            dt_step = (p_times[-1] - p_times[0]) / (n_samples - 1)
            freqs = np.fft.rfftfreq(n_samples, d=dt_step)
            all_spec = np.zeros_like(freqs)
            probe_specs = {}

            for k in probe_indices:
                Ez_k = np.array(probe_data[f'Ez_{k}'])
                Ez_ac = Ez_k - np.mean(Ez_k)
                spec_k = np.abs(np.fft.rfft(Ez_ac))
                max_s = np.max(spec_k)
                if max_s > 0:
                    spec_k /= max_s
                probe_specs[k] = spec_k
                all_spec = np.maximum(all_spec, spec_k)

            # Detect local peaks
            f_fund = (c_wave / 2.0) * math.sqrt((1.0 / Lx)**2 + (1.0 / Ly)**2)
            active_mask = (all_spec >= 0.04) & (freqs > 0)
            f_active_max = float(np.max(freqs[active_mask])) if np.any(active_mask) else 4.0 * f_fund
            max_plot_freq = min(freqs[-1], max(f_active_max * 1.18, 3.0 * f_fund))
            freq_mask = freqs <= max_plot_freq

            raw_peaks = []
            for p_i in range(1, len(freqs) - 1):
                if freq_mask[p_i] and freqs[p_i] > 0.05 and all_spec[p_i] > 0.05:
                    if all_spec[p_i] > all_spec[p_i - 1] and all_spec[p_i] > all_spec[p_i + 1]:
                        raw_peaks.append((float(freqs[p_i]), float(all_spec[p_i])))
            raw_peaks.sort(key=lambda x: x[1], reverse=True)

            for f_pk, amp in raw_peaks[:8]:
                dists = np.abs(np.array(theory_freqs) - f_pk)
                b_idx = int(np.argmin(dists))
                f_th = theory_freqs[b_idx]
                m_lbl = distinct_modes[b_idx][1]
                err = 100.0 * abs(f_pk - f_th) / f_th if f_th > 0 else 0.0
                peak_report.append((f_pk, amp, f_th, m_lbl, err))

        else:
            # Pure Python FFT fallback
            n_fft = 1
            while n_fft * 2 <= n_samples:
                n_fft *= 2
            dt_step = (p_times[n_fft - 1] - p_times[0]) / (n_fft - 1)
            df = 1.0 / (n_fft * dt_step)
            freqs = [k * df for k in range(n_fft // 2)]
            all_spec = [0.0] * (n_fft // 2)

            for k in probe_indices:
                sig = list(probe_data[f'Ez_{k}'])[:n_fft]
                mean_v = sum(sig) / n_fft
                sig_ac = [v - mean_v for v in sig]
                spec_k = [abs(v) for v in pure_fft(sig_ac)[:n_fft // 2]]
                max_s = max(spec_k) if max(spec_k) > 0 else 1.0
                for i in range(len(all_spec)):
                    all_spec[i] = max(all_spec[i], spec_k[i] / max_s)

            for p_i in range(1, len(freqs) - 1):
                if 0.05 < freqs[p_i] < 2.5 and all_spec[p_i] > 0.05:
                    if all_spec[p_i] > all_spec[p_i - 1] and all_spec[p_i] > all_spec[p_i + 1]:
                        best_f = min(theory_freqs, key=lambda ft: abs(ft - freqs[p_i]))
                        b_idx = theory_freqs.index(best_f)
                        m_lbl = distinct_modes[b_idx][1]
                        err = 100.0 * abs(freqs[p_i] - best_f) / best_f if best_f > 0 else 0.0
                        peak_report.append((freqs[p_i], all_spec[p_i], best_f, m_lbl, err))
            peak_report.sort(key=lambda x: x[1], reverse=True)
            peak_report = peak_report[:8]

    if peak_report:
        print("\n Cavity Resonant Mode Spectrum (Detected FFT Peaks vs. Theory):")
        print(f"{'Detected (f)':>14} | {'Amplitude':>10} | {'Theory (f)':>12} | {'Mode (m,n,p)':>20} | {'Error (%)':>10}")
        print("-" * 76)
        for f_pk, amp, f_th, m_lbl, err in peak_report:
            print(f"{f_pk:14.5f} | {amp:10.4f} | {f_th:12.5f} | {m_lbl:>20} | {err:9.2f}%")
        print("=" * 76 + "\n")

    # -------------------------------------------------------------------------
    # Visualization Dashboard Plot
    # -------------------------------------------------------------------------
    if not HAVE_MATPLOTLIB:
        print("[NekWave Analyze] Notice: matplotlib not found. Skipping figure generation.")
        return

    # Configure publication-grade styling
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

    has_fields = (f_init is not None and f_final is not None and 
                  'x' in f_init and 'x' in f_final and 'Ez' in f_init and 'Ez' in f_final)

    if has_fields:
        fig = plt.figure(figsize=(11, 8), dpi=150)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.38, wspace=0.25)
        ax_energy = fig.add_subplot(gs[0, 0])
        ax_cut = fig.add_subplot(gs[0, 1])
        ax_fft = fig.add_subplot(gs[1, :])
    else:
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
    # Panel 3: Multi-Probe Fourier Spectrum
    # -------------------------------------------------------------------------
    if probe_data and len(probe_indices) > 0 and len(freqs) > 0:
        probe_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        f_fund = (c_wave / 2.0) * math.sqrt((1.0 / Lx)**2 + (1.0 / Ly)**2)
        active_mask = (np.array(all_spec) >= 0.04) & (np.array(freqs) > 0)
        f_active_max = float(np.max(np.array(freqs)[active_mask])) if np.any(active_mask) else 4.0 * f_fund
        max_plot_freq = min(freqs[-1], max(f_active_max * 1.18, 3.0 * f_fund))
        freq_mask = np.array(freqs) <= max_plot_freq

        for i, k in enumerate(probe_indices):
            if HAVE_NUMPY and k in probe_specs:
                spec_k = probe_specs[k]
            else:
                spec_k = all_spec
            col = probe_colors[i % len(probe_colors)]
            ax_fft.plot(np.array(freqs)[freq_mask], np.array(spec_k)[freq_mask],
                        color=col, linewidth=1.2, alpha=0.85, label=f'Probe {k}')
            if i == 0:
                ax_fft.fill_between(np.array(freqs)[freq_mask], np.array(spec_k)[freq_mask],
                                    color=col, alpha=0.10)

        # Plot analytical mode lines
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
                        verticalalignment='bottom',
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.75, edgecolor='none'))

        ax_fft.set_xlim([0.0, max_plot_freq])
        ax_fft.set_ylim([0.0, 1.25])
        ax_fft.set_xlabel('Frequency $f$ [cycles / time]')
        ax_fft.set_ylabel('Normalized Amplitude $|\\hat{E}_z(f)|$')
        ax_fft.set_title('Cavity Resonant Mode Frequency Spectrum (1D Multi-Probe FFT)')
        ax_fft.legend(loc='upper right', frameon=False, ncol=len(probe_indices))
        ax_fft.grid(True, linestyle=':', alpha=0.5)
    else:
        ax_fft.text(0.5, 0.5, 'No probe data found for FFT spectrum', ha='center', va='center')

    target_dir = resolve_output_dir(output_dir, input_dir)
    out_file = os.path.join(target_dir, filename)
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"[NekWave Analyze] Master dashboard saved: {out_file}")

    # Sync to agent artifact directory if present
    artifact_dirs = [
        "/afs/pdc.kth.se/home/m/marcopas/.gemini/antigravity/brain/e9a93cc1-19fc-435d-be1d-a448fc998657",
        os.environ.get("ANTIGRAVITY_ARTIFACT_DIR", "")
    ]
    for ad in artifact_dirs:
        if ad and os.path.isdir(ad):
            dest = os.path.join(ad, filename)
            shutil.copyfile(out_file, dest)
            print(f"[NekWave Analyze] Synced to agent artifact: {dest}")


if __name__ == "__main__":
    analyze()
