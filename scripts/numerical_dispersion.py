#!/usr/bin/env python3
"""
================================================================================
NekWave: 1-Element Periodic DGTD Bloch Dispersion Analysis Tool
================================================================================

Analyzes Discontinuous Galerkin Time-Domain (DGTD / DG-SEM) numerical dispersion
directly from the C++ / CUDA solver (NekWave / NekCEM) on a 1-element periodic mesh.

Workflow:
  1. Configures or reads a 1-element periodic simulation (.par).
  2. Executes the GPU CUDA solver: `./nekwave <case.par>` (or loads existing output).
  3. Parses observation probe time series (`probe_history.csv`) at collocation nodes.
  4. Performs high-resolution temporal Fourier spectral analysis (Hann-windowed FFT
     with parabolic spectral peak refinement) to extract discrete numerical frequencies
     omega_num(k) and phase velocities vp_num(k) = omega_num / k.
  5. Computes Points Per Wavelength (PPW), normalized wavenumber k * dx, and error %.
  6. Prints an expanded, formatted ASCII summary table with generous column padding.
  7. Generates publication-quality dual-axis dispersion figures (PNG).

Usage:
  python3 tools/numerical_dispersion.py -c examples/numerical_dispersion/numerical_dispersion.par -i build/output
  python3 tools/numerical_dispersion.py --run
  python3 tools/numerical_dispersion.py --sweep
================================================================================
"""

import sys
import os
import csv
import math
import cmath
import argparse
import subprocess
import shutil

# Check optional rendering libraries
HAVE_NUMPY = False
HAVE_MATPLOTLIB = False
HAVE_CAIRO = False

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    pass

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAVE_MATPLOTLIB = True
except ImportError:
    pass

try:
    import cairo
    HAVE_CAIRO = True
except ImportError:
    pass


# =============================================================================
# 1. PAR CONFIGURATION & PROBE DATA LOADERS
# =============================================================================
# 1. PAR CONFIGURATION & PROBE DATA LOADERS
# =============================================================================
def load_par_config(par_path):
    """
    Parses NekWave case parameter file (.par).
    """
    cfg = {
        "order": 8,
        "elements_x": 1, "elements_y": 1, "elements_z": 1,
        "Lx": 1.0, "Ly": 1.0, "Lz": 1.0,
        "xmin": -0.5, "xmax": 0.5,
        "ymin": -0.5, "ymax": 0.5,
        "zmin": -0.5, "zmax": 0.5,
        "periodic_x": True, "periodic_y": True, "periodic_z": True,
        "c": 1.0,
        "C0": 0.0,
        "output_dir": "output",
        "wave_type": "bloch",
        "carrier_k": 2.0 * math.pi,
        "num_modes": 4,
        "angle_deg": 0.0,
        "num_probes": 8,
        "num_steps": 1000
    }

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        par_path,
        os.path.join(repo_root, par_path) if par_path else None,
        os.path.join("..", par_path) if par_path else None,
        os.path.join(repo_root, "examples/numerical_dispersion/numerical_dispersion.par"),
        "examples/numerical_dispersion/numerical_dispersion.par",
        "../examples/numerical_dispersion/numerical_dispersion.par"
    ]
    resolved_path = None
    for c in candidates:
        if c and os.path.exists(c):
            resolved_path = c
            break

    if not resolved_path:
        return cfg

    try:
        with open(resolved_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '#' in line:
                    line = line.split('#')[0].strip()
                if '=' in line:
                    k, v = [x.strip() for x in line.split('=', 1)]
                    k_lower = k.lower()
                    if k_lower in ["order", "n"]: cfg["order"] = int(v)
                    elif k_lower in ["elements_x", "nelx"]: cfg["elements_x"] = int(v)
                    elif k_lower in ["elements_y", "nely"]: cfg["elements_y"] = int(v)
                    elif k_lower in ["elements_z", "nelz"]: cfg["elements_z"] = int(v)
                    elif k_lower in ["l_x", "lx"]: cfg["Lx"] = float(v)
                    elif k_lower in ["l_y", "ly"]: cfg["Ly"] = float(v)
                    elif k_lower in ["l_z", "lz"]: cfg["Lz"] = float(v)
                    elif k_lower in ["xmin", "x_min"]: cfg["xmin"] = float(v)
                    elif k_lower in ["xmax", "x_max"]: cfg["xmax"] = float(v)
                    elif k_lower in ["ymin", "y_min"]: cfg["ymin"] = float(v)
                    elif k_lower in ["ymax", "y_max"]: cfg["ymax"] = float(v)
                    elif k_lower in ["zmin", "z_min"]: cfg["zmin"] = float(v)
                    elif k_lower in ["zmax", "z_max"]: cfg["zmax"] = float(v)
                    elif k_lower in ["c0", "c_0"]: cfg["C0"] = float(v)
                    elif k_lower in ["c", "wave_speed"]: cfg["c"] = float(v)
                    elif k_lower in ["output_dir"]: cfg["output_dir"] = v
                    elif k_lower in ["wave_type"]: cfg["wave_type"] = v.lower()
                    elif k_lower in ["carrier_k"]: cfg["carrier_k"] = float(v)
                    elif k_lower in ["num_modes"]: cfg["num_modes"] = int(v)
                    elif k_lower in ["angle_deg"]: cfg["angle_deg"] = float(v)
                    elif k_lower in ["num_probes"]: cfg["num_probes"] = int(v)
                    elif k_lower in ["num_steps"]: cfg["num_steps"] = int(v)
    except Exception as e:
        print(f"[Warning] Could not parse parameter file {par_path}: {e}")

    # Derived grid spacing
    cfg["dx"] = cfg["Lx"] / max(1, cfg["elements_x"])
    cfg["dy"] = cfg["Ly"] / max(1, cfg["elements_y"])
    return cfg


def load_probe_history(input_dir):
    """
    Loads probe_history.csv generated by NekWave C++ solver.
    """
    candidates = [
        os.path.join(input_dir, "probe_history.csv"),
        os.path.join(input_dir, "output", "probe_history.csv"),
        os.path.join("build", input_dir, "probe_history.csv"),
        os.path.join("build", "output", "probe_history.csv"),
        "output/probe_history.csv"
    ]

    csv_path = None
    for cand in candidates:
        if os.path.exists(cand):
            csv_path = cand
            break

    if not csv_path:
        raise FileNotFoundError(
            f"Cannot locate 'probe_history.csv' in '{input_dir}'.\n"
            f"Run the simulation first: ./build/nekwave examples/numerical_dispersion/numerical_dispersion.par\n"
            f"Or pass '--run' to execute automatically."
        )

    times = []
    ez_data = {}

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        col_t = header.index("time")
        ez_cols = [(idx, name) for idx, name in enumerate(header) if name.startswith("Ez_")]

        for idx, name in ez_cols:
            ez_data[name] = []

        for row in reader:
            if not row or len(row) <= col_t:
                continue
            times.append(float(row[col_t]))
            for idx, name in ez_cols:
                ez_data[name].append(float(row[idx]))

    return times, ez_data, csv_path


# =============================================================================
# 2. FOURIER TRANSFORM & NUMERICAL FREQUENCY ESTIMATOR
# =============================================================================
def get_periodic_wavenumber(angle_deg, Lx=1.0, Ly=1.0, mode=1):
    """
    Computes reciprocal lattice wavenumber magnitude k_m = |k_vec|
    and actual angle for periodic boundary conditions.
    """
    if abs(angle_deg - 0.0) < 1e-4:
        mx, my = 1, 0
    elif abs(angle_deg - 45.0) < 1e-4:
        mx, my = 1, 1
    elif abs(angle_deg - 90.0) < 1e-4:
        mx, my = 0, 1
    elif abs(angle_deg - 26.6) < 1.0 or abs(angle_deg - 30.0) < 5.0:
        mx, my = 2, 1
    elif abs(angle_deg - 18.4) < 1.0 or abs(angle_deg - 15.0) < 5.0:
        mx, my = 3, 1
    else:
        ang_rad = math.radians(angle_deg)
        mx = max(1, int(round(math.cos(ang_rad) * 4.0)))
        my = int(round(math.sin(ang_rad) * 4.0))

    kx = 2.0 * math.pi * mx / Lx
    ky = 2.0 * math.pi * my / Ly
    k_mag = math.sqrt(kx * kx + ky * ky)
    actual_ang = math.degrees(math.atan2(ky, kx))
    return mode * k_mag, actual_ang


def fft_1d(x):
    """
    Cooley-Tukey Radix-2 1D Fast Fourier Transform in pure Python.
    """
    n = len(x)
    if n <= 1:
        return x
    if (n & (n - 1)) != 0:
        next_pow2 = 1 << (n - 1).bit_length()
        x = list(x) + [0.0] * (next_pow2 - n)
        n = next_pow2

    even = fft_1d(x[0::2])
    odd  = fft_1d(x[1::2])
    terms = [cmath.exp(-2j * math.pi * k / n) * odd[k] for k in range(n // 2)]
    return [even[k] + terms[k] for k in range(n // 2)] + [even[k] - terms[k] for k in range(n // 2)]


def extract_mode_frequencies(times, signal, expected_omegas):
    """
    Extracts numerical oscillation frequencies from time series using Hann-windowed
    FFT with parabolic interpolation around spectral peaks.
    """
    num_steps = len(times)
    if num_steps < 2:
        return [w for w in expected_omegas]

    dt = (times[-1] - times[0]) / float(num_steps - 1)
    if dt <= 0:
        return [w for w in expected_omegas]

    # High-resolution zero-padded FFT length (at least 8192)
    n_fft = max(8192, 1 << (num_steps * 2 - 1).bit_length())

    # Hann window
    windowed = [signal[i] * 0.5 * (1.0 - math.cos(2.0 * math.pi * i / (num_steps - 1))) for i in range(num_steps)]
    windowed += [0.0] * (n_fft - num_steps)

    spec = fft_1d(windowed)[:n_fft // 2]
    freqs = [2.0 * math.pi * k / (n_fft * dt) for k in range(n_fft // 2)]
    mag = [abs(x) for x in spec]

    num_bins = len(freqs)
    w_nyquist = freqs[-1] if num_bins > 0 else 0.0

    extracted = []
    for w_exp in expected_omegas:
        # Check if frequency is strictly above Nyquist limit
        if w_exp > w_nyquist:
            extracted.append(float('nan'))
            continue

        # Search window within +/- 25% of expected frequency
        w_low = 0.75 * w_exp
        w_high = 1.25 * w_exp

        idx_min = min(range(num_bins), key=lambda i: abs(freqs[i] - w_low))
        idx_max = min(range(num_bins), key=lambda i: abs(freqs[i] - w_high))

        if idx_min > idx_max:
            idx_min, idx_max = idx_max, idx_min

        # Ensure indices stay strictly within valid range [0, num_bins - 1]
        idx_min = max(0, min(num_bins - 1, idx_min))
        idx_max = max(idx_min, min(num_bins - 1, idx_max))

        search_range = list(range(idx_min, idx_max + 1))
        best_idx = max(search_range, key=lambda i: mag[i])

        # Parabolic peak interpolation
        if 0 < best_idx < num_bins - 1:
            alpha = mag[best_idx - 1]
            beta  = mag[best_idx]
            gamma = mag[best_idx + 1]
            denom = alpha - 2.0 * beta + gamma
            delta = 0.5 * (alpha - gamma) / denom if denom != 0.0 else 0.0
            w_num = (best_idx + delta) * (2.0 * math.pi / (n_fft * dt))
        else:
            w_num = freqs[best_idx]

        extracted.append(w_num)

    return extracted


# =============================================================================
# 3. CUDA SIMULATION DRIVER (Single Element Periodic Grid)
# =============================================================================
def find_nekwave_binary():
    """
    Locates nekwave executable.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        "./nekwave",
        "./build/nekwave",
        os.path.join(repo_root, "build", "nekwave"),
        os.path.join(repo_root, "nekwave"),
        "../build/nekwave",
        "build/nekwave",
        "nekwave"
    ]
    for c in candidates:
        if os.path.exists(c) and os.access(c, os.X_OK):
            return c
    return "./build/nekwave"


def run_single_cuda_simulation(cfg, angle_deg=None, out_subdir="output"):
    """
    Executes NekWave CUDA solver on a 1-element periodic grid.
    """
    exe = find_nekwave_binary()
    if not os.path.exists(exe):
        raise FileNotFoundError(f"NekWave binary '{exe}' not found. Please run 'make' in build/ first.")

    order = cfg["order"]
    Lx = cfg["Lx"]
    c0 = cfg["C0"]
    angle = cfg["angle_deg"] if angle_deg is None else angle_deg
    num_modes = cfg.get("num_modes", 4)
    num_steps = cfg.get("num_steps", 1000)

    par_text = f"""# NekWave 1-Element Periodic Benchmark
elements_x = 1
elements_y = 1
elements_z = 1
order = {order}

L_x = {Lx}
L_y = {Lx}
L_z = {Lx}

xmin = {-0.5 * Lx}
xmax =  {0.5 * Lx}
ymin = {-0.5 * Lx}
ymax =  {0.5 * Lx}
zmin = {-0.5 * Lx}
zmax =  {0.5 * Lx}

periodic_x = true
periodic_y = true
periodic_z = true

cfl = auto
dt = auto
num_steps = {num_steps}
output_freq = 1
C0 = {c0}
output_dir = {out_subdir}
export_fields = false

wave_type = bloch
carrier_k = {2.0 * math.pi / Lx}
num_modes = {num_modes}
angle_deg = {angle}
num_probes = {order}
"""
    os.makedirs(out_subdir, exist_ok=True)
    temp_par = os.path.join(out_subdir, "case_run.par")
    with open(temp_par, 'w') as f:
        f.write(par_text)

    print(f"[NekWave CUDA] Running 1-element simulation: Order N = {order}, Angle = {angle:.1f} deg, C0 = {c0}...")
    res = subprocess.run([exe, temp_par], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        err_msg = res.stderr.decode('utf-8')
        print(f"[Error] CUDA simulation failed:\n{err_msg}")
        return None

    csv_path = os.path.join(out_subdir, "probe_history.csv")
    if not os.path.exists(csv_path):
        alt_path = os.path.join("build", out_subdir, "probe_history.csv")
        if os.path.exists(alt_path):
            csv_path = alt_path

    return csv_path


# =============================================================================
# 4. SUMMARY TABLE DISPLAY (Generous, Aligned Columns)
# =============================================================================
def print_summary_table(results_by_angle, order, c0, num_modes=4):
    """
    Prints a formatted ASCII summary of numerical phase velocity errors
    with generous column widths (>= 32 chars) and perfectly aligned borders.
    """
    angles = sorted(results_by_angle.keys())
    flux_str = f"Flux C0 = {c0}"
    title_str = f"NekWave C++ CUDA Simulation Dispersion Summary (1 Element, Order N = {order}, {flux_str})"

    # Dynamically expand column width so the title line never overflows
    min_title_len = len(title_str) + 4
    base_cols_w = 43  # '|  Mode  |  PPW (nodes)   |  k * dx [rad]  |' is 43 chars
    needed_per_col = max(32, math.ceil((min_title_len - base_cols_w) / max(1, len(angles))) - 3)
    col_w = max(32, needed_per_col)

    header = f"| {'Mode':^6} | {'PPW (nodes)':^14} | {'k * dx [rad]':^14} |"
    for ang in angles:
        header += f" {f'theta = {ang:2.0f} deg (v_p / c)':^{col_w}} |"

    total_width = len(header)
    double_bar = "=" * total_width
    single_bar = "-" * total_width

    print("\n" + double_bar)
    print(f"| {title_str:^{total_width - 4}} |")
    print(double_bar)
    print(header)
    print(single_bar)

    for m in range(1, num_modes + 1):
        ppw = (order - 1) / float(m)
        k_dx = 2.0 * math.pi * m
        row = f"| {m:^6d} | {ppw:^14.2f} | {k_dx:^14.4f} |"
        for ang in angles:
            if m in results_by_angle[ang]:
                v_rat, err_pct = results_by_angle[ang][m]
                if math.isnan(v_rat):
                    cell = "Sub-Nyquist"
                else:
                    cell = f"{v_rat:7.5f} ({err_pct:+6.2f}%)"
            else:
                cell = "N/A"
            row += f" {cell:^{col_w}} |"
        print(row)

    print(double_bar + "\n")


# =============================================================================
# 5. PUBLICATION PLOTTING (Cairo / Matplotlib)
# =============================================================================
def plot_dispersion_results(results_by_angle, out_png, order, c0):
    """
    Renders publication dispersion figure comparing C++ simulation curves.
    """
    angles = sorted(results_by_angle.keys())
    colors = [(0.12, 0.47, 0.71), (1.0, 0.5, 0.05), (0.17, 0.63, 0.17), (0.84, 0.15, 0.16)]
    hex_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    flux_lbl = f"Flux C0={c0}"

    if HAVE_MATPLOTLIB:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
        fig.suptitle(
            f"NekWave C++ CUDA Simulation Dispersion Analysis (1 Element, Order N = {order}, {flux_lbl} Flux)",
            fontsize=13, fontweight='bold'
        )

        ax1.axhline(1.0, color='gray', linestyle='--', linewidth=1.2, label=r'Exact: $v_p / c = 1.0$')

        for idx, ang in enumerate(angles):
            data = results_by_angle[ang]
            valid_modes = [m for m in sorted(data.keys()) if not math.isnan(data[m][0])]
            if not valid_modes:
                continue
            ppw_vals = [(order - 1) / float(m) for m in valid_modes]
            v_rats = [data[m][0] for m in valid_modes]
            ax1.plot(ppw_vals, v_rats, 'o-', color=hex_colors[idx % len(hex_colors)],
                     linewidth=2.0, markersize=7, label=f"$\\theta = {ang:.0f}^\\circ$ (CUDA)")

        ax1.set_xlabel(r'Points Per Wavelength (PPW = $\lambda / \Delta x$)', fontsize=11, fontweight='bold')
        ax1.set_ylabel(r'Normalized Phase Velocity $v_p / c$', fontsize=11, fontweight='bold')
        # Invert x-axis to match analytical plot (high PPW to low PPW)
        ax1.set_xlim(ax1.get_xlim()[::-1])
        ax1.set_xscale("log")
        ax1.set_ylim([0.30, 1.6])
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend(loc='lower left', frameon=True, framealpha=0.9)

        # Right panel: omega_num vs omega_exact
        all_valid_modes = []
        for ang in angles:
            all_valid_modes.extend([m for m in results_by_angle[ang].keys() if not math.isnan(results_by_angle[ang][m][0])])
        max_m = max(all_valid_modes) if all_valid_modes else 4
        max_k = 2.0 * math.pi * max_m
        k_line = [0.0, max_k * 1.1]
        ax2.plot(k_line, k_line, color='gray', linestyle='--', linewidth=1.2, label=r'Exact: $\omega = c k$')

        for idx, ang in enumerate(angles):
            data = results_by_angle[ang]
            valid_modes = [m for m in sorted(data.keys()) if not math.isnan(data[m][0])]
            if not valid_modes:
                continue
            k_vals = [2.0 * math.pi * m for m in valid_modes]
            w_nums = [data[m][0] * (2.0 * math.pi * m) for m in valid_modes]
            ax2.plot(k_vals, w_nums, 's-', color=hex_colors[idx % len(hex_colors)],
                     linewidth=2.0, markersize=7, label=f"$\\theta = {ang:.0f}^\\circ$ (CUDA)")

        ax2.set_xlabel(r'Wavenumber $k$ [rad/m]', fontsize=11, fontweight='bold')
        ax2.set_ylabel(r'Numerical Angular Frequency $\omega_{\mathrm{num}}$ [rad/s]', fontsize=11, fontweight='bold')
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend(loc='upper left', frameon=True, framealpha=0.9)

        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        print(f"[NekWave Bloch] Saved publication plot (Matplotlib): {out_png}")

    elif HAVE_CAIRO:
        width, height = 1100, 650
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(surface)
        ctx.set_source_rgb(1.0, 1.0, 1.0)
        ctx.paint()

        # Title
        ctx.set_source_rgb(0.1, 0.1, 0.1)
        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(15)
        ctx.move_to(50, 40)
        ctx.show_text(f"NekWave C++ CUDA Simulation Dispersion Analysis (1 Element, Order N = {order}, {flux_lbl} Flux)")

        # Plot frame
        x0, y0, x1, y1 = 90, 540, 1020, 120
        pw, ph = x1 - x0, y0 - y1
        ctx.set_source_rgb(0.96, 0.96, 0.97)
        ctx.rectangle(x0, y1, pw, ph)
        ctx.fill()
        ctx.set_source_rgb(0.7, 0.7, 0.7)
        ctx.rectangle(x0, y1, pw, ph)
        ctx.stroke()

        ymin, ymax = 0.40, 1.35
        kmin, kmax = 0.0, 30.0

        # Exact line: v/c = 1.0
        py_exact = y0 - (1.0 - ymin) / (ymax - ymin) * ph
        ctx.set_source_rgb(0.5, 0.5, 0.5)
        ctx.set_line_width(1.2)
        ctx.set_dash([4.0, 3.0])
        ctx.move_to(x0, py_exact); ctx.line_to(x1, py_exact); ctx.stroke()
        ctx.set_dash([])

        # Axis ticks and labels
        ctx.set_source_rgb(0.3, 0.3, 0.3)
        ctx.set_font_size(11)
        for kv in [0, 5, 10, 15, 20, 25, 30]:
            px = x0 + (kv - kmin) / (kmax - kmin) * pw
            ctx.move_to(px, y0); ctx.line_to(px, y0 + 5); ctx.stroke()
            ctx.move_to(px - 6, y0 + 20); ctx.show_text(str(kv))

        for vv in [0.5, 0.75, 1.0, 1.25]:
            py = y0 - (vv - ymin) / (ymax - ymin) * ph
            ctx.move_to(x0 - 5, py); ctx.line_to(x0, py); ctx.stroke()
            ctx.move_to(x0 - 45, py + 4); ctx.show_text(f"{vv:.2f}")

        # Axis labels
        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(12)
        ctx.move_to(x0 + pw / 2 - 120, y0 + 45)
        ctx.show_text("Normalized Wavenumber k * dx [rad]")

        ctx.save()
        ctx.translate(28, (y0 + y1) / 2 + 100)
        ctx.rotate(-math.pi / 2.0)
        ctx.move_to(0, 0)
        ctx.show_text("Normalized Phase Velocity v_p / c")
        ctx.restore()

        for idx, ang in enumerate(angles):
            data = results_by_angle[ang]
            modes = [m for m in sorted(data.keys()) if not math.isnan(data[m][0])]
            if not modes:
                continue
            col = colors[idx % len(colors)]
            ctx.set_source_rgb(*col)
            ctx.set_line_width(2.2)

            for i, m in enumerate(modes):
                ppw = (order - 1) / float(m)
                v_val = data[m][0]
                px = x0 + (kmax - ppw) / (kmax - kmin) * pw # Inverted X axis
                py = y0 - (v_val - ymin) / (ymax - ymin) * ph
                py = max(y1, min(y0, py))
                if i == 0: ctx.move_to(px, py)
                else: ctx.line_to(px, py)
            ctx.stroke()

            # Draw markers
            for m in modes:
                ppw = (order - 1) / float(m)
                v_val = data[m][0]
                px = x0 + (kmax - ppw) / (kmax - kmin) * pw
                py = y0 - (v_val - ymin) / (ymax - ymin) * ph
                py = max(y1, min(y0, py))
                ctx.arc(px, py, 4.0, 0, 2 * math.pi)
                ctx.fill()

            # Legend
            ctx.move_to(x0 + 20, y1 + 30 + idx * 22)
            ctx.show_text(f"— theta = {ang:.0f} deg (CUDA simulation)")

        surface.write_to_png(out_png)
        print(f"[NekWave Bloch] Saved publication plot (Cairo): {out_png}")


# =============================================================================
# 6. CLI ENTRY POINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="NekWave 1-Element Periodic DGTD Dispersion Analysis (C++ CUDA Solver Data)"
    )
    parser.add_argument("pos_case", nargs="?", default=None,
                        help="Optional positional path to .par file or output directory.")
    parser.add_argument("-c", "--case", "--par", dest="case_file", type=str, default=None,
                        help="Path to case parameter file (.par). Default: examples/numerical_dispersion/numerical_dispersion.par")
    parser.add_argument("-i", "--input-dir", dest="input_dir", type=str, default=None,
                        help="Directory containing probe_history.csv (default: output/ or build/output/)")
    parser.add_argument("-o", "--output-dir", dest="output_dir", type=str, default="output",
                        help="Directory to save dispersion plot (default: output/)")
    parser.add_argument("-f", "--filename", dest="filename", type=str, default="numerical_dispersion.png",
                        help="Output image filename (default: numerical_dispersion.png)")
    parser.add_argument("--run", action="store_true",
                        help="Force execution of the NekWave CUDA simulation before analysis.")
    parser.add_argument("--sweep", action="store_true",
                        help="Execute multi-angle (0, 15, 30, 45 deg) CUDA solver sweep on 1-element grid.")

    args = parser.parse_args()

    # Determine case file
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_par = os.path.join(repo_root, "examples/numerical_dispersion/numerical_dispersion.par")
    if not os.path.exists(default_par):
        default_par = "examples/numerical_dispersion/numerical_dispersion.par"

    if args.pos_case:
        if args.pos_case.endswith(".par") or os.path.isfile(args.pos_case):
            args.case_file = args.pos_case
        elif not args.input_dir:
            args.input_dir = args.pos_case

    case_path = args.case_file if args.case_file else default_par
    if not os.path.exists(case_path):
        alt = os.path.join(repo_root, case_path)
        if os.path.exists(alt):
            case_path = alt

    cfg = load_par_config(case_path)

    print("================================================================================")
    print("      NekWave: 1-Element Periodic DGTD Bloch Dispersion Analysis Tool           ")
    print("================================================================================")
    print(f"  Configuration File: {case_path}")
    print(f"  Grid Elements:      {cfg['elements_x']} x {cfg['elements_y']} x {cfg['elements_z']} (1-Element Unit Cell)")
    print(f"  Collocation Order:  N = {cfg['order']} (Degree P = {cfg['order'] - 1})")
    print(f"  Flux Formulation:   {'Central (C0 = 0.0)' if cfg['C0'] == 0.0 else 'Upwind (C0 = 1.0)'}")
    print(f"  Boundary Condition: Periodic in X, Y, Z")

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, args.filename)

    # Multi-angle sweep workflow
    if args.sweep:
        print("\n[Sweep Mode] Running C++ CUDA simulations for angles theta = 0, 15, 30, 45 deg...")
        angles = [0.0, 15.0, 30.0, 45.0]
        results_by_angle = {}

        for ang in angles:
            sub_dir = os.path.join(out_dir, f"sweep_angle_{int(ang)}")
            csv_path = run_single_cuda_simulation(cfg, angle_deg=ang, out_subdir=sub_dir)
            if not csv_path or not os.path.exists(csv_path):
                continue
            times, ez_data, _ = load_probe_history(sub_dir)
            first_probe = list(ez_data.keys())[0]
            sig = ez_data[first_probe]

            num_modes = cfg.get("num_modes", 4)
            exp_w = [cfg["c"] * get_periodic_wavenumber(ang, cfg["Lx"], cfg["Ly"], mode=m)[0] for m in range(1, num_modes + 1)]
            w_nums = extract_mode_frequencies(times, sig, exp_w)

            mode_results = {}
            for m, (w_exp, w_n) in enumerate(zip(exp_w, w_nums), start=1):
                if math.isnan(w_n):
                    mode_results[m] = (float('nan'), float('nan'))
                else:
                    v_rat = w_n / w_exp
                    err_pct = (v_rat - 1.0) * 100.0
                    mode_results[m] = (v_rat, err_pct)
            results_by_angle[ang] = mode_results

            # Clean temporary probe folder
            shutil.rmtree(sub_dir, ignore_errors=True)

        print_summary_table(results_by_angle, cfg["order"], cfg["C0"], cfg.get("num_modes", 4))
        plot_dispersion_results(results_by_angle, out_png, cfg["order"], cfg["C0"])
        return

    # Single-case analysis workflow
    input_dir = args.input_dir if args.input_dir else cfg.get("output_dir", "output")
    has_output = False
    for cand in [input_dir, os.path.join("build", input_dir), "output", "build/output"]:
        if os.path.exists(os.path.join(cand, "probe_history.csv")):
            input_dir = cand
            has_output = True
            break

    if args.run or not has_output:
        print("\n[Execution Mode] Triggering NekWave C++ CUDA solver simulation...")
        csv_path = run_single_cuda_simulation(cfg, out_subdir=input_dir)
        if not csv_path:
            sys.exit(1)

    print(f"\n[Analysis Mode] Loading CUDA simulation data from: {input_dir}")
    times, ez_data, actual_csv = load_probe_history(input_dir)
    print(f"  Loaded {len(times)} time steps from: {actual_csv}")

    first_probe = list(ez_data.keys())[0]
    sig = ez_data[first_probe]
    num_modes = cfg.get("num_modes", 4)
    current_angle = cfg.get("angle_deg", 0.0)
    exp_w = [cfg["c"] * get_periodic_wavenumber(current_angle, cfg["Lx"], cfg["Ly"], mode=m)[0] for m in range(1, num_modes + 1)]
    w_nums = extract_mode_frequencies(times, sig, exp_w)

    mode_results = {}
    for m, (w_exp, w_n) in enumerate(zip(exp_w, w_nums), start=1):
        if math.isnan(w_n):
            mode_results[m] = (float('nan'), float('nan'))
        else:
            v_rat = w_n / w_exp
            err_pct = (v_rat - 1.0) * 100.0
            mode_results[m] = (v_rat, err_pct)

    results_by_angle = {current_angle: mode_results}

    # Print summary table and generate publication figure
    print_summary_table(results_by_angle, cfg["order"], cfg["C0"], num_modes)
    plot_dispersion_results(results_by_angle, out_png, cfg["order"], cfg["C0"])


if __name__ == "__main__":
    main()
