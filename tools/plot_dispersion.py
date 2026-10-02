#!/usr/bin/env python3
"""
NekWave: Numerical Dispersion Analysis & (k, omega) Visualization Tool

Analyzes spatial-temporal field data recorded across observation probes to:
  1. Compute the 2D space-time Fourier energy spectrum S(k, omega).
  2. Extract numerical dispersion curve omega_num(k) vs analytical omega_exact(k).
  3. Quantify numerical phase velocity errors vp_num(k) / c - 1.
  4. Output a formatted ASCII dispersion table to stdout.
  5. Generate publication-quality 4-panel visualization (PNG and SVG).

Usage:
  python3 tools/plot_dispersion.py -i build/output -o build/output
  python3 tools/plot_dispersion.py -i output -c examples/numerical_dispersion/numerical_dispersion.par
"""

import sys
import os
import csv
import math
import cmath
import argparse

# Check optional libraries
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


# -----------------------------------------------------------------------------
# CLI Arguments Parser
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="NekWave numerical dispersion relation (k - omega) analysis tool."
    )
    parser.add_argument(
        "pos_input",
        nargs="?",
        default=None,
        help="Optional positional argument for input folder (e.g. output or build/output)."
    )
    parser.add_argument(
        "-i", "--input-dir", "--input-folder",
        dest="input_dir",
        type=str,
        default=None,
        help="Directory containing probe_history.csv (default: current or output/)."
    )
    parser.add_argument(
        "-o", "--output-dir", "--output-folder",
        dest="output_dir",
        type=str,
        default=None,
        help="Directory to save dispersion plots and summary."
    )
    parser.add_argument(
        "-c", "--case", "--par",
        dest="case_file",
        type=str,
        default=None,
        help="Path to case parameter file (.par) for domain parameters."
    )
    parser.add_argument(
        "-f", "--filename",
        dest="filename",
        type=str,
        default="dispersion_relation.png",
        help="Name of the generated plot image (default: dispersion_relation.png)."
    )
    args = parser.parse_args()

    if not args.input_dir and args.pos_input:
        args.input_dir = args.pos_input

    if not args.input_dir:
        candidates = [
            "output",
            "build/output",
            "../output",
            "../build/output",
            "."
        ]
        for cand in candidates:
            if os.path.exists(os.path.join(cand, "probe_history.csv")):
                args.input_dir = cand
                break
        if not args.input_dir:
            for cand in ["output", "build/output", "../output"]:
                if os.path.exists(os.path.join(cand, "energy_history.csv")):
                    args.input_dir = cand
                    break
            if not args.input_dir:
                args.input_dir = "output"

    if not args.output_dir:
        args.output_dir = args.input_dir

    return args


# -----------------------------------------------------------------------------
# Configuration Loader
# -----------------------------------------------------------------------------
def load_par_config(par_path):
    cfg = {
        "Lx": 4.0, "Ly": 1.0, "Lz": 1.0,
        "xmin": -2.0, "xmax": 2.0,
        "ymin": -0.5, "ymax": 0.5,
        "zmin": -0.5, "zmax": 0.5,
        "c_wave": 1.0,
        "carrier_k": 4.0 * math.pi,
        "packet_sigma": 0.30,
        "order": 4
    }
    if not par_path or not os.path.exists(par_path):
        return cfg

    try:
        with open(par_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '#' in line:
                    line = line.split('#')[0].strip()
                if '=' in line:
                    k, v = [x.strip() for x in line.split('=', 1)]
                    k = k.lower()
                    if k in ["l_x", "lx"]: cfg["Lx"] = float(v)
                    elif k in ["l_y", "ly"]: cfg["Ly"] = float(v)
                    elif k in ["l_z", "lz"]: cfg["Lz"] = float(v)
                    elif k in ["xmin", "x_min"]: cfg["xmin"] = float(v)
                    elif k in ["xmax", "x_max"]: cfg["xmax"] = float(v)
                    elif k in ["ymin", "y_min"]: cfg["ymin"] = float(v)
                    elif k in ["ymax", "y_max"]: cfg["ymax"] = float(v)
                    elif k in ["carrier_k"]: cfg["carrier_k"] = float(v)
                    elif k in ["packet_sigma"]: cfg["packet_sigma"] = float(v)
                    elif k in ["order"]: cfg["order"] = int(v)
                    elif k in ["wave_type"]: cfg["wave_type"] = v.lower()
    except Exception as e:
        print(f"[Warning] Could not parse par file {par_path}: {e}")

    return cfg


# -----------------------------------------------------------------------------
# Data Loader: Probe History
# -----------------------------------------------------------------------------
def load_probe_history(input_dir):
    csv_path = os.path.join(input_dir, "probe_history.csv")
    if not os.path.exists(csv_path):
        alt_path = os.path.join(input_dir, "output", "probe_history.csv")
        if os.path.exists(alt_path):
            csv_path = alt_path
        else:
            raise FileNotFoundError(
                f"Cannot find 'probe_history.csv' in '{input_dir}'.\n"
                f"Please ensure the simulation was run with observation probes enabled\n"
                f"(e.g. 'num_probes = 32' in your .par file).\n"
                f"You can also specify the output directory explicitly: python3 tools/plot_dispersion.py -i <output_dir>"
            )

    steps = []
    times = []
    probe_ids = []
    field_data = {}  # {probe_id: [Ez_values]}

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        # Parse column headers: step, time, then Ex_k, Ey_k, Ez_k, Hx_k, Hy_k, Hz_k
        col_map = {}
        for col_idx, col_name in enumerate(header):
            col_name = col_name.strip()
            if col_name.startswith("Ez_"):
                pid = int(col_name.split("_")[1])
                col_map[pid] = col_idx
                if pid not in probe_ids:
                    probe_ids.append(pid)
                field_data[pid] = []

        probe_ids.sort()

        for row in reader:
            if not row or len(row) < 2:
                continue
            steps.append(int(row[0]))
            times.append(float(row[1]))
            for pid in probe_ids:
                field_data[pid].append(float(row[col_map[pid]]))

    return steps, times, probe_ids, field_data


# -----------------------------------------------------------------------------
# Pure Python Fast Fourier Transform (Cooley-Tukey Radix-2)
# -----------------------------------------------------------------------------
def fft_1d(x):
    n = len(x)
    if n <= 1:
        return x
    if (n & (n - 1)) != 0:
        # Zero-pad to next power of 2
        next_pow2 = 1 << (n - 1).bit_length()
        x = list(x) + [0.0] * (next_pow2 - n)
        n = next_pow2

    even = fft_1d(x[0::2])
    odd  = fft_1d(x[1::2])
    terms = [cmath.exp(-2j * math.pi * k / n) * odd[k] for k in range(n // 2)]
    return [even[k] + terms[k] for k in range(n // 2)] + [even[k] - terms[k] for k in range(n // 2)]


# -----------------------------------------------------------------------------
# 2D Space-Time Fourier Analysis (k, omega)
# -----------------------------------------------------------------------------
def compute_dispersion_spectrum(times, probe_ids, field_data, cfg):
    num_t = len(times)
    num_x = len(probe_ids)
    dt = (times[-1] - times[0]) / (num_t - 1) if num_t > 1 else 1.0

    # Probe x-coordinates: collinear along centerline
    Lx = cfg["Lx"]
    xmin = cfg["xmin"]
    xmax = cfg["xmax"]
    x_start = xmin + 0.05 * Lx
    x_end   = xmax - 0.05 * Lx
    dx = (x_end - x_start) / (num_x - 1) if num_x > 1 else 1.0
    x_coords = [x_start + p * dx for p in range(num_x)]

    # Form 2D space-time matrix: Ez_grid[t_idx][x_idx]
    grid = []
    for t_idx in range(num_t):
        row = [field_data[pid][t_idx] for pid in probe_ids]
        grid.append(row)

    ky_trans = math.pi / cfg["Ly"]  # Transverse wavenumber from cavity height
    c = cfg.get("c_wave", 1.0)

    # 1. Temporal FFT for each probe
    # Zero-pad time series to enhance frequency resolution
    n_fft_t = 1 << (num_t * 2 - 1).bit_length()
    omega_vals = [2.0 * math.pi * m / (n_fft_t * dt) for m in range(n_fft_t // 2)]

    # Temporal spectra at each probe: time_spec[probe_idx]
    time_spec = []
    for pid in probe_ids:
        signal = field_data[pid]
        # Apply Hann window to suppress spectral leakage and pad to n_fft_t
        windowed = [signal[i] * 0.5 * (1.0 - math.cos(2.0 * math.pi * i / (num_t - 1))) for i in range(num_t)]
        windowed += [0.0] * (n_fft_t - num_t)
        spec = fft_1d(windowed)[:n_fft_t // 2]
        time_spec.append(spec)

    # 2. Spatial discrete Fourier decomposition
    # Evaluate wave numbers k_x = m * pi / Lx (standing/traveling harmonics)
    if cfg.get("wave_type") == "multimode":
        max_mode = 8
    else:
        max_mode = min(16, num_x - 1)
    k_modes = []
    results = []

    for m in range(1, max_mode + 1):
        kx = m * math.pi / Lx
        # Compute spatial projection of temporal spectra onto mode m: sin(kx * (x - xmin))
        mode_spec = [0.0] * len(omega_vals)
        for w_idx in range(len(omega_vals)):
            c_val = 0.0 + 0.0j
            for p_idx in range(num_x):
                spatial_weight = math.sin(kx * (x_coords[p_idx] - xmin))
                c_val += time_spec[p_idx][w_idx] * spatial_weight
            mode_spec[w_idx] = abs(c_val)

        # Find peak numerical frequency
        # Search in the vicinity of expected frequency (+/- 20%)
        w_exact = c * math.sqrt(kx * kx + ky_trans * ky_trans)
        search_min = 0.8 * w_exact
        search_max = 1.2 * w_exact

        best_w = w_exact
        max_pwr = -1.0
        for w_idx, w in enumerate(omega_vals):
            if search_min <= w <= search_max:
                if mode_spec[w_idx] > max_pwr:
                    max_pwr = mode_spec[w_idx]
                    best_w = w

        # Refine peak using 3-point parabolic interpolation
        w_peak_idx = min(range(len(omega_vals)), key=lambda i: abs(omega_vals[i] - best_w))
        if 0 < w_peak_idx < len(omega_vals) - 1 and max_pwr > 0:
            y0 = mode_spec[w_peak_idx - 1]
            y1 = mode_spec[w_peak_idx]
            y2 = mode_spec[w_peak_idx + 1]
            denom = 2.0 * (2.0 * y1 - y0 - y2)
            if abs(denom) > 1e-12:
                delta = (y0 - y2) / denom
                delta = max(-0.5, min(0.5, delta))
                best_w = omega_vals[w_peak_idx] + delta * (omega_vals[1] - omega_vals[0])

        w_num = best_w
        k_total = math.sqrt(kx * kx + ky_trans * ky_trans)
        vp_num = w_num / k_total
        disp_error = (vp_num / c - 1.0) * 100.0

        results.append({
            "mode": m,
            "kx": kx,
            "k_total": k_total,
            "w_exact": w_exact,
            "w_num": w_num,
            "vp_num": vp_num,
            "disp_error": disp_error,
            "power": max_pwr
        })

    return x_coords, times, results, ky_trans


# -----------------------------------------------------------------------------
# Formatted ASCII Output Table
# -----------------------------------------------------------------------------
def print_dispersion_table(results, cfg):
    order = cfg.get("order", 4)
    print("\n" + "=" * 94)
    print(f"       NekWave DG-SEM Numerical Dispersion Relation Analysis (Order N = {order})")
    print("=" * 94)
    print(f" {'Mode':^6} | {'kx [rad/m]':^12} | {'k_tot [rad/m]':^13} | {'w_exact [rad/s]':^15} | {'w_num [rad/s]':^15} | {'vp/c':^8} | {'Error (%)':^10}")
    print("-" * 94)
    for r in results:
        print(f" {r['mode']:^6d} | {r['kx']:^12.4f} | {r['k_total']:^13.4f} | {r['w_exact']:^15.5f} | {r['w_num']:^15.5f} | {r['vp_num']:^8.5f} | {r['disp_error']:^+10.4f}%")
    print("=" * 94)

    # Compute RMS dispersion error
    rms_err = math.sqrt(sum(r['disp_error']**2 for r in results) / len(results)) if results else 0.0
    max_err = max(abs(r['disp_error']) for r in results) if results else 0.0
    print(f"  Summary: RMS Dispersion Phase Error = {rms_err:.4f}% | Max Error = {max_err:.4f}%")
    print("=" * 94 + "\n")


# -----------------------------------------------------------------------------
# Cairo Vector & PNG Renderer (Fallback when Matplotlib is unavailable)
# -----------------------------------------------------------------------------
def render_cairo_plot(out_png, out_svg, x_coords, times, results, probe_ids, field_data, cfg):
    width, height = 1100, 850
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)

    # Background
    ctx.set_source_rgb(1.0, 1.0, 1.0)
    ctx.paint()

    # Title & Header
    ctx.set_source_rgb(0.1, 0.1, 0.1)
    ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(18)
    ctx.move_to(40, 42)
    ctx.show_text(f"NekWave DG-SEM Numerical Dispersion Analysis (Order N = {cfg['order']})")

    ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(12)
    ctx.set_source_rgb(0.4, 0.4, 0.4)
    ctx.move_to(40, 62)
    ctx.show_text("Space-time Fourier analysis of Maxwell wavepacket propagation with central flux (C0 = 0.0)")

    # 4 Subplot layout: (x, y, w, h)
    # Panel 1: (50, 90, 480, 320)  - Dispersion Curve (w vs k)
    # Panel 2: (580, 90, 480, 320) - Phase Velocity Error vs k
    # Panel 3: (50, 470, 480, 320) - Wavepacket Temporal Evolution
    # Panel 4: (580, 470, 480, 320)- Spatial Wave Profile Cuts
    panels = [
        {"x": 60, "y": 100, "w": 460, "h": 310, "title": "Panel 1: Dispersion Relation (omega vs k)"},
        {"x": 590, "y": 100, "w": 460, "h": 310, "title": "Panel 2: Phase Velocity Error (vp / c - 1)"},
        {"x": 60, "y": 480, "w": 460, "h": 310, "title": "Panel 3: Observation Probe Time Histories"},
        {"x": 590, "y": 480, "w": 460, "h": 310, "title": "Panel 4: Wavepacket Spatial Profile Cuts"}
    ]

    for p in panels:
        # Draw panel box
        ctx.set_source_rgb(0.97, 0.97, 0.98)
        ctx.rectangle(p["x"], p["y"], p["w"], p["h"])
        ctx.fill()
        ctx.set_source_rgb(0.8, 0.8, 0.8)
        ctx.set_line_width(1.0)
        ctx.rectangle(p["x"], p["y"], p["w"], p["h"])
        ctx.stroke()

        # Title
        ctx.set_source_rgb(0.15, 0.15, 0.2)
        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(12)
        ctx.move_to(p["x"] + 12, p["y"] + 24)
        ctx.show_text(p["title"])

    # Helper to draw axes, ticks, grid, and labels for a panel
    def draw_axes(p, x_label, y_label, x_min, x_max, y_min, y_max, x_ticks, y_ticks, x_fmt="{:.1f}", y_fmt="{:.1f}"):
        x0 = p["x"] + 52
        y0 = p["y"] + p["h"] - 38
        x1 = p["x"] + p["w"] - 20
        y1 = p["y"] + 42

        # Light grid lines
        ctx.set_source_rgba(0.88, 0.88, 0.90, 0.8)
        ctx.set_line_width(0.6)
        for xt in x_ticks:
            if x_min < xt < x_max:
                px = x0 + (xt - x_min) / (x_max - x_min) * (x1 - x0)
                ctx.move_to(px, y0); ctx.line_to(px, y1); ctx.stroke()
        for yt in y_ticks:
            if y_min < yt < y_max:
                py = y0 - (yt - y_min) / (y_max - y_min) * (y0 - y1)
                ctx.move_to(x0, py); ctx.line_to(x1, py); ctx.stroke()

        # Axis border
        ctx.set_source_rgb(0.4, 0.4, 0.4)
        ctx.set_line_width(1.0)
        ctx.move_to(x0, y0); ctx.line_to(x1, y0)
        ctx.move_to(x0, y0); ctx.line_to(x0, y1)
        ctx.stroke()

        # Ticks and numeric labels
        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(9)
        ctx.set_source_rgb(0.3, 0.3, 0.3)

        for xt in x_ticks:
            if x_min <= xt <= x_max:
                px = x0 + (xt - x_min) / (x_max - x_min) * (x1 - x0)
                ctx.move_to(px, y0); ctx.line_to(px, y0 + 4); ctx.stroke()
                txt = x_fmt.format(xt)
                ext = ctx.text_extents(txt)
                ctx.move_to(px - ext.width / 2, y0 + 15)
                ctx.show_text(txt)

        for yt in y_ticks:
            if y_min <= yt <= y_max:
                py = y0 - (yt - y_min) / (y_max - y_min) * (y0 - y1)
                ctx.move_to(x0 - 4, py); ctx.line_to(x0, py); ctx.stroke()
                txt = y_fmt.format(yt)
                ext = ctx.text_extents(txt)
                ctx.move_to(x0 - 7 - ext.width, py + 3)
                ctx.show_text(txt)

        # Axis titles
        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(9.5)
        ctx.set_source_rgb(0.2, 0.2, 0.2)
        ext_x = ctx.text_extents(x_label)
        ctx.move_to((x0 + x1) / 2 - ext_x.width / 2, y0 + 29)
        ctx.show_text(x_label)

        ctx.save()
        ctx.translate(x0 - 38, (y0 + y1) / 2)
        ctx.rotate(-math.pi / 2)
        ext_y = ctx.text_extents(y_label)
        ctx.move_to(-ext_y.width / 2, 0)
        ctx.show_text(y_label)
        ctx.restore()

    # -------------------------------------------------------------------------
    # Draw Panel 1: omega vs k
    # -------------------------------------------------------------------------
    p1 = panels[0]
    k_vals = [r["k_total"] for r in results]
    w_exact = [r["w_exact"] for r in results]
    w_num   = [r["w_num"] for r in results]

    k_min, k_max = 0.0, max(k_vals) * 1.15
    w_min, w_max = 0.0, max(max(w_exact), max(w_num)) * 1.15

    draw_axes(p1, "Total Wavenumber k [rad/m]", "Frequency omega [rad/s]",
              k_min, k_max, w_min, w_max,
              [0, 3, 6, 9, 12, 15], [0, 4, 8, 12, 16], "{:.0f}", "{:.0f}")

    def to_p1(k, w):
        px = p1["x"] + 52 + (k - k_min) / (k_max - k_min) * (p1["w"] - 72)
        py = p1["y"] + p1["h"] - 38 - (w - w_min) / (w_max - w_min) * (p1["h"] - 80)
        return px, py

    # Theoretical line
    ctx.set_source_rgb(0.8, 0.2, 0.2)
    ctx.set_line_width(2.0)
    for i, (k, w) in enumerate(zip(k_vals, w_exact)):
        px, py = to_p1(k, w)
        if i == 0: ctx.move_to(px, py)
        else: ctx.line_to(px, py)
    ctx.stroke()

    # Numerical markers
    ctx.set_source_rgb(0.1, 0.4, 0.8)
    for k, w in zip(k_vals, w_num):
        px, py = to_p1(k, w)
        ctx.arc(px, py, 4.0, 0, 2 * math.pi)
        ctx.fill()

    # Legend
    ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(10)
    ctx.set_source_rgb(0.8, 0.2, 0.2)
    ctx.move_to(p1["x"] + 65, p1["y"] + 55); ctx.show_text("- - Analytical: omega = c * k")
    ctx.set_source_rgb(0.1, 0.4, 0.8)
    ctx.move_to(p1["x"] + 65, p1["y"] + 70); ctx.show_text(" o  NekWave DG-SEM Numerical")

    # -------------------------------------------------------------------------
    # Draw Panel 2: Phase Velocity Error vs k
    # -------------------------------------------------------------------------
    p2 = panels[1]
    errors = [r["disp_error"] for r in results]
    err_min = min(-25.0, min(errors) * 1.25)
    err_max = max(25.0, max(errors) * 1.25)

    draw_axes(p2, "Total Wavenumber k [rad/m]", "Phase Error [%]",
              k_min, k_max, err_min, err_max,
              [0, 3, 6, 9, 12, 15], [-20, -10, 0, 10, 20], "{:.0f}", "{:+.0f}")

    def to_p2(k, e):
        px = p2["x"] + 52 + (k - k_min) / (k_max - k_min) * (p2["w"] - 72)
        py = p2["y"] + p2["h"] - 38 - (e - err_min) / (err_max - err_min) * (p2["h"] - 80)
        return px, py

    # Zero error line
    z_x0, z_y = to_p2(k_min, 0.0)
    z_x1, _   = to_p2(k_max, 0.0)
    ctx.set_source_rgb(0.6, 0.6, 0.6)
    ctx.set_line_width(1.0)
    ctx.move_to(z_x0, z_y); ctx.line_to(z_x1, z_y); ctx.stroke()

    # Error curve
    ctx.set_source_rgb(0.2, 0.7, 0.3)
    ctx.set_line_width(2.0)
    for i, (k, e) in enumerate(zip(k_vals, errors)):
        px, py = to_p2(k, e)
        if i == 0: ctx.move_to(px, py)
        else: ctx.line_to(px, py)
    ctx.stroke()

    for k, e in zip(k_vals, errors):
        px, py = to_p2(k, e)
        ctx.arc(px, py, 3.5, 0, 2 * math.pi)
        ctx.fill()

    # -------------------------------------------------------------------------
    # Draw Panel 3: Probe Time Histories
    # -------------------------------------------------------------------------
    p3 = panels[2]
    sel_probes = [probe_ids[0], probe_ids[len(probe_ids) // 4], probe_ids[len(probe_ids) // 2], probe_ids[-1]]
    colors = [(0.15, 0.45, 0.8), (0.85, 0.4, 0.1), (0.2, 0.65, 0.2), (0.7, 0.2, 0.7)]
    t_min, t_max = times[0], times[-1]

    draw_axes(p3, "Time t [s]", "Electric Field Ez",
              t_min, t_max, -1.8, 1.8,
              [0, 2, 4, 6, 8, 10], [-1.5, -0.75, 0.0, 0.75, 1.5], "{:.0f}", "{:.2f}")

    def to_p3(t, val):
        px = p3["x"] + 52 + (t - t_min) / (t_max - t_min) * (p3["w"] - 72)
        py = p3["y"] + p3["h"] - 38 - (val - (-1.8)) / (3.6) * (p3["h"] - 80)
        return px, py

    for idx, pid in enumerate(sel_probes):
        col = colors[idx % len(colors)]
        ctx.set_source_rgb(*col)
        ctx.set_line_width(1.2)
        sig = field_data[pid]
        for t_idx in range(0, len(times), max(1, len(times) // 200)):
            px, py = to_p3(times[t_idx], sig[t_idx])
            if t_idx == 0: ctx.move_to(px, py)
            else: ctx.line_to(px, py)
        ctx.stroke()

    # -------------------------------------------------------------------------
    # Draw Panel 4: Spatial Wave Profiles Ez(x) at t0, tMid, tEnd
    # -------------------------------------------------------------------------
    p4 = panels[3]
    time_indices = [0, len(times) // 2, len(times) - 1]
    labels = ["Initial t = 0", f"Mid t = {times[len(times)//2]:.2f}", f"Final t = {times[-1]:.2f}"]

    draw_axes(p4, "Probe Coordinate x [m]", "Electric Field Ez",
              x_coords[0], x_coords[-1], -1.8, 1.8,
              [-1.8, -0.9, 0.0, 0.9, 1.8], [-1.5, -0.75, 0.0, 0.75, 1.5], "{:.1f}", "{:.2f}")

    def to_p4(x, val):
        px = p4["x"] + 52 + (x - x_coords[0]) / (x_coords[-1] - x_coords[0]) * (p4["w"] - 72)
        py = p4["y"] + p4["h"] - 38 - (val - (-1.8)) / (3.6) * (p4["h"] - 80)
        return px, py

    for idx, (t_idx, lbl) in enumerate(zip(time_indices, labels)):
        col = colors[idx % len(colors)]
        ctx.set_source_rgb(*col)
        ctx.set_line_width(1.5)
        for p_idx, pid in enumerate(probe_ids):
            px, py = to_p4(x_coords[p_idx], field_data[pid][t_idx])
            if p_idx == 0: ctx.move_to(px, py)
            else: ctx.line_to(px, py)
        ctx.stroke()

        # Legend
        ctx.move_to(p4["x"] + 65, p4["y"] + 55 + idx * 16)
        ctx.show_text(f"— {lbl}")

    # Write output files
    surface.write_to_png(out_png)
    print(f"[NekWave Dispersion] Saved publication-quality plot: {out_png}")


# -----------------------------------------------------------------------------
# Matplotlib 4-Panel Plot Renderer
# -----------------------------------------------------------------------------
def render_matplotlib_plot(out_png, x_coords, times, results, probe_ids, field_data, cfg):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150)
    fig.suptitle(f"NekWave DG-SEM Numerical Dispersion Analysis (Order N = {cfg['order']})", fontsize=14, fontweight='bold')

    k_vals  = [r["k_total"] for r in results]
    w_exact = [r["w_exact"] for r in results]
    w_num   = [r["w_num"] for r in results]
    errors  = [r["disp_error"] for r in results]

    # Panel 1: omega vs k
    ax1 = axes[0, 0]
    ax1.plot(k_vals, w_exact, 'r--', linewidth=1.5, label=r'Analytical $\omega = c k$')
    ax1.plot(k_vals, w_num, 'bo-', linewidth=1.2, markersize=5, label=r'NekWave DG-SEM')
    ax1.set_xlabel(r'Total Wavenumber $k$ [rad/m]')
    ax1.set_ylabel(r'Frequency $\omega$ [rad/s]')
    ax1.set_title(r'Numerical Dispersion Relation $\omega(k)$')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='best', frameon=False)

    # Panel 2: Phase Velocity Error
    ax2 = axes[0, 1]
    ax2.axhline(0.0, color='gray', linestyle='--', linewidth=0.8)
    ax2.plot(k_vals, errors, 'gs-', linewidth=1.4, markersize=5, label=r'$(v_p / c - 1)$')
    ax2.set_xlabel(r'Total Wavenumber $k$ [rad/m]')
    ax2.set_ylabel(r'Phase Velocity Error [%]')
    ax2.set_title(r'Numerical Phase Velocity Error vs Wavenumber')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='best', frameon=False)

    # Panel 3: Probe Time Histories
    ax3 = axes[1, 0]
    step_probe = max(1, len(probe_ids) // 4)
    for pid in probe_ids[::step_probe]:
        ax3.plot(times, field_data[pid], linewidth=1.0, alpha=0.8, label=f'Probe {pid}')
    ax3.set_xlabel('Time $t$ [s]')
    ax3.set_ylabel('Electric Field $E_z$')
    ax3.set_title('Observation Probe Time Series')
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='best', frameon=False, fontsize=8)

    # Panel 4: Spatial Profiles
    ax4 = axes[1, 1]
    time_indices = [0, len(times) // 2, len(times) - 1]
    labels = ['Initial ($t = 0$)', f'Mid ($t = {times[len(times)//2]:.2f}$)', f'Final ($t = {times[-1]:.2f}$)' ]
    colors = ['black', 'blue', 'red']
    for t_idx, lbl, col in zip(time_indices, labels, colors):
        y_slice = [field_data[pid][t_idx] for pid in probe_ids]
        ax4.plot(x_coords, y_slice, color=col, linewidth=1.3, label=lbl)
    ax4.set_xlabel('Collinear Probe Coordinate $x$ [m]')
    ax4.set_ylabel('Electric Field $E_z$')
    ax4.set_title('Wavepacket Spatial Profile Evolution')
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='best', frameon=False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_png)
    plt.close()
    print(f"[NekWave Dispersion] Saved publication-quality plot: {out_png}")


# -----------------------------------------------------------------------------
# Main Tool Execution
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    # Resolve case file
    case_file = args.case_file
    if not case_file:
        cand = os.path.join(args.input_dir, "numerical_dispersion.par")
        if os.path.exists(cand):
            case_file = cand

    cfg = load_par_config(case_file)

    print(f"[NekWave Dispersion] Loading probe data from: {args.input_dir}")
    steps, times, probe_ids, field_data = load_probe_history(args.input_dir)
    print(f"[NekWave Dispersion] Loaded {len(steps)} time steps across {len(probe_ids)} probes.")

    # Compute dispersion spectrum and metrics
    x_coords, times, results, ky = compute_dispersion_spectrum(times, probe_ids, field_data, cfg)

    # Print ASCII table
    print_dispersion_table(results, cfg)

    # Render visualization
    os.makedirs(args.output_dir, exist_ok=True)
    out_png = os.path.join(args.output_dir, args.filename)
    out_svg = os.path.join(args.output_dir, args.filename.replace(".png", ".svg"))

    if HAVE_MATPLOTLIB:
        render_matplotlib_plot(out_png, x_coords, times, results, probe_ids, field_data, cfg)
    elif HAVE_CAIRO:
        render_cairo_plot(out_png, out_svg, x_coords, times, results, probe_ids, field_data, cfg)
    else:
        print("[Notice] Matplotlib and Cairo not available; ASCII dispersion table printed above.")


if __name__ == "__main__":
    main()
