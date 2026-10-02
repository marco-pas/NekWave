#!/usr/bin/env python3
"""
NekWave: Multi-Curve High-Order Convergence Plotting Utility
============================================================
Generates 3 distinct figures for C0 in {0.0, 0.5, 1.0} simulating 10 periods:
  - Each figure contains 2 subplots:
      1. Left subplot (h-refinement): 2 lines -> Degree P=3 and Degree P=6
      2. Right subplot (p-refinement): 2 lines -> Mesh 2x2x1 and Mesh 8x8x1
Saves figures directly into output/ directory in PNG and PDF formats.

Usage:
    python3 plot_convergence.py [output_dir]
"""

import sys
import json
import os
import glob
import shutil
import numpy as np

def plot_single_c0_figure(json_file, out_dir):
    with open(json_file, "r") as f:
        data = json.load(f)

    C0 = float(data.get("C0", 0.0))
    c0_str = f"{C0:.1f}"
    num_periods = data.get("num_periods", 10)

    h_line0 = data.get("h_refinement_N3", [])
    h_line1 = data.get("h_refinement_N5") or data.get("h_refinement_P3", [])
    h_line2 = data.get("h_refinement_N8") or data.get("h_refinement_P6", [])
    p_line1 = data.get("p_refinement_mesh4") or data.get("p_refinement_mesh2", [])
    p_line2 = data.get("p_refinement_mesh16") or data.get("p_refinement_mesh8", [])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not installed. Skipping plot generation.")
        return

    # Styling settings
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "lines.linewidth": 2.2,
        "lines.markersize": 8,
        "grid.alpha": 0.35,
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if C0 == 0.0:
        flux_label = "Central Flux ($C_0 = 0.0$, Energy-Conserving)"
    elif C0 == 1.0:
        flux_label = "Upwind Flux ($C_0 = 1.0$, Dissipative)"
    else:
        flux_label = f"Intermediate Flux ($C_0 = {C0:.1f}$)"

    # --------------------------------------------------------------------------
    # Subplot 1: h-Refinement (Lines: N=3, N=5, N=8)
    # --------------------------------------------------------------------------
    ax1 = axes[0]

    # Line 0: N=3 (P=2)
    if h_line0:
        N0 = h_line0[0]["N"]
        P0 = h_line0[0].get("degree_P", N0 - 1)
        h_vals0 = np.array([item["h"] for item in h_line0])
        l2_0 = np.array([item["l2_error"] for item in h_line0])
        ax1.loglog(h_vals0, l2_0, "^-", color="#2ca02c", label=rf"Order $N = {N0}$ ($P = {P0}$)", lw=2.4)

        # Reference slope O(h^N0)
        ref_h = np.linspace(h_vals0.min(), h_vals0.max(), 50)
        mid_idx = len(h_vals0) // 2
        ref_scale0 = l2_0[mid_idx] / (h_vals0[mid_idx] ** N0)
        ax1.loglog(ref_h, ref_scale0 * (ref_h ** N0), ":", color="#2ca02c", alpha=0.7, label=rf"Theoretical $\mathcal{{O}}(h^{{{N0}}})$")

    # Line 1: N=5 (P=4)
    if h_line1:
        N1 = h_line1[0]["N"]
        P1 = h_line1[0].get("degree_P", N1 - 1)
        h_vals1 = np.array([item["h"] for item in h_line1])
        l2_1 = np.array([item["l2_error"] for item in h_line1])
        ax1.loglog(h_vals1, l2_1, "o-", color="#1f77b4", label=rf"Order $N = {N1}$ ($P = {P1}$)", lw=2.4)

        # Reference slope O(h^N1)
        ref_h = np.linspace(h_vals1.min(), h_vals1.max(), 50)
        mid_idx = len(h_vals1) // 2
        ref_scale1 = l2_1[mid_idx] / (h_vals1[mid_idx] ** N1)
        ax1.loglog(ref_h, ref_scale1 * (ref_h ** N1), ":", color="#1f77b4", alpha=0.7, label=rf"Theoretical $\mathcal{{O}}(h^{{{N1}}})$")

    # Line 2: N=8 (P=7)
    if h_line2:
        N2 = h_line2[0]["N"]
        P2 = h_line2[0].get("degree_P", N2 - 1)
        h_vals2 = np.array([item["h"] for item in h_line2])
        l2_2 = np.array([item["l2_error"] for item in h_line2])
        ax1.loglog(h_vals2, l2_2, "s-", color="#d62728", label=rf"Order $N = {N2}$ ($P = {P2}$)", lw=2.4)

        # Reference slope O(h^N2)
        valid_indices = [i for i, v in enumerate(l2_2) if v > 1e-12]
        if valid_indices:
            ref_idx = valid_indices[len(valid_indices) // 2]
            ref_scale2 = l2_2[ref_idx] / (h_vals2[ref_idx] ** N2)
            ax1.loglog(ref_h, ref_scale2 * (ref_h ** N2), ":", color="#d62728", alpha=0.7, label=rf"Theoretical $\mathcal{{O}}(h^{{{N2}}})$")

    ax1.set_xlabel("Element Size $h = 2 / K$")
    ax1.set_ylabel(r"$L_2$ Error Norm at $t = 10\,T$")
    ax1.set_title(r"$h$-Refinement (Spatial Grid Convergence)")
    ax1.grid(True, which="both", ls="--")
    ax1.legend(frameon=True, loc="lower right")


    # --------------------------------------------------------------------------
    # Subplot 2: p-Refinement (2 Lines: e.g. Mesh 4x4x1 and Mesh 16x16x1)
    # --------------------------------------------------------------------------
    ax2 = axes[1]

    # Line 1: Coarse Mesh (e.g. 4x4x1)
    if p_line1:
        m1_str = f"{p_line1[0]['nelx']}\\times{p_line1[0]['nely']}\\times{p_line1[0]['nelz']}"
        deg_m1 = np.array([item["degree_P"] for item in p_line1])
        l2_m1 = np.array([item["l2_error"] for item in p_line1])
        ax2.semilogy(deg_m1, l2_m1, "o-", color="#2ca02c", label=rf"Coarse Mesh (${m1_str}$)", lw=2.4)

    # Line 2: Refined Mesh (e.g. 16x16x1)
    if p_line2:
        m2_str = f"{p_line2[0]['nelx']}\\times{p_line2[0]['nely']}\\times{p_line2[0]['nelz']}"
        deg_m2 = np.array([item["degree_P"] for item in p_line2])
        l2_m2 = np.array([item["l2_error"] for item in p_line2])
        ax2.semilogy(deg_m2, l2_m2, "s-", color="#9467bd", label=rf"Refined Mesh (${m2_str}$)", lw=2.4)

    ax2.set_xlabel("Polynomial Degree $P = N - 1$")
    ax2.set_ylabel(r"$L_2$ Error Norm at $t = 10\,T$")
    ax2.set_title(r"$p$-Refinement (Spectral Order Convergence)")
    ax2.grid(True, which="both", ls="--")
    if p_line1:
        ax2.set_xticks(deg_m1)
    ax2.legend(frameon=True, loc="upper right")


    fig.suptitle(f"NekWave DG-SEM Convergence (10 Wave Periods) — {flux_label}", y=0.98, fontsize=15)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    out_png = os.path.join(out_dir, f"convergence_study_C0_{c0_str}.png")
    out_pdf = os.path.join(out_dir, f"convergence_study_C0_{c0_str}.pdf")

    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close(fig)

    print(f"[INFO] Generated Figure: {out_png}")
    print(f"[INFO] Generated Vector PDF: {out_pdf}")

def main():
    target_out_dir = sys.argv[1] if len(sys.argv) > 1 else "examples/numerical_analysis/convergence_test/output"
    os.makedirs(target_out_dir, exist_ok=True)

    # Search candidates for JSON files in multiple locations
    search_dirs = [
        target_out_dir,
        "output",
        "examples/numerical_analysis/convergence_test/output",
        "."
    ]

    json_files = []
    found_dir = None
    for sdir in search_dirs:
        if os.path.exists(sdir):
            cands = sorted(glob.glob(os.path.join(sdir, "convergence_results_C0_*.json")))
            if cands:
                json_files = cands
                found_dir = sdir
                break

    if not json_files:
        print(f"[WARN] No convergence_results_C0_*.json found in {search_dirs}. Searching for single JSON files...")
        for sdir in search_dirs:
            if os.path.exists(sdir):
                cands = sorted(glob.glob(os.path.join(sdir, "*.json")))
                if cands:
                    json_files = cands
                    found_dir = sdir
                    break

    if not json_files:
        print("[ERROR] No JSON convergence result files found in any search path.")
        sys.exit(1)

    print(f"[INFO] Found {len(json_files)} result file(s) in '{found_dir}'.")
    print(f"[INFO] Output figures will be saved to: '{target_out_dir}'")

    for jf in json_files:
        # Copy JSON file into target_out_dir if not already there
        dest_json = os.path.join(target_out_dir, os.path.basename(jf))
        if os.path.abspath(jf) != os.path.abspath(dest_json):
            shutil.copy2(jf, dest_json)
        plot_single_c0_figure(dest_json, target_out_dir)

if __name__ == "__main__":
    main()
