import re

with open("tools/numerical_dispersion.py", "r") as f:
    content = f.read()

# 1. Update Matplotlib subplot 1
old_mpl = """            k_vals = [2.0 * math.pi * m for m in valid_modes]
            v_rats = [data[m][0] for m in valid_modes]
            ax1.plot(k_vals, v_rats, 'o-', color=hex_colors[idx % len(hex_colors)],
                     linewidth=2.0, markersize=7, label=f"$\\\\theta = {ang:.0f}^\\\\circ$ (CUDA)")

        ax1.set_xlabel(r'Normalized Wavenumber $k \\cdot \\Delta x$ [rad]', fontsize=11, fontweight='bold')
        ax1.set_ylabel(r'Normalized Phase Velocity $v_p / c$', fontsize=11, fontweight='bold')
        ax1.set_ylim([0.30, 1.6])"""

new_mpl = """            ppw_vals = [(order - 1) / float(m) for m in valid_modes]
            v_rats = [data[m][0] for m in valid_modes]
            ax1.plot(ppw_vals, v_rats, 'o-', color=hex_colors[idx % len(hex_colors)],
                     linewidth=2.0, markersize=7, label=f"$\\\\theta = {ang:.0f}^\\\\circ$ (CUDA)")

        ax1.set_xlabel(r'Points Per Wavelength (PPW = $\\lambda / \\Delta x$)', fontsize=11, fontweight='bold')
        ax1.set_ylabel(r'Normalized Phase Velocity $v_p / c$', fontsize=11, fontweight='bold')
        # Invert x-axis to match analytical plot (high PPW to low PPW)
        ax1.set_xlim(ax1.get_xlim()[::-1])
        ax1.set_ylim([0.30, 1.6])"""

content = content.replace(old_mpl, new_mpl)


# 2. Update Cairo section
old_cairo = """        # ---------------------------------------------------------
        # Fallback Plotter: Pure Python + Cairo (No Matplotlib)
        # ---------------------------------------------------------
        print("[Plotter] Matplotlib not found. Falling back to basic Cairo PNG rendering.")
        import cairo

        width, height = 800, 600
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(surface)

        # White background
        ctx.set_source_rgb(1, 1, 1)
        ctx.paint()

        # Grid box
        x0, y0 = 100, height - 100
        pw, ph = width - 150, height - 180
        x1, y1 = x0 + pw, y0 - ph

        ctx.set_source_rgb(0, 0, 0)
        ctx.set_line_width(1.5)
        ctx.rectangle(x0, y1, pw, ph)
        ctx.stroke()

        # Title
        ctx.set_font_size(16)
        ctx.move_to(x0, y1 - 30)
        ctx.show_text(f"NekWave Dispersion (Order {order}, {flux_lbl})")

        # Labels
        ctx.set_font_size(14)
        ctx.move_to(x0, y0 + 40)
        ctx.show_text("Normalized Wavenumber k*dx [rad]")
        ctx.move_to(x0 - 50, y0 - ph/2)
        ctx.save()
        ctx.rotate(-math.pi/2)
        ctx.show_text("v_p / c")
        ctx.restore()

        kmin, kmax = 0.0, 30.0
        ymin, ymax = 0.3, 1.6"""

new_cairo = """        # ---------------------------------------------------------
        # Fallback Plotter: Pure Python + Cairo (No Matplotlib)
        # ---------------------------------------------------------
        print("[Plotter] Matplotlib not found. Falling back to basic Cairo PNG rendering.")
        import cairo

        width, height = 800, 600
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(surface)

        # White background
        ctx.set_source_rgb(1, 1, 1)
        ctx.paint()

        # Grid box
        x0, y0 = 100, height - 100
        pw, ph = width - 150, height - 180
        x1, y1 = x0 + pw, y0 - ph

        ctx.set_source_rgb(0, 0, 0)
        ctx.set_line_width(1.5)
        ctx.rectangle(x0, y1, pw, ph)
        ctx.stroke()

        # Title
        ctx.set_font_size(16)
        ctx.move_to(x0, y1 - 30)
        ctx.show_text(f"NekWave Dispersion (Order {order}, {flux_lbl})")

        # Labels
        ctx.set_font_size(14)
        ctx.move_to(x0, y0 + 40)
        ctx.show_text("Points Per Wavelength (PPW)")
        ctx.move_to(x0 - 50, y0 - ph/2)
        ctx.save()
        ctx.rotate(-math.pi/2)
        ctx.show_text("v_p / c")
        ctx.restore()

        kmin, kmax = 2.0, max(12.0, order) # Acts as ppw_min, ppw_max
        ymin, ymax = 0.3, 1.6"""

content = content.replace(old_cairo, new_cairo)


old_cairo_plot = """            for i, m in enumerate(modes):
                kx = 2.0 * math.pi * m
                v_val = data[m][0]
                px = x0 + (kx - kmin) / (kmax - kmin) * pw
                py = y0 - (v_val - ymin) / (ymax - ymin) * ph
                py = max(y1, min(y0, py))
                if i == 0: ctx.move_to(px, py)
                else: ctx.line_to(px, py)
            ctx.stroke()

            # Draw markers
            for m in modes:
                kx = 2.0 * math.pi * m
                v_val = data[m][0]
                px = x0 + (kx - kmin) / (kmax - kmin) * pw
                py = y0 - (v_val - ymin) / (ymax - ymin) * ph
                py = max(y1, min(y0, py))
                ctx.arc(px, py, 4.0, 0, 2 * math.pi)
                ctx.fill()"""

new_cairo_plot = """            for i, m in enumerate(modes):
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
                ctx.fill()"""

content = content.replace(old_cairo_plot, new_cairo_plot)

with open("tools/numerical_dispersion.py", "w") as f:
    f.write(content)
