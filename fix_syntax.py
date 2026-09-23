with open("tools/analytical_dispersion.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'print(f"\\n' in line and line.strip() == 'print(f"\\n':
        new_lines.append('    print(f"\\n[Plotter] Dispersion plot saved to {out_file}")\n')
    else:
        new_lines.append(line)

with open("tools/analytical_dispersion.py", "w") as f:
    f.writelines(new_lines)
