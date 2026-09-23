import re

with open("tools/analytical_dispersion.py", "r") as f:
    content = f.read()

# 1. Update load_matrices to return order and C0
old_loader = r'    dx = data\["dx"\]'
new_loader = r'''    dx = data["dx"]
    order = data.get("order", "Unknown")
    c0 = data.get("C0", "Unknown")'''
content = re.sub(old_loader, new_loader, content)

old_loader_ret = r'    return M, S, neighbors, dx'
new_loader_ret = r'    return M, S, neighbors, dx, order, c0'
content = re.sub(old_loader_ret, new_loader_ret, content)

# 2. Update plot_dispersion definition
old_plot_def = r'def plot_dispersion\(results\):'
new_plot_def = r'def plot_dispersion(results, order, c0):'
content = re.sub(old_plot_def, new_plot_def, content)

# 3. Update title
old_title = r'    plt.title\("Analytical DGTD Numerical Dispersion \(Bloch-Floquet\)"\)'
new_title = r'    plt.title(f"Analytical DGTD Dispersion (Bloch-Floquet)\\nOrder $N={order}$, Flux $C_0={c0}$")'
content = re.sub(old_title, new_title, content)

# 4. Update main
old_main_call1 = r'    M, S, neighbors, dx = load_matrices\(filename\)'
new_main_call1 = r'    M, S, neighbors, dx, order, c0 = load_matrices(filename)'
content = re.sub(old_main_call1, new_main_call1, content)

old_main_call2 = r'    plot_dispersion\(results\)'
new_main_call2 = r'    plot_dispersion(results, order, c0)'
content = re.sub(old_main_call2, new_main_call2, content)

with open("tools/analytical_dispersion.py", "w") as f:
    f.write(content)
