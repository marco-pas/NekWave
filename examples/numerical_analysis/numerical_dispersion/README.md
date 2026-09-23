# Two-Probe Multi-Harmonic Numerical Dispersion Benchmark

This benchmark performs time-domain numerical dispersion analysis of the NekWave 3D DGTD CUDA solver using **collinear point observation probes** and **multi-harmonic discrete modal extraction**.

---

## 1. Overview & Physical Setup

In a numerical wave simulation, spatial discretization (discontinuous Galerkin spectral elements) and time integration (Runge-Kutta) introduce artificial phase errors ($v_p \neq c$). To measure these dispersion errors directly from time-domain GPU simulations, this benchmark propagates a composite multi-mode electromagnetic wave through a 1D periodic channel embedded in 3D:

* **Governing Equations:** 3D Maxwell TM system ($\partial_t E_z = \partial_x H_y - \partial_y H_x$, $\partial_t H_y = \partial_x E_z$).
* **Computational Domain:** $L_x = 16\pi \approx 50.26548246\text{ m}$, $L_y = 1.0\text{ m}$, $L_z = 1.0\text{ m}$.
* **Periodic Boundaries:** Exact periodic boundary conditions across all faces ($x, y, z$).
* **Mesh Discretization:** 64 elements along $x$ ($E_x = 64$), 1 element along $y$ and $z$.
* **Observation Probes:** Two point probes located at $(24.5, 0.5, 0.5)$ and $(25.5, 0.5, 0.5)$ with exact separation distance $d = 1.0\text{ m}$. Field values at probes are evaluated using high-order spectral element Lagrange polynomial interpolation ($O(\Delta x^N)$ accuracy).

---

## 2. Multi-Harmonic Periodic Plane Wave Excitation

Rather than running a single frequency per simulation, a composite wave packet exciting $M = 5$ linearly spaced spatial wavenumbers is initialized simultaneously:

$$E_z(x, y, z, 0) = \frac{1}{\sqrt{M}} \sum_{m=1}^M \cos(k_m x), \qquad H_y(x, y, z, 0) = -E_z(x, y, z, 0)$$

### Exact Periodic Boundary Continuity ($C^\infty$)
To prevent unphysical high-frequency jump discontinuities at periodic boundaries $x = 0$ and $x = L_x$, every excited wavenumber $k_m$ must have an integer number of full wave cycles $n_m \in \mathbb{Z}^+$ spanning the domain length $L_x$:

$$k_m = \frac{2\pi n_m}{L_x}$$

For $M = 5$ modes linearly spaced from $k_{\text{start}} = 0.25\text{ rad/m}$ to $k_{\text{end}} = 3.75\text{ rad/m}$ ($\Delta k = 0.875\text{ rad/m}$):
$$\gcd(0.25, 1.125, 2.00, 2.875, 3.75) = 0.125\text{ rad/m} = \frac{1}{8}\text{ rad/m}$$
$$\implies L_x = \frac{2\pi}{0.125} = 16\pi \approx 50.26548246\text{ m}$$

| Mode $m$ | Wavenumber $k$ [rad/m] | Wavelength $\lambda = 2\pi/k$ [m] | Full Cycles across $L_x = 16\pi$ ($n$) | Periodic Continuity |
| :---: | :---: | :---: | :---: | :---: |
| **1** | $0.250$ | $25.1327$ | **2** | Exact ($C^\infty$) |
| **2** | $1.125$ | $5.5851$ | **9** | Exact ($C^\infty$) |
| **3** | $2.000$ | $3.1416$ | **16** | Exact ($C^\infty$) |
| **4** | $2.875$ | $2.1855$ | **23** | Exact ($C^\infty$) |
| **5** | $3.750$ | $1.6755$ | **30** | Exact ($C^\infty$) |

With 64 elements, even at polynomial order $N = 2$ (linear elements, 1 DOF/elem = 64 DOFs), mode 5 ($n = 30$ cycles) is resolved above the Nyquist limit ($64 > 2 \times 30 = 60$). For $N \ge 3$, the DOFs are $\ge 128$, providing high resolution across all modes.

---

## 3. High-Resolution Signal Processing & Modal Phase Velocity

The post-processing script [`plot_two_probe_dispersion.py`](plot_two_probe_dispersion.py) extracts the numerical frequency and phase velocity of each mode without cross-talk:

1. **DC Bias Removal:**
   $$s_{\text{AC}}(t) = s(t) - \bar{s}$$
2. **Blackman-Harris 4-Term Windowing:**
   Eliminates spectral leakage between adjacent modes with $-92\text{ dB}$ sidelobe attenuation:
   $$w[n] = 0.35875 - 0.48829 \cos\left(\frac{2\pi n}{N_t-1}\right) + 0.14128 \cos\left(\frac{4\pi n}{N_t-1}\right) - 0.01168 \cos\left(\frac{6\pi n}{N_t-1}\right)$$
3. **Zero-Padded FFT:**
   Zero-pads the signal to $N_{\text{fft}} \ge 65536$ points, providing fine frequency sampling.
4. **Sub-Bin Parabolic Interpolation:**
   Locates the exact continuous peak frequency $\omega_{\text{num}}(m)$ around each expected harmonic:
   $$\delta = \frac{1}{2} \frac{\alpha - \gamma}{\alpha - 2\beta + \gamma}, \qquad \omega_{\text{num}} = \omega_{\text{peak}} + \delta \Delta\omega$$
5. **Modal Phase Velocity:**
   Since spatial wavenumber $k_m$ is fixed by the periodic box dimensions, the numerical phase velocity is:
   $$v_p(m) = \frac{\omega_{\text{num}}(m)}{k_m}$$
6. **Points Per Wavelength (PPW):**
   Evaluated with respect to the effective nodal spacing $\Delta x_{\text{nodal}} = L_x / (E_x \cdot (N-1))$:
   $$\text{PPW}(m) = \frac{2\pi}{k_m \, \Delta x_{\text{nodal}}}$$

---

## 4. Parameter File Configuration (`numerical_dispersion.par`)

```ini
# Geometry & Periodic Domain
elements_x = 64
elements_y = 1
elements_z = 1
order = 8

L_x = 50.26548245743669
L_y = 1.0
L_z = 1.0

xmin = 0.0
xmax = 50.26548245743669
ymin = 0.0
ymax = 1.0
zmin = 0.0
zmax = 1.0

periodic_x = true
periodic_y = true
periodic_z = true

# Time-stepping & output
cfl = auto
dt = auto
num_steps = 40000
output_freq = 5
C0 = 0.5
output_dir = output

# Collinear observation probes
probe = 24.5 0.5 0.5
probe = 25.5 0.5 0.5

# Multi-harmonic Bloch excitation
wave_type = bloch
k_start = 0.25
k_end = 3.75
num_modes = 5
angle_deg = 0.0
```

---

## 5. Usage & Execution

### Single Run
```bash
# 1. Compile C++ / CUDA solver
cmake --build build -j4

# 2. Run simulation
./build/ctests/numerical_dispersion/numerical_dispersion \
    examples/numerical_analysis/numerical_dispersion/numerical_dispersion.par

# 3. Analyze probe time series and generate plot
python3 examples/numerical_analysis/numerical_dispersion/plot_two_probe_dispersion.py \
    build/output/probe_history.csv results/dispersion_N8_C0.5.png
```

### Full Benchmark Sweep
Run the automated parameter sweep across orders $N \in \{2 \dots 10\}$ and numerical fluxes $C_0 \in \{0.0, 0.5, 1.0\}$:
```bash
./examples/numerical_analysis/numerical_dispersion/run_sweep.sh
```
All generated figures are saved to `examples/numerical_analysis/numerical_dispersion/results/`.

---

## 6. Visual Output

Each generated plot displays two panels:
* **Left Panel ($v_p / c$ vs. PPW):** Displays normalized phase velocity against Points Per Wavelength on a logarithmic scale from $628$ to $2$. Shows convergence toward exact velocity ($v_p/c = 1.0$) as PPW increases.
* **Right Panel ($\omega$ vs. $k$):** Displays the numerical dispersion curve $\omega(k)$ against exact physical dispersion $\omega = ck$ over $k \in [0, 4.0]\text{ rad/m}$ and $\omega \in [0, 4.0]\text{ rad/s}$.
