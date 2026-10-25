"""MELO: 1D cortisol diffusion through a PVA hydrogel.

Purpose
-------
Estimate how rapidly cortisol reaches the MIP-facing side of a hydrogel
before sensor fabrication. This is a TRANSPORT-ONLY screening model: no
MIP binding, chemical reaction, swelling, or finite sweat-volume depletion.

Governing equation (Fick's second law):
    dC/dt = D_eff * d²C/dx²

Model boundaries:
    x = 0:  C = C0               (constant-concentration sweat contact)
    x = L:  dC/dx = 0            (impermeable MIP-facing boundary)
    t = 0:  C = 0 inside gel     (initially cortisol-free)

The impermeable boundary does NOT mean that the MIP binds cortisol;
it simply prevents cortisol from leaving the modelled gel domain.

Numerical method: finite differences in space, backward Euler in time.
Each timestep solves a tridiagonal linear system.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

# Save plots and CSV in the same directory as this Python file.
OUT = Path(__file__).resolve().parent

# Cortisol molar mass, in grams per mole.
MW = 362.46


def ngml_to_nM(c_ng_ml):
    """Convert cortisol concentration from ng/mL to nanomolar (nM).

    1 ng/mL = 1e-6 g/L; convert g/L -> mol/L, then multiply by 1e9 nM/M.
    Therefore: concentration[nM] = concentration[ng/mL] * 1000 / MW.
    """
    return c_ng_ml * 1000 / MW


def diffusion(L_um=400, D=1e-10, C0_ngml=20, duration=120, nx=101, dt=0.5):
    """Solve 1D Fickian diffusion for a slab with one no-flux boundary.

    Parameters
    ----------
    L_um : float     Hydrogel thickness in micrometres (default 400 µm).
    D : float        Assumed effective diffusivity, in m²/s.
    C0_ngml : float  Cortisol concentration at sweat-facing surface, ng/mL.
    duration : float Total time to simulate, in seconds.
    nx : int         Number of spatial grid points from x=0 to x=L.
    dt : float       Target timestep in seconds.

    Returns
    -------
    x_um : array     Positions across the hydrogel, in µm.
    times : array    Sampled times, in seconds.
    sol : 2D array   Cortisol concentrations, in nM; sol[time_index, x_index].
    C0 : float       Imposed surface cortisol concentration, in nM.
    """
    # Convert thickness to metres, which matches SI diffusivity units.
    L = L_um * 1e-6

    # Divide gel into nx equally spaced points (including both surfaces).
    x = np.linspace(0, L, nx)
    dx = x[1] - x[0]  # Spatial step, m.

    # Create the time grid; adjust dt slightly to end exactly at duration.
    times = np.linspace(0, duration, int(round(duration / dt)) + 1)
    dt = times[1] - times[0]

    # Dimensionless diffusion number for a single time step.
    # Implicit backward Euler is stable even for r > 0.5, unlike explicit Euler.
    r = D * dt / dx**2

    # Convert the imposed boundary concentration to nM.
    C0 = ngml_to_nM(C0_ngml)

    # Initial condition: no cortisol in the hydrogel except at x=0.
    c = np.zeros(nx)
    c[0] = C0

    # Allocate all results: one concentration profile per time point.
    sol = np.zeros((len(times), nx))
    sol[0] = c

    # We already know c[0] = C0, so solve only for c[1], ..., c[nx-1].
    # For interior point j, backward Euler gives:
    # -r*c_new[j-1] + (1+2r)*c_new[j] - r*c_new[j+1] = c_old[j]
    n = nx - 1

    # SciPy's banded representation of the tridiagonal coefficient matrix:
    # band[0, 1:] = superdiagonal (values above the main diagonal)
    # band[1, :]  = main diagonal
    # band[2, :-1]= subdiagonal (values below the main diagonal)
    band = np.zeros((3, n))
    band[0, 1:] = -r
    band[1, :] = 1 + 2*r
    band[2, :-1] = -r

    # At the MIP-facing surface (x=L), enforce zero diffusive flux:
    # dC/dx = 0. With a mirrored ghost point, the second derivative becomes
    # 2*(c[nx-2] - c[nx-1])/dx².
    # Thus the last equation has coefficient -2r on the previous point.
    # In the banded format, the final subdiagonal is at band[2, -2].
    band[2, -2] = -2*r

    # Progress through time by solving the implicit diffusion equations.
    for i in range(1, len(times)):
        rhs = c[1:].copy()  # Concentrations from previous timestep.

        # At the first unknown node (x=dx), its left neighbour is the known
        # fixed boundary c[0]=C0; move -r*C0 to right-hand side as +r*C0.
        rhs[0] += r*C0

        # Solve the tridiagonal system efficiently for new concentrations.
        c[1:] = solve_banded((1, 1), band, rhs)
        c[0] = C0  # Re-impose the sweat-side constant-concentration condition.
        sol[i] = c

    # Return depth in µm (easy to interpret on plots), time in s, and C in nM.
    return x*1e6, times, sol, C0


if __name__ == '__main__':
    # This block runs only when the file is executed directly.
    # It creates three figures and a CSV table for your verification report.

    # ---------- FIGURE 1: concentration versus depth ----------
    x, t, C, C0 = diffusion()  # Default: L=400 µm, D=1e-10 m²/s.
    fig, ax = plt.subplots(figsize=(7.4, 4.7))

    for ti in (0, 15, 30, 60, 90, 120):
        # Find the simulated time nearest the chosen time ti.
        idx = np.argmin(np.abs(t - ti))
        # Normalize by applied concentration: 1 = 100% of C0.
        ax.plot(x, C[idx]/C0, label=f'{ti} s')

    ax.set(xlabel='Depth into hydrogel (µm)',
           ylabel='Normalized cortisol concentration C/C₀',
           title='MELO: concentration profiles (L = 400 µm, D = 10⁻¹⁰ m²/s)')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT/'concentration_profiles.png', dpi=180)

    # ---------- FIGURE 2: change diffusivity, hold thickness fixed ----------
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    Ds = [1e-10, 2.5e-10, 5e-10, 1e-9]  # Illustrative, NOT measured D values.

    for d in Ds:
        _, ts, cp, c0 = diffusion(D=d)
        # cp[:, -1] takes the final spatial point (MIP side) at ALL times.
        ax.plot(ts, cp[:, -1]/c0, label=f'D = {d:.1e} m²/s')

    ax.axvline(120, color='grey', linestyle='--', alpha=.7)
    ax.set(xlabel='Incubation time (s)',
           ylabel='C at MIP boundary / C₀',
           title='MELO: diffusivity sensitivity (L = 400 µm)')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT/'diffusivity_sweep.png', dpi=180)
   
    # ---------- FIGURE 3: change thickness, hold diffusivity fixed ----------
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    for l in [200, 300, 400, 500, 600]:
        _, ts, cp, c0 = diffusion(L_um=l, D=1e-10)
        ax.plot(ts, cp[:, -1]/c0, label=f'{l} µm')

    ax.set(xlabel='Incubation time (s)',
           ylabel='C at MIP boundary / C₀',
           title='MELO: gel thickness sensitivity (D = 10⁻¹⁰ m²/s)')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT/'thickness_sweep.png', dpi=180)
    
    plt.show()
    

    # ---------- NUMERICAL RESULTS / EXPORT ----------
    print('Baseline C0 (nM):', round(C0, 3))
    for d in Ds:
        _, _, cp, c0 = diffusion(D=d)
        # [-1,-1] is the concentration at the LAST time and LAST depth.
        print('D = %.2g m²/s: C_MIP/C0 at 120s = %.4f'
              % (d, cp[-1, -1]/c0))

    # Export the baseline depth profiles at selected time points as a CSV.
    # Each row is a depth, and each column is concentration at one time.
    selected_times = (0, 30, 60, 90, 120)
    selected_idx = [np.argmin(np.abs(t - ti)) for ti in selected_times]
    table = np.column_stack([x, C[selected_idx].T])
    np.savetxt(OUT/'baseline_profiles.csv', table, delimiter=',',
               header='depth_um,C_0s_nM,C_30s_nM,C_60s_nM,C_90s_nM,C_120s_nM',
               comments='')
# MELO Capstone Project - Hydrogel Transport Simulation