import numpy as np
from scipy.integrate import solve_ivp


BAR = 1e5
RS = 518.3  # J/(kg K), illustrative methane-rich natural gas
TEMP = 288.15  # K
Z = 1.0  # idealized constant compressibility factor
FRICTION = 0.014  # prescribed Darcy friction factor, not calculated from Re

# Edges are oriented purely for sign convention.
EDGES = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 4), (2, 5),
         (3, 4), (3, 6), (4, 5), (4, 6), (5, 7), (6, 7)]
# Physical properties are illustrative, not data for an actual gas grid.
LENGTH_M = np.array([1800, 2100, 1700, 1900, 1500, 1600, 900, 1800, 1250, 1950, 1400, 1600], float)
DIAMETER_M = np.array([.23, .23, .18, .18, .18, .18, .15, .16, .16, .16, .15, .16], float)
N = 8
M = len(EDGES)
SUPPLY_BAR = 10.0  # absolute pressure
DEMAND = np.array([.45, .55, .35, .68, .57, .43, .37])  # kg/s at nodes 1,...,7
POSITIONS = {0: (0., 1.), 1: (1.1, 2.), 2: (1.05, 0.), 3: (2.3, 2.1),
             4: (2.35, 1.), 5: (2.3, -.15), 6: (3.5, 1.9), 7: (3.6, .05)}

B = np.zeros((N, M))
for e, (i, j) in enumerate(EDGES):
    B[i, e] = 1.0  # flow leaving node is positive
    B[j, e] = -1.0


def resistance(length_m=LENGTH_M, diameter_m=DIAMETER_M,
               friction=FRICTION, temperature=TEMP, rs=RS, z=Z):
    """K [Pa^2/(kg/s)^2] for p_i^2-p_j^2 = K m|m|.

    Derivation uses steady isothermal Darcy-Weisbach with constant Z and f,
    neglecting gravity and acceleration; A=pi D²/4.
    """
    length_m = np.asarray(length_m, dtype=float)
    diameter_m = np.asarray(diameter_m, dtype=float)
    A = np.pi * diameter_m ** 2 / 4
    return friction * z * rs * temperature * length_m / (diameter_m * A * A)


def transient_linepack(num_cells=6, L=15000., D=.30, supply_bar=40.,
                       f=.012, base_demand=4., peak_demand=6.,
                       t_step=1200., t_recovery=3600., t_end=7200.):
    """Spatially lumped linepack demonstration with compressible storage.

    Source node pressure is prescribed. N downstream control volumes each
    store M_i = p_i * V_i /(Z R T). Each link uses the same steady isothermal
    friction law over a length L/N. Demands are at the end node.

    This is a deliberately simplified RC/compartment model, *not* full
    transient gas momentum equations. It supports direction reversal.
    """
    A = np.pi * D * D / 4
    cell_V = A * L / num_cells
    cap = cell_V / (Z * RS * TEMP)  # kg per Pa of pressure
    Kseg = resistance(np.array([L / num_cells]), np.array([D]),
                      friction=f)[0]
    supply_pa = supply_bar * BAR
    # Initial steady condition satisfies each resistive segment at base flow.
    p2 = supply_pa ** 2 - np.arange(1, num_cells + 1) * Kseg * base_demand ** 2
    if np.any(p2 <= 0): raise ValueError('Unphysical initial pressure squared')
    initial_p = np.sqrt(p2)

    def flux(p_vec):
        p_all = np.r_[supply_pa, p_vec]
        dp2 = p_all[:-1] ** 2 - p_all[1:] ** 2
        return np.sign(dp2) * np.sqrt(np.abs(dp2) / Kseg)

    def rhs(_, p_vec, d):
        q = flux(p_vec)
        incoming = q
        outgoing = np.r_[q[1:], d]
        return (incoming - outgoing) / cap

    chunks = []
    phases = [(0., t_step, base_demand), (t_step, t_recovery, peak_demand),
              (t_recovery, t_end, base_demand)]
    y0 = initial_p
    for a, b, d in phases:
        tgrid = np.linspace(a, b, max(3, int((b - a) / 20) + 1))
        sol = solve_ivp(lambda t, y: rhs(t, y, d), (a, b), y0, method='BDF',
                        t_eval=tgrid, rtol=1e-7, atol=1e-3, max_step=30)
        if not sol.success: raise RuntimeError(sol.message)
        qq = np.array([flux(sol.y[:, k]) for k in range(sol.y.shape[1])]).T
        chunks.append((sol.t, sol.y, qq, np.full(len(sol.t), d)))
        y0 = sol.y[:, -1]
    t = np.concatenate([v[0][(0 if k == 0 else 1):] for k, v in enumerate(chunks)])
    p = np.hstack([v[1][:, (0 if k == 0 else 1):] for k, v in enumerate(chunks)]) / BAR
    flow = np.hstack([v[2][:, (0 if k == 0 else 1):] for k, v in enumerate(chunks)])
    demands = np.concatenate([v[3][(0 if k == 0 else 1):] for k, v in enumerate(chunks)])
    inventory = cap * np.sum(p * BAR, axis=0)
    # hypothetical instantaneous steady-state response to same withdrawal
    p2_ss = supply_pa ** 2 - num_cells * Kseg * demands ** 2
    p_end_quasisteady = np.sqrt(np.maximum(p2_ss, 0)) / BAR
    assert np.all(p > 0)
    return {'time_s': t, 'pressure_bar': p, 'flow_kg_s': flow,
            'demand_kg_s': demands, 'inventory_kg': inventory,
            'quasisteady_end_bar': p_end_quasisteady,
            'initial_mass_kg': inventory[0], 'storage_coefficient_kg_pa': cap,
            'Kseg_pa2': Kseg, 'volume_m3': cell_V * num_cells,
            'initial_pressure_pa': initial_p, 'num_cells': num_cells,
            'source_bar': supply_bar}


if __name__ == '__main__':
    run = transient_linepack()
    print('Pipe length 15 km; diameter 0.30 m; source 40 bar(a)')
    print('Pipe volume [m³]:', round(run['volume_m3'], 1))
    print('Initial gas inventory [tonnes]:', round(run['initial_mass_kg'] / 1000, 2))
