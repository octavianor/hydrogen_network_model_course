import numpy as np
from scipy.optimize import least_squares


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


def resistance(length_m, diameter_m,
               friction, temperature, rs, z):
    """K [Pa^2/(kg/s)^2] for p_i^2-p_j^2 = K m|m|.

    Derivation uses steady isothermal Darcy-Weisbach with constant Z and f,
    neglecting gravity and acceleration; A=pi D²/4.
    """
    length_m = np.asarray(length_m, dtype=float)
    diameter_m = np.asarray(diameter_m, dtype=float)
    A = np.pi * diameter_m ** 2 / 4
    return friction * z * rs * temperature * length_m / (diameter_m * A * A)


def steady_solution(demand, supply_bar, length_m,
                    diameter_m, friction, temperature, rs, z, tol=2e-7):
    """Solve nodal conservation and pressure-square friction simultaneously.

    Unknowns = M signed mass flows (kg/s) + N-1 squared absolute pressures
    (bar²). Source pressure prescribed at node zero.  The unique reference
    pressure removes the pressure datum ambiguity.
    """
    demand = np.asarray(demand, float)
    if demand.shape != (N - 1,) or np.any(demand < 0):
        raise ValueError('demand must be 7 non-negative withdrawals, kg/s')
    K_bar2 = resistance(length_m, diameter_m, friction, temperature, rs, z) / BAR ** 2
    s0 = supply_bar ** 2
    # start from a linear resistor analogy, conserving mass at internal nodes
    g = 1.0 / np.maximum(K_bar2, 1e-6)
    Bred = B[1:]
    initial_s = np.linalg.solve((Bred * g) @ Bred.T,
                                -demand - Bred @ (g * B[0] * s0))
    initial_q = -(g * (B.T @ np.r_[s0, initial_s]))

    def residual(x):
        q = x[:M];
        s = np.r_[s0, x[M:]]
        # hydraulic residuals are scaled to units roughly comparable to mass
        return np.r_[Bred @ q + demand,
                     (B.T @ s - K_bar2 * q * np.abs(q)) / 10.0] # scaling factor

    def jac(x):
        q = x[:M]
        J = np.zeros((N - 1 + M, N - 1 + M))
        J[:N - 1, :M] = Bred
        J[N - 1:, :M] = -np.diag(2 * K_bar2 * np.abs(q)) / 10
        J[N - 1:, M:] = Bred.T / 10
        return J

    ans = least_squares(residual, np.r_[initial_q, initial_s], jac=jac,
                        xtol=1e-12, gtol=1e-12, ftol=1e-12, max_nfev=600)
    q = ans.x[:M]
    s = np.r_[s0, ans.x[M:]]
    mass = np.max(np.abs(Bred @ q + demand))
    hydraulic = np.max(np.abs(B.T @ s - K_bar2 * q * np.abs(q)))
    if (not ans.success or np.any(s <= 0) or mass > tol or hydraulic > tol):
        raise RuntimeError(f'Hydraulic solution failed: success={ans.success},'
                           f' mass={mass}, pressure²={hydraulic}, msg={ans.message}')
    return {'pressure_bar': np.sqrt(s), 'mass_flow_kg_s': q, 'K_bar2': K_bar2,
            'source_flow_kg_s': np.sum(demand), 'mass_max_kg_s': mass,
            'hydraulic_max_bar2': hydraulic, 'solver_evaluations': ans.nfev}


if __name__ == '__main__':
    base = steady_solution(DEMAND, SUPPLY_BAR, LENGTH_M, DIAMETER_M, FRICTION, TEMP, RS, Z)
    print('Nodal pressures [bar(a)] :', np.round(base['pressure_bar'], 4))
    print('Oriented pipe flows [kg/s]:', np.round(base['mass_flow_kg_s'], 4))
    print('Total source injection:', base['source_flow_kg_s'], 'kg/s')
    print('Conservation max error:', base['mass_max_kg_s'], 'kg/s')
    print('Hydraulic max error:', base['hydraulic_max_bar2'], 'bar²')
    print('Number of nonlinear function evaluations:', base['solver_evaluations'])