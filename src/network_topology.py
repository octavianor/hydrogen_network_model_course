import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


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

# color
NAVY = '#183047'
BLUE = '#187f97'
ORANGE = '#eb9b55'
CYAN = '#68b2bd'
GREY = '#637587'
GREEN = '#338769'
PURPLE = '#9074b7'

class GasNetwork:

    def plot_network(self):
        g = nx.Graph()
        g.add_nodes_from(range(N))
        g.add_edges_from(EDGES)
        fig, ax = plt.subplots(figsize=(10, 5.0))
        nx.draw_networkx_edges(g, POSITIONS, ax=ax, width=3.0, edge_color='#aac8d0')
        nx.draw_networkx_nodes(g, POSITIONS, nodelist=[0], node_color=ORANGE, node_size=930, ax=ax,
                               edgecolors='white', linewidths=2)
        nx.draw_networkx_nodes(g, POSITIONS, nodelist=list(range(1, N)), node_color=BLUE, node_size=750,
                               ax=ax, edgecolors='white', linewidths=2)
        nx.draw_networkx_labels(g, POSITIONS, ax=ax, font_color='white', font_weight='bold',
                                font_size=12)
        labels = {(u, v): f'e{i + 1}' for i, (u, v) in enumerate(EDGES)}
        nx.draw_networkx_edge_labels(g, POSITIONS, labels, ax=ax, font_size=9, rotate=False,
                                     bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                                               edgecolor='none', alpha=.92))
        ax.set_xlim(-.35, 4.05)
        ax.set_ylim(-.65, 2.6)
        ax.axis('off')
        ax.text(.05, .04, 'Orange = prescribed source pressure  |  Blue = demand junction',
                transform=ax.transAxes, fontsize=11, color=GREY)
        plt.show()


if __name__ == '__main__':
    GN = GasNetwork()
    GN.plot_network()
