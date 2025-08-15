import numpy as np
import pandas as pd
from functions import *  # Assumes normalize() is defined here
import matplotlib.pyplot as plt
import networkx as nx
import random

class CausalSimulator:
    def __init__(self, n_nodes=5, edge_prob=0.3, nonlinear_prob=0.5, self_dep_prob=1.0, seed=None):
        self.n = n_nodes
        self.T = 5000
        self.edge_prob = edge_prob
        self.self_dep_prob = self_dep_prob
        self.nonlinear_prob = nonlinear_prob
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Lower noise for clearer signals
        self.noise_stds = np.random.uniform(0.01, 0.05, size=self.n)
        self.noise_means = np.random.uniform(0.0, 0.1, size=self.n)

        self.adj = None
        self.graph = None
        self.data = None

    def _generate_random_dag(self):
        adj = np.triu((np.random.rand(self.n, self.n) < self.edge_prob).astype(int), 1)
        for i in range(self.n):
            if np.random.rand() < self.self_dep_prob:
                adj[i, i] = 1  # self-loop
        # np.fill_diagonal(adj, 1)  # Always add self-loops
        return adj

    def _nonlinear(self, x):
        # More complex nonlinear function with clipping to prevent explosion
        val = 0.2 * np.sin(2 * x) + 0.01 * (x ** 2)
        return np.clip(val, -3, 3)

    def simulate(self):
        self.adj = self._generate_random_dag()
        G = nx.DiGraph()

        # Faster cycles (shorter periodicity)
        freqs = np.linspace(0.1, 0.3, self.n)
        np.random.shuffle(freqs)
        phases = np.random.uniform(0, 2 * np.pi, size=self.n)
        amplitudes = np.random.uniform(0.5, 1.0, size=self.n)

        data = {}
        for i in range(self.n):
            time = np.arange(self.T)
            base_signal = amplitudes[i] * np.sin(freqs[i] * time + phases[i])
            noise = np.random.normal(loc=self.noise_means[i], scale=self.noise_stds[i], size=self.T)
            data[f'Z{i}'] = list(base_signal + noise)

        nonlinear_mask = (np.random.rand(self.n, self.n) < self.nonlinear_prob) & (self.adj == 1)
        # print(f'Nonlinear Mask:\n{nonlinear_mask}')

        for i in range(self.n):
            G.add_node(f'Z{i}')
        for i in range(self.n):
            for j in range(self.n):
                if self.adj[i, j] == 1 and i != j:
                    G.add_edge(f'Z{i}', f'Z{j}')
                elif i == j and self.adj[i, i] == 1:
                    G.add_edge(f'Z{i}', f'Z{i}')

        self.graph = G

        lags = np.random.randint(1, 4, size=(self.n, self.n))  # shorter lags
        coeffs = np.random.uniform(1.0, 5.0, size=(self.n, self.n))  # strong causal effect

        for t in range(self.T):
            for child in range(self.n):
                for parent in range(self.n):
                    if self.adj[parent, child] == 1 and parent != child:
                        lag = lags[parent, child]
                        if t - lag < 0:
                            continue
                        parent_val = data[f'Z{parent}'][t - lag]
                        coef = coeffs[parent, child]
                        if nonlinear_mask[parent, child]:
                            data[f'Z{child}'][t] += coef * self._nonlinear(parent_val)
                        else:
                            data[f'Z{child}'][t] += coef * parent_val
                # Clip after all parent influences to avoid explosion or vanishing
                data[f'Z{child}'][t] = np.clip(data[f'Z{child}'][t], -10, 10)

        self.data = pd.DataFrame(data).apply(normalize)
        return self.data, self.adj

    def draw_dag(self, layout='spring', figsize=(6, 6)):
        if self.graph is None:
            raise ValueError("Call simulate() first to generate the graph.")

        layout_func = {
            'spring': nx.spring_layout,
            'circular': nx.circular_layout,
            'kamada_kawai': nx.kamada_kawai_layout
        }.get(layout, nx.spring_layout)

        pos = layout_func(self.graph)
        plt.figure(figsize=figsize)
        nx.draw(self.graph, pos, with_labels=True,
                node_size=2000, node_color='lightblue',
                font_weight='bold', arrows=True, arrowstyle='-|>')
        plt.title("Generated Causal DAG (with possible self-loops)")
        plt.show()


# Example usage
if __name__ == "__main__":
    sim = RandomCausalSimulator(n_nodes=5, edge_prob=0.4, nonlinear_prob=0.6, seed=42)
    df, adj = sim.simulate()

    print("Adjacency Matrix (adj[i, j] == 1 means row i causes column j):")
    print(adj)
    print("\nSample Data:")
    print(df.head())

    sim.draw_dag()
