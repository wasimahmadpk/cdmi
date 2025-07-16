import numpy as np
import pandas as pd
from functions import *
import matplotlib.pyplot as plt
import networkx as nx
import random

class RandomCausalSimulator:
    def __init__(self, n_nodes=5, edge_prob=0.3, self_dep_prob=0.8, nonlinear_prob=0.5, seed=None):
        self.n = n_nodes
        self.T = 5000
        self.edge_prob = edge_prob
        self.self_dep_prob = self_dep_prob
        self.nonlinear_prob = nonlinear_prob
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.noise_stds = np.random.uniform(0.05, 0.3, size=self.n)
        self.noise_means = np.random.uniform(0.0, 0.5, size=self.n)
        self.adj = None
        self.graph = None
        self.data = None

    def _generate_random_dag(self):
        adj = np.triu((np.random.rand(self.n, self.n) < self.edge_prob).astype(int), 1)
        for i in range(self.n):
            if np.random.rand() < self.self_dep_prob:
                adj[i, i] = 1  # Self-loop (autoregressive)
        return adj

    def _nonlinear(self, x):
        return np.tanh(x) + 0.1 * np.sin(x)

    def simulate(self):
        self.adj = self._generate_random_dag()
        G = nx.DiGraph()
        data = {f'Z{i}': list(np.zeros(10)) for i in range(self.n)}
        nonlinear_mask = (np.random.rand(self.n, self.n) < self.nonlinear_prob) & (self.adj == 1)

        for i in range(self.n):
            G.add_node(f'Z{i}')
        for i in range(self.n):
            for j in range(self.n):
                if self.adj[i, j] == 1 and i != j:
                    G.add_edge(f'Z{i}', f'Z{j}')
                elif i == j and self.adj[i, i] == 1:
                    G.add_edge(f'Z{i}', f'Z{i}')  # Self-loop

        self.graph = G

        lags = np.random.randint(1, 6, size=(self.n, self.n))
        coeffs = np.random.uniform(0.15, 1.5, size=(self.n, self.n))

        # Add seasonality parameters
        periods = np.random.randint(20, 200, size=self.n)
        amplitudes = np.random.uniform(0.2, 2.0, size=self.n)
        phases = np.random.uniform(0, 2 * np.pi, size=self.n)

        for t in range(10, self.T):
            for child in range(self.n):
                val = np.random.normal(self.noise_means[child], self.noise_stds[child])
                for parent in range(self.n):
                    if self.adj[parent, child] == 1:
                        lag = lags[parent, child]
                        coef = coeffs[parent, child]
                        if t - lag < 0:
                            continue
                        parent_val = data[f'Z{parent}'][t - lag]
                        if nonlinear_mask[parent, child]:
                            val += coef * self._nonlinear(parent_val)
                        else:
                            val += coef * parent_val
                #  Add seasonal component
                seasonal = amplitudes[child] * np.sin(2 * np.pi * t / periods[child] + phases[child])
                val += seasonal

                data[f'Z{child}'].append(val)

        self.data = pd.DataFrame({k: v[33:] for k, v in data.items()})
        self.data = self.data.apply(normalize)
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
