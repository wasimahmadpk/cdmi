import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import random

def safe_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    if max_val - min_val < 1e-8:
        # inject tiny noise instead of flat zero series
        return np.random.normal(0, 1e-2, size=len(x))
    return (x - min_val) / (max_val - min_val)

class CausalSimulator:
    def __init__(self, n_nodes=5, edge_prob=0.3, nonlinear_prob=0.0,
                 self_dep_prob=1.0, timesteps=500, noise_scale=0, seed=None):
        self.n = n_nodes
        self.T = timesteps
        self.edge_prob = edge_prob
        self.self_dep_prob = self_dep_prob
        self.nonlinear_prob = nonlinear_prob
        self.seed = seed
        self.noise_scale = noise_scale

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.noise_stds = np.random.uniform(0.1, 0.50, size=self.n)
        self.noise_means = np.zeros(self.n)
        self.adj = None
        self.graph = None
        self.data = None

    def _generate_random_dag(self):
        adj = np.triu((np.random.rand(self.n, self.n) < self.edge_prob).astype(int), 1)
        for i in range(self.n):
            if np.random.rand() < self.self_dep_prob:
                adj[i, i] = 1
        return adj

    def _nonlinear(self, x, nonlin_prob):
        """
        Progressive nonlinear function that adds more nonlinearities
        cumulatively as nonlin_prob increases.
        Ensures stability by bounding outputs.
        """
        x = np.clip(x, -5, 5)  # keep input in safe range
        y = 0.5 * np.sin(x)

        nonlinear_terms = [
            lambda x: 0.2 * np.cos(x),
            lambda x: 0.2 * np.cos(2 * x) + 0.2 * np.sin(x),
            lambda x: 0.3 * (2 * x) / np.exp(-0.1 * x**2 + 1e-3),  # avoid div/0
            lambda x: 0.4 * x / (1 + 0.5 * x**2),  # bounded rational function
            lambda x: 0.5 * np.tanh(2 * x)        # tanh instead of raw tan
        ]

        thresholds = np.linspace(0.1, 1.0, len(nonlinear_terms))

        for t, func in zip(thresholds, nonlinear_terms):
            if nonlin_prob >= t:
                term = func(x)
                term = np.clip(term, -5, 5)  # clip each term
                y += term

        return 5 * np.tanh(y / 5)  # squash to [-5,5]

    def simulate(self):
        self.adj = self._generate_random_dag()
        G = nx.DiGraph()
        data = {f'Z{i}': np.random.normal(loc=self.noise_means[i],
                                        scale=self.noise_stds[i],
                                        size=self.T)
                for i in range(self.n)}

        for i in range(self.n):
            G.add_node(f'Z{i}')
        for i in range(self.n):
            for j in range(self.n):
                if self.adj[i, j] == 1:
                    G.add_edge(f'Z{i}', f'Z{j}')
        self.graph = G

        lags = np.random.randint(1, 5, size=(self.n, self.n))
        coeffs = np.random.uniform(1, 5, size=(self.n, self.n))  # mild linear coefficients

        # Simulation loop
        for t in range(self.T):
            for child in range(self.n):
                for parent in range(self.n):
                    if self.adj[parent, child] == 1 and t - lags[parent, child] >= 0:
                        lag = lags[parent, child]
                        parent_val = data[f'Z{parent}'][t - lag]
                        coef = coeffs[parent, child]

                        mixed_effect = self._nonlinear(parent_val, self.nonlinear_prob)

                        adaptive_noise = np.random.normal(self.noise_scale, 0.50 + self.noise_scale) # replace self.noise_scale with 0 mean
                        data[f'Z{child}'][t] += coef * mixed_effect + adaptive_noise

                data[f'Z{child}'][t] = np.clip(data[f'Z{child}'][t], -10, 10)

        # Normalize, with guaranteed non-constant output
        self.data = pd.DataFrame({col: safe_normalize(vals) for col, vals in data.items()})
        return self.data, self.adj

    def draw_dag(self, layout='spring', figsize=(6, 6)):
        if self.graph is None:
            raise ValueError("Call simulate() first.")
        layout_func = {'spring': nx.spring_layout,
                       'circular': nx.circular_layout,
                       'kamada_kawai': nx.kamada_kawai_layout}.get(layout, nx.spring_layout)
        pos = layout_func(self.graph)
        plt.figure(figsize=figsize)
        nx.draw(self.graph, pos, with_labels=True,
                node_size=2000, node_color='lightblue',
                font_weight='bold', arrows=True, arrowstyle='-|>')
        plt.title("Generated Causal DAG")
        plt.show()


if __name__ == "__main__":
    sim = CausalSimulator(n_nodes=5, edge_prob=0.4, nonlinear_prob=0.7, seed=42)
    df, adj = sim.simulate()
    print("Adjacency Matrix:")
    print(adj)
    print("\nSample Data:")
    print(df.head())
    sim.draw_dag()
