import os
import math
import random
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

# Parameters for signal generation
def get_sig_params():
    return {
        "sample_rate": 100,  # Hz
        "duration": 10       # seconds
    }

class StructuralCausalModels:
    def __init__(self, num_nodes, link_density=0.15, time_steps=2000, random_seed=None):
        self.num_nodes = num_nodes
        self.link_density = link_density
        self.time_steps = time_steps
        self.ts_length = time_steps - 1000
        self.Tao = range(1, 6)
        self.CoeffC = np.arange(0.25, 2.00, 0.25)
        self.CoeffE = np.linspace(0.75, 1.0, num=6)
        self.var = np.arange(0.25, 1.0, 0.25)
        self.lags = np.arange(0, 5)
        self.path = r'../datasets/synthetic_datasets'
        self.lag_list = []
        
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        # Generate adjacency matrix & links
        adj_mat = self.generate_adj_matrix()
        adj_mat_upp = np.triu(adj_mat)
        res = np.where(adj_mat_upp == 1)
        list_links_all = list(zip(res[0], res[1]))
        self.list_links = [(src, tgt) for src, tgt in list_links_all if src != tgt]
        self.num_links = len(self.list_links)
        self.node_labels = [f'Z{l + 1}' for l in range(num_nodes)]

        # Generate time series + causal structure immediately on init
        self.df, self.links = self._generate_df_and_links()

        # Generate binary causal matrix immediately
        self.binary_matrix = self.links_to_binary_matrix(self.links, self.num_nodes)

    def generate_ts(self):
        multivariate_ts = []
        for i in range(self.num_nodes):
            t = np.arange(0, self.time_steps)
            time_series = np.sin(0.75 * t) + np.random.normal(0.75, 0.33 * (i + 1), self.time_steps)
            multivariate_ts.append(time_series)
        return np.array(multivariate_ts)

    def generate_adj_matrix(self):
        nonzero_indices = np.random.choice(
            self.num_nodes ** 2,
            size=int((self.num_nodes ** 2) * self.link_density),
            replace=False
        )
        data = np.ones(len(nonzero_indices), dtype=int)
        row_indices = nonzero_indices // self.num_nodes
        col_indices = nonzero_indices % self.num_nodes
        sparse_binary_matrix = csr_matrix((data, (row_indices, col_indices)),
                                          shape=(self.num_nodes, self.num_nodes))
        return sparse_binary_matrix.toarray()

    def linear(self, cause, effect, lag_cause, lag_effect, coeff_c, coeff_e):
        dynamic_noise = np.random.normal(0, 0.5, 2 * self.time_steps)
        for t in range(max(self.lags), self.time_steps):
            effect[t] = coeff_e * effect[t - lag_effect] + coeff_c * cause[t - lag_cause] + dynamic_noise[t]
        return effect

    def non_linear(self, cause, effect, lag_cause, lag_effect, coeff_c, coeff_e):
        dynamic_noise = np.random.normal(0, 0.1, 2 * self.time_steps)
        for t in range(max(self.lags), self.time_steps):
            effect[t] = coeff_e * effect[t - lag_effect] + coeff_c * np.sin(cause[t - lag_cause]) + dynamic_noise[t]
        return effect

    def generate_ts_DAG(self):
        multivariate_dag_ts = self.generate_ts()
        base_ts = multivariate_dag_ts.copy()
        nonlinear_prob = [1 if random.random() > 0.75 else 0 for _ in range(self.num_links)]

        for idx in range(self.num_links):
            nonlinear_func = nonlinear_prob[idx]
            cnode, enode = self.list_links[idx]
            coeff_c, coeff_e = random.choice(self.CoeffC), random.choice(self.CoeffE)
            lag_cause, lag_effect = random.choice(self.lags), random.choice(self.lags)
            self.lag_list.append(lag_cause)

            if nonlinear_func:
                time_series = self.linear(base_ts[cnode], multivariate_dag_ts[enode],
                                          lag_cause, lag_effect, coeff_c, coeff_e)
            else:
                time_series = self.non_linear(base_ts[cnode], multivariate_dag_ts[enode],
                                              lag_cause, lag_effect, coeff_c, coeff_e)

            multivariate_dag_ts[enode] = time_series

        return multivariate_dag_ts

    def _generate_df_and_links(self):
        timeseries = self.generate_ts_DAG()
        data_dict = {self.node_labels[nodes]: timeseries[nodes][:] for nodes in range(self.num_nodes)}
        df = pd.DataFrame(data=data_dict, columns=self.node_labels)
        links_with_lags = list(zip(self.list_links, self.lag_list))
        return df, links_with_lags

    @staticmethod
    def links_to_binary_matrix(links, num_nodes):
        matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        for (src, tgt), lag in links:
            matrix[src, tgt] = 1
        return matrix

    def plot_ts(self):
        fig = plt.figure(figsize=(10, 2 * self.num_nodes))
        for i in range(len(self.node_labels)):
            ax = fig.add_subplot(self.num_nodes, 1, i + 1)
            ax.plot(self.df[self.df.columns[i]][100:1000].values)
            ax.set_ylabel(f'{self.df.columns[i]}')
        plt.tight_layout()
        plt.show()

    def draw_DAG(self):
        G = nx.DiGraph()
        for n in range(self.num_nodes):
            G.add_node(n + 1, label='Z$_{' + str(n + 1) + '}$')

        for e in range(len(self.list_links)):
            G.add_edge(self.list_links[e][0] + 1, self.list_links[e][1] + 1)

        pos = nx.circular_layout(G)
        labels = nx.get_node_attributes(G, 'label')
        nx.draw(
            G, pos, with_labels=True, labels=labels, node_size=1000,
            node_color='lightblue', font_size=12, font_color='black',
            font_weight='bold', edge_color='gray', width=1.5, arrows=True
        )
        plt.show()

# Example usage:
if __name__ == '__main__':
    nodes = 5
    scms = StructuralCausalModels(num_nodes=nodes, random_seed=42)
    
    # Now df and binary_matrix are directly available:
    print("\nGenerated Time Series Data (head):")
    print(scms.df.head())
    
    print("\nLinks & Lags Used:")
    print(scms.links)
    
    print("\nBinary Adjacency Matrix:")
    print(scms.binary_matrix)
    
    scms.plot_ts()
    scms.draw_DAG()
