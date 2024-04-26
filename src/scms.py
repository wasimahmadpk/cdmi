import os
import math
import random
import parameters
import numpy as np
import pandas as pd
import networkx as nx
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
# np.random.seed(1)

class SCMS:

    def __init__(self, num_nodes, link_density=0.15, time_steps=2000):

        self.num_nodes = num_nodes
        self.link_density = link_density
        self.time_steps = time_steps
        self.ts_length = time_steps - 1000
        self.Tao = range(1, 6)
        self.CoeffC, self.CoeffE = np.arange(0.25, 2.00, 0.25), np.arange(0.75, 1.0, 0.05)
        self.var = np.arange(0.25, 1.0, 0.25)
        self.lags = np.arange(0, 5)
        self.path = r'../datasets/synthetic_datasets'
        self.lag_list = []
        
        adj_mat = self.generate_adj_matrix()
        # print('Matrix:\n', adj_mat)
        adj_mat_upp = np.triu(adj_mat)
        res = np.where(adj_mat_upp==1)
        
        list_links_all = list(zip(res[0], res[1]))
        self.list_links = []

        for links in list_links_all:
            if links[0]!=links[1]:
                self.list_links.append(links)
        self.num_links = len(self.list_links)
                
        self.node_labels = [f'Z{l+1}'for l in range(num_nodes)]
        self.generate_ts_DAG()


    def generate_sine_ts(self, freq, sample_rate, duration):
        t = np.linspace(0, duration, sample_rate * duration, endpoint=False)
        frequencies = t*freq
        # 2pi because np.sin takes radians
        y = np.sin((1*np.pi)*frequencies)
        return t, y

    # Time series base
    def generate_ts(self):
         
        # Generate sine wave and the gaussian noise 
        pars = parameters.get_sig_params()
        sample_rate = pars.get("sample_rate")  # Hertz
        duration = pars.get("duration")  # Seconds

        multivariate_ts = []
        
        for i in range(self.num_nodes):
            # _, sin_ts = self.generate_sine_ts(6000, sample_rate, duration)
            # timeseries = np.random.normal(0, random.choice(self.var), self.time_steps)
            t = np.arange(0, self.time_steps)
            time_series = np.sin(0.75 * t) + np.random.normal(0.75, 0.33*(i+1), self.time_steps)
            multivariate_ts.append(time_series)
        return np.array(multivariate_ts)
  

    def generate_adj_matrix(self):

            # Generate random indices for non-zero elements
            nonzero_indices = np.random.choice(self.num_nodes**2, size=int((self.num_nodes**2) * self.link_density), replace=False)

            # Create a binary matrix with ones at the specified indices
            data = np.ones(len(nonzero_indices), dtype=int)
            row_indices = nonzero_indices // self.num_nodes
            col_indices = nonzero_indices % self.num_nodes

            # Create a sparse binary matrix using CSR format
            sparse_binary_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(self.num_nodes, self.num_nodes))

            # print(sparse_binary_matrix.toarray())
            adj_matrix = sparse_binary_matrix.toarray()
            return adj_matrix

    # Linear cause-effect relation
    def linear(self, cause, effect, lag_cause, lag_effect, coeff_c, coeff_e):
        
        dynamic_noise = np.random.normal(0, 0.5, 2*self.time_steps)
        for t in range(max(self.lags), self.time_steps):
            effect[t] = coeff_e*effect[t-lag_effect] + coeff_c*cause[t-lag_cause] + dynamic_noise[t]
        return effect, len(effect)
    
    # NOn-linear dependency
    def non_linear(self, cause, effect, lag_cause, lag_effect, coeff_c, coeff_e):
        
        dynamic_noise = np.random.normal(0, 0.10, 2*self.time_steps)
        for t in range(max(self.lags), self.time_steps):
            effect[t] = coeff_e*effect[t-lag_effect] + coeff_c*np.sin(cause[t-lag_cause]) + dynamic_noise[t]
        return effect, len(effect)
    
    def generate_ts_DAG(self):

        multivariate_dag_ts = self.generate_ts()
        nonlinear_prob = [1 if random.random() > 0.75 else 0 for _ in range(self.num_links)]
        for links in range(self.num_links):
            nonlinear_func = random.choice(nonlinear_prob)
            cnode, enode = self.list_links[links][0], self.list_links[links][1]

            coeff_c, coeff_e = random.choice(self.CoeffC), random.choice(self.CoeffE) 
            lag_cause, lag_effect = random.choice(self.lags), random.choice(self.lags)
            self.lag_list.append(lag_cause)
            
            if nonlinear_func:
                time_series , len = self.linear(multivariate_dag_ts[cnode], multivariate_dag_ts[enode], lag_cause, lag_effect, coeff_c, coeff_e)
            else:
                time_series , len = self.non_linear(multivariate_dag_ts[cnode], multivariate_dag_ts[enode], lag_cause, lag_effect, coeff_c, coeff_e)
            multivariate_dag_ts[enode] = time_series  

        return multivariate_dag_ts
    
    def df_timeseries(self):
        
        data_dict = {}
        timeseries = self.generate_ts_DAG()
        
        for nodes in range(self.num_nodes):
            data_dict[self.node_labels[nodes]] = timeseries[nodes][:]
        
        df = pd.DataFrame(data=data_dict, columns=self.node_labels)
        filename = 'synthetic_dataset.csv'
        df.to_csv(os.path.join(self.path, filename), index_label=False, header=True)
        return df, list(zip(self.list_links, self.lag_list))
    
    def plot_ts(self):

        fig = plt.figure()
        df, links = self.df_timeseries()
        for i in range(len(self.node_labels)):
            ax = fig.add_subplot(int(f'{len(self.node_labels)}1{i+1}'))
            ax.plot(df[df.columns[i]][100:1000].values)
            ax.set_ylabel(f'{df.columns[i]}')
        plt.show()

    def draw_DAG(self):
          
        # Create an empty graph
        G = nx.DiGraph()

        # Add nodes
        for n in range(self.num_nodes):
            G.add_node(n+1, label=f'Z$_{n+1}$')

        for e in range(len(self.list_links)):
            G.add_edge(self.list_links[e][0]+1, self.list_links[e][1]+1)

       # Draw the directed graph with labels
        pos = nx.circular_layout(G)
        labels = nx.get_node_attributes(G, 'label')  # Get labels from node attributes
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=1000, node_color='lightblue', font_size=12, font_color='black', font_weight='bold', edge_color='gray', width=2.0, arrows=True)

        # Display the directed graph with labels
        plt.show()



if __name__ == '__main__':
    
    nodes = 5
    scms = SCMS(nodes)
    df = scms.df_timeseries()
    scms.plot_ts()
