import numpy as np
import helper
import mutualinfo
import pandas as pd
from scipy import stats
# import statsmodels.api as sm
# from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import adfuller
plt.rcParams['figure.dpi'] = 200
import seaborn as sns

def get_regimes(data, wsize):
    
    winsize = wsize
    start = 0
    covmat = []
    columns = data.columns
    dim = len(columns) - 1
    cluster_idx = []
 
    while start+winsize < len(data)-1:
        cluster_idx.append(start)
#         print(f"Data shape: {data.shape}")
        data_batch = data[start: start + winsize]
#         print(f"Data batch: {data_batch.shape}")
        ls_data_batch = []
        
        for i in range(len(columns)):
            ls_data_batch.append(data_batch[columns[i]].values.tolist())

        cov = np.cov(np.array(ls_data_batch))
#         print(f"Covariance of {columns[14]} with other variables: {cov[14]}")
#         flat_cov = np.concatenate(cov).ravel().tolist()
        upper = np.triu(cov, k=0)
#         print(f"Length of Cov matrix: {len(upper[upper!=0])}")
        mask = np.triu_indices(dim)
        newupp = list(upper[mask])
        upp = list(upper[upper!=0])
        
#         mean_v = list(np.mean(np.array(ls_data_batch), axis=1))
        
        feat = stats.describe(np.array(ls_data_batch), axis=1)
        mean_val = feat.mean.tolist()
        skewness = feat.skewness.tolist()
        kurtosis = feat.kurtosis.tolist()
        
        plt.plot(helper.normalize(newupp, 'std'))
        plt.show()
        mix_feat = newupp + mean_val
#         print(f"Length of features pool: {len(mix_feat)}")
        covmat.append(mix_feat)
        start = start + winsize
#    
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=1).fit(covmat)
    clusters = list(kmeans.labels_)
#     print(f"Clusters: {list(kmeans.labels_)}")
#     print(f"Clusters indecis: {cluster_idx}")
    
    
    # Plot regimes

    # toplot = [ 'rain','strain_ns_corrected', 'tides_ns', 'temperature_outside', 'pressure_outside', 'gw_west']
    # toplot = ['temperature_outside', 'pressure_outside', 'strain_ew_corrected']
    toplot = ['Z1', 'Z2', 'Z3']
    # toplot = ['Hs', 'P', 'W' ]
    colors = ['r', 'g', 'b']

    t = np.arange(0, cluster_idx[-1]+winsize)
    start = 0

    for c in range(len(clusters)):
    
        if clusters[c] == 0:
                marker = '-'
        elif clusters[c] == 1:
                marker = '-'
        elif clusters[c] == 2:
                marker = '-'
        for i in range(len(toplot)):
        
        
            plt.plot(t[start: start+winsize], data[toplot[i]].values[start: start + winsize], colors[i]+marker)
#           plt.plot(t[start: start + winsize], data[toplot[i+1]].values[start: start + winsize], color)
#           plt.plot(t[start: start + winsize], data[toplot[i+2]].values[start: start + winsize], color)
        
        start = start + winsize
    plt.legend(toplot)
    for c in range(len(cluster_idx)):
        val = cluster_idx[c]
        if clusters[c] == 0:
            for v in range(winsize):
                plt.axvline(val+v, color="red", alpha=0.01)
        if clusters[c] == 1:
            for v in range(winsize):
                plt.axvline(val+v, color="green", alpha=0.01)
        if clusters[c] == 2:
            for v in range(winsize):
                plt.axvline(val+v, color="white", alpha=0.01)
    plt.savefig("regimes.png")
    plt.show()
    
    return clusters, cluster_idx