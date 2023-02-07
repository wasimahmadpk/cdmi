import mutualinfo
import parameters
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import preprocessing as prep
# from sklearn import metrics
# import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import adfuller
plt.rcParams['figure.dpi'] = 200


# Parameters
pars = parameters.get_syn_params()
win_size = pars.get("win_size")
slidingwin_size = pars.get("slidingwin_size")
plot_path = pars.get("plot_path")

def pyriemann_clusters(data, k=2):
    
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    
    kmeans = Kmeans(k, metric='riemann', tol=1e-3, init='random')
    kmeans.fit(data)
    labels = kmeans.predict(data)
    centroids = kmeans.centroids
    print(labels)
    
#     for k in K:
#         kmeans = KMeans(k, 'riemann', tol=1e-3, init='random')
#         kmeans.fit(data)
#         labels = kmeans.predict(data)
#         centroids = kmeans.centroids
#         print(labels)
        
#         distortions.append(sum(np.min(cdist(data, kmeans.centroids, 'euclidean'), axis=1)) / np.array(data).shape[0])
#         inertias.append(kmeans.inertia_)
#         mapping1[k] = sum(np.min(cdist(data, kmeans.centroids, 'euclidean'), axis=1)) / np.array(data).shape[0]
#         mapping2[k] = kmeans.inertia_
        
#     #   The elbow method for optimal number of clusters
#     plt.plot(K, inertias, 'bx-')
#     plt.xlabel('Values of K')
#     plt.ylabel('Distortion')
#     plt.title('The Elbow Method using Distortion')
#     plt.show()
    
    return labels

def get_regimes(data, wsize):
    
    winsize = wsize
    start = 0
    covmat = []
    columns = data.columns
    dim = len(columns)
    cluster_idx = []
 
    while start+winsize < len(data):
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
        
        # plt.plot(prep.normalize(newupp, 'std'))
        # plt.show()
        mix_feat = newupp + mean_val
#         print(f"Length of features pool: {len(mix_feat)}")
        covmat.append(mix_feat)
        start = start + winsize
#    
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=1).fit(covmat)
    clusters = list(kmeans.labels_)
#     print(f"Clusters: {list(kmeans.labels_)}")
#     print(f"Clusters indecis: {cluster_idx}")
    
    clusters2 = pyriemann_clusters(cov, k=2)

    clusters_extended = []
    for i in range(len(clusters)):

        val = clusters[i]
        for j in range(slidingwin_size):
            clusters_extended.append(val)
    
    newdf = data.iloc[:len(clusters_extended), :].copy()
    newdf['Clusters'] = clusters_extended

    dfs = []
    for c in range(len(list(set(clusters)))):
        dfs.append(newdf.loc[newdf['Clusters'] == list(set(clusters))[c]])

    return dfs, clusters, cluster_idx, newdf


def get_reduced_set(df):
    
    corr = data.corr()
    cls = corr.iloc[0][:].values.tolist()
    selected_idx = np.where(cls>0.50)[0].tolist()

    reduced_df = df.iloc[:, selected_idx].copy()
    return reduced_df


def plot_regimes(data, clusters, cluster_idx, winsize, toplot):

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