# Imports
import numpy as np
import pandas as pd
import matplotlib
from riverdata import RiverData
from matplotlib import pyplot as plt
# %matplotlib inline
## use `%matplotlib notebook` for interactive figures
# plt.style.use('ggplot')
import sklearn

import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
from tigramite.models import LinearMediation, Prediction

from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr
import tigramite.data_processing as pp
np.random.seed(7)
win_size = 1


def normalize(var):
    nvar = (np.array(var) - np.mean(var)) / np.std(var)
    return nvar


def deseasonalize(var, interval):

    deseasonalize_data = []
    for i in range(interval, len(var)):
        value = var[i] - var[i - interval]
        deseasonalize_data.append(value)
    return deseasonalize_data


def down_sample(data, win_size, partition=None):
    agg_data = []
    daily_data = []
    for i in range(len(data)):
        daily_data.append(data[i])

        if (i % win_size) == 0:

            if partition == None:
                agg_data.append(sum(daily_data) / win_size)
                daily_data = []
            elif partition == 'gpp':
                agg_data.append(sum(daily_data[24: 30]) / 6)
                daily_data = []
            elif partition == 'reco':
                agg_data.append(sum(daily_data[40: 48]) / 8)
                daily_data = []
    return agg_data

# Example process to play around with
# Each key refers to a variable and the incoming links are supplied
# as a list of format [((driver, -lag), coeff), ...]

links_coeffs = {0: [((0, -1), 0.8)],
                1: [((1, -2), 0.8), ((0, -1), 0.75)],
                2: [((2, -1), 0.8), ((0, -1), 0.5)],
                }
# #
# var_names = [r"$Xts$", r"$Yts$", r"$Zts$", r"$Rts$"]
# col_names = ['Xts', 'Yts', 'Zts', 'Rts']

var_names = [r"$Z1$", r"$Z2$", r"$Z3$", r"$Z4$", r"$Z5$", r"$Z6$", r"$Z7$", r"$Z8$", r"$Z9$", r"$Z10$"]
col_names = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9', 'Z10']
# # #
# var_names = [r"$Rg$", r"$T$", r"$GPP$", r"$Reco$"]
# col_names = ['Rg', 'T', 'GPP' 'Reco']

# var_names = [r"$Kts$", r"$Dts$", r"$Lts$"]
# col_names = ['Kts', 'Dts', 'Lts']

# data, true_parents = pp.var_process(links_coeffs, T=1000)
# # Data must be array of shape (time, variables)
# print(data.shape)
# print(data)
# dataframe = pp.DataFrame(data)

# # *********************************************************
# # "Load fluxnet 2015 data for grassland IT-Mbo site-Hourly data"
# fluxnet = pd.read_csv("/home/ahmad/PycharmProjects/deepCause/datasets/fluxnet2015/FLX_IT-MBo_FLUXNET2015_SUBSET_2003-2013_1-4/FLX_IT-MBo_FLUXNET2015_SUBSET_HH_2003-2013_1-4.csv")
# print(fluxnet.shape)
# rg = normalize(fluxnet['SW_IN_F'])
# temp = normalize(fluxnet['TA_F'])
# vpd = normalize(fluxnet['VPD_F'])
# swc = normalize(fluxnet['SWC_F_MDS_2'])
# # heat = normalize(fluxnet['H_CORR_75'])
# # oppt = normalize(fluxnet['P_F'])
# nee = normalize(fluxnet['NEE_VUT_25'])
# gpp = normalize(fluxnet['GPP_NT_VUT_25'])
# reco = normalize(fluxnet['RECO_NT_VUT_25'])


# # # External dataframe
syndata = pd.read_csv("/home/ahmad/PycharmProjects/deepCause/datasets/ncdata/synthetic_data.csv", sep=',')

x1 = normalize(down_sample(np.array(syndata['Z1']), win_size))
x2 = normalize(down_sample(np.array(syndata['Z2']), win_size))
x3 = normalize(down_sample(np.array(syndata['Z3']), win_size))
x4 = normalize(down_sample(np.array(syndata['Z4']), win_size))
x5 = normalize(down_sample(np.array(syndata['Z5']), win_size))
x6 = normalize(down_sample(np.array(syndata['Z6']), win_size))
x7 = normalize(down_sample(np.array(syndata['Z7']), win_size))
x8 = normalize(down_sample(np.array(syndata['Z8']), win_size))
x9 = normalize(down_sample(np.array(syndata['Z9']), win_size))
x10 = normalize(down_sample(np.array(syndata['Z10']), win_size))

# # River Discharge Data
# dataobj = RiverData()
# data = dataobj.get_data()
# kts = data['Kempten']
# dts = data['Dillingen']
# lts = data['Lenggries']

# "Load semi-synthetic data"
# syndata = pd.read_csv("/home/ahmad/PycharmProjects/deepCause/datasets/ncdata/artificial_data.csv", sep=',')
# rg = normalize(down_sample(np.array(syndata['Rg']), win_size)[:1000])
# temp = normalize(down_sample(np.array(syndata['T']), win_size)[:1000])
# gpp = normalize(down_sample(np.array(syndata['GPP']), win_size)[:1000])
# reco = normalize(down_sample(np.array(syndata['Reco']), win_size)[:1000])

# interval = 100
# # LOad synthetic data *************************
# syndata = pd.read_csv("/home/ahmad/PycharmProjects/deepCause/datasets/ncdata/artificial_data.csv")
# rg = normalize(down_sample(np.array(syndata['Rg']), win_size))
# temp = normalize(down_sample(np.array(syndata['T']), win_size))
# gpp = normalize(down_sample(np.array(syndata['GPP']), win_size))
# reco = normalize(down_sample(np.array(syndata['Reco']), win_size))

data = np.array([x1[:150], x2[:150], x3[:150], x4[:150], x5[:150], x6[:150], x7[:150], x8[:150], x9[:150], x10[:150]])
# # data = np.array([kts, dts, lts])
# data = np.array([rg[7000:7500], temp[7000:7500], gpp[7000:7500], reco[7000:7500]])
print(data)
data = data.transpose()


# Initialize dataframe object, specify time axis and variable names
dataframe = pp.DataFrame(data,
                         var_names=var_names)
# df = pd.DataFrame(data,
#                   columns=col_names)
# df.to_csv(r'/home/ahmad/PycharmProjects/deepCause/datasets/ncdata/synthetic_data.csv', index_label=False, header=True)

# med = LinearMediation(dataframe=dataframe)
# med.fit_model(all_parents=true_parents, tau_max=4)
#
# print("Link coefficient (0, -2) --> 2: ", med.get_coeff(i=0, tau=-2, j=2))
# print("Causal effect (0, -2) --> 2: ", med.get_ce(i=0, tau=-2, j=2))
# print("Mediated Causal effect (0, -2) --> 2 through 1: ", med.get_mce(i=0, tau=-2, j=2, k=1))
#
# i=0; tau=4; j=2
# graph_data = med.get_mediation_graph_data(i=i, tau=tau, j=j)
# tp.plot_mediation_time_series_graph(
#     var_names=var_names,
#     path_node_array=graph_data['path_node_array'],
#     tsg_path_val_matrix=graph_data['tsg_path_val_matrix']
#     )
# tp.plot_mediation_graph(
#                     var_names=var_names,
#                     path_val_matrix=graph_data['path_val_matrix'],
#                     path_node_array=graph_data['path_node_array'],
#                     )

cond_ind_test = GPDC()             # ParCorr()  # #CMIknn()
pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=1)
results = pcmci.run_pcmciplus(tau_min=1, tau_max=5, pc_alpha=.05)
pcmci.print_significant_links(p_matrix=results['p_matrix'],
                                     val_matrix=results['val_matrix'],
                                     alpha_level=0.05)
## Significant parents at alpha = 0.05: