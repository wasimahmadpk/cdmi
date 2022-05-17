import lingam
import graphviz
import numpy as np
import pandas as pd
import statsmodels.api as sm
from riverdata import RiverData
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.base.datetools import dates_from_str
from lingam.utils import print_causal_directions, print_dagc, make_dot


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


win_size = 1

# # # "Load fluxnet 2015 data for grassland IT-Mbo site"
# fluxnet = pd.read_csv("/home/ahmad/PycharmProjects/deepCause/datasets/fluxnet2015/FLX_IT-MBo_FLUXNET2015_SUBSET_2003-2013_1-4/FLX_IT-MBo_FLUXNET2015_SUBSET_HH_2003-2013_1-4.csv")
# org = fluxnet['SW_IN_F']
# otemp = fluxnet['TA_F']
# ovpd = fluxnet['VPD_F']
# # oppt = fluxnet['P_F']
# nee = fluxnet['NEE_VUT_50']
# ogpp = fluxnet['GPP_NT_VUT_50']
# oreco = fluxnet['RECO_NT_VUT_50']
#
# rg = normalize(down_sample(org, win_size))
# temp = normalize(down_sample(otemp, win_size))
# gpp = normalize(down_sample(nee, win_size, partition='gpp'))
# reco = normalize(down_sample(nee, win_size, partition='reco'))
# # ppt = normalize(down_sample(oppt, win_size))
# vpd = normalize(down_sample(ovpd, win_size))
# print("Length:", len(rg))

# "Load synthetic data"
syndata = pd.read_csv("/home/ahmad/PycharmProjects/deepCausality/datasets/ncdata/synthetic_data.csv", sep=',')

x1 = normalize(down_sample(np.array(syndata['Z1']), win_size))
x2 = normalize(down_sample(np.array(syndata['Z2']), win_size))
x3 = normalize(down_sample(np.array(syndata['Z3']), win_size))
x4 = normalize(down_sample(np.array(syndata['Z4']), win_size))
x5 = normalize(down_sample(np.array(syndata['Z5']), win_size))


# # "Load semi-synthetic data"
# syndata = pd.read_csv("/home/ahmad/PycharmProjects/deepCause/datasets/ncdata/artificial_data.csv", sep=',')
# rg = normalize(down_sample(np.array(syndata['Rg']), win_size))
# temp = normalize(down_sample(np.array(syndata['T']), win_size))
# gpp = normalize(down_sample(np.array(syndata['GPP']), win_size))
# reco = normalize(down_sample(np.array(syndata['Reco']), win_size))
#
#
col_list = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5']

# # col_list = ['Kts', 'Dts', 'Lts']
# col_list = ['Rg', 'T', 'GPP', 'Reco']
dt = {'Z1': x1, 'Z2': x2, 'Z3': x3, 'Z4': x4, 'Z5': x5}

data = pd.DataFrame(dt, columns=col_list)
# # data = pd.DataFrame({'Kts': kts, 'Dts': dts, 'Lts': lts}, columns=col_list)
# data = pd.DataFrame({'Rg': rg[0:1000], 'T': temp[0:1000], 'GPP': gpp[0: 1000], 'Reco': reco[0: 1000]}, columns=col_list)
# print(data.head(100))

# # *********************************************************
# # "Load fluxnet 2015 data for grassland IT-Mbo site-Hourly data"
# fluxnet = pd.read_csv("/home/ahmad/PycharmProjects/deepCause/datasets/fluxnet2015/FLX_IT-MBo_FLUXNET2015_SUBSET_2003-2013_1-4/FLX_IT-MBo_FLUXNET2015_SUBSET_HH_2003-2013_1-4.csv")
# print(fluxnet.columns)
# rg = normalize(fluxnet['SW_IN_F'])
# temp = normalize(fluxnet['TA_F'])
# vpd = normalize(fluxnet['VPD_F'])
# swc = normalize(fluxnet['SWC_F_MDS_2'])
# # heat = normalize(fluxnet['H_CORR_75'])
# # oppt = normalize(fluxnet['P_F'])
# nee = normalize(fluxnet['NEE_VUT_25'])
# gpp = normalize(fluxnet['GPP_NT_VUT_25'])
# reco = normalize(fluxnet['RECO_NT_VUT_25'])


# col_list = ['Rg', 'T', 'GPP', 'Reco']
# dt = {'Rg': rg[7000:7700], 'T': temp[7000:7700], 'GPP': gpp[7000:7700], 'Reco': reco[7000:7700]}
# data = pd.DataFrame(dt, columns=col_list)


# x3 = np.random.uniform(size=1000)
# x0 = 3.0*x3 + np.random.uniform(size=1000)
# x2 = 6.0*x3 + np.random.uniform(size=1000)
# x1 = 3.0*x0 + 2.0*x2 + np.random.uniform(size=1000)
# x5 = 4.0*x0 + np.random.uniform(size=1000)
# x4 = 8.0*x0 - 1.0*x2 + np.random.uniform(size=1000)
# data = pd.DataFrame(np.array([x0, x1, x2, x3, x4, x5]).T ,columns=['x0', 'x1', 'x2', 'x3', 'x4', 'x5'])
# print(data.head())


# To run causal discovery, we create a DirectLiNGAM object and call the fit method.
model = lingam.DirectLiNGAM()
# model.fit(data)
result = model.bootstrap(data, n_sampling=5)
cdc = result.get_causal_direction_counts(n_directions=10, min_causal_effect=0.01, split_by_causal_effect_sign=True)
print_causal_directions(cdc, 5)

# # Using the causal_order_ properties,
# # we can see the causal ordering as a result of the causal discovery.
# print(model.causal_order_)
#
# # Also, using the adjacency_matrix_ properties,
# # we can see the adjacency matrix as a result of the causal discovery.
# print(model.adjacency_matrix_)
# dot = make_dot(model.adjacency_matrix_)
# print(dot)
# # p_values = model.get_error_independence_p_values(data)
# # print(p_values)

