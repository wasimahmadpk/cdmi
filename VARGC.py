import numpy as np
import pandas as pd
from riverdata import RiverData
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.base.datetools import dates_from_str


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

# mdata = sm.datasets.macrodata.load_pandas().data
# # prepare the dates index
# dates = mdata[['year', 'quarter']].astype(int).astype(str)
# quarterly = dates["year"] + "Q" + dates["quarter"]
# quarterly = dates_from_str(quarterly)
# mdata = mdata[['realgdp','realcons','realinv']]
# mdata.index = pandas.DatetimeIndex(quarterly)
# data = np.log(mdata).diff().dropna()
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

# dataobj = RiverData()
# data = dataobj.get_data()
# kts = data['Kempten']
# dts = data['Dillingen']
# lts = data['Lenggries']

# # "Load semi-synthetic data"
# syndata = pd.read_csv("/home/ahmad/PycharmProjects/deepCause/datasets/ncdata/artificial_data.csv", sep=',')
# rg = normalize(down_sample(np.array(syndata['Rg']), win_size))
# temp = normalize(down_sample(np.array(syndata['T']), win_size))
# gpp = normalize(down_sample(np.array(syndata['GPP']), win_size))
# reco = normalize(down_sample(np.array(syndata['Reco']), win_size))
#
#
col_list = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9', 'Z10']
# # col_list = ['Kts', 'Dts', 'Lts']
# col_list = ['Rg', 'T', 'GPP', 'Reco']
dt = {'Z1': x1, 'Z2': x2, 'Z3': x3, 'Z4': x4, 'Z5': x5, 'Z6': x6, 'Z7': x7, 'Z8': x8, 'Z9': x9, 'Z10': x10}
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

# make a VAR model
model = VAR(data)
results = model.fit(7)
# print(results.summary())
for i in range(len(col_list)):
    for j in range(len(col_list)):
        print(results.test_causality(col_list[j], col_list[i], kind='f', signif=0.05))