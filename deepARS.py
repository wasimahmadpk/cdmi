import math
# import netCDF
import pickle
import random
import pathlib
import numpy as np
import mxnet as mx
import pandas as pd
from os import path
from math import sqrt
from netCDF4 import Dataset
from itertools import islice
from datetime import datetime
from deepcauses import deepCause
import matplotlib.pyplot as plt
from knockoffs import Knockoffs
from scipy.special import stdtr
from model_test import modelTest
from gluonts.trainer import Trainer
from gluonts.evaluation import Evaluator
from sklearn.metrics import mean_squared_error
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.deepar._network import DeepARTrainingNetwork
from gluonts.evaluation.backtest import make_evaluation_predictions
from scipy.stats import ttest_ind, ttest_ind_from_stats, ttest_1samp
from gluonts.distribution.multivariate_gaussian import MultivariateGaussianOutput

np.random.seed(1)
mx.random.seed(2)


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


def SNR(s, n):
    Ps = np.sqrt(np.mean(np.array(s) ** 2))
    Pn = np.sqrt(np.mean(np.array(n) ** 2))
    SNR = Ps / Pn
    return 10 * math.log(SNR, 10)


def mean_absolute_percentage_error(y_true, y_pred):

    return np.mean(np.abs((y_true - y_pred) / y_true))


def running_avg_effect(y, yint):

    rae = 0
    for i in range(len(y)):
        ace = 1/((params.get("train_len") + 1 + i) - params.get("train_len")) * (rae + (y[i] - yint[i]))
    return rae


# # Parameters for fluxnet2006
# freq = 'D'
# epochs = 100
#
# training_length = 750  # data for 15 days
# prediction_length = 15  # data for 1 days
#
# start = 0
# train_stop = start + training_length
# test_stop = train_stop + prediction_length
# win_size = 48

# # Parameters for River discharge data
# freq = 'D'
# dim = 3
# epochs = 150
# win_size = 1

# # Parameters for ecosystem data
# freq = 'D'
# dim = 5
# epochs = 150
# win_size = 48
#
# prediction_length = 14
# num_samples = 50

# Synthetic data
freq = '30min'
epochs = 100
win_size = 1

training_length = 666
prediction_length = 24
num_samples = 10
# *********************************************************

# "Load fluxnet-2006 data"
# nc_f = '/home/ahmad/PycharmProjects/deepCause/datasets/ncdata/DE-Hai.2000.2006.hourly.nc'  # Your filename
# nc_fid = Dataset(nc_f, 'r')   # Dataset is the class behavior to open the file                         # and create an instance of the ncCDF4 class
# nc_attrs, nc_dims, nc_vars = netCDF.ncdump(nc_fid);
#
# # Extract data from NetCDF file
# vpd = normalize(nc_fid.variables['VPD_f'][:].ravel().data)  # extract/copy the data
# temp = normalize(nc_fid.variables['Tair_f'][:].ravel().data)
# rg = normalize(nc_fid.variables['Rg_f'][:].ravel().data)
# swc1 = normalize(nc_fid.variables['SWC1_f'][:].ravel().data)
# nee = normalize(nc_fid.variables['NEE_f'][:].ravel().data)
# gpp = normalize(nc_fid.variables['GPP_f'][:].ravel().data)
# reco = normalize(nc_fid.variables['Reco'][:].ravel().data)
# le = normalize(nc_fid.variables['LE_f'][:].ravel().data)
# h = normalize(nc_fid.variables['H_f'][:].ravel().data)
# time = nc_fid.variables['time'][:].ravel().data

# #========== Load fluxnet data (Micro's nature paper)============
# fluxnet = pd.read_csv("/home/ahmad/PycharmProjects/deepCause/datasets/fluxnet2015/fluxnetmirco.csv", sep=',', delimiter=None)
# print(fluxnet.columns)
#
# temp = fluxnet['Tair']         # T_air
# vpd = fluxnet['VPD']           # VPD
# csw = fluxnet['CSWI']          # CSW
# ppt = fluxnet['P']             # P
# rg = fluxnet['SWin']           # SW_IN
#
# lai = fluxnet['LAImax']        #LAI_MAX
# agb = fluxnet['AGB']           #AGB
# hc = fluxnet['Hc']             #H_C
# ncon = fluxnet['Nmass']        #N%
#
# gpp = fluxnet['GPPsat']       # GPP_SAT
# nep = fluxnet['NEPmax']       # NEP_MAX
# et = fluxnet['ETmax']         # ET_MAX
#
# wue = fluxnet['uWUE']         # uWUE
#
# cue = fluxnet['aCUE']         # aCUE
# rb = fluxnet['Rb']            # R_B
#
# data = {'Tair': temp, 'VPD': vpd, 'CSW': csw, 'PPT': ppt, 'Rg': rg, 'LAI': lai, 'AGB': agb, 'Hc': hc, 'Ncon': ncon,
#           'GPP': gpp, 'NEP': nep, 'ET': et, 'WUE': wue,
#            'CUE': cue, 'Rb': rb }
# df = pd.DataFrame(data, columns=['Tair', 'VPD', 'CSW', 'PPT', 'Rg', 'LAI', 'AGB', 'Hc', 'Ncon',
#          'GPP', 'NEP', 'ET', 'WUE', 'CUE', 'Rb'])
#
# print(df.columns)

# for n in range(len(df.columns)):
# 
#     plt.plot(df[df.columns[n]])
#     plt.ylabel(df.columns[n])
#     plt.show()
# # *********************************************************
# # "Load fluxnet 2015 data for grassland IT-Mbo site-Half hourly data"
# fluxnet = pd.read_csv("/home/ahmad/PycharmProjects/deepCause/datasets/fluxnet2015/FLX_IT-MBo_FLUXNET2015_SUBSET_2003-2013_1-4/FLX_IT-MBo_FLUXNET2015_SUBSET_HH_2003-2013_1-4.csv")
# print(fluxnet.columns)
# org = fluxnet['SW_IN_F']
# otemp = fluxnet['TA_F']
# ovpd = fluxnet['VPD_F']
# oswc = fluxnet['SWC_F_MDS_2']
# oheat = fluxnet['H_CORR_75']
# # oppt = fluxnet['P_F']
# nee = fluxnet['NEE_VUT_25']
# ogpp = fluxnet['GPP_NT_VUT_25']
# oreco = fluxnet['RECO_NT_VUT_25']

# # *********************************************************
# # "Load fluxnet 2015 data for grassland IT-Mbo site hourly data"
# fluxnet = pd.read_csv("/home/ahmad/PycharmProjects/deepCause/datasets/fluxnet2015/FLX_IT-MBo_FLUXNET2015_SUBSET_2003-2013_1-4/FLX_IT-MBo_FLUXNET2015_SUBSET_HH_2003-2013_1-4.csv")
# print(fluxnet.columns)
# rg = normalize(fluxnet['SW_IN_F'])
# temp = normalize(fluxnet['TA_F'])
# vpd = normalize(fluxnet['VPD_F'])
# swc = normalize(fluxnet['SWC_F_MDS_2'])
# # heat = normalize(deseasonalize(fluxnet['H_CORR_75'], 1))
# # oppt = normalize(deasonalize(fluxnet['P_F'], 1))
# nee = normalize(fluxnet['NEE_VUT_25'])
# gpp = normalize(fluxnet['GPP_NT_VUT_25'])
# reco = normalize(fluxnet['RECO_NT_VUT_25'])

# LOad synthetic data *************************
df = pd.read_csv("/home/ahmad/PycharmProjects/deepCausality/datasets/ncdata/synthetic_data.csv")
# X1 = normalize(down_sample(np.array(df['Z1']), win_size))
# X2 = normalize(down_sample(np.array(df['Z2']), win_size))
# X3 = normalize(down_sample(np.array(df['Z3']), win_size))
# X4 = normalize(down_sample(np.array(df['Z4']), win_size))
# X5 = normalize(down_sample(np.array(df['Z5']), win_size))
# X6 = normalize(down_sample(np.array(df['Z6']), win_size))
# X7 = normalize(down_sample(np.array(df['Z7']), win_size))
# X8 = normalize(down_sample(np.array(df['Z8']), win_size))
# X9 = normalize(down_sample(np.array(df['Z9']), win_size))
# X10 = normalize(down_sample(np.array(df['Z10']), win_size))

# # ************* LOad FLUXNET2015 data ************************
# rg = normalize(down_sample(org, win_size))
# temp = normalize(down_sample(otemp, win_size))
# gpp = normalize(down_sample(nee, win_size, partition='gpp'))
# reco = normalize(down_sample(nee, win_size, partition='reco'))
# # ppt = normalize(down_sample(oppt, win_size))
# vpd = normalize(down_sample(ovpd, win_size))
# swc = normalize(down_sample(oswc, win_size))
# heat = normalize(down_sample(oheat, win_size))
# # # print("Length:", len(rg))

# data = {'Rg': rg[7000: 17000], 'T': temp[7000: 17000], 'GPP': gpp[7000: 17000], 'Reco': reco[7000: 17000]}
# df = pd.DataFrame(data, columns=['Rg', 'T', 'GPP', 'Reco'])

# data = {'Z1': X1[:3000], 'Z2': X2[:3000], 'Z3': X3[:3000], 'Z4': X4[:3000], 'Z5': X5[:3000],
#         'Z6': X6[:3000], 'Z7': X7[:3000], 'Z8': X8[:3000], 'Z9': X9[:3000], 'Z10': X10[:3000]}
# df = pd.DataFrame(data, columns=['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9', 'Z10'])
# df.replace([np.inf, -np.inf], np.nan, inplace=True)
# print("Nans:", df.isnull().values.any())

# # Plot fluxnet after normalization and (daily) aggregation
# fig = plt.figure()
# ax1 = fig.add_subplot(411)
# ax1.plot(reco[7000:17000])
# # ax1.plot(recoo)
# ax1.set_ylabel('Reco')
#
# ax2 = fig.add_subplot(412)
# ax2.plot(temp[7000:17000])
# # ax2.plot(tempp)
# ax2.set_ylabel("Temp")
#
# ax3 = fig.add_subplot(413)
# ax3.plot(gpp[7000:17000])
# # ax3.plot(gppp)
# ax3.set_ylabel("GPP")
#
# ax4 = fig.add_subplot(414)
# ax4.plot(rg[7000:17000])
# # ax4.plot(rgg)
# ax4.set_ylabel("Rg")
# plt.show()

# /////////////////////////////////////////////////////////////
original_data = []
train_data = []
columns = df.columns
dim = len(df.columns)
print(f"Dimension {dim} and Columns: {df.columns}")

for col in df:
    # print("Col1:", len(df[col]))
    original_data.append(df[col])
    # original_data.append(normalize(down_sample(df[col], win_size)))

original_data = np.array(original_data)

# dataobj = RiverData()
# data = dataobj.get_data()
# xts = data['Kempten']
# yts = data['Dillingen']
# zts = data['Lenggries']

train_ds = ListDataset(
    [
        {'start': "01/01/1961 00:00:00",
         'target': original_data[:, 0: training_length].tolist()
         }
    ],
    freq=freq,
    one_dim_target=False
)

# create estimator
estimator = DeepAREstimator(
    prediction_length=prediction_length,
    context_length=prediction_length,
    freq=freq,
    num_layers=7,
    num_cells=70,
    dropout_rate=0.1,
    trainer=Trainer(
        ctx="cpu",
        epochs=epochs,
        hybridize=False,
        batch_size=48
    ),
    distr_output=MultivariateGaussianOutput(dim=dim)
)

# model_path = "models/trained_model_eco22Dec.sav"
model_path = "models/trained_model_syn26Jan.sav"
filename = pathlib.Path(model_path)
if not filename.exists():
    print("Training forecasting model....")
    predictor = estimator.train(train_ds)
    # save the model to disk
    pickle.dump(predictor, open(filename, 'wb'))

# # Test the model

# path = "models/counterfactual_model.sav"
# train_vars = [rg, temp, gpp, reco]
# test_vars = [temp, gpp, reco]
# target = rg
# category = 1
# obj = Counterfactuals(path, train_vars, test_vars, target, category)
# counterfactuals = obj.generate()
# np.random.shuffle(counterfactuals)

# plt.plot(counterfactuals)
# plt.plot(target[500: 1500])
# plt.show()

# Generate Knockoffs
category = 3
data_actual = np.array(original_data[:, :]).transpose()
n = len(original_data[:, 0])
obj = Knockoffs()
knockoffs = obj.GenKnockoffs(n, dim, data_actual)

# counterfactuals = np.array(knockoffs[:, category-1])
# print("Deep Knockoffs: \n", counterfactuals)
# print("Deep Knockoffs size: \n", len(counterfactuals))

# plt.plot(np.arange(0, len(counterfactuals)), original_data[3][: len(counterfactuals)], counterfactuals)
# plt.show()

# corr = np.corrcoef(counterfactuals, target[0: len(counterfactuals)])
# print(f"Correlation Coefficient (Variable, Counterfactual): {corr}")
prior_graph = np.array([[1, 0, 1, 0, 0, 0, 0, 0, 0],   [0, 1, 1, 0, 0, 0, 0, 0, 0],   [0, 0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0, 0, 0],   [0, 0, 0, 0, 0, 0, 0, 0, 0],   [0, 0, 0, 0, 1, 0, 1, 0, 0],
               [0, 0, 0, 1, 0, 0, 0, 1, 1],   [0, 0, 0, 0, 0, 0, 0, 1, 0],   [0, 0, 0, 0, 0, 0, 0, 0, 1],
               [0, 0, 0, 0, 0, 0, 0, 0, 0]])

params = {
    'num_samples': num_samples,
    'col': columns,
    'pred_len': prediction_length,
    'train_len': training_length,
    'prior_graph': prior_graph,
    'dim': dim,
    'freq': freq
    }

deepCause(original_data, knockoffs, model_path, params)