import math
import pickle
import random
import pathlib
import parameters
import parameters
import numpy as np
from os import path
import pandas as pd
from math import sqrt
from datetime import datetime
from scipy.special import stdtr
import matplotlib.pyplot as plt
from riverdata import RiverData
from sklearn.feature_selection import f_regression, mutual_info_regression

np.random.seed(1)
pars = parameters.get_real_params()

win_size = pars.get("win_size")
training_length = pars.get("train_len")
prediction_length = pars.get("pred_len")


def get_shuffled_ts(SAMPLE_RATE, DURATION, root):
    # Number of samples in normalized_tone
    N = SAMPLE_RATE * DURATION
    yf = rfft(root)
    xf = rfftfreq(N, 1 / SAMPLE_RATE)
    # plt.plot(xf, np.abs(yf))
    # plt.show()
    new_ts = irfft(shuffle(yf))
    return new_ts


def deseasonalize(var, interval):
    deseasonalize_data = []
    for i in range(interval, len(var)):
        value = var[i] - var[i - interval]
        deseasonalize_data.append(value)
    return deseasonalize_data


# def running_avg_effect(y, yint):

#  Break temporal dependency and generate a new time series
#     pars = parameters.get_sig_params()
#     SAMPLE_RATE = pars.get("sample_rate")  # Hertz
#     DURATION = pars.get("duration")  # Seconds
#     rae = 0
#     for i in range(len(y)):
#         ace = 1/((training_length + 1 + i) - training_length) * (rae + (y[i] - yint[i]))
#     return rae


def normalize(var):
    nvar = (np.array(var) - np.mean(var)) / np.std(var)
    return nvar


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


def mutual_information(x, y):
    mi = mutual_info_regression(x, y)
    mi /= np.max(mi)
    return mi


def load_river_data():
    # Load river discharges data
    dataobj = RiverData()
    data = dataobj.get_data()
    dillingen = data.iloc[:, 1].tolist()
    kempton = data.iloc[:, 2].tolist()
    lenggries = data.iloc[:, 3].tolist()

    data = {'Kt': normalize(kempton), 'Dt': normalize(dillingen), 'Lt': normalize(lenggries)}
    df = pd.DataFrame(data, columns=['Kt', 'Dt', 'Lt'])

    return df


def load_flux_data():

    # "Load fluxnet 2015 data for grassland IT-Mbo site"
    fluxnet = pd.read_csv("/home/ahmad/PycharmProjects/deepCausality/datasets/fluxnet2015/FLX_IT-MBo_FLUXNET2015_SUBSET_2003-2013_1-4/FLX_IT-MBo_FLUXNET2015_SUBSET_HH_2003-2013_1-4.csv")
    org = fluxnet['SW_IN_F']
    otemp = fluxnet['TA_F']
    ovpd = fluxnet['VPD_F']
    # oppt = fluxnet['P_F']
    # nee = fluxnet['NEE_VUT_50']
    ogpp = fluxnet['GPP_NT_VUT_50']
    oreco = fluxnet['RECO_NT_VUT_50']
    #
    # ************* LOad FLUXNET2015 data ************************

    rg = normalize(down_sample(org, win_size))
    temp = normalize(down_sample(otemp, win_size))
    # gpp = normalize(down_sample(nee, win_size, partition='gpp'))
    # reco = normalize(down_sample(nee, win_size, partition='reco'))
    gpp = normalize(down_sample(ogpp, win_size))
    reco = normalize(down_sample(oreco, win_size))
    # ppt = normalize(down_sample(oppt, win_size))
    vpd = normalize(down_sample(ovpd, win_size))
    # swc = normalize(down_sample(oswc, win_size))
    # heat = normalize(down_sample(oheat, win_size))

    data = {'Rg': rg[8000:12000], 'T': temp[8000:12000], 'GPP': gpp[8000:12000], 'Reco': reco[8000:12000]}
    df = pd.DataFrame(data, columns=['Rg', 'T', 'GPP', 'Reco'])

    return df


def load_syn_data():
    # Load synthetic data *************************
    df = pd.read_csv("/home/ahmad/PycharmProjects/deepCausality/datasets/ncdata/synthetic_data.csv")
    return df