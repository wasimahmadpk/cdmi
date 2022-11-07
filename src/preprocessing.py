import math
import h5py
import pickle
import random
import pathlib
import parameters
import numpy as np
from os import path
import pandas as pd
from math import sqrt
from datetime import datetime
from scipy.special import stdtr
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression, mutual_info_regression

np.random.seed(1)
pars = parameters.get_geo_params()

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
    stations = ["dillingen", "kempten", "lenggries"]
        # Read the average daily discharges at each of these stations and combine them into a single pandas dataframe
    average_discharges = None

    for station in stations:

        filename = pathlib.Path("../datasets/river_discharge_data/data_" + station + ".csv")
        new_frame = pd.read_csv(filename, sep=";", skiprows=range(10))
        new_frame = new_frame[["Datum", "Mittelwert"]]

        new_frame = new_frame.rename(columns={"Mittelwert": station.capitalize(), "Datum": "Date"})
        new_frame.replace({",": "."}, regex=True, inplace=True)

        new_frame[station.capitalize()] = new_frame[station.capitalize()].astype(float)

        if average_discharges is None:
            average_discharges = new_frame
        else:
            average_discharges = average_discharges.merge(new_frame, on="Date")
    
    dillingen = average_discharges.iloc[:, 1].tolist()
    kempton = average_discharges.iloc[:, 2].tolist()
    lenggries = average_discharges.iloc[:, 3].tolist()

    data = {'Kt': kempton, 'Dt': dillingen, 'Lt': lenggries}
    df = pd.DataFrame(data, columns=['Kt', 'Dt', 'Lt'])
    df = df.apply(normalize)

    return df


def load_climate_data():
    # Load river discharges data
    
    df = pd.read_csv('/home/ahmad/PycharmProjects/deepCausality/datasets/environment_dataset/light.txt', sep=" ", header=None)
    df.columns = ["NEP", "PPFD"]
    df = df.apply(normalize)

    return df


def load_geo_data():
    # Load river discharges data
    path = '/home/ahmad/PycharmProjects/deepCausality/datasets/geo_dataset/moxa_data.csv'
    # vars = ['tides_ew', 'tides_ns', 'rain', 'temperature_outside', 'pressure_outside', 'gw_mb', 'gw_sr', 'gw_west', 'snow_load', 'wind_x', 'wind_y', 'humidity', 'glob_radiaton', 'strain_ew_corrected', 'strain_ns_corrected']
    # vars = ['DateTime', 'strain_ew', 'strain_ns', 'tides_ns', 'temperature_outside', 'pressure_outside', 'gw_mb', 'gw_west',
    #         'snow_load', 'wind_x', 'humidity', 'glob_radiaton']
    vars = ['DateTime', 'strain_ew', 'strain_ns', 'gw_west', 'wind_x', 'wind_y', 'humidity']
    data = pd.read_csv(path, usecols=vars)

    # Read spring and summer season geo-climatic data
    mask = (data['DateTime'] > '2015-05-01') & (data['DateTime'] <= '2015-08-29')
    df = data.loc[mask]
    df = df.set_index('DateTime')
    df = df.apply(normalize)

    # df = pd.DataFrame(data[vars].values[:1500], columns=list(vars))

    return df



def load_hackathon_data():
    # Load river discharges data
    bot, bov = simple_load_csv("/home/ahmad/PycharmProjects/deepCausality/datasets/hackathon_data/blood-oxygenation_interpolated_3600_pt_avg_14.csv")
    wt, wv = simple_load_csv("/home/ahmad/PycharmProjects/deepCausality/datasets/hackathon_data/weight_interpolated_3600_pt_avg_6.csv")
    hrt, hrv = simple_load_csv("/home/ahmad/PycharmProjects/deepCausality/datasets/hackathon_data/resting-heart-rate_interpolated_3600_iv_avg_4.csv")
    st, sv = simple_load_csv("/home/ahmad/PycharmProjects/deepCausality/datasets/hackathon_data/step-amount_interpolated_3600_iv_ct_15.csv")
    it, iv = simple_load_csv("/home/ahmad/PycharmProjects/deepCausality/datasets/hackathon_data/in-bed_interpolated_3600_iv_sp_19.csv")
    at, av = simple_load_csv("/home/ahmad/PycharmProjects/deepCausality/datasets/hackathon_data/awake_interpolated_3600_iv_sp_18.csv")

        # plt.plot(bov)
        # plt.plot(wv)
        # plt.plot(hrv)
        # plt.show()

        # v15 = np.nan_to_num(aggregate_avg(ts_15, v_15, 60 * 60))
        # v3 = np.nan_to_num(aggregate_avg(ts_3, v_3, 60 * 60))
        # v2 = np.nan_to_num(aggregate_avg(ts_2, v_2, 60 * 60))
        # v1 = np.nan_to_num(aggregate_avg(ts_1, v_1, 60 * 60))

    data = {'BO': bov[7500:10000], 'WV': wv[7500:10000], 'HR': hrv[7500:10000], 'Step': sv[7500:10000], 'IB': iv[7500:10000], 'Awake': av[7500:10000]}
    df = pd.DataFrame(data, columns=['BO', 'WV', 'HR', 'Step', 'IB', 'Awake'])

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
    data = pd.read_csv("../datasets/synthetic_datasets/synthetic_data.csv")
    df = data.apply(normalize)
    return df