import math
import pickle
import random
import pathlib
import numpy as np
import parameters
import mxnet as mx
import pandas as pd
from os import path
from math import sqrt
from datetime import datetime
from riverdata import RiverData
import matplotlib.pyplot as plt
from knockoffs import Knockoffs
from scipy.special import stdtr
from gluonts.trainer import Trainer
from gluonts.evaluation import Evaluator
from sklearn.metrics import mean_squared_error

np.random.seed(1)
mx.random.seed(2)


def deseasonalize(var, interval):

    deseasonalize_data = []
    for i in range(interval, len(var)):
        value = var[i] - var[i - interval]
        deseasonalize_data.append(value)
    return deseasonalize_data


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

