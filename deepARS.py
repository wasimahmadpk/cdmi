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

# Parameters for synthetic data
freq = '30min'
epochs = 100
win_size = 1

training_length = 555
prediction_length = 50
num_samples = 10

# LOad synthetic data *************************
df = pd.read_csv("/home/ahmad/PycharmProjects/deepCausality/datasets/ncdata/synthetic_data.csv")

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

# /////////////////////////////////////////////////////////////
original_data = []
train_data = []
columns = df.columns
dim = len(df.columns)
print(f"Dimension {dim} and Columns: {df.columns}")

for col in df:
    original_data.append(df[col])
    # original_data.append(normalize(down_sample(df[col], win_size)))

original_data = np.array(original_data)
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
    num_layers=3,
    num_cells=30,
    dropout_rate=0.1,
    trainer=Trainer(
        ctx="cpu",
        epochs=epochs,
        hybridize=False,
        batch_size=32
    ),
    distr_output=MultivariateGaussianOutput(dim=dim)
)

# model_path = "models/trained_model_eco22Dec.sav"
model_path = "models/trained_model_syn10Feb.sav"
filename = pathlib.Path(model_path)
if not filename.exists():
    print("Training forecasting model....")
    predictor = estimator.train(train_ds)
    # save the model to disk
    pickle.dump(predictor, open(filename, 'wb'))

# Generate Knockoffs
category = 3
data_actual = np.array(original_data[:, :]).transpose()
n = len(original_data[:, 0])
obj = Knockoffs()
knockoffs = obj.GenKnockoffs(n, dim, data_actual)

# counterfactuals = np.array(knockoffs[:, category-1])
# Show variables with its knockoff copy
# plt.plot(np.arange(0, len(counterfactuals)), original_data[3][: len(counterfactuals)], counterfactuals)
# plt.show()

# Check for correlation between knockoff and original variable
# corr = np.corrcoef(counterfactuals, target[0: len(counterfactuals)])
# print(f"Correlation Coefficient (Variable, Counterfactual): {corr}")

# Causal skeletion based on prior assumptions/ expert knowledge
prior_graph = np.array([[1, 1, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]])

# Parameters dict
params = {
    'num_samples': num_samples,
    'col': columns,
    'pred_len': prediction_length,
    'train_len': training_length,
    'prior_graph': prior_graph,
    'dim': dim,
    'freq': freq
    }

# Function for estimating causal impact among variables
deepCause(original_data, knockoffs, model_path, params)