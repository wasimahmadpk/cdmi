import math
import pickle
import random
import pathlib
import parameters
import numpy as np
import mxnet as mx
import pandas as pd
from os import path
from math import sqrt
from itertools import islice
import preprocessing as prep
from datetime import datetime
from riverdata import RiverData
from climatedata import ClimateData
from deepcause import deepCause
import matplotlib.pyplot as plt
from knockoffs import Knockoffs
from scipy.special import stdtr
from forecast import modelTest
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

# Parameters
pars = parameters.get_climate_params()
freq = pars.get("freq")
epochs = pars.get("epochs")
win_size = pars.get("win_size")
training_length = pars.get("train_len")
prediction_length = pars.get("pred_len")
num_samples = pars.get("num_samples")
num_layers = pars.get("num_layers")
num_cells = pars.get("num_cells")
dropout_rate = pars.get("dropout_rate")
batch_size = pars.get("batch_size")
plot_path = pars.get("plot_path")

# Load river discharges data
# df = prep.load_river_data()
df = prep.load_climate_data()
df = df.dropna().reset_index(drop=True)
df.plot.scatter(x='PPFD', y='NEP', c='blue')
plt.xlabel("PPFD ($\mu$ mol photons $m^{2}s^{-1}$)")
plt.ylabel("NEP ($\mu$ mol $CO_2$ $m^{2}s^{-1}$)")
filename = pathlib.Path(plot_path + "PPFD->NEP_Scatter.pdf")
plt.savefig(filename)
plt.show()

print(len(df))


original_data = []
dim = len(df.columns)
columns = df.columns
# print(f"Dimension {dim} and Columns: {df.columns}")

for col in df:
    original_data.append(df[col])

original_data = np.array(original_data)

# training set
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
    num_layers=num_layers,  # 3
    num_cells=num_cells,  # 33
    dropout_rate=dropout_rate,
    trainer=Trainer(
        ctx="cpu",
        epochs=epochs,
        hybridize=False,
        batch_size=batch_size
    ),
    distr_output=MultivariateGaussianOutput(dim=dim)
)

# load model if not already trained
# model_path = "../models/trained_model_syn08Jun.sav"
# model_path = "../models/trained_model_river16Jun.sav"
model_path = "../models/trained_model_climate03Aug.sav"
filename = pathlib.Path(model_path)
if not filename.exists():
    print("Training forecasting model....")
    predictor = estimator.train(train_ds)
    # save the model to disk
    pickle.dump(predictor, open(filename, 'wb'))

# Generate Knockoffs
data_actual = np.array(original_data[:, :]).transpose()
n = len(original_data[:, 0])
obj = Knockoffs()
knockoffs = obj.GenKnockoffs(n, dim, data_actual)

params = {"dim": dim, "col": columns}
# Function for estimating causal impact among variables
deepCause(original_data, knockoffs, model_path, params)