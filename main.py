import math
import pickle
import random
import pathlib
import parameters
import numpy as np
import mxnet as mx
import pandas as pd
from os import path
import preprocessing
from math import sqrt
from itertools import islice
from datetime import datetime
from riverdata import RiverData
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


prep = preprocessing()
# Parameters for River discharge data
real_pars = parameters.get_real_params()
syn_pars = parameters.get_syn_params()
freq = 'D'
dim = 3
epochs = 150
win_size = 1

# # Parameters for synthetic data
# freq = '30min'
# epochs = 150
# win_size = 1

training_length = 555    # 555
prediction_length = 28   # 28
num_samples = 10

# # Parameters for ecosystem data
# freq = '30min'
# dim = 4
# epochs = 150
# win_size = 1
# #
# training_length = 666
# prediction_length = 48
# num_samples = 10


# Load river discharges data
dataobj = RiverData()
data = dataobj.get_data()

dillingen = data.iloc[:, 1].tolist()
kempton = data.iloc[:, 2].tolist()
# kempton = [0, 0, 0, 0, 0] + kempton
lenggries = data.iloc[:, 3].tolist()

# Load synthetic data *************************
# df = pd.read_csv("/home/ahmad/PycharmProjects/deepCausality/datasets/ncdata/synthetic_data.csv")


# # "Load fluxnet 2015 data for grassland IT-Mbo site"
# fluxnet = pd.read_csv("/home/ahmad/PycharmProjects/deepCausality/datasets/fluxnet2015/FLX_IT-MBo_FLUXNET2015_SUBSET_2003-2013_1-4/FLX_IT-MBo_FLUXNET2015_SUBSET_HH_2003-2013_1-4.csv")
# org = fluxnet['SW_IN_F']
# otemp = fluxnet['TA_F']
# ovpd = fluxnet['VPD_F']
# # oppt = fluxnet['P_F']
# # nee = fluxnet['NEE_VUT_50']
# ogpp = fluxnet['GPP_NT_VUT_50']
# oreco = fluxnet['RECO_NT_VUT_50']
# #
# # ************* LOad FLUXNET2015 data ************************

# rg = normalize(down_sample(org, win_size))
# temp = normalize(down_sample(otemp, win_size))
# # gpp = normalize(down_sample(nee, win_size, partition='gpp'))
# # reco = normalize(down_sample(nee, win_size, partition='reco'))
# gpp = normalize(down_sample(ogpp, win_size))
# reco = normalize(down_sample(oreco, win_size))
# # ppt = normalize(down_sample(oppt, win_size))
# vpd = normalize(down_sample(ovpd, win_size))
# # swc = normalize(down_sample(oswc, win_size))
# # heat = normalize(down_sample(oheat, win_size))
#
# data = {'Rg': rg[8000:12000], 'T': temp[8000:12000], 'GPP': gpp[8000:12000], 'Reco': reco[8000:12000]}
# df = pd.DataFrame(data, columns=['Rg', 'T', 'GPP', 'Reco'])

data = {'Kt': prep.normalize(kempton), 'Dt': prep.normalize(dillingen), 'Lt': prep.normalize(lenggries)}
df = pd.DataFrame(data, columns=['Kt', 'Dt', 'Lt'])
# /////////////////////////////////////////////////////////////
original_data = []
dim = len(df.columns)
print(f"Dimension {dim} and Columns: {df.columns}")

for col in df:
    original_data.append(df[col])

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
    num_layers=3,  # 3
    num_cells=33,  # 33
    dropout_rate=0.1,
    trainer=Trainer(
        ctx="cpu",
        epochs=epochs,
        hybridize=False,
        batch_size=32
    ),
    distr_output=MultivariateGaussianOutput(dim=dim)
)

# model_path = "models/trained_model_syn08Jun.sav"
model_path = "models/trained_model_river16Jun.sav"
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

# Causal skeletion based on prior assumptions/ expert knowledge
# prior_graph = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
prior_graph = np.array([[1, 1, 1, 1, 1], [0, 1, 0, 0, 1], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])

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