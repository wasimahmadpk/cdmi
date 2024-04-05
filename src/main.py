import pickle
import time
import pathlib
import parameters
import numpy as np
import mxnet as mx
import pandas as pd
import functions as func
import dataloader as datasets
import matplotlib.pyplot as plt
from knockoffs import Knockoffs
from regimes import get_regimes
from deepcause import deepCause
from gluonts.trainer import Trainer
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.distribution.multivariate_gaussian import MultivariateGaussianOutput

np.random.seed(1)
mx.random.seed(2)

start_time = time.time()
# Parameters
pars = parameters.get_syn_params()
freq = pars.get("freq")
epochs = pars.get("epochs")
win_size = pars.get("win_size")
slidingwin_size = pars.get("slidingwin_size")
training_length = pars.get("train_len")
prediction_length = pars.get("pred_len")
num_samples = pars.get("num_samples")
num_layers = pars.get("num_layers")
num_cells = pars.get("num_cells")
dropout_rate = pars.get("dropout_rate")
batch_size = pars.get("batch_size")
plot_path = pars.get("plot_path")
model_name = pars.get("model_name")

df = datasets.load_syn_data()
df.plot(figsize=(9, 4), linewidth=0.66, use_index=True)
plt.show()
func.corr_heatmap(df)

# # --------Identify Regimes in Time series--------
# regimes, _, _, newdf = get_regimes(data, slidingwin_size)
# # -----------------------------------------------

# for i in range(len(regimes)):
    # print(regimes[i].head(5))

# df = data.loc[:1000].copy()
print(f'Shape: {df.shape}')

# df = regimes[1].drop('Clusters', axis=1)
# df.plot.scatter(x='BO', y='Awake', c='blue')
# plt.xlabel("PPFD ($\mu$ mol photons $m^{2}s^{-1}$)")
# plt.ylabel("NEP ($\mu$ mol $CO_2$ $m^{2}s^{-1}$)")
# filename = pathlib.Path(plot_path + "PPFD->NEP_Scatter.pdf")
# plt.savefig(filename)
# plt.show()

original_data = []
dim, columns = len(df.columns), df.columns

for col in df:
    original_data.append(df[col])

original_data = np.array(original_data)
# training set
train_ds = ListDataset(
    [
        {'start': "01/03/2015 00:00:00",
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
    num_layers=num_layers,
    num_cells=num_cells,
    dropout_rate=dropout_rate,
    trainer=Trainer(
        ctx="cpu",
        epochs=epochs,
        hybridize=False,
        learning_rate=1E-4,
        batch_size=32
    ),
    distr_output=MultivariateGaussianOutput(dim=dim)
)

# load model if not already trained
path = pars.get('model_path')
model_path = pathlib.Path(path + model_name)
# model_path = "../models/trained_model_georegime_cli9.sav"
# model_path = "../models/trained_model_syn22Sep.sav"
# model_path = "../models/trained_model_river16Jun.sav"

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
params = {"length": n, "dim": dim, "col": columns}
knockoffs = obj.GenKnockoffs(data_actual, params)

# Function for estimating causal impact among variables
end_time = deepCause(original_data, knockoffs, model_path, params)

# Calculate difference
elapsed_time = end_time - start_time

# Print elapsed time
print("Computation time:", round(elapsed_time/60), "mins")
