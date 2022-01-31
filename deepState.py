from gluonts.dataset import common
from gluonts.model import deepstate
from gluonts.trainer import Trainer
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from netCDF4 import Dataset
from itertools import islice
import confidence
import numpy as np
import crps
import pandas as pd
import matplotlib.pyplot as plt
import netCDF


"Load fluxnet data"
nc_f = '/home/ahmad/PycharmProjects/deepCause/datasets/ncdata/DE-Hai.2000.2006.hourly.nc'  # Your filename
nc_fid = Dataset(nc_f, 'r')   # Dataset is the class behavior to open the file                         # and create an instance of the ncCDF4 class
nc_attrs, nc_dims, nc_vars = netCDF.ncdump(nc_fid);

# Extract data from NetCDF file
vpd = nc_fid.variables['VPD_f'][:].ravel().data  # extract/copy the data
temp = nc_fid.variables['Tair_f'][:].ravel().data
rg = nc_fid.variables['Rg_f'][:].ravel().data
swc1 = nc_fid.variables['SWC1_f'][:].ravel().data
nee = nc_fid.variables['NEE_f'][:].ravel().data
gpp = nc_fid.variables['GPP_f'][:].ravel().data
reco = nc_fid.variables['Reco'][:].ravel().data
le = nc_fid.variables['LE_f'][:].ravel().data
h = nc_fid.variables['H_f'][:].ravel().data
time = nc_fid.variables['time'][:].ravel().data

# Parameters
freq = '30min'
epochs = 100

training_length = 1008  # data for 3 weeks
prediction_length = 144  # dat for 3 days

start = 29000
train_stop = start + training_length
test_stop = train_stop + prediction_length

train_data = common.ListDataset(
    [
        {'start': "01/01/2006 00:00:00", 'target': reco[start:train_stop],
         'cat':[1]
         },
        {'start': "01/01/2006 00:00:00", 'target': reco[start:train_stop],
         'dynamic_feat': [temp[start:test_stop]],
         'cat':[2]
         }
    ],
    freq=freq)

test_data = common.ListDataset(
    [
        {'start': "01/01/2006 00:00:00", 'target': reco[start:test_stop],
         'dynamic_feat': [temp[start:test_stop]],
         'cat':[2]
        }
    ],
    freq=freq)

trainer = Trainer(epochs=epochs, hybridize=True)
estimator = deepstate.DeepStateEstimator(
    freq=freq, prediction_length=prediction_length, cardinality=[2],
    use_feat_static_cat=False, trainer=trainer)
predictor = estimator.train(training_data=train_data)

forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_data,  # test dataset
    predictor=predictor,  # predictor
    num_samples=prediction_length,  # number of sample paths we want for evaluation
)


def plot_forecasts(tss, forecasts, past_length, num_plots):
    counter = 0
    for target, forecast in islice(zip(tss, forecasts), num_plots):
        ax = target[-past_length:].plot(figsize=(12, 5), linewidth=2)
        forecast.plot(color='g')
        plt.grid(which='both')
        plt.legend(["observations", "median prediction", "90% confidence interval", "50% confidence interval"])
        plt.title("Forecasting " + titles[counter] + " time series")
        plt.xlabel("Timestamp")
        plt.ylabel(titles[counter])
        plt.show()
        counter += 1


forecasts = list(forecast_it)
tss = list(ts_it)
titles = ['Reco', 'Temperature']
# titles = ['Reco', 'Temperature', 'Rg', 'GPP']
plot_forecasts(tss, forecasts, past_length=600, num_plots=2)

evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9], seasonality=48)

agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_data))
print("Performance metrices", agg_metrics)
