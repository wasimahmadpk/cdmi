import math
import netCDF
import pickle
import pandas as pd
import numpy as np
import mxnet as mx
import pathlib
from os import path
from math import sqrt
from netCDF4 import Dataset
from datetime import datetime
import matplotlib.pyplot as plt
from itertools import islice
from gluonts.evaluation import Evaluator
from gluonts.dataset.common import ListDataset
from sklearn.metrics import mean_absolute_error
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.deepar._network import DeepARTrainingNetwork
from gluonts.trainer import Trainer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from gluonts.distribution.multivariate_gaussian import MultivariateGaussianOutput
from gluonts.evaluation.backtest import make_evaluation_predictions
import warnings
# np.random.seed(1)
# mx.random.seed(2)


def running_avg_effect(y, yint):

    rae = 0
    rae_list = []
    for i in range(len(y)):
        rae = 1/(1 + i) * (rae + (abs(y[i] - yint[i])))
        rae_list.append()
    return np.mean(rae_list)

def mean_absolute_percentage_error(y_true, y_pred):

    mape = np.mean(np.abs((y_true - y_pred)/y_true))*100
    print("MAPE: ", mape)
    return mape


def modelTest(model_path, test_ds, num_samples, data, idx, prediction_length, count, intervention, in_type):
    filename = pathlib.Path(model_path)
    # load the model from disk
    predictor = pickle.load(open(filename, 'rb'))

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,  # test dataset
        predictor=predictor,  # predictor
        num_samples=num_samples,  # number of sample paths we want for evaluation
    )

    if intervention == True:
        heuristic_itn_types = ['In-dist', 'Out-dist', 'Mean']
        int_title = 'After ' + heuristic_itn_types[in_type] + ' Intervention'
    else:
        int_title = ''

    def plot_forecasts(tss, forecasts, past_length, num_plots):

        for target, forecast in islice(zip(tss, forecasts), num_plots):

            ax = target[-past_length:][idx].plot(figsize=(14, 10), linewidth=2)
            forecast.copy_dim(idx).plot(color='g')
            plt.grid(which='both')
            plt.legend(["observations", "median prediction", "90% confidence interval", "50% confidence interval"])
            plt.title(f"Forecasting time series {int_title}")
            plt.xlabel("Timestamp")
            plt.ylabel('Target Time-series')
            plt.show()

    forecasts = list(forecast_it)
    tss = list(ts_it)
    y_pred = []

    for i in range(num_samples):
        y_pred.append(forecasts[0].samples[i].transpose()[idx].tolist())

    y_pred = np.array(y_pred)
    y_true = data[-prediction_length:]

    # mape = mean_absolute_percentage_error(y_true, np.mean(y_pred, axis=0))
    mape = mean_absolute_error(y_true, np.mean(y_pred, axis=0))*100
    mse = mean_squared_error(y_true, np.mean(y_pred, axis=0))
    mae = mean_absolute_error(y_true, np.mean(y_pred, axis=0))

    # meanerror = np.mean(np.mean(y_pred, axis=0))

    counter = -1
    if count < counter:

        plot_forecasts(tss, forecasts, past_length=100, num_plots=1)
        evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
        agg_metrics, item_metrics = evaluator(iter([pd.DataFrame((tss[0][:][idx]))]),
                                              iter([forecasts[0].copy_dim(idx)]), num_series=len(test_ds))
        print("Performance metrics", agg_metrics)

    # return agg_metrics['MSE'], agg_metrics['MAPE'], list(np.mean(y_pred, axis=0))
    return mse, mape, list(np.mean(y_pred, axis=0))