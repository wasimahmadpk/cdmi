import math
import pickle
import pathlib
import numpy as np
import mxnet as mx
from os import path
import pandas as pd
from math import sqrt
from itertools import islice
import matplotlib.pyplot as plt
from gluonts.evaluation import Evaluator
from gluonts.dataset.common import ListDataset
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
from gluonts.evaluation.backtest import make_evaluation_predictions
import warnings
# np.random.seed(1)
# mx.random.seed(2)


def mean_absolute_percentage_error(y_true, y_pred):

    mape = np.mean(np.abs((y_true - y_pred)/y_true))*100
    print("MAPE: ", mape)
    return mape


def modelTest(model_path, test_ds, test_dsint, num_samples, data, idx, prediction_length, count, intervention, in_type):
    filename = pathlib.Path(model_path)
    # load the model from disk
    predictor = pickle.load(open(filename, 'rb'))

    if intervention == True:
        heuristic_itn_types = ['In-dist', 'Out-dist', 'Mean', 'Uniform']
        int_title = 'After ' + heuristic_itn_types[in_type] + ' Intervention'
        test_data = test_dsint
    else:
        int_title = ''
        test_data = test_ds

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,  # test dataset
        predictor=predictor,  # predictor
        num_samples=num_samples,  # number of sample paths we want for evaluation
    )

    forecast_itint, ts_itint = make_evaluation_predictions(
        dataset=test_dsint,  # test dataset
        predictor=predictor,  # predictor
        num_samples=num_samples,  # number of sample paths we want for evaluation
    )

    def plot_forecasts(tss, forecasts, forecastint, past_length, num_plots):

        for target, forecast in islice(zip(tss, forecasts), num_plots):

            ax = target[-past_length:][idx].plot(figsize=(14, 10), linewidth=2)
            forecast.copy_dim(idx).plot(color='g')
            plt.grid(which='both')
            # plt.legend(["observations", "median prediction", "90% confidence interval", "50% confidence interval"])
            plt.title(f"Forecasting time series {int_title}")
            plt.xlabel("Timestamp")
            plt.ylabel('NEP')

        for target, forecast in islice(zip(tss, forecastint), num_plots):
            # ax = target[-past_length:][idx].plot(figsize=(14, 10), linewidth=2)
            forecast.copy_dim(idx).plot(color='r')
            plt.grid(which='both')
            plt.legend(["observations", "median prediction", "90% confidence interval", "50% confidence interval"])
            plt.title(f"Forecasting time series {int_title}")
            plt.xlabel("Timestamp")
            plt.ylabel('NEP')
            plt.show()



    forecasts = list(forecast_it)
    tss = list(ts_it)

    forecasts_int = list(forecast_itint)
    tss_int = list(ts_itint)
    y_pred = []

    for i in range(num_samples):
        y_pred.append(forecasts[0].samples[i].transpose()[idx].tolist())

    y_pred = np.array(y_pred)
    y_true = data[-prediction_length:]
    
    # mape = mean_absolute_percentage_error(y_true, np.mean(y_pred, axis=0))
    mape = mean_absolute_error(y_true, np.mean(y_pred, axis=0))
    mse = mean_squared_error(y_true, np.mean(y_pred, axis=0))
    mae = mean_absolute_error(y_true, np.mean(y_pred, axis=0))

    # meanerror = np.mean(np.mean(y_pred, axis=0))

    counter = 1
    print(f"TSS when intervention is : {intervention}-> {tss}")

    if count < counter:

        plot_forecasts(tss, forecasts, forecasts_int, past_length=150, num_plots=1)
        # if intervention == True:
        #     plot_forecasts(tss, forecasts_int, past_length=250, num_plots=1)
        evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
        agg_metrics, item_metrics = evaluator(iter([pd.DataFrame((tss[0][:][idx]))]),
                                                  iter([forecasts[0].copy_dim(idx)]), num_series=len(test_ds))
        print("Performance metrics", agg_metrics)

    # return agg_metrics['MSE'], agg_metrics['MAPE'], list(np.mean(y_pred, axis=0))

    return mse, mape, list(np.mean(y_pred, axis=0))