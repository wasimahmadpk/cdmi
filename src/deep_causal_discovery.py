import os
import re
import time
import pickle
import pathlib
import numpy as np
import pandas as pd
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt
from functions import *
from knockoffs import Knockoffs
from forecast import model_inference
from gluonts.dataset.common import ListDataset
from scipy.stats import ks_2samp, spearmanr

np.random.seed(1)

def execute_causal_pipeline(df, model_path, pars):

    # Extract Parameters
    num_samples = pars.get("num_samples")
    training_length = pars.get("train_len")
    prediction_length = pars.get("pred_len")
    frequency = pars.get("freq")
    plot_path = pars.get("plot_path")
    num_windows = pars.get("num_sliding_win")
    step_size = pars.get("step_size")
    model_name = pars.get("model_name")
    columns = pars.get('col')
    plot_forecasts = pars.get('plot_forecasts', False)

    if plot_forecasts and plot_path:
        os.makedirs(plot_path, exist_ok=True)

    n = df.shape[1]
    pars['dim'] = n
    pars['length'] = n

    filename = pathlib.Path(model_path)

    conf_mat_all = [[] for _ in range(4)]
    pvalues_all = [[] for _ in range(4)]
    kvalues_all = [[] for _ in range(4)]

    for i in range(n):
        indist_cause, outdist_cause, mean_cause, uni_cause = [], [], [], []
        pval_lists = [[] for _ in range(4)]
        kval_lists = [[] for _ in range(4)]

        data_actual = df.iloc[:training_length + prediction_length].to_numpy()
        obj = Knockoffs()
        knockoffs = obj.Generate_Knockoffs(data_actual, pars)
        knockoff_sample = np.array(knockoffs[:, i])

        mean = np.random.normal(0, 0.05, len(knockoff_sample)) + df.iloc[:, i].mean()
        outdist = np.random.normal(3, 3, len(knockoff_sample))
        uniform = np.random.uniform(df.iloc[:, i].min(), df.iloc[:, i].max(), len(knockoff_sample))
        interventionlist = [knockoff_sample, outdist, mean, uniform]
        intervention_methods = ['Knockoffs', 'Out-dist', 'Mean', 'Uniform']

        for j in range(n):
            results = {k: [] for k in range(4)}
            results_int = {k: [] for k in range(4)}

            for m, intervention in enumerate(interventionlist):
                start = 0
                for win in range(num_windows):
                    test_data = df.iloc[start:start + training_length + prediction_length].copy()
                    int_data = test_data.copy()
                    int_data.iloc[:, i] = intervention

                    test_ds = ListDataset([{
                        'start': test_data.index[0],
                        'target': test_data.values.T.tolist()
                    }], freq=frequency, one_dim_target=False)

                    test_dsint = ListDataset([{
                        'start': test_data.index[0],
                        'target': int_data.values.T.tolist()
                    }], freq=frequency, one_dim_target=False)

                    # Run forecasting with intervention
                    forecast_actual, mse, mape = model_inference(model_path, test_ds, num_samples, test_data.iloc[:, j], j, prediction_length, 0, False, 0)
                    forecast_int, mseint, mapeint = model_inference(model_path, test_dsint, num_samples, test_data.iloc[:, j], j, prediction_length, 0, True, m)

                    results[m].append(mape)
                    results_int[m].append(mapeint)

                    if plot_forecasts and plot_path:
                        plt.figure(figsize=(8, 4))
                        true_values = test_data.iloc[-prediction_length:, j].values
                        plt.plot(true_values, label="True", linestyle='--')
                        plt.plot(forecast_actual, label="Forecast (original)", color='blue')
                        plt.plot(forecast_int, label=f"Forecast (intervened: {intervention_methods[m]})", color='red')
                        plt.title(f"Z{i} → Z{j} [{intervention_methods[m]}]")
                        plt.xlabel("Time")
                        plt.ylabel(f"Z{j}")
                        plt.legend()
                        filename = f"{plot_path}/forecasts/forecast_{i}_to_{j}_{intervention_methods[m].lower()}.pdf"
                        plt.tight_layout()
                        plt.savefig(filename, dpi=600)
                        plt.close()

                    start += step_size

            for m in range(4):
                corr, pv_corr = spearmanr(results[m], results_int[m])
                t, p = ks_2samp(np.array(results[m]), np.array(results_int[m]))
                kld = kl_divergence(np.array(results[m]), np.array(results_int[m]))

                decision = 1 if p < pars['alpha'] else 0

                cause_type = intervention_methods[m]
                status = "REJECTED" if decision == 1 else "ACCEPTED"
                print(f"[{cause_type}] Hypothesis for link {columns[i]} → {columns[j]}: {status} (p = {p:.4f})")

                if m == 0: indist_cause.append(decision)
                if m == 1: outdist_cause.append(decision)
                if m == 2: mean_cause.append(decision)
                if m == 3: uni_cause.append(decision)

                pval_lists[m].append(p)
                kval_lists[m].append(kld)

            print('-------------***-------------***---------------***------------')

        for m in range(4):
            pvalues_all[m].append(pval_lists[m])
            kvalues_all[m].append(kval_lists[m])

        conf_mat_all[0] += indist_cause
        conf_mat_all[1] += outdist_cause
        conf_mat_all[2] += mean_cause
        conf_mat_all[3] += uni_cause

    pred = np.array(pvalues_all[0])
    actual = remove_diagonal_and_flatten(np.array(pars['ground_truth']))
    pred_score = remove_diagonal_and_flatten(pred)
    _, fmax = f1_max(actual, pred_score)
    print(f'F-max: {fmax:.2f}')

    pred_conf_mat = np.array(conf_mat_all[3]).reshape(n, n)

    print(f'Actual: \n {pars["ground_truth"]}')
    print(f'Predicted: \n {pred_conf_mat}')

    metrics = evaluate_best_predicted_graph(np.array(pars['ground_truth']), np.array([pred_conf_mat]))

    causal_matrix_thresholded = np.where(np.abs(np.array(pvalues_all[0])) < 0.10, 1, 0)
    plot_causal_graph(causal_matrix_thresholded, columns, model_name)
    evaluate(np.array(pars['ground_truth']).flatten(), conf_mat_all, intervention_methods)

    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")

    return metrics, pred_conf_mat, time.time()
