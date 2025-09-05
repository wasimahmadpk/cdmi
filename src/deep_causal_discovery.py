import os
import time
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from functions import *
from knockoffs import Knockoffs
from forecast import model_inference
from gluonts.dataset.common import ListDataset
from scipy.stats import ks_2samp, spearmanr, ttest_ind, anderson_ksamp

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
        os.makedirs(f"{plot_path}/forecasts", exist_ok=True)
        os.makedirs(f"{plot_path}/kde", exist_ok=True)

    n = df.shape[1]
    pars['dim'] = n
    pars['length'] = n

    conf_mat_all = [[] for _ in range(4)]
    pvalues_all = [[] for _ in range(4)]
    kvalues_all = [[] for _ in range(4)]

    for i in range(n):
        indist_cause, outdist_cause, mean_cause, uni_cause = [], [], [], []
        pval_lists = [[] for _ in range(4)]
        kval_lists = [[] for _ in range(4)]

        for j in range(n):
            results = {k: [] for k in range(4)}
            results_int = {k: [] for k in range(4)}

            for win in range(num_windows): #
                start = win * step_size
                end = start + training_length + prediction_length
                test_data = df.iloc[start:end].copy()

                data_actual = test_data.to_numpy()
                obj = Knockoffs()
                knockoff_samples = obj.Generate_Knockoffs(data_actual, pars)

                knockoffs = np.array(knockoff_samples[:, i]) + np.random.normal(20.0, 20.00, len(knockoff_samples[:, i])) #0, 2 SCMs
                mean = np.random.normal(0, 0.05, len(knockoffs)) + test_data.iloc[:, i].mean()
                outdist = np.random.normal(3, 3, len(knockoffs))
                uniform = np.random.uniform(test_data.iloc[:, i].min(), test_data.iloc[:, i].max(), len(knockoffs))

                interventionlist = [knockoffs] #, outdist, mean, uniform
                intervention_methods = ['Knockoffs'] #, 'Out-dist', 'Mean', 'Uniform'

                for m, intervention in enumerate(interventionlist):
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

                    forecast_actual, _, mape = model_inference(model_path, test_ds, num_samples, test_data.iloc[:, j], j,
                                                               prediction_length, 0, False, 0)
                    forecast_int, _, mapeint = model_inference(model_path, test_dsint, num_samples, test_data.iloc[:, j], j,
                                                               prediction_length, 0, True, m)
    
                    # mape, mapeint = forecast_actual, forecast_int
                    results[m].append(mape)
                    results_int[m].append(mapeint)

                    if plot_forecasts and plot_path and win < 1:
                        plt.figure(figsize=(8, 4))
                        true_values = test_data.iloc[-prediction_length:, j].values

                        # Generate horizon indices [1, 2, ..., N]
                        horizon = np.arange(1, len(true_values) + 1)

                        plt.plot(horizon, true_values, label="True", linestyle='--', color='green')
                        plt.plot(horizon, forecast_actual, label="Forecast", color='blue')

                        plt.xlabel("Forecast horizon", fontsize=20)
                        plt.ylabel(f"$Z_{{{j}}}$", fontsize=20)  # subscript j
                        plt.xticks(horizon, fontsize=18)
                        plt.yticks(fontsize=18)
                        # plt.ylim(top=1.3)
                        plt.legend(fontsize=16, loc='upper right')

                        # Generate a random integer
                        rnd = random.randint(1, 9999)
                        filename = f"{plot_path}/forecasts/forecast_Z{j}_win{win}_{rnd}.pdf"
                        plt.tight_layout()
                        plt.savefig(filename, dpi=600, format='pdf')

                        plt.plot(horizon, forecast_int, label=f"Counterfactual: {intervention_methods[m]}", color='red')
                        plt.legend(fontsize=16, loc='upper right')
                        filename = f"{plot_path}/forecasts/forecast_Z{i}_to_Z{j}_{intervention_methods[m].lower()}_win{win}.pdf"
                        plt.tight_layout()
                        plt.savefig(filename, dpi=600, format='pdf')
                        plt.close()

            # KDE plot
            if plot_forecasts and plot_path:
                for m in range(len(intervention_methods)):
                    baseline_arr = np.array(results[m])
                    intervened_arr = np.array(results_int[m])

                    plt.figure(figsize=(8, 5))
                    sns.kdeplot(baseline_arr, label="Actual", color='#008080', fill=True, alpha=0.77)
                    sns.kdeplot(intervened_arr, label=f"Intervened", color='#FFA500', fill=True, alpha=0.6)

                    plt.xlabel('Residuals', fontsize=18)
                    plt.ylabel(f"$Z_{{{i}}} \\; \\rightarrow \\; Z_{{{j}}}$", fontsize=18)
                    plt.xticks(fontsize=16)
                    plt.yticks(fontsize=16)
                    plt.legend(fontsize=16)

                    plt.tight_layout()
                    kde_file = f"{plot_path}/kde/kde_Z{i}_to_Z{j}_{intervention_methods[m].lower()}.pdf"
                    plt.savefig(kde_file, dpi=600, format='pdf')
                    plt.close()


            # Statistical tests
            for m in range(len(intervention_methods)):

                corr, pv_corr = spearmanr(results[m], results_int[m])
                t, p = ks_2samp(np.array(results[m]).ravel(), np.array(results_int[m]).ravel())
                # t, p = ttest_ind(np.array(results[m]), np.array(results_int[m]), equal_var=False)
                # result = anderson_ksamp([np.array(results[m]), np.array(results_int[m])])
                # t, p = result.statistic, result.significance_level
                kld = kl_divergence(np.array(results[m]), np.array(results_int[m]))
                decision = 1 if p < pars['alpha'] else 0

                status = "REJECTED" if decision else "ACCEPTED"
                print(f"[{intervention_methods[m]}] Hypothesis {columns[i]} → {columns[j]}: {status} (p = {p:.4f})")

                [indist_cause, outdist_cause, mean_cause, uni_cause][m].append(decision)
                pval_lists[m].append(p)
                kval_lists[m].append(kld)

            print('-------------***-------------***---------------***------------')

        for m in range(len(intervention_methods)):
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

    pred_conf_mat = np.array(conf_mat_all[0]).reshape(n, n)
    print(f'Actual: \n{pars["ground_truth"]}')
    print(f'Predicted: \n{pred_conf_mat}')

    metrics = evaluate_best_predicted_graph(np.array(pars['ground_truth']), np.array([pred_conf_mat]))
    causal_matrix_thresholded = np.where(np.abs(np.array(pvalues_all[0])) < 0.10, 1, 0)
    plot_causal_graph(causal_matrix_thresholded, columns, model_name)
    # evaluate(np.array(pars['ground_truth']).flatten(), conf_mat_all[0], intervention_methods)
    # metrics['Fscore'] = fmax
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")

    return metrics, pred_conf_mat, time.time()
