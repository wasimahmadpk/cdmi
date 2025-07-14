import re
import time
import pickle
import pathlib
import numpy as np
import pandas as pd
from math import sqrt
import seaborn as sns
from functions import *
from knockoffs import Knockoffs
import matplotlib.pyplot as plt
from forecast import modelTest
from gluonts.dataset.common import ListDataset
from scipy.stats import ttest_ind, ttest_ind_from_stats, ttest_1samp, ks_2samp, kstest, spearmanr

np.random.seed(1)

def deepCause(odata, model, pars):

    num_samples = pars.get("num_samples")
    step = pars.get("step_size")
    training_length = pars.get("train_len")
    prediction_length = pars.get("pred_len")
    frequency = pars.get("freq")
    plot_path = pars.get("plot_path")
    num_windows = pars.get('num_sliding_win')
    step_size = pars.get('step_size')
    model_name = pars.get("model_name")
    n = pars.get('dim')

    columns = pars.get('col')
    mutual_info = []
    for a in range(len(odata)):
            x = odata[:].T
            y = odata[a].T
            mi = mutual_information(x, y)
            # print("MI Value: ", mi)
            mutual_info.append(mi)

    filename = pathlib.Path(model)
    if not filename.exists():
        print("Training forecasting model....")
        predictor = estimator.train(train_ds)
        # save the model to disk
        pickle.dump(predictor, open(filename, 'wb'))

    conf_mat = []
    conf_mat_mean = []
    conf_mat_indist = []
    conf_mat_outdist = []
    conf_mat_uniform = []
    
    pvalues = []
    pval_indist, pval_outdist, pval_mean, pval_uniform = [], [], [], []

    kvalues = []
    kval_indist, kval_outdist, kval_mean, kval_uniform = [], [], [], []

    for i in range(len(odata)):

        int_var = odata[i]
        int_var_name = "Z_" + str(i + 1) + ""
        var_list = []
        causal_decision = []
        mean_cause = []
        indist_cause = []
        outdist_cause = []
        uni_cause = []

        # P-Values
        pvi, pvo, pvm, pvu = [], [], [], []
        # KL-Divergence
        kvi, kvo, kvm, kvu = [], [], [], []

         # Generate Knockoffs
        data_actual = np.array(odata[: , 0: training_length + prediction_length]).transpose()
        n = len(data_actual[:, 0])
        pars.update({'length': n})
        obj = Knockoffs()
        knockoffs = obj.Generate_Knockoffs(data_actual, pars)
        knockoff_sample = np.array(knockoffs[0: training_length+prediction_length, i])

        mean = np.random.normal(0, 0.05, len(knockoff_sample)) + np.mean(odata[i])
        outdist = np.random.normal(150, 120, len(knockoff_sample))
        uniform = np.random.uniform(np.min(odata[i]), np.max(odata[i]), len(knockoff_sample))
        interventionlist = [knockoff_sample, outdist[: len(knockoff_sample)], mean, uniform]
        intervention_methods = ['In-dist', 'Out-dist', 'Mean', 'Uniform']

        for j in range(len(odata)):
            # back_door_int = []
            # back_door = prior_graph[:, j].nonzero()[0]
            # print("-------------*****-----------------------*****-------------")
            # print(f"Front/Backdoor Paths: {np.array(back_door) + 1} ---> {j + 1}")
            print("-------------*****-----------------------*****-------------")

            mselol = []
            mapelol = []
            mselolint = []
            mapelolint = []

            for m in range(len(interventionlist)):  # apply all types of intervention methods

                intervene = interventionlist[m]

                mselist = []      # list of MSE values for multiple realization without intervention
                mselistint = []   # list of MSE values for multiple realization with intervention
                mapelist = []     # list of MAPE values for multiple realization without intervention
                mapelistint = []  # list of MAPE values for multiple realization with intervention
                start = 0

                for iter in range(num_windows):  # 30
    
                    mselist_batch = []
                    mselistint_batch = []
                    mapelist_batch = []
                    mapelistint_batch = []
                    for r in range(1):

                        test_data = odata[:, start: start + training_length + prediction_length].copy()
                        test_ds = ListDataset(
                            [
                                {'start': "01/04/2001 00:00:00",
                                 'target': test_data
                                 }
                            ],
                            freq=frequency,
                            one_dim_target=False
                        )
                        int_data = odata[:, start: start + training_length + prediction_length].copy()
                        int_data[i, :] = intervene
                        test_dsint = ListDataset(
                            [
                                {'start': "01/04/2001 00:00:00",
                                 'target': int_data
                                 }
                            ],
                            freq=frequency,
                            one_dim_target=False
                        )

                        mse, mape, ypred = modelTest(model, test_ds, num_samples, test_data[j], j,
                                                     prediction_length, iter, False, 0)

                        mseint, mapeint, ypredint = modelTest(model, test_dsint, num_samples,
                                                              test_data[j], j,
                                                              prediction_length, iter, True, m)

                        if m == 0:
                            # Generate multiple version Knockoffs
                            data_actual = np.array(odata[:, start: start + training_length + prediction_length]).transpose()
                            obj = Knockoffs()
                            n = len(odata[:, 0])
                            knockoffs = obj.Generate_Knockoffs(data_actual, pars)
                            knockoff_sample = np.array(knockoffs[:, i])
                            intervene = knockoff_sample

                        mselist_batch.append(mse)
                        mapelist_batch.append(mape)
                        mselistint_batch.append(mseint)
                        mapelistint_batch.append(mapeint)

                    start = start + step_size                                       # Step size for sliding window # 10
                    mselist.append(np.mean(mselist_batch))                  # mselist = mselist_batch
                    mapelist.append(np.mean(mapelist_batch))                # mapelist = mapelist_batch
                    mselistint.append(np.mean(mselistint_batch))            # mselistint = mselistint_batch
                    mapelistint.append(np.mean(mapelistint_batch))          # mapelistint = mapelistint_batch

                # mselist = np.mean(mselist, axis=0)
                # mapelist = np.mean(mapelist, axis=0)
                # mselistint = np.mean(mselistint, axis=0)
                # mapelistint = np.mean(mapelistint, axis=0)

                msevar = np.var(mselist)
                mapevar = np.var(mapelist)
                mselol.append(mselist)
                mapelol.append(mapelist)
                mselolint.append(mselistint)
                mapelolint.append(mapelistint)

            var_list.append(np.var(mapelolint[1]))
            # print(f"MSE(Mean): {list(np.mean(mselol, axis=0))}")
            if len(columns) > 0:

                print(f"Causal Link: {columns[i]} --------------> {columns[j]}")
                print("-------------*****-----------------------*****-------------")
                fnamehist = plot_path + "{columns[i]}_{columns[j]}:hist"
            else:
                print(f"Causal Link: Z_{i + 1} --------------> Z_{j + 1}")
                print("-------------*****-----------------------*****-------------")
                fnamehist = plot_path + "{Z_[i + 1]}_{Z_[j + 1]}:hist"
            
            pvals = []
            kvals = []

            # ------------------------- plot residuals ---------------------------------------

            fig = plt.figure()
            ax = fig.add_subplot(111)

            # Calculate Spearman correlation coefficient and its p-value
            corr, p_val = spearmanr(mapelol[0], mapelolint[0])

            plt.plot(mapelol[0], color='g', alpha=0.7, label='Actual $\epsilon$')
            plt.plot(mapelolint[0], color='r', alpha=0.7, label='Counterfactual $\epsilon$')
            plt.title(f'corr: {round(corr, 2)}, p-val: {round(p_val, 2)}')
            if len(columns) > 0:
                # effect_var = re.sub(r'(\d+)', lambda match: f'$_{match.group(1)}$', columns[j])
                ax.set_ylabel(f'{columns[i]} ---> {columns[j]}')
            else:
                # plt.ylabel(f"CSS: Z_{i + 1} ---> Z_{j + 1}")
                ax.set_ylabel(f'Z_{i + 1} ---> Z_{j + 1}')

            plt.gcf()
            ax.legend()
            filename = pathlib.Path(plot_path + f'res_{columns[i]} ---> {columns[j]}.pdf')
            plt.savefig(filename)
            plt.close()

            # plt.show()
            # ---------------------------------------------------------------------------------
            
            for z in range(len(intervention_methods)):

                 # Calculate Spearman correlation coefficient and its p-value
                corr, pv_corr = spearmanr(mapelol[z], mapelolint[z])

                print("Intervention: " + intervention_methods[z])
                
                # print(f"Mean: {np.mean(mapelol[z])}, Mean Intervention: {np.mean(mapelolint[z])}")
                # print(f"Variance: {np.var(mapelol[z])}, Variance Intervention: {np.var(mapelolint[z])}")
                # t, p = ttest_ind(np.array(mapelolint[z]), np.array(mapelol[z]), equal_var=True)
                
                # invariance test
                t, p = ks_2samp(np.array(mapelol[z]), np.array(mapelolint[z]))
                t, p = round(t, 2), round(p, 2)
                # t, p = kstest(np.array(mapelolint[z]), np.array(mapelol[z]))
                
                kld = round(kl_divergence(np.array(mapelol[z]), np.array(mapelolint[z])), 2)
                kvals.append(kld)
                
                if i==j:
                    pvals.append(0)
                    p = 0
                else:
                    pvals.append(p)
                
                print(f'Test statistic: {t}, p-value: {p}, KLD: {kld}')
                if p < 0.05 or i==j:         # or mutual_info[i][j] > 0.90:
                    print("\033[92mNull hypothesis is rejected\033[0m")
                    causal_decision.append(1)
                else:
                     if pv_corr > 0.05:
                        print("\033[92mNull hypothesis is rejected\033[0m")
                        causal_decision.append(1)
                     else:
                        print("\033[94mFail to reject null hypothesis\033[0m")
                        causal_decision.append(0)

            pvi.append(pvals[0])
            pvo.append(pvals[1])
            pvm.append(pvals[2])
            pvu.append(pvals[3])

            kvi.append(kvals[0])
            kvo.append(kvals[1])
            kvm.append(kvals[2])
            kvu.append(kvals[3])

            # plot residuals distribution
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            sns.kdeplot(mapelol[0], color='red', label='Actual')
            sns.kdeplot(mapelolint[0], color='green', label='Counterfactual')

            if len(columns) > 0:
                # plt.ylabel(f"CSS: {columns[i]} ---> {columns[j]}")
                ax1.set_ylabel(f"{columns[i]} ---> {columns[j]}")
            else:
                # plt.ylabel(f"CSS: Z_{i + 1} ---> Z_{j + 1}")
                ax1.set_ylabel(f"Z_{i + 1} ---> Z_{j + 1}")

            plt.gcf()
            ax1.legend()
            filename = pathlib.Path(plot_path + f"{columns[i]} ---> {columns[j]}.pdf")
            plt.savefig(filename,  dpi=350)
            plt.close()

            indist_cause.append(causal_decision[0])
            outdist_cause.append(causal_decision[1])
            mean_cause.append(causal_decision[2])
            uni_cause.append(causal_decision[3])
            causal_decision = []

        pval_indist.append(pvi)
        pval_outdist.append(pvo)
        pval_mean.append(pvm)
        pval_uniform.append(pvu)

        kval_indist.append(kvi)
        kval_outdist.append(kvo)
        kval_mean.append(kvm)
        kval_uniform.append(kvu)
        
        conf_mat_indist = conf_mat_indist + indist_cause
        conf_mat_outdist = conf_mat_outdist + outdist_cause
        conf_mat_mean = conf_mat_mean + mean_cause
        conf_mat_uniform = conf_mat_uniform + uni_cause
        
        indist_cause, outdist_cause, mean_cause, uni_cause = [], [], [], []

    pvalues.append(pval_indist)
    pvalues.append(pval_outdist)
    pvalues.append(pval_mean)
    pvalues.append(pval_uniform)
    # print("P-Values: ", pvalues)

    kvalues.append(kval_indist)
    kvalues.append(kval_outdist)
    kvalues.append(kval_mean)
    kvalues.append(kval_uniform)
    # print("KL-Divergence: ", kvalues)

    conf_mat.append(conf_mat_indist)
    conf_mat.append(conf_mat_outdist)
    conf_mat.append(conf_mat_mean)
    conf_mat.append(conf_mat_uniform)

    true_conf_mat = pars.get("ground_truth")
    # --------------- f1-max ------------------
    pred = np.array(pval_indist)   #1 - np.array(kld_matrix)
    actual_lab = remove_diagonal_and_flatten(np.array(true_conf_mat))
    pred_score = remove_diagonal_and_flatten(pred)
    threshod, fmax = f1_max(actual_lab, pred_score)
    print(f'F1-Max: {fmax}')
    # -----------------------------------------
    # Reshape the list into a n x n array (causal matrix)
    causal_matrix = np.array(pval_indist).reshape(n, n)
    pred_conf_mat = np.array(conf_mat[0]).reshape(n, n)

    print(f'True: {true_conf_mat}')
    print(f'Predicted: {pred_conf_mat}')

     # Calculate metrics
    metrics = evaluate_predicted_graph(np.array(true_conf_mat), np.array(pred_conf_mat))

    # Apply condition to the covariance matrix
    causal_matrix_thresholded = np.where(np.abs(causal_matrix) < 0.10, 1, 0)
    print("-------------*****-----------------------*****-------------")
    # print(f'Discovered Causal Structure:\n {causal_matrix_thresholded}')
    # causal_heatmap(causal_matrix_thresholded, columns)
    print(f'Actual: {np.array(true_conf_mat)}')
    print(f'Predicted: {np.array(pred_conf_mat)}')
    evaluate(np.array(true_conf_mat).flatten().tolist(), conf_mat, intervention_methods)
    plot_causal_graph(causal_matrix_thresholded, columns, model_name)

    for metric, value in metrics.items():
       print(f"{metric}: {value:.2f}")

    return metrics, conf_mat, time.time()
