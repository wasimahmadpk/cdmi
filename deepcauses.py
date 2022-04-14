import math
import netCDF
import pickle
import random
import pathlib
import parameters
import numpy as np
import mxnet as mx
import pandas as pd
from os import path
from math import sqrt
from netCDF4 import Dataset
from scipy.fft import irfft
from itertools import islice
from datetime import datetime
import matplotlib.pyplot as plt
from knockoffs import Knockoffs
from scipy.special import stdtr
from model_test import modelTest
from sklearn.utils import shuffle
from gluonts.trainer import Trainer
from gluonts.evaluation import Evaluator
from scipy.fftpack import fft, irfft, fftfreq, rfft, rfftfreq
from sklearn.metrics import mean_squared_error
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.deepar._network import DeepARTrainingNetwork
from gluonts.evaluation.backtest import make_evaluation_predictions
from scipy.stats import ttest_ind, ttest_ind_from_stats, ttest_1samp
from sklearn.feature_selection import f_regression, mutual_info_regression
from gluonts.distribution.multivariate_gaussian import MultivariateGaussianOutput
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

np.random.seed(1)
mx.random.seed(2)


def mutual_information(x, y):
    mi = mutual_info_regression(x, y)
    mi /= np.max(mi)
    return mi


def get_shuffled_ts(SAMPLE_RATE, DURATION, root):
    # Number of samples in normalized_tone
    N = SAMPLE_RATE * DURATION
    yf = rfft(root)
    xf = rfftfreq(N, 1 / SAMPLE_RATE)
    # plt.plot(xf, np.abs(yf))
    # plt.show()
    new_ts = irfft(shuffle(yf))
    return new_ts


def running_avg_effect(y, yint):

    rae = 0
    for i in range(len(y)):
        ace = 1/((params.get("train_len") + 1 + i) - params.get("train_len")) * (rae + (y[i] - yint[i]))
    return rae


def deepCause(odata, knockoffs, model, params):

    mutual_info = []
    for a in range(len(odata)):
            x = odata[:].T
            y = odata[a].T
            mi = mutual_information(x, y)
            print("MI Value: ", mi)
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

    # Get prior skeleton
    prior_graph = params.get('prior_graph')
    print("Length of Original data:", len(odata))
    for i in range(len(odata)):

        int_var = odata[i]
        int_var_name = "Z_" + str(i + 1) + ""
        causal_decision = []
        mean_cause = []
        var_list = []
        indist_cause = []
        outdist_cause = []

        # Break temporal dependency and generate a new time series
        pars = parameters.get_sig_params()
        SAMPLE_RATE = pars.get("sample_rate")  # Hertz
        DURATION = pars.get("duration")  # Seconds

        # Generate Knockoffs
        data_actual = np.array(odata[:, 0: params.get('train_len') + params.get('pred_len')]).transpose()
        obj = Knockoffs()
        n = len(odata[:, 0])
        knockoffs = obj.GenKnockoffs(n, params.get("dim"), data_actual)
        knockoff_sample = np.array(knockoffs[:, i])

        mean = np.random.normal(0, 0.05, len(knockoff_sample)) + np.mean(odata[i])
        outdist = np.random.normal(10, 10, len(knockoff_sample))
        # outdist = get_shuffled_ts(SAMPLE_RATE, DURATION, odata[i])
        uniform = np.random.uniform(np.min(odata[i]), np.min(odata[i]), len(knockoff_sample))
        interventionlist = [knockoff_sample, outdist[: len(knockoff_sample)], mean, uniform]
        heuristic_itn_types = ['In-dist', 'Out-dist', 'Mean', 'Uniform']

        # Show variable with its knockoff copy
        # plt.plot(np.arange(0, len(counterfactuals)), target[: len(counterfactuals)], counterfactuals)
        # plt.show()

        # Check correlation of knockoff samples with its original variable
        # corr = np.corrcoef(knockoff_sample, odata[j][0: len(knockoff_sample)])
        # print(f"Correlation Coefficient (Variable, Counterfactual): {corr}")

        for j in range(len(odata)):
            back_door_int = []
            back_door = prior_graph[:, j].nonzero()[0]
            print(f"Front/Backdoor Paths: {np.array(back_door) + 1} ---> {j + 1}")
            for g in range(len(back_door)):
                # back_door_int.append(np.array(knockoffs[:, g]))
                # back_door_int.append(get_shuffled_ts(SAMPLE_RATE, DURATION, odata[g]))
                back_door_int.append(np.random.normal(10, 10, len(knockoff_sample)))

            columns = params.get('col')
            pred_var = odata[j]
            pred_var_name = "Z_" + str(j + 1) + ""

            css_list = []
            css_list_new = []
            css_score_new = []
            mselol = []
            mapelol = []
            acelol = []
            mselolint = []
            mapelolint = []

            for m in range(len(interventionlist)):  # apply all types of intervention methods

                intervene = interventionlist[m]

                mselist = []  # list of MSE values for multiple realization without intervention
                mselistint = []  # list of MSE values for multiple realization with intervention
                acelist = []
                mapelist = []  # list of MAPE values for multiple realization without intervention
                mapelistint = []  # list of MAPE values for multiple realization with intervention
                css_score = []  # list of causal scores for multiple realization
                diff = []
                start = 10

                for iter in range(25):

                    mselist_batch = []
                    mselistint_batch = []
                    mapelist_batch = []
                    mapelistint_batch = []
                    for r in range(4):

                        test_data = odata[:, start: start + params.get('train_len') + params.get('pred_len')].copy()

                        # for q in range(len(back_door)):
                        #     if back_door[q] != j or back_door[q] != i:
                        #         test_data[q, :] = back_door_int[q][0:550]

                        test_ds = ListDataset(
                            [
                                {'start': "01/01/1961 00:00:00",
                                 'target': test_data
                                 }
                            ],
                            freq=params.get('freq'),
                            one_dim_target=False
                        )
                        # rg[0:-50] + list(intervene[-50:])
                        int_data = odata[:, start: start + params.get('train_len') + params.get('pred_len')].copy()
                        # for v in range(len(back_door)):
                        #     if back_door[v] != j:
                        # int_data[v, :] = back_door_int[v][0:550]
                        int_data[i, :] = intervene

                        test_dsint = ListDataset(
                            [
                                {'start': "01/01/1961 00:00:00",
                                 'target': int_data
                                 }
                            ],
                            freq=params.get('freq'),
                            one_dim_target=False
                        )

                        mse, mape, ypred = modelTest(model, test_ds, params.get('num_samples'), test_data[j], j,
                                                     params.get('pred_len'), iter, False, 0)

                        mseint, mapeint, ypredint = modelTest(model, test_dsint, params.get('num_samples'),
                                                              test_data[j], j,
                                                              params.get('pred_len'), iter, True, m)

                        # Visualize impact of intervention on target variable
                        # plt.plot(np.arange(100, 100 + params.get("pred_len")), ypred,  '-g', label="Prediction")
                        # plt.plot(np.arange(100, 100 + params.get("pred_len")), ypredint, '-r', label="Counterfactual")
                        # plt.plot(test_data[j, -148:], '--b', label="Actual")

                        # # x coordinates for the lines
                        # xcoords = [100]
                        # # colors for the lines
                        # colors = ['black']
                        #
                        # for xc, c in zip(xcoords, colors):
                        #     plt.axvline(x=xc, label="Intervention", ls='--', c=c,)
                        #
                        # plt.legend()
                        # plt.show()

                        if (m == 0):
                            # Generate multiple version Knockoffs
                            data_actual = np.array(odata[:, start: start + params.get("train_len") + params.get(
                                "pred_len")]).transpose()
                            obj = Knockoffs()
                            n = len(odata[:, 0])
                            knockoffs = obj.GenKnockoffs(n, params.get("dim"), data_actual)
                            knockoff_sample = np.array(knockoffs[:, i])
                            intervene = knockoff_sample

                        np.random.shuffle(intervene)

                        mselist_batch.append(mse)
                        mapelist_batch.append(mape)
                        mselistint_batch.append(mseint)
                        mapelistint_batch.append(mapeint)
                        # step = step + 96

                    start = start + 25  # Step size for sliding window
                    mselist.append(np.mean(mselist_batch))  # mselist = mselist_batch
                    mapelist.append(np.mean(mapelist_batch))  # mapelist = mapelist_batch
                    mselistint.append(np.mean(mselistint_batch))  # mselistint = mselistint_batch
                    mapelistint.append(np.mean(mapelistint_batch))  # mapelistint = mapelistint_batch

                # mselist = np.mean(mselist, axis=0)
                # mapelist = np.mean(mapelist, axis=0)
                # mselistint = np.mean(mselistint, axis=0)
                # mapelistint = np.mean(mapelistint, axis=0)

                msevar = np.var(mselist)
                mapevar = np.var(mapelist)
                mselol.append(mselist)
                mapelol.append(mapelist)
                # acelol.append(acelist)
                # print(f"MSE: {mselist}, MAPE: {mape}%")
                # print(f"ACE: {acelist}")

                mselolint.append(mselistint)
                mapelolint.append(mapelistint)
                # print(f"MSE: {mselistint}, MAPE: {mape}%")
                # avg_diff = np.mean(diff, axis=0)
                # plt.plot(avg_diff)
                # plt.show()

                for k in range(len(mselist)):
                    # Calculate causal significan score (CSS)
                    css_score.append(np.log(mapelistint[k] / mapelist[k]))
                    # css_score.append(np.log(mselistint[k] / mselist[k]))
                    # css_score.append(mapelistint[k] - mapelist[k])
                    # css_score.append(mapelistint[k] / mapelist[k])

                # Absolute casual score
                # css_score = [abs(x) if x < 0 else x for x in css_score]
                print("-----------------------------------------------------------------------------")
                css_list.append(css_score)
                print("CSS: ", css_score)
                print("-----------------------------------------------------------------------------")

            var_list.append(np.var(mapelolint[1]))
            # print(f"MSE(Mean): {list(np.mean(mselol, axis=0))}")
            if len(columns) > 0:

                print(f"Time series: {columns[i]} --------------> {columns[j]}")
                print("-----------------------------------------------------------------------------")
                fnamehist = f"/home/ahmad/PycharmProjects/deepCausality/plots/{columns[i]}_{columns[j]}:hist"
            else:
                print(f"Time series: Z_{i + 1} --------------> Z_{j + 1}")
                print("-----------------------------------------------------------------------------")
                fnamehist = f"/home/ahmad/PycharmProjects/deepCausality/plots/{Z_[i + 1]}_{Z_[j + 1]}:hist"

            for z in range(len(heuristic_itn_types)):

                # months = ['7 Aug', '14 Aug', '21 Aug', '28 Aug', '7 Sep', '14 Sep', '21 Sep', '28 Sep']
                plt.plot(css_list[z])
                plt.xlabel("Vegetation season")

                if len(columns) > 0:
                    plt.ylabel(f"CSS: {columns[i]} ---> {columns[j]}")
                    fnamecss = f"/home/ahmad/PycharmProjects/deepCausality/plots/{columns[i]}_{columns[j]}:css"
                else:
                    plt.ylabel(f"CSS: Z_{i + 1} ---> Z_{j + 1}")
                    fnamecss = f"/home/ahmad/PycharmProjects/deepCausality/plots/{Z_[i + 1]}_{Z_[j + 1]}:css"

                # Hypothesis testing for causal decision
                print(
                    f"Average Causal Strength using {heuristic_itn_types[z]} Intervention: {np.mean(css_list[z])}")
                # print(f"Average Causal Strength using {heuristic_itn_types[z]} Intervention: {np.mean(np.array(mselolint[z]) - np.array(mselol[z]))}")
                # print("CSS: ", css_score)
                t, p = ttest_ind(np.array(mapelolint[z]), np.array(mapelol[z]), equal_var=True)
                # t, p = ttest_1samp(css_list[z], popmean=0.0, alternative="greater")  #
                # plt.hist(mselolint[z])
                # plt.hist(mselol[z])
                # plt.show()
                print(f'Test statistic: {t}, p-value: {p}')
                if p < 0.05 or mutual_info[i][j] > 0.33:
                    print("Null hypothesis is rejected")
                    causal_decision.append(1)
                else:
                    print("Fail to reject null hypothesis")
                    causal_decision.append(0)

            plt.plot(np.arange(1, 22), np.zeros(21), 'b--')
            plt.legend(heuristic_itn_types)
            plt.gcf()
            # plt.show()
            plt.savefig(fnamecss, dpi=100, facecolor='w', edgecolor='w',
                        orientation='portrait', papertype=None, format=None,
                        transparent=False, bbox_inches=None, pad_inches=0.1,
                        frameon=None, metadata=None)
            plt.close()

            plt.hist(css_list[0], bins=5)
            # plt.hist(mapelol[1], bins=7, color='red')
            # plt.hist(mapelolint[1], bins=7, color='green')

            if len(columns) > 0:
                plt.ylabel(f"CSS: {columns[i]} ---> {columns[j]}")
            else:
                plt.ylabel(f"CSS: Z_{i + 1} ---> Z_{j + 1}")

            plt.gcf()
            # plt.show()
            plt.savefig(fnamehist, dpi=100, facecolor='w', edgecolor='w',
                        orientation='portrait', papertype=None, format=None,
                        transparent=False, bbox_inches=None, pad_inches=0.1,
                        frameon=None, metadata=None)
            plt.close()

            mean_cause.append(causal_decision[0])
            indist_cause.append(causal_decision[1])
            outdist_cause.append(causal_decision[2])
            causal_decision = []
            print("-------------******----------------*******-------------*******--------------")
            print("Variances:", var_list)
            print("-------------******--------------*******-------------*******----------------")


        conf_mat_mean = conf_mat_mean + mean_cause
        conf_mat_indist = conf_mat_indist + indist_cause
        conf_mat_outdist = conf_mat_outdist + outdist_cause
        mean_cause, indist_cause, outdist_cause = [], [], []

    conf_mat.append(conf_mat_mean)
    conf_mat.append(conf_mat_indist)
    conf_mat.append(conf_mat_outdist)

    print("Confusion Matrix:", conf_mat)
    true_conf_mat = [1, 1, 1, 1, 1,    0, 1, 0, 0, 1,   0, 0, 1, 0, 1,   0, 0, 0, 1, 1,    0, 0, 0, 0, 1]

    for ss in range(len(conf_mat)):

        fscore = round(f1_score(true_conf_mat, conf_mat[ss], average='binary'), 2)
        acc = accuracy_score(true_conf_mat, conf_mat[ss])
        tn, fp, fn, tp = confusion_matrix(true_conf_mat, conf_mat[ss], labels=[0, 1]).ravel()
        precision = precision_score(true_conf_mat, conf_mat[ss])
        recall = recall_score(true_conf_mat, conf_mat[ss])
        
        print("---------***-----------***----------***----------")
        print(f"Intervention: {heuristic_itn_types[ss]}")
        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")  
        print(f"Accuracy: {acc}")
        print(f"F-score: {fscore}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print("---------***-----------***----------***----------")