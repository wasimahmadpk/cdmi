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
import seaborn as sns
import preprocessing as prep
from itertools import islice
from datetime import datetime
import matplotlib.pyplot as plt
from knockoffs import Knockoffs
from scipy.special import stdtr
from forecast import modelTest
from sklearn.utils import shuffle
from gluonts.trainer import Trainer
from gluonts.evaluation import Evaluator
from sklearn.metrics import mean_squared_error
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.deepar._network import DeepARTrainingNetwork
from gluonts.evaluation.backtest import make_evaluation_predictions
from scipy.stats import ttest_ind, ttest_ind_from_stats, ttest_1samp, ks_2samp, kstest
from sklearn.feature_selection import f_regression, mutual_info_regression
from gluonts.distribution.multivariate_gaussian import MultivariateGaussianOutput
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

np.random.seed(1)
mx.random.seed(2)


pars = parameters.get_climate_params()
num_samples = pars.get("num_samples")
training_length = pars.get("train_len")
prediction_length = pars.get("pred_len")
frequency = pars.get("freq")
plot_path = pars.get("plot_path")
# Get prior skeleton
prior_graph = pars.get('prior_graph')
true_conf_mat = pars.get("true_graph")


def deepCause(odata, knockoffs, model, params):

    mutual_info = []
    for a in range(len(odata)):
            x = odata[:].T
            y = odata[a].T
            mi = prep.mutual_information(x, y)
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

    for i in range(len(odata)):

        int_var = odata[i]
        int_var_name = "Z_" + str(i + 1) + ""
        var_list = []
        causal_decision = []
        mean_cause = []
        indist_cause = []
        outdist_cause = []
        uni_cause = []

        # Generate Knockoffs
        data_actual = np.array(odata[:, 0: training_length + prediction_length]).transpose()
        obj = Knockoffs()
        n = len(odata[:, 0])
        knockoffs = obj.GenKnockoffs(n, params.get("dim"), data_actual)
        knockoff_sample = np.array(knockoffs[:, i])

        mean = np.random.normal(0, 0.05, len(knockoff_sample)) + np.mean(odata[i])
        outdist = np.random.normal(150, 120, len(knockoff_sample))
        # outdist = get_shuffled_ts(SAMPLE_RATE, DURATION, odata[i])
        uniform = np.random.uniform(np.min(odata[i]), np.min(odata[i]), len(knockoff_sample))
        interventionlist = [knockoff_sample, outdist[: len(knockoff_sample)], mean, uniform]
        heuristic_itn_types = ['In-dist', 'Out-dist', 'Mean', 'Uniform']

        for j in range(len(odata)):
            back_door_int = []
            back_door = prior_graph[:, j].nonzero()[0]
            print("----------*****-----------------------*****-----------------******-----------")
            print(f"Front/Backdoor Paths: {np.array(back_door) + 1} ---> {j + 1}")
            print("----------*****-----------------------*****-----------------******-----------")

            columns = params.get('col')
            pred_var = odata[j]
            pred_var_name = "Z_" + str(j + 1) + ""

            css_list = []
            css_list_new = []
            css_score_new = []
            mselol = []
            mapelol = []
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

                for iter in range(20):  # 30

                    mselist_batch = []
                    mselistint_batch = []
                    mapelist_batch = []
                    mapelistint_batch = []
                    for r in range(5):

                        test_data = odata[:, start: start + training_length + prediction_length].copy()
                        test_ds = ListDataset(
                            [
                                {'start': "01/01/1961 00:00:00",
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
                                {'start': "01/01/1961 00:00:00",
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

                        if (m == 0):
                            # Generate multiple version Knockoffs
                            data_actual = np.array(odata[:, start: start + training_length + prediction_length]).transpose()
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
                        # start = start + 96

                    start = start + 5 # Step size for sliding window # 10
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
                mselolint.append(mselistint)
                mapelolint.append(mapelistint)

            var_list.append(np.var(mapelolint[1]))
            # print(f"MSE(Mean): {list(np.mean(mselol, axis=0))}")
            if len(columns) > 0:

                print(f"Causal Link: {columns[i]} --------------> {columns[j]}")
                print("-----------------------------------------------------------------------------")
                fnamehist = plot_path + "{columns[i]}_{columns[j]}:hist"
            else:
                print(f"Causal Link: Z_{i + 1} --------------> Z_{j + 1}")
                print("-----------------------------------------------------------------------------")
                fnamehist = plot_path + "{Z_[i + 1]}_{Z_[j + 1]}:hist"

            for z in range(len(heuristic_itn_types)):

                print("Intervention: " + heuristic_itn_types[z])
                # print(f"Mean: {np.mean(mapelol[z])}, Mean Intervention: {np.mean(mapelolint[z])}")
                # print(f"Variance: {np.var(mapelol[z])}, Variance Intervention: {np.var(mapelolint[z])}")
                # t, p = ttest_ind(np.array(mapelolint[z]), np.array(mapelol[z]), equal_var=True)
                t, p = ks_2samp(np.array(mapelol[z]), np.array(mapelolint[z]))
                # t, p = kstest(np.array(mapelolint[z]), np.array(mapelol[z]))

                print(f'Test statistic: {t}, p-value: {p}')
                if p < 0.10 or mutual_info[i][j] > 0.90:
                    print("Null hypothesis is rejected")
                    causal_decision.append(1)
                else:
                    print("Fail to reject null hypothesis")
                    causal_decision.append(0)

            # plot residuals distribution
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            sns.distplot(mapelol[0], color='red', label='Actual')
            sns.distplot(mapelolint[0], color='green', label='Counterfactual')

            if len(columns) > 0:
                # plt.ylabel(f"CSS: {columns[i]} ---> {columns[j]}")
                ax1.set_ylabel(f"{columns[i]} ---> {columns[j]}")
            else:
                # plt.ylabel(f"CSS: Z_{i + 1} ---> Z_{j + 1}")
                ax1.set_ylabel(f"Z_{i + 1} ---> Z_{j + 1}")

            plt.gcf()
            ax1.legend()
            filename = pathlib.Path(plot_path + f"{columns[i]} ---> {columns[j]}.pdf")
            plt.savefig(filename)
            plt.show()
            # plt.close()

            mean_cause.append(causal_decision[0])
            indist_cause.append(causal_decision[1])
            outdist_cause.append(causal_decision[2])
            uni_cause.append(causal_decision[3])
            causal_decision = []

        conf_mat_mean = conf_mat_mean + mean_cause
        conf_mat_indist = conf_mat_indist + indist_cause
        conf_mat_outdist = conf_mat_outdist + outdist_cause
        conf_mat_uniform = conf_mat_uniform + uni_cause
        mean_cause, indist_cause, outdist_cause, uni_cause = [], [], [], []

    conf_mat.append(conf_mat_mean)
    conf_mat.append(conf_mat_indist)
    conf_mat.append(conf_mat_outdist)
    conf_mat.append(conf_mat_uniform)
    print("-----------------------------------------------------------------------------")
    print("Discovered Causal Graphs: ", conf_mat)

    for ss in range(len(conf_mat)):

        fscore = round(f1_score(true_conf_mat, conf_mat[ss], average='binary'), 2)
        acc = accuracy_score(true_conf_mat, conf_mat[ss])
        tn, fp, fn, tp = confusion_matrix(true_conf_mat, conf_mat[ss], labels=[0, 1]).ravel()
        precision = precision_score(true_conf_mat, conf_mat[ss])
        recall = recall_score(true_conf_mat, conf_mat[ss])
        
        print("---------***-----------***----------***----------")
        print(f"Intervention: {heuristic_itn_types[ss]}")
        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"Accuracy: {acc}")
        print(f"F-score: {fscore}")
        print("---------***-----------***----------***----------")