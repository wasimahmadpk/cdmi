import math
import h5py
import pathlib
import parameters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score


np.random.seed(1)
pars = parameters.get_syn_params()

win_size = pars.get("win_size")
training_length = pars.get("train_len")
prediction_length = pars.get("pred_len")


def get_shuffled_ts(SAMPLE_RATE, DURATION, root):
    # Number of samples in normalized_tone
    N = SAMPLE_RATE * DURATION
    yf = rfft(root)
    xf = rfftfreq(N, 1 / SAMPLE_RATE)
    # plt.plot(xf, np.abs(yf))
    # plt.show()
    new_ts = irfft(shuffle(yf))
    return new_ts


def deseasonalize(var, interval):
    deseasonalize_data = []
    for i in range(interval, len(var)):
        value = var[i] - var[i - interval]
        deseasonalize_data.append(value)
    return deseasonalize_data


def running_avg_effect(y, yint):

#  Break temporal dependency and generate a new time series
    pars = parameters.get_sig_params()
    SAMPLE_RATE = pars.get("sample_rate")  # Hertz
    DURATION = pars.get("duration")  # Seconds
    rae = 0
    for i in range(len(y)):
        ace = 1/((training_length + 1 + i) - training_length) * (rae + (y[i] - yint[i]))
    return rae


# Normalization (MixMax/ Standard)
def normalize(data, type='minmax'):

    if type == 'std':
        return (np.array(data) - np.mean(data))/np.std(data)

    elif type == 'minmax':
        return (np.array(data) - np.min(data))/(np.max(data) - np.min(data))


def down_sample(data, win_size, partition=None):
    agg_data = []
    daily_data = []
    for i in range(len(data)):
        daily_data.append(data[i])

        if (i % win_size) == 0:

            if partition == None:
                agg_data.append(sum(daily_data) / win_size)
                daily_data = []
            elif partition == 'gpp':
                agg_data.append(sum(daily_data[24: 30]) / 6)
                daily_data = []
            elif partition == 'reco':
                agg_data.append(sum(daily_data[40: 48]) / 8)
                daily_data = []
    return agg_data


def SNR(s, n):
    Ps = np.sqrt(np.mean(np.array(s) ** 2))
    Pn = np.sqrt(np.mean(np.array(n) ** 2))
    SNR = Ps / Pn
    return 10 * math.log(SNR, 10)


def mean_absolute_percentage_error(y_true, y_pred):

    return np.mean(np.abs((y_true - y_pred) / y_true))


def mutual_information(x, y):
    mi = mutual_info_regression(x, y)
    mi /= np.max(mi)
    return mi


def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def corr_heatmap(df):

    # Assuming you have a DataFrame named df
    # df = pd.DataFrame(...)

    # Compute the correlation matrix
    corr_matrix = df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, center=0, square=True, linewidths=0.5)
    print(f'------------------------------ Correlation Matrix ---------------------------- \n {corr_matrix}')
    # Add column names as xticklabels
    plt.xticks(ticks=np.arange(0.5, len(df.columns)), labels=df.columns, rotation=25, ha='right')

    # Add row names as yticklabels
    plt.yticks(ticks=np.arange(0.5, len(df.columns)), labels=df.columns, rotation=0)

    # Add title
    plt.title('Correlation Heatmap')

    # Show the plot
    # plt.show()

def causal_heatmap(cmatrix, columns):

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(cmatrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, center=0, square=True, linewidths=0.5)
    # Add column names as xticklabels
    plt.xticks(ticks=np.arange(0.5, len(columns)), labels=columns, rotation=25, ha='right')

    # Add row names as yticklabels
    plt.yticks(ticks=np.arange(0.5, len(columns)), labels=columns, rotation=0)

    # Add title
    plt.title('Discovered Causal Structure')
    plot_path = r"../deepCausality/plots/"
    filename = pathlib.Path(plot_path + f"causal_matrix.pdf")
    plt.savefig(filename)

    # Show the plot
    plt.show()

def evaluate(true_conf_mat, conf_mat):
    print('Called from Fun(c)!')

    for ss in range(len(conf_mat)):

        # true_conf_mat = conf_mat[ss]
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