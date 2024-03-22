import numpy as np
import pandas as pd


def get_sig_params():
    pars = dict()
    pars["sample_rate"] = 44100  # Hertz
    pars["duration"] = 5   # seconds
    return pars


def GetDistributionParams(model, p):
    """
    Returns parameters for generating different data distributions
    """
    params = dict()
    params["model"] = model
    params["p"] = p
    if model == "gaussian":
        params["rho"] = 0.5
    elif model == "gmm":
        params["rho-list"] = [0.3,0.5,0.7]
    elif model == "mstudent":
        params["df"] = 3
        params["rho"] = 0.5
    elif model == "sparse":
        params["sparsity"] = int(0.3*p)
    else:
        raise Exception('Unknown model generating distribution: ' + model)
    
    return params
        

def GetTrainingHyperParams(model):
    """
    Returns the default hyperparameters for training deep knockoffs
    as described in the paper
    """
    params = dict()
    
    params['GAMMA'] = 1.0
    if model == "gaussian":
        params['LAMBDA'] = 1.0
        params['DELTA'] = 1.0
    elif model == "gmm":
        params['LAMBDA'] = 1.0
        params['DELTA'] = 1.0
    elif model == "mstudent":
        params['LAMBDA'] = 0.01
        params['DELTA'] = 0.01
    elif model == "sparse":
        params['LAMBDA'] = 0.1
        params['DELTA'] = 1.0
    else:
        raise Exception('Unknown data distribution: ' + model)
        
    return params


def GetFDRTestParams(model):
    """
    Returns the default hyperparameters for performing controlled
    variable selection experiments as described in the paper
    """
    params = dict()
    # Test parameters for each model
    if model in ["gaussian", "gmm"]:
        params["n"] = 150
        params["elasticnet_alpha"] = 0.1
    elif model in ["mstudent"]:
        params["n"] = 200
        params["elasticnet_alpha"] = 0.0
    elif model in ["sparse"]:
        params["n"] = 200
        params["elasticnet_alpha"] = 0.0
    
    return params


def get_syn_params():
    # Parameters for synthetic data
    params = {

        'epochs': 100,
        'pred_len': 14,
        'train_len': 555,
        'num_layers': 4,
        'num_cells': 44,
        'num_samples': 10,
        'dropout_rate': 0.1,
        'win_size': 1,
        'slidingwin_size': 100,
        'step_size': 10,
        'dim': 5,
        'batch_size': 32,
        'prior_graph': np.array([[1, 1, 1, 1],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 1],
                                 [0, 0, 0, 1],
                                 ]),
        'true_graph': [1, 1, 1, 1,
                       0, 1, 0, 0,
                       0, 0, 1, 1,
                       0, 0, 0, 1,
                       ],
        'freq': '30min',
        'plot_path': "/home/ahmad/Projects/deepCausality/plots/"
    }

    return params


def get_real_params():

    params = {
        'epochs': 150,
        'pred_len': 28,
        'train_len': 555,
        'num_layers': 3,
        'num_samples': 10,
        'num_cells': 33,
        'dropout_rate': 0.1,
        'win_size': 1,
        'dim': 3,
        'batch_size': 32,
        'prior_graph': np.array([[1, 1, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]]),
        'true_graph': [1, 1, 0,
                       0, 1, 0,
                       0, 0, 1],
        'freq': 'D',
        'plot_path': "/home/ahmad/Projects/deepCausality/plots/"
    }
    return params


def get_climate_params():

    params = {
        'epochs': 100,
        'pred_len': 48,
        'train_len': 375,
        'num_layers': 3,
        'num_samples': 10,
        'num_cells': 30,
        'dropout_rate': 0.1,
        'win_size': 1,
        'dim': 2,
        'batch_size': 24,
        'prior_graph': np.array([[1, 0],
                                 [1, 1]]),
        'true_graph': [1, 0,
                       1, 1],
        'freq': 'H',
        'plot_path': "/home/ahmad/Projects/deepCausality/plots/"
    }
    return params


def get_geo_params():

    params = {
        'epochs': 100,    # 125
        'pred_len': 15,    # 15
        'train_len': 1000,  # 1500
        'num_layers': 4,    # 5
        'num_samples': 10,
        'num_cells': 40,     # 50
        'dropout_rate': 0.1,
        'win_size': 1,
        'slidingwin_size': 100,
        'dim': 4,
        'batch_size': 32,
        'prior_graph': np.array([[1, 0, 0, 0, 1],
                                 [0, 1, 0, 0, 1],
                                 [0, 0, 1, 0, 1],
                                 [0, 0, 0, 1, 1],
                                 [0, 0, 0, 0, 1]]),
        'true_graph': [1, 0, 0, 0, 1,
                       0, 1, 0, 0, 1,
                       0, 0, 1, 0, 1,
                       0, 0, 0, 1, 1,
                       0, 0, 0, 0, 1],
        'freq': 'H',
        'plot_path': "/home/ahmad/Projects/deepCausality/plots/"
    }
    return params


def get_hack_params():

    params = {
        'epochs': 100,
        'pred_len': 24,
        'train_len': 21*24,
        'num_layers': 6,
        'num_samples': 10,
        'num_cells': 50,
        'dropout_rate': 0.1,
        'win_size': 1,
        'dim': 6,
        'batch_size': 32,
        'prior_graph': np.array([[1, 0, 1, 0, 1, 1],
                                 [1, 0, 1, 0, 0, 1],
                                 [1, 1, 0, 1, 1, 0],
                                 [1, 0, 1, 0, 1, 1],
                                 [1, 0, 1, 0, 1, 1],
                                 [1, 0, 1, 0, 1, 1]]),
        'true_graph': [1, 0, 1, 0, 1, 1,
                       1, 0, 1, 0, 1, 1,
                       1, 1, 0, 1, 1, 0,
                       1, 0, 1, 0, 1, 0,
                       1, 0, 1, 0, 1, 1,
                       1, 0, 1, 0, 1, 1],
        'freq': 'D',
        'plot_path': "/home/ahmad/Projects/deepCausality/plots/"
    }
    return params



