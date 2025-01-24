import pickle
import os
import time
import pathlib
import parameters
import numpy as np
import mxnet as mx
import pandas as pd
import functions as func
import dataloader as datasets
import matplotlib.pyplot as plt
from regimes import get_regimes
from deepcause import deepCause
from gluonts.trainer import Trainer
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.distribution.multivariate_gaussian import MultivariateGaussianOutput

np.random.seed(1)
mx.random.seed(2)

def causal_graph(input, pars):

    start_time = time.time()
    # Default parameters dictionary
    default_params = {
        'epochs': 50,
        'pred_len': 12,
        'train_len': 500,
        'num_layers': 4,
        'num_samples': 40,
        'num_cells': 20,
        'dropout_rate': 0.1,
        'win_size': 1,
        'step_size': 3,
        'num_sliding_win': 20,
        'sliding_win_size': 100,
        'dim': 5,
        'batch_size': 32,
        'freq': '30min',
        'plot_path': "/home/ahmad/Projects/cdmi/plots/",
        'model_path': "/home/ahmad/Projects/cdmi/models/",
        'model_name': 'trained_model_syntest.sav'
    }
    
    # Use default parameters if params is None
    if pars is None:
        pars = default_params
        print("Default parameters are used. Please provide task-specific parameters for customization.")
    else:
        # print("Task-specific parameters are provided.")
        pass

    freq = pars.get("freq")
    epochs = pars.get("epochs")
    win_size = pars.get("win_size")
    slidingwin_size = pars.get("slidingwin_size")
    training_length = pars.get("train_len")
    prediction_length = pars.get("pred_len")
    num_samples = pars.get("num_samples")
    num_layers = pars.get("num_layers")
    num_cells = pars.get("num_cells")
    dropout_rate = pars.get("dropout_rate")
    batch_size = pars.get("batch_size")
    plot_path = pars.get("plot_path")
    model_name = pars.get("model_name")
    print(f'COnfMat: {len(np.array(pars.get("true_graph")).shape)}')
    
    
    # Your function logic using data and params


    if isinstance(input, str):
        if input.endswith('.csv'):
            try:
                df = pd.read_csv(input)
                return df
            except FileNotFoundError:
                print(f"Error: File '{input}' not found.")
                return None
        else:
            print("Error: Input is not a path to a CSV file.")
            return None
    elif isinstance(input, pd.DataFrame):
        # Input data is already a DataFrame
        df = input
    else:
        print("Error: Input must be a path to a CSV file or a pandas DataFrame.")

    original_data = []
    dim, columns = len(df.columns), df.columns

    for col in df:
        original_data.append(df[col])

    original_data = np.array(original_data)
    # training set
    train_ds = ListDataset(
        [
            {'start': df.index[0],
            'target': original_data[:, 0: training_length].tolist()
            }
        ],
        freq=freq,
        one_dim_target=False
    )

    # create estimator
    estimator = DeepAREstimator(
        prediction_length=prediction_length,
        context_length=prediction_length,
        freq=freq,
        num_layers=num_layers,
        num_cells=num_cells,
        dropout_rate=dropout_rate,
        trainer=Trainer(
            ctx="cpu",
            epochs=epochs,
            hybridize=False,
            learning_rate=1E-4,
            batch_size=32
        ),
        distr_output=MultivariateGaussianOutput(dim=dim)
    )

    # load model if not already trained
    path = pars.get('model_path')
    model_path = pathlib.Path(path + model_name)
   
    filename = pathlib.Path(model_path)
    if not filename.exists():
        print("Training forecasting model....")
        predictor = estimator.train(train_ds)
        # save the model to disk
        pickle.dump(predictor, open(filename, 'wb'))

    # # Generate Knockoffs
    # data_actual = np.array(original_data[:, :]).transpose()
    # n = len(original_data[:, 0])
    # obj = Knockoffs()
    pars.update({"dim": dim, "col": columns})
    # knockoffs = obj.GenKnockoffs(data_actual, params)

    # Function for estimating causal impact among variables
    causal_matrix_thresholded, predicted_graph, fmax, end_time = deepCause(original_data, model_path, pars)

    # Calculate difference
    elapsed_time = end_time - start_time
    # Print elapsed time
    print("Computation time:", round(elapsed_time/60), "mins")

    return  causal_matrix_thresholded, predicted_graph, fmax, end_time