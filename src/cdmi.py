import pickle
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
from deep_causal_discovery import execute_causal_pipeline
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
    training_length = pars.get("train_len")
    prediction_length = pars.get("pred_len")
    num_layers = pars.get("num_layers")
    num_cells = pars.get("num_cells")
    dropout_rate = pars.get("dropout_rate")
    model_name = pars.get("model_name")

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

    dim, columns = len(df.columns), df.columns
    # training set
    train_ds = ListDataset(
        [
            {'start': df.index[0],
             'target': df.iloc[:training_length].values.T.tolist()
            }
        ],
        freq=freq,
        one_dim_target=False
    )

    # create estimator
    estimator = DeepAREstimator(
        prediction_length=prediction_length,
        context_length=2*prediction_length,
        freq=freq,
        num_layers=num_layers,
        num_cells=num_cells,
        dropout_rate=dropout_rate,
        trainer=Trainer(
            ctx="cpu",
            epochs=epochs,
            hybridize=False,
            learning_rate=1E-3,
            batch_size=48
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

    pars.update({"dim": dim, "col": columns})
    # Function for estimating causal impact among variables
    # metrics, predicted_graph, end_time = deepCause(df, model_path, pars)
    metrics, predicted_graph, end_time = execute_causal_pipeline(df, model_path, pars)

    # Calculate difference
    elapsed_time = end_time - start_time
    # Print elapsed time
    print("Computation time:", round(elapsed_time/60), "mins")

    return  metrics, predicted_graph, end_time