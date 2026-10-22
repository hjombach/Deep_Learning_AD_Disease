from datetime import datetime
import errno
from functools import partial
import json
import os
import pickle
import random
import shutil
import time
import warnings

import deepchem as dc
from deepchem import models
from deepchem.models import GraphConvModel
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from openpyxl import Workbook
import pandas as pd
from pytz import timezone
from sklearn.metrics import log_loss
import torch

import data_preparation
import model_evaluation

warnings.filterwarnings('ignore')


def define_parameters(target: str) -> dict:
    """
    Based on the specified target, load the optimal architecture.

    Args:
        target (str): The target of the model. Must be either 'tau' or 'app'.
    
    Returns:
        dict - dictionary containing the model parameters for the specified target
            - 'graph_conv_layers' (list): List of integers specifying the number of graph convolution layers
            - 'batch_normalize' (bool): Whether to use batch normalization
            - 'dropout' (float): Dropout rate
            - 'dense_layer_size' (list): List of integers specifying the number of neurons in each dense layer
            - 'batch_size' (int): Batch size
            - 'probability_threshold' (float): Threshold for classifying compounds as active

    Raises:
        ValueError: If the target is not 'tau' or 'app'
    """
    with open('/home/paperspace/Desktop/DL_Project/data/final_model_params.json', 'r') as f:
        model_params=json.load(f)

    target_specific_params=model_params[target]

    return target_specific_params


def load_model_weights(target: str) -> str:
    """
    Based on the target, return the file path for the pre-trained model weights

    Args:
        target (str): The target of the model. Must be either 'tau' or 'app'.

    Returns:
        str - path to the appropriate model weights file
    """
    base_path="/home/paperspace/Desktop/DL_Project/py_files/"

    if target == 'tau':
        # epoch 130
        # file_path="tau_interim_weights/epoch_130_checkpoint/ckpt-309.index"
        file_path="tau_interim_weights/epoch_130_checkpoint/ckpt-305.index"
    elif target=='app':
        # epoch 150
        # file_path="app_interim_weights/epoch_150_checkpoint/ckpt-262.index"
        file_path="app_interim_weights/epoch_140_checkpoint/ckpt-261.index"

    return base_path + file_path

def load_prediction_set() -> pd.DataFrame:
    """
    Load the prediction set from the pickle file.

    Returns:
        pd.DataFrame - DataFrame containing the prediction set

    Raises:
        FileNotFoundError: If the pickle file is not found
    """
    base_path="/home/paperspace/Desktop/DL_Project/data/"
    df_file="full_prediction_set_fp_graphs.pickle"
    pred_set=pd.read_pickle(base_path+df_file)
    
    return pred_set


def run_model(pred_set_df: pd.DataFrame, targets=['app', 'tau']):
    """
    Run the model on the prediction set.

    Args:
        pred_set_df (pd.DataFrame): DataFrame containing the prediction set
        targets (list): List of targets to run the model on

    Returns:
        None

    Raises:
        ValueError: If the targets are not 'app' or 'tau'
    """
    base_path="/home/paperspace/Desktop/DL_Project/py_files/"
    # with the chosen parameters, create models for both tasks
    for target in targets:

        parameters=define_parameters(target)
        # threshold for active classification
        probability_threshold = parameters['probability_threshold']
        task = f'{target}_inhibition'

        final_model = dc.models.GraphConvModel(n_tasks=1, mode='classification',
                                graph_conv_layers=parameters["graph_conv_layers"],
                                batch_normalize=parameters['batch_normalize'],
                                dropout=parameters['dropout'],
                                dense_layer_size=parameters['dense_layer_size'],
                                batch_size=parameters['batch_size'],
                                model_dir='model_model')

        # load best weights
        checkpoint_file=load_model_weights(target)
        final_model.restore(checkpoint_file)
        # make Predictions
        pred_set=data_preparation.prepare_prediction_set(pred_set_df)
        predictions = final_model.predict(pred_set)
        # Convert to Binary Label
        binary_preds = np.array([1 if x[0][1] >= probability_threshold else 0 for x in predictions]) # 1 = 'Active

        pred_set_df['binary_pred']=binary_preds
        pred_set_df['active_probability']=[x[0][1] for x in predictions]

        filtered=pred_set_df.drop(columns=['morgan_fp', 'uncharged_mol', 'molecule', 'dc_graph'])
        filtered.to_excel(f"{base_path}{target}_predictions.xlsx", index=False)


if __name__=='__main__':
    """
    Load the drugbank and chembl combined prediction set. Make predictions.
    """
    # df = data_preparation.load_data(testing=False)
    prediction_set=load_prediction_set()
    print(prediction_set.head())

    run_model(prediction_set, targets=["tau", "app"])