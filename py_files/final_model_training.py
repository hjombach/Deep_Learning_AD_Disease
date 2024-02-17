import data_preparation
import model_evaluation
import model_development
import deepchem as dc
from deepchem import models
from deepchem.models import GraphConvModel
import matplotlib.pyplot as plt
from matplotlib import ticker
from openpyxl import Workbook
from functools import partial
import os
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import torch
import time
from pytz import timezone
import random
from sklearn.metrics import log_loss
import json
import warnings
warnings.filterwarnings('ignore')
from keras.callbacks import ModelCheckpoint
import shutil
import errno

def dc_metrics() -> list:
    '''
    Deepchem classifier metrics to log throughout training
    '''
    # metrics to log throughout training
    m1 = dc.metrics.Metric(dc.metrics.roc_auc_score)
    m2 = dc.metrics.Metric(dc.metrics.accuracy_score)
    m3 = dc.metrics.Metric(dc.metrics.balanced_accuracy_score)

    return [m1, m2, m3]


def define_parameters(target: str) -> dict:
    """
    Return the model architectures determined after grid search based on the target.
    """
    with open('/home/paperspace/Desktop/DL_Project/data/final_model_params.json', 'r') as f:
        model_params=json.load(f)

    target_specific_params=model_params[target]

    return target_specific_params

def define_random_parameters() -> dict:
    """
    Based on the specified choices, randomly select the parameters for the current model iteration.
    
    Returns:
    dictionary with keys representing the parameter id and the values as the parameter
    """
    graph_conv_layers_options = [
                                [64, 64],
                                # [64, 64, 64, 64],
                                [64, 128, 64],
                                [64, 128, 256, 384, 256, 128, 64],
                                # [64, 128, 256, 256, 128, 64]
                                ]

    batchnorm_options =[True, False]
    dropout_options = [0, 0.2]
    dense_layer_size_options = [128, 256]
    # dense_layer_size_options=[128]
    batch_size_options = [500]
    # epoch_list = [150]
    epoch_list=[50]

    parameters={}

    parameters['probability_threshold']=0.2
    parameters['batch_size']=random.choice(batch_size_options)
    parameters['graph_conv_layers']=random.choice(graph_conv_layers_options)
    parameters['batchnorm']=random.choice(batchnorm_options)
    parameters['dropout']=random.choice(dropout_options)
    parameters['dense_layer_size']=random.choice(dense_layer_size_options)
    parameters['epochs']=random.choice(epoch_list)

    return parameters


def copy_weights(src: str, dst: str) -> None:
    """
    Copy the weights saved in the save_checkpoint function to a new file.
    """
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(src, dst)
        else: raise

def load_prediction_set() -> pd.DataFrame:
    """
    Load drugbank compounds from pickle file for predictions.
    """
    base_path="/home/paperspace/Desktop/DL_Project/data/"
    df_file="full_prediction_set_fp_graphs.pickle"
    pred_set=pd.read_pickle(base_path+df_file)
    
    return pred_set


def make_final_preds(model, target: str) -> None:
    """
    Generate final predictions using a trained model and save the results to an Excel file.

    Parameters:
        model (object): The trained machine learning model for making predictions.
        target (str): The target variable for which predictions are being generated.

    Returns:
        None
    """
    base_path="/home/paperspace/Desktop/DL_Project/py_files/"

    print(f"Making predictions for {target}")
    prediction_set=load_prediction_set()
    pred_set=data_preparation.prepare_prediction_set(prediction_set)
    predictions = model.predict(pred_set)
    # Convert to Binary Label
    probability_threshold=0.2
    binary_preds = np.array([1 if x[0][1] >= probability_threshold else 0 for x in predictions]) # 1 = 'Active

    prediction_set['binary_pred']=binary_preds
    prediction_set['active_probability']=[x[0][1] for x in predictions]

    filtered=prediction_set.drop(columns=['morgan_fp', 'uncharged_mol', 'molecule', 'dc_graph'], errors='ignore')
    filtered.to_excel(f"{base_path}{target}_predictions.xlsx", index=False)


def run_model(df: pd.DataFrame, targets=['app', 'tau'], save_images=True):
    """
    Pipeline to run through the process of loading data, defining model architectures,
    splitting, training, logging, and predicting.
    """
    base_path="/home/paperspace/Desktop/DL_Project/py_files/"
    # with the chosen parameters, create models for both tasks
    for target in targets:

        # parameters=define_parameters(target)
        parameters=define_random_parameters()
        # threshold for active classification
        probability_threshold = parameters['probability_threshold']
        # split pubchem data
        train, valid, test = data_preparation.prepare_data(df, target)

        task = f'{target}_inhibition'

        # dictionary to store performance of model, parameters, true labels
        all_data_dictionary = data_preparation.create_model_dictionary(task, parameters, valid.y)

        # global model
        model = dc.models.GraphConvModel(n_tasks=1, mode='classification',
                                        graph_conv_layers=parameters["graph_conv_layers"],
                                        batchnorm=parameters['batchnorm'],
                                        dropout=parameters['dropout'],
                                        dense_layer_size=parameters['dense_layer_size'],
                                        batchsize=parameters['batch_size']
                                        )                     
        # display model type
        model_evaluation.print_dictionary(all_data_dictionary['model_parameters'])

        all_metrics = {}

        epochs_per_fit=5
        # if target=='tau':
        #     epochs=130
        # else:
        #     epochs=parameters['epochs']

        # model.fit(train, nb_epoch=epochs)
        # make_final_preds(model, target)
        for i in range(1, int((parameters['epochs'])/epochs_per_fit)+1):
            model.fit(train, nb_epoch=epochs_per_fit)

            training_preds=model.predict(train)
            training_metrics={}
            training_metrics['log_loss'] = log_loss(train.y, [i[0] for i in training_preds])

            validation_preds = model.predict(valid)
            validation_metrics = model.evaluate(valid, metrics=dc_metrics()) # attempt to diagnose model seeing valid
            # validation_metrics={}
            validation_metrics['log_loss'] = log_loss(valid.y, [i[0] for i in validation_preds])

            test_preds=model.predict(test)
            test_metrics={}
            test_metrics['log_loss'] = log_loss(test.y, [i[0] for i in test_preds])

            all_training_metrics = model_evaluation.prediction_metrics(training_metrics, "training", training_preds,
                                                train.y, probability_threshold)

            all_validation_metrics = model_evaluation.prediction_metrics(validation_metrics, "validation", validation_preds,
                                                valid.y, probability_threshold)

            all_test_metrics = model_evaluation.prediction_metrics(test_metrics, "test", test_preds,
                                                test.y, probability_threshold)       

            all_metrics=all_training_metrics.copy()
            all_metrics.update(all_validation_metrics)
            all_metrics.update(all_test_metrics)                                                                 

            # store performance results
            all_data_dictionary["results"][i*epochs_per_fit] = all_metrics

            print(f'\nEpoch {i*epochs_per_fit}\n------------------------')
            # output results
            model_evaluation.log_interim_results(all_metrics, target, i*epochs_per_fit)

            # model.model_dir=f'{base_path}{target}_model_checkpoints/epoch_{i*epochs_per_fit}_checkpoint'
            ## every 10 epochs save the weights - not working
            # model.save_checkpoint()

            # if (target=='tau') & (epochs_per_fit*i == 130):
            #     make_final_preds(model, target)
            #     break
            # if (target=='app') & (epochs_per_fit*i == 150):
            #     make_final_preds(model, target)
            #     break

        # save results to external files
        model_folder=model_evaluation.save_metrics(all_data_dictionary)
         # plots and saves
        model_evaluation.plot_performance(all_data_dictionary, save_image=save_images, save_location=model_folder)
        # top_level_checkpoints=f'{base_path}{target}_model_checkpoints' -- not working
        # copy_weights(top_level_checkpoints, f'{base_path}{target}_interim_weights')



if __name__=='__main__':
    """
    adfadf
    """
    df = data_preparation.load_data(testing=False)
    print(df.head())

    for j in range(3):
        run_model(df, targets=['tau', 'app'], save_images=True)