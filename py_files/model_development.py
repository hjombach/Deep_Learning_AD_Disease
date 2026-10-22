from datetime import datetime
from functools import partial
import json
import os
import pickle
import random
import time
import warnings

import deepchem as dc
from deepchem import models
from deepchem.models import GraphConvModel
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from openpyxl import Workbook
import pandas as pd
from pytz import timezone
from rdkit import RDLogger
from sklearn.metrics import log_loss
import torch

import data_preparation
import model_evaluation

warnings.filterwarnings('ignore')
RDLogger.DisableLog("rdApp.*")



def dc_metrics() -> list:
    """
    Deepchem classifier metrics to log throughout training.

    Returns:
        list - list of Deepchem classifier metrics

    Raises:
        ValueError: If the metrics are not 'roc_auc_score', 'accuracy_score', or 'balanced_accuracy_score'
    """
    # metrics to log throughout training
    m1 = dc.metrics.Metric(dc.metrics.roc_auc_score)
    m2 = dc.metrics.Metric(dc.metrics.accuracy_score)
    m3 = dc.metrics.Metric(dc.metrics.balanced_accuracy_score)

    return [m1, m2, m3]


def define_parameters() -> dict:
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
    epoch_list=[10, 15, 20]

    parameters={}

    parameters['probability_threshold']=0.5
    parameters['batch_size']=random.choice(batch_size_options)
    parameters['graph_conv_layers']=random.choice(graph_conv_layers_options)
    parameters['batch_normalize']=random.choice(batchnorm_options)
    parameters['dropout']=random.choice(dropout_options)
    parameters['dense_layer_size']=random.choice(dense_layer_size_options)
    parameters['epochs']=random.choice(epoch_list)

    return parameters


def run_model(df: pd.DataFrame, 
                targets=['app', 'tau'], 
                save_images: bool =False, 
                testing : bool = False):
    """ 
    Run the model on the training data.

    Args:
        df (pd.DataFrame): DataFrame containing the training data
        targets (list): List of targets to run the model on
        save_images (bool): Whether to save the images of the performance plots
        testing (bool): Whether the model being run on a sample of data

    Returns:
        None

    Raises:
        ValueError: If the targets are not 'app' or 'tau'
    """
    parameters=define_parameters()

    # threshold for active classification
    probability_threshold = parameters['probability_threshold']
    # tasks to model

    # with the chosen parameters, create models for both tasks
    for target in targets:
        # split pubchem data
        train, valid, test = data_preparation.prepare_data(df, target)
        task = f'{target}_inhibition'

        # dictionary to store performance of model, parameters, true labels
        all_data_dictionary = data_preparation.create_model_dictionary(task, parameters, valid.y)

        # global model
        model = dc.models.GraphConvModel(n_tasks=1, mode='classification',
                                        graph_conv_layers=parameters["graph_conv_layers"],
                                        batch_normalize=parameters['batch_normalize'],
                                        dropout=parameters['dropout'],
                                        dense_layer_size=parameters['dense_layer_size'],
                                        batch_size=parameters['batch_size'])
        # display model type
        model_evaluation.print_dictionary(all_data_dictionary['model_parameters'])

        all_metrics = {}
        for i in range(1, parameters['epochs']+1):

            model.fit(train, nb_epoch=1)

            # every n epochs evaluate the model
            if i % 2 == 0:
                
                all_metrics = {}
                # for data_set, data_points in enumerate([('training', train), ('validation', valid)]):
                for data_set, data_points in {'training': train,
                                              'validation': valid}.items():

                    preds = model.predict(data_points)
                    metrics = model.evaluate(data_points, metrics=dc_metrics())
                    metrics['log_loss'] = log_loss(data_points.y, [i[0] for i in preds])

                    all_metrics_subset = model_evaluation.prediction_metrics(
                                                    all_metrics_dictionary = metrics, 
                                                    data_set = data_set, 
                                                    predictions = preds,
                                                    true_labels = data_points.y, 
                                                    probability_threshold = probability_threshold
                                                    )

                    # all_metrics[data_set] = {}
                    all_metrics[data_set] = all_metrics_subset.copy()
                    # store performance results
                    all_data_dictionary["results"][i] = all_metrics

                print(f'\nEpoch {i}\n------------------------')
                # output results
                model_evaluation.print_dictionary(all_metrics['validation'])

        # add elements to dictionary with highest scores for metrics
        all_data_dictionary = model_evaluation.find_best_results(all_data_dictionary)
        # save results to external files
        model_folder=model_evaluation.save_metrics(all_data_dictionary, testing = testing)
         # plots and saves
        model_evaluation.plot_performance(all_data_dictionary, save_image=save_images, save_location=model_folder)
        # return all_data_dictionary



if __name__=='__main__':

    TESTING = False
    df = data_preparation.load_data(testing=TESTING)

    ## each test performed for app and tau
    number_tests=4
    for i in range(number_tests):
        # run_model(df, targets=['app'], save_images=True, testing=TESTING)
        run_model(df, save_images=True, testing=TESTING)