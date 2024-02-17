import data_preparation
import model_evaluation
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


def dc_metrics() -> list:
    '''
    adfdf
    '''
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
                                [64, 64]
                                # [64, 64, 64, 64],
                                # [64, 128, 64],
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


def run_model(df: pd.DataFrame, targets=['app', 'tau'], save_images=False):

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
                                        batchnorm=parameters['batchnorm'],
                                        dropout=parameters['dropout'],
                                        dense_layer_size=parameters['dense_layer_size'],
                                        batchsize=parameters['batch_size'])
        # display model type
        model_evaluation.print_dictionary(all_data_dictionary['model_parameters'])

        all_metrics = {}
        for i in range(1, parameters['epochs']+1):

            model.fit(train, nb_epoch=1)

            # every n epochs evaluate the model
            if i % 2 == 0:
                validation_preds = model.predict(valid)
                metrics = model.evaluate(valid, metrics=dc_metrics())
                metrics['log_loss'] = log_loss(valid.y, [i[0] for i in validation_preds])

                all_metrics = model_evaluation.prediction_metrics(metrics, validation_preds,
                                                 valid.y, probability_threshold)
                # store performance results
                all_data_dictionary["results"][i] = all_metrics

                print(f'\nEpoch {i}\n------------------------')
                # output results
                model_evaluation.print_dictionary(all_metrics)

            # if after 10 epochs we don't have any predictions as active, stop training the model and try new
                if i > 10 and all_data_dictionary['results'][i]['num_predicted_hits'] == 0:
                    print("Model training ended early due to lack of predictions")
                    all_data_dictionary = model_evaluation.find_best_results(all_data_dictionary)
                    model_folder=model_evaluation.save_metrics(all_data_dictionary, i)
                    model_evaluation.plot_performance(all_data_dictionary, save_image=save_images, save_location=model_folder)
                    break

        # add elements to dictionary with highest scores for metrics
        all_data_dictionary = model_evaluation.find_best_results(all_data_dictionary)
        # save results to external files
        model_folder=model_evaluation.save_metrics(all_data_dictionary)
         # plots and saves
        model_evaluation.plot_performance(all_data_dictionary, save_image=save_images, save_location=model_folder)
        # return all_data_dictionary



# if __name__=='__main__':
#     df = data_preparation.load_data(testing=True)
#     print(df.head())

#     ## each test performed for app and tau
#     number_tests=1
#     for i in range(number_tests):
#         run_model(df, targets=['app'], save_images=True)
#         # run_model(df, save_images=True)