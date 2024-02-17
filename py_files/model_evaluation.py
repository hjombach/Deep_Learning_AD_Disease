import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from datetime import datetime
import os
import time
from pytz import timezone
import deepchem as dc

# https://bobbyhadz.com/blog/python-typeerror-object-of-type-int64-is-not-json-serializable
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def print_dictionary(dictionary_to_print: dict) -> None:
    """
    """
    for key, value in dictionary_to_print.items():

        if (key == 'validation_true_labels' or key == 'predictions'):
            continue
        if (isinstance(value, np.floating) or type(value)==float):
            value=np.round(value, 3)

        print(key, ' : ', value)

    print("------------------------------------")



def log_interim_results(dictionary_to_print: dict, target: str, epoch: int) -> None:
    """
    """
    base_path="/home/paperspace/Desktop/DL_Project/py_files/"

    tz = timezone('US/Eastern')
    now = datetime.now(tz)

    # mm/dd/YY H:M:S format
    s1 = now.strftime("%m_%d_%Y-%H_%M")

    with open(f"{base_path}{target}_final_interim_metrics.txt", "a") as file:
        
        file.write(f"\nEpoch {epoch} - {target} Evaluation - {s1}\n")
        file.write("------------------------------------\n")

        for dataset in dictionary_to_print.keys():
            file.write(f"{dataset} metrics\n")
            # for "validation" then "test"
            for key, value in dictionary_to_print[dataset].items():

                if (key == 'validation_true_labels' or key == 'predictions'):
                    continue
                if (isinstance(value, np.floating) or type(value)==float):
                    value=np.round(value, 3)

                file.write(f"{key} : {value}\n")
            file.write("------------------------------------\n")



def save_metrics(all_data_dictionary: dict, epochs_cut= None) -> None:
    """
    """
    # current date and time
    tz = timezone('US/Eastern')
    now = datetime.now(tz)
    # mm/dd/YY H:M:S format
    s1 = now.strftime("%m_%d_%Y-%H_%M")

    # class to overrride Encoder needed to dump dictionary
    json_str = json.dumps(all_data_dictionary, cls=NpEncoder)

    # variables for naming file
    task=all_data_dictionary['model_parameters']['task']
    target=task.replace("_inhibition", "")
    layers=all_data_dictionary['model_parameters']['graph_conv_layers']
    dropout=all_data_dictionary['model_parameters']['dropout']
    epochs=all_data_dictionary['model_parameters']['epochs']
    batch_norm=all_data_dictionary['model_parameters']['batch_norm']
    dense_layer_size=all_data_dictionary['model_parameters']['dense_layer_size']
    batch_size=all_data_dictionary['model_parameters']['batch_size']

    strings_for_name=[s1, target, layers, dropout,
                      batch_norm, dense_layer_size, batch_size, epochs]

    full_name=""
    for i in range(len(strings_for_name)):

        variable= strings_for_name[i]
        full_name+=str(variable)+"_"

    base_path="/home/paperspace/Desktop/DL_Project"
    full_directory=f"{base_path}/model_results/" + full_name + "results"

    if not os.path.exists(full_directory):
        os.mkdir(full_directory)

    # # subfolder to store images / other data from run
    file_name=full_name + "results_file.txt"
    full_path=full_directory+"/"+file_name

    # # create file for results
    f = open(f'{full_path}', "w+")
    f.write(json_str)
    f.close()

    # record_best_results(all_data_dictionary, s1, file_name, epochs_cut)

    return full_directory


def record_best_results(all_data_dictionary: dict, model_time: str,
                        filename: str, epochs_cut=None) -> None:
    """
    """
    base_path="/home/paperspace/Desktop/DL_Project"
    summary_filename = f'{base_path}/model_results/analysis_summaries.xlsx'

    if not os.path.isfile(summary_filename):
        wb = Workbook()
        wb.save(f'{summary_filename}')
    ### read the file as a pandas data frame - append the new results as new row
    ### rewrite the results with the newest data to the excel file

    results_summary=pd.read_excel(summary_filename)

    data_to_store=all_data_dictionary['model_parameters'].copy()
    data_to_store.update(all_data_dictionary['best_results'])
    data_to_store['run_date']=model_time
    data_to_store['results_file']=filename

    if epochs_cut is not None:
        data_to_store['epochs_run']=epochs_cut

    new_results=pd.DataFrame([data_to_store])

    with_new=pd.concat([results_summary, new_results])
    with_new.to_excel(summary_filename, index=False)


def prediction_metrics(all_metrics_dictionary: dict,
                       data_set: str,
                       predictions: list,
                       true_labels: pd.Series,
                       probability_threshold=0.2
                       ) -> dict:
    """
    """
    # create predictions and score
    binary_predictions = np.array([1 if x[0][1] >= probability_threshold \
                                   else 0 for x in predictions]) # 1 = 'Active

    num_predicted_hits = binary_predictions.sum()
    num_true_hits = true_labels.sum()
    # accuracy = round(accuracy_score(true_labels, binary_predictions)*100, 2)

    num_true_positives = sum([1 for x, y in zip(binary_predictions, true_labels) if x == y == 1])
    num_false_positives = num_predicted_hits - num_true_positives
    num_false_negatives = sum([1 for x, y in zip(binary_predictions, true_labels) if x == 0 and y == 1])

    precision = round(num_true_positives/(num_true_positives + num_false_positives), 3)
    recall = round(num_true_positives/(num_true_positives + num_false_negatives), 3)

    ## consider removing the dataset prefix and creating a sub dictionary within 
    overall_dictionary={}
    # all_metrics_dictionary[data_set]={}

    # data_set_dict={}
    all_metrics_dictionary[f'num_predicted_hits'] = num_predicted_hits
    all_metrics_dictionary[f'num_true_hits'] = num_true_hits # doesn't change (objective to task)
    # all_metrics_dictionary['accuracy'] = accuracy
    all_metrics_dictionary[f'num_true_positives'] = num_true_positives
    all_metrics_dictionary[f'num_false_positives'] = num_false_positives
    all_metrics_dictionary[f'precision (TP/(TP + FP))'] = precision
    all_metrics_dictionary[f'recall (TP/TP + FN))'] = recall
    all_metrics_dictionary[f'predictions']=predictions

    overall_dictionary[data_set]={}
    overall_dictionary[data_set]=all_metrics_dictionary

    return overall_dictionary


def find_best_results(all_data_dictionary: dict):
    """
    Doesn't seem to keep the best results in the same epoch - can be spread
    across multiple epochs - not as helpful
    """
    # default to zero
    max_roc=max_balanced_acc=max_true_pos=max_precision=max_recall=0

    for key, vals in all_data_dictionary['results'].items():

        all_data_dictionary['best_results']['roc_auc_score'] = max([max_roc, vals['roc_auc_score']])
        all_data_dictionary['best_results']['balanced_accuracy'] = max([max_balanced_acc, vals['balanced_accuracy_score']])
        all_data_dictionary['best_results']['num_true_positives'] = max([max_true_pos, vals['num_true_positives']])
        all_data_dictionary['best_results']['precision (TP/(TP + FP))'] = max([max_precision, vals['precision (TP/(TP + FP))']])
        all_data_dictionary['best_results']['recall (TP/TP + FN))'] = max([max_recall, vals['recall (TP/TP + FN))']])

    print(f"\nBest Results Throughout Model Training:")
    print_dictionary(all_data_dictionary['best_results'])

    return all_data_dictionary


def extract_numeric_performance(all_data_dictionary: dict) -> list:
    """
    Called from within plot_performance function
    """
    # the last digit is how often the model was evaluated (eg: every 3rd epoc)
    epochs = list(all_data_dictionary['results'].keys())
    epoch_numbers=epochs
    overall_data={}
    overall_data['epochs']=epoch_numbers
    # data_to_plot = defaultdict(list)
    # data_to_plot['epochs']=epoch_numbers

    # then will pull out "validation" "test" dictionaries
    for dataset in ['training', 'validation', 'test']:
        data_to_plot = defaultdict(list)
        # data_to_plot['epochs']=epoch_numbers
        overall_data[dataset]={}
        for key, val in all_data_dictionary['results'].items():
            
            # data_to_plot['roc_auc_vals'].append(all_data_dictionary['results'][key][dataset]['roc_auc_score'])
            data_to_plot['num_predicted_hits'].append(all_data_dictionary['results'][key][dataset]['num_predicted_hits'])
            data_to_plot['true_positive_list'].append(all_data_dictionary['results'][key][dataset]['num_true_positives'])
            data_to_plot['loss_vals'].append(all_data_dictionary['results'][key][dataset]['log_loss'])
            # data_to_plot['accuracy_vals'].append(all_data_dictionary['results'][key]['accuracy'])
            data_to_plot['precision_vals'].append(all_data_dictionary['results'][key][dataset]['precision (TP/(TP + FP))'])
            data_to_plot['recall_vals'].append(all_data_dictionary['results'][key][dataset]['recall (TP/TP + FN))'])

        overall_data[dataset]=data_to_plot
    return overall_data


def plot_performance(all_data_dictionary: dict, save_image=False, save_location=""):
    """
    abadafd
    """
    plt.style.use("ggplot")

    data_to_plot=extract_numeric_performance(all_data_dictionary)
    epochs=data_to_plot['epochs']
    # true_hit_num=all_data_dictionary['results'][epochs[0]]['num_true_hits']
    print_dictionary(all_data_dictionary['model_parameters'])

    title_size=12
    label_size=10
    fig, ax = plt.subplots(3, 4, figsize = (16, 8), tight_layout = True, sharey=False)
    plt.setp(ax, xlabel='Epochs')
    
    for data_set, row in zip(['training', 'validation', 'test'], [0, 1, 2]):
        epochs=data_to_plot['epochs']
        true_hit_num=all_data_dictionary['results'][epochs[0]][data_set][f'num_true_hits']
        ax[row][0].plot(epochs, data_to_plot[data_set][f'loss_vals'], label = 'cross entropy loss')
        ax[row][0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax[row][0].legend()
        ax[row][0].set_ylim([0, 1])
        ax[row][0].set_ylabel("Cross Entropy Loss", size=label_size)
        ax[row][0].set_title(f"{data_set.title()} Set Cross Entropy Loss", size=title_size)

        ax[row][1].plot(epochs, data_to_plot[data_set][f'recall_vals'], label = 'recall')
        ax[row][1].plot(epochs, data_to_plot[data_set][f'precision_vals'], label = 'precision')
        ax[row][1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax[row][1].set_ylim([0, 1])
        ax[row][1].set_ylabel("Metric Performance")
        ax[row][1].set_title(f"{data_set.title()} Set Precision and Recall", size=title_size)
        ax[row][1].legend()

        # ax[row][2].plot(epochs, data_to_plot[data_set][f'roc_auc_vals'], label = 'roc_auc')
        # ax[row][2].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        # ax[row][2].set_ylim([0, 1])
        # ax[row][2].set_ylabel("Metric Performance", size=label_size)
        # ax[row][2].set_title(f"{data_set.title()} Set ROC AUC Values", size=title_size)
        # ax[row][2].legend()

        ax[row][2].bar(epochs, data_to_plot[data_set][f'num_predicted_hits'], label='predicted hits')
        ax[row][2].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        # ax[row][2].set_ylim([0, 1])
        ax[row][2].set_ylabel("Metric Performance", size=label_size)
        ax[row][2].set_title(f"{data_set.title()} Set - Prediction Activity", size=title_size)
        ax[row][2].legend()

        # ax[row][3].bar(epochs, data_to_plot[data_set][f'num_predicted_hits'], label='predicted hits')
        ax[row][3].bar(epochs, data_to_plot[data_set][f'true_positive_list'], label = 'correctly predicted hits')
        ax[row][3].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax[row][3].axhline(y = true_hit_num, color = 'green', linestyle = 'dashed', label = "True Number of Hits", alpha=0.5)
        ax[row][3].set_ylabel("Compound Count", size=label_size)
        ax[row][3].set_title(f"{data_set.title()} Set - True Positives Identification", size=title_size)
        ax[row][3].legend()

    tz = timezone('US/Eastern')
    now = datetime.now(tz)

    # mm/dd/YY H:M:S format
    s1 = now.strftime("%m_%d_%Y-%H_%M")

    task=all_data_dictionary['model_parameters']['task']

    if save_image:
        plt.savefig(f"{save_location}/{s1}_{task}_model_performance.png")

    # plt.show()