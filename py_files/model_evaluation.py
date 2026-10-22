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
from openpyxl import Workbook

# https://bobbyhadz.com/blog/python-typeerror-object-of-type-int64-is-not-json-serializable
class NpEncoder(json.JSONEncoder):
    """JSON encoder that converts NumPy types to native Python types."""

    def default(self, obj):
        """
        Serialize NumPy scalars and arrays for JSON dumping.

        Args:
            obj: Object being encoded. Handles np.integer, np.floating,
                and np.ndarray; otherwise defers to the parent encoder.

        Returns:
            A JSON-serializable Python int, float, or list, or the result
            of the parent encoder's default method.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def print_dictionary(dictionary_to_print: dict) -> None:
    """
    Pretty-print a metrics dictionary, skipping large array fields.

    Float values are rounded to 3 decimal places. Keys named
    'validation_true_labels' or 'predictions' are omitted from output.

    Args:
        dictionary_to_print (dict): Metrics or parameter dictionary to print.
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
    Append per-epoch evaluation metrics to a target-specific text log.

    Writes validation and test metrics for the given epoch to
    ``{target}_final_interim_metrics.txt``, skipping large array fields
    and rounding floats to 3 decimal places.

    Args:
        dictionary_to_print (dict): Nested metrics dictionary keyed by
            dataset name (e.g. 'validation', 'test'), each mapping metric
            names to values.
        target (str): Target identifier used in the output filename
            (e.g. 'app', 'tau').
        epoch (int): Epoch number being logged.
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
                print("key: " + key)
                if (key == 'validation_true_labels' or key == 'predictions'):
                    continue
                if (isinstance(value, np.floating) or type(value)==float):
                    value=np.round(value, 3)

                file.write(f"{key} : {value}\n")
            file.write("------------------------------------\n")



def save_metrics(all_data_dictionary: dict, epochs_cut= None) -> None:
    """
    Persist a full training run's metrics and parameters to disk.

    Serializes ``all_data_dictionary`` as JSON into a uniquely named
    subdirectory under ``model_results/``, then appends a summary row to
    the analysis Excel workbook via ``record_best_results``.

    Args:
        all_data_dictionary (dict): Complete run record containing
            'model_parameters', 'results', and 'best_results'.
        epochs_cut (optional): If provided, recorded as the number of
            epochs actually run when early stopping or truncation applies.

    Returns:
        str: Absolute path to the created results directory.
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
    batch_norm=all_data_dictionary['model_parameters']['batch_normalize']
    dense_layer_size=all_data_dictionary['model_parameters']['dense_layer_size']
    batch_size=all_data_dictionary['model_parameters']['batch_size']

    strings_for_name=[s1, target, layers, dropout,
                      batch_norm, dense_layer_size, batch_size, epochs]

    full_name=""
    for i in range(len(strings_for_name)):

        variable= strings_for_name[i]
        full_name+=str(variable)+"_"

    base_path="/home/paperspace/Deep_Learning_AD_Disease"
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

    record_best_results(all_data_dictionary, s1, file_name, epochs_cut)

    return full_directory


def record_best_results(all_data_dictionary: dict, model_time: str,
                        filename: str, epochs_cut=None) -> None:
    """
    Append model parameters and best metrics to the summary Excel workbook.

    Creates ``analysis_summaries.xlsx`` if it does not exist, then appends
    a new row combining model parameters, best results, run timestamp, and
    results filename.

    Args:
        all_data_dictionary (dict): Run record with 'model_parameters' and
            'best_results' keys.
        model_time (str): Timestamp string identifying when the run finished.
        filename (str): Name of the results file written by ``save_metrics``.
        epochs_cut (optional): If provided, stored as 'epochs_run' in the
            summary row.
    """
    # base_path="/home/paperspace/Desktop/DL_Project"
    base_path = "/home/paperspace/Deep_Learning_AD_Disease"
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
    Compute hit-based classification metrics from model probability outputs.

    Converts class-1 probabilities to binary predictions using
    ``probability_threshold``, then calculates predicted/true hit counts,
    true/false positives, precision, and recall. Results are written into
    ``all_metrics_dictionary`` in place.

    Args:
        all_metrics_dictionary (dict): Dictionary updated in place with
            computed metric values and raw predictions.
        data_set (str): Dataset name (e.g. 'training', 'validation');
            currently unused in the returned structure but retained for
            call-site clarity.
        predictions (list): Model outputs; each entry is expected to
            contain class probabilities such that ``x[0][1]`` is the
            active-class probability.
        true_labels (pd.Series): Ground-truth binary labels (1 = active).
        probability_threshold (float, optional): Minimum class-1
            probability to count a compound as a predicted hit.
            Defaults to 0.2.

    Returns:
        dict: The updated ``all_metrics_dictionary`` with keys including
            num_predicted_hits, num_true_hits, num_true_positives,
            num_false_positives, precision, recall, and predictions.
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
    all_metrics_dictionary[f'precision'] = precision
    all_metrics_dictionary[f'recall'] = recall
    all_metrics_dictionary[f'predictions']=predictions

    overall_dictionary[data_set]={}
    overall_dictionary[data_set]=all_metrics_dictionary
    return all_metrics_dictionary

    # return overall_dictionary


def find_best_results(all_data_dictionary: dict):
    """
    Record the best validation metric values observed across all epochs.

    For each tracked metric (ROC AUC, balanced accuracy, true positives,
    precision, recall), stores the maximum value seen over validation
    results. Peak values may come from different epochs.

    Args:
        all_data_dictionary (dict): Run record with a 'results' mapping of
            epoch -> dataset metrics, and a 'best_results' dict to update.

    Returns:
        dict: The same ``all_data_dictionary`` with 'best_results' filled in.
    """
    # default to zero
    max_roc=max_balanced_acc=max_true_pos=max_precision=max_recall=0

    ## iterate over every epoch's validation data results
    for key, vals in all_data_dictionary['results'].items():

        all_data_dictionary['best_results']['roc_auc_score'] = max([max_roc, vals['validation']['roc_auc_score']])
        all_data_dictionary['best_results']['balanced_accuracy'] = max([max_balanced_acc, vals['validation']['balanced_accuracy_score']])
        all_data_dictionary['best_results']['num_true_positives'] = max([max_true_pos, vals['validation']['num_true_positives']])
        all_data_dictionary['best_results']['precision'] = max([max_precision, vals['validation']['precision']])
        all_data_dictionary['best_results']['recall'] = max([max_recall, vals['validation']['recall']])

    print(f"\nBest Results Throughout Model Training:")
    print_dictionary(all_data_dictionary['best_results'])

    return all_data_dictionary


def extract_numeric_performance(all_data_dictionary: dict) -> list:
    """
    Reshape epoch-wise metrics into lists suitable for plotting.

    Iterates over training and validation results and builds per-metric
    sequences (ROC AUC, predicted hits, true positives, log loss,
    precision, recall) aligned with the recorded epoch keys.

    Args:
        all_data_dictionary (dict): Run record whose 'results' key maps
            epoch identifiers to per-dataset metric dictionaries.

    Returns:
        dict: Mapping with an 'epochs' list plus 'training' and
            'validation' entries, each a dict of metric name -> list of
            values across epochs.
    """
    # the last digit is how often the model was evaluated (eg: every 3rd epoc)
    epochs = list(all_data_dictionary['results'].keys())
    epoch_numbers=epochs
    overall_data={}
    overall_data['epochs']=epoch_numbers
    # data_to_plot = defaultdict(list)
    # data_to_plot['epochs']=epoch_numbers

    # then will pull out "validation" "test" dictionaries
    # for dataset in ['training', 'validation', 'test']:
    for dataset in ['training', 'validation']:
        data_to_plot = defaultdict(list)
        # data_to_plot['epochs']=epoch_numbers
        overall_data[dataset]={}
        for key, val in all_data_dictionary['results'].items():
            
            data_to_plot['roc_auc_vals'].append(all_data_dictionary['results'][key][dataset]['roc_auc_score'])
            data_to_plot['num_predicted_hits'].append(all_data_dictionary['results'][key][dataset]['num_predicted_hits'])
            data_to_plot['true_positive_list'].append(all_data_dictionary['results'][key][dataset]['num_true_positives'])
            data_to_plot['loss_vals'].append(all_data_dictionary['results'][key][dataset]['log_loss'])
            # data_to_plot['accuracy_vals'].append(all_data_dictionary['results'][key]['accuracy'])
            data_to_plot['precision_vals'].append(all_data_dictionary['results'][key][dataset]['precision'])
            data_to_plot['recall_vals'].append(all_data_dictionary['results'][key][dataset]['recall'])

        overall_data[dataset]=data_to_plot
    return overall_data


def plot_performance(all_data_dictionary: dict, save_image=False, save_location=""):
    """
    Plot training and validation performance metrics across epochs.

    Creates a multi-panel figure (loss, precision/recall, ROC AUC, true
    positives, and predicted hits) for each dataset present in the results.
    Optionally saves the figure to ``save_location``.

    Args:
        all_data_dictionary (dict): Run record with 'model_parameters' and
            epoch-wise 'results'.
        save_image (bool, optional): If True, write a PNG of the figure.
            Defaults to False.
        save_location (str, optional): Directory path for the saved image
            when ``save_image`` is True. Defaults to "".
    """
    plt.style.use("ggplot")

    data_to_plot=extract_numeric_performance(all_data_dictionary)
    epochs=data_to_plot['epochs']
    print_dictionary(all_data_dictionary['model_parameters'])

    title_size=12
    label_size=10
    fig, ax = plt.subplots(2, 5, figsize = (18, 8), tight_layout = True, sharey=False, squeeze=False)
    plt.setp(ax, xlabel='Epochs')
    
    sources = list(next(iter(all_data_dictionary['results'].values())).keys())

    for row, data_set in enumerate(sources):

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

        ax[row][2].plot(epochs, data_to_plot[data_set][f'roc_auc_vals'], label = 'roc_auc')
        ax[row][2].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax[row][2].set_ylim([0, 1])
        ax[row][2].set_ylabel("Metric Performance", size=label_size)
        ax[row][2].set_title(f"{data_set.title()} Set ROC AUC Values", size=title_size)
        ax[row][2].legend()

        ax[row][3].bar(epochs, data_to_plot[data_set][f'true_positive_list'], label = 'correctly predicted hits')
        ax[row][3].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax[row][3].axhline(y = true_hit_num, color = 'green', linestyle = 'dashed', label = "True Number of Hits", alpha=0.5)
        ax[row][3].set_ylabel("Compound Count", size=label_size)
        ax[row][3].set_title(f"{data_set.title()} Set - True Positives Identification", size=title_size)
        ax[row][3].legend()

        ax[row][4].bar(epochs, data_to_plot[data_set][f'num_predicted_hits'], label='predicted hits')
        ax[row][4].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax[row][4].set_ylabel("Metric Performance", size=label_size)
        ax[row][4].set_title(f"{data_set.title()} Set - Prediction Activity", size=title_size)
        ax[row][4].legend()

    tz = timezone('US/Eastern')
    now = datetime.now(tz)

    # mm/dd/YY H:M:S format
    s1 = now.strftime("%m_%d_%Y-%H_%M")

    task=all_data_dictionary['model_parameters']['task']

    if save_image:
        plt.savefig(f"{save_location}/{s1}_{task}_model_performance.png")

    # plt.show()