import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
import json
import os
import numpy as np
import deepchem as dc
 
def load_data(testing=False):
    base_path="/home/paperspace/Desktop/DL_Project"
    if testing:
        # print(os.getcwd())
        df = pd.read_pickle(f'{base_path}/data/sample.pickle')
    else:
        df = pd.read_pickle(f'{base_path}/data/cleaned_smiles_both_assays_pubchem_fp_graphs_2023Dec24.pickle')
    # Create Binary Labels from Text Labels
    df['app_label'] = (df["app_inhibitor"] == 'Active').astype(int)
    df['tau_label'] = (df["tau_inhibitor"] == 'Active').astype(int)

    # Descriptive info
    print(f"{df['app_label'].sum():,} compounds inhibit APP")
    print(f"{df['tau_label'].sum():,} compounds inhibit Tau")
    # display(df.head(3))

    return df

def prepare_prediction_set(df_copy: pd.DataFrame):
    """
    adsfaf
    """
    X = np.array(df_copy['dc_graph'])
    # transform
    dataset = dc.data.NumpyDataset(X = X)

    return dataset


def prepare_data(df_copy: pd.DataFrame, target: str):
    """
    """
    task = f'{target}_inhibition'

    # filter compounds not tested on the target of interest
    df = df_copy[~df_copy[f'{target}_inhibitor'].isnull()]

    X = np.array(df['dc_graph'])
    y = np.array(df[f"{target}_label"])

    # transform
    dataset = dc.data.NumpyDataset(X = X, y = y)
    # take into account imbalance - assigns weights of both classes
    # to be equivalent, even though class labels are very imbalanced
    transformer = dc.trans.BalancingTransformer(dataset = dataset)
    dataset = transformer.transform(dataset)

    # split the data to be in equal proportions through sets
    splitters = {
                'index': dc.splits.IndexSplitter(),
                'random stratified': dc.splits.RandomStratifiedSplitter(),
                'scaffold': dc.splits.ScaffoldSplitter(),
    }

    # default 0.8, 0.1, 0.1 data split
    split = 'random stratified'
    splitter = splitters[split]
    # split stratified to minimize
    train, valid, test = splitter.train_valid_test_split(dataset)
    # make sure the validation and test sets are eqiuvalent (plus or minus 1)
    assert valid.y.sum()-1 <= test.y.sum() <= valid.y.sum()+1

    return train, valid, test


def confirm_split(train, valid, test, target: str) -> None:
    """

    """
    total_positives = sum(train.y) + sum(valid.y) + sum(test.y)
    expected_train_samples = int(total_positives * 0.8)
    expected_val_samples = int(total_positives * 0.1)
    expected_test_samples = int(total_positives * 0.1)
    print(total_positives)

    assert sum(train.y) == expected_train_samples, f"Train set should have {expected_train_samples} positive samples."
    assert sum(valid.y) == expected_val_samples, f"Validation set should have {expected_val_samples} positive samples."
    assert sum(test.y) == expected_test_samples, f"Test set should have {expected_test_samples} positive samples."

    print(f"Training Data Distribution - {target} inhibition")
    print(f"{sum(train.y)} Active Compounds")

    print("\nValidation Data Distribution")
    print(f"{sum(valid.y)} Active Compounds")

    print("\nTest Data Distribution")
    print(f"{sum(test.y)} Active Compounds")


def create_model_dictionary(task: str, model_params: dict,
                            validation_labels: list) -> dict:
    """
    """
    model_parameters= {
            'model_parameters': {
                'task': task,
                'probability_threshold': model_params["probability_threshold"],
                'graph_conv_layers': model_params["graph_conv_layers"],
                'epochs': model_params["epochs"],
                'batch_norm': model_params['batchnorm'],
                'dropout': model_params['dropout'],
                'dense_layer_size': model_params['dense_layer_size'],
                'batch_size': model_params['batch_size'],
                'validation_true_labels': validation_labels
            },
            'results': {},
            'best_results': {}
    }

    return model_parameters


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
