import json
import os
import pickle

import deepchem as dc
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

RDLogger.DisableLog("rdApp.*")


def load_data(testing: bool=False):
    """ 
    Load the cleaned compound dataset and generate binary activity labels.

    Loads either a sample dataset for testing or the full cleaned dataset
    containing compound SMILES, assay results, fingerprints, and molecular
    graph representations. Converts APP and Tau inhibitor activity outcomes
    into binary labels and prints the number of active compounds for each
    target.

    Args:
        testing (bool, optional): If True, loads a smaller sample dataset
            for testing. Otherwise, loads the full dataset. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame containing the loaded compound data and two
            additional binary label columns:
            - 'app_label': 1 if the compound is active against APP, otherwise 0.
            - 'tau_label': 1 if the compound is active against Tau, otherwise 0.
    """
#    /data/final_data/training_data_both_assays_no_graph.parquet
    base_path = "/home/paperspace/Deep_Learning_AD_Disease"
    if testing:
        # print(os.getcwd())
        # df = pd.read_pickle(f'{base_path}/data/sample.pickle')
        df = pd.read_parquet(f'{base_path}/data/final_data/training_data_both_assays_no_graph.parquet').head(10000)
    else: 
        # df = pd.read_pickle(f'{base_path}/data/cleaned_smiles_both_assays_pubchem_fp_graphs_2023Dec24.pickle')
        df = pd.read_parquet(f'{base_path}/data/final_data/training_data_both_assays_no_graph.parquet')
    # Create Binary Labels from Text Labels
    df['app_label'] = (df["app_inhibitor"] == 'Active').astype(int)
    df['tau_label'] = (df["tau_inhibitor"] == 'Active').astype(int)

    # Descriptive info
    print(f"{df['app_label'].sum():,} compounds inhibit APP")
    print(f"{df['tau_label'].sum():,} compounds inhibit Tau")
    # display(df.head(3))

    df = generate_deepchem_graphs(df)

    # df = df[['inchikey', 'dc_graph', 'app_label', 'tau_label']]
    df = df.set_index("inchikey")

    return df


def generate_deepchem_graphs(df):
    df = df.copy()

    # Parse SMILES and identify invalid molecules
    mols = df["canonical_tautomer_smiles"].apply(
        lambda s: Chem.MolFromSmiles(s) if isinstance(s, str) else None
    )

    valid_mask = mols.notna()

    print(f"Total compounds: {len(df)}")
    print(f"Invalid SMILES: {(~valid_mask).sum()}")

    # Keep only valid molecules
    df = df.loc[valid_mask].copy()
    valid_mols = mols.loc[valid_mask].tolist()

    # Featurize valid molecules
    featurizer = dc.feat.ConvMolFeaturizer(
        per_atom_fragmentation=False
    )
    graphs = featurizer.featurize(valid_mols)

    # Confirm every graph is a ConvMol
    graph_mask = np.array([
        isinstance(graph, dc.feat.mol_graphs.ConvMol)
        for graph in graphs
    ])

    # print(f"Valid graphs: {graph_mask.sum()}")
    # print(f"Invalid graphs: {(~graph_mask).sum()}")

    df = df.iloc[np.flatnonzero(graph_mask)].copy()
    df["dc_graph"] = graphs[graph_mask]

    return df

def prepare_prediction_set(df_copy: pd.DataFrame):
    """
    Prepare a dataset of molecular graph representations for model prediction.

    Extracts graph representations from the input DataFrame and converts
    them into a DeepChem NumpyDataset for use with a trained machine
    learning model.

    Args:
        df_copy (pd.DataFrame): DataFrame containing molecular graph
            representations in the 'dc_graph' column.

    Returns:
        dc.data.NumpyDataset: DeepChem dataset containing the molecular
            graph representations, without labels, for prediction.

    Notes:
        - The function does not apply class balancing or split the dataset.
        - The input DataFrame must contain a 'dc_graph' column.
    """
    X = np.array(df_copy['dc_graph'])
    # transform
    dataset = dc.data.NumpyDataset(X = X)

    return dataset


def prepare_data(df_copy: pd.DataFrame, target: str, split: str = 'scaffold'):
    """
    Prepare a target-specific dataset for graph-based machine learning.

    Filters compounds to retain those tested against the specified target,
    extracts graph representations and binary activity labels, applies
    class-balancing weights, and splits the dataset into training,
    validation, and test sets using the selected splitting strategy.

    Args:
        df_copy (pd.DataFrame): DataFrame containing compound graph
            representations ('dc_graph'), target-specific inhibitor
            indicators, and binary activity labels.
        target (str): Name of the target of interest. Used to identify
            the corresponding inhibitor and label columns.
        split (str, optional): Dataset splitting strategy. Supported values:
            - 'index': Split by dataset index.
            - 'random stratified': Random stratified split.
            - 'scaffold': Split based on molecular scaffolds.
            Defaults to 'scaffold'.

    Returns:
        tuple: A tuple containing the training, validation, and test
            DeepChem datasets, in that order.

    Raises:
        KeyError: If the specified target columns are missing or if an
            unsupported split strategy is provided.
        AssertionError: If the number of positive samples in the validation
            and test sets differs by more than one.
    """
    task = f'{target}_inhibition'

    # filter compounds not tested on the target of interest
    df = df_copy[~df_copy[f'{target}_label'].isnull()]

    X = np.array(df['dc_graph'])
    y = np.array(df[f"{target}_label"])

    # transform
    dataset = dc.data.NumpyDataset(X = X, y = y)

    # print("Dataset X dtype:", dataset.X.dtype)
    # print("Dataset first item:", type(dataset.X[0]))
    # print("Expected ConvMol:", isinstance(
    #     dataset.X[0], dc.feat.mol_graphs.ConvMol
    # ))
    # take into account imbalance - assigns weights of both classes
    # to be equivalent, even though class labels are very imbalanced
    transformer = dc.trans.BalancingTransformer(dataset = dataset)
    dataset = transformer.transform(dataset)


    if split in ['index', 'random_stratified']:
        # split the data to be in equal proportions through sets
        splitters = {
                    'index': dc.splits.IndexSplitter(),
                    'random stratified': dc.splits.RandomStratifiedSplitter(),
                    'scaffold': dc.splits.ScaffoldSplitter(),
        }

        # default 0.8, 0.1, 0.1 data split
        splitter = splitters[split]
    # split stratified to minimize
        train, valid, test = splitter.train_valid_test_split(dataset)

    elif split == 'scaffold':
        sgkf = StratifiedGroupKFold(n_splits=10, shuffle = True, random_state=42)
        folds = [idx for _, idx in sgkf.split(np.zeros(len(y)), y, groups = df['bm_scaffold'])]
        test_idx, valid_idx = folds[0], folds[1]
        train_idx = np.concatenate(folds[2:])
        train, valid, test = dataset.select(train_idx), dataset.select(valid_idx), dataset.select(test_idx)

    else:
        print("invalid selection")

    # make sure the validation and test sets are eqiuvalent (plus or minus 1)
    # assert valid.y.sum()-1 <= test.y.sum() <= valid.y.sum()+1
    print("DISTRIBUTION of HITS")
    print(f"Train: {train.y.sum()}")
    print(f"Validation: {valid.y.sum()}")
    print(f"Test: {test.y.sum()}")

    return train, valid, test


def confirm_split(train, valid, test, target: str) -> None:
    """
    Validate and report the distribution of positive samples across datasets.

    Checks whether the number of positive samples in the training,
    validation, and test datasets follows an expected 80/10/10 split.
    Prints the total number of positive samples and the distribution
    across the three datasets.

    Args:
        train: Training dataset containing binary labels in its `y` attribute.
        valid: Validation dataset containing binary labels in its `y` attribute.
        test: Test dataset containing binary labels in its `y` attribute.
        target (str): Name of the target being evaluated, used in the
            training distribution heading.

    Returns:
        None: Prints dataset distribution information and raises an
            AssertionError if the expected positive sample counts do not match.
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
    Initialize a dictionary to store model parameters and evaluation results.

    Organizes the specified task, model hyperparameters, and validation
    labels into a structured dictionary. Initializes empty dictionaries
    for storing general and best model results.

    Args:
        task (str): Name of the prediction task.
        model_params (dict): Dictionary containing model hyperparameters,
            including probability_threshold, graph_conv_layers, epochs,
            batch_normalize, dropout, dense_layer_size, and batch_size.
        validation_labels (list): True labels for the validation dataset,
            used for model evaluation.

    Returns:
        dict: Dictionary containing three top-level keys:
            - 'model_parameters': Task name, model hyperparameters, and
              validation true labels.
            - 'results': Empty dictionary for storing evaluation results.
            - 'best_results': Empty dictionary for storing the best
              evaluation results.
    """
    model_parameters= {
            'model_parameters': {
                'task': task,
                'probability_threshold': model_params["probability_threshold"],
                'graph_conv_layers': model_params["graph_conv_layers"],
                'epochs': model_params["epochs"],
                'batch_normalize': model_params['batch_normalize'],
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
    """
    Custom JSON encoder for NumPy data types.

    Extends json.JSONEncoder to support serialization of NumPy integers,
    floating-point numbers, and arrays by converting them into native Python
    types that are compatible with JSON serialization.

    Methods:
        default(obj): Converts supported NumPy objects into JSON-serializable
            Python objects. Delegates unsupported types to the default
            implementation of json.JSONEncoder, which raises a TypeError
            if the object is not serializable.
    """
    def default(self, obj):
        """
        Convert NumPy objects into JSON-serializable Python types.

        Args:
            obj: Object to serialize.

        Returns:
            int, float, or list: Converted value for NumPy integers,
                floating-point numbers, or arrays, respectively.

        Raises:
            TypeError: If the object is not serializable by the default
                JSON encoder.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
