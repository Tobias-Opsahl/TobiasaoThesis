import os
import yaml
import torch
import pickle
import shutil
import warnings
import matplotlib.pyplot as plt
from pathlib import Path

from src.constants import (RESULTS_FOLDER, HYPERPARAMETERS_FOLDER, HISTORY_FOLDER, PLOTS_FOLDER, SAVED_MODELS_FOLDER,
                           DATA_FOLDER, SHAPES_FOLDER, CUB_FOLDER, FAST_HYPERPARAMETERS_FILENAME_SHAPES,
                           DEFAULT_HYPERPARAMETERS_FILENAME_SHAPES)


def _check_just_file(filename):
    """
    Checks that `filename` does not contain a folder, for example `plots/plot.png`. Raises ValueError if it does.
    Also checks that `filename` is either `str` or `pathtlib.Path`.

    Args:
        filename (str or pathlib.Path): The filename to check.

    Raises:
        ValueError: If `filename` contains a folder.
        TypeError: If `filename` is not of type `str` or `pathlib.Path`.
    """
    message = f"Filename must not be inside a directory, but only be the filename. Was {filename}. "
    if isinstance(filename, Path):
        filename = filename.as_posix()  # Convert to string
    if not isinstance(filename, str):
        raise TypeError(f"Filename must be of type `str` or `pathlib.Path`. Was {type(filename)}. ")
    if "/" in filename or "\\" in filename:
        raise ValueError(message)


def create_folder(folder_path, exist_ok=True):
    """
    Create a folder.

    Args:
        folder_path (str or pathlib.Path): The folder-path, including the foldername.
        exist_ok (bool, optional): If True, will not raise Exception if folder already exists. Defaults to True.
    """
    os.makedirs(folder_path, exist_ok=exist_ok)


def make_file_path(folder, filename, check_folder_exists=True):
    """
    Merges a path to a folder `folder` with a filename `filename`.
    If `check_folder_exists` is True, will create the folder `folder` if it is not there.
    Argument `filename` can not be inside a folder, it can only be a filename (to ensure that the correct and full
    folder path gets created).

    Args:
        folder (str or pathlib.Path): Path to the folder.
        filename (str or pathlib.Path): Filename of the file. Must not be inside a folder, for example `plots/plot.png`
            is not allowed, `plots/` should be a part of `folder`.
        check_folder_exists (bool): If `True`, will check that `folder` exists, and create it if it does not.

    Returns:
        pathlib.Path: The merged path.
    """
    _check_just_file(filename)  # Check that filename does not have a folder.
    folder_path = Path(folder)
    if check_folder_exists:
        create_folder(folder_path, exist_ok=True)
    file_path = folder_path / filename
    return file_path


def get_shapes_folder_name(n_classes, n_attr, signal_strength=98):
    """
    Get the folder-name according to the basic shapes folder-name structure.
    This is for example "c10_a5_s100", where 10 is the amount of classes, 5 is the amount of attributes and
    100 is the signal-strength.

    Args:
        n_classes (int): The amount of classes.
        n_attr (int): The amount of attributes (concepts)
        signal_strength (int, optional): The signal-strength. Defaults to 98.

    Returns:
        pathlib.Path: The foldername according to the arguments.
    """
    folder_name = Path(f"c{n_classes}_a{n_attr}_s{signal_strength}/")
    return folder_name


def get_dataset_folder(dataset_name):
    """
    Returns the foldername of the dataset.

    Args:
        dataset_name (str): The name of the dataset. Must be in ["shapes", "cub"].

    Raises:
        ValueError: If dataset_name is not in ["shapes", "cub"].

    Returns:
        pathlib.Path: The folder of the dataset. Is loaded from `constants.py`.
    """
    dataset_name = dataset_name.lower().strip().replace("/", "").replace("\\", "")
    if dataset_name == "shapes":
        return Path(SHAPES_FOLDER)
    elif dataset_name == "cub":
        return Path(CUB_FOLDER)
    raise ValueError(f"Argument `dataset_name` must be in [\"shapes\", \"cub\"]. Was {dataset_name}. ")


def get_full_shapes_folder_path(n_classes, n_attr, signal_strength, relative_folder):
    """
    Returns the full nested structure of a shapes folder, inside one of the results-folders.

    Args:
        n_classes (int): The amount of classes.
        n_attr (int): The amount of attributes (concepts)
        signal_strength (int, optional): The signal-strength.
        relative_folder (str or pathlib.Path): The name of the specific results folder.
            Must be in one of the paths from constants.py.

    Returns:
        pathlib.Path: The full path of the folder
    """
    base_folder = Path(RESULTS_FOLDER)
    dataset_folder = get_dataset_folder("shapes")
    folder_name = get_shapes_folder_name(n_classes=n_classes, n_attr=n_attr, signal_strength=signal_strength)
    full_shapes_folder = base_folder / relative_folder / dataset_folder / folder_name
    return full_shapes_folder


def get_shapes_dataset_path(n_classes, n_attr, signal_strength, check_already_exists=True):
    """
    Given classes, attributes and the signal-strength, makes the path to a shapes-dataset.
    This does not create the dataset, so it is assumed that it is already created (with `make_shapes_datasets.py`).
    Datasets should be stored in "data/shapes".
    Names of the dataset are on the form "shapes_1k_c10_a5_s100" for 1k images in each class, 10 classes, 5 attributes
    and signal-strength 100. Note that all datasets has 1k images in each class, except from 10-classes 5-attributes
    signal-strenght 98, which has 2k. Also note that datasets with signal-strength 98 will omit the signal-strength
    in the name, for example "shapes_1k_c21_a9" for 21 classes, 9 attributes and signal_strength 98.

    Args:
        n_classes (int): The amount of classes in the dataset.
        n_attr (int): The amonut of attributes (concepts) in the dataset.
        signal_strength (int, optional): Signal-strength used to make dataset.
        check_already_exists (bool, optional): If True, will raise ValueError if dataset does not exist.
            Defaults to True.

    Raises:
        ValueError: If `check_already_exists` is True and dataset_path does not exit.

    Returns:
        pathlib.Path: Path to the dataset.
    """
    base_path = Path(DATA_FOLDER + SHAPES_FOLDER)
    if n_classes == 10 and n_attr == 5 and signal_strength == 98:
        base_folder_name = Path("shapes_2k_")  # Dataset c10_a5 (s98) has 2k images of each class. The rest has 1k.
    else:
        base_folder_name = Path("shapes_1k_")

    signal_string = "" if signal_strength == 98 else f"_s{signal_strength}"
    folder_name = Path(f"{base_folder_name}c{n_classes}_a{n_attr}{signal_string}/")
    dataset_path = base_path / folder_name
    if check_already_exists:
        if not dataset_path.exists():
            message = f"Shapes dataset for {n_classes=}, {n_attr=}, {signal_strength=} does not exist. "
            message += f"Check if there were a typo, or create the dataset with `src/datasets/make_shapes_datasets.py. "
            raise ValueError(message)
    return dataset_path


def load_data_list_shapes(n_classes, n_attr, signal_strength, n_subset=None, mode="train"):
    """
    Loads a data-list, a list of dictionaries saved as a pickle file with labels and images-paths for the dataset.

    Args:
        n_classes (int): The amount of classes in the dataset.
        n_attr (int): The amonut of attributes (concepts) in the dataset.
        signal_strength (int, optional): Signal-strength used to make dataset.
        n_subset (int, optional): The subset of data to load data-list for. If `None`, will use full dataset.
        mode (str, optional): Should be in ["train", "val", "test"]. Determines which
            data-list to load. Defaults to "train".

    Returns:
        list: The data list
    """
    dataset_path = get_shapes_dataset_path(n_classes, n_attr, signal_strength)
    filename = Path(f"{mode}_data.pkl")
    if n_subset is not None:
        data_list_path = dataset_path / "tables/" / f"sub{n_subset}" / filename
    else:  # Load the full dataset-data-list
        data_list_path = dataset_path / "tables/" / filename
    try:
        data_list = pickle.load(open(data_list_path, "rb"))
    except Exception as e:
        message = f"Data-list {data_list_path} not found. Try creating datalist with make_subset_shapes() from "
        message += f"src.datasets.datasets_shapes.py, or create datasets with src.datasets.make_shapes_datasets.py. "
        raise FileNotFoundError(message)
    return data_list


def write_data_list_shapes(n_classes, n_attr, signal_strength, n_subset, train_data, val_data):
    """
    Given a train and validation data-list for a subset of a shapes-dataset, saves or overwrites the data-lists.
    Args:
        n_classes (int): The amount of classes in the dataset.
        n_attr (int): The amonut of attributes (concepts) in the dataset.
        signal_strength (int, optional): Signal-strength used to make dataset.
        n_subset (int, optional): The subset of data to load data-list for. If `None`, will use full dataset.
        train_data (list): The list of the train-data dictionaries
        val_data (list): The list of the validation-data dictionaries

    Returns:
        list: The data list
    """
    dataset_path = get_shapes_dataset_path(n_classes, n_attr, signal_strength)
    tables_path = dataset_path / "tables" / f"sub{n_subset}"
    if os.path.exists(tables_path):
        shutil.rmtree(tables_path)  # Delete previous folder and re-create
    os.makedirs(tables_path)
    train_filename = make_file_path(tables_path, "train_data.pkl")
    val_filename = make_file_path(tables_path, "val_data.pkl")
    with open(train_filename, "wb") as outfile:
        pickle.dump(train_data, outfile)
    with open(val_filename, "wb") as outfile:
        pickle.dump(val_data, outfile)


def load_history_shapes(n_classes, n_attr, signal_strength, n_bootstrap, n_subset):
    """
    Load models history, made by `src/evaluation.py`.

    Args:
        n_classes (int): The amount of classes in the dataset.
        n_attr (int): The amonut of attributes (concepts) in the dataset.
        signal_strength (int, optional): Signal-strength used to make dataset.
        n_bootstrap (int): The amount of bootstrap iterations that were used.
        n_subset (int): The amount of instances in each class used in the subset.

    Returns:
        dict: Dictionary of histories.
    """
    folder_name = get_full_shapes_folder_path(n_classes, n_attr, signal_strength, relative_folder=HISTORY_FOLDER)
    filename = Path(f"histories_sub{n_subset}_b{n_bootstrap}.pkl")
    file_path = make_file_path(folder_name, filename, check_folder_exists=False)
    history = pickle.load(open(file_path, "rb"))
    return history


def save_history_shapes(n_classes, n_attr, signal_strength, n_bootstrap, n_subset, history):
    """
    Save a history dictionary, made by `src/evaluation.py`.

    Args:
        n_classes (int): The amount of classes in the dataset.
        n_attr (int): The amonut of attributes (concepts) in the dataset.
        signal_strength (int, optional): Signal-strength used to make dataset.
        n_bootstrap (int): The amount of bootstrap iterations that were used.
        n_subset (int): The amount of instances in each class used in the subset.
        history (dict): The histories to save.
    """
    folder_name = get_full_shapes_folder_path(n_classes, n_attr, signal_strength, relative_folder=HISTORY_FOLDER)
    filename = Path(f"histories_sub{n_subset}_b{n_bootstrap}.pkl")
    file_path = make_file_path(folder_name, filename, check_folder_exists=True)
    with open(file_path, "wb") as outfile:
        pickle.dump(history, outfile)


def load_model_shapes(n_classes, n_attr, signal_strength, n_subset, model_type, metric="loss"):
    """
    Load a state_dict for a model.

    Args:
        n_classes (int): The amount of classes in the dataset.
        n_attr (int): The amonut of attributes (concepts) in the dataset.
        signal_strength (int, optional): Signal-strength used to make dataset.
        n_subset (int): The amount of instances in each class used in the subset.
        model_type (str): Model to load. Must be in ["cnn", "cbm", "cbm_res", "cbm_skip", "scm"].
        metric (str, optional): Should be in ["loss", "accuracy"]. Determines if one loads the best
            validation loss model or best validation accuracy loss. Defaults to "loss".

    Returns:
        dict: The state-dict to the model.
    """
    model_type = model_type.strip().lower()
    if model_type not in ["cnn", "cbm", "cbm_res", "cbm_skip", "scm"]:
        message = f"Argument `model_type` must be in [\"cnn\", \"cbm\", \"cbm_res\", \"cbm_skip\", \"scm\"]. "
        message += f"Was {model_type}. "
        raise ValueError(message)

    folder_name = get_full_shapes_folder_path(n_classes, n_attr, signal_strength, relative_folder=SAVED_MODELS_FOLDER)
    filename = Path(f"{model_type}_sub{n_subset}_{metric}.pth")
    file_path = make_file_path(folder_name, filename, check_folder_exists=False)
    state_dict = torch.load(file_path)
    return state_dict


def save_model_shapes(n_classes, n_attr, signal_strength, n_subset, state_dict, model_type, metric="loss"):
    """
    Saves a models state_dict.

    Args:
        n_classes (int): The amount of classes in the dataset.
        n_attr (int): The amonut of attributes (concepts) in the dataset.
        signal_strength (int, optional): Signal-strength used to make dataset.
        n_subset (int): The amount of instances in each class used in the subset.
        state_dict (dict): The state-dict of the model to save.
        model_type (str): String name of the model to save. Must be in ["cnn", "cbm", "cbm_res", "cbm_skip", "scm"].
        metric (str, optional): Nam. Defaults to "loss".
    """
    model_type = model_type.strip().lower()
    if model_type not in ["cnn", "cbm", "cbm_res", "cbm_skip", "scm"]:
        message = f"Argument `model_type` must be in [\"cnn\", \"cbm\", \"cbm_res\", \"cbm_skip\", \"scm\"]. "
        message += f"Was {model_type}. "
        raise ValueError(message)

    folder_name = get_full_shapes_folder_path(n_classes, n_attr, signal_strength, relative_folder=SAVED_MODELS_FOLDER)
    filename = Path(f"{model_type}_sub{n_subset}_{metric}.pth")
    file_path = make_file_path(folder_name, filename, check_folder_exists=True)
    torch.save(state_dict, file_path)


def save_training_plot_shapes(n_classes, n_attr, signal_strength, n_bootstrap, n_subset, attr=False):
    """
    Save a training history plot.

    Args:
        n_classes (int): The amount of classes in the dataset.
        n_attr (int): The amonut of attributes (concepts) in the dataset.
        signal_strength (int, optional): Signal-strength used to make dataset.
        n_bootstrap (int): The amount of bootstrap iterations that were used.
        n_subset (int): The amount of instances in each class used in the subset.
        attr (bool, optional): Set to True if the training plot is of the attributes / concepts. Defaults to False.
    """
    folder_name = get_full_shapes_folder_path(n_classes, n_attr, signal_strength, relative_folder=PLOTS_FOLDER)
    attr_string = "" if not attr else "attr_"
    filename = Path(f"training_history_{attr_string}sub{n_subset}_b{n_bootstrap}.pdf")
    file_path = make_file_path(folder_name, filename, check_folder_exists=True)
    plt.savefig(file_path)
    plt.close()


def save_test_plot_shapes(n_classes, n_attr, signal_strength, n_bootstrap):
    """
    Saves a test-accuracies plot.

    Args:
        n_classes (int): The amount of classes in the dataset.
        n_attr (int): The amonut of attributes (concepts) in the dataset.
        signal_strength (int, optional): Signal-strength used to make dataset.
        n_bootstrap (int): The amount of bootstrap iterations that were used.
    """
    folder_name = get_full_shapes_folder_path(n_classes, n_attr, signal_strength, relative_folder=PLOTS_FOLDER)
    filename = Path(f"test_accuracies_b{n_bootstrap}.pdf")
    file_path = make_file_path(folder_name, filename, check_folder_exists=True)
    plt.savefig(file_path)
    plt.close()


def load_hyperparameters_shapes(n_classes=None, n_attr=None, signal_strength=None, n_subset=None,
                                fast=False, default=False, hard=False):
    """
    Loads a set of hyperparameters. This will try to load hyperparameters specific for this combination of
    classes, attributes and n_subset. If this is not found, it will use default hyperparameters instead.
    Alternatively, if `fast` is True, it will load parameters with very low `n_epochs`, so that one can test fast.
    Keep in mind that this returns a dict with hyperparameters for all possible models.

    Args:
        n_classes (int): The amount of classes in the dataset.
        n_attr (int): The amonut of attributes (concepts) in the dataset.
        signal_strength (int, optional): Signal-strength used to make dataset.
        n_subset (int): The amount of instances in each class used in the subset.
        fast (bool, optional): If True, will use hyperparameters with very low `n_epochs`. Defaults to False.
        default (bool, optional): If True, will use default hyperparameters, disregarding n_classe, n_attr etc.

    Raises:
        ValueError: If `default` and `fast` are False, `n_classes`, `n_attr` and `n_subset` must be provided.
        Raises ValueError if not.

    Returns:
        dict: The hyperparameters.
    """
    if (not default and not fast) and ((n_classes is None) or (n_attr is None) or (n_subset is None)):
        message = f"When getting non-default or non-fast hyperparameters, arguments `n_classes`, `n_attr` and "
        message += f"`n_subset` must be provided (all of them must be int). Was {n_classes=}, {n_attr=}, {n_subset=}. "
        raise ValueError(message)

    results_folder = Path(RESULTS_FOLDER)
    base_folder = results_folder / HYPERPARAMETERS_FOLDER / SHAPES_FOLDER
    if fast:  # Here n_epohcs = 2, for fast testing
        fast_path = make_file_path(base_folder, FAST_HYPERPARAMETERS_FILENAME_SHAPES)
        with open(fast_path, "r") as infile:
            hyperparameters = yaml.safe_load(infile)
        return hyperparameters

    folder_name = get_full_shapes_folder_path(
        n_classes, n_attr, signal_strength, relative_folder=HYPERPARAMETERS_FOLDER)
    filename = Path(f"hyperparameters_sub{n_subset}.yaml")
    file_path = make_file_path(folder_name, filename, check_folder_exists=False)

    if (not default) and (not os.path.exists(file_path)):  # No hp for this class-attr-sub
        message = f"No hyperparameters found for {n_classes=}, {n_attr=} and {n_subset=}. "
        message += "Falling back on default hyperparameters. "
        warnings.warn(message)
        default = True

    if default:
        default_path = base_folder / DEFAULT_HYPERPARAMETERS_FILENAME_SHAPES
        with open(default_path, "r") as infile:
            hyperparameters = yaml.safe_load(infile)
        return hyperparameters

    with open(file_path, "r") as infile:
        hyperparameters = yaml.safe_load(infile)
    return hyperparameters


def save_hyperparameters_shapes(n_classes, n_attr, signal_strength, n_subset, hyperparameters_dict, model_type):
    """
    Saves a set of hyperparameters. Hyperparameters should be of single model, with name `model_type`.

    Args:
        n_classes (int): The amount of classes in the dataset.
        n_attr (int): The amonut of attributes (concepts) in the dataset.
        signal_strength (int, optional): Signal-strength used to make dataset.
        n_subset (int): The amount of instances in each class used in the subset.
        hyperparameters_dict (dict): The hyperparameters to save.
        model_type (str): The name of the model with the given hyperparameters
    """
    if model_type not in ["cnn", "cbm", "cbm_res", "cbm_skip", "scm"]:
        message = f"Argument `model_type` must be in [\"cnn\", \"cbm\", \"cbm_res\", \"cbm_skip\", \"scm\"]. "
        message += f"Was {model_type}. "
        raise ValueError(message)
    folder_name = get_full_shapes_folder_path(
        n_classes, n_attr, signal_strength, relative_folder=HYPERPARAMETERS_FOLDER)
    filename = Path(f"hyperparameters_sub{n_subset}.yaml")
    file_path = make_file_path(folder_name, filename, check_folder_exists=False)
    if not os.path.exists(file_path):  # No yaml file for this setting
        os.makedirs(folder_name, exist_ok=True)
        with open(file_path, "w") as outfile:
            hyperparameters_full = {model_type: hyperparameters_dict}
            yaml.dump(hyperparameters_full, outfile, default_flow_style=False)
        return

    # File already exists. We have to read, overwrite only this model_type, and write again.
    with open(file_path, "r") as yaml_file:
        hyperparameters_full = yaml.safe_load(yaml_file)
    if hyperparameters_full is None:
        hyperparameters_full = {}
    hyperparameters_full[model_type] = hyperparameters_dict  # Overwrite or create new dict of model_type only
    with open(file_path, "w") as yaml_file:
        yaml.dump(hyperparameters_full, yaml_file, default_flow_style=False)
