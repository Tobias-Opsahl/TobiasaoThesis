import os
import torch
import shutil
import random
import pickle
import logging
import numpy as np

from src.common.path_utils import load_hyperparameters_shapes
from src.models.models_shapes import ShapesCNN, ShapesCBM, ShapesCBMWithResidual, ShapesCBMWithSkip, ShapesSCM
from src.constants import MODEL_STRINGS


def set_global_log_level(level):
    LOG_LEVEL_MAP = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }
    if isinstance(level, str):
        level = level.strip().lower()
        level = LOG_LEVEL_MAP[level]
    logging.getLogger().setLevel(level)
    logging.StreamHandler().setLevel(level)


def get_logger(name):
    """
    Get a logger with the given name.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.hasHandlers():  # Only if logger has not been set up before
        ch = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def seed_everything(seed=57):
    """
    Set all the seeds.

    Args:
        seed (int, optional): The seed to set to. Defaults to 57.
    """
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_int_list(values):
    """
    Tries to parse from string with ints to list of ints. If it does not work, parses list of single int.

    Args:
        values (str): The string to pars

    Returns:
        ist of int: The list of ints
    """
    try:
        return [int(val) for val in values.split(",")]
    except ValueError:
        return [int(values)]


def split_dataset(data_list, tables_dir, include_test=True, seed=57):
    """
    Splits a dataset into "train", "validation" and "test".

    Args:
        data_list (list): List of rows, each element is an instance-dict.
        tables_dir (str): The path to save the data-tables.
        include_test (bool): If True, will split in train, val and test. If False, will split in train and val.
        seed (int, optional): Seed for the rng. Defaults to 57.
    """
    random.seed(seed)
    n_images = len(data_list)
    random.shuffle(data_list)
    if include_test:
        train_size = int(0.5 * n_images)
        val_size = int(0.3 * n_images)
    else:
        train_size = int(0.6 * n_images)

    train_data = data_list[: train_size]
    if include_test:
        val_data = data_list[train_size: train_size + val_size]
        test_data = data_list[train_size + val_size:]
    else:
        val_data = data_list[train_size:]

    if os.path.exists(tables_dir):
        shutil.rmtree(tables_dir)  # Delete previous folder and re-create
    os.makedirs(tables_dir)
    with open(tables_dir + "train_data.pkl", "wb") as outfile:
        pickle.dump(train_data, outfile)
    with open(tables_dir + "val_data.pkl", "wb") as outfile:
        pickle.dump(val_data, outfile)
    if include_test:
        with open(tables_dir + "test_data.pkl", "wb") as outfile:
            pickle.dump(test_data, outfile)


def load_single_model(model_type, n_classes, n_attr, hyperparameters):
    """
    Loads a model, given its hyperparameters.

    Args:
        model_type (str): The model to load. Must be in ["cnn", "cbm", "cbm_res", "cbm_skip"]
        n_classes (int): The amount of classes used in the dataset.
        n_attr (int): The amount of attributes used in the dataset.
        hyperparameters (dict): The hyperparameters for the model. Can be loaded with `load_hyperparameters_shapes()`.

    Raises:
        ValueError: If model_type is not supported

    Returns:
        model: The pytorch model.
    """
    model_type = model_type.strip().lower()
    if model_type not in MODEL_STRINGS:
        raise ValueError(f"The model type must be in {MODEL_STRINGS}. Was {model_type}. ")
    hp = hyperparameters

    if model_type == "cnn":
        cnn = ShapesCNN(
            n_classes=n_classes, n_linear_output=hp["n_linear_output"],
            dropout_probability=hp["dropout_probability"])
        return cnn

    if hp.get("hard") is None:  # This hp was added later, so add this for backwards compability
        hp["hard"] = False

    if model_type == "cbm":
        cbm = ShapesCBM(
            n_classes=n_classes, n_attr=n_attr, n_linear_output=hp["n_linear_output"],
            attribute_activation_function=hp["activation"], hard=hp["hard"], two_layers=hp["two_layers"],
            dropout_probability=hp["dropout_probability"])
        return cbm

    elif model_type == "cbm_res":
        cbm_res = ShapesCBMWithResidual(
            n_classes=n_classes, n_attr=n_attr, n_linear_output=hp["n_linear_output"],
            attribute_activation_function=hp["activation"], hard=hp["hard"],
            dropout_probability=hp["dropout_probability"])
        return cbm_res

    elif model_type == "cbm_skip":
        cbm_skip = ShapesCBMWithSkip(
            n_classes=n_classes, n_attr=n_attr, n_hidden=hp["n_hidden"], n_linear_output=hp["n_linear_output"],
            attribute_activation_function=hp["activation"], hard=hp["hard"],
            dropout_probability=hp["dropout_probability"])
        return cbm_skip

    elif model_type == "scm":
        scm = ShapesSCM(
            n_classes=n_classes, n_attr=n_attr, n_hidden=hp["n_hidden"], n_linear_output=hp["n_linear_output"],
            attribute_activation_function=hp["activation"], hard=hp["hard"],
            dropout_probability=hp["dropout_probability"])
        return scm


def load_models_shapes(n_classes, n_attr, signal_strength=98, n_subset=None, hyperparameters=None,
                       hard_bottleneck=None, fast=False):
    """
    Loads the shapes model with respect to the hyperparameters.
    The classes and attributes must be equal for all the models, but the hyperparameters
    (like nodes in layers and such) can be different.

    Args:
        n_classes (int): Amount of classes.
        n_attr (int): Amount of attribues.
        signal_strength (int). The signal-strength used for creating the dataset.
        n_subset (int): The amount of data used to train the model. Used for loading hyperparameters.
        hyperparameters (dict of dict, optional): Dictionary of the hyperparameter-dictionaries.
            Should be read from yaml file in "hyperparameters/". If `None`, will read
            default or fast hyperparameters. Defaults to None.
        hard_bottleneck (bool): If True, will load hard-bottleneck concept layer for the concept models.
            If not, will use what is in the hyperparameters.
        fast (bool, optional): If True, will load hyperparameters with very low `n_epochs`. This
            can be used for fast testing of the code. Defaults to False.

    Returns:
        list of models: List of the models loaded.
    """
    hp = hyperparameters
    if hp is None:
        hp = load_hyperparameters_shapes(n_classes, n_attr, signal_strength, n_subset,
                                         fast=fast, hard_bottleneck=hard_bottleneck)
    cnn = load_single_model(model_type="cnn", hyperparameters=hp["cnn"], n_classes=n_classes, n_attr=n_attr)
    cbm = load_single_model(model_type="cbm", hyperparameters=hp["cbm"], n_classes=n_classes, n_attr=n_attr)
    cbm_res = load_single_model(model_type="cbm_res", hyperparameters=hp["cbm_res"], n_classes=n_classes, n_attr=n_attr)
    cbm_skip = load_single_model(model_type="cbm_skip", hyperparameters=hp["cbm_skip"],
                                 n_classes=n_classes, n_attr=n_attr)
    scm = load_single_model(model_type="scm", hyperparameters=hp["scm"], n_classes=n_classes, n_attr=n_attr)

    models = [cnn, cbm, cbm_res, cbm_skip, scm]
    return models


def add_histories(histories_total, histories_new, n_bootstrap):
    """
    Adds together two training histories and averages. This means looping over the dictionaries and summing them,
    and also dividing new sum by `n_bootstrap` to average over runs.
    This is also supported for first iteration where `histories_total` is None.

    Args:
        histories_total (dict): First dictionary to add. May be None.
        histories_new (dict): Second dictionary to add.
        n_bootstrap (int): The amount of bootstrap iteration, will divide by this number.

    Returns:
        dict: The added and averaged dictionary.
    """

    metric_keys = ["train_class_loss", "train_class_accuracy", "val_class_loss", "val_class_accuracy",
                   "train_attr_loss", "train_attr_accuracy", "val_attr_loss", "val_attr_accuracy", "test_accuracy"]

    if histories_total is None:  # First iteration. We do not need to add, but must divide.
        for i in range(len(histories_new)):  # loop over each models history
            for key in metric_keys:  # Loop over the keys to use
                if histories_new[i].get(key) is None:
                    continue
                for j in range(len(histories_new[i][key])):
                    histories_new[i][key][j] /= n_bootstrap
        return histories_new

    for i in range(len(histories_new)):  # loop over each models history
        for key in metric_keys:  # Loop over the keys to use
            if histories_total[i].get(key) is None:
                continue
            for j in range(len(histories_new[i][key])):
                histories_total[i][key][j] += histories_new[i][key][j] / n_bootstrap
    return histories_total
