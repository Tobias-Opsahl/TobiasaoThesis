import logging
import os
import pickle
import random
import shutil

import numpy as np
import torch

from src.common.path_utils import load_data_list_cub, load_hyperparameters_cub, load_hyperparameters_shapes
from src.constants import (MODEL_STRINGS_ALL_CUB, MODEL_STRINGS_ALL_SHAPES, MODEL_STRINGS_CUB, MODEL_STRINGS_SHAPES,
                           N_ATTR_CUB, N_CLASSES_CUB)
from src.models.models_cub import CubCBM, CubCBMWithResidual, CubCBMWithSkip, CubCNN, CubLogisticOracle, CubNNOracle
from src.models.models_shapes import (ShapesCBM, ShapesCBMWithResidual, ShapesCBMWithSkip, ShapesCNN,
                                      ShapesLogisticOracle, ShapesNNOracle, ShapesSCM)
from src.models.resnet_scm import ResNet18SCM


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
        model_type (str): The model to load. Must be in ["cnn", "cbm", "cbm_res", "cbm_skip", "scm"]
        n_classes (int): The amount of classes used in the dataset.
        n_attr (int): The amount of attributes used in the dataset.
        hyperparameters (dict): The hyperparameters for the model. Can be loaded with `load_hyperparameters_shapes()`.

    Raises:
        ValueError: If model_type is not supported

    Returns:
        model: The pytorch model.
    """
    model_type = model_type.strip().lower()
    if model_type not in MODEL_STRINGS_ALL_SHAPES:
        raise ValueError(f"The model type must be in {MODEL_STRINGS_ALL_SHAPES}. Was {model_type}. ")
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

    elif model_type == "lr_oracle":
        lr_oracle = ShapesLogisticOracle(n_classes=n_classes, n_attr=n_attr)
        return lr_oracle

    elif model_type == "nn_oracle":
        lr_oracle = ShapesNNOracle(n_classes=n_classes, n_attr=n_attr)
        return lr_oracle


def load_models_shapes(n_classes, n_attr, signal_strength=98, n_subset=None, model_strings=None, hyperparameters=None,
                       hard_bottleneck=None, fast=False):
    """
    Loads the shapes models with respect to the hyperparameters.
    The classes and attributes must be equal for all the models, but the hyperparameters
    (like nodes in layers and such) can be different.

    Args:
        n_classes (int): Amount of classes.
        n_attr (int): Amount of attribues.
        signal_strength (int). The signal-strength used for creating the dataset.
        n_subset (int): The amount of data used to train the model. Used for loading hyperparameters.
        model_strings (list of str): List of names of models to load, from `src.constants.py`.
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
    models = []
    if model_strings is None:
        model_strings = MODEL_STRINGS_SHAPES
    for model_string in model_strings:
        model = load_single_model(model_type=model_string, hyperparameters=hp[model_string],
                                  n_classes=n_classes, n_attr=n_attr)
        models.append(model)

    return models


def add_histories(histories_total, histories_new, model_strings):
    """
    Add training new bootstrap iteration of training histories to existing list of training histories.
    Histories should be from one of the trianing functions, and may have added test-accuracy.

    Args:
        histories_total (dict): Accumulated dictionary of histories
        histories_new (dict): New dictionary of training history.
        model_strings (list of str): List of model names, which should be keys in the history-dicts.

    Returns:
        dict: The merged histories
    """
    for model_string in model_strings:
        for key in histories_new[model_string]:
            if key not in histories_total[model_string]:  # First iteration
                histories_total[model_string][key] = [histories_new[model_string][key]]
            else:  # Already in the histories_total
                histories_total[model_string][key].append(histories_new[model_string][key])
    return histories_total


def find_class_imbalance(n_subset=None, multiple_attr=False):
    """
    From the CBM code.
    Calculate class imbalance ratio for the binary attribute labels.
    This means the imbalance between positive and negative concepts (there are a lot more 0s than 1s).
    If `multiple_attr` is `True`, then the imbalance is different for each concept. If not, it is the average of all.

    Args:
        n_subset (int): The amount of images used for each class in the subset. If `None`, will use all data.
        multiple_attr (bool): If `True`, will make a separate weight for each concept. If `False`, will average
            across all concepts

    Returns:
        list: List of imbalance-weights for the attributes.
    """
    imbalance_ratio = []
    data = load_data_list_cub("train", n_subset)
    n = len(data)
    n_attr = len(data[0]["attribute_label"])
    if multiple_attr:
        n_ones = [0] * n_attr
        total = [n] * n_attr
    else:
        n_ones = [0]
        total = [n * n_attr]
    for d in data:
        labels = d["attribute_label"]
        if multiple_attr:
            for i in range(n_attr):
                n_ones[i] += labels[i]
        else:
            n_ones[0] += sum(labels)

    for j in range(len(n_ones)):
        imbalance_ratio.append(total[j] / n_ones[j] - 1)
    if not multiple_attr:  # e.g. [9.0] --> [9.0] * 312
        imbalance_ratio *= n_attr
    return imbalance_ratio


def load_single_model_cub(model_type, hyperparameters, n_attr=112):
    """
    Loads a model, given its hyperparameters.

    Args:
        model_type (str): The model to load. Must be in ["cnn", "cbm", "cbm_res", "cbm_skip"]
        hyperparameters (dict): The hyperparameters for the model. Can be loaded with `load_hyperparameters_shapes()`.
        n_attr (int): The amount of attributes used in the dataset.

    Raises:
        ValueError: If model_type is not supported

    Returns:
        model: The pytorch model.
    """
    model_type = model_type.strip().lower()
    if model_type not in MODEL_STRINGS_ALL_CUB:
        raise ValueError(f"The model type must be in {MODEL_STRINGS_ALL_CUB}. Was {model_type}. ")
    hp = hyperparameters

    if model_type == "cnn":
        cnn = CubCNN(
            n_classes=N_CLASSES_CUB, n_linear_output=hp["n_linear_output"],
            dropout_probability=hp["dropout_probability"])
        return cnn

    if hp.get("hard") is None:  # This hp was added later, so add this for backwards compability
        hp["hard"] = False

    if model_type == "cbm":
        cbm = CubCBM(
            n_classes=N_CLASSES_CUB, n_attr=n_attr, n_linear_output=hp["n_linear_output"],
            attribute_activation_function=hp["activation"], hard=hp["hard"], two_layers=hp["two_layers"],
            dropout_probability=hp["dropout_probability"])
        return cbm

    elif model_type == "cbm_res":
        cbm_res = CubCBMWithResidual(
            n_classes=N_CLASSES_CUB, n_attr=n_attr, n_linear_output=hp["n_linear_output"],
            attribute_activation_function=hp["activation"], hard=hp["hard"],
            dropout_probability=hp["dropout_probability"])
        return cbm_res

    elif model_type == "cbm_skip":
        cbm_skip = CubCBMWithSkip(
            n_classes=N_CLASSES_CUB, n_attr=n_attr, n_hidden=hp["n_hidden"], n_linear_output=hp["n_linear_output"],
            attribute_activation_function=hp["activation"], hard=hp["hard"],
            dropout_probability=hp["dropout_probability"])
        return cbm_skip

    elif model_type == "scm":
        scm = ResNet18SCM(
            n_classes=N_CLASSES_CUB, n_attr=n_attr, n_hidden=hp["n_hidden"], n_linear_output=hp["n_linear_output"],
            attribute_activation_function=hp["activation"], hard=hp["hard"],
            dropout_probability=hp["dropout_probability"])
        return scm

    elif model_type == "lr_oracle":
        lr_oracle = CubLogisticOracle(n_classes=N_CLASSES_CUB, n_attr=n_attr)
        return lr_oracle

    elif model_type == "nn_oracle":
        lr_oracle = CubNNOracle(n_classes=N_CLASSES_CUB, n_attr=n_attr)
        return lr_oracle


def load_models_cub(model_strings=None, n_subset=None, hyperparameters=None, hard_bottleneck=None,
                    fast=False, n_attr=N_ATTR_CUB):
    """
    Loads the cub models with respect to the hyperparameters.

    Args:
        model_strings (list of str): List of names of models to load, from `src.constants.py`.
        n_subset (int): The amount of data used to train the model. Used for loading hyperparameters.
        hyperparameters (dict of dict, optional): Dictionary of the hyperparameter-dictionaries.
            Should be read from yaml file in "hyperparameters/". If `None`, will read
            default or fast hyperparameters. Defaults to None.
        hard_bottleneck (bool): If True, will load hard-bottleneck concept layer for the concept models.
            If not, will use what is in the hyperparameters.
        fast (bool, optional): If True, will load hyperparameters with very low `n_epochs`. This
            can be used for fast testing of the code. Defaults to False.
        n_attr (int): Amount of attribues.

    Returns:
        list of models: List of the models loaded.
    """
    hp = hyperparameters
    if hp is None:
        hp = load_hyperparameters_cub(n_subset, fast=fast, hard_bottleneck=hard_bottleneck)

    models = []
    if model_strings is None:
        model_strings = MODEL_STRINGS_CUB
    for model_string in model_strings:
        model = load_single_model_cub(model_type=model_string, hyperparameters=hp[model_string], n_attr=n_attr)
        models.append(model)

    return models


def count_parameters(model):
    """
    Returns parameters (trainable and not) of a model.

    Args:
        model (pytorch model): The model to count parameters for.

    Returns:
        (int, int, int): Total parameters, trainable parameters, non-trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    return total_params, trainable_params, frozen_params
