import os
import yaml
import random
import pickle
import warnings
import torch
import numpy as np
from shapes.models_shapes import ShapesCNN, ShapesCBM, ShapesCBMWithResidual, ShapesCBMWithSkip, ShapesSCM
from constants import MODEL_STRINGS


def seed_everything(seed=57):
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def split_dataset(data_list, tables_dir, seed=57):
    """
    Splits a dataset into "train", "validation" and "test".

    Args:
        data_list (list): List of rows, each element is an instance-dict.
        tables_dir (str): The path to save the data-tables
        seed (int, optional): Seed for the rng. Defaults to 57.
    """
    random.seed(seed)
    n_images = len(data_list)
    random.shuffle(data_list)
    train_size = int(0.5 * n_images)
    val_size = int(0.3 * n_images)

    train_data = data_list[: train_size]
    val_data = data_list[train_size: train_size + val_size]
    test_data = data_list[train_size + val_size:]
    with open(tables_dir + "train_data.pkl", "wb") as outfile:
        pickle.dump(train_data, outfile)
    with open(tables_dir + "val_data.pkl", "wb") as outfile:
        pickle.dump(val_data, outfile)
    with open(tables_dir + "test_data.pkl", "wb") as outfile:
        pickle.dump(test_data, outfile)


def get_hyperparameters(n_classes=None, n_attr=None, n_subset=None, default=False, fast=False,
                        base_dir="hyperparameters/", dataset_type="shapes/"):
    """
    Loads a set of hyperparameters. This will try to load hyperparameters specific for this combination of
    classes, attributes and subset. If this is not found, it will use default hyperparameters instead.
    Alternatively, if `fast` is True, it will load parameters with very low `n_epochs`, so that one can test fast.
    Keep in mind that this returns a dict with hyperparameters for all possible models.

    Args:
        n_classes (int): The amount of classes the hyperparameter was optimized with.
        n_attr (int): The amount of attributes the hyperparameter was optimized with.
        n_subset (int): The amount of instances in each class the hyperparameter was optimized with.
        fast (bool, optional): If True, will use hyperparameters with very low `n_epochs`. Defaults to False.
        default (bool, optional): If True, will use default hyperparameters, disregarding n_classe, n_attr etc.
        base_dir (str, optional): Base directory to the hyperparameters.. Defaults to "hyperparameters/".
        dataset_dir (str, optional): The dataset-directory. Defaults to "shapes/".

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

    base_dir = base_dir + dataset_type
    if fast:  # Here n_epohcs = 2, for fast testing
        with open(base_dir + "fast.yaml", "r") as infile:
            hyperparameters = yaml.safe_load(infile)
        return hyperparameters

    folder_name = "c" + str(n_classes) + "_a" + str(n_attr) + "/"
    filename = "hyperparameters_sub" + str(n_subset) + ".yaml"

    if (not default) and (not os.path.exists(base_dir + folder_name + filename)):  # No hp for this class-attr-sub
        message = f"No hyperparameters found for {n_classes=}, {n_attr=} and {n_subset=}. "
        message += "Falling back on default hyperparameters. "
        warnings.warn(message)
        default = True

    if default:
        with open(base_dir + "default.yaml", "r") as infile:
            hyperparameters = yaml.safe_load(infile)
        return hyperparameters

    with open(base_dir + folder_name + filename, "r") as infile:
        hyperparameters = yaml.safe_load(infile)
    return hyperparameters


def load_single_model(model_type, n_classes, n_attr, hyperparameters):
    """
    Loads a model, given its hyperparameters.

    Args:
        model_type (str): The model to load. Must be in ["cnn", "cbm", "cbm_res", "cbm_skip"]
        n_classes (int): The amount of classes used in the dataset.
        n_attr (int): The amount of attributes used in the dataset.
        hyperparameters (dict): The hyperparameters for the model. Can be loaded with `get_hyperparameters()`.

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

    if model_type == "cbm":
        cbm = ShapesCBM(
            n_classes=n_classes, n_attr=n_attr, n_linear_output=hp["n_linear_output"],
            attribute_activation_function=hp["activation"], two_layers=hp["two_layers"],
            dropout_probability=hp["dropout_probability"])
        return cbm

    elif model_type == "cbm_res":
        cbm_res = ShapesCBMWithResidual(
            n_classes=n_classes, n_attr=n_attr, n_linear_output=hp["n_linear_output"],
            attribute_activation_function=hp["activation"], dropout_probability=hp["dropout_probability"])
        return cbm_res

    elif model_type == "cbm_skip":
        cbm_skip = ShapesCBMWithSkip(
            n_classes=n_classes, n_attr=n_attr, n_hidden=hp["n_hidden"], n_linear_output=hp["n_linear_output"],
            attribute_activation_function=hp["activation"], dropout_probability=hp["dropout_probability"])
        return cbm_skip

    elif model_type == "scm":
        hp["n_hidden"] = 16  # TODO: Remove this when new hyperparameters are ran.
        scm = ShapesSCM(
            n_classes=n_classes, n_attr=n_attr, n_hidden=hp["n_hidden"], n_linear_output=hp["n_linear_output"],
            attribute_activation_function=hp["activation"], dropout_probability=hp["dropout_probability"])
        return scm


def load_models_shapes(n_classes, n_attr, n_subset=None, hyperparameters=None, fast=False):
    """
    Loads the shapes model with respect to the hyperparameters.
    The classes and attributes must be equal for all the models, but the hyperparameters
    (like nodes in layers and such) can be different.

    Args:
        n_classes (int): Amount of classes.
        n_attr (int): Amount of attribues.
        n_subset (int): The amount of data used to train the model. Used for loading hyperparameters.
        hyperparameters (dict of dict, optional): Dictionary of the hyperparameter-dictionaries.
            Should be read from yaml file in "hyperparameters/". If `None`, will read
            default or fast hyperparameters. Defaults to None.
        fast (bool, optional): If True, will load hyperparameters with very low `n_epochs`. This
            can be used for fast testing of the code. Defaults to False.

    Returns:
        list of models: List of the models loaded.
    """
    hp = hyperparameters
    if hp is None:
        hp = get_hyperparameters(n_classes, n_attr, n_subset, fast=fast)
    cnn = load_single_model(model_type="cnn", hyperparameters=hp["cnn"], n_classes=n_classes, n_attr=n_attr)
    cbm = load_single_model(model_type="cbm", hyperparameters=hp["cbm"], n_classes=n_classes, n_attr=n_attr)
    cbm_res = load_single_model(model_type="cbm_res", hyperparameters=hp["cbm_res"], n_classes=n_classes, n_attr=n_attr)
    cbm_skip = load_single_model(model_type="cbm_skip", hyperparameters=hp["cbm_skip"],
                                 n_classes=n_classes, n_attr=n_attr)
    scm = load_single_model(model_type="scm", hyperparameters=hp["scm"],
                            n_classes=n_classes, n_attr=n_attr)

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
