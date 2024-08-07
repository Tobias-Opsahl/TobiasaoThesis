import torch
import torch.nn as nn

from src.common.path_utils import load_history_cub, load_hyperparameters_cub, save_history_cub, save_model_cub
from src.common.utils import add_histories, get_logger, load_models_cub, seed_everything
from src.constants import CONCEPT_MODELS_STRINGS_CUB, FAST_MAX_EPOCHS_CUB, MAX_EPOCHS, MODEL_STRINGS_CUB
from src.datasets.datasets_cub import load_data_cub, make_subset_cub
from src.evaluation import evaluate_on_test_set
from src.plotting import plot_mpo_scores_cub, plot_test_accuracies_cub, plot_training_histories_cub
from src.train import train_cbm, train_simple

logger = get_logger(__name__)


def train_and_evaluate_cub(
        n_subset, train_loader, val_loader, test_loader, model_strings=None,
        hyperparameters=None, hard_bottleneck=None, fast=False, device=None, non_blocking=False, seed=None):
    """
    Trains the cub models given some hyperparameters, classes, attributes and data subdirectory.
    The data subdirectory should be generated by `make_subset_cub()`.

    Args:
        n_subset (int): The amount of instances in each class for this subset.
        train_loader (dataloader): The training dataloader.
        val_loader (dataloader): The validation dataloader.
        test_loader (dataloader): The test dataloader.
        model_strings (list of str): List of strings of the models to evaluate. Load from `src.constants.py`.
        hyperparameters (dict of dict, optional): Dictionary of the hyperparameter-dictionaries.
            Should be read from yaml file in "hyperparameters/". If `None`, will read
            default or fast hyperparameters. Defaults to None.
        hard_bottleneck (bool): If True, will load hard-bottleneck concept layer for the concept models.
            If not, will use what is in the hyperparameters.
        fast (bool, optional): If True, will load hyperparameters with very low `n_epochs`. This
        device (str): Use "cpu" for cpu training and "cuda" for gpu training.
            can be used for fast testing of the code. Defaults to False.
        non_blocking (bool): If True, allows for asyncronous transfer between RAM and VRAM.
            This only works together with `pin_memory=True` to dataloader and GPU training.
        n_bootstrap (int): The amount of bootstrap iterations. This is only used to give the saved model unique names,
            so that many different bootstrap iterations can be ran in parallel.
        seed (int): Seed to seed training.

    Returns:
        list of dict: list of model training-histories, including test_accuracy.
    """
    if device is None or device == "":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hp = hyperparameters
    if hp is None:
        hp = load_hyperparameters_cub(n_subset, hard_bottleneck=hard_bottleneck)

    if "lr_oracle" not in hp:  # Add oracle hyperparameters
        default_hp = load_hyperparameters_cub(default=True)
        hp["lr_oracle"] = default_hp["lr_oracle"]
    if "nn_oracle" not in hp:
        default_hp = load_hyperparameters_cub(default=True)
        hp["nn_oracle"] = default_hp["nn_oracle"]

    models = load_models_cub(model_strings=model_strings, hyperparameters=hp)
    histories = {}
    criterion = nn.CrossEntropyLoss()
    attr_criterion = nn.BCEWithLogitsLoss()

    for i in range(len(models)):
        model = models[i]
        model_string = model_strings[i]
        logger.info(f"Running model {model.name}:")
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=hp[model_string]["learning_rate"])
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=hp[model_string]["gamma"])

        n_epochs = MAX_EPOCHS
        if fast:
            n_epochs = FAST_MAX_EPOCHS_CUB
        if seed is not None:  # Seed before training, for reprorudibility
            seed_everything(seed)
        if model_string == "cnn":
            history, models_dict = train_simple(
                model, criterion, optimizer, train_loader, val_loader, scheduler=exp_lr_scheduler, n_epochs=n_epochs,
                n_early_stop=False, device=device, non_blocking=non_blocking)
        elif model_string in ["lr_oracle", "nn_oracle"]:
            history, models_dict = train_simple(
                model, criterion, optimizer, train_loader, val_loader, scheduler=exp_lr_scheduler, n_epochs=n_epochs,
                n_early_stop=False, device=device, non_blocking=non_blocking, oracle=True)
        else:  # Concept model
            history, models_dict = train_cbm(
                model, criterion, attr_criterion, optimizer, train_loader, val_loader, n_epochs=n_epochs,
                attr_weight=hp[model_string]["attr_weight"], scheduler=exp_lr_scheduler, n_early_stop=False,
                device=device, non_blocking=non_blocking)

        state_dict = models_dict["best_model_loss_state_dict"]
        save_model_cub(n_subset, state_dict, model_type=model.short_name,
                       metric="loss", hard_bottleneck=hard_bottleneck)
        model.load_state_dict(state_dict)
        test_accuracy, mpo_list = evaluate_on_test_set(model, test_loader, device=device, non_blocking=non_blocking)
        history["test_accuracy"] = [test_accuracy]
        logger.info(f"Test accuracy: {test_accuracy}. \n")
        history["mpo"] = mpo_list
        histories[model_string] = history

    return histories


def run_models_on_subsets_and_plot(
        subsets, model_strings=None, n_bootstrap=1, fast=False,
        batch_size=16, hard_bottleneck=None, device=None, non_blocking=False, num_workers=0, pin_memory=False,
        persistent_workers=False, base_seed=57):
    """
    Run all available models (from MODEL_STRINGS_CUB) on different subsets. For each subset, plots the models
    training-history together, and also plots the test error for each subset.
    To even out variance, the subsets can be bootstrapped with "n_bootstrap".
    Note that this will only bootstrap the training and validation set, while the test-set remains the same.

    Args:
        subsets (list of int): List of the subsets to run on.
        n_bootstrap (int, optional): The amount of times to draw new subset and run models. Defaults to 1.
        model_strings (list of str): List of strings of the models to evaluate. Load from `src.constants.py`.
        fast (bool, optional): If True, will load hyperparameters with low `n_epochs`. Defaults to False.
        batch_size (int, optional): Batch-size of the training. Defaults to 16.
        hard_bottleneck (bool): If True, will load hard-bottleneck concept layer for the concept models.
            If not, will use what is in the hyperparameters.
        device (str): Use "cpu" for cpu training and "cuda" for gpu training.
        non_blocking (bool): If True, allows for asyncronous transfer between RAM and VRAM.
            This only works together with `pin_memory=True` to dataloader and GPU training.
        num_workers (int): The amount of subprocesses used to load the data from disk to RAM.
            0, default, means that it will run as main process.
        pin_memory (bool): Whether or not to pin RAM memory (make it non-pagable).
            This can increase loading speed from RAM to VRAM (when using `to("cuda:0")`,
            but also increases the amount of RAM necessary to run the job. Should only
            be used with GPU training.
        persistent_workers (bool): If `True`, will not shut down workers between epochs.
        base_seed (int, optional): Seed for the subset generation. Will iterate with 1 for every bootstrap.
            Defaults to 57.
    """
    if model_strings is None:
        model_strings = MODEL_STRINGS_CUB
    concept_model_strings = []
    for model_string in model_strings:
        if model_string in CONCEPT_MODELS_STRINGS_CUB:
            concept_model_strings.append(model_string)

    for n_subset in subsets:
        seed = base_seed
        logger.info(
            f"\n    Beginning evaluation on subset {n_subset} / {subsets}: \n")
        histories_total = {}
        for model_string in model_strings:
            histories_total[model_string] = {}
        # Load test-set for full dataset (not subset)
        if fast:
            test_name = "test_small"  # Only load 200 of the test-images
        else:
            test_name = "test"  # Load all 5794 of the test-images
        test_loader = load_data_cub(
            n_subset=None, mode=test_name, batch_size=batch_size, drop_last=False, num_workers=num_workers,
            pin_memory=pin_memory, persistent_workers=persistent_workers)
        if fast:
            test_loader = test_loader
        for i in range(n_bootstrap):
            logger.info(f"Beginning bootstrap iteration [{i + 1} / {n_bootstrap}]")
            if n_subset is not None:  # Do not make subset of data-list if we use all data
                make_subset_cub(n_images_class=n_subset, seed=seed)
            seed += 1  # Change seed so that subset will be different for the bootstrapping

            train_loader, val_loader = load_data_cub(
                mode="train-val", n_subset=n_subset, batch_size=batch_size, drop_last=False,
                num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)

            histories = train_and_evaluate_cub(
                n_subset=n_subset, train_loader=train_loader, val_loader=val_loader,
                test_loader=test_loader, model_strings=model_strings, hyperparameters=None,
                hard_bottleneck=hard_bottleneck, fast=fast, device=device, non_blocking=non_blocking, seed=seed)
            histories_total = add_histories(histories_total, histories, model_strings)

        # All bootstrap runs for one subset are finished
        save_history_cub(n_bootstrap, n_subset, histories_total, hard_bottleneck=hard_bottleneck)
        plot_training_histories_cub(
            n_bootstrap, n_subset, histories=histories_total,
            model_strings=model_strings, hard_bottleneck=hard_bottleneck, attributes=False)

        # Exclude cnn model when plotting attributes / concept training
        if concept_model_strings != {}:
            plot_training_histories_cub(
                n_bootstrap, n_subset, histories=histories_total,
                model_strings=concept_model_strings, hard_bottleneck=hard_bottleneck, attributes=True)
            plot_mpo_scores_cub(
                n_bootstrap, n_subset, histories=histories_total, model_strings=concept_model_strings,
                hard_bottleneck=hard_bottleneck)

    plot_test_accuracies_cub(
        subsets=subsets, n_bootstrap=n_bootstrap, hard_bottleneck=hard_bottleneck, model_strings=model_strings)


def only_plot(
        subsets, model_strings=None, n_bootstrap=1, hard_bottleneck=None, plot_train=True, plot_test=True):
    """
    Assume histories are made, and only do the plotting. This is useful for changing the plots slightly after a run.

    Args:
        subsets (list of int): List of the subsets to run on.
        model_strings (list of str): List of strings of the models to evaluate. Load from `src.constants.py`.
        n_bootstrap (int, optional): The amount of times to draw new subset and run models. Defaults to 1.
        hard_bottleneck (bool): If True, will load hard-bottleneck concept layer for the concept models.
            If not, will use what is in the hyperparameters.
        plot_train (bool): If `True`, will plot the training histories.
        plot_test (bool): If `True`, will plot the test accuracies.
    """
    if model_strings is None:
        model_strings = MODEL_STRINGS_CUB

    concept_model_strings = []
    for model_string in model_strings:
        if model_string in CONCEPT_MODELS_STRINGS_CUB:
            concept_model_strings.append(model_string)

    if plot_train:
        n_subset = subsets[-1]
        histories = load_history_cub(
            n_bootstrap, n_subset=n_subset, hard_bottleneck=hard_bottleneck)

        plot_training_histories_cub(
            n_bootstrap, n_subset, histories=histories,
            model_strings=model_strings, hard_bottleneck=hard_bottleneck, attributes=False)

        if concept_model_strings != {}:
            # Exclude cnn model when plotting attributes / concept training
            plot_training_histories_cub(
                n_bootstrap, n_subset, histories=histories,
                model_strings=concept_model_strings, hard_bottleneck=hard_bottleneck, attributes=True)
            plot_mpo_scores_cub(
                n_bootstrap, n_subset, histories=histories, model_strings=concept_model_strings,
                hard_bottleneck=hard_bottleneck)

    if plot_test and len(subsets) > 1:
        plot_test_accuracies_cub(
            subsets=subsets, n_bootstrap=n_bootstrap,
            hard_bottleneck=hard_bottleneck, model_strings=model_strings)
