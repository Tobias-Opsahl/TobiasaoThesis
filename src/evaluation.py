import torch
import torch.nn as nn

from src.train import train_simple, train_cbm
from src.datasets.datasets_shapes import load_data_shapes, make_subset_shapes
from src.plotting import plot_training_histories, plot_test_accuracies
from src.common.utils import seed_everything, get_logger, load_models_shapes, add_histories, average_histories
from src.common.path_utils import load_hyperparameters_shapes, save_history_shapes, save_model_shapes
from src.constants import MODEL_STRINGS, COLORS, MAX_EPOCHS, FAST_MAX_EPOCHS, BOOTSTRAP_CHECKPOINTS


logger = get_logger(__name__)


def evaluate_on_test_set(model, test_loader, device=None, non_blocking=False):
    """
    Evaluates a model on the test set and returns the test accuracy.

    Args:
        model (model): Pytorch pretrained model
        test_loader (_type_): _description_
        device (_type_, optional): _description_. Defaults to None.
        non_blocking (bool, optional): _description_. Defaults to False.
        test_loader (dataloader): The test-dataloader
        device (str): Use "cpu" for cpu training and "cuda:0" for gpu training.
        non_blocking (bool): If True, allows for asyncronous transfer between RAM and VRAM.
            This only works together with `pin_memory=True` to dataloader and GPU training.

    Returns:
        float: The test accuracy
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device, non_blocking=non_blocking)
    model.eval()
    test_correct = 0
    for input, labels, attr_labels, paths in test_loader:
        input = input.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        outputs = model(input)
        if isinstance(outputs, tuple):  # concept model, returns (outputs, attributes)
            outputs = outputs[0]
        _, preds = torch.max(outputs, 1)
        test_correct += (preds == labels).sum().item()

    test_accuracy = 100 * test_correct / len(test_loader.dataset)
    return test_accuracy


def train_and_evaluate_shapes(
        n_classes, n_attr, signal_strength, n_subset, train_loader, val_loader, test_loader,
        hyperparameters=None, hard_bottleneck=None, fast=False, device=None, non_blocking=False, seed=None):
    """
    Trains the shapes models given some hyperparameters, classes, attributes and data subdirectory.
    The data subdirectory should be generated by `make_subset_shapes()`.

    Args:
        n_classes (int): Amount of classes.
        n_attr (int): Amount of attribues.
        signal_strength (int): The signal-strength used for creating the dataset.
        n_subset (int): The amount of instances in each class for this subset.
        train_loader (dataloader): The training dataloader.
        val_loader (dataloader): The validation dataloader.
        test_loader (dataloader): The test dataloader.
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
        hp = load_hyperparameters_shapes(n_classes, n_attr, signal_strength, n_subset, hard_bottleneck=hard_bottleneck)

    models = load_models_shapes(n_classes, n_attr, hyperparameters=hp)
    histories = []
    criterion = nn.CrossEntropyLoss()
    attr_criterion = nn.BCEWithLogitsLoss()

    for i in range(len(models)):
        model = models[i]
        model_string = MODEL_STRINGS[i]
        logger.info(f"Running model {model.name}:")
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=hp[model_string]["learning_rate"])
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=hp[model_string]["gamma"])

        n_epochs = MAX_EPOCHS
        if fast:
            n_epochs = FAST_MAX_EPOCHS
        if seed is not None:  # Seed before training, for reprorudibility
            seed_everything(seed)
        if model.name == "ShapesCNN":
            history, models_dict = train_simple(
                model, criterion, optimizer, train_loader, val_loader, scheduler=exp_lr_scheduler, n_epochs=n_epochs,
                n_early_stop=False, device=device, non_blocking=non_blocking)
        else:
            history, models_dict = train_cbm(
                model, criterion, attr_criterion, optimizer, train_loader, val_loader, n_epochs=n_epochs,
                attr_weight=hp[model_string]["attr_weight"], scheduler=exp_lr_scheduler, n_early_stop=False,
                device=device, non_blocking=non_blocking)

        state_dict = models_dict["best_model_loss_state_dict"]
        save_model_shapes(n_classes, n_attr, signal_strength, n_subset, state_dict,
                          model_type=model.short_name, metric="loss", hard_bottleneck=hard_bottleneck)
        model.load_state_dict(state_dict)
        test_accuracy = evaluate_on_test_set(model, test_loader, device=device, non_blocking=non_blocking)
        history["test_accuracy"] = [test_accuracy]
        histories.append(history)

    return histories


def run_models_on_subsets_and_plot(
        n_classes, n_attr, signal_strength, subsets, n_bootstrap=1, bootstrap_checkpoints=None, fast=False,
        batch_size=16, hard_bottleneck=None, device=None, non_blocking=False, num_workers=0, pin_memory=False,
        persistent_workers=False, base_seed=57):
    """
    Run all available models (from MODEL_STRINGS) on different subsets. For each subset, plots the models
    training-history together, and also plots the test error for each subset.
    To even out variance, the subsets can be bootstrapped with "n_bootstrap".
    Note that this will only bootstrap the training and validation set, while the test-set remains the same.

    Args:
        n_classes (int): The amount of classes in the dataset.
        n_attr (int): The amount of attributes in the dataset
        subsets (list of int): List of the subsets to run on.
        signal_strength (int): The signal_strength the dataset is created with.
        n_bootstrap (int, optional): The amount of times to draw new subset and run models. Defaults to 1.
        bootstrap_checkpoints (list of int): List of bootstrap iterations to save and plot after.
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
    if bootstrap_checkpoints is None:
        bootstrap_checkpoints = BOOTSTRAP_CHECKPOINTS

    for n_subset in subsets:
        seed = base_seed
        logger.info(f"\n    Beginning evaluation on subset {n_subset} / {subsets}: \n")
        histories_total = None
        # Load test-set for full dataset (not subset)
        test_loader = load_data_shapes(
            n_classes, n_attr, signal_strength, n_subset=None, mode="test", batch_size=batch_size,
            drop_last=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
        for i in range(n_bootstrap):
            logger.info(f"Begginging bootstrap iteration [{i + 1} / {n_bootstrap}]")
            make_subset_shapes(n_classes, n_attr, signal_strength, n_images_class=n_subset, seed=seed)
            seed += 1  # Change seed so that subset will be different for the bootstrapping # TODO: Uncomment

            train_loader, val_loader = load_data_shapes(
                n_classes, n_attr, signal_strength, n_subset, mode="train-val", batch_size=batch_size, drop_last=False,
                num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)

            histories = train_and_evaluate_shapes(
                n_classes, n_attr, signal_strength, n_subset=n_subset, train_loader=train_loader, val_loader=val_loader,
                test_loader=test_loader, hyperparameters=None, hard_bottleneck=hard_bottleneck,
                fast=fast, device=device, non_blocking=non_blocking, seed=base_seed)
            histories_total = add_histories(histories_total, histories)

            n_boot = i + 1
            if n_boot in bootstrap_checkpoints:

                averaged_histories = average_histories(histories_total, n_boot)
                save_history_shapes(n_classes, n_attr, signal_strength, n_boot, n_subset, averaged_histories,
                                    hard_bottleneck=hard_bottleneck)
                plot_training_histories(n_classes, n_attr, signal_strength, n_boot, n_subset,
                                        histories=averaged_histories, names=MODEL_STRINGS,
                                        hard_bottleneck=hard_bottleneck, colors=COLORS, attributes=False)
                # Exclude cnn model when plotting attributes / concept training
                plot_training_histories(n_classes, n_attr, signal_strength, n_boot, n_subset,
                                        histories=averaged_histories[1:], names=["cbm", "cbm_res", "cbm_skip", "scm"],
                                        hard_bottleneck=hard_bottleneck, colors=COLORS[1:], attributes=True)

    for n_boot in bootstrap_checkpoints:
        if n_boot > n_bootstrap:  # Incase a checkpoint is higher than the actual iterations ran
            break
        plot_test_accuracies(n_classes, n_attr, signal_strength, subsets=subsets, n_bootstrap=n_boot,
                             hard_bottleneck=hard_bottleneck)
