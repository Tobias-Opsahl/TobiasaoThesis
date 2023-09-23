import os
import yaml
import optuna
import torch
import torch.nn as nn
from shapes.datasets_shapes import load_data_shapes, make_subset_shapes
from train import train_simple, train_cbm
from utils import load_single_model
from constants import MODEL_STRINGS


class HyperparameterOptimizationShapes:
    """
    Class for doing hyperparameter search with optuna.
    This tests many hyperparameters with smart statistical methods, instead of normal grid-search.
    However, in order to be effient, one needs many runs.

    This class both loads models, trains them, sets up optuna study and saves hyperparameters to yaml file.
    One class is needed for every dataset and every model.
    """

    def __init__(self, model_type, dataset_path, n_classes, n_attr=None, subset_dir="", batch_size=16,
                 eval_loss=True, device=None, num_workers=0, pin_memory=False, persistent_workers=False,
                 non_blocking=False, min_epochs=10, max_epochs=50, seed=57):
        """
        Args:
            model_type (str): Type of model.
            dataset_path (str): Path to the dataset_directory
            n_classes (int): The amount of classes in the dataset.
            n_attr (int): The amount of attributes in the dataset
            subset_dir (str, optional): Directory to subset of data to use. Defaults to "".
            batch_size (int, optional): Batch-size for training. Defaults to 16.
            eval_loss (bool, optional): If `True`, will use evalulation set loss as metric for optimizing.
                If false, will use evaluation accuracy. Defaults to True.
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
            min_epochs (int, optional): Minimum amount of epochs to run. Defaults to 10.
            max_epochs (int, optional): Maximum amount of epochs to run. Defaults to 50.
            seed (int, optional): Seed, used in case subdataset needs to be created.. Defaults to 57.

        Raises:
            ValueError: If model_type is not in MODEL_STRINGS.
        """
        model_type = model_type.strip().lower()
        if model_type not in MODEL_STRINGS:
            raise ValueError(f"The model type must be in {MODEL_STRINGS}. Was {model_type}. ")
        self.model_type = model_type

        self.device = device
        if device is None or device == "":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset_path = dataset_path
        self.n_classes = n_classes
        self.n_attr = n_attr
        self.subset_dir = subset_dir  # TODO: Deal with subset_dir / n_subset nicely
        self.batch_size = batch_size
        self.eval_loss = eval_loss
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.non_blocking = non_blocking
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.study_ran = False

        if not os.path.exists(dataset_path + "tables/" + subset_dir):
            make_subset_shapes(dataset_path, subset_dir, n_classes, seed=seed)

    def objective(self, trial):
        """
        Optunas objective function. This function is ran every test run for an optuna study.
        Runs the model with one set of hyperparameters and returns the evlaluation metric.

        Args:
            trial (trial): Optuna trial

        Returns:
            metric: Either the evaluation loss or the evaluation accuracy.
        """
        train_loader, val_loader, _ = load_data_shapes(
            path=self.dataset_path, subset_dir=self.subset_dir, batch_size=self.batch_size, drop_last=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)

        hp = {}  # Hyperparameters
        hp["learning_rate"] = trial.suggest_float("learning_rate", 0.0002, 0.02, log=True)
        hp["gamma"] = trial.suggest_float("gamma", 0.5, 1, log=False)
        hp["n_linear_output"] = trial.suggest_int("n_linear_output", 32, 128, log=True)
        hp["n_epochs"] = trial.suggest_int("n_epochs", self.min_epochs, self.max_epochs, log=False)
        if self.model_type != "cnn":
            hp["activation"] = trial.suggest_categorical("activation", ["relu", "sigmoid", "none"])
            hp["attr_weight"] = trial.suggest_float("attr_weight", 0.01, 10, log=True)
        if self.model_type == "cbm":
            hp["two_layers"] = trial.suggest_categorical("two_layers", [True, False])
        if self.model_type == "cbm_skip":
            hp["n_hidden"] = trial.suggest_int("n_hidden", 8, 32, log=True)

        model = load_single_model(self.model_type, n_classes=self.n_classes, n_attr=self.n_attr, hyperparameters=hp)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hp["learning_rate"])
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=hp["gamma"])
        if self.model_type == "cnn":  # Train non-bottleneck model
            history = train_simple(
                model, criterion, optimizer, train_loader, val_loader, trial=trial, scheduler=exp_lr_scheduler,
                n_epochs=hp["n_epochs"], device=self.device, non_blocking=self.non_blocking, verbose=0)
        else:  # Concept-bottleneck model
            attr_criterion = nn.BCEWithLogitsLoss()
            history = train_cbm(
                model, criterion, attr_criterion, optimizer, train_loader, val_loader, n_epochs=hp["n_epochs"],
                attr_weight=hp["attr_weight"], trial=trial, scheduler=exp_lr_scheduler, device=self.device,
                non_blocking=self.non_blocking, verbose=0)

        if self.eval_loss:
            return history["val_class_loss"][-1]
        else:
            return history["val_class_accuracy"][-1]

    def run_optuna_hyperparameter_search(self, n_trials=50, write=True, base_dir="hyperparameters/shapes",
                                         verbose="warning"):
        """
        Runs a whole optuna study.

        Args:
            n_trials (int, optional): Amount of trials to run. Defaults to 50.
            write (bool, optional): If True, writes hyperparameters to file. Defaults to True.
            base_dir (str, optional): Path to hyperparameter base directory. Defaults to "hyperparameters/shapes/".
            verbose (str, optional): Controls the verbosity of optuna study. Defaults to "warning".
        """
        if self.eval_loss:  # Minimize loss
            direction = "minimize"
        else:  # Maximize acccuracy
            direction = "maximize"
        verbose = str(verbose).strip().lower()

        if verbose == "info" or verbose == "2":
            optuna.logging.set_verbosity(optuna.logging.INFO)
        if verbose == "warning" or verbose == "1":
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        if verbose == "error" or verbose == "0":
            optuna.logging.set_verbosity(optuna.logging.ERROR)

        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1)
        study = optuna.create_study(direction=direction, pruner=pruner)
        study.optimize(self.objective, n_trials=n_trials)
        self.study = study
        self.study_ran = True
        self.test_accuracy = 0
        if write:
            self.write_to_yaml(base_dir=base_dir)

    def round_dict_values(self, hyperparameters_dict, n_round=4):
        """
        Rounds off float values in dict.

        Args:
            hyperparameters_dict (dict): The dictionary.
            n_round (int, optional): The amount of digits to round to. Defaults to 4.

        Returns:
            dict: The rounded dict.
        """
        for key, value in hyperparameters_dict.items():
            if isinstance(value, float):
                hyperparameters_dict[key] = round(value, n_round)
        return hyperparameters_dict

    def write_to_yaml(self, base_dir="hyperparameters/shapes/"):
        """
        Writes discovered hyperparameters to file.

        Args:
            base_dir (str, optional): Path to hyperparameter base directory. Defaults to "hyperparameters/shapes/".

        Raises:
            Exception: If the hyperparameters are not yet searched for.
        """
        if not self.study_ran:
            raise Exception(f"run_optuna_hyperparameter_search must be called before write_to_yaml().")
        trial = self.study.best_trial
        # Add shared hyperparameters for every model-type
        hyperparameters = {
            "val_loss": trial.value, "test_accuracy": self.test_accuracy,
            "learning_rate": trial.params["learning_rate"], "n_epochs": trial.params["n_epochs"],
            "n_linear_output": trial.params["n_linear_output"], "gamma": trial.params["gamma"]}
        if self.model_type != "cnn":
            hyperparameters["attr_weight"] = trial.params["attr_weight"]
            hyperparameters["activation"] = trial.params["activation"]
        if self.model_type == "cbm":
            hyperparameters["two_layers"] = trial.params["two_layers"]
        if self.model_type == "cbm_skip":
            hyperparameters["n_hidden"] = trial.params["n_hidden"]

        hyperparameters = self.round_dict_values(hyperparameters)
        folder_name = "c" + str(self.n_classes) + "_a" + str(self.n_attr) + "/"
        hp_path = base_dir + folder_name
        hp_filename = "hyperparameters_" + self.subset_dir.strip("/") + ".yaml"
        if not os.path.exists(hp_path + hp_filename):  # No yaml file for this setting
            os.makedirs(hp_path, exist_ok=True)
            with open(hp_path + hp_filename, "w") as outfile:
                hyperparameters_full = {self.model_type: hyperparameters}
                yaml.dump(hyperparameters_full, outfile, default_flow_style=False)
            return

        # File already exists. We have to read, overwrite only this model_type, and write again.
        with open(hp_path + hp_filename, "r") as yaml_file:
            hyperparameters_full = yaml.safe_load(yaml_file)
        hyperparameters_full[self.model_type] = hyperparameters  # Overwrite or create new dict of model_type only
        with open(hp_path + hp_filename, "w") as yaml_file:
            yaml.dump(hyperparameters_full, yaml_file, default_flow_style=False)


def run_hyperparameter_optimization_all_models(
        dataset_path, n_classes, n_attr, subsets, base_dir="hyperparameters/shapes/", n_trials=10, batch_size=16,
        eval_loss=True, device=None, num_workers=0, pin_memory=False, persistent_workers=False, non_blocking=False,
        min_epochs=10, max_epochs=50, write=True, verbose="warning"):
    """
    Run hyperparameter search for every model for many subsets.

    Args:
        dataset_path (str): Path to the dataset_directory
        n_classes (int): The amount of classes in the dataset.
        n_attr (int): The amount of attributes in the dataset
        subsets (list of int): List of the subsets to run on.
        base_dir (str, optional): Path to hyperparameter base directory. Defaults to "hyperparameters/shapes/".
        n_trials (int, optional): Amount of trials to run. Defaults to 10.
        batch_size (int, optional): Batch-size. Defaults to 16.
        eval_loss (bool, optional): If `True`, will use evalulation set loss as metric for optimizing.
            If false, will use evaluation accuracy. Defaults to True.
        device (str): Use "cpu" for cpu training and "cuda" for gpu training.
        non_blocking (bool): If True, allows for asyncronous transfer between RAM and VRAM.
            This only works together with `pin_memory=True` to dataloader and GPU training.
        num_workers (int): The amount of subprocesses used to load the data from disk to RAM.
            0, default, means that it will run as main process.
        pin_memory (bool): Whether or not to pin RAM memory (make it non-pagable).
            This can increase loading speed from RAM to VRAM (when using `to("cuda:0")`,
            but also increases the amount of RAM necessary to run the job. Should only
            be used with GPU training.
        min_epochs (int, optional): Minimum amount of epochs to run. Defaults to 10.
        max_epochs (int, optional): Maximum amount of epochs to run. Defaults to 50.
        write (bool, optional): Whether to write hyperparameters to file or not.
        verbose (str, optional): Controls the verbosity of optuna study. Defaults to "warning".
    """

    for subset in subsets:
        print(f"\nRunning hyperparameter search for {subset} subsets. \n")
        subset_dir = "sub" + str(subset) + "/"
        for model_type in MODEL_STRINGS:
            print(f"\nRunning hyperparameter optimization on model {model_type}. \n")
            obj = HyperparameterOptimizationShapes(
                model_type=model_type, dataset_path=dataset_path, n_classes=n_classes, n_attr=n_attr,
                subset_dir=subset_dir, batch_size=batch_size, eval_loss=eval_loss, device=device,
                num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers,
                non_blocking=non_blocking, min_epochs=min_epochs, max_epochs=max_epochs)
            obj.run_optuna_hyperparameter_search(n_trials=n_trials, write=write, base_dir=base_dir, verbose=verbose)
            print(f"\nFinnished hyperparameter search for model{model_type} with value {obj.study.best_trial.value}.\n")


if __name__ == "__main__":
    run_hyperparameter_optimization_all_models(dataset_path="data/shapes/shapes_2k_c10_a5/", n_classes=10,
                                               n_attr=5, n_trials=1, subsets=[60, 70], batch_size=16, min_epochs=2,
                                               max_epochs=3, verbose="info", write=True)
