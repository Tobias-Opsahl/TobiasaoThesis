import os
import yaml
import optuna
import torch
import torch.nn as nn
from shapes.datasets_shapes import load_data_shapes, make_subset_shapes
from train import train_simple, train_cbm
from utils import load_single_model, get_hyperparameters
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
                 non_blocking=False, fast=False, seed=57):
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
            fast (bool, optional): If True, will use hyperparameters with very low `n_epochs`. Defaults to False.
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
        self.subset_dir = subset_dir
        self.batch_size = batch_size
        self.eval_loss = eval_loss
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.non_blocking = non_blocking
        self.study_ran = False
        self.hyperparameter_names = self._get_hyperparameter_names()  # Names of possible hyperparameter
        self.default_hyperparameters = get_hyperparameters(0, 0, 0, fast=fast, default=True)

        if not os.path.exists(dataset_path + "tables/" + subset_dir):  # Make subset of dataset if it does not exist
            make_subset_shapes(dataset_path, subset_dir, n_classes, seed=seed)
        self.subset_dir = subset_dir  # Save for study-name

    def _get_hyperparameter_names(self):
        """
        Get the names of the possible hyperparameters. Note that this is different for the five possible
        models, hence this function.

        Returns:
            list of str: List of the hyperparameter names.
        """
        # Common hyperparameters
        hyperparameter_names = ["learning_rate", "dropout_probability", "gamma", "n_linear_output", "n_epochs"]
        if self.model_type != "cnn":  # Hyperparameters only for concept models
            hyperparameter_names.append("activation")
            hyperparameter_names.append("attr_weight")
            hyperparameter_names.append("attr_weight_decay")
        if self.model_type == "cbm":
            hyperparameter_names.append("two_layers")
        elif self.model_type in ["cbm_skip", "scm"]:
            hyperparameter_names.append("n_hidden")
        return hyperparameter_names

    def _get_default_hyperparameters_to_search(self):
        """
        Return a default dictionary of which hyperparameters to look for during hyperparameter optimization.
        This function if called if this dict is not passed to the `run_hyperparameter_optimization()` method.

        Returns:
            dict: Dictionary of hyperparameter-names pointing to booleans.
        """
        hp = {"learning_rate": True, "dropout_probability": True, "gamma": True, "attr_schedule": True,
              "attr_weight": False, "attr_weight_decay": False, "n_epochs": False,
              "n_linear_output": False, "activation": False, "two_layers": False, "n_hidden": False}
        return hp

    def _check_hyperparameters_to_search(self, hyperparameters_to_search):
        """
        Checks that the argument `hyperparameters_to_search` is of correct format.
        This is the argument given to `run_hyperparameter_search()` that determines which hyperparameters
        that are searched for. It needs to be a dict that maps all hyperparameter-names to booleans.
        Additionally, the if "attr_schedule" is True, then "attr_weight" and "attr_weight_decay" needs to be False,
        since "attr_schedule" determines both of them.
        The "attr_schedule" tries to either set "attr_weight" to a moderate constant and "attr_weight_decay" to 1 (no
        decay), or to set "attr_weight" to a high value and set "attr_weight_decay" to < 1. This setup reduces the
        complexity in the hyperparameter search by removing unecessary searches (like low "attr_weight" and high
        "attr_weight_decay", or vice versa).

        Args:
            hyperparameters_to_search (dict): The argument to check

        Raises:
            ValueError or TypeError: If the dict is on wrong format.
        """
        if not isinstance(hyperparameters_to_search, dict):
            message = f"Argument `hyperparameters_to_search` must be of type dict, and map hyperparameter-names to "
            message += f"booleans. Was type {type(hyperparameters_to_search)} with value {hyperparameters_to_search}. "
            raise TypeError(message)
        if hyperparameters_to_search.get("attr_schedule") is not None:
            attr_schedule = hyperparameters_to_search["attr_schedule"]
            attr_weight = hyperparameters_to_search["attr_weight"]
            attr_weight_decay = hyperparameters_to_search["attr_weight_decay"]
            if attr_schedule and (attr_weight or attr_weight_decay):
                message = f"Argument `hyperparameters_to_search`: When `attr_schedule` is True, `attr_weight` and "
                message += f"`attr_weight_schedule` must be False. Was {attr_weight=}, {attr_weight_decay=}. "
                raise ValueError(message)
        for hyperparameter_name in self.hyperparameter_names:
            if hyperparameters_to_search.get(hyperparameter_name) is None:
                message = f"Argument `hyperparameters_to_search` (dict) did not include all needed arguments. "
                message += f"Was missing argument {hyperparameter_name}. "
                message += f"Need the following keys mapped to a boolean: {self.hyperparameter_names}. "
                raise ValueError(message)

    def _get_single_hyperparameter_for_trial(self, hp_name, trial):
        """
        Get a single suggestion for a hyperparameter suggested by optunas trial.

        Args:
            hp_name (str): The name of the hyperparameter to get a suggestion for.
            trial (trial): Optuna trial.

        Raises:
            ValueError: If the hyperparameter-name is not in `self.hyperparameter_names`.

        Returns:
            object: Trial suggestion of the value to test.
        """
        if hp_name == "learning_rate":
            return trial.suggest_float("learning_rate", 0.0001, 0.1, log=True)
        elif hp_name == "dropout_probability":
            return trial.suggest_float("dropout_probability", 0, 0.5, log=False)
        elif hp_name == "gamma":
            return trial.suggest_float("gamma", 0.1, 1, log=False)
        elif hp_name == "n_epochs":
            return trial.suggest_int("n_epochs", 15, 50, log=False)
        elif hp_name == "n_linear_epochs":
            return trial.suggest_int("n_linear_output", 32, 128, log=True)
        elif hp_name == "n_hidden":
            return trial.suggest_int("n_hidden", 8, 32, log=True)
        elif hp_name == "activation":
            return trial.suggest_categorical("activation", ["relu", "sigmoid", "none"])
        elif hp_name == "two_layers":
            return trial.suggest_categorical("two_layers", [True, False])
        elif hp_name == "attr_weight":
            return trial.suggest_float("attr_weight", 1, 10, log=False)
        elif hp_name == "attr_weight_decay":
            return trial.suggest_float("attr_weight_decay", 0.5, 1, log=False)
        elif hp_name == "attr_schedule":
            attr_schedule = trial.suggest_categorical("attr_schedule", [0.7, 0.8, 0.9, 1, 3, 5, 10])
            if attr_schedule < 1:
                attr_weight = 100
                attr_weight_decay = attr_schedule
            else:
                attr_weight = attr_schedule
                attr_weight_decay = 1
            trial.set_user_attr("attr_weight", attr_weight)  # Set values in trial so we can find them with best_trial
            trial.set_user_attr("attr_weight_decay", attr_weight_decay)
            return attr_weight, attr_weight_decay
        else:
            raise ValueError(f"Hyperparameter name not valid, got {hp_name}")

    def _get_hyperparameters_for_trial(self, trial):
        """
        Returns a full set of hyperparameters to use for a single trial in an optuna study.
        Some of the values are suggested, some are set to default values. This depends on `hyperparameters_to_search`.

        Args:
            trial (trial): Optuna trial.

        Returns:
            dict: Dictionary of the hyperparameter-names pointing to the hyperparameter values.
        """
        hyperparameters = {}
        for hp_name in self.hyperparameter_names:
            if self.hyperparameters_to_search[hp_name]:  # Make trial suggest value
                hyperparameters[hp_name] = self._get_single_hyperparameter_for_trial(hp_name, trial)
            else:  # Set default value
                hyperparameters[hp_name] = self.default_hyperparameters[self.model_type][hp_name]
        if (self.model_type != "cnn") and self.hyperparameters_to_search.get("attr_schedule"):
            # This determines two arguments.
            attr_weight, attr_weight_decay = self._get_single_hyperparameter_for_trial("attr_schedule", trial)
            hyperparameters["attr_weight"] = attr_weight
            hyperparameters["attr_weight_decay"] = attr_weight_decay
        return hyperparameters

    def _get_hyperparameters_from_best_trial(self, best_trial):
        """
        Get all hyperparameters from the best trial (or any trial). Note that the hyperparameters not
        searched for will be set from the default hyperparameters.
        Note that `attr_schedule` is handled separately, since if this is True, then both `attr_weight` and
        `attr_weight_decay` is in trial, even though they are False in `self.hyperparameters_to_search`.

        Args:
            best_trial (trial): The best trial from a optuna study (or any trial).

        Returns:
            dict: Dictionary of the hyperparameter-names pointing to the hyperparameter values.
        """
        hyperparameters = {}
        for hp_name in self.hyperparameter_names:
            if self.hyperparameters_to_search[hp_name]:  # Get value from trial
                hyperparameters[hp_name] = best_trial.params[hp_name]
            else:  # Get value from default parameters
                hyperparameters[hp_name] = self.default_hyperparameters[self.model_type][hp_name]
        if (self.model_type != "cnn") and self.hyperparameters_to_search.get("attr_schedule"):
            # This means we search for the two values below.
            hyperparameters["attr_weight"] = best_trial.user_attrs["attr_weight"]
            hyperparameters["attr_weight_decay"] = best_trial.user_attrs["attr_weight_decay"]
        # Write the validation stats
        hyperparameters["best_val_loss"] = best_trial.user_attrs["best_val_loss"]
        hyperparameters["best_val_accuracy"] = best_trial.user_attrs["best_val_accuracy"]
        if not self.hyperparameters_to_search["n_epochs"]:  # Overwite with best epoch-number
            if self.eval_loss:
                hyperparameters["n_epochs"] = best_trial.user_attrs["best_epoch_loss"]
            else:
                hyperparameters["n_epochs"] = best_trial.user_attrs["best_epoch_accuracy"]
        return hyperparameters

    def _get_search_space(self):
        """
        Generates the search-space that Grid-search uses. Note that only the values suggested by the trial,
        controlled by `self.hyperparameters_to_search`, are used.

        Returns:
            dict: Dictionary of the hyperparameter-names pointing to a list of possible values to try.
        """
        all_possibilities = {
            "learning_rate": [0.05, 0.01, 0.005, 0.001],
            "gamma": [0.1, 0.5, 0.7, 0.9, 1],
            "dropout_probability": [0, 0.2, 0.4],
            "n_epochs": [20, 30, 50, 100],
            "n_linear_output": [16, 64, 128, 256],
            "n_hidden": [16, 32, 64],
            "activation": ["sigmoid", "relu", "none"],
            "two_layers": [True, False],
            "attr_weight": [1, 3, 5, 10],
            "attr_weight_decay": [0.5, 0.7, 0.9, 1],
            "attr_schedule": [0.8, 0.9, 1, 5, 10]
        }

        search_space = {}  # Add only the hyperparameters we are going to search for in the space
        for hyperparameter_name in self.hyperparameter_names:
            if self.hyperparameters_to_search[hyperparameter_name]:
                search_space[hyperparameter_name] = all_possibilities[hyperparameter_name]
        if (self.model_type != "cnn") and self.hyperparameters_to_search.get("attr_schedule"):
            search_space["attr_schedule"] = all_possibilities["attr_schedule"]

        return search_space

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

        hp = self._get_hyperparameters_for_trial(trial)  # Get suggestions for hyperparameters
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

        trial.set_user_attr("best_val_loss", history["best_val_loss"])  # Save these values to write later
        trial.set_user_attr("best_val_accuracy", history["best_val_accuracy"])
        trial.set_user_attr("best_epoch_loss", history["best_epoch_loss"])
        trial.set_user_attr("best_epoch_accuracy", history["best_epoch_accuracy"])
        if self.eval_loss:
            return history["best_val_loss"]
        else:
            return history["best_val_accuracy"]

    def run_hyperparameter_search(self, hyperparameters_to_search=None, n_trials=50, write=True,
                                  base_dir="hyperparameters/shapes", grid_search=True, verbose="warning"):
        """
        Runs a whole optuna study.

        Args:
            n_trials (int, optional): Amount of trials to run. Defaults to 50.
            hyperparameters_to_search (dict): Dictionary of hyperparameter-names for keys,
                and boolenas for values, corresponding to which hyperparameters to search for.
                See `self._get_default_hyperparameters_to_search()` for an example, or
                `self._check_hyperparameters_to_search()` for more info.
            write (bool, optional): If True, writes hyperparameters to file. Defaults to True.
            base_dir (str, optional): Path to hyperparameter base directory. Defaults to "hyperparameters/shapes/".
            grid_search (bool): If True, will use standard grid-search. If not, will use default optuna
                sampler, which will be set to TPE sampler if n_trials is less than 1000.
            verbose (str, optional): Controls the verbosity of optuna study. Defaults to "warning".
        """
        self.hyperparameters_to_search = hyperparameters_to_search
        if hyperparameters_to_search is None:
            self.hyperparameters_to_search = self._get_default_hyperparameters_to_search()
        self._check_hyperparameters_to_search(self.hyperparameters_to_search)

        if self.eval_loss:  # Minimize loss
            direction = "minimize"
        else:  # Maximize acccuracy
            direction = "maximize"
        verbose = str(verbose).strip().lower()

        if verbose == "info" or verbose == "2" or verbose == "1":
            optuna.logging.set_verbosity(optuna.logging.INFO)
        if verbose == "warning":
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        if verbose == "error" or verbose == "0":
            optuna.logging.set_verbosity(optuna.logging.ERROR)

        study_name = "Study_" + self.subset_dir.strip("/") + "_" + self.model_type
        if not grid_search:
            pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0, interval_steps=1)
            study = optuna.create_study(direction=direction, pruner=pruner, study_name=study_name)
        else:  # Grid search
            pruner = optuna.pruners.NopPruner()  # One actually have to specify no-pruner, else MedianPruner is used.
            search_space = self._get_search_space()
            study = optuna.create_study(direction=direction, pruner=pruner, study_name=study_name,
                                        sampler=optuna.samplers.GridSampler(search_space=search_space))
        study.optimize(self.objective, n_trials=n_trials)
        self.study = study
        self.study_ran = True
        if write:
            self.write_to_yaml(base_dir=base_dir)

    def _round_dict_values(self, hyperparameters_dict, n_round=4):
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
            raise Exception(f"run_hyperparameter_search must be called before write_to_yaml().")
        trial = self.study.best_trial
        hyperparameters = self._get_hyperparameters_from_best_trial(trial)
        hyperparameters = self._round_dict_values(hyperparameters)

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
        if hyperparameters_full is None:
            hyperparameters_full = {}
        hyperparameters_full[self.model_type] = hyperparameters  # Overwrite or create new dict of model_type only
        with open(hp_path + hp_filename, "w") as yaml_file:
            yaml.dump(hyperparameters_full, yaml_file, default_flow_style=False)


def run_hyperparameter_optimization_all_models(
        dataset_path, n_classes, n_attr, subsets, hyperparameters_to_search=None, grid_search=False,
        base_dir="hyperparameters/shapes/", n_trials=10, batch_size=16, eval_loss=True, device=None, num_workers=0,
        pin_memory=False, persistent_workers=False, non_blocking=False, fast=False, write=True, verbose="warning"):
    """
    Run hyperparameter search for every model for many subsets.

    Args:
        dataset_path (str): Path to the dataset_directory
        n_classes (int): The amount of classes in the dataset.
        n_attr (int): The amount of attributes in the dataset
        subsets (list of int): List of the subsets to run on.
        hyperparameters_to_search (dict, optinal): Determines the hyperparameters to search for. See the
            hyperparameter-class for more doctumentation.
        grid_search (bool, optional): If True, will run grid-search optimization. If not, uses optunas default sampler.
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
        fast (bool, optional): If True, will use hyperparameters with very low `n_epochs`. Defaults to False.
        verbose (str, optional): Controls the verbosity of optuna study. Defaults to "warning".
    """
    set_hyperparameters_to_search = False
    if hyperparameters_to_search is None:
        set_hyperparameters_to_search = True
    for subset in subsets:
        print(f"\nRunning hyperparameter search for {subset} subsets. \n")
        subset_dir = "sub" + str(subset) + "/"
        for model_type in MODEL_STRINGS:
            if set_hyperparameters_to_search:
                if model_type == "cnn":
                    hyperparameters_to_search = {
                        "learning_rate": True, "dropout_probability": True, "gamma": True, "attr_schedule": False,
                        "attr_weight": False, "attr_weight_decay": False, "n_epochs": False, "n_linear_output": False,
                        "activation": False, "two_layers": False, "n_hidden": False}
                else:
                    hyperparameters_to_search = {
                        "learning_rate": True, "dropout_probability": True, "gamma": False, "attr_schedule": True,
                        "attr_weight": False, "attr_weight_decay": False, "n_epochs": False, "n_linear_output": False,
                        "activation": False, "two_layers": False, "n_hidden": False}

            print(f"\nRunning hyperparameter optimization on model {model_type}. \n")
            obj = HyperparameterOptimizationShapes(
                model_type=model_type, dataset_path=dataset_path, n_classes=n_classes, n_attr=n_attr,
                subset_dir=subset_dir, batch_size=batch_size, eval_loss=eval_loss, device=device,
                num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers,
                non_blocking=non_blocking, fast=fast)
            obj.run_hyperparameter_search(n_trials=n_trials, hyperparameters_to_search=hyperparameters_to_search,
                                          grid_search=grid_search, write=write, base_dir=base_dir, verbose=verbose)
            print(f"\nFinnished search for model {model_type} with value {obj.study.best_trial.value}.\n")


if __name__ == "__main__":
    run_hyperparameter_optimization_all_models(dataset_path="data/shapes/shapes_2k_c10_a5/", n_classes=10,
                                               n_attr=5, n_trials=1, subsets=[60, 70], batch_size=16, min_epochs=2,
                                               max_epochs=3, verbose="info", write=True)
