"""
Testing function where the code can be run.
"""
import argparse
import torch
import torch.nn as nn

from src.train import train_simple, train_cbm
from src.datasets.datasets_cub import load_data_cub, make_subset_cub
from src.models.models_cub import CubCNN, CubCBM, CubCBMWithResidual, CubCBMWithSkip
from src.common.utils import seed_everything, get_logger, set_global_log_level, find_class_imbalance, parse_int_list
from src.common.path_utils import load_data_list_cub
from src.evaluation_cub import run_models_on_subsets_and_plot
from src.hyperparameter_optimization_cub import run_hyperparameter_optimization_all_models


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for parsing command-line arguments.")
    
    parser.add_argument("--run_hyperparameters", action="store_true", help="Run hyperparameter search.")
    parser.add_argument("--no_grid_search", action="store_true", help="Do not run grid-search, but TPEsampler.")
    parser.add_argument("--evaluate_and_plot", action="store_true", help="Evaluate models and plot.")

    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--n_bootstrap", type=int, default=1, help="number of bootstrap iterations to run")
    help = "Number of bootstrap iterations to save at. Can be single int or list of int, for example `1` or `1,5,10`."
    parser.add_argument("--bootstrap_checkpoints", type=parse_int_list, help=help)
    # parser.add_argument("--n_bootstrap", type=int, default=1, help="Number of bootstrap iterations.")
    help = "Sizes of subsets to run on. Can be single int or list of int, for example `5` or `5,10,15`. " 
    help += " Use `30` for full dataset."
    parser.add_argument("--subsets", type=parse_int_list, default=[1], help=help)
    parser.add_argument("--n_trials", type=int, default=100, help="Number of trials for hyperparameter search.")
    parser.add_argument("--fast", action="store_true", help="Use fast testing hyperparameters.")
    parser.add_argument("--hard_bottleneck", action="store_true", help="If True, will use hard bottleneck")

    parser.add_argument("--device", type=str, default=None, help="Device to train on. cuda:0 or CPU.")
    parser.add_argument("--non_blocking", action="store_true", help="Allows asynchronous RAM to VRAM operations.")
    parser.add_argument("--num_workers", type=int, default=0, help="Amount of workers to load data to RAM.")
    parser.add_argument("--pin_memory", action="store_true", help="Pins the RAM memory (makes it non-pagable).")
    parser.add_argument("--persistent_workers", action="store_true", help="Do not reload workers between epochs")
    parser.add_argument("--logging_level", type=str, default="info", help="Verbosisty level of the logger.")
    parser.add_argument("--optuna_verbosity", type=int, default=1, help="Verbosity of optuna (2, 1 or 0).")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    seed_everything(57)
    args = parse_arguments()

    set_global_log_level(args.logging_level)
    logger = get_logger(__name__)

    if args.device is None or args.device == "":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i in range(len(args.subsets)):
        if args.subsets[i] >= 30:
            args.subsets[i] = None

    grid_search = not args.no_grid_search

    if args.run_hyperparameters:
        run_hyperparameter_optimization_all_models(
            n_trials=args.n_trials, grid_search=grid_search, subsets=args.subsets, batch_size=args.batch_size,
            eval_loss=True, hard_bottleneck=args.hard_bottleneck, device=device, num_workers=args.num_workers,
            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers, non_blocking=args.non_blocking,
            fast=args.fast, optuna_verbosity=args.optuna_verbosity)

    if args.evaluate_and_plot:
        logger.info(f"\nBeginning evaluation with {args.n_bootstrap} bootstrap iterations.\n")
        run_models_on_subsets_and_plot(
            subsets=args.subsets, n_bootstrap=args.n_bootstrap, bootstrap_checkpoints=args.bootstrap_checkpoints,
            fast=args.fast, batch_size=args.batch_size, hard_bottleneck=args.hard_bottleneck,
            non_blocking=args.non_blocking, num_workers=args.num_workers, pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers)
