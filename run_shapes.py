"""
Testing function where the code can be run.
"""
import argparse
import torch
import torch.nn as nn
import numpy as np
from utils import seed_everything
from hyperparameter_optimization import run_hyperparameter_optimization_all_models
from evaluation import run_models_on_subsets_and_plot


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for parsing command-line arguments.")

    parser.add_argument("--run_hyperparameters", action="store_true", help="Run hyperparameter search.")
    parser.add_argument("--evaluate_and_plot", action="store_true", help="Evaluate models and plot.")

    # Path stuff
    parser.add_argument("--dataset_folder", type=str, default="shapes_testing/", help="Name of dataset.")
    parser.add_argument("--base_dir", type=str, default="data/shapes/", help="Path to where dataset is.")
    parser.add_argument("--subset_dir", type=str, default="", help="Name of optional subset.")

    # Parameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--n_classes", type=int, default=4, help="Number of classes.")
    parser.add_argument("--n_attr", type=int, default=5, help="Number of attributes.")
    parser.add_argument("--n_bootstrap", type=int, default=1, help="Number of bootstrap iterations.")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of trials for hyperparameter search.")
    parser.add_argument("--fast", action="store_true", help="Use fast testing hyperparameters.")

    # Hardware and div stuff:
    parser.add_argument("--device", type=str, default=None, help="Device to train on. cuda:0 or CPU.")
    parser.add_argument("--non_blocking", action="store_true", help="Allows asynchronous RAM to VRAM operations.")
    parser.add_argument("--num_workers", type=int, default=0, help="Amount of workers to load data to RAM.")
    parser.add_argument("--pin_memory", action="store_true", help="Pins the RAM memory (makes it non-pagable).")
    parser.add_argument("--persistent_workers", action="store_true", help="Do not reload workers between epochs")
    parser.add_argument("--verbose", type=int, default=2, help="Verbosity of training (2, 1 or 0).")

    # Old stuff:
    parser.add_argument("--n_epochs", type=int, default=15, help="Number of epochs for training.")
    parser.add_argument("--n_hidden", type=int, default=16, help="Number of hidden units in cbm skip connection.")
    parser.add_argument("--n_linear_output", type=int, default=64, help="Number of nodes in first linear layer")
    parser.add_argument("--attr_weight", type=int, default=1, help="Weight for attributes.")
    parser.add_argument("--sigmoid", action="store_true", help="Use sigmoid activation in concept layer.")
    parser.add_argument("--relu", action="store_true", help="Use relu activation in concept layer.")
    parser.add_argument("--two_layers", action="store_true", help="Use two layers in CBM model after bottleneck.")
    parser.add_argument("--small", action="store_true", help="Use a small model. If not specified, defaults to False.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    seed_everything(57)
    args = parse_arguments()

    device = args.device
    if args.device is None or args.device == "":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = args.base_dir + args.dataset_folder
    subsets = [50, 100, 150, 200, 250]

    if args.run_hyperparameters:
        if args.fast:
            min_epochs = 2
            max_epochs = 3
            base_dir = "hyperparameters/shapes/testing/"
        else:
            min_epochs = 10
            max_epochs = 30
            base_dir = "hyperparameters/shapes/"
        run_hyperparameter_optimization_all_models(
            path, args.n_classes, args.n_attr, n_trials=args.n_trials, base_dir=base_dir, subsets=subsets,
            batch_size=args.batch_size, eval_loss=True, device=device,
            num_workers=args.num_workers, pin_memory=args.pin_memory, persistent_workers=args.persistent_workers,
            non_blocking=args.non_blocking, min_epochs=min_epochs, max_epochs=max_epochs, verbose=args.verbose)
    if args.evaluate_and_plot:
        run_models_on_subsets_and_plot(
            path, args.n_classes, args.n_attr, subsets=subsets, n_bootstrap=args.n_bootstrap,
            fast=args.fast, batch_size=args.batch_size, non_blocking=args.non_blocking, num_workers=args.num_workers,
            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers, verbose=args.verbose)
