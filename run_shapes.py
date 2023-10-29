"""
Testing function where the code can be run.
"""
import argparse
import torch

from src.common.utils import seed_everything, parse_int_list
from src.hyperparameter_optimization import run_hyperparameter_optimization_all_models
from src.evaluation import run_models_on_subsets_and_plot


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for parsing command-line arguments.")

    parser.add_argument("--run_hyperparameters", action="store_true", help="Run hyperparameter search.")
    parser.add_argument("--no_grid_search", action="store_true", help="Do not run grid-search, but TPEsampler.")
    parser.add_argument("--evaluate_and_plot", action="store_true", help="Evaluate models and plot.")

    # Parameters
    parser.add_argument("--n_classes", type=int, default=10, help="Number of classes.")
    parser.add_argument("--signal_strength", type=int, default=98, help="Signal strength for dataset")
    parser.add_argument("--n_attr", type=int, default=5, help="Number of attributes.")
    parser.add_argument("--n_bootstrap", type=int, default=1, help="number of bootstrap iterations to run")
    help = "Number of bootstrap iterations to save at. Can be single int or list of int, for example `1` or `1,5,10`."
    parser.add_argument("--bootstrap_checkpoints", type=parse_int_list, help=help)
    # parser.add_argument("--n_bootstrap", type=int, default=1, help="Number of bootstrap iterations.")
    help = "Sizes of subsets to run on. Can be single int or list of int, for example `50` or `50,100,150`."
    parser.add_argument("--subsets", type=parse_int_list, default=[29, 31], help=help)
    parser.add_argument("--n_trials", type=int, default=100, help="Number of trials for hyperparameter search.")
    parser.add_argument("--fast", action="store_true", help="Use fast testing hyperparameters.")

    # Hardware and div stuff:
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--device", type=str, default=None, help="Device to train on. cuda:0 or CPU.")
    parser.add_argument("--non_blocking", action="store_true", help="Allows asynchronous RAM to VRAM operations.")
    parser.add_argument("--num_workers", type=int, default=0, help="Amount of workers to load data to RAM.")
    parser.add_argument("--pin_memory", action="store_true", help="Pins the RAM memory (makes it non-pagable).")
    parser.add_argument("--persistent_workers", action="store_true", help="Do not reload workers between epochs")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity of training (2, 1 or 0).")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    seed_everything(57)
    args = parse_arguments()

    device = args.device
    if args.device is None or args.device == "":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    grid_search = not args.no_grid_search

    if args.run_hyperparameters:
        run_hyperparameter_optimization_all_models(
            args.n_classes, args.n_attr, signal_strength=args.signal_strength, n_trials=args.n_trials,
            grid_search=grid_search, subsets=args.subsets, batch_size=args.batch_size,
            eval_loss=True, device=device, num_workers=args.num_workers, pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers, non_blocking=args.non_blocking, fast=args.fast,
            verbose=args.verbose)

    if args.evaluate_and_plot:
        print(f"\nBeginning evaluation with {args.n_bootstrap} bootstrap iterations.\n")
        run_models_on_subsets_and_plot(
            args.n_classes, args.n_attr, signal_strength=args.signal_strength, subsets=args.subsets,
            n_bootstrap=args.n_bootstrap, bootstrap_checkpoints=args.bootstrap_checkpoints, fast=args.fast,
            batch_size=args.batch_size, non_blocking=args.non_blocking, num_workers=args.num_workers,
            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers, verbose=args.verbose)
