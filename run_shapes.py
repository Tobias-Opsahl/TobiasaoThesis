"""
Testing function where the code can be run.
"""
import argparse
import torch
import torch.nn as nn
import numpy as np
from train import train_simple, train_cbm
from shapes.datasets_shapes import load_data_shapes
from shapes.models_shapes import ShapesCNN, ShapesCBM, ShapesCBMWithResidual, ShapesCBMWithSkip
from utils import seed_everything


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for parsing command-line arguments.")

    parser.add_argument("--dataset_folder", type=str, default="shapes_testing/", help="Name of dataset.")
    parser.add_argument("--base_dir", type=str, default="data/shapes/", help="Path to where dataset is.")
    parser.add_argument("--subset_dir", type=str, default="", help="Name of optional subset.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--n_classes", type=int, default=4, help="Number of classes.")
    parser.add_argument("--n_attr", type=int, default=7, help="Number of attributes.")
    parser.add_argument("--n_epochs", type=int, default=15, help="Number of epochs for training.")
    parser.add_argument("--n_hidden", type=int, default=16, help="Number of hidden units in cbm skip connection.")
    parser.add_argument("--n_linear_output", type=int, default=64, help="Number of nodes in first linear layer")
    parser.add_argument("--attr_weight", type=int, default=1, help="Weight for attributes.")
    parser.add_argument("--sigmoid", action="store_true", help="Use sigmoid activation in concept layer.")
    parser.add_argument("--relu", action="store_true", help="Use relu activation in concept layer.")
    parser.add_argument("--two_layers", action="store_true", help="Use two layers in CBM model after bottleneck.")
    parser.add_argument("--small", action="store_true", help="Use a small model. If not specified, defaults to False.")
    parser.add_argument("--device", type=str, default=None, help="Device to train on. cuda:0 or CPU.")
    parser.add_argument("--non_blocking", action="store_true", help="Allows asynchronous RAM to VRAM operations.")
    parser.add_argument("--num_workers", type=int, default=0, help="Amount of workers to load data to RAM.")
    parser.add_argument("--pin_memory", action="store_true", help="Pins the RAM memory (makes it non-pagable).")
    parser.add_argument("--persistent_workers", action="store_true", help="Do not reload workers between epochs")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    seed_everything(57)
    args = parse_arguments()

    if args.device is None or args.device == "":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    path = args.base_dir + args.dataset_folder
    train_loader, val_loader, test_loader = load_data_shapes(path=path,
                                                             subset_dir=args.subset_dir,
                                                             batch_size=args.batch_size,
                                                             drop_last=True,
                                                             num_workers=args.num_workers,
                                                             pin_memory=args.pin_memory,
                                                             persistent_workers=args.persistent_workers)
    # Make models
    cnn = ShapesCNN(n_classes=args.n_classes, n_linear_output=args.n_linear_output)
    cbm = ShapesCBM(n_classes=args.n_classes, n_attr=args.n_attr, n_linear_output=args.n_linear_output,
                    use_sigmoid=args.sigmoid, use_relu=args.relu, two_layers=args.two_layers)
    cbm_res = ShapesCBMWithResidual(n_classes=args.n_classes, n_attr=args.n_attr, n_linear_output=args.n_linear_output,
                                    use_sigmoid=args.sigmoid, use_relu=args.relu)
    cbm_skip = ShapesCBMWithSkip(n_classes=args.n_classes, n_attr=args.n_attr, n_hidden=args.n_hidden,
                                 n_linear_output=args.n_linear_output, use_sigmoid=args.sigmoid, use_relu=args.relu)

    criterion = nn.CrossEntropyLoss()
    attr_criterion = nn.BCEWithLogitsLoss()
    models = [cnn, cbm, cbm_res, cbm_skip]

    # Loop over models and train
    for model in models:
        print(f"Running model {model.name}:")
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
        if model.name == "ShapesCNN":
            model = train_simple(model, criterion, optimizer, train_loader, val_loader, scheduler=exp_lr_scheduler,
                                 n_epochs=args.n_epochs, device=device, non_blocking=args.non_blocking)
        else:
            model = train_cbm(model, criterion, attr_criterion, optimizer, train_loader, val_loader,
                              n_epochs=args.n_epochs, attr_weight=args.attr_weight, scheduler=exp_lr_scheduler,
                              device=device, non_blocking=args.non_blocking)

    from IPython import embed
    embed()
