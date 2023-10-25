"""
Testing function where the code can be run.
"""
import argparse
import torch
import torch.nn as nn
import numpy as np
from train import train_simple, train_cbm

from src.datasets.datasets_cub import load_data_cub
from src.models.models_cub import (make_mobilenetv3, CBMWithSkip, make_cbm, SequentialConceptModel)
from src.common.utils import seed_everything


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for parsing command-line arguments.")

    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--n_classes", type=int, default=10, help="Number of classes.")
    parser.add_argument("--n_attr", type=int, default=112, help="Number of attributes.")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("--n_hidden", type=int, default=128, help="Number of hidden units in cbm skip connection.")
    parser.add_argument("--n_linear_output", type=int, default=64, help="Number of nodes in first linear layer")
    parser.add_argument("--attr_weight", type=int, default=1, help="Weight for attributes.")
    parser.add_argument("--sigmoid", action="store_true", help="Use sigmoid activation in concept layer.")
    parser.add_argument("--relu", action="store_true", help="Use relu activation in concept layer.")
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

    train_loader, val_loader, test_loader = load_data_cub(
        subset=args.n_classes, n_attr=args.n_attr, batch_size=args.batch_size, drop_last=True,
        num_workers=args.num_workers, pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)

    cbm = make_cbm(n_classes=args.n_classes, n_attr=args.n_attr, sigmoid=args.sigmoid, double_top=True,
                   n_hidden=args.n_hidden, small=args.small)
    cbm_skip1 = CBMWithSkip(n_attr=args.n_attr, n_output=args.n_classes, n_hidden=args.n_hidden,
                            concat=False, sigmoid=args.sigmoid, small=args.small)
    cbm_skip2 = CBMWithSkip(n_attr=args.n_attr, n_output=args.n_classes, n_hidden=args.n_hidden,
                            concat=True, sigmoid=args.sigmoid, small=args.small)
    standard = make_mobilenetv3(n_output=args.n_classes, small=args.small)

    criterion = nn.CrossEntropyLoss()
    if args.sigmoid:
        attr_criterion = nn.BCELoss()
    else:
        attr_criterion = nn.BCEWithLogitsLoss()

    models = [cbm, cbm_skip1, cbm_skip2]
    model_names = ["cbm", "cbm_skip1", "cbm_skip2"]
    for i, model in enumerate([cbm, cbm_skip1, cbm_skip2]):
        print(f"Running model {i + 1}, {model_names[i]}\n")
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.3)
        model = train_cbm(model, criterion, attr_criterion, optimizer, train_loader, val_loader,
                          n_epochs=args.n_epochs, attr_weight=args.attr_weight, scheduler=exp_lr_scheduler,
                          device=device, non_blocking=args.non_blocking)

    for model in [standard]:
        print("Running standard model\n")
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.3)
        model = train_simple(model, criterion, optimizer, train_loader, val_loader,
                             n_epochs=args.n_epochs, scheduler=exp_lr_scheduler,
                             device=device, non_blocking=args.non_blocking)

    from IPython import embed
    embed()
