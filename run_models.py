"""
Testing function where the code can be run
"""
import argparse
import torch
import torch.nn as nn
import numpy as np
from dataset import load_data
from torch.optim import lr_scheduler
from models import (make_mobilenetv3, make_attr_model_single, make_attr_model_double,
                    CombineModels, LinearWithInput, CBMWithSkip, make_cbm, SequentialConceptModel)
from train import train_simple, train_cbm


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for parsing command-line arguments.")

    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--n_classes", type=int, default=200, help="Number of classes.")
    parser.add_argument("--n_attr", type=int, default=112, help="Number of attributes.")
    parser.add_argument("--n_epochs", type=int, default=20, help="Number of epochs for training.")
    parser.add_argument("--attr_weight", type=int, default=1, help="Weight for attributes.")
    parser.add_argument("--sigmoid", action="store_true", help="Use sigmoid activation in concept layer.")
    parser.add_argument("--n_hidden", type=int, default=256, help="Number of hidden unitsin cbm skip connection.")
    parser.add_argument("--small", action="store_true", help="Use a small model. If not specified, defaults to False.")
    parser.add_argument("--device", type=str, default=None, help="Device to train on. cuda:0 or CPU.")
    parser.add_argument("--non_blocking", action="store_true", help="Allows asynchronous RAM to VRAM operations.")
    parser.add_argument("--num_workers", type=int, default=0, help="Amount of workers to load data to RAM.")
    parser.add_argument("--pin_memory", action="store_true", help="Pins the RAM memory (makes it non-pagable).")
    parser.add_argument("--persistent_workers", action="store_true", help="Do not reload workers between epochs")

    args = parser.parse_args()
    return args


def seed_everything(seed=57):
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    seed_everything(57)
    train_cbm_models = True
    train_standard = True
    args = parse_arguments()
    batch_size = args.batch_size
    n_classes = args.n_classes
    n_attr = args.n_attr
    n_epochs = args.n_epochs
    attr_weight = args.attr_weight
    sigmoid = args.sigmoid
    n_hidden = args.n_hidden
    small = args.small
    non_blocking = args.non_blocking
    num_workers = args.num_workers
    pin_memory = args.pin_memory
    persistent_workers = args.persistent_workers
    device = args.device
    if args.device is None or args.device == "":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = load_data(subset=n_classes, n_attr=n_attr, batch_size=batch_size,
                                                      drop_last=True, num_workers=num_workers, pin_memory=pin_memory,
                                                      persistent_workers=persistent_workers)

    cbm = make_cbm(n_classes=n_classes, n_attr=n_attr, sigmoid=sigmoid, double_top=True,
                   n_hidden=n_hidden, small=small)
    cbm_skip1 = CBMWithSkip(n_attr=n_attr, n_output=n_classes, n_hidden=n_hidden,
                            concat=False, sigmoid=sigmoid, small=small)
    cbm_skip2 = CBMWithSkip(n_attr=n_attr, n_output=n_classes, n_hidden=n_hidden,
                            concat=True, sigmoid=sigmoid, small=small)
    standard = make_mobilenetv3(n_output=n_classes, small=small)

    criterion = nn.CrossEntropyLoss()
    if sigmoid:
        attr_criterion = nn.BCELoss()
    else:
        attr_criterion = nn.BCEWithLogitsLoss()

    if train_cbm_models:
        models = [cbm, cbm_skip1, cbm_skip2]
        model_names = ["cbm", "cbm_skip1", "cbm_skip2"]
        for i, model in enumerate([cbm, cbm_skip1, cbm_skip2]):
            print(f"Running model {i + 1}, {model_names[i]}\n")
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.3)
            model = train_cbm(model, criterion, attr_criterion, optimizer, train_loader, val_loader,
                              n_epochs=n_epochs, attr_weight=attr_weight, scheduler=exp_lr_scheduler,
                              device=device, non_blocking=non_blocking)

    if train_standard:
        for model in [standard]:
            print("Running standard model\n")
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.3)
            model = train_simple(model, criterion, optimizer, train_loader, val_loader,
                                 n_epochs=n_epochs, scheduler=exp_lr_scheduler,
                                 device=device, non_blocking=non_blocking)

    from IPython import embed
    embed()
