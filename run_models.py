"""
Testing function where the code can be run
"""
import torch
import torch.nn as nn
from dataset import load_data
from torch.optim import lr_scheduler
from models import (make_mobilenetv3, make_attr_model_single, make_attr_model_double,
                    CombineModels, LinearWithInput, CBMWithSkip, make_cbm)
from train import train_simple, train_cbm


if __name__ == "__main__":
    n_classes = 10
    n_attr = 112
    n_epochs = 10
    sigmoid = True
    n_hidden = 64
    small = True
    train_loader, val_loader, test_loader = load_data(subset=n_classes)

    cbm = make_cbm(n_classes=n_classes, n_attr=n_attr, sigmoid=sigmoid, double_top=True,
                   n_hidden=n_hidden, small=small)
    cbm_skip1 = CBMWithSkip(n_attr=n_attr, n_output=n_classes, n_hidden=n_hidden,
                            concat=False, sigmoid=sigmoid, small=small)
    cbm_skip2 = CBMWithSkip(n_attr=n_attr, n_output=n_classes, n_hidden=n_hidden,
                            concat=True, sigmoid=sigmoid, small=small)
    standard = make_mobilenetv3(n_output=200, small=small)

    criterion = nn.CrossEntropyLoss()
    if sigmoid:
        attr_criterion = nn.BCELoss()
    else:
        attr_criterion = nn.BCEWithLogitsLoss()

    for i, model in enumerate([cbm, cbm_skip1, cbm_skip2]):
        print(i)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.2)
        model = train_cbm(model, criterion, attr_criterion, optimizer, train_loader, val_loader,
                          n_epochs=n_epochs, attr_weight=2, scheduler=exp_lr_scheduler)

    for model in [standard]:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.2)
        model = train_simple(model, criterion, optimizer, train_loader, val_loader,
                             n_epochs=n_epochs, scheduler=exp_lr_scheduler)
    from IPython import embed
    embed()
