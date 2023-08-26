"""
Testing function where the code can be run
"""
import torch
import torch.nn as nn
from dataset import load_data
from torch.optim import lr_scheduler
from models import make_mobilenetv3, make_attr_model_single, make_attr_model_double, CombineModels
from train import train_model_simple, train_example, train_cbm


if __name__ == "__main__":
    n_classes = 10
    n_attr = 112
    train_loader, val_loader, test_loader = load_data(subset=10)

    mobilenetv3 = make_mobilenetv3(n_output=n_attr, small=True)
    # top_model = make_attr_model_double(n_input=n_attr, n_hidden=50, n_output=n_classes)
    top_model = make_attr_model_single(n_input=n_attr, n_output=n_classes)
    activation_func = "sigmoid"
    # activation_func = None
    model = CombineModels(mobilenetv3, top_model, activation_func)

    criterion = nn.CrossEntropyLoss()
    if activation_func == "sigmoid":
        attr_criterion = nn.BCELoss()
    else:
        attr_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.2)
    model = train_cbm(model, criterion, attr_criterion, optimizer, train_loader, val_loader,
                      n_epochs=10, attr_weight=1, scheduler=exp_lr_scheduler)

    from IPython import embed
    embed()
