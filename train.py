import os
import pickle
import torch


def train_simple(model, criterion, optimizer, train_loader, val_loader=None, n_epochs=10,
                 scheduler=None, device=None, non_blocking=False, model_dir="saved_models/", model_save_name=None,
                 history_dir="history/", history_save_name=None, verbose=2):
    """
    Trains a model and calculate training and valudation stats, given the model, loader, optimizer
    and some hyperparameters.
    If `val_loader` and `model_save_name` is not None, will save the model that had the best accuracy on validation set.

    Args:
        model (model): The model to train. Freeze layers ahead of calling this function.
        criterion (callable): Pytorch loss function.
        optimizer (optim): Pytorch Optimizer.
        train_loader (dataloader): Data loader for training set
        val_loader (dataloader, optional): Optinal validation data loader.
            If not None, will calculate validation loss and accuracy after each epoch.
        n_epochs (int, optional): Amount of epochs to run. Defaults to 10.
        scheduler (scheduler, optional): Optional learning rate scheduler.
        device (str): Use "cpu" for cpu training and "cuda:0" for gpu training.
        non_blocking (bool): If True, allows for asyncronous transfer between RAM and VRAM.
            This only works together with `pin_memory=True` to dataloader and GPU training.
        model_dir (str): The directory to save the best model, if `model_save_name` is not None.
        model_save_name (str): If not None, will be the name of the saved best model.
        history_dir (str): The directory to save the pickled file of the training history, if
            `history_save_name` is not None.
        history_save_name (str): If not None, will save the training history as this name.
        verbose (int): Determines how much output is printed. If 2, will print stats after every epoch.
            If 1, will print after last epoch only. If 0, will not print anything.

    Returns:
        dict: A dictionary of the training history. Will contain lists of training loss and accuracy over
            epochs, and for validation loss and accuracy if `val_loader` is not None.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device, non_blocking=non_blocking)
    train_class_loss_list = []  # Initialize training history variables
    train_class_accuracy_list = []
    val_class_loss_list = []  # These will remain empty if `val_loader` is None
    val_class_accuracy_list = []
    best_epoch_number = 0
    best_val_accuracy = 0
    best_model = None  # This will only be saved if `val_loader` is not None

    for epoch in range(n_epochs):  # Train
        train_loss = 0
        train_correct = 0
        model.train()
        for input, labels, attr_labels, paths in train_loader:
            input = input.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)

            optimizer.zero_grad()
            outputs = model(input)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * input.shape[0]
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()

        average_train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = 100 * train_correct / len(train_loader.dataset)
        train_class_loss_list.append(average_train_loss)
        train_class_accuracy_list.append(train_accuracy)

        if val_loader is not None:  # Eval
            model.eval()
            val_loss = 0
            val_correct = 0
            for input, labels, attr_labels, paths in val_loader:
                input = input.to(device, non_blocking=non_blocking)
                labels = labels.to(device, non_blocking=non_blocking)
                optimizer.zero_grad()
                outputs = model(input)

                loss = criterion(outputs, labels)

                val_loss += loss.item() * input.shape[0]
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()

            average_val_loss = val_loss / len(val_loader.dataset)
            val_accuracy = 100 * val_correct / len(val_loader.dataset)
            val_class_loss_list.append(average_val_loss)
            val_class_accuracy_list.append(val_accuracy)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_epoch_number = epoch
                best_model = model.state_dict()

        if (verbose == 2) or ((verbose == 1) and (epoch + 1 == n_epochs)):
            print(f"Epoch [{epoch + 1} / {n_epochs}]")
            print(f"Train loss: {average_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}%")
            if val_loader is not None:
                print(f"Validation loss: {average_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}%")
            print()

        if scheduler is not None:
            scheduler.step()

    history = {"train_class_loss": train_class_loss_list, "train_class_accuracy": train_class_accuracy_list,
               "val_class_loss": val_class_loss_list, "val_class_accuracy": val_class_accuracy_list,
               "best_epoch": best_epoch_number, "best_val_accuracy": best_val_accuracy,
               "model_save_name": model_save_name}

    if model_save_name is not None:
        if not model_save_name.endswith(".pth"):
            model_save_name += ".pth"
        os.makedirs(model_dir, exist_ok=True)
        torch.save(best_model, model_dir + model_save_name)
    if history_save_name is not None:
        if not history_save_name.endswith(".pkl"):
            history_save_name += ".pkl"
        with open(history_dir + history_save_name, "wb") as outfile:
            pickle.dump(history, outfile)
    return history


def train_cbm(model, criterion, attr_criterion, optimizer, train_loader, val_loader=None, n_epochs=10, attr_weight=0.01,
              scheduler=None, device=None, non_blocking=False, model_dir="saved_models/", model_save_name=None,
              history_dir="history/", history_save_name=None, verbose=2):
    """
    Trains and evaluates a Joint Concept Bottleneck Model. This means it is both trained normal
    output cross entropy loss, but also on the intermediary attribute loss.
    If `val_loader` and `model_save_name` is not None, will save the model that had the best accuracy on validation set.

    Args:
        model (model): The model to train. Freeze layers ahead of calling this function.
        criterion (callable): Pytorch loss function for the output.
        attr_criterion (callable): Pytorch loss function for the attributes.
        optimizer (optim): Pytorch Optimizer.
        train_loader (dataloader): Data loader for training set
        val_loader (dataloader, optional): Optinal validation data loader.
            If not None, will calculate validation loss and accuracy after each epoch.
        n_epochs (int, optional): Amount of epochs to run. Defaults to 10.
        attr_weight (float): The weight of the attribute loss function. Equal to Lambda in
            the Concept Bottleneck Models paper.
        scheduler (scheduler, optional): Optional learning rate scheduler.
        device (str): Use "cpu" for cpu training and "cuda:0" for gpu training.
        non_blocking (bool): If True, allows for asyncronous transfer between RAM and VRAM.
            This only works together with `pin_memory=True` to dataloader and GPU training.
        model_dir (str): The directory to save the best model, if `model_save_name` is not None.
        model_save_name (str): If not None, will be the name of the saved best model.
        history_dir (str): The directory to save the pickled file of the training history, if
            `history_save_name` is not None.
        history_save_name (str): If not None, will save the training history as this name.
        verbose (int): Determines how much output is printed. If 2, will print stats after every epoch.
            If 1, will print after last epoch only. If 0, will not print anything.

    Returns:
        dict: A dictionary of the training history. Will contain lists of training loss and accuracy over
            epochs, and for validation loss and accuracy if `val_loader` is not None. This is done for both the
            class and the attributes.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device, non_blocking=non_blocking)
    best_model = None  # This will only be saved if `val_loader` is not None
    train_class_loss_list = []  # Initialize training history variables
    train_class_accuracy_list = []
    val_class_loss_list = []  # These will remain empty if `val_loader` is None
    val_class_accuracy_list = []
    train_attr_loss_list = []
    train_attr_accuracy_list = []
    val_attr_loss_list = []
    val_attr_accuracy_list = []
    best_epoch_number = 0
    best_val_accuracy = 0
    best_model = None  # This will only be saved if `val_loader` is not None

    for epoch in range(n_epochs):  # Train
        train_attr_loss = 0
        train_attr_correct = 0
        train_class_loss = 0
        train_class_correct = 0
        model.train()
        for input, labels, attr_labels, paths in train_loader:
            input = input.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            attr_labels = attr_labels.to(device, non_blocking=non_blocking)
            optimizer.zero_grad()
            class_outputs, attr_outputs = model(input)

            class_loss = criterion(class_outputs, labels)
            attr_loss = attr_weight * attr_criterion(attr_outputs, attr_labels)
            loss = class_loss + attr_loss
            loss.backward()
            optimizer.step()

            train_attr_loss += attr_loss.item() * input.shape[0]
            attr_preds = (attr_outputs > 0.5).float()
            train_attr_correct += (attr_preds == attr_labels).sum().item()

            train_class_loss += class_loss.item() * input.shape[0]
            _, preds = torch.max(class_outputs, 1)
            train_class_correct += (preds == labels).sum().item()

        average_train_attr_loss = train_attr_loss / len(train_loader.dataset)
        train_attr_accuracy = 100 * train_attr_correct / (len(train_loader.dataset) * attr_labels.shape[1])
        average_train_class_loss = train_class_loss / len(train_loader.dataset)
        train_class_accuracy = 100 * train_class_correct / len(train_loader.dataset)

        train_attr_loss_list.append(average_train_attr_loss)
        train_attr_accuracy_list.append(train_attr_accuracy)
        train_class_loss_list.append(average_train_class_loss)
        train_class_accuracy_list.append(train_class_accuracy)

        if val_loader is not None:  # Eval
            model.eval()
            val_attr_loss = 0
            val_attr_correct = 0
            val_class_loss = 0
            val_class_correct = 0
            for input, labels, attr_labels, paths in val_loader:
                input = input.to(device, non_blocking=non_blocking)
                labels = labels.to(device, non_blocking=non_blocking)
                attr_labels = attr_labels.to(device, non_blocking=non_blocking)
                optimizer.zero_grad()
                class_outputs, attr_outputs = model(input)

                class_loss = criterion(class_outputs, labels)
                attr_loss = attr_weight * attr_criterion(attr_outputs, attr_labels)
                loss = class_loss + attr_loss

                val_attr_loss += attr_loss.item() * input.shape[0]
                attr_preds = (attr_outputs > 0.5).float()
                val_attr_correct += (attr_preds == attr_labels).sum().item()

                val_class_loss += class_loss.item() * input.shape[0]
                _, preds = torch.max(class_outputs, 1)
                val_class_correct += (preds == labels).sum().item()

            average_val_attr_loss = val_attr_loss / len(val_loader.dataset)
            val_attr_accuracy = 100 * val_attr_correct / (len(val_loader.dataset) * attr_labels.shape[1])
            average_val_class_loss = val_class_loss / len(val_loader.dataset)
            val_class_accuracy = 100 * val_class_correct / len(val_loader.dataset)

            val_attr_loss_list.append(average_val_attr_loss)
            val_attr_accuracy_list.append(val_attr_accuracy)
            val_class_loss_list.append(average_val_class_loss)
            val_class_accuracy_list.append(val_class_accuracy)

            if val_class_accuracy > best_val_accuracy:
                best_val_accuracy = val_class_accuracy
                best_epoch_number = epoch
                best_model = model.state_dict()

        if (verbose == 2) or ((verbose == 1) and (epoch + 1 == n_epochs)):
            print(f"Epoch [{epoch + 1} / {n_epochs}]")
            print(f"Train atribute loss: {average_train_attr_loss:.4f}, ", end="")
            print(f"Train attribute accuracy: {train_attr_accuracy:.4f}%")
            print(f"Train class loss: {average_train_class_loss:.4f}, Train class accuracy: {train_class_accuracy:.4f}%")

            if val_loader is not None:
                print(f"Val atribute loss: {average_val_attr_loss:.4f}, ", end="")
                print(f"Val attribute accuracy: {val_attr_accuracy:.4f}%")
                print(f"Val class loss: {average_val_class_loss:.4f}, Val class accuracy: {val_class_accuracy:.4f}%")
            print()

        if scheduler is not None:
            scheduler.step()

    history = {"train_class_loss": train_class_loss_list, "train_class_accuracy": train_class_accuracy_list,
               "val_class_loss": val_class_loss_list, "val_class_accuracy": val_class_accuracy_list,
               "train_attr_loss": train_attr_loss_list, "train_attr_accuracy": train_attr_accuracy_list,
               "val_attr_loss": val_attr_loss_list, "val_attr_accuracy": val_attr_accuracy_list,
               "best_epoch": best_epoch_number, "best_val_accuracy": best_val_accuracy,
               "model_save_name": model_save_name}

    if model_save_name is not None:
        if not model_save_name.endswith(".pth"):
            model_save_name += ".pth"
        os.makedirs(model_dir, exist_ok=True)
        torch.save(best_model, model_dir + model_save_name)
    if history_save_name is not None:
        if not history_save_name.endswith(".pkl"):
            history_save_name += ".pkl"
        with open(history_dir + history_save_name, "wb") as outfile:
            pickle.dump(history, outfile)
    return history
