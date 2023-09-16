import random
import pickle
import torch
import numpy as np


def seed_everything(seed=57):
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def split_dataset(data_list, tables_dir, seed=57):
    """
    Splits a dataset into "train", "validation" and "test".

    Args:
        data_list (list): List of rows, each element is an instance-dict.
        tables_dir (str): The path to save the data-tables
        seed (int, optional): Seed for the rng. Defaults to 57.
    """
    random.seed(seed)
    n_images = len(data_list)
    random.shuffle(data_list)
    train_size = int(0.5 * n_images)
    val_size = int(0.3 * n_images)

    train_data = data_list[: train_size]
    val_data = data_list[train_size: train_size + val_size]
    test_data = data_list[train_size + val_size:]
    with open(tables_dir + "train_data.pkl", "wb") as outfile:
        pickle.dump(train_data, outfile)
    with open(tables_dir + "val_data.pkl", "wb") as outfile:
        pickle.dump(val_data, outfile)
    with open(tables_dir + "test_data.pkl", "wb") as outfile:
        pickle.dump(test_data, outfile)
