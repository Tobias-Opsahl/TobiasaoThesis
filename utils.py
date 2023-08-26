import pickle
import numpy as np


def make_correct_paths(base_path="data/"):
    """
    Reads the preprocessed CUB dictionaries and overwrites the base path.
    The original published dataset had the authors full path in the `img_path`
    attribute.
    Saves the new pickled list of dicts as `train.pkl` for train, val and test.

    Args:
        base_path (str, optional): The base-path one. Defaults to "data/".
            The folders `CUB_processed/class_attr_data_10/` should be inside
            this folder.
    """
    mid_path = "CUB_processed/class_attr_data_10/"
    for dataset in ["train", "val", "test"]:
        full_path = base_path + mid_path + dataset + ".pkl"
        data_list = pickle.load(open(full_path, "rb"))
        # Data is a list of each observation
        for row in data_list:  # Loop over each dict
            # Overwrite each path with correct path
            row["img_path"] = base_path + row["img_path"].split("datasets/")[1]

        with open(base_path + mid_path + dataset + ".pkl", "wb") as outfile:
            pickle.dump(data_list, outfile)


def make_small_dataset(n_classes=10, random_choice=True):
    """
    Makes a subset of the list of dictionaries, only containing some of the
    original classes. The original image-files are untouched, since the dictionaries
    contains paths to the images.

    The new pickle files will be stored in the same place as the old ones,
    where the name starts with the amount of classes followed by underscore.

    Args:
        n_classes (int, optional): Amonut of classes to use. Defaults to 10.
        random_choice (bool, optional): If True, will chose indices (among the 200)
            at random. If false, chooses the first from 1 to `n_classes` + 1.
            Defaults to True.
    """
    total_classes = 200
    if random_choice:  # Choose indices
        indices = np.random.choice(total_classes, n_classes, replace=False)
    else:
        indices = np.arange(n_classes)
    indices = indices + 1  # Image folders start indexing at 1
    indices.sort()

    preprocess_dir = "data/CUB_processed/class_attr_data_10/"
    for mode in ["train", "val", "test"]:  # Make a new dict of every mode
        new_list = []
        full_path = preprocess_dir + mode + ".pkl"
        data_list = pickle.load(open(full_path, "rb"))
        for im_dict in data_list:  # For every dict in the list
            index = int(im_dict["img_path"].split("/")[-2].split(".")[0])
            if index in indices:  # Check if img-path is of correct class
                new_list.append(im_dict)
        with open(preprocess_dir + str(n_classes) + "_" + mode + ".pkl", "wb") as outfile:
            pickle.dump(new_list, outfile)

    mapping = {}  # Save mapping from indices to 0, ..., n_classes (for training)
    for i in range(n_classes):
        mapping[indices[i] - 1] = i  # Indices in labels start from 0

    with open(preprocess_dir + str(n_classes) + "_class_mapping.pkl", "wb") as outfile:
        pickle.dump(mapping, outfile)
