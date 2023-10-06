"""
Code that is depricated, but might be interesting to look back at frequently. It is placed here to save time
on navigating old files with Git.
"""
import os
import pickle
import random
from utils import split_dataset


def make_subset_shapes(path, n_images_class, n_classes, split_data=True, include_test=False, seed=57):
    """
    Makes a subset of a data-table. This can then be split into "train", "validation" and "test".
    This makes it possible to easily train with less data for a given dataset. For example, if there are
    5k images for each class in a dataset, one can call this function with `n_images_class=100` to make
    new data-tables. The path of these can then be sent into the dataloader, so that only 100 images
    (which are randomly drawn here) will be used in the dataloader.

    Args:
        path (str): Path to where the data-list lies.
        n_images_class (int): The amount of images to include for each class. Must be less than the total number
            of images in each class.
        n_classes (int): The total amount of classes in the dataset.
        split_data (bool, optional): If True, splits into "train", "validation" and maybe "test". Defaults to True.
        include_test (bool, optional): If True, will incldue test-set in data-split. If False, will only include
            "train" and "validation".
        seed (int, optional): The seed for the rng. Defaults to 57.
    """
    random.seed(seed)
    if not path.endswith("tables/"):
        path = path + "tables/"
    data_list = pickle.load(open(path + "train_data.pkl", "rb"))  # Use train data to exclude test-dataset

    class_dict = {}  # Sort dicts by label
    for i in range(n_classes):
        class_dict[i] = []
    for i in range(len(data_list)):
        class_dict[data_list[i]["class_label"]].append(data_list[i])

    new_list = []  # The sub-list of datapoints we want to create
    for i in range(n_classes):
        # shuffled_list = class_dict[i]  # New way, was not in use.
        # random.shuffle(shuffled_list)
        # sub_list = shuffled_list[:n_images_class]
        sub_list = random.sample(class_dict[i], n_images_class)  # Old way
        new_list.extend(sub_list)

    tables_dir = path + "sub" + str(n_images_class) + "/"
    if split_data:
        split_dataset(new_list, tables_dir, include_test=include_test)
    os.makedirs(tables_dir, exist_ok=True)
    with open(tables_dir + "data_list.pkl", "wb") as outfile:
        pickle.dump(new_list, outfile)


def make_shapes_test_set(path, n_images_class, n_classes, n_images_class_test_set=500, split_data=True, base_seed=57):
    """
    This function is probably only used temporarly, at a time where `make_subset_shapes` drew from `data_list.pkl`
    instead of `train_data.pkl`, making the need of this function for making realiable test data.

    Make a big test set for a sub-list of a dataset, avoiding the images that were used to tune hyperparameters.
    The `base_seed` should be the seed that were used to make the sub-list of data, with `make_subset_shapes`,
    that were then used to tune hyperparameters on. We want to avoid all of those instances, to avoid overlap
    between images used for tuning and images used in this test-set.
    We first get the paths that were used in the original sub-list, then make a dictionary including all images in
    the dataset *except* the paths used in the original sub-list. Finally, we make a subset of size `n_images_class_test_set`
    of this.
    Note that this could be computed in `make_subsets_shapes` slightly more efficient, but this function were added for
    backwards compatibility. The operation only needs to be ran once, and takes second at most, so the inefficiency
    is not considered significant.

    Args:
        path (str): Path to the dataset.
        n_images_class (int): Amount of subset of data used in each class in the sub-list.
        n_classes (int): Amount of classes in the dataset.
        n_images_class_test_set (int, optional): Amount of images for each class in the test set. Defaults to 500.
        split_data (bool, optional): Wether to split data in sub-list. Defaults to True.
        base_seed (int, optional): The seed used for the original sub-list. Defaults to 57.
    """
    # Make the subset to be sure the seed is correct
    make_subset_shapes(path, n_images_class, n_classes, split_data=split_data, seed=base_seed)
    if not path.endswith("tables/"):
        path = path + "tables/"
    tables_dir = path + "sub" + str(n_images_class) + "/"
    # First find all the indices used in sub data_list, so we can avoid them in the test-set
    data_list = pickle.load(open(tables_dir + "data_list.pkl", "rb"))  # Data-list of subset
    banned_paths = []  # Paths we do not want in our test-set
    for i in range(len(data_list)):
        banned_paths.append(data_list[i]["img_path"])

    # Now start making the real test-dataset
    full_data_list = pickle.load(open(path + "data_list.pkl", "rb"))  # Data-list of full dataset
    test_class_dict = {}  # Make dict with class-labels pointing to lists of instances of that class
    for i in range(n_classes):
        test_class_dict[i] = []
    for i in range(len(full_data_list)):
        if full_data_list[i]["img_path"] not in banned_paths:
            test_class_dict[full_data_list[i]["class_label"]].append(full_data_list[i])
    random.seed(base_seed)
    new_list = []  # The sub-list of datapoints we want to create
    for i in range(n_classes):  # Make balanced list of each class
        sub_list = random.sample(test_class_dict[i], n_images_class_test_set)
        new_list.extend(sub_list)
    with open(tables_dir + "full_test_data_" + str(n_images_class_test_set) + ".pkl", "wb") as outfile:
        pickle.dump(new_list, outfile)
