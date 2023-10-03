import os
import pickle
import random
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from utils import split_dataset


class ShapesDataset(Dataset):
    """
    Dataset for shapes dataset. The `shapes` datasets are created in `make_shapes_datasets.py`.
    """

    def __init__(self, data_path, data_list, transform=None):
        """

        Args:
            data_path (str): Path to the directory where the dataset is stored.
                Includes both the path and the foldername.
            data_list (str): The path to the pickle file with table-data.
            transform (torch.transform, optional): Optional transformation on the data.
        """
        self.data_path = data_path
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path = self.data_list[idx]["img_path"]
        class_label = self.data_list[idx]["class_label"]
        # Convert from dict to list to tensor
        attribute_label = torch.tensor(list(self.data_list[idx]["attribute_label"].values())).to(torch.float32)

        image = Image.open(img_path).convert("RGB")  # Images have alpa channel

        if self.transform:
            image = self.transform(image)
        else:
            image = torchvision.transforms.ToTensor()(image)

        return image, class_label, attribute_label, img_path


def get_transforms_shapes():
    """
    Transforms for shapes dataset. No color changing or flips are done, since these can be meaningful
    concepts in the data. The images are only turned into tensors and normalized.
    Note that there is not really a need for data augmentation, since more data can be created.

    Returns:
        torchvision.tranforms.Compose: The transforms.
    """
    normalize = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),  # Turns image into [0, 1] float tensor
        normalize,
    ])
    return transform


def load_data_shapes(mode="all", path="data/shapes/shapes_testing/", subset_dir="", full_test_set=None,
                     batch_size=4, shuffle=True, drop_last=False, num_workers=0, pin_memory=False,
                     persistent_workers=False):
    """
    Makes dataloaders for the Shapes dataset.

    Will either just load "train", "val" or "test" loader (depending on argument `mode`),
    or a list of all of them if `mode == "all"`.

    Args:
        mode (str, optional): The datasetmode to loader. If "all", will return a tuple of (train, val, test).
            if "train-val", will return a tuple of (train, val). Defaults to "all".
        path (str, optional): Path to dataset-folder.
        subset_dir (str, optional): Optional name of subset-directory, which may be created
            with `make_shapes_dataset.make_subset_shapes()`.
        full_test_set (str): Set to name of full-test-set (made with `make_shapes_test_set`) to load full test-set,
            instead of `mode`. If `None`, will ignore this and load normal datasets.
        batch_size (int, optional): Batch size to use. Defaults to 4.
        shuffle (bool, optional): Determines wether the sampler will shuffle or not.
            It is recommended to use `True` for training and `False` for validating.
        drop_last (bool, optional): Determines wether the last iteration of an epoch
            is dropped when the amount of elements is less than `batch_size`.
        num_workers (int): The amount of subprocesses used to load the data from disk to RAM.
            0, default, means that it will run as main process.
        pin_memory (bool): Whether or not to pin RAM memory (make it non-pagable).
            This can increase loading speed from RAM to VRAM (when using `to("cuda:0")`,
            but also increases the amount of RAM necessary to run the job. Should only
            be used with GPU training.
        persistent_workers (bool): If `True`, will not shut down workers between epochs.

    Returns:
        Dataloader: The dataloader, or the list of the two or three dataloaders.
    """
    if full_test_set is not None:  # Be sure to only load one dataset
        mode = "test"
    if mode.lower() == "all":
        modes = ["train", "val", "test"]
    elif mode.lower() in ["train-val", "train val", "train-validation" "train validation"]:
        modes = ["train", "val"]
    else:
        modes = [mode]
    dataloaders = []
    for mode in modes:
        if full_test_set is not None:
            data_list = pickle.load(open(path + "tables/" + subset_dir + full_test_set, "rb"))
        else:
            data_list = pickle.load(open(path + "tables/" + subset_dir + mode + "_data.pkl", "rb"))
        transform = get_transforms_shapes()
        dataset = ShapesDataset(path, data_list, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                                num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
        dataloaders.append(dataloader)

    if len(dataloaders) == 1:  # Just return the datalaoder, not list
        return dataloaders[0]
    return dataloaders  # List of (train, val, test) dataloaders


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
        sub_list = random.sample(class_dict[i], n_images_class)
        new_list.extend(sub_list)

    tables_dir = path + "sub" + str(n_images_class) + "/"
    os.makedirs(tables_dir, exist_ok=True)
    with open(tables_dir + "data_list.pkl", "wb") as outfile:
        pickle.dump(new_list, outfile)
    if split_data:
        split_dataset(new_list, tables_dir, include_test=include_test)


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


def change_dataset_name(old_path, new_path):
    """
    Changes a name of a dataset folder, and correspondingly changes all of the image-paths in the
    data-lists and splits corresponding to it.

    The data-lists and splits are stored in the folder `dataset_path/tables/`, and if there are subset of them,
    created with `make_subset_shapes()`, they are stored in `dataset_path/tables/sub100/`.
    Replaces all of the data-lists files img_paths.

    Args:
        old_path (str): The path to the dataset to be renamed. Must contain both the path and the folder name.
        new_path (str): The name of the new path and folder. Must contain both the path and the folder name.
    """
    os.rename(old_path, new_path)  # Rename the actual folder

    tables_paths = [new_path + "tables/"]  # Find all the directories containing data-lists and splits
    for name in os.listdir(new_path + "tables/"):  # Check for folders inside `tables/`
        new_full_path = new_path + "tables/" + name
        if os.path.isdir(new_full_path):
            tables_paths.append(new_full_path)

    for table_path in tables_paths:  # Loop over the possible table-folders
        for filename in os.listdir(table_path):  # Loop over the files inside the folder
            if not filename.endswith(".pkl"):
                continue
            file_path = table_path + "/" + filename
            data_list = pickle.load(open(file_path, "rb"))
            for i in range(len(data_list)):  # Loop over the instances in the data-list.
                data_list[i]["img_path"] = data_list[i]["img_path"].replace(old_path, new_path)
            with open(file_path, "wb") as outfile:  # Overwrite
                pickle.dump(data_list, outfile)
