import torch
import random
import pickle
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from src.common.utils import seed_everything
from src.common.path_utils import (load_data_list_cub, write_data_list_cub, get_feature_selection_cub,
                                   write_test_data_list_cub)
from src.constants import N_CLASSES_CUB


class CUBDataset(Dataset):
    """
    Dataset object for CUB dataset. The Dataloader returns
    (PIL-image, class-label-int, list-of-attribute-labels, img_path).

    This uses the CUB dataset and the preprocessed
    version from the CBM 2020 paper. Note that the CBM paper have the authors full
    path as image path arguments. This can be updated by calling `make_correct_paths()`
    from `initialize.py`, or by `python initialize.py --base_path base_path`.

    The original CUB dataset should be stored in the given `base_path`, where is `data/`
    is used for the author. This folder should be in the same folder as this file if one
    uses relative paths.
    Additionally, the preprocessed pickle files from the CBM paper should also be in the
    same folder. This pickle files contains a list of a object-level dict, with paths,
    class labels, attribute labels and attribute certainty.
    """

    def __init__(self, data_list, transform=None, attr_mask=None):
        """
        To access the CBM pickle data.
        Please read `data_list` with `load_data_list_cub()` from `src.common.path_utils.py`.

        Args:
            data_list (list of dict): List containing dictionaries with `img_path`
                `class_label`, and `attribute_label` keys. Should be read with `load_data_list_cub()` from
                `src.common.path_utils.py`.
            transform (callable, optional): Optional transform to be applied on the image.
                One can use `get_transforms_cub()` to get this.
            n_attr (np.array of int): If not None, will load a subset of the attributes.
                The elements of the list are the indices of the features to be loaded.
                Which features to use can be determined by `feature_selection.py`.
                Note: Has to be array, list is not possible.
        """
        self.data_list = data_list
        self.transform = transform
        self.attr_mask = attr_mask

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path = self.data_list[idx]["img_path"]
        class_label = self.data_list[idx]["class_label"]
        attribute_label = torch.tensor(self.data_list[idx]["attribute_label"]).to(torch.float32)

        if self.attr_mask is not None:  # Load only a subset of attribues
            attribute_label = attribute_label[self.attr_mask]

        with Image.open(img_path) as img:  # Open and close properly
            image = img.convert("RGB")  # Some images are grayscale

        if self.transform:
            image = self.transform(image)
        else:
            image = torchvision.transforms.ToTensor()(image)

        return image, class_label, attribute_label, img_path


def get_transforms_cub(mode, resol=224, normalization="imagenet"):
    """
    Returns the callable for transforming the images for the CUB dataset.
    The `normalization` argument either scales to the interval [-1, 1], or
    uses the imagenet standard normalization values. `resol` is the height
    and width expected by the model, which is 224 for MobileNet and ResNet.

    Naturally, the train transforms does random cropping and flipping
    (and maybe jittering), while the validation and test transforms makes a
    deterministic center crop. The image is resized first to make sure the center
    is not very zoomed in.

    The CBM paper 2020 uses 0.5 for std instead of 2, scaling to [-0.25, 0.25].
    It also does not resize the validation transform before centercropping.

    Args:
        mode (str): Either `train` or `val`, to indicate which transform to use.
        resol (int, optional): The expected input height and width by the model,
            which the transfroms transforms to. Defaults to 224.
        normalization (str, optional): The normalization method.

    Returns:
        callable: The transforms callable.
    """
    assert mode.lower() in ["train", "val", "test", "test_small"]
    if normalization in ["[-1, 1]", "-1, 1", "[-1,1]", "-1,1"]:
        normalize = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
    elif normalization.lower() == "imagenet":
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])

    if mode.lower() == "train":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(resol),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),  # Turns image into [0, 1] float tensor
            normalize,
        ])
    elif mode.lower() in ["val", "test", "test_small"]:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(resol),
            torchvision.transforms.ToTensor(),
            normalize,
        ])
    return transform


def make_dataloader(dataset, sampler=None, batch_size=4, shuffle=True, drop_last=False,
                    num_workers=0, pin_memory=False, persistent_workers=False):
    """
    Converts a Dataset to a Dataloader.
    If one uses a costum sampler, send this as `sampler`.

    Args:
        dataset (Dataset): The Dataset to use
        sampler (callable, optional): Costum sampler. If None, will use standard
            sampler. Defaults to None.
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
        Dataloader: The Dataloader
    """
    if sampler is not None:
        batch_sampler = torch.utils.data.BatchSampler(sampler(dataset), batch_size=batch_size,
                                                      shuffle=shuffle, drop_last=drop_last)
        dataloader = DataLoader(dataset, sampler=batch_sampler, num_workers=num_workers, pin_memory=pin_memory,
                                persistent_workers=persistent_workers)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                                num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    return dataloader


def load_data_cub(mode="train-val", n_subset=None, n_attr=112,
                  resol=224, normalization="imagenet", sampler=None, batch_size=4, shuffle=True, drop_last=False,
                  num_workers=0, pin_memory=False, persistent_workers=False):
    """
    Main function to call for loading the CUB dataset.

    Function that calls CUBDataset, get_transforms_cub() and make_dataloader().
    This is meant as a convenient way of just calling one function to get the
    final dataloaders. Takes in all of the arguments that the other functions does.
    Note that `mode` can be in ["train", "val", "test", "all"]. If it is one of the
    three first, this function returns only the dataloader corresponding to that set,
    and if it is "all" then it returns a list of train, val, test loaders.

    Args:
        mode (str, optional): The datasetmode to loader. If "all", will return a
            tuple of (train, val, test). Defaults to "train-val", which means train and validation loader.
        n_subset (int, optional): If not None, will only use a subset of the classes from the CUB dataset.
        n_attr (int): The amount of attributes (out of 112) to use. If not 112, the
            features determined by the feature selection in `feature_selection.py` will
            be used. This file should already be called.
        resol (int, optional): The expected input height and width by the model,
            which the transfroms transforms to. Defaults to 224.
        normalization (str, optional): The normalization method. Defaults to "[-1, 1]".
        sampler (callable, optional): Costum sampler. If None, will use standard
            sampler. Defaults to None.
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
        Dataloader: The dataloader, or the list of the three dataloaders.
    """
    if mode.lower() == "all":
        modes = ["train", "val", "test"]
    elif mode.lower() in ["train-val", "train val", "train-validation" "train validation"]:
        modes = ["train", "val"]
    else:
        modes = [mode]

    attr_mask = None
    if n_attr < 112:  # Use only a subset of attributes, based on feature-selection
        info_dict = get_feature_selection_cub()
        attr_mask = np.array(info_dict[n_attr]["features"])

    dataloaders = []
    for mode in modes:  # Loop over the train, val, test (or just one of them)
        transform = get_transforms_cub(mode, resol=resol, normalization=normalization)
        data_list = load_data_list_cub(mode=mode, n_subset=n_subset)
        dataset = CUBDataset(data_list=data_list,  transform=transform, attr_mask=attr_mask)
        dataloader = make_dataloader(dataset=dataset, sampler=sampler, batch_size=batch_size, shuffle=shuffle,
                                     drop_last=drop_last, num_workers=num_workers, pin_memory=pin_memory,
                                     persistent_workers=persistent_workers)
        dataloaders.append(dataloader)

    if len(dataloaders) == 1:  # Just return the datalaoder, not list
        return dataloaders[0]
    return dataloaders  # List of (train, val, test) dataloaders


def make_subset_cub(n_images_class, seed=57):
    """
    Makes subsets of the data-lists for CUB.
    This means we pick `n_images_class` images for every of the 200 classes to use for training and validation.
    80% of the images will be in train, and 20% for validation. This means that if `n_images_class` = 10, then
    8 images per class will be used for training, and 2 for validaiton.
    Nothing will be done with the original test-data-list. This will be a global test-dataset for all of the subsets.

    This reads the data-lists, which are the `CUB_processed` made by the `Concept Bottleneck Models` paper,
    and saves a sublist in a folder called sub{n_image_class}, in the same folder as the original datalists.

    Args:
        n_images_class (int): The amount of images to use from every class, for training and validation added up.
        seed (int, optional): Seed for drawing the data. Defaults to 57.
    """
    seed_everything(seed)
    train_list = load_data_list_cub(mode="train")
    val_list = load_data_list_cub(mode="val")

    class_mapping_train = {}  # Each key (class-number) points to a list of instances in that class
    class_mapping_val = {}
    for i in range(N_CLASSES_CUB):
        class_mapping_train[i] = []
        class_mapping_val[i] = []

    for data_dict in train_list:  # Put the dicts into the class-mappings
        class_mapping_train[data_dict["class_label"]].append(data_dict)
    for data_dict in val_list:
        class_mapping_val[data_dict["class_label"]].append(data_dict)
    
    new_train_list = []
    new_val_list = []
    for i in range(N_CLASSES_CUB):  # Take the first `n_image_class * split` from each class.
        sub_list_train = class_mapping_train[i][:max(int(n_images_class * 0.8), 1)]
        sub_list_val = class_mapping_val[i][:max(int(n_images_class * 0.2), 1)]
        new_train_list.extend(sub_list_train)
        new_val_list.extend(sub_list_val)

    write_data_list_cub(n_subset=n_images_class, train_data=new_train_list, val_data=new_val_list)


def make_small_test_set(n_size=200):
    """
    Make a subset of the test-set for fast code-testing. The original test-set has 5794 images, and takes some time.

    Args:
        n_size (int, optional): The size of the small test-set. Defaults to 200.
    """
    full_test = load_data_list_cub("test")
    random.shuffle(full_test)
    small_test = full_test[:n_size]
    write_test_data_list_cub(small_test, "test_small.pkl")

    
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
