import pickle
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image


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

    def __init__(self, data_path=None, data_list=None, transform=None, mapping=None, attr_mask=None):
        """
        To access the CBM pickle data, please either proved `data_path` as a path to where
        it lies, or `data_list` where the pickle files are already read.

        Args:
            data_list (list of dict): List containing dictionaries with `img_path`
                `class_label`, and `attribute_label` keys. Should be read from
                `CUB_processed/class_attr_data_10/train.pkl`.
                Optionally, one can use `data_path` instead of this.
            data_path (str): Path to where to pkl files with image-info are stored.
                This is used if `data_list` is None.
                Default place is "data/CUB_processed/class_attr_data_10/train.pkl",
                which one can reach by setting `data_path` to "default_train",
                "default_val", or "default_test".
            transform (callable, optional): Optional transform to be applied on the image.
                One can use `get_transforms_cub()` to get this.
            mapping (dict): In case the data_list use is a subset of all classes,
                send this argument to convert the labels from bird indecies to indices in
                0, ..., n_classes. See utils.make_subset_cub() for info of the subset,
                this is useful for exploring since it runs much faster.
            n_attr (np.array of int): If not None, will load a subset of the attributes.
                The elements of the list are the indices of the features to be loaded.
                Which features to use can be determined by `feature_selection.py`.
                Note: Has to be array, list is not possible.
        """
        if data_list is not None:
            self.data_list = data_list
        elif data_path is not None:
            if data_path.lower() == "default_train":
                data_path = "data/CUB_processed/class_attr_data_10/train.pkl"
            elif data_path.lower() == "default_val":
                data_path = "data/CUB_processed/class_attr_data_10/val.pkl"
            elif data_path.lower() == "default_test":
                data_path = "data/CUB_processed/class_attr_data_10/test.pkl"
            try:
                data_list = pickle.load(open(data_path, "rb"))
            except Exception as e:
                message = f"Could not load file in path {data_path}. "
                message += "To resolve this issue, please download the CUB dataset "
                message += "and run `python initialize.py make_paths --base_path data/` "
                message += "to set the correct paths. See README.md for more information."
                raise FileNotFoundError(message)
        else:
            message += "Either `data_list` or `data_path` must be sent to constructor."
            raise ValueError(message)

        self.data_list = data_list
        self.transform = transform
        self.mapping = mapping
        self.attr_mask = attr_mask

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path = self.data_list[idx]["img_path"]
        class_label = self.data_list[idx]["class_label"]
        attribute_label = torch.tensor(self.data_list[idx]["attribute_label"]).to(torch.float32)

        if self.attr_mask is not None:  # Load only a subset of attribues
            attribute_label = attribute_label[self.attr_mask]

        image = Image.open(img_path).convert("RGB")  # Some images are grayscale

        if self.transform:
            image = self.transform(image)
        else:
            image = torchvision.transforms.ToTensor()(image)

        if self.mapping is not None:  # Map bird index to label
            class_label = self.mapping[class_label]

        return image, class_label, attribute_label, img_path


def get_transforms_cub(mode, resol=224, normalization="[-1, 1]",
                       brightness=32 / 255, saturation=[0.5, 1.5]):
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
        normalization (str, optional): The normalization method. Defaults to "[-1, 1]".
        brigthness (float): The brigthness to randomly increase or decreause the
            training image with, in jittering. Use 0 for no jitter.
        saturation (list): The interval of possible saturations to randomly jitter
            image to. Use [1, 1] for no saturation.

    Returns:
        callable: The transforms callable.
    """
    assert mode.lower() in ["train", "val", "test"]
    if normalization in ["[-1, 1]", "-1, 1", "[-1,1]", "-1,1"]:
        normalize = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
    elif normalization.lower() == "imagenet":
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])

    if mode.lower() == "train":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(brightness=brightness, saturation=saturation),
            torchvision.transforms.RandomResizedCrop(resol),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),  # Turns image into [0, 1] float tensor
            normalize,
        ])
    elif mode.lower() in ["val", "test"]:
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


def load_data_cub(mode="all", subset=None, n_attr=112, path="data/CUB_processed/class_attr_data_10/",
                  resol=224, normalization="[-1, 1]", brightness=32 / 255,
                  saturation=[0.5, 1.5], sampler=None, batch_size=4, shuffle=True, drop_last=False,
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
            tuple of (train, val, test). Defaults to "all".
        subset (int, optional): If not None, will only use a subset of the classes from the
            CUB dataset. Note that `utils.make_subset_cub()` should be called with
            the `n_classes` corresponding to `subset`, in order to first create the files.
        n_attr (int): The amount of attributes (out of 112) to use. If not 112, the
            features determined by the feature selection in `feature_selection.py` will
            be used. This file should already be called, and saved as
            "CUB_processed/CUB_feature_selection.pkl".
        path (str, optional): Path to the data-folder. Defaults to "data/CUB_processed/class_attr_data_10/".
        resol (int, optional): The expected input height and width by the model,
            which the transfroms transforms to. Defaults to 224.
        normalization (str, optional): The normalization method. Defaults to "[-1, 1]".
        brigthness (float): The brigthness to randomly increase or decreause the
            training image with, in jittering. Use 0 for no jitter.
        saturation (list): The interval of possible saturations to randomly jitter
            image to. Use [1, 1] for no saturation.
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
    else:
        modes = [mode]

    dataloaders = []
    mapping = None
    if subset == 200:  # We are using all the classes, so no actual subset will be used
        subset = None
    if subset is not None:  # Map class indices to labels
        mapping_path = path + str(subset) + "_class_mapping.pkl"
        # Make sure `python initialize.py --make_dataset --n_classes subset`
        # is called to make this dict:
        mapping = pickle.load(open(mapping_path, "rb"))

    attr_mask = None
    if n_attr < 112:
        feature_path = "data/CUB_processed/CUB_feature_selection.pkl"
        # Make sure `feature_selection.py` is called to make this dict:
        info_dict = pickle.load(open(feature_path, "rb"))
        attr_mask = np.array(info_dict[n_attr]["features"])

    for mode in modes:  # Loop over the train, val, test (or just one of them)
        transform = get_transforms_cub(mode, resol=resol, normalization=normalization,
                                       brightness=brightness, saturation=saturation)
        if subset is not None:
            full_path = path + str(subset) + "_" + mode + ".pkl"
        else:
            full_path = path + mode + ".pkl"
        dataset = CUBDataset(data_path=full_path, transform=transform,
                             mapping=mapping, attr_mask=attr_mask)
        dataloader = make_dataloader(dataset=dataset, sampler=sampler, batch_size=batch_size, shuffle=shuffle,
                                     drop_last=drop_last, num_workers=num_workers, pin_memory=pin_memory,
                                     persistent_workers=persistent_workers)
        dataloaders.append(dataloader)

    if len(dataloaders) == 1:  # Just return the datalaoder, not list
        return dataloaders[0]
    return dataloaders  # List of (train, val, test) dataloaders


def make_subset_cub(n_classes=10, random_choice=True):
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
