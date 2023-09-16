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


def load_data_shapes(mode="all", path="data/shapes/shapes_testing/", subset_dir="",
                     batch_size=4, shuffle=True, drop_last=False,
                     num_workers=0, pin_memory=False, persistent_workers=False):
    """
    Makes dataloaders for the Shapes dataset.

    Will either just load "train", "val" or "test" loader (depending on argument `mode`),
    or a list of all of them if `mode == "all"`.

    Args:
        mode (str, optional): The datasetmode to loader. If "all", will return a
            tuple of (train, val, test). Defaults to "all".
        path (str, optional): Path to dataset-folder.
        subset_dir (str, optional): Optional name of subset-directory, which may be created
            with `make_shapes_dataset.make_subset_shapes()`.
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
    for mode in modes:
        data_list = pickle.load(open(path + "tables/" + subset_dir + mode + "_data.pkl", "rb"))
        transform = get_transforms_shapes()
        dataset = ShapesDataset(path, data_list, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                                num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
        dataloaders.append(dataloader)

    if len(dataloaders) == 1:  # Just return the datalaoder, not list
        return dataloaders[0]
    return dataloaders  # List of (train, val, test) dataloaders


def make_subset_shapes(path, n_images_class, n_classes, split_data=True, seed=57):
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
        split_data (bool, optional): If True, splits into "train", "validation" and "test". Defaults to True.
        seed (int, optional): The seed for the rng. Defaults to 57.
    """
    random.seed(seed)
    data_list = pickle.load(open(path + "data_list.pkl", "rb"))

    class_dict = {}  # Sort dicts by label
    for i in range(n_classes):
        class_dict[i] = []
    for i in range(len(data_list)):
        class_dict[data_list[i]["class_label"]].append(data_list[i])

    new_list = []
    for i in range(n_classes):
        sub_list = random.sample(class_dict[i], n_images_class)
        new_list.extend(sub_list)

    tables_dir = path + "sub" + str(n_images_class) + "/"
    os.makedirs(tables_dir, exist_ok=True)
    with open(tables_dir + "data_list.pkl", "wb") as outfile:
        pickle.dump(data_list, outfile)
    if split_data:
        split_dataset(new_list, tables_dir)


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
            file_path = table_path + filename
            data_list = pickle.load(open(file_path, "rb"))
            for i in range(len(data_list)):  # Loop over the instances in the data-list.
                data_list[i]["img_path"] = data_list[i]["img_path"].replace(old_path, new_path)
            with open(file_path, "wb") as outfile:  # Overwrite
                pickle.dump(data_list, outfile)
