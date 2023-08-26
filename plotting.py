"""
File for visualizing pictures form the datasets
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import rc


def plot_image_single(image_path, pred=None, label=None, transformed=False, show=True, title=None):
    """
    Display a single picture.

    Arguments:
        image_path (str): Path to image.
        pred (int): Prediction corresponding to the image.
            If provided, will include the prediction in the title.
        label (int): label corresponding to the images. If given, will
            include label as the title to the image.
        tranformed (bool): If False, will reshape from p to h x w, according to
            the argument "dataset".
        dataset (str): The dataset to be plotted. This determines the reshape size, if
            "transformed" is False. Must be in ["MNIST", "CIFAR"].
        show (bool): If True, will call plt.show().
        title (str): Title for the plot.
    """
    image = np.array(Image.open(image_path))
    plt.imshow(image, cmap="gray")
    plt.xticks(np.array([]))  # Remove ticks
    plt.yticks(np.array([]))
    if title is None:
        title = ""  # Construct title
    else:
        title += ". "
    if pred is not None:
        title += f"Prediction: {pred}"
    if pred is not None and label is not None:
        title += ", "
    if label is not None:
        title += f"True label: {label}"
    plt.title(title)
    if show:
        plt.show()


def plot_images_random(image_paths, preds=None, labels=None, show=True, n_random=10, n_cols=5, title=None):
    """
    Plot random images of the data given.

    Arguments:
        image_paths (list of str): List of string paths to images.
        preds (np.array): [n] array of predicted labels corresponding to the images.
            If provided, will include predictions as titles to the images.
        labels (np.array): [n] array of labels corresponding to the images.
            If provided, will include labels as titles to the images.
        show (bool): If True, will call plt.show().
        n_random (int): The amount of images that will be plotted.
        n_cols (int): The amount of images in each columns
        title (str): Title for the plot.
    """
    images = []  # Read images
    for path in image_paths:
        img = np.array(Image.open(path))
        images.append(img)

    fig = plt.figure()
    n = len(images)
    indices = np.random.choice(n, n_random, replace=False)  # Chose indices for random images to plot
    n_rows = int(np.ceil(n_random / n_cols))
    for i in range(n_random):
        if preds is not None and labels is not None:  # Make title blue in wrong predictions
            if preds[indices[i]] != labels[indices[i]]:
                rc("text", color="red")
            else:
                rc("text", color="blue")

        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        image = images[indices[i]]
        ax.imshow(image)  # Plot the image

        sub_title = ""  # Construct sub_title
        if preds is not None:
            sub_title += f"P: {preds[indices[i]]}"
        if preds is not None and labels is not None:
            sub_title += ", "
        if labels is not None:
            sub_title += f"Y: {labels[indices[i]]}"

        ax.title.set_text(sub_title)

        plt.xticks(np.array([]))  # Remove ticks
        plt.yticks(np.array([]))

    rc("text", color="black")  # Set text back to black
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    if show:
        plt.show()


def plot_images_mislabeled(image_paths, preds, labels, show=True, n_random=10, n_cols=5, title=None):
    """
    Plot random mislabeled images from the data provided.

    Arguments:
        image_paths (list of str): List of string paths to images.
        preds (np.array): [n] array of predicted labels corresponding to the images.
        labels (np.array): [n] array of labels corresponding to the images.
        show (bool): If True, will call plt.show().
        n_random (int): The amount of images that will be plotted.
        n_cols (int): The amount of images in each columns.
        title (str): Title for the plot.
    """
    paths = np.array(paths)  # For indexing
    indices = (preds != labels)
    wrong_paths = image_paths[indices]  # Index wrongly labeled images
    preds = preds[indices]
    labels = labels[indices]
    # Use random plotting function
    plot_images_random(wrong_paths, preds, labels, show, n_random, n_cols, title)


def plot_images_worst(image_paths, logits, labels,
                      show=True, n_images=10, n_cols=5, title=None):
    """
    Plot (probably mislabeled) images that corresponds to the worst predictions. This means
    the value for the true class and the predicted logit value is as different as possible.

    Arguments:
        image_paths (list of str): List of string paths to images.
        logits (np.array): [n] array of predicted logits values
            (either softmax or other activation function outputs).
        labels (np.array): [n] array of labels corresponding to the images.
        show (bool): If True, will call plt.show().
        n_images (int): The amount of images that will be plotted.
        n_cols (int): The amount of images in each columns
        title (str): Title for the plot.
    """
    paths = np.array(paths)  # For indexing
    predicted_logits = logits[np.arange(len(labels)), labels]
    indices = predicted_logits.argsort()
    wrong_paths = image_paths[indices[:n_images]]
    logits = logits[indices[:n_images]]
    labels = labels[indices[:n_images]]
    preds = logits.argmax(axis=1)
    plot_images_random(wrong_paths, preds, labels, show, n_images, n_cols, title)
