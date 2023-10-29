"""
File for visualizing pictures form the datasets
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import rc

from src.constants import MODEL_STRINGS, COLORS
from src.common.path_utils import load_history_shapes, save_test_plot_shapes, save_training_plot_shapes


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


def plot_training_histories(n_classes, n_attr, signal_strength, n_bootstrap, n_subset, histories, names,
                            colors=None, attributes=False, title=None):
    """
    Plots training and validation loss and accuracy, in four subplots.
    Hisotries can be a list of many training histories, which will results in the history being
    plotted together, so one can easily compare.

    Args:
        n_classes (int): The amount of classes in the dataset.
        n_attr (int): The amonut of attributes (concepts) in the dataset.
        signal_strength (int, optional): Signal-strength used to make dataset.
        n_bootstrap (int): The amount of bootstrap iterations that were used.
        n_subset (int): The amount of instances in each class used in the subset.
        histories (list of dict): List of training histories, as returned or saved by the train functions
            in `train.py`.
        names (list of str): List of the names of the models. Must be of same length as histories.
        colors (list of str, optional): List of colors to use for the different models. Defaults to None.
        attributes (bool, optional): If True, will plot the attribute stats. Must be a concept-model.
            Defaults to False.
        title (str, optional): Title for the plot. Defaults to None.
        save_dir (str, optional): Path to where the plot will be saved. Defaults to "plots/".
        save_name (str, optional): Name of saved figure. Defaults to "test.png".
    """
    fig = plt.figure()
    fig.suptitle(title)  # Main title
    n_hist = len(histories)
    if colors is None:  # Matplotlib will chose colors for us
        colors = [None] * n_hist

    if attributes:  # Set name for either class or attributes
        mid_name = "attr"
    else:
        mid_name = "class"

    # Initialize the four subplots
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title("Train Loss Over Epochs")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title("Validation Loss Over Epochs")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title("Train Accuracy Over Epochs")
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("Accuracy")
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title("Validation Accuracy Over Epochs")
    ax4.set_xlabel("Epochs")
    ax4.set_ylabel("Accuracy")
    axes = [ax1, ax2, ax3, ax4]

    for i in range(n_hist):  # Loop over histories and plot
        history = histories[i]
        name = names[i]
        color = colors[i]
        axes[0].plot(history["train_" + mid_name + "_loss"], label=f"{name}", c=color)
        axes[1].plot(history["val_" + mid_name + "_loss"], label=f"{name}", c=color)
        axes[2].plot(history["train_" + mid_name + "_accuracy"], label=f"{name}", c=color)
        axes[3].plot(history["val_" + mid_name + "_accuracy"], label=f"{name}", c=color)

    for ax in axes:
        ax.legend()

    plt.tight_layout()  # To avoid overlap
    save_training_plot_shapes(n_classes, n_attr, signal_strength, n_bootstrap, n_subset, attr=attributes)


def plot_test_accuracies(n_classes, n_attr, signal_strength, subsets, n_bootstrap=1,
                         model_strings=None, colors=None):
    """
    Plots the test-accuracies as pdf format and with minimal border, so that plot is suitable for using in LaTeX.
    The histories file must already exist.

    Args:
        n_classes (int): The amount of classes to plot for.
        n_attr (int): The amount of attributes.
        signal_strength (int): The signal strength used for the dataset.
        subsets (list of int): The list of subsets to plot for.
        n_bootstrap (int, optional): The amount of bootstrap iterations used, in order to access correct history.
        history_folder (str, optional): Folder for where histories are saved. Defaults to "history/".
        model_strings (list of str, optional): List of string name of models to use. If None,
            will use the constants value of names. Defaults to None.
        colors (list of str, optional): List of colors to use for the different models. Defaults to None.
    """
    if model_strings is None:
        model_strings = MODEL_STRINGS
    test_accuracies_lists = [[] for _ in range(len(model_strings))]
    for n_subset in subsets:
        histories = load_history_shapes(n_classes, n_attr, signal_strength, n_bootstrap, n_subset)
        for i in range(len(model_strings)):
            test_accuracies_lists[i].append(histories[i]["test_accuracy"])

    x_values = subsets
    n_models = len(test_accuracies_lists)
    fig = plt.figure()
    if colors is None:
        colors = COLORS

    for i in range(n_models):
        error_list = test_accuracies_lists[i]
        name = model_strings[i]
        color = colors[i]
        plt.plot(x_values, error_list, label=name, c=color)  # Lines
        plt.scatter(x_values, error_list, marker="s", facecolor=color, edgecolor=color)  # Squares at each point

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)
    plt.legend()
    plt.xlabel("Size of Subset for each class")
    save_test_plot_shapes(n_classes, n_attr, signal_strength, n_bootstrap)