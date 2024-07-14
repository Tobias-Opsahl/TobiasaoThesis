import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import rc
from matplotlib.ticker import MaxNLocator

from src.constants import (MODEL_STRINGS_SHAPES, MODEL_STRINGS_CUB, MAP_NAMES_TO_DOCUMENT_NAMES,
                           MAP_MODEL_TO_COLOR_SHAPES, MAP_MODEL_TO_LINESTYLES_SHAPES)
from src.common.path_utils import (
    load_history_shapes, save_test_plot_shapes, save_training_plot_shapes, load_history_cub, save_test_plot_cub,
    save_training_plot_cub, save_single_image, save_adversarial_image_shapes, save_mpo_plot_shapes, save_mpo_plot_cub,
    get_attribute_mapping, get_attribute_names, get_class_names, save_adversarial_text_file)


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
    # Chose indices for random images to plot
    indices = np.random.choice(n, n_random, replace=False)
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


def plot_mpo_scores(histories, model_strings):
    """
    Plots the MPO score Misprediction Prediction Overlap metric from https://arxiv.org/pdf/2010.13233.pdf.

    Args:
        histories (dict): Dictionary of training histories, with mpos.
        model_strings (list of string): List of only the concept models that should be used for the MPO plot
    """
    _ = plt.figure()

    for model_string in model_strings:
        plot_model_string = MAP_NAMES_TO_DOCUMENT_NAMES[model_string]  # Name for plotting
        mpo_array = np.array(histories[model_string]["mpo"])
        n_bootstrap = mpo_array.shape[0]
        n_epochs = mpo_array.shape[1]
        color = MAP_MODEL_TO_COLOR_SHAPES[model_string]
        linestyle = MAP_MODEL_TO_LINESTYLES_SHAPES[model_string]

        mean_accuracies = np.mean(mpo_array, axis=0)
        std_accuracies = np.std(mpo_array, axis=0)  # For confidence intervals
        z_score = 1.96  # 95% confidence intervals
        confidence_interval = z_score * std_accuracies / np.sqrt(n_bootstrap)
        x_values = np.arange(1, n_epochs + 1)
        ax = plt.gca()  # Get current axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Remove non-integer x-tick values

        plt.plot(x_values, mean_accuracies, label=plot_model_string, c=color, linestyle=linestyle)  # Lines
        plt.fill_between(x_values, (mean_accuracies - confidence_interval),
                         (mean_accuracies + confidence_interval), color=color, alpha=0.10)

    # plt.ylim(bottom=-0.000001)
    plt.subplots_adjust(left=0.065, right=0.95, top=0.95, bottom=0.1)
    plt.legend()


def plot_training_histories(histories, model_strings, attributes=False, title=None):
    """
    Plots training and validation loss and accuracy, in four subplots.
    Hisotries can be a list of many training histories, which will results in the history being
    plotted together, so one can easily compare.

    Args:
        histories (list of dict): List of training histories, as returned or saved by the train functions
            in `train.py`.
        model_strings (list of str): List of the names of the models. Must be of same length as histories.
        attributes (bool, optional): If True, will plot the attribute stats. Must be a concept-model.
            Defaults to False.
        title (str, optional): Title for the plot. Defaults to None.
    """
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle(title)  # Main title

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

    for model_string in model_strings:
        plot_model_string = MAP_NAMES_TO_DOCUMENT_NAMES[model_string]  # Name for plotting
        history = histories[model_string]
        color = MAP_MODEL_TO_COLOR_SHAPES[model_string]
        linestyle = MAP_MODEL_TO_LINESTYLES_SHAPES[model_string]

        metrics = ["train_" + mid_name + "_loss", "val_" + mid_name + "_loss",
                   "train_" + mid_name + "_accuracy", "val_" + mid_name + "_accuracy"]
        for i, metric in enumerate(metrics):
            metric_history = np.array(history[metric])
            x_values = np.arange(metric_history.shape[1])
            n_bootstrap = metric_history.shape[0]
            mean = np.mean(metric_history, axis=0)
            std = np.std(metric_history, axis=0)
            # z_score = 0.674  # 50% confidence interval
            z_score = 1.96  # 95% confidence intervals
            confidence_interval = z_score * std / np.sqrt(n_bootstrap)
            axes[i].plot(x_values, mean, label=plot_model_string, c=color, linestyle=linestyle)
            axes[i].fill_between(x_values, (mean - confidence_interval), (mean + confidence_interval),
                                 color=color, alpha=0.05)

    ax1.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def plot_test_accuracies(histories_dict, subsets, model_strings, n_bootstrap):
    """
    Plots the test-accuracies as pdf format and with minimal border, so that plot is suitable for using in LaTeX.
    The histories file must already exist.

    Args:
        histories_dict (dict): Dictinaries of training histories, read form pickle files.
        subsets (list of int): The list of subsets to plot for.
        model_strings (list of str): List of string name of models to use. If None,
            will use the constants value of names.
        n_bootstrap (int, optional): The amount of bootstrap iterations used, in order to access correct history.
    """
    # Save the test-accuracies on the currect format.
    # For every model, we will have a 2d arrray, where rows are bootstrap runs, and columns are different subsets
    test_accuracies = {}
    for model_string in model_strings:
        test_accuracies[model_string] = np.zeros((n_bootstrap, len(subsets)))
        for n_boot in range(n_bootstrap):
            for i, n_subset in enumerate(subsets):
                test_accuracy = histories_dict[n_subset][model_string]["test_accuracy"][n_boot][0]
                test_accuracies[model_string][n_boot][i] = test_accuracy

    x_values = subsets
    _ = plt.figure()

    for model_string in test_accuracies:
        plot_model_string = MAP_NAMES_TO_DOCUMENT_NAMES[model_string]  # Name for plotting
        test_accuracies_array = test_accuracies[model_string]
        color = MAP_MODEL_TO_COLOR_SHAPES[model_string]
        linestyle = MAP_MODEL_TO_LINESTYLES_SHAPES[model_string]

        mean_accuracies = np.mean(test_accuracies_array, axis=0)
        std_accuracies = np.std(test_accuracies_array, axis=0)  # For confidence intervals
        z_score = 1.96  # 95% confidence intervals
        confidence_interval = z_score * std_accuracies / np.sqrt(n_bootstrap)

        plt.plot(x_values, mean_accuracies, label=plot_model_string, c=color, linestyle=linestyle)  # Lines
        plt.scatter(x_values, mean_accuracies, marker="s", facecolor=color, edgecolor=color)  # Squares at each point
        plt.fill_between(x_values, (mean_accuracies - confidence_interval),
                         (mean_accuracies + confidence_interval), color=color, alpha=0.15)

    # Change left to 0.075 if there are 4 digits in the y-axis.
    # plt.ylim(top=100.1)
    plt.subplots_adjust(left=0.065, right=0.95, top=0.95, bottom=0.1)
    # plt.legend(loc="upper left")
    plt.legend()


def plot_mpo_scores_shapes(n_classes, n_attr, signal_strength, n_bootstrap, n_subset, histories=None,
                           model_strings=None, hard_bottleneck=False):
    """
    Plots the MPO metrics for m values from 1 to k, the number of concepts.

    Args:
        n_classes (int): The amount of classes in the dataset.
        n_attr (int): The amonut of attributes (concepts) in the dataset.
        signal_strength (int, optional): Signal-strength used to make dataset.
        n_bootstrap (int): The amount of bootstrap iterations that were used.
        n_subset (int): The amount of instances in each class used in the subset.
        histories (list of dict): List of training histories, as returned or saved by the train functions
            in `train.py`.
        model_strings (list of str): List of the names of the models. Must be of same length as histories.
        hard_bottleneck (bool): If True, will load histories with hard-bottleneck, and save with "_hard" in name.
        attributes (bool, optional): If True, will plot the attribute stats. Must be a concept-model.
            Defaults to False.
        title (str, optional): Title for the plot. Defaults to None.
    """
    if model_strings is None:
        model_strings = MODEL_STRINGS_SHAPES

    if histories is None:
        histories = load_history_shapes(n_classes, n_attr, signal_strength, n_bootstrap, n_subset,
                                        hard_bottleneck=hard_bottleneck)

    plot_mpo_scores(histories=histories, model_strings=model_strings)
    save_mpo_plot_shapes(n_classes, n_attr, signal_strength, n_bootstrap, n_subset, hard_bottleneck=hard_bottleneck)


def plot_training_histories_shapes(n_classes, n_attr, signal_strength, n_bootstrap, n_subset, histories=None,
                                   model_strings=None, hard_bottleneck=False, attributes=False, title=None):
    """
    Plots the training histories for the Shapes dataset. Saves it in the folder-structure for shapes.
    Histories must already exist.

    Args:
        n_classes (int): The amount of classes in the dataset.
        n_attr (int): The amonut of attributes (concepts) in the dataset.
        signal_strength (int, optional): Signal-strength used to make dataset.
        n_bootstrap (int): The amount of bootstrap iterations that were used.
        n_subset (int): The amount of instances in each class used in the subset.
        histories (list of dict): List of training histories, as returned or saved by the train functions
            in `train.py`.
        model_strings (list of str): List of the names of the models. Must be of same length as histories.
        hard_bottleneck (bool): If True, will load histories with hard-bottleneck, and save with "_hard" in name.
        attributes (bool, optional): If True, will plot the attribute stats. Must be a concept-model.
            Defaults to False.
        title (str, optional): Title for the plot. Defaults to None.
    """
    if model_strings is None:
        model_strings = MODEL_STRINGS_SHAPES

    if histories is None:
        histories = load_history_shapes(n_classes, n_attr, signal_strength, n_bootstrap, n_subset,
                                        hard_bottleneck=hard_bottleneck)

    plot_training_histories(histories=histories, model_strings=model_strings,
                            attributes=attributes, title=title)
    save_training_plot_shapes(n_classes, n_attr, signal_strength, n_bootstrap, n_subset, attr=attributes,
                              hard_bottleneck=hard_bottleneck)


def plot_test_accuracies_shapes(n_classes, n_attr, signal_strength, subsets, n_bootstrap=1, hard_bottleneck=False,
                                model_strings=None):
    """
    Plots test accuracies for different subsets for the Shapes dataset. Histories must already exist.
    Plots the test-accuracies as pdf format and with minimal border, so that plot is suitable for using in LaTeX.
    Saves the plot in the appropriate Shapes file-structure.

    Args:
        n_classes (int): The amount of classes to plot for.
        n_attr (int): The amount of attributes.
        signal_strength (int): The signal strength used for the dataset.
        subsets (list of int): The list of subsets to plot for.
        n_bootstrap (int, optional): The amount of bootstrap iterations used, in order to access correct history.
        hard_bottleneck (bool): If True, will load histories with hard-bottleneck, and save with "_hard" in name.
        model_strings (list of str, optional): List of string name of models to use. If None,
            will use the constants value of names. Defaults to None.
    """
    if model_strings is None:
        model_strings = MODEL_STRINGS_SHAPES

    histories_dict = {}
    for n_subset in subsets:  # Load all of the histories, one for each subset
        histories = load_history_shapes(n_classes, n_attr, signal_strength, n_bootstrap, n_subset,
                                        hard_bottleneck=hard_bottleneck)
        histories_dict[n_subset] = histories

    plot_test_accuracies(histories_dict, subsets, model_strings=model_strings, n_bootstrap=n_bootstrap)
    save_test_plot_shapes(n_classes, n_attr, signal_strength,
                          n_bootstrap, hard_bottleneck=hard_bottleneck)


def plot_mpo_scores_cub(n_bootstrap, n_subset, histories=None, model_strings=None, hard_bottleneck=False):
    """

    Args:
        n_bootstrap (int): The amount of bootstrap iterations that were used.
        n_subset (int): The amount of instances in each class used in the subset.
        histories (list of dict): List of training histories, as returned or saved by the train functions
            in `train.py`.
        model_strings (list of str): List of the names of the models. Must be of same length as histories.
        hard_bottleneck (bool): If True, will load histories with hard-bottleneck, and save with "_hard" in name.
        attributes (bool, optional): If True, will plot the attribute stats. Must be a concept-model.
            Defaults to False.
        title (str, optional): Title for the plot. Defaults to None.
    """
    if model_strings is None:
        model_strings = MODEL_STRINGS_CUB

    if histories is None:
        histories = load_history_cub(n_bootstrap, n_subset, hard_bottleneck=hard_bottleneck)

    plot_mpo_scores(histories=histories, model_strings=model_strings)
    save_mpo_plot_cub(n_bootstrap, n_subset, hard_bottleneck=hard_bottleneck)


def plot_training_histories_cub(n_bootstrap, n_subset, histories=None, model_strings=None, hard_bottleneck=False,
                                attributes=False, title=None):
    """
    Plots the training histories for the CUB dataset. Saves it in the folder-structure for shapes.
    Histories must already exist.

    Args:
        n_bootstrap (int): The amount of bootstrap iterations that were used.
        n_subset (int): The amount of instances in each class used in the subset.
        histories (list of dict): List of training histories, as returned or saved by the train functions
            in `train.py`.
        model_strings (list of str): List of the names of the models. Must be of same length as histories.
        hard_bottleneck (bool): If True, will load histories with hard-bottleneck, and save with "_hard" in name.
        attributes (bool, optional): If True, will plot the attribute stats. Must be a concept-model.
            Defaults to False.
        title (str, optional): Title for the plot. Defaults to None.
    """
    if model_strings is None:
        model_strings = MODEL_STRINGS_CUB

    if histories is None:
        histories = load_history_cub(n_bootstrap, n_subset, hard_bottleneck=hard_bottleneck)

    plot_training_histories(histories=histories, model_strings=model_strings, attributes=attributes, title=title)
    save_training_plot_cub(n_bootstrap, n_subset, hard_bottleneck=hard_bottleneck, attr=attributes)


def plot_test_accuracies_cub(subsets, n_bootstrap=1, hard_bottleneck=False, model_strings=None):
    """
    Plots test accuracies for different subsets for the CUB dataset. Histories must already exist.
    Plots the test-accuracies as pdf format and with minimal border, so that plot is suitable for using in LaTeX.
    Saves the plot in the appropriate Shapes file-structure.

    Args:
        subsets (list of int): The list of subsets to plot for.
        n_bootstrap (int, optional): The amount of bootstrap iterations used, in order to access correct history.
        history_folder (str, optional): Folder for where histories are saved. Defaults to "history/".
        hard_bottleneck (bool): If True, will load histories with hard-bottleneck, and save with "_hard" in name.
        model_strings (list of str, optional): List of string name of models to use. If None,
            will use the constants value of names. Defaults to None.
    """
    if model_strings is None:
        model_strings = MODEL_STRINGS_CUB

    histories_dict = {}
    for n_subset in subsets:
        histories = load_history_cub(n_bootstrap, n_subset, hard_bottleneck=hard_bottleneck)
        if n_subset is not None:
            histories_dict[n_subset] = histories
        else:
            histories_dict[30] = histories

    subsets = subsets.copy()
    for i in range(len(subsets)):  # n_subset = None means full data set, which is about 30 images per class.
        if subsets[i] is None:
            subsets[i] = 30

    plot_test_accuracies(histories_dict, subsets, model_strings=model_strings, n_bootstrap=n_bootstrap)
    save_test_plot_cub(n_bootstrap, hard_bottleneck=hard_bottleneck)


def plot_oracles_cub():
    pass


def plot_perturbed_images(perturbed_images, original_images, original_preds, new_preds, iterations_list,
                          cols=4, max_rows=3, dataset_name="shapes", adversarial_filename=None):
    """
    Plots a grid of original and perturbed images showing their original and new predictions.

    Args:
        perturbed_images (list): List of perturbed image tensors.
        original_images (list): List of original image tensors.
        original_preds (list): List of original predictions.
        new_preds (list): List of new predictions after perturbation.
        iterations_list (list): List of iterations ran
        cols (int): Number of columns in the subplot grid.
        dataset_name (str, optional): Shapes or CUB. Defaults to "shapes".
        adversarial_filename (name, optional): Name of file. Defaults to None.
    """
    plt.figure(figsize=(cols * 4, max_rows * 4))

    for i, (original_image, perturbed_image) in enumerate(zip(original_images, perturbed_images)):
        if i > max_rows + 2:
            break
        # Plot original image
        plt.subplot(max_rows, cols, 2 * i + 1)
        plot_tensor(original_image)
        plt.title(f"Original: {original_preds[i]}", fontsize=14)
        plt.axis("off")

        # Plot perturbed image
        plt.subplot(max_rows, cols, 2 * i + 2)
        plot_tensor(perturbed_image)
        plt.title(f"Perturbed: {new_preds[i]}, Iterations: {iterations_list[i]}", fontsize=14)
        plt.axis("off")

    plt.tight_layout()
    save_adversarial_image_shapes(dataset_name, adversarial_filename)
    plt.close()


def plot_perturbed(output, alpha, concept_threshold, dataset_name, n_classes=None):
    # Get mapping of concepts, make path-utils function
    n_images = len(output["perturbed_images"])
    attribute_mapping = get_attribute_mapping(dataset_name)
    attribute_names = get_attribute_names(dataset_name)
    class_names_mapping = get_class_names(dataset_name, n_classes)
    attr_string = ""
    for i in range(n_images):
        plot_tensor(output["original_images"][i])
        save_single_image(path=output["paths"][i][0], dataset_name=dataset_name, alpha=alpha,
                          concept_threshold=concept_threshold, type="original", index=i)
        plot_tensor(output["perturbed_images"][i])
        save_single_image(path=output["paths"][i][0], dataset_name=dataset_name, alpha=alpha,
                          concept_threshold=concept_threshold, type="perturbed", index=i)
        concepts_list = []
        attr_predictions = output["attr_predictions"][i].squeeze()  # All the attr-predictions for image number i
        n_concepts = len(attr_predictions)
        n_predicted_concepts = 0
        for j in range(n_concepts):
            attr_prediction = attr_predictions[j].int().item()  # Attr prediction number j for image number i
            if attr_prediction == 0:  # Attribute not predicted, continue
                continue
            n_predicted_concepts += 1
            concept = attribute_names[attribute_mapping[j]]
            concepts_list.append(concept)
        attr_string += output["paths"][i][0]
        attr_string += f": \nNumber of Predicted Concepts: {n_predicted_concepts} \n"
        attr_string += f"Concept Accuracy: {output['attr_accuracy'][i]:.4f}, "
        attr_string += f"Concept Precision: {output['attr_precision'][i]:.4f}, "
        attr_string += f"Concept Recall: {output['attr_recall'][i]:.4f} \n"
        attr_string += str(concepts_list)
        attr_string += "\n"
        new_prediction = class_names_mapping[output["new_predictions"][i]]
        original_prediction = class_names_mapping[output["original_predictions"][i]]

        attr_string += f"Original Prediction: {original_prediction}, New Prediction: {new_prediction=}\n\n"
    save_adversarial_text_file(attr_string, dataset_name, alpha, concept_threshold)


def plot_tensor(image_tensor):
    """
    Plots an image tensor. This means detaching it, moving to cpu, squeezing and transposing it.

    Args:
        image_tensor (torch tensor): The tensor to plot
    """
    numpy_image = image_tensor.detach().cpu().numpy().squeeze()
    numpy_image = numpy_image.transpose(1, 2, 0)
    height, width, _ = numpy_image.shape
    fig, ax = plt.subplots(figsize=(width / height, 1), dpi=300)
    plt.imshow(numpy_image)
    plt.axis("off")  # Remove ticks and stuff
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove whitespace
