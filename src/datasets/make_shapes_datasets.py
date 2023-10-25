"""
Functions for making various versions of the artificial `shapes` toy-dataset.
Here is the structure of the functions:
    `generate_shapes_dataset()`, and the functions that calls it, makes a full dataset.
    `create_and_save_image()` makes a single images, sometimes with many shapes in it.
    `make_single_shape()` makes a single shape, to be put into an image.

The specifications of the contents are determined by the image `attributes`, which contain information
on position, rotation, color, thickness and more. This is drawn by `draw_attributes()`.

The attributes can be made correlatied with the classes, as the intention of this dataset.
Each image gets assigned `concepts` from `draw_concept_labels()`, where a concept might be
`big_figure` or `thick_outline` (True or False). The attribute is still drawn randomly in some interval,
but the interval is different depending on the concept.

The probabilities of a image getting assigned a concept is draw by `draw_concept_probabilities()`. If
`equal_probabilities` is `True`, this will make all proabilities 0.5, resulting in no corelation between
class and concept. If it is `False` however, the function contains a scheme of drawing correlated probabilites
for each class for different amount of total classes.

The datasets gets saved and organised in folders, with one folder of images for each class. The concept-labels,
class labels and image-paths can be found in `data-list`, which is stored as a pickle file. This can be split
into "train", "validation" and "test". There are also help function for renaming and making subsets of data,
found in `datasets_shapes.py`.
"""

import os
import shutil
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.common.utils import split_dataset


def draw_concept_probabilities(n_classes, class_number, signal_strength=0.98, equal_probabilities=False,
                               use_position_concepts=False, use_background_concepts=False):
    """
    Draws the probabilites of drawing concept-lables. Draws one probability dict for one image.
    This function returns the probability that a shape will have a concept, which is drawn in `draw_concept_labels()`.

    This is made in order to have a strong corelation between some of the classes and some of the concepts,
    as determined in this function. The corelation scheme is unique for different number of classes.
    If `equal_probabilities` is `True`, all of the probabilites will be 0.5 (no correlation).

    The concept-labels contains `True` or `False` for concepts such as `big_figure` and `thick_outline`,
    which is then used to draw the exact values from `draw_attributes()`, which draws randomly,
    but the concepts determines the interval they are drawn in.

    Args:
        n_classes (int): The amount of classes to draw concepts from.
        class_number (int): The class that are currently drawn from (between 0 and `n_classes`).
        signal_strength (float, optional): The signal between classes and the probability they
            get assigned. If `0.98`, a class will have a 98% or 2% probability of having a concept.
        equal_probabilities (bool, optional): If True, will make dataset with no correlation. Defaults to False.
        use_position_concepts (bool, optional): If True, the position is drawn according to
            `concept_labels["right_side"]` and `concept_labels["upper_side"]`. This means that
            the concept_labels determines the quadrant the shape is made in. If False, the shape
            will be drawn randomly in the whole grid. It is recommended to set `True` when there
            is just one shape in an image (since then there are two more concepts to use), but
            `False` when there are two or more shapes (since they will heavily overlap otherwise.
        use_background_concepts (bool, optional): If True, will use four background concepts. These are dark-color
            upper half, dark color lower half, stripes upper half and striped lower half. If not, background will
            be completely black.

    Raises:
        ValueError: If the correlation scheme is not implemented for a given `n_classes`.

    Returns:
        dict: Dictionary of probabilites, to be given to `draw_concept_labels()`.
    """
    if equal_probabilities:  # Do not draw any corelations
        return None
    if n_classes == 4:
        # Bad combination: stripes, thick_outline, small_figure
        thick_outline_prob = set_probability(class_number, [0, 1], signal_strength)
        big_figure_prob = set_probability(class_number, [0, 2], signal_strength)
        dark_facecolor_prob = set_probability(class_number, [0, 3], signal_strength)
        dark_outline_prob = set_probability(class_number, [1, 2], signal_strength)
        stripes_prob = set_probability(class_number, [2, 3], signal_strength)
        if use_position_concepts:
            right_side_prob = set_probability(class_number, [2, 3], signal_strength)
            upper_side_prob = set_probability(class_number, [0, 1], signal_strength)
        if use_background_concepts:
            dark_upper_background_prob = set_probability(class_number, [1, 3], signal_strength)
            dark_lower_background_prob = set_probability(class_number, [0], signal_strength)
            upper_background_stripes_prob = set_probability(class_number, [1, 2, 3], signal_strength)
            lower_background_stripes_prob = set_probability(class_number, [0, 2], signal_strength)
    elif n_classes == 5:
        thick_outline_prob = set_probability(class_number, [0], signal_strength)
        big_figure_prob = set_probability(class_number, [2, 4], signal_strength)
        dark_facecolor_prob = set_probability(class_number, [4], signal_strength)
        dark_outline_prob = set_probability(class_number, [3, 4], signal_strength)
        stripes_prob = set_probability(class_number, [2], signal_strength)
        if use_position_concepts:
            right_side_prob = set_probability(class_number, [0, 1], signal_strength)
            upper_side_prob = set_probability(class_number, [1, 2], signal_strength)
        if use_background_concepts:
            dark_upper_background_prob = set_probability(class_number, [1, 3], signal_strength)
            dark_lower_background_prob = set_probability(class_number, [2, 4], signal_strength)
            upper_background_stripes_prob = set_probability(class_number, [1, 4], signal_strength)
            lower_background_stripes_prob = set_probability(class_number, [0, 4], signal_strength)
    elif n_classes == 10:
        thick_outline_prob = set_probability(class_number, [0, 2, 4, 6, 8], signal_strength)
        big_figure_prob = set_probability(class_number, [0, 1, 2, 3, 4], signal_strength)
        dark_facecolor_prob = set_probability(class_number, [1, 4, 5, 8, 9], signal_strength)
        dark_outline_prob = set_probability(class_number, [0, 6, 7], signal_strength)
        stripes_prob = set_probability(class_number, [3, 7, 9], signal_strength)
        if use_position_concepts:
            right_side_prob = set_probability(class_number, [0, 1], signal_strength)
            upper_side_prob = set_probability(class_number, [1, 2], signal_strength)
        if use_background_concepts:
            dark_upper_background_prob = set_probability(class_number, [1, 3, 5, 6, 7], signal_strength)
            dark_lower_background_prob = set_probability(class_number, [0, 3, 7, 8, 9], signal_strength)
            upper_background_stripes_prob = set_probability(class_number, [5, 6, 9], signal_strength)
            lower_background_stripes_prob = set_probability(class_number, [2, 4, 5], signal_strength)
    elif n_classes == 15:
        thick_outline_prob = set_probability(class_number, [0, 2, 4, 6, 8, 10, 12, 14], signal_strength)
        big_figure_prob = set_probability(class_number, [0, 1, 2, 3, 4, 5, 6, 7], signal_strength)
        dark_facecolor_prob = set_probability(class_number, [1, 4, 5, 8, 9, 11, 13], signal_strength)
        dark_outline_prob = set_probability(class_number, [0, 6, 7, 10], signal_strength)
        stripes_prob = set_probability(class_number, [3, 7, 9, 12], signal_strength)
        if use_position_concepts:
            right_side_prob = set_probability(class_number, [0, 1], signal_strength)
            upper_side_prob = set_probability(class_number, [1, 2], signal_strength)
        if use_background_concepts:
            dark_upper_background_prob = set_probability(class_number, [1, 5, 8, 9, 11, 14], signal_strength)
            dark_lower_background_prob = set_probability(class_number, [1, 3, 7, 12, 13], signal_strength)
            upper_background_stripes_prob = set_probability(class_number, [4, 7, 10, 14], signal_strength)
            lower_background_stripes_prob = set_probability(class_number, [8, 9, 10, 13], signal_strength)
    elif n_classes == 21:
        thick_outline_prob = set_probability(class_number, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20], signal_strength)
        big_figure_prob = set_probability(class_number, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], signal_strength)
        dark_facecolor_prob = set_probability(class_number, [1, 4, 5, 8, 9, 11, 13, 15, 19], signal_strength)
        dark_outline_prob = set_probability(class_number, [3, 6, 7, 10, 17], signal_strength)
        stripes_prob = set_probability(class_number, [3, 7, 9, 12, 14, 16, 18, 19], signal_strength)
        if use_position_concepts:
            right_side_prob = set_probability(class_number, [0, 1], signal_strength)
            upper_side_prob = set_probability(class_number, [1, 2], signal_strength)
        if use_background_concepts:
            dark_upper_background_prob = set_probability(class_number, [2, 7, 18, 19, 20], signal_strength)
            dark_lower_background_prob = set_probability(class_number, [3, 6, 8, 14, 16, 19], signal_strength)
            upper_background_stripes_prob = set_probability(class_number, [1, 4, 15, 18, 20], signal_strength)
            lower_background_stripes_prob = set_probability(class_number, [4, 5, 10, 11, 12], signal_strength)
    else:
        message = f"Drawing probabilites not yet defined for {n_classes} classes. Only defined for [4, 5]. "
        message += f"Please use `equal_probabilities=True` or implement probabilites for {n_classes} classes. "
        raise ValueError(message)
    prob_dict = {"thick_outline_prob": thick_outline_prob, "big_figure_prob": big_figure_prob,
                 "dark_facecolor_prob": dark_facecolor_prob, "dark_outline_prob": dark_outline_prob,
                 "stripes_prob": stripes_prob}
    if use_position_concepts:
        prob_dict["right_side_prob"] = right_side_prob
        prob_dict["upper_side_prob"] = upper_side_prob
    if use_background_concepts:
        prob_dict["dark_upper_background_prob"] = dark_upper_background_prob
        prob_dict["dark_lower_background_prob"] = dark_lower_background_prob
        prob_dict["upper_background_stripes_prob"] = upper_background_stripes_prob
        prob_dict["lower_background_stripes_prob"] = lower_background_stripes_prob
    return prob_dict


def set_probability(class_number, in_classes, signal_strength):
    """
    Help-function for `draw_concept_labels()`. Given a class-number and the `in_classes` (a list of
    classes that will have high signal_strength), sets the probability.
    If `class_number` is in `in_classes`, the probability will be `signal_strength`, if not it will
    be `1 - signal_strength`.

    Args:
        class_number (int): The class number for the image.
        in_classes (list of int): The classes to have high singal-strength.
        signal_strength (float): The probability the in-classes gets assigned.

    Returns:
        float: The probability, either `signal_strength` or `1 - signal_strength`.
    """
    if class_number in in_classes:
        probability = signal_strength
    else:
        probability = 1 - signal_strength
    return probability


def draw_concept_labels(probability_dict=None, use_position_concepts=False, use_background_concepts=False):
    """
    Draws the concept-labels, given the probabilities.
    The concepts labels are `True` and `False` for things like `big_figure` and `thick_outline`.
    The booleans are drawn according to probabilities in `probability_dict`, which is returned from
    `draw_concepts_probabilities()`.
    The concepts are used to draw the actual image attributes in `draw_attributes()`.

    Args:
        probability_dict (dict, optional): The dictionary of probabilites for each concepts.
            Is returned from `draw_concept_probabilities()`. If None, will use 0.5 for each probability,
            which results in no correlation between classes and concepts.
        use_position_concepts (bool, optional): If True, the position is drawn according to
            `concept_labels["right_side"]` and `concept_labels["upper_side"]`. This means that
            the concept_labels determines the quadrant the shape is made in. If False, the shape
            will be drawn randomly in the whole grid. It is recommended to set `True` when there
            is just one shape in an image (since then there are two more concepts to use), but
            `False` when there are two or more shapes (since they will heavily overlap otherwise.
        use_background_concepts (bool, optional): If True, will use four background concepts. These are dark-color
            upper half, dark color lower half, stripes upper half and striped lower half. If not, background will
            be completely black.

    Returns:
        dict: A dictionary with concepts as keys and 1 and 0 (representing booleans) as values.
            Should be sent to `make_single_shape()`.
    """
    if probability_dict is None:  # Set all probabilites to default value 0.5
        probability_dict = {"thick_outline_prob": 0.5, "big_figure_prob": 0.5, "dark_facecolor_prob": 0.5,
                            "dark_outline_prob": 0.5, "stripes_prob": 0.5, "right_side_prob": 0.5,
                            "upper_side_prob": 0.5, "dark_upper_background_prob": 0.5,
                            "dark_lower_background_prob": 0.5, "upper_background_stripes_prob": 0.5,
                            "lower_background_stripes_prob": 0.5}
    thick_outline = False
    big_figure = False
    dark_facecolor = False
    dark_outline = False
    stripes = False
    right_side = False
    upper_side = False
    dark_upper_background = False
    dark_lower_background = False
    upper_background_stripes = False
    lower_background_stripes = False
    if np.random.uniform() < probability_dict["thick_outline_prob"]:
        thick_outline = True
    if np.random.uniform() < probability_dict["big_figure_prob"]:
        big_figure = True
    if np.random.uniform() < probability_dict["dark_facecolor_prob"]:
        dark_facecolor = True
    if np.random.uniform() < probability_dict["dark_outline_prob"]:
        dark_outline = True
    if np.random.uniform() < probability_dict["stripes_prob"]:
        stripes = True
    if use_position_concepts:
        if np.random.uniform() < probability_dict["right_side_prob"]:
            right_side = True
        if np.random.uniform() < probability_dict["upper_side_prob"]:
            upper_side = True
    if use_background_concepts:
        if np.random.uniform() < probability_dict["dark_upper_background_prob"]:
            dark_upper_background = True
        if np.random.uniform() < probability_dict["dark_lower_background_prob"]:
            dark_lower_background = True
        if np.random.uniform() < probability_dict["upper_background_stripes_prob"]:
            upper_background_stripes = True
        if np.random.uniform() < probability_dict["lower_background_stripes_prob"]:
            lower_background_stripes = True

    # Pack all the information above into a dict
    concept_labels = {"thick_outline": int(thick_outline), "big_figure": int(big_figure),
                      "dark_facecolor": int(dark_facecolor), "dark_outline": int(dark_outline), "stripes": int(stripes)}
    if use_position_concepts:  # Add positional arguments
        concept_labels["right_side"] = int(right_side)
        concept_labels["upper_side"] = int(upper_side)
    if use_background_concepts:
        concept_labels["dark_upper_background"] = int(dark_upper_background)
        concept_labels["dark_lower_background"] = int(dark_lower_background)
        concept_labels["upper_background_stripes"] = int(upper_background_stripes)
        concept_labels["lower_background_stripes"] = int(lower_background_stripes)
    return concept_labels


def draw_attributes(concept_labels, x_lim=10, y_lim=10, edge_diff=1, use_position_concepts=True):
    """
    Draws the attributes of a shape, given the concepts determined by `concept_labels`, which is
    a dict with concepts as keys and booleans as values. The concepts determines things like `big_figure`
    and `thick_outline`, but the actual size and linewidth are still draw randomly in an interval
    by `draw_attributes()`, which is called in this function.

    Args:
        concept_labels (dict): The concepts, returned by `draw_concept_labels()`. This determines
            things like the intervals the height, width or radius are drawn from (concept_labels["big_figure"]),
            and the interval the linewidth is drawn from (concept_labels["thick_outline"]).
        x_lim (int, optional): The ax limit, set with `ax.set_xlim()`. This is the virtual interval the plot
            is set to be in. Defaults to 10.
        y_lim (int, optional): The ax limit, set with `ax.set_ylim()`. This is the virtual interval the plot
            is set to be in. Defaults to 10.
        edge_diff (int, optional): How long from the edge the figure can be drawn from. The more `edge_diff`,
            the less a figure can be outside the plot. Defaults to 1.
        use_position_concepts (bool, optional): If True, the position is drawn according to
            `concept_labels["right_side"]` and `concept_labels["upper_side"]`. This means that
            the concept_labels determines the quadrant the shape is made in. If False, the shape
            will be drawn randomly in the whole grid. It is recommended to set `True` when there
            is just one shape in an image (since then there are two more concepts to use), but
            `False` when there are two or more shapes (since they will heavily overlap otherwise.

    Returns:
        dict: A dict containing the exact attributes for the figure. Note that it may contain something
            that is not used. For example, every dict contains `radius`, but this is not used for polygons.
    """
    if not use_position_concepts:  # Draw position randomly in the whole plot
        x1 = np.random.uniform(0 + edge_diff, x_lim - edge_diff)
        y1 = np.random.uniform(0 + edge_diff, y_lim - edge_diff)
    else:  # Draw position according to concept_labels
        if concept_labels["right_side"]:
            x1 = np.random.uniform(x_lim / 2 + edge_diff, x_lim - edge_diff)
        else:
            x1 = np.random.uniform(0 + edge_diff, x_lim / 2 - edge_diff)
        if concept_labels["upper_side"]:
            y1 = np.random.uniform(y_lim / 2 + edge_diff, y_lim - edge_diff)
        else:
            y1 = np.random.uniform(0 + edge_diff, y_lim / 2 - edge_diff)

    orientation = np.random.randint(0, 360)  # Rotation of the figure, sometimes called `angle`.
    theta1 = np.random.uniform(0, 360)  # The thetas are used for the wedges only.
    theta2 = np.random.uniform(theta1 + 100, theta1 + 110)

    if concept_labels["thick_outline"]:
        linewidth = np.random.uniform(1, 1.2)
    else:  # Thin outline
        linewidth = np.random.uniform(0.2, 0.5)

    if concept_labels["big_figure"]:
        radius = np.random.uniform(2.8, 3.3)
        width = np.random.uniform(2, 2.5)
        height = np.random.uniform(2, 2.5)
    else:
        radius = np.random.uniform(1.5, 2)
        width = np.random.uniform(1, 1.2)
        height = np.random.uniform(1, 1.2)

    if concept_labels["dark_facecolor"]:
        facecolor = "b"
    else:
        facecolor = "y"

    if concept_labels["dark_outline"]:
        edgecolor = "r"
    else:
        edgecolor = "w"

    if concept_labels["stripes"]:
        hatch = "////"  # This will make the shapes stripy.
    else:
        hatch = None

    attributes = {"linewidth": linewidth, "facecolor": facecolor, "edgecolor": edgecolor, "hatch": hatch,
                  "radius": radius, "height": height, "width": width, "x1": x1, "y1": y1, "orientation": orientation,
                  "theta1": theta1, "theta2": theta2}
    return attributes


def make_circle(attributes):
    """
    Given the shape-attributes, draws a matplotlib.patches Patch.
    This function is called from `make_single_shape()`.

    Args:
        attributes (dict): The attributes, retuned by `draw_attributes()`.

    Returns:
        matplotlib.patches.Circle: The patch, to be out into a figure.
    """
    patch = patches.Circle((attributes["x1"], attributes["y1"]), radius=attributes["radius"],
                           linewidth=attributes["linewidth"], facecolor=attributes["facecolor"],
                           edgecolor=attributes["edgecolor"], hatch=attributes["hatch"])
    return patch


def make_rectangle(attributes):
    """
    Given the shape-attributes, draws a matplotlib.patches Patch.
    This function is called from `make_single_shape()`.

    Args:
        attributes (dict): The attributes, retuned by `draw_attributes()`.

    Returns:
        matplotlib.patches.Rectangle: The patch, to be out into a figure.
    """
    patch = patches.Rectangle((attributes["x1"], attributes["y1"]), width=attributes["width"],
                              height=attributes["height"], linewidth=attributes["linewidth"],
                              angle=attributes["orientation"], facecolor=attributes["facecolor"],
                              edgecolor=attributes["edgecolor"], hatch=attributes["hatch"])
    return patch


def make_regular_polygon(n_vertices, attributes):
    """
    Given the shape-attributes, draws a matplotlib.patches Patch.
    This function is called from `make_single_shape()`.

    Args:
        n_verticies (int): The amount of verticies for the polygon. `3` will result in
            triangle, `4` in square, and so on. All of the figures have sides of the same length.
        attributes (dict): The attributes, retuned by `draw_attributes()`.

    Returns:
        matplotlib.patches.Polygon: The patch, to be out into a figure.
    """
    patch = patches.RegularPolygon((attributes["x1"], attributes["y1"]), numVertices=n_vertices,
                                   radius=attributes["radius"], linewidth=attributes["linewidth"],
                                   orientation=attributes["orientation"], facecolor=attributes["facecolor"],
                                   edgecolor=attributes["edgecolor"], hatch=attributes["hatch"])
    return patch


def make_wedge(attributes):
    """
    Given the shape attributes, draws a matplotlib.patches Patch.
    This function is called from `make_single_shape()`.
    A wedge is a "pie slice" shapes object. The arguments `theta1` and `theta2` determines where in
    the circle the wedge starts and stops. This therefore determines both the orientation and the size
    of the "slice". For example, a difference of 90 between the thetas will result in a quarter of a pie slice.

    Args:
        attributes (dict): The attributes, retuned by `draw_attributes()`.

    Returns:
        matplotlib.patches.Wedge: The patch, to be out into a figure.
    """
    patch = patches.Wedge((attributes["x1"], attributes["y1"]), r=attributes["radius"], theta1=attributes["theta1"],
                          theta2=attributes["theta2"], linewidth=attributes["linewidth"],
                          facecolor=attributes["facecolor"], edgecolor=attributes["edgecolor"],
                          hatch=attributes["hatch"])
    return patch


def make_ellipse(attributes):
    """
    Given the shape attributes, draws a matplotlib.patches Patch.
    This function is called from `make_single_shape()`.

    Args:
        attributes (dict): The attributes, retuned by `draw_attributes()`.

    Returns:
        matplotlib.patches.Ellipse: The patch, to be out into a figure.
    """
    patch = patches.Ellipse((attributes["x1"], attributes["y1"]), width=attributes["width"],
                            height=attributes["height"], linewidth=attributes["linewidth"],
                            angle=attributes["orientation"], facecolor=attributes["facecolor"],
                            edgecolor=attributes["edgecolor"], hatch=attributes["hatch"])
    return patch


def make_single_shape(shape, concept_labels, use_position_concepts=True):
    """
    Makes a single shape (matplotlib.patches.Path).
    This funnction first finds the shapes attributes given `concept_labels`, and then calls the
    respective shape function to make that shape.
    The `concept_labels` can be obtained with `draw_concept_labels()`, which given a probability
    draws the concepts for a shape. The concepts determines things like `big_figure` and
    `thick_outline`, but the actual size and linewidth are still draw randomly in an interval
    by `draw_attributes()`, which is called in this function.

    Args:
        shape (str): Shape to be made. Must be in ["triangle", "pentagon", "hexagon",
            "rectangle", "ellipse", "circle", "wedge"].
        concept_labels (dict): Dictionary of attributes for the shapes. Is returned by
            `draw_concept_labels()`, see that function.
        use_position_concepts (bool, optional): If True, the position is drawn according to
            `concept_labels["right_side"]` and `concept_labels["upper_side"]`. This means that
            the concept_labels determines the quadrant the shape is made in. If False, the shape
            will be drawn randomly in the whole grid. It is recommended to set `True` when there
            is just one shape in an image (since then there are two more concepts to use), but
            `False` when there are two or more shapes (since they will heavily overlap otherwise.

    Raises:
        ValueError: If `shape` is not a correct shape.

    Returns:
        matplotlib.pathces.Path: The shape drawn, as a matplotlib patch.
    """
    shape = shape.strip().lower()
    if shape not in ["triangle", "pentagon", "hexagon", "rectangle", "ellipse", "circle", "wedge"]:
        message = f"Argument `shape` must be in [\"triangle\", \"pentagon\", \"hexagon\", \"rectangle\", "
        message += f"\"ellipse\", \"circle\", \"wedge\"]. Got {shape}"
        raise ValueError(message)
    n_vertices = 0
    if shape == "triangle":
        n_vertices = 3
    elif shape == "pentagon":
        n_vertices = 5
    elif shape == "hexagon":
        n_vertices = 6
    if shape in ["triangle", "pentagon", "hexagon"]:
        shape = "polygon"  # The shape generator abstracts over n-gons.

    attributes = draw_attributes(concept_labels, use_position_concepts=use_position_concepts)
    if shape == "circle":
        return make_circle(attributes)
    elif shape == "rectangle":
        return make_rectangle(attributes)
    elif shape == "polygon":
        return make_regular_polygon(n_vertices, attributes)
    elif shape == "wedge":
        return make_wedge(attributes)
    elif shape == "ellipse":
        return make_ellipse(attributes)


def add_background_concepts(ax, concept_labels):
    if concept_labels["dark_upper_background"]:
        upper_background_color = "magenta"
    else:
        upper_background_color = "palegreen"
    if concept_labels["upper_background_stripes"]:
        upper_background_hatch = "////"
    else:
        upper_background_hatch = None
    upper_rectangle = patches.Rectangle((0, 5), 10, 5, facecolor=upper_background_color,
                                        edgecolor="black", hatch=upper_background_hatch,
                                        linewidth=0.001)
    ax.add_patch(upper_rectangle)

    if concept_labels["dark_lower_background"]:
        lower_background_color = "indigo"
    else:
        lower_background_color = "darkseagreen"
    if concept_labels["lower_background_stripes"]:
        lower_background_hatch = "////"
    else:
        lower_background_hatch = None
    lower_rectangle = patches.Rectangle((0, 0), 10, 5, facecolor=lower_background_color,
                                        edgecolor="black", hatch=lower_background_hatch,
                                        linewidth=0.001)
    ax.add_patch(lower_rectangle)
    return ax


def create_and_save_image(shapes, concept_labels=None, height=64, width=64, dpi=100, base_dir="data/shapes/",
                          class_subdir="", fig_name="fig.png",
                          use_position_concepts=True, use_background_concepts=False):
    """
    Makes a single images and saves it in a respective folder. A single image may contain many shapes.
    This is called from `create_and_save_images()`, which will systematically call this function many times to make
    a complete dataset.

    Args:
        shapes (list of str): List of the shapes to be put into the images. For example ["triangle", "ellipse"]
            for a triangle and ellipse in one figure, or ["rectangle"] for only one rectangle.
        concept_labels (dict): The concepts, returned by `draw_concept_labels()`. This determines
            things like the intervals the height, width or radius are drawn from (concept_labels["big_figure"]),
            and the interval the linewidth is drawn from (concept_labels["thick_outline"]).
            This is used int `draw_attributes()`
        height (int, optional): Height of the plot in inches. Total vertical pixels are
            `height * dpi`. Defaults to 64.
        width (int, optional): Width of the plot in inches. Total horizontal pixels are
            `width * dpi`. Defaults to 64.
        dpi (int, optional): `Dots Per Inch`. Determines how many pixels to assign to an inch in the
            plot when it is saved. Defaults to 100.
        base_dir (str, optional): Directory where the directories of images will be saved. Figures are saved as
            `base_dir` + `class_subdir` + `fig_name`. Defaults to "data/shapes/".
        class_subdir (str, optional): Directory of a given combination of shapes.
        fig_name (str, optional): The name of the figure. Defaults to "fig.png".
        use_position_concepts (bool, optional): If True, the position is drawn according to
            `concept_labels["right_side"]` and `concept_labels["upper_side"]`. This means that
            the concept_labels determines the quadrant the shape is made in. If False, the shape
            will be drawn randomly in the whole grid. It is recommended to set `True` when there
            is just one shape in an image (since then there are two more concepts to use), but
            `False` when there are two or more shapes (since they will heavily overlap otherwise.
        use_background_concepts (bool, optional): If True, will use four background concepts. These are dark-color
            upper half, dark color lower half, stripes upper half and striped lower half. If not, background will
            be completely black.

    Returns:
        concept_labels: The concept labels drawn for the image.
    """
    fig, ax = plt.subplots(figsize=(height / dpi, width / dpi))
    ax = fig.add_axes([0, 0, 1, 1], aspect="auto")  # Make plot cover the whole image
    ax.set_xlim(0, 10)  # Virtual coords in image are from 0 to 10.
    ax.set_ylim(0, 10)
    ax.set_facecolor("black")

    if use_background_concepts:
        ax = add_background_concepts(ax, concept_labels)

    for shape in shapes:
        patch = make_single_shape(shape, concept_labels, use_position_concepts=use_position_concepts)
        ax.add_patch(patch)

    ax.add_patch(patch)

    plt.savefig(base_dir + class_subdir + fig_name, dpi=dpi, pad_inches=0)
    plt.close()
    return concept_labels


def generate_shapes_dataset(class_names, shape_combinations, n_images_class=10, equal_probabilities=False,
                            use_position_concepts=True, use_background_concepts=False, signal_strength=0.98,
                            split_data=True, base_dir="data/shapes/shapes_testing/", seed=57, verbose=True):
    """
    Makes a shapes-dataset. This function makes a directory of a dataset, and calls `create_and_save_image()` many times
    to create each image.

    Args:
        class_names (list of str): List of the names for each class.
        shape_combinations (list of list of str): Nested list structure. Each element corresponds to each class.
            The elements are lists of the shapes that will be in that class. For example
            [["triangle, "circle"], ["rectangle"], ...] where first class has two shapes and second has one.
        n_images_class (int, optional): The amount of images in each class. Defaults to 10.
        equal_probabilities (bool, optional): If True, the probabilites for concepts will all be set
            to 0.5, resulting in no corelation. Defaults to False.
        use_position_concepts (bool, optional): If True, the position is drawn according to
            `concept_labels["right_side"]` and `concept_labels["upper_side"]`. This means that
            the concept_labels determines the quadrant the shape is made in. If False, the shape
            will be drawn randomly in the whole grid. It is recommended to set `True` when there
            is just one shape in an image (since then there are two more concepts to use), but
            `False` when there are two or more shapes (since they will heavily overlap otherwise.
        use_background_concepts (bool, optional): If True, will use four background concepts. These are dark-color
            upper half, dark color lower half, stripes upper half and striped lower half. If not, background will
            be completely black.
        signal_strength (float, optional): The probability that determines the correlation between classes
            and concepts when `equal_probabilities` is `False`. (probabilites are either `signal_strength`
            or `1 - signal_strenth`). Defaults to 0.98.
        split_data (bool, optional): If True, will split the data-table in "train", "validation" and
            "test". Defaults to True.
        base_dir (str, optional): The directory of the dataset. Contains both the path and the name of the
            folder for the dataset. Defaults to "data/shapes/shapes_testing/".
        seed (int, optional): A seed for the rng.. Defaults to 57.
        verbose (bool, optional): If True, will print the progress. Defaults to True.
    """
    np.random.seed(seed)

    if os.path.exists(base_dir):  # Delete folder and make new if it already exists
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)

    data_list = []  # List of instances" img_path, class_label, attribute_label, class_name
    for i in range(len(class_names)):
        shape_name = class_names[i]
        shape_combination = shape_combinations[i]
        if verbose:
            print(f"Beginning shape combination {shape_name}:")
        class_subdir = str(i) + "_" + shape_name + "/"
        full_dir_path = base_dir + class_subdir

        os.makedirs(full_dir_path)

        for j in range(n_images_class):
            if verbose and ((j + 1) % 100 == 0):
                print(f"Image number [{j + 1} / {n_images_class}].")
            fig_name = shape_name + "_" + str(j) + ".png"
            # Draw concept-probabilites to use for single image
            concept_probabilities = draw_concept_probabilities(n_classes=len(class_names), class_number=i,
                                                               signal_strength=signal_strength,
                                                               equal_probabilities=equal_probabilities,
                                                               use_position_concepts=use_position_concepts,
                                                               use_background_concepts=use_background_concepts)
            # Draw the actual probabilites for the image
            concept_labels = draw_concept_labels(probability_dict=concept_probabilities,
                                                 use_position_concepts=use_position_concepts,
                                                 use_background_concepts=use_background_concepts)
            # Create and save the image.
            create_and_save_image(shapes=shape_combination, concept_labels=concept_labels, base_dir=base_dir,
                                  class_subdir=class_subdir, fig_name=fig_name,
                                  use_position_concepts=use_position_concepts,
                                  use_background_concepts=use_background_concepts)
            data_list.append({"img_path": base_dir + class_subdir + fig_name, "class_label": i,
                              "attribute_label": concept_labels, "class_name": shape_name})

    if verbose:
        print("Begin writing tables:")
    tables_dir = base_dir + "tables/"  # Folder where data with labels and paths should be saved
    if os.path.exists(tables_dir):
        shutil.rmtree(tables_dir)
    os.makedirs(tables_dir)
    with open(tables_dir + "data_list.pkl", "wb") as outfile:
        pickle.dump(data_list, outfile)

    if split_data:  # Split data in train, validation and test-set.
        split_dataset(data_list, tables_dir, include_test=True)
    if verbose:
        print("Finished writing tables.")


def make_shapes_10k_c4_correlation():
    """
    Makes a dataset with 10k images of 4 classes, with concept-class correlation.
    """
    class_names = ["triangle", "rectangle", "hexagon", "ellipse"]
    shape_combinations = [[shape_name] for shape_name in class_names]
    generate_shapes_dataset(
        class_names=class_names, shape_combinations=shape_combinations, n_images_class=10000, split_data=True,
        base_dir="data/shapes/shapes_10k_c4_correlation_1/", use_position_concepts=True)


def make_shapes_2k_c5_correlation():
    """
    Makes a dataset with 2k images of 5 classes, with concept-class correlation.
    """
    class_names = ["triangle", "rectangle", "hexagon", "ellipse", "wedge"]
    shape_combinations = [[shape_name] for shape_name in class_names]
    generate_shapes_dataset(
        class_names=class_names, shape_combinations=shape_combinations, n_images_class=2000, split_data=True,
        base_dir="data/shapes/shapes_2k_c5_correlation/", use_position_concepts=True)


def make_shapes_2k_c10():
    """
    Makes a dataset with 2k images of 10 classes, with concept-class correlation.
    """
    class_names = ["triangle_triangle", "triangle_rectangle", "triangle_ellipse", "triangle_hexagon",
                   "rectangle_rectangle", "rectangle_ellipse", "rectangle_hexagon",
                   "ellipse_ellipse", "ellipse_hexagon", "hexagon_hexagon"]
    shape_combinations = []  # Nested list structure
    for shape_name in class_names:
        shape_combinations.append([shape_name.split("_")[0], shape_name.split("_")[1]])
    base_dir = "data/shapes/shapes_2k_c10_a5/"
    generate_shapes_dataset(class_names=class_names, shape_combinations=shape_combinations, n_images_class=2000,
                            split_data=True, base_dir=base_dir, use_position_concepts=False)


def make_shapes_1k_c15_a5():
    class_names = ["triangle_triangle", "triangle_rectangle", "triangle_hexagon", "triangle_ellipse", "triangle_wedge",
                   "rectangle_rectangle", "rectangle_hexagon", "rectangle_ellipse", "rectangle_wedge",
                   "hexagon_hexagon", "hexagon_ellipse", "hexagon_wedge", "ellipse_ellipse", "ellipse_wedge",
                   "wedge_wedge"]
    shape_combinations = []
    for shape_name in class_names:
        shape_combinations.append([shape_name.split("_")[0], shape_name.split("_")[1]])
    base_dir = "data/shapes/shapes_1k_c15_a5/"
    generate_shapes_dataset(class_names=class_names, shape_combinations=shape_combinations, n_images_class=1000,
                            split_data=True, base_dir=base_dir, use_position_concepts=False)


def make_shapes_1k_c21_a5():
    class_names = ["triangle_triangle", "rectangle_triangle", "pentagon_triangle", "hexagon_triangle",
                   "ellipse_triangle", "triangle_wedge", "rectangle_rectangle", "pentagon_rectangle",
                   "hexagon_rectangle", "ellipse_rectangle", "rectangle_wedge", "pentagon_pentagon",
                   "hexagon_pentagon", "ellipse_pentagon", "pentagon_wedge", "hexagon_hexagon", "ellipse_hexagon",
                   "hexagon_wedge", "ellipse_ellipse", "ellipse_wedge", "wedge_wedge"]
    shape_combinations = []
    for shape_name in class_names:
        shape_combinations.append([shape_name.split("_")[0], shape_name.split("_")[1]])
    base_dir = "data/shapes/shapes_1k_c21_a5/"
    generate_shapes_dataset(class_names=class_names, shape_combinations=shape_combinations, n_images_class=1000,
                            split_data=True, base_dir=base_dir, use_position_concepts=False)


def make_shapes_1k_c10_a9():
    """
    Makes a dataset with 1k images of 10 classes, with concept-class correlation.
    Use background concepts
    """
    class_names = ["triangle_triangle", "triangle_rectangle", "triangle_ellipse", "triangle_hexagon",
                   "rectangle_rectangle", "rectangle_ellipse", "rectangle_hexagon",
                   "ellipse_ellipse", "ellipse_hexagon", "hexagon_hexagon"]
    shape_combinations = []  # Nested list structure
    for shape_name in class_names:
        shape_combinations.append([shape_name.split("_")[0], shape_name.split("_")[1]])
    base_dir = "data/shapes/shapes_1k_c10_a9/"
    generate_shapes_dataset(class_names=class_names, shape_combinations=shape_combinations, n_images_class=1000,
                            split_data=True, base_dir=base_dir, use_position_concepts=False,
                            use_background_concepts=True)


def make_shapes_1k_c15_a9():
    """
    Makes a dataset with 1k images of 10 classes, with concept-class correlation.
    Use background concepts
    """
    class_names = ["triangle_triangle", "triangle_rectangle", "triangle_hexagon", "triangle_ellipse", "triangle_wedge",
                   "rectangle_rectangle", "rectangle_hexagon", "rectangle_ellipse", "rectangle_wedge",
                   "hexagon_hexagon", "hexagon_ellipse", "hexagon_wedge", "ellipse_ellipse", "ellipse_wedge",
                   "wedge_wedge"]
    shape_combinations = []  # Nested list structure
    for shape_name in class_names:
        shape_combinations.append([shape_name.split("_")[0], shape_name.split("_")[1]])
    base_dir = "data/shapes/shapes_1k_c15_a9/"
    generate_shapes_dataset(class_names=class_names, shape_combinations=shape_combinations, n_images_class=1000,
                            split_data=True, base_dir=base_dir, use_position_concepts=False,
                            use_background_concepts=True)


def make_shapes_1k_c21_a9():
    """
    Makes a dataset with 1k images of 10 classes, with concept-class correlation.
    Use background concepts
    """
    class_names = ["triangle_triangle", "rectangle_triangle", "pentagon_triangle", "hexagon_triangle",
                   "ellipse_triangle", "triangle_wedge", "rectangle_rectangle", "pentagon_rectangle",
                   "hexagon_rectangle", "ellipse_rectangle", "rectangle_wedge", "pentagon_pentagon",
                   "hexagon_pentagon", "ellipse_pentagon", "pentagon_wedge", "hexagon_hexagon", "ellipse_hexagon",
                   "hexagon_wedge", "ellipse_ellipse", "ellipse_wedge", "wedge_wedge"]
    shape_combinations = []  # Nested list structure
    for shape_name in class_names:
        shape_combinations.append([shape_name.split("_")[0], shape_name.split("_")[1]])
    base_dir = "data/shapes/shapes_1k_c21_a9/"
    generate_shapes_dataset(class_names=class_names, shape_combinations=shape_combinations, n_images_class=1000,
                            split_data=True, base_dir=base_dir, use_position_concepts=False,
                            use_background_concepts=True)

# Signal strength 100


def make_shapes_1k_c10_a5_s100():
    """
    Makes a dataset with 2k images of 10 classes, with concept-class correlation.
    """
    class_names = ["triangle_triangle", "triangle_rectangle", "triangle_ellipse", "triangle_hexagon",
                   "rectangle_rectangle", "rectangle_ellipse", "rectangle_hexagon",
                   "ellipse_ellipse", "ellipse_hexagon", "hexagon_hexagon"]
    shape_combinations = []  # Nested list structure
    for shape_name in class_names:
        shape_combinations.append([shape_name.split("_")[0], shape_name.split("_")[1]])
    base_dir = "data/shapes/shapes_1k_c10_a5_s100/"
    generate_shapes_dataset(class_names=class_names, shape_combinations=shape_combinations, n_images_class=1000,
                            split_data=True, base_dir=base_dir, use_position_concepts=False, signal_strength=1)


def make_shapes_1k_c15_a5_s100():
    class_names = ["triangle_triangle", "triangle_rectangle", "triangle_hexagon", "triangle_ellipse", "triangle_wedge",
                   "rectangle_rectangle", "rectangle_hexagon", "rectangle_ellipse", "rectangle_wedge",
                   "hexagon_hexagon", "hexagon_ellipse", "hexagon_wedge", "ellipse_ellipse", "ellipse_wedge",
                   "wedge_wedge"]
    shape_combinations = []
    for shape_name in class_names:
        shape_combinations.append([shape_name.split("_")[0], shape_name.split("_")[1]])
    base_dir = "data/shapes/shapes_1k_c15_a5_s100/"
    generate_shapes_dataset(class_names=class_names, shape_combinations=shape_combinations, n_images_class=1000,
                            split_data=True, base_dir=base_dir, use_position_concepts=False, signal_strength=1)


def make_shapes_1k_c21_a5_s100():
    class_names = ["triangle_triangle", "rectangle_triangle", "pentagon_triangle", "hexagon_triangle",
                   "ellipse_triangle", "triangle_wedge", "rectangle_rectangle", "pentagon_rectangle",
                   "hexagon_rectangle", "ellipse_rectangle", "rectangle_wedge", "pentagon_pentagon",
                   "hexagon_pentagon", "ellipse_pentagon", "pentagon_wedge", "hexagon_hexagon", "ellipse_hexagon",
                   "hexagon_wedge", "ellipse_ellipse", "ellipse_wedge", "wedge_wedge"]
    shape_combinations = []
    for shape_name in class_names:
        shape_combinations.append([shape_name.split("_")[0], shape_name.split("_")[1]])
    base_dir = "data/shapes/shapes_1k_c21_a5_s100/"
    generate_shapes_dataset(class_names=class_names, shape_combinations=shape_combinations, n_images_class=1000,
                            split_data=True, base_dir=base_dir, use_position_concepts=False, signal_strength=1)


def make_shapes_1k_c10_a9_s100():
    """
    Makes a dataset with 1k images of 10 classes, with concept-class correlation.
    Use background concepts
    """
    class_names = ["triangle_triangle", "triangle_rectangle", "triangle_ellipse", "triangle_hexagon",
                   "rectangle_rectangle", "rectangle_ellipse", "rectangle_hexagon",
                   "ellipse_ellipse", "ellipse_hexagon", "hexagon_hexagon"]
    shape_combinations = []  # Nested list structure
    for shape_name in class_names:
        shape_combinations.append([shape_name.split("_")[0], shape_name.split("_")[1]])
    base_dir = "data/shapes/shapes_1k_c10_a9_s100/"
    generate_shapes_dataset(class_names=class_names, shape_combinations=shape_combinations, n_images_class=1000,
                            split_data=True, base_dir=base_dir, use_position_concepts=False,
                            use_background_concepts=True, signal_strength=1)


def make_shapes_1k_c15_a9_s100():
    """
    Makes a dataset with 1k images of 10 classes, with concept-class correlation.
    Use background concepts
    """
    class_names = ["triangle_triangle", "triangle_rectangle", "triangle_hexagon", "triangle_ellipse", "triangle_wedge",
                   "rectangle_rectangle", "rectangle_hexagon", "rectangle_ellipse", "rectangle_wedge",
                   "hexagon_hexagon", "hexagon_ellipse", "hexagon_wedge", "ellipse_ellipse", "ellipse_wedge",
                   "wedge_wedge"]
    shape_combinations = []  # Nested list structure
    for shape_name in class_names:
        shape_combinations.append([shape_name.split("_")[0], shape_name.split("_")[1]])
    base_dir = "data/shapes/shapes_1k_c15_a9_s100/"
    generate_shapes_dataset(class_names=class_names, shape_combinations=shape_combinations, n_images_class=1000,
                            split_data=True, base_dir=base_dir, use_position_concepts=False,
                            use_background_concepts=True, signal_strength=1)


def make_shapes_1k_c21_a9_s100():
    """
    Makes a dataset with 1k images of 10 classes, with concept-class correlation.
    Use background concepts
    """
    class_names = ["triangle_triangle", "rectangle_triangle", "pentagon_triangle", "hexagon_triangle",
                   "ellipse_triangle", "triangle_wedge", "rectangle_rectangle", "pentagon_rectangle",
                   "hexagon_rectangle", "ellipse_rectangle", "rectangle_wedge", "pentagon_pentagon",
                   "hexagon_pentagon", "ellipse_pentagon", "pentagon_wedge", "hexagon_hexagon", "ellipse_hexagon",
                   "hexagon_wedge", "ellipse_ellipse", "ellipse_wedge", "wedge_wedge"]
    shape_combinations = []  # Nested list structure
    for shape_name in class_names:
        shape_combinations.append([shape_name.split("_")[0], shape_name.split("_")[1]])
    base_dir = "data/shapes/shapes_1k_c21_a9_s100/"
    generate_shapes_dataset(class_names=class_names, shape_combinations=shape_combinations, n_images_class=1000,
                            split_data=True, base_dir=base_dir, use_position_concepts=False,
                            use_background_concepts=True, signal_strength=1)


if __name__ == "__main__":
    all_shapes = ["circle", "rectangle", "triangle", "pentagon", "hexagon", "ellipse", "wedge"]
    make_shapes_1k_c10_a5_s100()
    make_shapes_1k_c15_a5_s100()
    make_shapes_1k_c21_a5_s100()
    # make_shapes_1k_c10_a9_s100()
    # make_shapes_1k_c15_a9_s100()
    # make_shapes_1k_c21_a9_s100()
