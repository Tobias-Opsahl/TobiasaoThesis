"""
File for feature selecting the 112 attributes from the CUB dataset.
We try to see if they can be accurately represented by a smaller subset.
"""
import pickle

import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def read_processed_cub(path="data/CUB_processed/class_attr_data_10/", subset=""):
    """
    Read the processed cub data set, converts to arrays and returns.
    This is the attribute processed dataset from the CBM 2020 paper.
    The original data is stored in lists of dictionaries. This functions turns the
    attribute lables (112 amounts) into a [row x 112] x-array, and the class-labels
    as y-arrays. Does this for train, val, test and returns.

    Args:
        path (str, optional): Path to read files.
            Defaults to "data/CUB_processed/class_attr_data_10/".
        subset (str, optional): Provide if one only wants a subset for the 200 classes.
            Either int n of classes, or string "n_" as the files are saved.
            If one provides `n`, then `python initialize.py make_dataset --n_classes n`
            should be ran first to make the dataset.. Defaults to "".

    Returns:
        list: List of dataset tuples.
            [(x_train, y_train), (x_val, y_val), (x_test, y_test)].
    """
    if (subset == 200) or (subset is None) or (subset == "200_"):
        subset = ""  # Convert to correct format (all 200 classes included)
    if isinstance(subset, int):  # Convert from int to correct string
        subset = str(subset) + "_"
    types = ["train", "val", "test"]
    datasets = []  # The data arrays we will make
    for mode in types:
        # Read list of dictionaries
        data_list = pickle.load(open(path + subset + mode + ".pkl", "rb"))
        n = len(data_list)  # Rows
        n_attr = len(data_list[0]["attribute_label"])  # Amount of features
        features = np.zeros((n, n_attr))
        labels = np.zeros(n)
        for i in range(n):  # Convert from list to arrays
            labels[i] = data_list[i]["class_label"]
            features[i, :] = np.array(data_list[i]["attribute_label"])
        datasets.append((features, labels))

    return datasets


def get_rfe_ranking(x_train, y_train, max_iter=200):
    """
    Does Recurse Feature Elimination for feature selection on with logistic regression.
    This means that the model with all features are firstly fitted, then the feature
    with the lowest coefficient is removed. The processed is iterated until one feature
    is left (controlled by `n_features_to_selec`). Finally, this ranking of features is
    returned. The return element is an array where ranking[i] corresponds to feature
    number i's ranking, were ranking 1 is the most important ranking.

    With the CUB dataset, this process takes about 5 minutes.

    Args:
        x_train (np.array): The training features.
        y_train (np.array): The training labels
        max_iter (int, optional): Max number of iterations for logistic regression.
            The default of 100 is usally not sufficient. Defaults to 200.

    Returns:
        np.array: The rankings of features. ranking[i] corresponds to the ranking
            of feature number i, where ranking 1 is the most important feature.
    """
    model = LogisticRegression(solver="lbfgs", max_iter=max_iter, verbose=0)
    rfe = RFE(model, n_features_to_select=1, step=1, verbose=1)
    rfe.fit(x_train, y_train)

    return rfe.ranking_


def find_features_from_ranking(x_train, y_train, x_val, y_val, x_test=None, y_test=None,
                               ranking=None, n_attr=112, max_features=20, step=1):
    """
    Calculates the accuracy for different choices of feature selection.
    Given a ranking (if None, will use `get_rfe_ranking()`), where ranking[i]
    corresponds to features number i's ranking, where 1 is the most important
    ranking, this function calculates the accuracy for any feature subset up to
    `max_features`. Both train and validation (and optinally test) accuracies
    are calculated, and returned in a dictionary.

    Args:
        x_train (np.array): Training features.
        y_train (np.array): Training labels
        x_val (np.array): Validation features.
        y_val (np.array): Validation labels
        x_test (np.array, optional): Testing features. Defaults to None.
        y_test (np.array, optional): Testing labels. Defaults to None.
        ranking (np.array, optional): The rankings. If None, will be calculated
            with `get_rfe_ranking()`. Defaults to None.
        n_attr (int, optional): The amount of feautres. Defaults to 112.
        max_features (int, optional): The maxkimum amount of features to calculate
            accuracy on. Defaults to 20.
        step (int, optional): The step of features to calculate on. If 5, will
            calculate for 1, 6, 11, ... features.

    Returns:
        dict: Dictionary of information on accuracies. The keys are the amount
            of features. The values are dictionaris with accuracies.
    """
    if ranking is None:  # Calculate rankings
        ranking = get_rfe_ranking(x_train, y_train)

    feature_indices = np.arange(n_attr)
    main_dict = {"complete_ranking": ranking}  # Store full rankings

    for n_features in range(1, max_features):
        features = feature_indices[ranking <= n_features]  # Index features
        model = LogisticRegression(solver="lbfgs", max_iter=200)
        model.fit(x_train[:, features], y_train)

        # Predict and calculate accuracy
        train_preds = model.predict(x_train[:, features])
        train_accuracy = accuracy_score(y_train, train_preds)
        val_preds = model.predict(x_val[:, features])
        val_accuracy = accuracy_score(y_val, val_preds)

        info_dict = {"train_accuracy": train_accuracy,
                     "val_accuracy": val_accuracy}
        if x_test is not None and y_test is not None:
            test_preds = model.predict(x_test[:, features])
            test_accuracy = accuracy_score(y_test, test_preds)
            info_dict["test_accuracy"] = test_accuracy
        info_dict["n_features"] = n_features
        info_dict["features"] = features
        main_dict[n_features] = info_dict

    return main_dict


if __name__ == "__main__":
    # Get rankings, calculate accuracies for different features subset, and write to file.
    # Load data:
    subset = ""
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = read_processed_cub(subset=subset)
    ranking = get_rfe_ranking(x_train, y_train)  # Calculate feature rankings
    info_dict = find_features_from_ranking(x_train, y_train, x_val, y_val, x_test, y_test,
                                           ranking, max_features=30)
    path = "data/CUB_processed/"
    with open(path + "CUB_feature_selection.pkl", "wb") as outfile:
        pickle.dump(info_dict, outfile)

    # Benchmarks from above
    # Less than 10 features gives very poor accuracy.

    # 10 features gives 70 percent accuracy.
    # [2,  11,  22,  61,  74,  76,  78,  83, 100, 106]

    # 15 features gives 95 percent accuracy.
    # [2,  11,  22,  31,  51,  61,  74,  76,  78,  83,  86,  96,  97, 100, 106]

    # 20 features gives 98 percent accuracy.
    # [2,  11,  13,  22,  31,  38,  48,  51,  61,  74,  76,  78,  83,
    #  86,  89,  96,  97, 100, 106, 111]

    # 25 features gives 99 percent accuracy.
    # [2,  10,  11,  13,  22,  31,  38,  48,  51,  61,  65,  67,  74,
    #  76,  78,  82,  83,  86,  89,  96,  97, 100, 106, 108, 111]
