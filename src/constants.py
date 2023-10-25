MODEL_STRINGS = ["cnn", "cbm", "cbm_res", "cbm_skip", "scm"]
COLORS = ["y", "r", "blueviolet", "black", "skyblue"]
MAX_EPOCHS = 50
FAST_MAX_EPOCHS = 2  # Use in order to quickly test if implementations work
DEFAULT_HYPERPARAMETERS_FILENAME_SHAPES = "default.yaml"
FAST_HYPERPARAMETERS_FILENAME_SHAPES = "fast.yaml"

# Folder paths. Should not need to be edited, unless you for some reason want to save results somewhere else.
# RESULTS_FOLDER gets added to the beginning of the other folders, and it can be an absolute path. It will
# also make all the other sub-folder absolute
RESULTS_FOLDER = "results/"
HYPERPARAMETERS_FOLDER = "hyperparameters/"
HISTORY_FOLDER = "history/"
SAVED_MODELS_FOLDER = "saved_models/"
PLOTS_FOLDER = "plots/"

# Data paths. Does not need to be edited, if datasets are put in "data/shapes/", "data/cub/", etc.
# DATA_FOLDER gets added to the beginning of the other folders, and it can be an absolute path. It will
# also make all the other sub-folder absolute.
DATA_FOLDER = "data/"
SHAPES_FOLDER = "shapes/"
CUB_FOLDER = "CUB/"
