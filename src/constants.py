MODEL_STRINGS_SHAPES = ["cnn", "cbm", "cbm_res", "cbm_skip", "scm"]
MODEL_STRINGS_CUB = ["cnn", "cbm", "cbm_res", "cbm_skip"]
MODEL_STRINGS_ORACLE = ["lr_oracle", "nn_oracle"]
MODEL_STRINGS_ALL_SHAPES = ["cnn", "cbm", "cbm_res", "cbm_skip", "scm", "lr_oracle", "nn_oracle"]
MODEL_STRINGS_ALL_CUB = ["cnn", "cbm", "cbm_res", "cbm_skip", "lr_oracle", "nn_oracle"]
COLORS_SHAPES = ["y", "r", "blueviolet", "black", "skyblue"]
COLORS_CUB = ["y", "r", "blueviolet", "black"]
COLORS_ORACLE = ["hotpink", "deeppink"]
COLORS_ALL_SHAPES = ["y", "r", "blueviolet", "black", "skyblue", "hotpink", "deeppink"]
COLORS_ALL_CUB= ["y", "r", "blueviolet", "black", "deeppink", "hotpink"]
MAX_EPOCHS = 50
FAST_MAX_EPOCHS_SHAPES = 2  # Use in order to quickly test if implementations work
FAST_MAX_EPOCHS_CUB = 1
N_EARLY_STOP_DEFAULT = 7
DEFAULT_HYPERPARAMETERS_FILENAME_SHAPES = "default.yaml"
FAST_HYPERPARAMETERS_FILENAME_SHAPES = "fast.yaml"
DEFAULT_HYPERPARAMETERS_FILENAME_SHAPES_HARD = "default_hard.yaml"
FAST_HYPERPARAMETERS_FILENAME_SHAPES_HARD = "fast_hard.yaml"

BOOTSTRAP_CHECKPOINTS = [1, 3, 5, 10]
USE_XAVIER_INIT_IN_BOTTLENECK = False

# Folder paths. Should not need to be edited, unless you for some reason want to save results somewhere else.
# RESULTS_FOLDER gets added to the beginning of the other folders, and it can be an absolute path. It will
# also make all the other sub-folder absolute
RESULTS_FOLDER = "results/"
HYPERPARAMETERS_FOLDER = "hyperparameters/"
HISTORY_FOLDER = "history/"
SAVED_MODELS_FOLDER = "saved_models/"
PLOTS_FOLDER = "plots/"

# DATA_FOLDER gets added to the beginning of the other folders, and it can be an absolute path. It will
# also make all the other sub-foldes absolute.
DATA_FOLDER = "data/"
SHAPES_FOLDER = "shapes/"
CUB_FOLDER = "cub/"
CUB_TABLES_FOLDER = "CUB_processed/class_attr_data_10"
CUB_PROCESSED_FOLDER = "CUB_processed/"
CUB_FEATURE_SELECTION_FILENAME = "CUB_feature_selection.pkl"

N_CLASSES_CUB = 200
N_ATTR_CUB = 112
