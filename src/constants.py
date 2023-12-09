MODEL_STRINGS_SHAPES = ["cnn", "cbm", "cbm_res", "cbm_skip", "scm"]
MODEL_STRINGS_CUB = ["cnn", "cbm", "cbm_res", "cbm_skip", "scm"]
SCM_ONLY = ["scm"]
MODEL_STRINGS_ORACLE = ["lr_oracle", "nn_oracle"]
CONCEPT_MODELS_STRINGS_SHAPES = ["cbm", "cbm_res", "cbm_skip", "scm"]
CONCEPT_MODELS_STRINGS_CUB = ["cbm", "cbm_res", "cbm_skip", "scm"]
MODEL_STRINGS_ALL_SHAPES = ["cnn", "cbm", "cbm_res", "cbm_skip", "scm", "lr_oracle"]
MODEL_STRINGS_ALL_CUB = ["cnn", "cbm", "cbm_res", "cbm_skip", "scm", "lr_oracle", "nn_oracle"]
COLORS_SHAPES = ["y", "r", "blueviolet", "black", "skyblue"]
COLORS_CUB = ["y", "r", "blueviolet", "black"]
COLORS_ORACLE = ["hotpink", "deeppink"]
COLORS_ALL_SHAPES = ["y", "r", "blueviolet", "black", "skyblue", "hotpink", "deeppink"]
COLORS_ALL_CUB = ["y", "r", "blueviolet", "black", "deeppink", "hotpink"]
MAX_EPOCHS = 50
FAST_MAX_EPOCHS_SHAPES = 3  # Use in order to quickly test if implementations work
FAST_MAX_EPOCHS_CUB = 1
N_EARLY_STOP_DEFAULT = 7
DEFAULT_HYPERPARAMETERS_FILENAME_SHAPES = "default.yaml"
FAST_HYPERPARAMETERS_FILENAME_SHAPES = "fast.yaml"
DEFAULT_HYPERPARAMETERS_FILENAME_SHAPES_HARD = "default_hard.yaml"
FAST_HYPERPARAMETERS_FILENAME_SHAPES_HARD = "fast_hard.yaml"

USE_XAVIER_INIT_IN_BOTTLENECK = False

# Folder paths. Should not need to be edited, unless you for some reason want to save results somewhere else.
# RESULTS_FOLDER gets added to the beginning of the other folders, and it can be an absolute path. It will
# also make all the other sub-folder absolute
RESULTS_FOLDER = "results/"
HYPERPARAMETERS_FOLDER = "hyperparameters/"
HISTORY_FOLDER = "history/remote_final1_history/"
SAVED_MODELS_FOLDER = "saved_models/"
PLOTS_FOLDER = "plots/"
ORACLE_FOLDER = "oracles/"

# DATA_FOLDER gets added to the beginning of the other folders, and it can be an absolute path. It will
# also make all the other sub-foldes absolute.
DATA_FOLDER = "data/"
SHAPES_FOLDER = "shapes/"
CUB_FOLDER = "cub/"
CUB_TABLES_FOLDER = "CUB_processed/class_attr_data_10"
CUB_PROCESSED_FOLDER = "CUB_processed/"
CUB_FEATURE_SELECTION_FILENAME = "CUB_feature_selection.pkl"

ADVERSARIAL_FOLDER = "adversarial"
ADVERSARIAL_FILENAME = "adversarial_image.png"
ADVERSARIAL_TEXT_FILENAME = "adversarial_summary.txt"

ATTRIBUTE_NAMES_PATH_CUB = "CUB_200_2011/attributes/attributes.txt"
ATTRIBUTE_MAPPING_PATH_CUB = "CUB_processed/attribute_mapping.txt"
ATTRIBUTE_NAMES_PATH_SHAPES = "shapes/attributes.txt"

CLASS_NAMES_PATH_CUB = "CUB_200_2011/classes.txt"
CLASS_NAMES_PATH_C10_SHAPES = "shapes/classes_c10.txt"
CLASS_NAMES_PATH_C15_SHAPES = "shapes/classes_c15.txt"
CLASS_NAMES_PATH_C21_SHAPES = "shapes/classes_c21.txt"


N_CLASSES_CUB = 200
N_ATTR_CUB = 112

NAMES_TO_SHORT_NAMES_SHAPES = {
    "ShapesCNN": "cnn",
    "ShapesCBM": "cbm",
    "ShapesCBMWithResidual": "cbm_res",
    "ShapesCBMWithSkip": "cbm_skip",
    "ShapesSCM": "scm",
    "ShapesLogisticOracle": "lr_oracle",
    "ShapesNNOracle": "nn_oracle"
}

NAMES_TO_SHORT_NAMES_CUB = {
    "CubCNN": "cnn",
    "CubCBM": "cbm",
    "CubCBMWithResidual": "cbm_res",
    "CubCBMWithSkip": "cbm_skip",
    "CubSCM": "scm",
    "CubLogisticOracle": "lr_oracle",
    "CubNNOracle": "nn_oracle",
    "ShapesLogisticOracle": "lr_oracle",
    "ShapesNNOracle": "nn_oracle"
}

MAP_MODEL_TO_COLOR_SHAPES = {
    "cnn": "y",
    "cbm": "yellowgreen",
    "cbm_res": "blueviolet",
    "cbm_skip": "indigo",
    "scm": "violet",
    "lr_oracle": "deeppink",
    "nn_oracle": "hotpink"
}

MAP_MODEL_TO_LINESTYLES_SHAPES = {
    "cnn": "-",
    "cbm": ":",
    "cbm_res": "-",
    "cbm_skip": "-.",
    "scm": ":",
    "lr_oracle": "-",
    "nn_oracle": ":"
}

MAP_NAMES_TO_DOCUMENT_NAMES = {
    "cnn": "Standard Model",
    "cbm": "Vanilla CBM",
    "cbm_res": "CBM-Res",
    "cbm_skip": "CBM-Skip",
    "scm": "SCM",
    "lr_oracle": "Oracle"
}
