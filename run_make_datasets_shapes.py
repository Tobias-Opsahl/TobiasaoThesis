from src.datasets.make_shapes_datasets import make_shapes_1k_c10_a9_s50
from src.common.utils import get_logger, set_global_log_level

if __name__ == "__main__":
    set_global_log_level("debug")
    logger = get_logger(__name__)
    make_shapes_1k_c10_a9_s50()