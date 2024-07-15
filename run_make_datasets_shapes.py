from src.common.utils import get_logger, set_global_log_level
from src.datasets.make_shapes_datasets import (make_shapes_1k_c10_a9_s60, make_shapes_1k_c10_a9_s70,
                                               make_shapes_1k_c10_a9_s80, make_shapes_1k_c10_a9_s90)

if __name__ == "__main__":
    set_global_log_level("debug")
    logger = get_logger(__name__)
    make_shapes_1k_c10_a9_s60()
    make_shapes_1k_c10_a9_s70()
    make_shapes_1k_c10_a9_s80()
    make_shapes_1k_c10_a9_s90()
