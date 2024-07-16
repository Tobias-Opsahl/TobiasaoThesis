from src.datasets.datasets_cub import make_correct_paths, make_small_test_set
from src.common.utils import get_logger, set_global_log_level

logger = get_logger(__name__)


def initialize_cub():
    make_correct_paths()
    make_small_test_set()


if __name__ == "__main__":
    set_global_log_level("info")
    initialize_cub()
