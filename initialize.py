import argparse
from utils import make_correct_paths, make_subset_cub


def parse_arguments():
    parser = argparse.ArgumentParser(description="Perform actions.")

    help = "Action to perform. Choices: [make_paths, make_dataset]. `make_paths` "
    help += "replaces the img_paths in the dataset with the correct path. Provide "
    help += "`base_path` where the CUB dataset is stored for you locally. \n"
    help += "`make_dataset` makes a smaller dataset one can use for testing. Provide "
    help += "`n_classes`, the amount of the 200 classes to be used in the subset."
    parser.add_argument("action", type=str, choices=["make_paths", "make_dataset"], help=help)

    parser.add_argument("--base_path", type=str, default="data/",
                        help="Base path for `make_paths`. Defaults to 'data/'")

    parser.add_argument("--n_classes", type=int, required=False,
                        help="Number of classes for `make_dataset`.")

    random_help = "Optional boolean flag for `make_dataset`. Determines wether the "
    random_help += "classes are chosen randomly or chronologically."
    parser.add_argument("--random_choice", type=bool, default=False, help=random_help)

    args = parser.parse_args()

    if args.action == "make_dataset" and args.n_classes is None:
        parser.error("The --n_classes argument is required when action is 'make_dataset'.")

    return args


if __name__ == "__main__":
    # Run stuff marked in the CLI
    args = parse_arguments()
    if args.action == "make_paths":  # Corrects the paths in the data dictionaries
        make_correct_paths(args.base_path)
    elif args.action == "make_dataset":  # Makes subsets of data (only dictionaries)
        make_subset_cub(n_classes=args.n_classes, random_choice=args.random_choice)
