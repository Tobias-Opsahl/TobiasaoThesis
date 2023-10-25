import os


def make_initial_folders():
    os.makedirs("data/shapes/", exist_ok=True)
    os.makedirs("data/CUB/", exist_ok=True)
    os.makedirs("results/hyperparameters/", exist_ok=True)
    os.makedirs("results/history/", exist_ok=True)
    os.makedirs("results/saved_models/", exist_ok=True)
    os.makedirs("results/plots/", exist_ok=True)


if __name__ == "__main__":
    make_initial_folders()
