from src.plotting import plot_test_accuracies_pdf


def make_pdf_test_accuracies_shapes():
    """
    Plots the test-accuracies for shapes a5 and a9 in pdf format and saves.
    """
    n_bootstrap = 10
    subsets = [50, 100, 150, 200, 250]
    for n_attr in [5, 9]:
        for n_classes in [10, 15, 21]:
            save_name = "c" + str(n_classes) + "_a" + str(n_attr) + "_b" + str(n_bootstrap) + ".pdf"
            plot_test_accuracies_pdf(n_classes=n_classes, n_attr=n_attr, subsets=subsets, n_bootstrap=n_bootstrap,
                                     history_folder="history/remote_grid_search/",
                                     save_dir="plots/pdf_testing_accuracies/", save_name=save_name)


if __name__ == "__main__":
    make_pdf_test_accuracies_shapes()
