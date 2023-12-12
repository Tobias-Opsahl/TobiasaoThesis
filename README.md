# Hybrid Concept-based Models

This repo is used for the master thesis `Achieving Data-efficient Neural Networks with Hybrid Concept-based Models`.

## Overview of work and contents

Here is a brief overview of the work, motivation and content this repo was used for. For a more detailed explanation, please look at the thesis. Afterwards, we describe the code and how to run stuff.

We defined a `concept` as a human-meaningful feature in the dataset that is different from the target label. For instance, if the downstream task is to use images of birds to predict the bird species, a concept might be the color of the birds legs, the size of the bird or the length of the beak.

We work with datasets that have concept-labels in addition to target labels, and models that utilise the concept-labels during training. Previous work on concept-based models were motivated by interpretability, but since recent experiments question their interpretable qualities, we design concept-based models with the intent of improving performance. Our models outperform both existing concept-based models and standard CNNs on several datasets.

The main concept-based architecture we base our work on is the `Concept Bottleneck Models` ([CBM](https://proceedings.mlr.press/v119/koh20a.html)).

There are three main contributions:

- **Novel hybrid concept-based architectures:** Two new model architectures are proposed. They train with both concept-labels and class labels. They differ from existing concept-based models in that they are motivated by performance, not interpretability. In order to make target predictions, they use both concept predictions, and information not interfering with concepts.
- **ConceptShapes datasets:** We argue that existing concept datasets are insufficient to properly benchmark model performance, and propose a new set of flexible synthetic concept datasets called `ConceptShapes`. The amount of classes, concepts and correlation between concepts and classes can be adjusted. The code for generating the datasets are in this repo, and an ordinary laptop with only CPU support is sufficient.
- **Adversarial concept attacks:** We design an algorithm that creates adversarial examples such that the perturbed images look identical to the original ones, the concept predictions for a concept-based model are identical, but the class predictions are different. Since the concept-predictions are used as interpretations for concept-based models, we suggest that this further questions the interpretable qualities of concept-based models. If two identical interpretations produce different predictions, how can the interpretation help understand the models behaviour?

## Creating ConceptShapes datasets

Run `python run_make_datasets.py` to create a dataset. The main function for creating the dataset, `generate_shapes_dataset()`, can be called with whatever arguments one like. Hardcoded functions for creating the datasets used in the thesis are available, simply call them to create the datasets.

## Code structure

The source code can be found in `src/`, the results (hyperparameters, histories, plots and models) will be saved in `results/`, and the datasets should be saved in `data/`. The files for running everything, `run_shapes.py` and `run_cub.py` are saved at root level.

The code should be well documented and formatted. My formatter went deprecated during the last weeks of submissions, so a few slight deviations will occur.

The code uses the word `attributes` as `concepts` are used in the literature. The `ConceptShapes` datasets are called `Shapes`.

## Running code

To run the code, either call `run_shapes.py` for the ConceptShapes datasets or `run_cub.py` for the [CUB dataset](https://authors.library.caltech.edu/records/cvm3y-5hh21). When running ConceptShapes, see the subsection below on how to specify the arguments for the datasets.

Use `--evaluate_and_plot` to train models, evaluate them and plot results. This will run all the hybrid concept-models, the vanilla CBM and the standard CNN. They will be run at the subsets sizes specified with `--subsets` (for example `--subsets 50,100,150,200,250` for ConceptShapes and `--subsets 5,10,15,20,25,30` for CUB).

Use `--only_plot` to only plot the results, if results for the same hyperparameters are already run. By running `--evaluate_and_plot`, all the training and evaluation runs will be pickled and saved in `results/history/`, so one does not need to train again.

Use `--run_hyperparameters` to run hyperparameter searches. The hyperparameters found in the thesis are saved here, so there is no need to run them again. The ranges for hyperparameters are hard-coded in `src/hyperparameter_optimization.py` for ConceptShapes and `src/hyperparameter_optimization_cub.py` for CUB.

Use `--run_adversarial_attacks` to run adversarial attacks. Use `--adversarial_grid_search` to run hyperparameter search on the adversarial hyperparameters.

There are many more hyperparameters to choose from, please use `--help` or read the code. The scripts used to produce the results in the thesis are available in `scripts.txt`.

### Setup

### ConceptShapes

In order to run stuff from the ConceptShapes datasets, call `python run_shapes.py`. Specify the number of classes, concepts and correlation between concepts and classes with `--n_classes`, `--n_attr` and `--signal_strength`. To run on 10 classes, 9 concepts and signal strength 98, run `python run_shapes.py --n_classes 10 --n_attr 9 --signal_strenght 98`.

The datasets must be created in advance. That means, you should already have called `make_shapes_1k_c10_a9_s98()` to run the code above.

### CUB

### Adversarial concept attacks

The work we do on adversarial concept attacks differs from [this paper](https://ojs.aaai.org/index.php/AAAI/article/view/26765), which constructs adversarial attack algorithm that changes the concept predictions, but keeps the class prediction the same. We perform attacks where the concept predictions are identical, but the class predictions differ. We construct our algorithm to challenge the belief of interpretability in CBMs.

<!-- The repo works a lot with the Caltech-UCSD Birds-200-2011 (CUB) dataset [link](https://authors.library.caltech.edu/27452/1/CUB_200_2011.pdf). In order to get the code working, the user should download this dataset, in addition to the processed version from the CBM paper. The files can be found [here](https://worksheets.codalab.org/worksheets/0x362911581fcd4e048ddfd84f47203fd2) (download `CUB_200_2011` and `CUB_processed`). It is recommended to save these two folders inside a folder called `data/`. -->

<!-- The processed dataset are pickled lists of dictionaries, which among other things contains the path to the images. The paths provided by the CBM paper are specific to their computer, but this can be easily adjusted by running:

```terminal
python initialize.py make_paths --base_path `your_path`
``` -->

![no gif :(](https://media.giphy.com/media/mcsPU3SkKrYDdW3aAU/giphy.gif)
