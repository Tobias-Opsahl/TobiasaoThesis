# TobiasaoThesis

Working repo for my master thesis.

I currently work with training neural networks with concept, mainly based on these two papers, [TCAV](https://arxiv.org/abs/1711.11279) and [CBM](https://arxiv.org/abs/2007.04612).

More high level documentation will come here after a while, but the code is documented as I code.

## CUB dataset

The repo works a lot with the Caltech-UCSD Birds-200-2011 (CUB) dataset [link](https://authors.library.caltech.edu/27452/1/CUB_200_2011.pdf). In order to get the code working, the user should download this dataset, in addition to the processed version from the CBM paper. The files can be found [here](https://worksheets.codalab.org/worksheets/0x362911581fcd4e048ddfd84f47203fd2) (download `CUB_200_2011` and `CUB_processed`). It is recommended to save these two folders inside a folder called `data/`.

The processed dataset are pickled lists of dictionaries, which among other things contains the path to the images. The paths provided by the CBM paper are specific to their computer, but this can be easily adjusted by running:

```terminal
python initialize.py make_paths --base_path `your_path`
```

If one wishes to make datasets that are smaller, so that one can do rapid testing with less than the 200 classes, one can run:

```terminal
ptyhon initialize.py make_dataset --n_classes 10 --random_choice True
````

This creates lists of dictionaries of only `n_classes` of the 200 classes.

![no gif :(](https://media.giphy.com/media/mcsPU3SkKrYDdW3aAU/giphy.gif)
