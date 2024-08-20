# BrainAlignNet
BrainAlignNet is a deep neural network that registers neurons in the deforming head of freely-moving C. elegans. This repository contains the source code for data preprocessing, as well as network training and testing.

## citation
To cite this work, please refer to our preprint:

**Deep Neural Networks to Register and Annotate the Cells of the C. elegans Nervous System**

Adam A. Atanas, Alicia Kun-Yang Lu, Jungsoo Kim, Saba Baskoylu, Di Kang, Talya S. Kramer, Eric Bueno, Flossie K. Wan, Steven W. Flavell

bioRxiv 2024.07.18.601886; doi: https://doi.org/10.1101/2024.07.18.601886

## table of contents
- [installation](#installation)
- [data preparation](#preparation)
- [usage](#usage)

## installation
BrainAlignNet runs on two other packages: `DeepReg` and `euler_gpu`, which need to be installed separately.

### `DeepReg`

`DeepReg` is a deep learning toolkit for image registration. BrainAlignNet uses a [custom version of `DeepReg`](https://github.com/flavell-lab/DeepReg) with a novel network objective.

Clone or download our custom DeepReg; then run `git install .` at its root directory to install the package.

### `euler_gpu`

`euler_gpu` is a GPU-accelerated implementation of Euler registration using pytorch.

Clone or download [`euler_gpu`](https://github.com/flavell-lab/euler_gpu) and run `git install .` at the root directory.

## data preparation


## usage

