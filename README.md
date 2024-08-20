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

### DeepReg

`DeepReg` is a deep learning toolkit for image registration. BrainAlignNet uses a [custom version of `DeepReg`](https://github.com/flavell-lab/DeepReg) with a novel network objective.

Clone or download our custom DeepReg; then run `git install .` at its root directory to install the package.

### euler_gpu

`euler_gpu` is a GPU-accelerated implementation of Euler registration using pytorch.

Clone or download [`euler_gpu`](https://github.com/flavell-lab/euler_gpu) and run `git install .` at the root directory.

## data preparation

The inputs to BrainAlignNet are images with their centroid labels. Each registration problem in the training and validation set is composed of six items:
* `fixed_image` & `moving_image`
* `fixed_roi` & `moving_roi`
* `fixed_label` & `moving_label`

These datasets for all registration problems are written in `*.h5` files. Each `.h5` file contains multiple keys. Each key, formatted as `<t_moving>to<t_fixed>`, represents a registration problem.

During training, BrainAlignNet is tasked with optimally registering the `t_moving` frame to the `t_fixed` frame, where `t_moving` and `t_fixed` are two different timepoints from a single calcium imaging recording.

The ROI images, `fixed_roi` and `moving_roi`, display each neuron on the RFP images with a unique color. Each label is a list of centriods of these neuronal R0Is.

To prepare data for training (and validation), preprocessing needs to accomplish the following tasks
* crop all RFP and ROI images to the same size: (284, 120, 64)
*


## usage

