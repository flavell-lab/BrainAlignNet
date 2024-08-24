# BrainAlignNet
BrainAlignNet is a deep neural network that registers neurons in the deforming head of freely-moving C. elegans. This repository contains the source code for data preprocessing, as well as network training and testing.

## citation
To cite this work, please refer to our preprint:

**Deep Neural Networks to Register and Annotate the Cells of the C. elegans Nervous System**

Adam A. Atanas, Alicia Kun-Yang Lu, Jungsoo Kim, Saba Baskoylu, Di Kang, Talya S. Kramer, Eric Bueno, Flossie K. Wan, Steven W. Flavell

bioRxiv 2024.07.18.601886; doi: https://doi.org/10.1101/2024.07.18.601886

## table of contents
- [installation](#installation)
- [data preparation](#data-preparation)
    - [cropping & Euler registration](#cropping--euler-registration)
    - [create centroids](#create-centroids)
- [usage](#usage)

## installation
BrainAlignNet runs on two other packages: `DeepReg` and `euler_gpu`, which need to be installed separately.

### DeepReg

`DeepReg` is a deep learning toolkit for image registration. BrainAlignNet uses a [custom version of `DeepReg`](https://github.com/flavell-lab/DeepReg) with a novel network objective.

Clone or download our custom DeepReg; then run `pip install .` at its root directory to install the package.

### euler_gpu

`euler_gpu` is a GPU-accelerated implementation of Euler registration using pytorch.

Clone or download [`euler_gpu`](https://github.com/flavell-lab/euler_gpu) and run `pip install .` at the root directory.

## data preparation

*For a demonstration of our data preprocessing pipeline, check out our [demo notebook](https://github.com/flavell-lab/BrainAlignNet/blob/main/demo_notebook/demo_pipeline.ipynb).*


The inputs to BrainAlignNet are images with their centroid labels. Each registration problem in the training and validation set is composed of six items:
* `fixed_image` & `moving_image`
* `fixed_roi` & `moving_roi`
* `fixed_label` & `moving_label`

These datasets for all registration problems are written in `.h5` files. Each `.h5` file contains multiple keys. Each key, formatted as `<t_moving>to<t_fixed>`, represents a registration problem.

During training, BrainAlignNet is tasked with optimally registering the `t_moving` frame to the `t_fixed` frame, where `t_moving` and `t_fixed` are two different timepoints from a single calcium imaging recording.

The ROI images, `fixed_roi` and `moving_roi`, display each neuron on the RFP images with a unique color. Each label is a list of centriods of these neuronal R0Is.

To prepare the data for training and validation, the preprocessing steps should accomplish the following tasks:

* **crop images:** crop all RFP and ROI images to the same size: (284, 120, 64)
* **Euler registration:** perform Euler registration on both RFP and ROI images, using parameters that optimize the registration of RFP images.
* **create centroids:** identify and extract the centroids of all neurons from the ROI images.

#### cropping & Euler registration
Both image cropping and Euler registration are performed on our raw data in `.nrrd` format, which is available upon request. The processed images for training and validaton are freely and publicly available on [DropBox](https://www.dropbox.com/scl/fo/ealblchspq427pfmhtg7h/ANRojNDjEY018KFywtEZ8-k/BrainAlignNet?dl=0&rlkey=1e6tseyuwd04rbj7wmn2n6ij7&subfolder_nav_tracking=1).

The following code block processes RFP images and creates a  `resources` folder, where it writes two `.json` files: `center_of_mass.json` and `euler_parameters.json`. These files store the parameters for cropping and registering the ROI images.

Additionally, the code outputs the processed RFP images from registration problems specified in `problem_file`. The outputs are `fixed_images.h5` and `moving_images.h5`, which are saved under the specified `save_directory`.

```python
from euler_register import EulerRegistrationProcessor

target_image_shape = (284, 120, 64)
save_directory = "/home/user/demo_data/euler_registered_RFP"
problem_file = "/home/user/demo_data/registration_problems.json"

processor = EulerRegistrationProcessor(
    target_image_shape,
    save_directory,
    problem_file
)
processor.process_datasets()
```
Then, the same Euler parameters for registering RFP images are applied to register their corresponding ROI images. The outputs are `fixed_rois.h5` and `moving_rois.h5` under the specified `save_directory`.

```python
from warp_roi import generate_rois

device_name = "cuda:2"
target_image_shape = (284, 120, 64)
problem_file = "/home/user/BrainAlignNet/demo_data/registration_problems_roi.json"
save_directory = "/home/user//BrainAlignNet/demo_data/euler_registered_roi"

generate_rois(
    device_name,
    target_image_shape,
    problem_file,
    save_directory,
    True)
```
#### create centroids

The neuronal centroids are computed after `fixed_rois.h5` and `moving_rois.h5` are created. To compute them, simply specify the path to the ROI images.

```python
from label_centroids import CentroidLabel

dataset_path = "/home/user/BrainAlignNet/demo_data/euler_registered_roi"
centroid_labeler = CentroidLabel(dataset_path)
centroid_labeler.create_all_labels()
```

## usage

*A demonstration of training and applying BrainAlignNet on unseen data is available [here](https://github.com/flavell-lab/BrainAlignNet/blob/main/demo_notebook/demo_network.ipynb)*.
