from julia.api import Julia
from scipy import ndimage
from typing import Any, Dict, Tuple
import SimpleITK as sitk
import argparse
import glob
import json
import numpy as np
import os

jl = Julia(compiled_modules=False)
jl.eval('include("adjust.jl")')
ADJUST_IMAGE_SIZE = jl.eval("adjust_image_cm")

def write_to_json(
    input_: Dict[str, Any], 
    output_file: str,
    folder: str = "resources"
):
    """ Write a dictionary to a JSON file.

    Args:
        input_ (Dict[str, Any]): The dictionary to be written to the JSON file.
        output_file (str): The name of the JSON file to be created (without extension).
        folder (str, optional): The folder where the JSON file will be saved.
            Default is "resources". """

    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.float32):
                return float(obj)
            return json.JSONEncoder.default(self, obj)

    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(f"{folder}/{output_file}.json", "w") as f:
        json.dump(input_, f, indent=4, cls=CustomEncoder)

    print(f"{output_file} written under {folder}.")


def locate_dataset(dataset_name: str):

    """ Given the name of the dataset, this function locates the directory
    where this data file can be found.

    Args:
        dataset_name (str): The name of the dataset; e.g., `2022-03-16-02`.

    Returns:
        str: The full path to the dataset directory.

    Raises:
        FileNotFoundError: If the dataset cannot be found in any of the
            specified directories. """

    swf360_path = "/data3/adam/SWF360_test_datasets"
    if dataset_name == "2022-01-06-01":
        return \
        f"{swf360_path}/{dataset_name}-SWF360-animal1-610LP_newunet_output"
    elif dataset_name == "2022-03-30-01":
        return f"{swf360_path}/{dataset_name}-SWF360-animal1-610LP_output"
    elif dataset_name == "2022-03-30-02":
        return \
        f"{swf360_path}/{dataset_name}-SWF360-animal2-610LP_diffnorm_ckpt287"
    elif dataset_name == "2022-03-31-01":
        return f"{swf360_path}/{dataset_name}-SWF360-animal1-610LP_output"

    neuropal_dir = "/store1/prj_neuropal/data_processed"
    kfc_dir = "/store1/prj_kfc/data_processed"
    rim_dir = "/store1/prj_rim/data_processed"

    dir_dataset_dict = {
        neuropal_dir: os.listdir(neuropal_dir),
        kfc_dir: os.listdir(kfc_dir),
        rim_dir: os.listdir(rim_dir)
    }

    for base_dir, dataset_dirs in dir_dataset_dict.items():

        if any(dataset_name in dataset_dir for dataset_dir in dataset_dirs):
            dataset_path = glob.glob(f"{base_dir}/{dataset_name}_*")
            assert len(dataset_path) == 1, \
                f"More than one path for {dataset_name} found: {dataset_path}"
            return dataset_path[0]

    raise FileNotFoundError(
        f'Dataset {dataset_name} not found in any specified directories.')


def filter_and_crop(
    image_T: np.ndarray,
    image_median: float,
    target_image_shape: Tuple[int, int, int]
) -> np.ndarray:
    """ Subtract the median pixel value regarded as the image background from
    the image and resize it to the target shape.

    Args:
        image_T (np.ndarray): The image of shape (x_dim, y_dim, z_dim).
        image_median (float): The median pixel value of the image.
        target_image_shape (Tuple[int, int, int]): The target image dimensions;
            given in the order (x_dim, y_dim, z_dim).

    Returns:
        np.ndarray: The image of the target shape with background
            subtracted. """

    filtered_image_CM = get_image_CM(image_T)
    filtered_image_T = filter_image(image_T, image_median)

    return get_cropped_image(
                filtered_image_T,
                filtered_image_CM,
                target_image_shape, -1).astype(np.float32)


def filter_image(
    image: np.ndarray,
    threshold: float
) -> np.ndarray:
    """ Subtract the threshold value from each image pixel and set the
    resultant negative pixels to zero.

    Args:
        image (np.ndarray): The input image of shape (x_dim, y_dim, z_dim).
        threshold (float): The value to be subtracted from each image pixel.

    Returns:
        np.ndarray: The thresholded image. """

    filtered_image = image - threshold
    filtered_image[filtered_image < 0] = 0

    return filtered_image


def get_cropped_image(
        image_T: np.ndarray,
        center: Tuple[int, int, int],
        target_image_shape: Tuple[int, int, int],
        projection: int
) -> np.ndarray:
    """ Resize image to the target image shape and optionally project along the
    maximum value of the given axis.

    Args:
        image_T (np.ndarray): The input image of shape (x_dim, y_dim, z_dim).
        center (Tuple[int, int, int]): The center of mass of the image.
        target_image_shape (Tuple[int, int, int]): The target image dimensions;
            given in the order (x_dim, y_dim, z_dim).
        projection (int): The axis to perform maximum projection; options are
            0, 1, 2. If no projection is needed, set to -1.

    Returns:
        np.ndarray: The resized (and possibly projected) image. """

    if projection in [0, 1, 2]:
        return ADJUST_IMAGE_SIZE(
                image_T,
                center,
                target_image_shape).max(projection)
    elif projection == -1:
        return ADJUST_IMAGE_SIZE(
                image_T,
                center,
                target_image_shape)
    else:
        raise Exception(f"projection is not 0, 1, 2, but {projection}")


def get_image_T(image_path: str) -> np.ndarray: 
    """ Read a .NRRD image from the given path and return it as a transposed
    numpy array.

    Args:
        image_path (str): The path of the image file.

    Returns:
        np.ndarray: The image as a numpy array, transposed to the shape (x_dim,
            y_dim, z_dim). """

    image_nrrd = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(image_nrrd)
    if image.ndim == 4:
        image = image.squeeze()
    image_T = np.transpose(image, (2,1,0))

    return image_T


def get_image_CM(image_T: np.ndarray) -> Tuple[int, int, int]:
    """ Find the center of mass of an image.

    Args:
        image_T (np.ndarray): The image array.

    Returns:
        Tuple[int, int, int]: The center of mass of the image. """

    # subtract the median pixel from the image; zero out the negative pixels
    image_T_wo_background = image_T - np.median(image_T)
    image_T_wo_background[image_T_wo_background < 0] = 0
    x, y, z = ndimage.center_of_mass(image_T_wo_background)

    return (round(x), round(y), round(z))


def extract_all_problems(
    dataset_name: str,
    problem_file_path: str
) -> None:
    """ Read all registration problems from a text file and write them to a
    JSON file with formatting `<MOVING>to<FIXED>`.

    Args:
        dataset_name (str): The name of the dataset.
        problem_file_path (str): The path to the directory containing the
            registration problems text file.

    Raises:
        FileNotFoundError: If the registration problems text file does not
            exist. """

    if os.path.exists(f"{problem_file_path}/registration_problems.txt"):
        lines = open(
            f"{problem_file_path}/registration_problems.txt", "r").readlines()
        problems = [line.strip().replace(" ", "to") for line in lines]
    else:
        raise FileNotFoundError(
            f"Can't find {dataset_path}/registration_problems.txt")

    write_to_json(
        {"train": {dataset_name: problems}},
        "registration_problems"
    )

