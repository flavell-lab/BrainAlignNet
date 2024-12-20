from data_utils import (locate_dataset, get_cropped_image, get_image_T)
from euler_gpu.preprocess import initialize
from euler_gpu.transform import transform_image_3d
from typing import Dict, List, Optional, Tuple, Union, Any
import deepreg.model.layer as layer
import json
import nibabel as nib
import numpy as np
import tensorflow as tf
import torch


class ImageWarper:

    def __init__(
        self,
        ddf_directory: str,
        dataset_name: str,
        registration_problem: str,
        image_shape: Tuple[int, int, int],
        problem_file: str,
        device_name: str = "cuda:2",
        simply_crop: bool = False,
    ):
        """ Initialize the ImageWarper class.

        Args:
            ddf_directory (str): Directory where the displacement fields are stored.
            dataset_name (str): The name of the dataset.
            registration_problem (str): The registration problem identifier.
            image_shape (Tuple[int, int, int]): The shape of the images (x_dim, y_dim, z_dim).
            problem_file (str): The path to the problem file.
            device_name (str, optional): The name of the device to use for processing.
                Default is "cuda:2".
            simply_crop (bool, optional): Flag to indicate whether to simply crop the images.
                Default is False. """

        self.ddf_directory = ddf_directory
        self.dataset_name = dataset_name
        self._registration_problem = registration_problem
        self.image_shape = image_shape
        self.device = torch.device(device_name)
        tag = problem_file.split(".")[0].split("_")[-1]
        if simply_crop:
            self.CM_dict = self._load_json(
                    f"resources/center_of_mass_{tag}.json")
        else:
            self.CM_dict = self._load_json(
                    f"resources/center_of_mass_{tag}.json")
            self.euler_parameters_dict = self._load_json(
                    f"resources/euler_parameters_{tag}.json")

        # if `dataset_name` and `registration_problem` are both provided
        # update `problem_id` and the corresponding pair number
        if dataset_name and registration_problem:
            self._update_problem()

    def _load_json(self, file_path: str):
        """Load JSON resources"""
        with open(file_path, "r") as f:
            return json.load(f)

    @property
    def registration_problem(self):
        return self._registration_problem

    @registration_problem.setter
    def registration_problem(self, value: str):
        self._registration_problem = value
        self._update_problem()

    def _update_problem(self):
        self.problem_id = f"{self.dataset_name}/{self._registration_problem}"

    def get_deepreg_outputs(
        self,
        target: str = "all"
    ) -> Union[dict, np.ndarray, None]:
        """ Get the DDF warped moving images, DDF, original fixed and moving
        images from the DeepReg packge outputs.

        Args:
            target (str, optional): The item to retrieve; options are `all`,
                `ddf`, `warped_moving_image`, `fixed_image`, `moving_image`.
                Default is `all`.

        Returns:
            Union[dict, np.ndarray, None]: The requested item(s) from the network outputs.
                If `target` is `all`, returns a dictionary containing all items.
                If `target` is one of the specific items, returns the corresponding numpy array.
                If `target` is invalid, returns None. """

        network_output_path = self._get_problem_path()
        network_outputs = self._load_images(network_output_path)

        if target == "all":
            return network_outputs
        else:
            return network_outputs[target]

    def get_image_roi(self) -> Dict[str, Any]:
        """ Get image ROI.

        Returns:
            Dict[str, Any]: A dictionary containing the ROIs. If the dictionary is not empty,
                            it will also contain the warped moving image ROI. """

        roi_dict = self._preprocess_image_roi()
        if len(roi_dict) > 0:
            roi_dict["warped_moving_image_roi"] = self._warp_moving_image_roi(
                    roi_dict["euler_tfmed_moving_image_roi"]
            )
        return roi_dict

    def _warp_moving_image_roi(
        self,
        moving_image_roi: np.ndarray,
        input_ddf: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """ Warp the moving image ROI with DDF.

        Args:
            moving_image_roi (np.ndarray): The region of interest of the moving image to be warped.
            input_ddf (Optional[np.ndarray], optional): The displacement field to use for warping.
                If not provided, it will be retrieved using `get_deepreg_outputs`. Defaults to None.

        Returns:
            np.ndarray: The warped region of interest. """

        moving_image_roi_tf = tf.cast(
                tf.expand_dims(moving_image_roi, axis=0),
                dtype=tf.float32
        )
        # TODO: set `batch_size` required by the latest centroid_label network
        warping = layer.Warping(fixed_image_size = self.image_shape,
                interpolation = "nearest")
        if intput_ddf == None:
            ddf = self.get_deepreg_outputs("ddf")
        else:
            ddf = input_ddf
        warped_moving_image_roi_tf = warping(
            inputs = [ddf, moving_image_roi_tf])

        return warped_moving_image_roi_tf.numpy()[0]

    def _get_problem_path(self):
        return f"{self.ddf_directory}/test/pair_{self.pair_num}"

    def _load_images(
        self,
        network_output_path: str
    ) -> Dict[str, np.ndarray]:
        """ Load images and displacement fields (DDF) from the specified network output path.

        Args:
            network_output_path (str): The path to the directory containing the network output
                files.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the DDF, warped moving image, fixed
                image, and moving image arrays. """

        ddf_nii_path = f"{network_output_path}/ddf.nii.gz"
        ddf_array = nib.load(ddf_nii_path).get_fdata()
        warped_moving_image = nib.load(
                f"{network_output_path}/pred_fixed_image.nii.gz").get_fdata()
        fixed_image = nib.load(
                f"{network_output_path}/fixed_image.nii.gz").get_fdata()
        moving_image = nib.load(
                f"{network_output_path}/moving_image.nii.gz").get_fdata()

        return {
                "ddf": ddf_array,
                "warped_moving_image": warped_moving_image,
                "fixed_image": fixed_image,
                "moving_image": moving_image
        }

    def _preprocess_image_roi(self) -> Dict[str, np.ndarray]:
        """ Resize the ROI images and Euler-transform them with the same parameters
        as their corresponding RFP images.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the resized and transformed ROIs:
                - "fixed_image_roi": The resized fixed image ROI.
                - "moving_image_roi": The resized moving image ROI.
                - "euler_tfmed_moving_image_roi": The Euler-transformed moving image ROI. """

        nrrd_images_path = locate_dataset(self.dataset_name)
        t_moving, t_fixed = self.registration_problem.split('to')
        fixed_image_roi_path = f"{nrrd_images_path}/img_roi_watershed/{t_fixed}.nrrd"
        moving_image_roi_path = f"{nrrd_images_path}/img_roi_watershed/{t_moving}.nrrd"

        # resize the fixed and moving image ROIs
        resized_fixed_image_roi = self._resize_image_roi(
                fixed_image_roi_path,
                self.CM_dict[self.problem_id]["fixed"]
        )
        resized_moving_image_roi = self._resize_image_roi(
                moving_image_roi_path,
                self.CM_dict[self.problem_id]["moving"]
        )
        euler_transformed_moving_image_roi = self._euler_transform_image_roi(
                resized_moving_image_roi
        )
        return {
            "fixed_image_roi": resized_fixed_image_roi,
            "moving_image_roi": resized_moving_image_roi,
            "euler_tfmed_moving_image_roi": euler_transformed_moving_image_roi
        }

    def _resize_image_roi(
        self,
        image_roi_path: str,
        image_CM: List[int]
    ) -> np.ndarray:
        """ Resize the image ROI based on the center of mass (CM).

        Args:
            image_roi_path (str): The path to the image ROI file.
            image_CM (Tuple[int, int, int]): The center of mass for resizing.

        Returns:
            np.ndarray: The resized image ROI. """

        image_roi_T = get_image_T(image_roi_path)

        return get_cropped_image(
                image_roi_T,
                image_CM,
                self.image_shape, -1).astype(np.float32)

    def _euler_transform_image_roi(
        self,
        moving_image_roi: np.ndarray,
        interpolation: str = "nearest"
    ) -> np.ndarray:
        """ Apply Euler transformation to the moving image ROI.

        Args:
            moving_image_roi (np.ndarray): The region of interest of the moving image to be
                transformed.
            interpolation (str, optional): The interpolation method to use. Default is "nearest".

        Returns:
            np.ndarray: The Euler-transformed moving image ROI. """

        x_dim, y_dim, z_dim = self.image_shape
        _memory_dict_xy = self._initialize_memory_dict(x_dim, y_dim, z_dim)

        return self._apply_euler_parameters(
                moving_image_roi,
                _memory_dict_xy,
                interpolation
        )

    def _initialize_memory_dict(
        self,
        dim_1: int,
        dim_2: int,
        dim_3: int
    ) -> Dict[str, torch.Tensor]:
        """ Initialize the memory dictionary required for Euler transformation.

        Args:
            dim_1 (int): x dimenstion.
            dim_2 (int): y dimension.
            dim_3 (int): z dimension. 

        Returns:
            Dict[str, torch.Tensor]: The initialized memory dictionary for Euler-GPU. """

        _memory_dict = initialize(
                np.zeros((dim_1, dim_2)).astype(np.float32),
                np.zeros((dim_1, dim_2)).astype(np.float32),
                np.zeros(dim_3),
                np.zeros(dim_3),
                np.zeros(dim_3),
                dim_3,
                self.device
        )
        return _memory_dict

    def _apply_euler_parameters(
        self,
        moving_image_roi: np.ndarray,
        memory_dict: Dict[str, torch.Tensor],
        interpolation: str,
        dimension: str = "xy",
    ) -> np.ndarray:
        """ Apply Euler parameters to transform the moving image ROI.

        Args:
            moving_image_roi (np.ndarray): The region of interest of the moving image to be
                transformed.
            memory_dict (Dict[str, torch.Tensor]): The parameters of Euler-GPU.
            interpolation (str): The interpolation method to use.
            dimension (str, optional): The dimension to transform ('xy', 'xz', 'yz').
                Default is 'xy'.

        Returns:
            np.ndarray: The transformed and translated moving image ROI. """

        best_transformation = torch.tensor(
            self.euler_parameters_dict[self.problem_id][dimension]
        ).to(self.device)

        moving_image_roi = self._adjust_image_shape(
                moving_image_roi,
                dimension
        )
        if dimension == "xy":
            axis = 2
        elif dimension == "xz":
            axis = 1
        elif dimension == "yz":
            axis = 0
        transformed_moving_image_roi = transform_image_3d(
                moving_image_roi,
                memory_dict,
                best_transformation,
                self.device,
                axis,
                interpolation
        )
        translated_moving_image_roi = self._translate_image(
                self._adjust_image_shape(
                    transformed_moving_image_roi,
                    "xy"
                )
        )

        return translated_moving_image_roi

    def _adjust_image_shape(
        self,
        image: np.ndarray,
        dimension: str,
    ) -> np.ndarray:

        x_dim, y_dim, z_dim = self.image_shape
        reshaping_dict = {
            (y_dim, z_dim, x_dim): {
                "xy": (2, 0, 1),
                "xz": (2, 1, 0),
                "yz": (0, 1, 2)
            },
            (x_dim, z_dim, y_dim): {
                "xy": (0, 2, 1),
                "yz": (2, 1, 0),
                "xz": (0, 1, 2)
            },
            (x_dim, y_dim, z_dim): {
                "yz": (1, 2, 0),
                "xz": (0, 2, 1),
                "xy": (0, 1, 2)
            }
        }
        order = reshaping_dict[image.shape][dimension]

        return np.transpose(image, order)

    def _translate_image(
        self,
        image: np.ndarray
    ) -> np.ndarray:

        dz = self.euler_parameters_dict[self.problem_id]["dz"]
        translated_image = np.full(self.image_shape, 0)

        if dz < 0:
            translated_image[:, :, :dz] = image[:, :, -dz:]
        elif dz > 0:
            translated_image[:, :, dz:] = image[:, :, :-dz]
        elif dz == 0:
            translated_image = image

        return translated_image
