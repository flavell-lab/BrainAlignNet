from euler_gpu.grid_search import grid_search
from euler_gpu.preprocess import (initialize,
    max_intensity_projection_and_downsample)
from euler_gpu.transform import (transform_image_3d, translate_along_z)
from evaluate import calculate_gncc
from tqdm import tqdm
from typing import Dict, List, Tuple, Callable
from data_utils import (write_to_json, locate_dataset, filter_and_crop,
        get_image_T, get_image_CM, filter_image)
import glob
import h5py
import json
import numpy as np
import os
import random
import torch


class EulerRegistrationProcessor:

    def __init__(
        self,
        target_image_shape: Tuple[int, int, int],
        save_directory: str,
        problem_file: str,
        batch_size: int = 200,
        device_name: str = "cuda:2",
        downsample_factor: int = 4,
        euler_search = True):
        """ Initialize the class with the given parameters.

        Args:
            target_image_shape (Tuple[int, int, int]): Shape of the image
                (x_dim, y_dim, z_dim).
            save_directory (str): Directory to save the registered images.
            problem_file (str): Name of the JSON file that contains all the
                registration problems for training, validation, and testing.
            batch_size (int, optional): The size of a batch to process with
                Euler-GPU. Default is 200.
            device_name (str, optional): Name of the device to use for
                processing. Default is "cuda:2".
                downsample_factor (int, optional): Factor to downsample the
                images during the grid search stage of running Euler-GPU.
                Default is 4.
            euler_search (bool, optional): Flag to enable or disable Euler
                search. Default is True. """

        self.target_image_shape = target_image_shape
        self.save_directory = save_directory
        self.problem_file = problem_file
        self.batch_size = batch_size
        self.device_name = device_name
        self.downsample_factor = downsample_factor
        self.euler_search = euler_search

        self.euler_parameters_dict = dict()
        self.outcomes = dict()
        self.CM_dict = dict()

        if self.euler_search:
            self.memory_dict_xy, self._memory_dict_xy = \
                    self._initialize_memory_dict()

        self._ensure_directory_exists(self.save_directory)
        self.tag = problem_file.split(".")[0].split("_")[-1]


    def _ensure_directory_exists(self, path):
        """ Create the given directory if it does not already exist. """

        if not os.path.exists(path):
            os.makedirs(path)

    def _initialize_memory_dict(self):
        """ Initialize memory dictory for storing Euler-GPU parameters. """

        x_dim, y_dim, z_dim = self.target_image_shape
        z_translation_range = range(-z_dim, z_dim)
        x_translation_range_xy = np.sort(np.concatenate((
            np.linspace(-0.24, 0.24, 49),
            np.linspace(-0.46, -0.25, 8),
            np.linspace(0.25, 0.46, 8),
            np.linspace(0.5, 1, 3),
            np.linspace(-1, -0.5, 3)
        )))
        y_translation_range_xy = np.sort(np.concatenate((
            np.linspace(-0.28, 0.28, 29),
            np.linspace(-0.54, -0.3, 5),
            np.linspace(0.3, 0.54, 5),
            np.linspace(0.6, 1.4, 3),
            np.linspace(-1.4, -0.6, 3)
        )))
        theta_rotation_range_xy = np.sort(np.concatenate((
            np.linspace(0, 19, 20),
            np.linspace(20, 160, 29),
            np.linspace(161, 199, 39),
            np.linspace(200, 340, 29),
            np.linspace(341, 359, 19)
        )))
        memory_dict_xy = initialize(
            np.zeros((
                x_dim // self.downsample_factor,
                y_dim // self.downsample_factor)).astype(np.float32),
            np.zeros((
                x_dim // self.downsample_factor,
                y_dim // self.downsample_factor)).astype(np.float32),
            x_translation_range_xy,
            y_translation_range_xy,
            theta_rotation_range_xy,
            self.batch_size,
            self.device_name
        )
        _memory_dict_xy = initialize(
            np.zeros((x_dim, y_dim)).astype(np.float32),
            np.zeros((x_dim, y_dim)).astype(np.float32),
            np.zeros(z_dim),
            np.zeros(z_dim),
            np.zeros(z_dim),
            z_dim,
            self.device_name
        )
        return memory_dict_xy, _memory_dict_xy

    def process_datasets(self, augment: bool = False):
        """ Process datasets by applying Euler-transformations to the moving
        images using the set of parameters that maximize the NCC (Normalized
        Cross-Correlation) between the fixed and moving images.

        Args:
            augment (bool, optional): Flag to enable or disable data
                augmentation. Default is False.

        Example:
            >>> processor = EulerRegistrationProcessor(
            >>>     (284, 120, 64),
            >>>     save_directory="/data/prj_register",
            >>>     problem_file="/data/registration_problems.json"
            >>>     euler_search=True,
            >>>     device_name="cuda:0"
            >>> )
            >>> processor.process_datasets() """

        with open(self.problem_file, 'r') as f:
            problem_dicts = json.load(f)

        for dataset_type, problem_dict in problem_dicts.items():
            # process training, validation, and testing datasets respectively
            self.process_dataset_type(dataset_type, problem_dict)

    def process_dataset_type(
        self,
        dataset_type: str,
        problem_dict: Dict[str, List[str]]):
        """ Process datasets that belong to the same type.

        Args:
            dataset_type (str): The type of dataset (i.e., `train`, `valid`, or `test`).
            problem_dict (Dict[str, List[str]]): A dictionary of problems for each dataset.
                The dictionary should have the following structure:
                {
                    "YYYY-MM-DD-XX": [xtox, ...],
                    "YYYY-MM-DD-XX": [xtox, ...],
                    ...
                } """

        dataset_type_dir = f"{self.save_directory}/{dataset_type}"
        self._ensure_directory_exists(f"{dataset_type_dir}/nonaugmented")

        for dataset_name, problems in problem_dict.items():

            print(f"=====Processing {dataset_name} in {dataset_type}=====")
            self.process_dataset(
                dataset_name,
                problems,
                f"{dataset_type_dir}/nonaugmented"
            )

            if self.euler_search:
                write_to_json(self.outcomes, f"eulergpu_outcomes_{self.tag}")
                write_to_json(self.CM_dict, f"center_of_mass_{self.tag}")
                write_to_json(self.euler_parameters_dict,
                              f"euler_parameters_{self.tag}")

    def process_dataset(
        self,
        dataset_name: str,
        problems: List[str],
        dataset_type_dir: str):
        """ Process a given dataset with Euler transformation.

        Args:
            dataset_name (str): The name of the dataset.
            problems (List[str]): A list of problems associated with this dataset.
            dataset_type_dir (str): The directory to save the processed dataset. """

        save_path = f"{dataset_type_dir}/{dataset_name}"
        self._ensure_directory_exists(save_path)
        dataset_path = locate_dataset(dataset_name)

        with h5py.File(f'{save_path}/moving_images.h5', 'w') as hdf5_m_file, \
            h5py.File(f'{save_path}/fixed_images.h5', 'w') as hdf5_f_file:

            for problem in tqdm(problems):
                problem_id = f"{dataset_name}/{problem}"
                processed_image_dict = self.process_problem(
                        problem_id,
                        dataset_path,
                        hdf5_m_file,
                        hdf5_f_file,
                )
                if len(processed_image_dict) != 0:
                    hdf5_f_file.create_dataset(
                        problem,
                        data = processed_image_dict["fixed_image"])

                    hdf5_m_file.create_dataset(
                        problem,
                        data = processed_image_dict["moving_image"])
                    if self.euler_search:
                        write_to_json(self.outcomes,
                                f"eulergpu_outcomes_{self.tag}")

                        write_to_json(self.CM_dict, f"center_of_mass_{self.tag}")
                        write_to_json(self.euler_parameters_dict,
                                      f"euler_parameters_{self.tag}")

    def process_problem(
        self,
        problem_id: str,
        dataset_path: str,
        hdf5_m_file: h5py.File,
        hdf5_f_file: h5py.File
    ) -> Callable:
        """ Process a specific registration problem using Euler transformation
        or simply cropping to the right image size.

        Args:
            problem_id (str): The identifier for the registration problem.
            dataset_path (str): The path to the dataset containing the problem.
            hdf5_m_file (h5py.File): The HDF5 file object for the moving images.
            hdf5_f_file (h5py.File): The HDF5 file object for the fixed images.

        Returns:
            Callable: A function to either simply crop or crop and transform
                the images. """

        if not self.euler_search:
            return self.simply_crop(
                    problem_id,
                    dataset_path,
                    hdf5_m_file,
                    hdf5_f_file
            )
        else:
            return self.crop_and_transform(
                    problem_id,
                    dataset_path,
                    hdf5_m_file,
                    hdf5_f_file
            )

    def simply_crop(
        self,
        problem_id: str,
        dataset_path: str,
        hdf5_m_file: h5py.File,
        hdf5_f_file: h5py.File
    ) -> Dict[str, np.ndarray]:
        """ Preprocess images by simply adjusting the sizes of the fixed and
        moving images without using Euler transformation. This method resizes
        images according to a target image shape.

        Args:
            problem_id (str): The ID that identifies the dataset name and problem name.
            dataset_path (str): The path to save the dataset.
            hdf5_m_file (h5py.File): The HDF5 file that contains the moving images.
            hdf5_f_file (h5py.File): The HDF5 file that contains the fixed images.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing resized moving and
            fixed images. """

        t_moving, t_fixed = problem_id.split('/')[1].split('to')
        t_moving_4 = t_moving.zfill(4)
        t_fixed_4 = t_fixed.zfill(4)
        fixed_image_path = glob.glob(
            f'{dataset_path}/NRRD_filtered/*_t{t_fixed_4}_ch2.nrrd')[0]
        moving_image_path = glob.glob(
            f'{dataset_path}/NRRD_filtered/*_t{t_moving_4}_ch2.nrrd')[0]
        fixed_image_T = get_image_T(fixed_image_path)
        fixed_image_median = np.median(fixed_image_T)
        moving_image_T = get_image_T(moving_image_path)
        moving_image_median = np.median(moving_image_T)

        try:
            resized_fixed_image_xyz = filter_and_crop(
                    fixed_image_T,
                    fixed_image_median,
                    self.target_image_shape
            )
            resized_moving_image_xyz = filter_and_crop(
                    moving_image_T,
                    moving_image_median,
                    self.target_image_shape
            )

        except Exception as e:
            print(f"an error occured: {e}")
            # log the CMs for later cropping ROI labels
            return {}

        self.CM_dict[problem_id] = {
            "moving": get_image_CM(moving_image_T),
            "fixed": get_image_CM(fixed_image_T)
        }
        return {
            "fixed_image": resized_fixed_image_xyz,
            "moving_image": resized_moving_image_xyz
        }

    def crop_and_transform(
        self,
        problem_id: str,
        dataset_path: str,
        hdf5_m_file: h5py.File,
        hdf5_f_file: h5py.File,
    ) -> Dict[str, np.ndarray]:
        """ Resize both fixed and moving images to the target shape, and then
        Euler-transform the moving image of a given registration problem.

        Args:
            problem_id (str): The ID that identifies the dataset name and
                problem name.
            dataset_path (str): The path where the dataset is saved.
            hdf5_m_file (h5py.File): The HDF5 file object that contains the
                moving images.
            hdf5_f_file (h5py.File): The HDF5 file object that contains the
                fixed images.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the resized and transformed
            fixed and moving images. """

        self.outcomes[problem_id] = dict()
        self.euler_parameters_dict[problem_id] = dict()

        t_moving, t_fixed = problem_id.split('/')[1].split('to')
        t_moving_4 = t_moving.zfill(4)
        t_fixed_4 = t_fixed.zfill(4)
        fixed_image_path = glob.glob(
                f'{dataset_path}/NRRD_filtered/*_t{t_fixed_4}_ch2.nrrd'
        )[0]
        moving_image_path = glob.glob(
                f'{dataset_path}/NRRD_filtered/*_t{t_moving_4}_ch2.nrrd'
        )[0]
        fixed_image_T = get_image_T(fixed_image_path).astype(int)
        fixed_image_median = np.median(fixed_image_T)
        moving_image_T = get_image_T(moving_image_path).astype(int)
        moving_image_median = np.median(moving_image_T)
        resized_fixed_image_xyz = filter_and_crop(
                fixed_image_T,
                fixed_image_median,
                self.target_image_shape
        )
        resized_moving_image_xyz = filter_and_crop(
                moving_image_T,
                moving_image_median,
                self.target_image_shape
        )
        # project onto the x-y plane along the maximum z
        downsampled_resized_fixed_image_xy = \
                max_intensity_projection_and_downsample(
                        resized_fixed_image_xyz,
                        self.downsample_factor,
                        projection_axis = 2).astype(np.float32)
        downsampled_resized_moving_image_xy = \
                max_intensity_projection_and_downsample(
                        resized_moving_image_xyz,
                        self.downsample_factor,
                        projection_axis = 2).astype(np.float32)

        # update the memory dictionary for grid search on x-y image
        self.memory_dict_xy["fixed_images_repeated"][:] = torch.tensor(
                downsampled_resized_fixed_image_xy,
                device = self.device_name,
                dtype = torch.float32
            ).unsqueeze(0).repeat(self.batch_size, 1, 1, 1)
        self.memory_dict_xy["moving_images_repeated"][:] = torch.tensor(
                downsampled_resized_moving_image_xy,
                device = self.device_name,
                dtype = torch.float32
            ).unsqueeze(0).repeat(self.batch_size, 1, 1, 1)

        # search optimal parameters with projected image on the x-y plane
        best_score_xy, best_transformation_xy = grid_search(self.memory_dict_xy)
        # transform the 3d image with the searched parameters
        transformed_moving_image_xyz = transform_image_3d(
                    resized_moving_image_xyz,
                    self._memory_dict_xy,
                    best_transformation_xy,
                    self.device_name,
                    2
        )
        # search for the optimal dz translation
        z_dim = self.target_image_shape[2]
        z_translation_range = range(-z_dim, z_dim)
        dz, gncc, transformed_moving_image_xyz = translate_along_z(
                    z_translation_range,
                    resized_fixed_image_xyz,
                    transformed_moving_image_xyz,
                    moving_image_median
        )
        # log the results
        self.CM_dict[problem_id] = {
                "moving": get_image_CM(moving_image_T),
                "fixed": get_image_CM(fixed_image_T)
        }
        self.euler_parameters_dict[problem_id]["xy"] = [
                score.item() for score in list(best_transformation_xy)
        ]
        self.euler_parameters_dict[problem_id]["dz"] = dz
        self.outcomes[problem_id]["registered_image_xy_gncc"] = \
                best_score_xy.item()
        self.outcomes[problem_id]["registered_image_yz_gncc"] = \
                calculate_gncc(
                    resized_fixed_image_xyz.max(0),
                    transformed_moving_image_xyz.max(0)
                ).item()
        self.outcomes[problem_id]["registered_image_xz_gncc"] = \
                calculate_gncc(
                    resized_fixed_image_xyz.max(1),
                    transformed_moving_image_xyz.max(1)
                ).item()
        self.outcomes[problem_id]["registered_image_xyz_gncc"] = gncc

        return {
            "fixed_image": resized_fixed_image_xyz,
            "moving_image": transformed_moving_image_xyz
        }
