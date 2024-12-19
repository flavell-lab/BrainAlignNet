from data_utils import get_cropped_image, write_to_json
from tqdm import tqdm
from typing import List, Dict, Tuple
from warp import ImageWarper
import glob
import h5py
import json
import nibabel as nib
import numpy as np
import os


class ROIWarper(ImageWarper):

    def __init__(self, *args, **kwargs):

        super().__init__(None, *args, **kwargs)
        self.label_path = "/scratch/nar8991/computer_vision/data"
        self.bad_labels = []

    def warp_label(
        self,
        simply_crop: bool,
    ) -> Dict[str, np.ndarray]:
        """ Warp the ROI images with the Euler parameters obtained from
        preprocessing the registration problems for training and validation. """

        if self._label_exists():
            return self._preprocess_image_roi(simply_crop)
        else:
            print(f"{self.dataset_name} has no labels")
            return {}

    def _label_exists(self):

        return self.registration_problem in \
            os.listdir(f"{self.label_path}/{self.dataset_name}/register_labels/")

    def _update_problem(self):
        self.problem_id = f"{self.dataset_name}/{self._registration_problem}"

    def _preprocess_image_roi(
        self,
        simply_crop: bool,
    ) -> Dict[str, np.ndarray]:
        """ Redefine this method for ROIWarper. """

        roi_path = \
            f"{self.label_path}/{self.dataset_name}/register_labels/{self.registration_problem}"
        fixed_image_roi = nib.load(
                f"{roi_path}/img_fixed.nii.gz").get_fdata().astype(np.float32)
        moving_image_roi = nib.load(
                f"{roi_path}/img_moving.nii.gz").get_fdata().astype(np.float32)

        if self.nonzero_labels(fixed_image_roi) and self.nonzero_labels(moving_image_roi):

            resized_fixed_image_roi = self._resize_image_roi(
                    fixed_image_roi,
                    self.CM_dict[self.problem_id]["fixed"]
            )
            resized_moving_image_roi = self._resize_image_roi(
                    moving_image_roi,
                    self.CM_dict[self.problem_id]["moving"]
            )
            if simply_crop:
                return {
                    "fixed_image_roi": resized_fixed_image_roi,
                    "moving_image_roi": resized_moving_image_roi
                }
            else:
                # pass interpolation method
                euler_transformed_moving_image_roi = self._euler_transform_image_roi(
                        resized_moving_image_roi
                )
                return {
                    "fixed_image_roi": resized_fixed_image_roi,
                    "moving_image_roi": resized_moving_image_roi,
                    "euler_tfmed_moving_image_roi": euler_transformed_moving_image_roi
                }
        else:
            return {}

    def _resize_image_roi(
        self,
        image_roi: np.ndarray,
        image_CM: List[int]
    ) -> np.ndarray:

        return get_cropped_image(
                image_roi,
                image_CM,
                self.image_shape, -1).astype(np.float32)

    def nonzero_labels(self, label_roi):

        if len(np.unique(label_roi)) == 1:
            self.bad_labels.append(self.registration_problem)
            return False
        else:
            return True


def generate_rois(
        device_name: str,
        target_image_shape: Tuple[int, int, int],
        problem_file: str,
        save_directory: str,
        simply_crop: bool,
    ):

    warper = ROIWarper(
        None,  # dataset_name and registration_problem are set later
        None,
        target_image_shape,
        problem_file,
        device_name,
        simply_crop, # default set to False in warp.ImageWarper
    )

    with open(problem_file, "r") as f:
        problem_dict = json.load(f)

    dataset_types = {
        "train": list(problem_dict["train"].keys()),
        "valid": list(problem_dict["valid"].keys()),
        "test": list(problem_dict["test"].keys())
    }
    all_bad_problems = {"train": {}, "valid": {}, "test": {}}
    for dataset_type, datasets in dataset_types.items():
        all_bad_problems[dataset_type] = generate_roi(
                datasets,
                dataset_type,
                warper,
                save_directory,
                problem_dict,
                simply_crop
        )
    write_to_json(all_bad_problems, f"bad_registration_problems")


def generate_roi(
        datasets: List[str],
        dataset_type: str,
        warper: ROIWarper,
        save_directory: str,
        problem_dict: Dict[str, Dict[str, List[str]]],
        simply_crop: bool,
    ):
    bad_problems = dict()

    for dataset in datasets:
        label_path = f"{save_directory}/{dataset_type}/nonaugmented/{dataset}"

        if not os.path.exists(label_path):
            os.makedirs(label_path)

        problems = problem_dict[dataset_type][dataset]
        warper.dataset_name = dataset

        with h5py.File(f"{label_path}/moving_rois.h5", "w") as h5_m_file, \
             h5py.File(f"{label_path}/fixed_rois.h5", "w") as h5_f_file:

            for problem in tqdm(problems):
                warper.registration_problem = problem
                label_dict = warper.warp_label(simply_crop=simply_crop)
                if len(label_dict) > 0:
                    if simply_crop:
                        moving_image_roi = label_dict["moving_image_roi"]
                    else:
                        moving_image_roi = label_dict["euler_tfmed_moving_image_roi"]
                    h5_m_file.create_dataset(
                            problem,
                            data = moving_image_roi
                    )
                    fixed_image_roi = label_dict["fixed_image_roi"]
                    h5_f_file.create_dataset(
                            problem,
                            data = fixed_image_roi
                    )
        print(f"{dataset} generated!")
        bad_problems[dataset] = warper.bad_labels

    return bad_problems
