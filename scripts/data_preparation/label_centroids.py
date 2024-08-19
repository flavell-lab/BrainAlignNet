from numpy.typing import NDArray
from tqdm import tqdm
import h5py
import numpy as np
import os


class CentroidLabel:
    """ A class to generate centroid labels for datasets.

    Args:
        dataset_path (str): Path to the dataset directory. """

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def create_all_labels(self):
        """ Create labels for all datasets in the 'train' and 'valid' directories. """

        train_dataset_names = os.listdir(f"{self.dataset_path}/train/nonaugmented")
        valid_dataset_names = os.listdir(f"{self.dataset_path}/valid/nonaugmented")

        for dataset_name in train_dataset_names:
            self.create_one_dataset_labels(dataset_name, "train")
        for dataset_name in valid_dataset_names:
            self.create_one_dataset_labels(dataset_name, "valid")

    def create_one_dataset_labels(
        self,
        dataset_name: str,
        dataset_type: str,
        max_centroids: int = 200):
        """ Create labels for one dataset.

        Args:
            dataset_name (str): Name of the dataset to process.
            dataset_type (str): Type of the dataset ('train', 'valid', etc.).
            max_centroids (int, optional): Maximum number of centroids to compute. Default is 200.
        """

        save_directory = f"{self.dataset_path}/{dataset_type}/nonaugmented/{dataset_name}"
        self._ensure_directory_exists(save_directory)

        fixed_roi_path = f"{save_directory}/fixed_rois.h5"
        moving_roi_path = f"{save_directory}/moving_rois.h5"

        problems = list(h5py.File(moving_roi_path, "r").keys())

        with h5py.File(f"{save_directory}/moving_labels.h5", "w") as hdf5_m_file, \
                h5py.File(f"{save_directory}/fixed_labels.h5", "w") as hdf5_f_file:

            for problem in tqdm(problems):

                fixed_roi = self._read_roi(fixed_roi_path, problem)
                moving_roi = self._read_roi(moving_roi_path, problem)

                fixed_centroids = self._compute_centroids_3d(
                        fixed_roi,
                        max_centroids
                )
                moving_centroids = self._compute_centroids_3d(
                        moving_roi,
                        max_centroids
                )
                # IMPORTANT: flip moving and fixed labels
                hdf5_m_file[problem] = fixed_centroids
                hdf5_f_file[problem] = moving_centroids

    def _read_roi(self, roi_path: str, key: str) -> NDArray[np.int32]:
        """ Read the ROI data from an HDF5 file.

        Args:
            roi_path (str): Path to the ROI HDF5 file.
            key (str): Key to access the specific dataset within the HDF5 file.

        Returns:
            NDArray[np.int32]: Numpy array containing the ROI data. """

        return h5py.File(roi_path, "r")[key][:]

    def _ensure_directory_exists(self, path):
        """ Create the given directory if it does not already exist.

        Args:
            path (str): Directory to create if it does not exist. """

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    def _compute_centroids_3d(
        self,
        image: NDArray[np.int32],
        max_centroids: int
    ) -> NDArray[np.int32]:
        """ Compute the centroids of all pixels with each unique value in a 3D image.

        Args:
            image (NDArray[np.int32]): A 3D numpy array representing the image with dimensions
                (x, y, z).
            max_centroids (int): The maximum number of centroids contained in a given label;
                included because the network expects all labels to have shape (max_centroids, 3).

        Returns:
            NDArray[np.int32]: A (max_centroids, 3) numpy array where each row corresponds to the
                centroid coordinates (x, y, z) for each value. """

        centroids = np.zeros((max_centroids, 3), dtype=np.int32) - 1  # Initialize the centroids array

        for val in range(1, max_centroids + 1):
            # Find the indices of pixels that have the current value
            indices = np.argwhere(image == val)

            # Compute the centroid if the value is present in the image
            if len(indices) > 0:
                centroid_x = np.mean(indices[:, 0])  # x-coordinate
                centroid_y = np.mean(indices[:, 1])  # y-coordinate
                centroid_z = np.mean(indices[:, 2])  # z-coordinate
                centroids[val-1] = [centroid_x, centroid_y, centroid_z]

        return centroids

