from data_utils import get_image_T, write_to_json, filter_and_crop
from deepreg.predict import unwrapped_predict
from tqdm import tqdm
import deepreg.model.layer as layer
import h5py
import json
import numpy as np
import os
import tensorflow as tf


def set_GPU(device: int):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[device], 'GPU')
        except RuntimeError as e:
            print(e)


def register_all_swf360_problems(
    dataset: str,
    model_ckpt_path: str,
    model_config_path: str,
    network_version: str,
):
    dataset_names = {
        "ALv4": "2022-01-06-01",
        "ALv6": "2022-03-30-01",
        "ALv7": "2022-03-30-02",
        "ALv8": "2022-03-31-01",
    }
    # image and label should be constant
    target_image_shape = (284, 120, 64)
    target_label_shape = (200, 3)
    # note: DeepReg creates empty folders under `output_dir` after executing the code
    # which can be safely ignored (and deleted).
    output_dir = 'outputs'

    def get_swf360_problems(dataset):
        """ dataset := one of 'ALv4, ALv6', 'ALv7', 'ALv8' """
        dataset_name = dataset_names[dataset]
        return json.load(
                open(
                    f"resources/registration_problems_{dataset}.json"
                ))["train"][dataset_name]

    def _print_score(experiment, outputs_dict):
        print(
            f"""{experiment} ch1 NCC: {'{:.2f}'.format(outputs_dict['ch1']['ncc'])}
            {experiment} ch2 NCC: {'{:.2f}'.format(outputs_dict['ch2']['ncc'])}"""
        )

    ncc_scores_dict = dict()
    problems = get_swf360_problems(dataset)

    for problem in tqdm(problems[:5]):

        ncc_scores_dict[problem] = {}
        network_outputs = register_single_image_pair(
            dataset,
            problem,
            target_image_shape,
            target_label_shape,
            model_ckpt_path,
            model_config_path,
            output_dir
        )
        ncc_scores_dict[problem][network_version] = [
            network_outputs["ch1"]["ncc"],
            network_outputs["ch2"]["ncc"]
        ]
        #_print_score(full_experiment, full_network_outputs)
        #_print_score(control_experiment, control_network_outputs)
    write_to_json(
        ncc_scores_dict,
        f"{dataset}_ncc_{network_version}",
        "scores"
    )


def register_single_image_pair(
    dataset: str,
    problem: str,
    target_image_shape: tuple,
    target_label_shape: tuple,
    model_ckpt_path: str,
    model_config_path: str,
    output_dir: str,
):
    def compute_score(score_class, *args):

        if len(args) == 2:
            return score_class().call(args[0], args[1]).numpy()[0]
        elif len(args) == 1:
            return score_class(
                img_size=target_image_shape
            ).call(args[0]).numpy()[0]

    def reformat(numpy_array):
        return tf.convert_to_tensor(
                numpy_array[np.newaxis, ...].astype(np.float32))

    def _warp_ch2_with_ddf(problem, fixed_image, moving_image):

        batched_fixed_image = normalize_batched_image(
            np.expand_dims(fixed_image, axis=0)
        )
        batched_moving_image = normalize_batched_image(
            np.expand_dims(moving_image, axis=0)
        )
        if batched_moving_image.dtype == np.float32:

            ddf_output, pred_fixed_image, model = unwrapped_predict(
                batched_fixed_image,
                batched_moving_image,
                output_dir,
                target_label_shape,
                target_label_shape,
                model = None,
                model_ckpt_path = model_ckpt_path,
                model_config_path = model_config_path,
            )
            raw_ncc = calculate_ncc(
                batched_moving_image.numpy().squeeze(),
                batched_fixed_image.numpy().squeeze()
            )
            ncc = calculate_ncc(
                pred_fixed_image.squeeze(),
                batched_fixed_image.numpy().squeeze()
            )

        return {
                "fixed_image": batched_fixed_image.numpy().squeeze(),
                "moving_image": batched_moving_image.numpy().squeeze(),
                "warped_moving_image": pred_fixed_image.squeeze(),
                "raw_ncc": raw_ncc,
                "ncc": ncc,
                "ddf": ddf_output
            }

    def _warp_ch1_with_ddf(problem, ch2_ddf, fixed_image, moving_image):

        batched_fixed_image = normalize_batched_image(
                np.expand_dims(fixed_image, axis=0))
        batched_moving_image = normalize_batched_image(
                np.expand_dims(moving_image, axis=0))

        warping = layer.Warping(
                fixed_image_size=batched_fixed_image.shape[1:4],
                batch_size=1)
        warped_moving_image = warping(inputs=[ch2_ddf, batched_moving_image])

        raw_ncc = calculate_ncc(
                batched_moving_image.numpy().squeeze(),
                batched_fixed_image.numpy().squeeze())
        ncc = calculate_ncc(
                warped_moving_image.numpy().squeeze(),
                batched_fixed_image.numpy().squeeze())

        return {
                "fixed_image": fixed_image,
                "moving_image": moving_image,
                "warped_moving_image": warped_moving_image.numpy().squeeze(),
                "raw_ncc": raw_ncc,
                "ncc": ncc
                }

    all_outputs = {"ch1": {}, "ch2": {}}

    ch2_moving_image, ch2_fixed_image = read_problem_h5(dataset, problem, channel=2)
    all_outputs["ch2"] = _warp_ch2_with_ddf(problem, ch2_fixed_image, ch2_moving_image)

    ch1_moving_image, ch1_fixed_image = read_problem_h5(dataset, problem, channel=1)
    all_outputs["ch1"] = _warp_ch1_with_ddf(
            problem,
            all_outputs["ch2"]["ddf"],
            ch1_fixed_image,
            ch1_moving_image
        )

    return all_outputs


def read_problem_h5(dataset, problem, channel):

    dataset_names = {
        "ALv4": "2022-01-06-01",
        "ALv6": "2022-03-30-01",
        "ALv7": "2022-03-30-02",
        "ALv8": "2022-03-31-01",
    }
    dataset_name = dataset_names[dataset]
    dataset_path = \
    f"/data3/prj_register/{dataset}_swf360_ch{channel}/train/nonaugmented/{dataset_name}"

    with h5py.File(f"{dataset_path}/fixed_images.h5", "r") as f:
        fixed_image = f[problem][:].astype(np.float32)

    with h5py.File(f"{dataset_path}/moving_images.h5", "r") as f:
        moving_image = f[problem][:].astype(np.float32)

    return moving_image, fixed_image


def normalize_batched_image(batched_image, eps=1e-7):

    """ Normalizes each image in a batch to [0, 1] range separately. """

    eps = tf.constant(eps, dtype=tf.float32)
    # calculate the min and max values for each image in the batch
    min_vals = tf.math.reduce_min(batched_image, axis=[1, 2, 3], keepdims=True)
    max_vals = tf.math.reduce_max(batched_image, axis=[1, 2, 3], keepdims=True)

    # normalize each image separately
    batched_image = batched_image - min_vals
    batched_image = batched_image / tf.maximum(tf.cast(max_vals - min_vals,
        tf.float32), eps)

    return batched_image


def calculate_ncc(moving, fixed):

    """ Computes the NCC (Normalized Cross-Correlation) of two image arrays
    `moving` and `fixed` corresponding to a registration. """

    assert fixed.shape == moving.shape, "Fixed and moving must have the same shape."

    med_f = np.median(np.max(fixed, axis=2))
    med_m = np.median(np.max(moving, axis=2))

    fixed_new = np.maximum(fixed - med_f, 0)
    moving_new = np.maximum(moving - med_m, 0)

    mu_f = np.mean(fixed_new)
    mu_m = np.mean(moving_new)

    fixed_new = (fixed_new / mu_f) - 1
    moving_new = (moving_new / mu_m) - 1

    numerator = np.sum(fixed_new * moving_new)
    denominator = np.sqrt(np.sum(fixed_new ** 2) * np.sum(moving_new ** 2))

    return numerator / denominator


def compute_centroid_labels(image, max_centroids = 200):

    centroids = np.zeros((max_centroids, 3), dtype=np.int32) - 1

    for val in range(1, max_centroids + 1):
        indices = np.argwhere(image == val)
        if len(indices) > 0:
            centroid_x = np.mean(indices[:, 0])
            centroid_y = np.mean(indices[:, 1])
            centroid_z = np.mean(indices[:, 2])
            centroids[val-1] = [centroid_x, centroid_y, centroid_z]

    return centroids
