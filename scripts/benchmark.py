# These functions were adapted from `deepreg/model/kernel.py` in the 'DeepReg' repository
# on GitHub. The original implementation can be found at:
# https://github.com/flavell-lab/DeepReg/deepreg/loss
# The code is used under the MIT License.
from typing import Tuple
import numpy as np
import tensorflow as tf
import benchmark_utils as utils

EPS = 1.0e-5

######################
##### Label loss #####
######################
class CentroidDistScore(tf.keras.losses.Loss):

    def __init__(
        self,
        smooth_nr: float = EPS,
        smooth_dr: float = EPS,
        name: str = "CentroidDistance",
        **kwargs,
    ):
        """
        Init.

        :param binary: if True, project y_true, y_pred to 0 or 1.
        :param background_weight: weight for background, where y == 0.
        :param smooth_nr: small constant added to numerator in case of zero covariance.
        :param smooth_dr: small constant added to denominator in case of zero variance.
        :param name: name of the loss.
        :param kwargs: additional arguments.
        """
        super().__init__(name=name, **kwargs)
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.flatten = tf.keras.layers.Flatten()

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Return loss for a batch.

        :param y_true: shape = (batch, ...)
        :param y_pred: shape = (batch, ...)
        :return: shape = (batch,)
        """
        y_true = tf.cast(y_true, tf.float32)
        # values that weren't in the original moving image
        mask_true = tf.math.reduce_all(tf.equal(y_true, -1.0), axis=-1)
        # values that weren't in the original fixed image (only way to get -1 is to be outside the image)
        mask_pred = tf.math.reduce_all(tf.equal(y_pred, -1.0), axis=-1)
        mask = tf.math.logical_or(mask_true, mask_pred)

        mask_expanded = tf.expand_dims(mask, axis=-1)
        displacement = tf.where(mask_expanded, 0.0, y_pred - tf.cast(y_true, tf.float32))
        distance = tf.norm(displacement, axis=-1)

        return (tf.reduce_sum(distance, axis=-1) + \
                self.smooth_nr) / (tf.reduce_sum(
                    1.0 - tf.cast(mask, dtype=tf.float32),
                    axis=-1) + self.smooth_dr)

######################
##### Image loss #####
#####################
class LocalNormalizedCrossCorrelation(tf.keras.losses.Loss):
    """
    Local squared zero-normalized cross-correlation.

    Denote y_true as t and y_pred as p. Consider a window having n elements.
    Each position in the window corresponds a weight w_i for i=1:n.

    Define the discrete expectation in the window E[t] as

        E[t] = sum_i(w_i * t_i) / sum_i(w_i)

    Similarly, the discrete variance in the window V[t] is

        V[t] = E[t**2] - E[t] ** 2

    The local squared zero-normalized cross-correlation is therefore

        E[ (t-E[t]) * (p-E[p]) ] ** 2 / V[t] / V[p]

    where the expectation in numerator is

        E[ (t-E[t]) * (p-E[p]) ] = E[t * p] - E[t] * E[p]

    Different kernel corresponds to different weights.

    For now, y_true and y_pred have to be at least 4d tensor, including batch axis.

    Reference:

        - Zero-normalized cross-correlation (ZNCC):
            https://en.wikipedia.org/wiki/Cross-correlation
        - Code: https://github.com/voxelmorph/voxelmorph/blob/legacy/src/losses.py
    """

    kernel_fn_dict = dict(
        gaussian=utils.gaussian_kernel1d,
        rectangular=utils.rectangular_kernel1d,
        triangular=utils.triangular_kernel1d,
    )

    def __init__(
        self,
        kernel_size: int = 9,
        kernel_type: str = "rectangular",
        smooth_nr: float = EPS,
        smooth_dr: float = EPS,
        name: str = "LocalNormalizedCrossCorrelation",
        **kwargs,
    ):
        """
        Init.

        :param kernel_size: int. Kernel size or kernel sigma for kernel_type='gauss'.
        :param kernel_type: str, rectangular, triangular or gaussian
        :param smooth_nr: small constant added to numerator in case of zero covariance.
        :param smooth_dr: small constant added to denominator in case of zero variance.
        :param name: name of the loss.
        :param kwargs: additional arguments.
        """
        super().__init__(name=name, **kwargs)
        if kernel_type not in self.kernel_fn_dict.keys():
            raise ValueError(
                f"Wrong kernel_type {kernel_type} for LNCC loss type. "
                f"Feasible values are {self.kernel_fn_dict.keys()}"
            )
        self.kernel_fn = self.kernel_fn_dict[kernel_type]
        self.kernel_type = kernel_type
        self.kernel_size = kernel_size
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr

        # (kernel_size, )
        self.kernel = self.kernel_fn(kernel_size=self.kernel_size)
        # E[1] = sum_i(w_i), ()
        self.kernel_vol = tf.reduce_sum(
            self.kernel[:, None, None]
            * self.kernel[None, :, None]
            * self.kernel[None, None, :]
        )

    def calc_ncc(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor
    ) -> tf.Tensor:
        """
        Return NCC for a batch.

        The kernel should not be normalized, as normalizing them leads to computation
        with small values and the precision will be reduced.
        Here both numerator and denominator are actually multiplied by kernel volume,
        which helps the precision as well.
        However, when the variance is zero, the obtained value might be negative due to
        machine error. Therefore a hard-coded clipping is added to
        prevent division by zero.

        :param y_true: shape = (batch, dim1, dim2, dim3, 1)
        :param y_pred: shape = (batch, dim1, dim2, dim3, 1)
        :return: shape = (batch, dim1, dim2, dim3. 1)
        """

        # t = y_true, p = y_pred
        # (batch, dim1, dim2, dim3, 1)
        t2 = y_true * y_true
        p2 = y_pred * y_pred
        tp = y_true * y_pred

        # sum over kernel
        # (batch, dim1, dim2, dim3, 1)
        t_sum = utils.separable_filter(y_true, kernel=self.kernel)  # E[t] * E[1]
        p_sum = utils.separable_filter(y_pred, kernel=self.kernel)  # E[p] * E[1]
        t2_sum = utils.separable_filter(t2, kernel=self.kernel)  # E[tt] * E[1]
        p2_sum = utils.separable_filter(p2, kernel=self.kernel)  # E[pp] * E[1]
        tp_sum = utils.separable_filter(tp, kernel=self.kernel)  # E[tp] * E[1]

        # average over kernel
        # (batch, dim1, dim2, dim3, 1)
        t_avg = t_sum / self.kernel_vol  # E[t]
        p_avg = p_sum / self.kernel_vol  # E[p]

        # shape = (batch, dim1, dim2, dim3, 1)
        cross = tp_sum - p_avg * t_sum  # E[tp] * E[1] - E[p] * E[t] * E[1]
        t_var = t2_sum - t_avg * t_sum  # V[t] * E[1]
        p_var = p2_sum - p_avg * p_sum  # V[p] * E[1]

        # ensure variance >= 0
        t_var = tf.maximum(t_var, 0)
        p_var = tf.maximum(p_var, 0)

        # (E[tp] - E[p] * E[t]) ** 2 / V[t] / V[p]
        ncc = (cross * cross) / (t_var * p_var + self.smooth_dr)

        return ncc

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor
    ) -> tf.Tensor:
        """
        Return loss for a batch.

        TODO: support channel axis dimension > 1.

        :param y_true: shape = (batch, dim1, dim2, dim3)
            or (batch, dim1, dim2, dim3, 1)
        :param y_pred: shape = (batch, dim1, dim2, dim3)
            or (batch, dim1, dim2, dim3, 1)
        :return: shape = (batch,)
        """
        # sanity checks
        if len(y_true.shape) == 4:
            y_true = tf.expand_dims(y_true, axis=4)
        if y_true.shape[4] != 1:
            raise ValueError(
                "Last dimension of y_true is not one. " f"y_true.shape = {y_true.shape}"
            )
        if len(y_pred.shape) == 4:
            y_pred = tf.expand_dims(y_pred, axis=4)
        if y_pred.shape[4] != 1:
            raise ValueError(
                "Last dimension of y_pred is not one. " f"y_pred.shape = {y_pred.shape}"
            )

        ncc = self.calc_ncc(y_true=y_true, y_pred=y_pred)
        return tf.reduce_mean(ncc, axis=[1, 2, 3, 4])


class GlobalNormalizedCrossCorrelation(tf.keras.losses.Loss):
    """
    Global squared zero-normalized cross-correlation.

    Compute the squared cross-correlation between the reference and moving images
    y_true and y_pred have to be at least 4d tensor, including batch axis.

    Reference:

        - Zero-normalized cross-correlation (ZNCC):
            https://en.wikipedia.org/wiki/Cross-correlation

    """

    def __init__(
        self,
        name: str = "GlobalNormalizedCrossCorrelation",
        **kwargs,
    ):
        """
        Init.

        :param name: name of the loss
        :param kwargs: additional arguments.
        """
        super().__init__(name=name, **kwargs)

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor
    ) -> tf.Tensor:
        """
        Return loss for a batch.

        :param y_true: shape = (batch, ...)
        :param y_pred: shape = (batch, ...)
        :return: shape = (batch,)
        """

        axis = [a for a in range(1, len(y_true.shape))]
        mu_pred = tf.reduce_mean(y_pred, axis=axis, keepdims=True)
        mu_true = tf.reduce_mean(y_true, axis=axis, keepdims=True)
        var_pred = tf.math.reduce_variance(y_pred, axis=axis)
        var_true = tf.math.reduce_variance(y_true, axis=axis)
        numerator = tf.abs(
            tf.reduce_mean((y_pred - mu_pred) * (y_true - mu_true), axis=axis)
        )

        return (numerator * numerator) / (var_pred * var_true + EPS)


###########################################
##### Deformation/Regularization loss #####
###########################################
class NonRigidPenalty(tf.keras.layers.Layer):
    """
    Calculate the L1/L2 norm of ddf using central finite difference.

    Take difference between the norm and the norm of a reference grid to penalize any non-rigid transformation.

    y_true and y_pred have to be at least 5d tensor, including batch axis.
    """

    def __init__(
        self,
        img_size: Tuple[int, int, int] = (0, 0, 0),
        l1: bool = False,
        name: str = "NonRigidPenalty",
        **kwargs
    ):
        """
        Init.

        :param img_size: size of the 3d images, for initializing reference grid
        :param l1: bool true if calculate L1 norm, otherwise L2 norm
        :param name: name of the loss
        :param kwargs: additional arguments.
        """
        super().__init__(name=name)
        self.l1 = l1

        # Assert that img_size has been changed from the default value
        assert img_size != (0, 0, 0), "img_size must be set to a value other than (0, 0, 0)"

        self.img_size = img_size
        grid_ref = tf.expand_dims(utils.get_reference_grid(grid_size=self.img_size), axis=0)
        self.ddf_ref = -grid_ref

    def call(
        self,
        inputs: tf.Tensor,
        **kwargs
    ) -> tf.Tensor:
        """
        Return a scalar loss.

        :param inputs: shape = (batch, m_dim1, m_dim2, m_dim3, 3)
        :param kwargs: additional arguments.
        :return: shape = (batch, )
        """
        assert len(inputs.shape) == 5
        ddf = inputs
        # first order gradient
        # (batch, m_dim1-2, m_dim2-2, m_dim3-2, 3)
        dfdx = utils.gradient_dxyz(ddf - self.ddf_ref, utils.gradient_dx)
        dfdy = utils.gradient_dxyz(ddf - self.ddf_ref, utils.gradient_dy)
        dfdz = utils.gradient_dxyz(ddf - self.ddf_ref, utils.gradient_dz)
        if self.l1:
            norms = tf.abs(utils.stable_f(tf.abs(dfdx) + tf.abs(dfdy) + tf.abs(dfdz)) - 2.0)
        else:
            norms = tf.abs(utils.stable_f(dfdx ** 2 + dfdy ** 2 + dfdz ** 2) - 2.0)
        return tf.reduce_mean(norms, axis=[1, 2, 3, 4])


class DifferenceNorm(tf.keras.layers.Layer):

    """
    Calculate the average displacement of a pixel in the image, using taxicab metric.

    y_true and y_pred have to be at least 5d tensor, including batch axis.
    """
    def __init__(
        self,
        l1: bool = False,
        name: str = "DifferenceNorm",
        **kwargs
    ):
        """
        Init.

        :param l1: bool true if calculate L1 norm, otherwise L2 norm
        :param name: name of the loss
        :param kwargs: additional arguments.
        """
        super().__init__(name=name)
        self.l1 = l1

    def call(
        self,
        inputs: tf.Tensor,
    **kwargs) -> tf.Tensor:
        """
        Return a scalar loss.

        :param inputs: shape = (batch, m_dim1, m_dim2, m_dim3, 3)
        :param kwargs: additional arguments.
        :return: shape = (batch, )
        """
        assert len(inputs.shape) == 5
        ddf = inputs
        # first order gradient
        # (batch, m_dim1-2, m_dim2-2, m_dim3-2, 3)
        if self.l1:
            norms = tf.abs(ddf)
        else:
            norms = ddf ** 2
        return tf.reduce_mean(norms, axis=[1, 2, 3, 4])


def calculate_ncc(moving, fixed):
    """
    Computes the NCC (Normalized Cross-Correlation) of two image arrays
    `moving` and `fixed` corresponding to a registration.
    """
    assert fixed.shape == moving.shape, "Fixed and moving images must have the same shape."

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

