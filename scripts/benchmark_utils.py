# These functions were adapted from `deepreg/model/kernel.py` in the 'DeepReg' repository
# on GitHub. The original implementation can be found at:
# https://github.com/flavell-lab/DeepReg/deepreg/loss
# The code is used under the MIT License.
from typing import Callable, List, Tuple, Union
import math
import tensorflow as tf


def get_reference_grid(
        grid_size: Union[Tuple[int, ...], 
        List[int]]
    ) -> tf.Tensor:
    """
    Generate a 3D grid with given size.

    Reference:

    - volshape_to_meshgrid of neuron
      https://github.com/adalca/neurite/blob/legacy/neuron/utils.py

      neuron modifies meshgrid to make it faster, however local
      benchmark suggests tf.meshgrid is better

    Note:

    for tf.meshgrid, in the 3-D case with inputs of length M, N and P,
    outputs are of shape (N, M, P) for ‘xy’ indexing and
    (M, N, P) for ‘ij’ indexing.

    :param grid_size: list or tuple of size 3, [dim1, dim2, dim3]
    :return: shape = (dim1, dim2, dim3, 3),
             grid[i, j, k, :] = [i j k]
    """

    # dim1, dim2, dim3 = grid_size
    # mesh_grid has three elements, corresponding to i, j, k
    # for i in range(dim1)
    #     for j in range(dim2)
    #         for k in range(dim3)
    #             mesh_grid[0][i,j,k] = i
    #             mesh_grid[1][i,j,k] = j
    #             mesh_grid[2][i,j,k] = k
    mesh_grid = tf.meshgrid(
        tf.range(grid_size[0]),
        tf.range(grid_size[1]),
        tf.range(grid_size[2]),
        indexing="ij",
    )  # has three elements, each shape = (dim1, dim2, dim3)
    grid = tf.stack(mesh_grid, axis=3)  # shape = (dim1, dim2, dim3, 3)
    grid = tf.cast(grid, dtype=tf.float32)

    return grid


def gaussian_kernel1d(kernel_size: int) -> tf.Tensor:
    """
    Return a the 1D Gaussian kernel for LocalNormalizedCrossCorrelation.

    :param kernel_size: scalar, size of the 1-D kernel
    :return: filters, of shape (kernel_size, )
    """
    mean = (kernel_size - 1) / 2.0
    sigma = kernel_size / 3

    grid = tf.range(0, kernel_size, dtype=tf.float32)
    filters = tf.exp(-tf.square(grid - mean) / (2 * sigma ** 2))

    return filters


def triangular_kernel1d(kernel_size: int) -> tf.Tensor:
    """
    Return a the 1D triangular kernel for LocalNormalizedCrossCorrelation.

    Assume kernel_size is odd, it will be a smoothed from
    a kernel which center part is zero.
    Then length of the ones will be around half kernel_size.
    The weight scale of the kernel does not matter as LNCC will normalize it.

    :param kernel_size: scalar, size of the 1-D kernel
    :return: kernel_weights, of shape (kernel_size, )
    """
    assert kernel_size >= 3
    assert kernel_size % 2 != 0

    padding = kernel_size // 2
    kernel = tf.constant(
        [0] * math.ceil(padding / 2)
        + [1] * (kernel_size - padding)
        + [0] * math.floor(padding / 2),
        dtype=tf.float32,
    )

    # (padding*2, )
    filters = tf.ones(shape=(kernel_size - padding, 1, 1), dtype=tf.float32)

    # (kernel_size, 1, 1)
    kernel = tf.nn.conv1d(
        kernel[None, :, None], filters=filters, stride=[1, 1, 1], padding="SAME"
    )

    return kernel[0, :, 0]


def rectangular_kernel1d(kernel_size: int) -> tf.Tensor:
    """
    Return a the 1D rectangular kernel for LocalNormalizedCrossCorrelation.

    :param kernel_size: scalar, size of the 1-D kernel
    :return: kernel_weights, of shape (kernel_size, )
    """

    kernel = tf.ones(shape=(kernel_size,), dtype=tf.float32)
    return kernel


def separable_filter(tensor: tf.Tensor, kernel: tf.Tensor) -> tf.Tensor:
    """
    Create a 3d separable filter.

    Here `tf.nn.conv3d` accepts the `filters` argument of shape
    (filter_depth, filter_height, filter_width, in_channels, out_channels),
    where the first axis of `filters` is the depth not batch,
    and the input to `tf.nn.conv3d` is of shape
    (batch, in_depth, in_height, in_width, in_channels).

    :param tensor: shape = (batch, dim1, dim2, dim3, 1)
    :param kernel: shape = (dim4,)
    :return: shape = (batch, dim1, dim2, dim3, 1)
    """
    strides = [1, 1, 1, 1, 1]
    kernel = tf.cast(kernel, dtype=tensor.dtype)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    tensor = tf.nn.conv3d(
        tf.nn.conv3d(
            tf.nn.conv3d(
                tensor,
                filters=tf.reshape(kernel, [-1, 1, 1, 1, 1]),
                strides=strides,
                padding="SAME",
            ),
            filters=tf.reshape(kernel, [1, -1, 1, 1, 1]),
            strides=strides,
            padding="SAME",
        ),
        filters=tf.reshape(kernel, [1, 1, -1, 1, 1]),
        strides=strides,
        padding="SAME",
    )
    return tensor


def gradient_dx(fx: tf.Tensor) -> tf.Tensor:
    """
    Calculate gradients on x-axis of a 3D tensor using central finite difference.

    It moves the tensor along axis 1 to calculate the approximate gradient, the x axis,
    dx[i] = (x[i+1] - x[i-1]) / 2.

    :param fx: shape = (batch, m_dim1, m_dim2, m_dim3)
    :return: shape = (batch, m_dim1-2, m_dim2-2, m_dim3-2)
    """
    return (fx[:, 2:, 1:-1, 1:-1] - fx[:, :-2, 1:-1, 1:-1]) / 2


def gradient_dy(fy: tf.Tensor) -> tf.Tensor:
    """
    Calculate gradients on y-axis of a 3D tensor using central finite difference.

    It moves the tensor along axis 2 to calculate the approximate gradient, the y axis,
    dy[i] = (y[i+1] - y[i-1]) / 2.

    :param fy: shape = (batch, m_dim1, m_dim2, m_dim3)
    :return: shape = (batch, m_dim1-2, m_dim2-2, m_dim3-2)
    """
    return (fy[:, 1:-1, 2:, 1:-1] - fy[:, 1:-1, :-2, 1:-1]) / 2


def gradient_dz(fz: tf.Tensor) -> tf.Tensor:
    """
    Calculate gradients on z-axis of a 3D tensor using central finite difference.

    It moves the tensor along axis 3 to calculate the approximate gradient, the z axis,
    dz[i] = (z[i+1] - z[i-1]) / 2.

    :param fz: shape = (batch, m_dim1, m_dim2, m_dim3)
    :return: shape = (batch, m_dim1-2, m_dim2-2, m_dim3-2)
    """
    return (fz[:, 1:-1, 1:-1, 2:] - fz[:, 1:-1, 1:-1, :-2]) / 2


def gradient_dxyz(fxyz: tf.Tensor, fn: Callable) -> tf.Tensor:
    """
    Calculate gradients on x,y,z-axis of a tensor using central finite difference.

    The gradients are calculated along x, y, z separately then stacked together.

    :param fxyz: shape = (..., 3)
    :param fn: function to call
    :return: shape = (..., 3)
    """
    return tf.stack([fn(fxyz[..., i]) for i in [0, 1, 2]], axis=4)


def stable_f(x, min_value=1e-6):
    """
    Perform the operation f(x) = x + 1/x in a numerically stable way.

    This function is intended to penalize growing and shrinking equally.

    :param x: Input tensor.
    :param min_value: The minimum value to which x will be clamped.
    :return: The result of the operation.
    """
    x_clamped = tf.clip_by_value(x, min_value, tf.float32.max)
    return x_clamped + 1.0 / x_clamped

