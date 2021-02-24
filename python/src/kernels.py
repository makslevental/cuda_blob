import numbers
from functools import lru_cache

import cupy as cp
import numpy as np
from cupy.cudnn import pooling_forward
from numba import cuda

from src.profiling import GPUTimer

componentwise_mult_raw_kernel = cp.RawKernel(
    r"""
extern "C" __global__ void componentwiseMatrixMul1vsBatchfloat2(
    const float2* singleIn,
    const float2* batchIn,
    float2* out,
    int batch_size,
    int rows,
    int cols) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int batch = blockIdx.z;

    size_t yy_xx = row * cols + col;
    size_t zz_yy_xx = batch * (rows * cols) + yy_xx;

    if (batch < batch_size && row < rows && col < cols) {
        out[zz_yy_xx].x =
            singleIn[yy_xx].x * batchIn[zz_yy_xx].x - singleIn[yy_xx].y * batchIn[zz_yy_xx].y;
        out[zz_yy_xx].y =
            singleIn[yy_xx].x * batchIn[zz_yy_xx].y + singleIn[yy_xx].y * batchIn[zz_yy_xx].x;
    }
}""",
    "componentwiseMatrixMul1vsBatchfloat2",
)


@cuda.jit("void(complex64[:], complex64[:], complex64[:], int64, int64, int64)")
def componentwise_mult_numba_depr(single_in, batch_in, out, batch_size, rows, cols):
    col = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    row = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    batch = cuda.blockIdx.z

    yy_xx = row * cols + col
    zz_yy_xx = batch * (rows * cols) + yy_xx
    if batch < batch_size and row < rows and col < cols:
        out[zz_yy_xx] = single_in[yy_xx] * batch_in[zz_yy_xx]


@cuda.jit(
    "void(complex64[:, :], complex64[:, :, :], complex64[:, :, :], uint64, uint64, uint64)"
)
def componentwise_mult_numba(single_in, batch_in, out, batch_size, rows, cols):
    col = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    row = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    batch = cuda.blockIdx.z

    if batch < batch_size and row < rows and col < cols:
        out[batch, row, col] = single_in[row, col] * batch_in[batch, row, col]
        # out[zz_yy_xx].real = (
        #     single_in[yy_xx].real * batch_in[zz_yy_xx].real
        #     - single_in[yy_xx].imag * batch_in[zz_yy_xx].imag
        # )
        # out[zz_yy_xx].imag = (
        #     single_in[yy_xx].real * batch_in[zz_yy_xx].real
        #     + single_in[yy_xx].imag * batch_in[zz_yy_xx].imag
        # )


def cupy_mult(kernel_freqs, img_freqs):
    return kernel_freqs * img_freqs


@lru_cache(maxsize=None)
def gaussian_kernel(width: int = 21, sigma: int = 3, dim: int = 2) -> np.ndarray:
    """Gaussian kernel
    Parameters
    ----------
    width: bandwidth of the kernel
    sigma: std of the kernel
    dim: dimensions of the kernel (images -> 2)

    Returns
    -------
    kernel : gaussian kernel
    """
    assert width > 2

    if isinstance(width, numbers.Number):
        width = [width] * dim
    if isinstance(sigma, numbers.Number):
        sigma = [sigma] * dim
    kernel = 1
    meshgrids = np.meshgrid(*[np.arange(size, dtype=np.float32) for size in width])
    for size, std, mgrid in zip(width, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= (
            1 / (std * np.sqrt(2 * np.pi)) * np.exp(-(((mgrid - mean) / std) ** 2) / 2)
        )

    # Make sure sum of values in gaussian kernel equals 1.
    return kernel / np.sum(kernel)


# TODO(max): this should be created on different GPUs
def create_embedded_kernel(
    sigmas,
    height,
    width,
    truncate=5.0,
) -> cp.core.core.ndarray:
    # can't just instantiate on the gpu because then writes will be compiled
    # to a ton of individual kernels

    kernel = np.zeros((len(sigmas), height, width))
    for i, s in enumerate(sigmas):
        radius = int(truncate * s + 0.5)
        embedded_kernel = gaussian_kernel(width=2 * radius + 1, sigma=s)
        for r in range(height // 2 - radius, height // 2 + radius + 1):
            for c in range(width // 2 - radius, width // 2 + radius + 1):
                # fmt: off
                kernel[i][r][c] = embedded_kernel[r - height // 2 - radius][c - width // 2 - radius]
                # fmt: on

    # roll so that final images don't need to be rolled
    # imagine convolving with a centered dirac delta - you'll induce a shift
    kernel = np.roll(kernel, (height // 2, width // 2), axis=(-2, -1))
    return cp.asarray(kernel)


CUDNN_POOLING_MAX = 0


def max_pool_nd(
    inp: cp.ndarray, size=(3, 3, 3), stride=(1, 1, 1), mode=CUDNN_POOLING_MAX
) -> cp.ndarray:
    pad = tuple((s - 1) // 2 for s in size)
    out = cp.empty((1, 1) + inp.shape, dtype=inp.dtype)
    pooling_forward(inp[cp.newaxis, cp.newaxis, :], out, size, stride, pad, mode)
    return out.squeeze()


# TODO(max): convert back to cupy so i can time correctly
def get_local_maxima(dog_images, sigmas, threshold):
    with GPUTimer("argmax"):
        idx = dog_images.argmax(axis=0)
        m, n = dog_images.shape[1:]
        I, J = cp.ogrid[:m, :n]
        max_values = dog_images[idx, I, J]
        max_values[max_values < threshold] = 0.0
        local_maxima = max_pool_nd(max_values, size=(3, 3), stride=(1, 1))
        mask = (local_maxima == max_values) & (
            max_values > cp.float32(0.0)
        )
        local_maxima = local_maxima[mask]
        coords = idx[mask]
        blobs = cp.asarray(
            (*mask.nonzero(), cp.asarray(sigmas)[coords])
        ).T
    return blobs, local_maxima


resize_images_interpolate_bilinear = cp.ElementwiseKernel(
    "raw T x, S v, S u, T vw, T uw, S H, S W, S outsize",
    "T y",
    """
        // indices
        S v0 = v;
        S v1 = min(v + 1, (S)(H - 1));
        S u0 = u;
        S u1 = min(u + 1, (S)(W - 1));
        // weights
        T w0 = (1 - vw) * (1 - uw);
        T w1 = (1 - vw) * uw;
        T w2 = vw * (1 - uw);
        T w3 = vw * uw;
        // fetch
        S offset = i / outsize * H * W;
        T px0 = x[offset + v0 * W + u0];
        T px1 = x[offset + v0 * W + u1];
        T px2 = x[offset + v1 * W + u0];
        T px3 = x[offset + v1 * W + u1];
        // interpolate
        y = (w0 * px0 + w1 * px1) + (w2 * px2 + w3 * px3);
        """,
    "resize_images_interpolate_bilinear",
)
