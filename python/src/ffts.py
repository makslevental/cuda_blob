import cupy as cp
from cupy.cuda.cufft import CUFFT_FORWARD, CUFFT_INVERSE
from cupy.fft._fft import _fftn as fftn
from cupyx.scipy.fftpack import get_fft_plan

from kernels import componentwise_mult_numba
from util import get_grid_block_dims

cp.fft.config.enable_nd_planning = False


def get_fft(inp: cp.ndarray, forward_plan, axes=(-2, -1)) -> cp.ndarray:
    # with s=None the sizes of the input along the `axes` dimensions
    freqs = fftn(
        inp,
        s=None,
        axes=axes,
        norm="ortho",  # divides by sqrt(2)
        direction=CUFFT_FORWARD,
        value_type="R2C",
        order="C",
        plan=forward_plan,
        overwrite_x=False,
    )
    return freqs


def get_inverse_fft(freqs, inverse_plan, axes=(-2, -1)) -> cp.ndarray:
    return fftn(
        freqs,
        s=None,
        axes=axes,
        norm="ortho",
        direction=CUFFT_INVERSE,
        value_type="C2R",
        order="C",
        plan=inverse_plan,
        overwrite_x=False,
    )


def get_forward_plans(img, kernel, axes=(-2, -1)):
    # re axes
    # only image axes (not batch); this is so that the plans
    # become batch plans for the kernel fft
    img_forward_plan = get_fft_plan(img, axes=axes, value_type="R2C")
    kernel_forward_plan = get_fft_plan(kernel, axes=axes, value_type="R2C")
    return img_forward_plan, kernel_forward_plan


def filter_imgs(img_freqs, kernel_freqs, kernel_inverse_plan):
    filtered_imgs_freqs = cp.empty_like(kernel_freqs, dtype=kernel_freqs.dtype)
    grid_block_dims = get_grid_block_dims(*filtered_imgs_freqs.shape)
    componentwise_mult_numba[grid_block_dims](
        img_freqs, kernel_freqs, filtered_imgs_freqs, *filtered_imgs_freqs.shape
    )

    assert kernel_freqs.shape == filtered_imgs_freqs.shape
    filtered_imgs = get_inverse_fft(filtered_imgs_freqs, kernel_inverse_plan)
    return filtered_imgs
