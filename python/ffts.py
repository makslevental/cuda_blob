import cupy as cp
from cupy.cuda.cufft import CUFFT_FORWARD, CUFFT_INVERSE
from cupy.fft._fft import _fftn as fftn

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
