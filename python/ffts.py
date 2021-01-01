import cupy as cp
from cupy.cuda.cufft import CUFFT_FORWARD, CUFFT_INVERSE
from cupy.fft._fft import _fftn
from cupyx.scipy.fftpack import get_fft_plan

cp.fft.config.enable_nd_planning = False


# TODO: cache the plans

def get_fft(inp):
    axes = (-2, -1)
    forward_plan = get_fft_plan(inp, axes=axes, value_type="R2C")
    # with s=None the sizes of the input along the `axes` dimensions
    freqs = _fftn(
        inp,
        s=None,
        axes=axes,
        norm="ortho",
        direction=CUFFT_FORWARD,
        value_type="R2C",
        order="C",
        plan=forward_plan,
        overwrite_x=False,
    )
    return freqs


def get_inverse_fft(freqs):
    axes = (-2, -1)
    inverse_plan = get_fft_plan(freqs, axes=axes, value_type="C2R")
    return _fftn(
        freqs,
        s=None,
        axes=(-2, -1),
        norm="ortho",
        direction=CUFFT_INVERSE,
        value_type="C2R",
        order="C",
        plan=inverse_plan,
        overwrite_x=False,
    )
