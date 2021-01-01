import cupy as cp
import numpy as np
from PIL import Image, ImageChops
from PIL.Image import BILINEAR
from cupy.cuda.cufft import CUFFT_FORWARD, CUFFT_INVERSE
from cupy.fft._fft import _fftn
from cupyx.scipy.fftpack import get_fft_plan

from util import gaussian_kernel

DEBUG = True

cp.fft.config.enable_nd_planning = False


# just happens to work for both the kernel and the image
def get_fft(inp):
    axes = (-2, -1)
    shape = list(inp.shape[-2:])
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


def create_embedded_kernel(batch_size, height, width) -> cp.core.core.ndarray:
    # can't just instantiate on the gpu because then writes will be compiled
    # to a ton of individual kernels
    kernel = np.zeros((batch_size, height, width))
    for k in range(batch_size):
        radius = k + 1
        assert 2 * radius + 1 <= height
        # TODO: not right
        embedded_kernel = gaussian_kernel(width=2 * radius + 1)
        for r in range(height // 2 - radius, height // 2 + radius + 1):
            for c in range(width // 2 - radius, width // 2 + radius + 1):
                # fmt: off
                kernel[k][r][c] = embedded_kernel[r - height // 2 - radius][c - width // 2 - radius]
                # fmt: on

    return cp.asarray(kernel)


def trim(img):
    bg = Image.new(img.mode, img.size, img.getpixel((0, 0)))
    diff = ImageChops.difference(img, bg)
    diff = ImageChops.add(diff, diff, scale=2.0, offset=-100)
    bbox = diff.getbbox()
    if bbox:
        return img.crop(bbox)
    else:
        return img


def load_tiff(fp, resize=None):
    # TODO: pinned memory?
    # convert to I so that there's no clipping (float for CUDA)
    img = trim(Image.open(fp)).convert("I").convert("F")
    if resize is None:
        # nearest low 2^k for fft
        resize = 2 ** int(np.log2(min(*img.size)))
        resize = (resize, resize)

    img = img.resize(resize, resample=BILINEAR)
    return cp.array(img) / 255.0


def main():
    fp = "/home/max/dev_projects/cuda_blob/data/S_000_1752450056/Tile_r1-c1_S_000_1752450056.tif"
    with cp.cuda.Device(0):
        img = load_tiff(fp)
        kernel = create_embedded_kernel(30, *img.shape)
        kernel_freqs = get_fft(kernel)
        print(kernel_freqs.astype(cp.csingle).dtype)
        # componentwiseMatrixMul1vsBatchfloat2((5,), (5,), (x1, x2, y))  # grid, block and arguments


if __name__ == "__main__":
    main()
