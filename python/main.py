import cupy as cp
import numpy as np
from PIL import Image, ImageChops
from PIL.Image import BILINEAR
from cupy.cuda.cufft import CUFFT_FORWARD
from cupy.fft._fft import _fftn
from cupyx.scipy.fftpack import get_fft_plan

from util import print_nd_array, gaussian_kernel

DEBUG = True

cp.fft.config.enable_nd_planning = False


def get_kernel_fft(kernel):
    axes = (-2, -1)
    forward_plan = get_fft_plan(kernel, axes=axes, value_type="R2C")
    inverse_plan = get_fft_plan(kernel, axes=axes, value_type="C2R")
    # with s=None the sizes of the input along the `axes` dimensions
    kernel_freqs = _fftn(
        kernel,
        s=None,
        axes=axes,
        norm="ortho",
        direction=CUFFT_FORWARD,
        value_type="R2C",
        order="C",
        plan=forward_plan,
        overwrite_x=False,
    )
    return kernel_freqs, inverse_plan


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


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, scale=2.0, offset=-100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    else:
        return im


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
        kernel_freqs, _ = get_kernel_fft(kernel)


if __name__ == "__main__":
    main()
