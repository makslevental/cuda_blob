import cupy as cp
from cupy.cuda.cufft import CUFFT_FORWARD

from ffts import get_fft
from kernels import create_embedded_kernel
from util import load_tiff


def main():
    fp = "/home/max/dev_projects/cuda_blob/data/S_000_1752450056/Tile_r1-c1_S_000_1752450056.tif"
    with cp.cuda.Device(0):
        img = load_tiff(fp)
        kernel = create_embedded_kernel(30, *img.shape)
        kernel_freqs = get_fft(kernel)
        print(kernel_freqs.astype(cp.csingle).dtype)


if __name__ == "__main__":
    main()
