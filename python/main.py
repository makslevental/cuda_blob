import math

import cupy as cp
import numpy as np
from cupy.cuda.cufft import CUFFT_FORWARD
from cupyx.scipy.fftpack import get_fft_plan
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

from ffts import get_fft, get_inverse_fft
from kernels import create_embedded_kernel, componentwise_mult_numba, max_pool_3d, componentwise_mult_numba_depr, \
    componentwise_mult_raw_kernel
from util import (
    load_img,
    get_grid_block_dims,
    prune_blobs,
    make_fig,
    make_hist,
    stretch_composite_histogram,
)


def multi_gpu(fp):
    batch_size = 30
    num_threads = 32
    img = load_img(fp)
    h, w = img.shape

    recv_img = cp.empty(img.shape, dtype="f")
    comm.Scatter(img, recv_img, root=0)

    img_freqs = get_fft(recv_img)

    kernel = create_embedded_kernel(batch_size, h, w).astype(dtype="f")

    recv_kernel = cp.empty((batch_size // size, h, w), dtype="f")
    comm.Scatter(kernel, recv_kernel, root=0)

    kernel_freqs = get_fft(recv_kernel)

    n_blocks_h, n_blocks_w = h // num_threads, w // num_threads
    if n_blocks_w % batch_size or n_blocks_w == 0:
        n_blocks_w += 1
    if n_blocks_h % batch_size or n_blocks_h == 0:
        n_blocks_h += 1

    out = cp.empty_like(kernel_freqs, dtype="f")
    componentwise_mult_numba[
        (n_blocks_w, n_blocks_h, batch_size), (num_threads, num_threads)
    ](img_freqs, kernel_freqs, out, batch_size, h, w)


def single_gpu(
    img: cp.ndarray,
    min_sigma: int = 1,
    max_sigma: int = 10,
    n_sigma_bins: int = 10,
    truncate: float = 4.0,
    threshold: float = 0.0001,
    prune: bool = True,
    overlap: float = 0.5,
    gpu_device: int = 0,
):
    with cp.cuda.Device(gpu_device):
        # move img to GPU
        img = stretch_composite_histogram(img)
        make_fig(img, title="stretched image").show()
        img = cp.asarray(img)

        # calculate sigmas (corresponding to radii)
        img_h, img_w = img.shape
        sigmas = np.linspace(
            min_sigma,
            max_sigma
            + (max_sigma - min_sigma)
            / (
                n_sigma_bins - 1
            ),  # go one increment higher so that we include max_sigma
            n_sigma_bins + 1,
        )
        max_radius = int(truncate * max(sigmas) + 0.5)
        assert max_radius < img_h // 2 and max_radius < img_w // 2

        # only image axes (not batch); this is so that the plans
        # become batch plans for the kernel fft
        axes = (-2, -1)
        kernel = create_embedded_kernel(sigmas, img_h, img_w, truncate).astype(
            dtype="f"
        )
        # for d in kernel:
        #     make_fig(d.get()).show()
        img_forward_plan = get_fft_plan(img, axes=axes, value_type="R2C")
        kernel_forward_plan = get_fft_plan(kernel, axes=axes, value_type="R2C")

        img_freqs = get_fft(img, img_forward_plan)
        kernel_freqs = get_fft(kernel, kernel_forward_plan)
        assert (
            img_freqs.shape[-2:] == kernel_freqs.shape[-2:]
        ), f"img_freqs {img_freqs.shape} kernel_freqs {kernel_freqs.shape}"

        kernel_inverse_plan = get_fft_plan(kernel_freqs, axes=axes, value_type="C2R")

        filtered_imgs_freqs = cp.empty_like(kernel_freqs, dtype=kernel_freqs.dtype)
        grid_block_dims = get_grid_block_dims(*filtered_imgs_freqs.shape)
        componentwise_mult_numba[grid_block_dims](
            img_freqs, kernel_freqs, filtered_imgs_freqs, *filtered_imgs_freqs.shape
        )

        assert kernel_freqs.shape == filtered_imgs_freqs.shape
        filtered_imgs = get_inverse_fft(filtered_imgs_freqs, kernel_inverse_plan)
        assert (
            kernel.shape == filtered_imgs.shape
        ), f"kernel shape {kernel.shape} filtered images shape {filtered_imgs.shape}"

        # for d in filtered_imgs:
        #     make_fig(d.get()).show()

        dog_images = (filtered_imgs[:-1] - filtered_imgs[1:]) * (
            cp.asarray(sigmas[:-1])[cp.newaxis, cp.newaxis, :].T
        )

        # for d in dog_images:
        #     make_fig(d.get()).show()

        local_maxima = max_pool_3d(dog_images)
        # for d in local_maxima:
        #     make_fig(d.get()).show()
        mask = (local_maxima == dog_images) & (dog_images > threshold)
        assert mask.any(), "no maxima - that's bad"
        local_maxima = local_maxima[mask]
        coords = np.asarray(mask.get().nonzero()).T
        # translate final column of cds, which contains the index of the
        # sigma that produced the maximum intensity value, into the sigma
        sigmas_of_peaks = sigmas[coords[:, 0]]
        # Remove sigma index and replace with sigmas
        cds = np.hstack([coords[:, 1:], sigmas_of_peaks[np.newaxis].T])
        print(f"number blobs {len(cds)}")
        if prune:
            blobs = prune_blobs(
                blobs_array=cds,
                overlap=overlap,
                local_maxima=local_maxima.get(),
                sigma_dim=1,
            )

        blobs[:, 2] = blobs[:, 2] * math.sqrt(2)
        return blobs


def main():
    fp = "/home/max/dev_projects/cuda_blob/data/S_000_1752450056/Tile_r1-c1_S_000_1752450056.tif"
    # fp = "/home/max/dev_projects/cuda_blob/python/tests/simulation.png"
    img = load_img(fp, resize=(1024, 1024))
    make_fig(img, title="loaded image").show()
    blobs = single_gpu(img)
    make_hist(blobs[:, 2]).show()
    make_fig(img, blobs).show()


if __name__ == "__main__":
    main()
