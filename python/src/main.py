import math
import os
from glob import glob

import cupy as cp
import numpy as np
import pandas as pd
from cupy.cuda.cufft import CUFFT_FORWARD
from mpi4py import MPI

from src.plots import plot_focus_res

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

from ffts import get_fft, get_forward_plans, filter_imgs
from kernels import (
    create_embedded_kernel,
    get_local_maxima,
)
from util import (
    load_img,
    prune_blobs,
    stretch_composite_histogram,
    get_sigmas,
)


def multi_gpu(
    img: cp.ndarray,
    min_sigma: int = 1,
    max_sigma: int = 10,
    n_sigma_bins: int = 10,
    truncate: float = 4.0,
    threshold: float = 0.0001,
    prune: bool = True,
    overlap: float = 0.5,
):
    # move img to GPU
    img = cp.asarray(img)
    img_h, img_w = img.shape
    # scatter img
    recv_img = cp.empty(img.shape, dtype="f")
    comm.Scatter(img, recv_img, root=0)

    if rank == 0:
        sigmas = get_sigmas(img_h, img_w, min_sigma, max_sigma, n_sigma_bins, truncate)
        assert len(sigmas) % size == 0, f"len(sigmas) {len(sigmas)} size {size}"
    else:
        sigmas = np.empty(n_sigma_bins + 1)

    # scatter sigmas
    recv_sigmas = np.empty(len(sigmas) // size, dtype="float64")
    comm.Scatter(sigmas, recv_sigmas, root=0)

    kernel = create_embedded_kernel(recv_sigmas, img_h, img_w, truncate).astype(
        dtype="f"
    )

    img_forward_plan, kernel_forward_plan = get_forward_plans(recv_img, kernel)
    img_freqs = get_fft(recv_img, img_forward_plan)

    kernel_freqs = get_fft(kernel, kernel_forward_plan)
    assert (
        img_freqs.shape[-2:] == kernel_freqs.shape[-2:]
    ), f"img_freqs {img_freqs.shape} kernel_freqs {kernel_freqs.shape}"

    filtered_imgs = filter_imgs(img_freqs, kernel_freqs)
    assert (
        kernel.shape == filtered_imgs.shape
    ), f"kernel shape {kernel.shape} filtered images shape {filtered_imgs.shape}"

    # gather to root
    recv_filtered_imgs = cp.empty((n_sigma_bins + 1, img_h, img_w), dtype="f")
    comm.Gather(filtered_imgs, recv_filtered_imgs, root=0)

    if rank == 0:
        dog_images = (recv_filtered_imgs[:-1] - recv_filtered_imgs[1:]) * (
            cp.asarray(sigmas[:-1])[cp.newaxis, cp.newaxis, :].T
        )

        blob_params, local_maxima = get_local_maxima(dog_images, sigmas, threshold)
        if prune:
            blobs = prune_blobs(
                blobs_array=blob_params,
                overlap=overlap,
                local_maxima=local_maxima.get(),
                sigma_dim=1,
            )
        else:
            blobs = blob_params

        blobs[:, 2] = blobs[:, 2] * math.sqrt(2)
        return blobs
    else:
        return None


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
        img = cp.asarray(img)
        img_h, img_w = img.shape

        sigmas = get_sigmas(img_h, img_w, min_sigma, max_sigma, n_sigma_bins, truncate)
        kernel = create_embedded_kernel(sigmas, img_h, img_w, truncate).astype(
            dtype="f"
        )

        img_forward_plan, kernel_forward_plan = get_forward_plans(img, kernel)
        img_freqs = get_fft(img, img_forward_plan)
        kernel_freqs = get_fft(kernel, kernel_forward_plan)
        assert (
            img_freqs.shape[-2:] == kernel_freqs.shape[-2:]
        ), f"img_freqs {img_freqs.shape} kernel_freqs {kernel_freqs.shape}"

        filtered_imgs = filter_imgs(img_freqs, kernel_freqs)
        assert (
            kernel.shape == filtered_imgs.shape
        ), f"kernel shape {kernel.shape} filtered images shape {filtered_imgs.shape}"
        dog_images = (filtered_imgs[:-1] - filtered_imgs[1:]) * (
            cp.asarray(sigmas[:-1])[cp.newaxis, cp.newaxis, :].T
        )

        blob_params, local_maxima = get_local_maxima(dog_images, sigmas, threshold)
        if prune:
            blobs = prune_blobs(
                blobs_array=blob_params,
                overlap=overlap,
                local_maxima=local_maxima.get(),
                sigma_dim=1,
            )
        else:
            blobs = blob_params

        blobs[:, 2] = blobs[:, 2] * math.sqrt(2)
        return blobs


def main():
    # fp = "/home/max/dev_projects/cuda_blob/data/S_000_1752450056/Tile_r1-c1_S_000_1752450056.tif"
    # fp = "/home/max/dev_projects/cuda_blob/python/tests/simulation.png"
    # for img_fp in glob("/home/max/dev_projects/cuda_blob/data/*/*/*.tif")[:10]:
    df = pd.read_csv("/home/max/dev_projects/cuda_blob/python/tests/image_focus.csv")
    for img_fp in glob(
        "/home/max/dev_projects/mouse_brain_data/focus_series/*/*/*.tif"
    ):
        section_name = os.path.split(os.path.split(img_fp)[0])[1]
        img = load_img(img_fp, resize=(1024, 1024))
        # make_fig(img, title=f"{section_name} loaded image").savefig(
        #     f"/home/max/dev_projects/cuda_blob/data/results/{section_name} loaded image.png"
        # )
        img = stretch_composite_histogram(img)
        # make_fig(img, title=f"{section_name} stretched image").savefig(
        #     f"/home/max/dev_projects/cuda_blob/data/results/{section_name} stretched image.png"
        # )
        blobs = single_gpu(img, n_sigma_bins=29, max_sigma=30)
        # make_hist(
        #     blobs[:, 2], title=f"{section_name} hist #blobs {len(blobs)}"
        # ).savefig(
        #     f"/home/max/dev_projects/cuda_blob/data/results/{section_name} hist #blobs {len(blobs)}.png"
        # )
        # make_fig(img, blobs, title=f"{section_name} blobs #blobs {len(blobs)}").savefig(
        #     f"/home/max/dev_projects/cuda_blob/data/results/{section_name} blobs #blobs {len(blobs)}.png"
        # )
        if rank == 0:
            focus = df[df["image name"] == int(section_name[-4:])][
                "focal depth (mm)"
            ].values[0]
            print(f"focus {focus}")
            # make_fig(
            #     img, blobs, title=f"{section_name} blobs #blobs {len(blobs)} focus {focus}"
            # ).show()


if __name__ == "__main__":
    # main()
    plot_focus_res()
