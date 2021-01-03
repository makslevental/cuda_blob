import cupy as cp
from cupy.cuda.cufft import CUFFT_FORWARD
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

from ffts import get_fft
from kernels import create_embedded_kernel, componentwise_mult_numba
from util import load_tiff

def multi_gpu(fp):
    batch_size = 30
    num_threads = 32
    img = load_tiff(fp)
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
    componentwise_mult_numba[(n_blocks_w, n_blocks_h, batch_size), (num_threads, num_threads)](
        img_freqs, kernel_freqs, out, batch_size, h, w
    )


def main():
    fp = "/home/max/dev_projects/cuda_blob/data/S_000_1752450056/Tile_r1-c1_S_000_1752450056.tif"
    with cp.cuda.Device(rank):
        multi_gpu(fp)


if __name__ == "__main__":
    main()
