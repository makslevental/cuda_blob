import unittest

import cupy as cp
import numpy as np
from mpi4py import MPI

from main import get_kernel_fft
from util import print_nd_array


class MyTestCase(unittest.TestCase):
    def test_single_gpu_ffts(self):
        with cp.cuda.Device(0):
            # img = load_tiff(fp)
            # kernel = create_embedded_kernel(30, *img.shape)
            kernel = cp.arange(2 * 10 * 10).reshape((2, 10, 10))
            print_nd_array(kernel)

            kernel_freqs, _ = get_kernel_fft(kernel)

            kernel_np = np.arange(2 * 10 * 10).reshape((2, 10, 10))
            kernel_freqs_np = np.fft.rfft2(kernel_np, axes=(-2, -1), norm="ortho")
            print_nd_array(kernel_freqs_np)
            print_nd_array(kernel_freqs)

            self.assertTrue(
                np.linalg.norm(cp.asnumpy(kernel_freqs) - kernel_freqs_np).sum() < 1e-6
            )

    def test_mpi(self):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        # Allreduce
        sendbuf = cp.arange(10, dtype="i")
        recvbuf = cp.empty_like(sendbuf)
        comm.Allreduce(sendbuf, recvbuf)
        assert cp.allclose(recvbuf, sendbuf * size)

        # Bcast
        if rank == 0:
            buf = cp.arange(100, dtype=cp.complex64)
        else:
            buf = cp.empty(100, dtype=cp.complex64)
        comm.Bcast(buf)
        assert cp.allclose(buf, cp.arange(100, dtype=cp.complex64))

        # Send-Recv
        if rank == 0:
            buf = cp.arange(20, dtype=cp.float64)
            comm.Send(buf, dest=1, tag=88)
        else:
            buf = cp.empty(20, dtype=cp.float64)
            comm.Recv(buf, source=0, tag=88)
            assert cp.allclose(buf, cp.arange(20, dtype=cp.float64))


if __name__ == "__main__":
    unittest.main()
