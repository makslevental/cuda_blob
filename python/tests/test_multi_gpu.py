import unittest

import cupy as cp
import numpy as np
from cupyx.scipy.fftpack import get_fft_plan
from mpi4py import MPI

from src.ffts import get_fft
from src.util import print_nd_array

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


# import pydevd_pycharm
# port_mapping = [65300, 65303]
# pydevd_pycharm.settrace(
#     "localhost", port=port_mapping[rank], stdoutToServer=True, stderrToServer=True
# )


class MyTestCase(unittest.TestCase):
    b, h, w = 20, 4096, 4096
    print(f"multi gpu tests with b {b}, h {h}, w {w}")

    def test_basic_mpi(self):
        # Allreduce
        sendbuf = cp.arange(10, dtype="i")
        recvbuf = cp.empty_like(sendbuf)
        comm.Allreduce(sendbuf, recvbuf)
        self.assertTrue(cp.allclose(recvbuf, sendbuf * size))

        # Bcast
        if rank == 0:
            buf = cp.arange(100, dtype=cp.complex64)
        else:
            buf = cp.empty(100, dtype=cp.complex64)
        comm.Bcast(buf)
        self.assertTrue(cp.allclose(buf, cp.arange(100, dtype=cp.complex64)))

        # Send-Recv
        if rank == 0:
            buf = cp.arange(20, dtype=cp.float64)
            comm.Send(buf, dest=1, tag=88)
        else:
            buf = cp.empty(20, dtype=cp.float64)
            comm.Recv(buf, source=0, tag=88)
            self.assertTrue(cp.allclose(buf, cp.arange(20, dtype=cp.float64)))

        # Scatter
        sendbuf = None
        if rank == 0:
            sendbuf = cp.empty((size, 100), dtype="i")
            sendbuf.T[:, :] = cp.arange(size)
        recvbuf = cp.empty(100, dtype="i")
        # root: Rank of sending process (integer).
        comm.Scatter(sendbuf, recvbuf, root=0)
        # element wise comparison ("all close to each other")
        self.assertTrue(cp.allclose(recvbuf, rank))

    def test_scatter_fft(self):
        kernels = [
            (i + 1) * np.ones((self.b // size, self.h, self.w)) for i in range(size)
        ]
        kernel = None
        if rank == 0:
            # mpi4py.MPI.Exception: MPI_ERR_TRUNCATE: message truncated
            # means add dtype
            kernel = cp.asarray(np.stack(kernels), dtype="f")

        recv_kernel = cp.empty((self.b // size, self.h, self.w), dtype="f")
        comm.Scatter(kernel, recv_kernel, root=0)
        kernel_forward_plan = get_fft_plan(recv_kernel, axes=(-2, -1), value_type="R2C")
        kernel_freqs = get_fft(recv_kernel, kernel_forward_plan)

        kernel_freqs_np = np.fft.rfft2(kernels[rank], axes=(-2, -1), norm="ortho")

        try:
            self.assertTrue(cp.allclose(kernel_freqs, kernel_freqs_np, atol=1.0e-6))
        except:
            diff = np.abs(cp.asnumpy(kernel_freqs) - kernel_freqs_np)
            print(diff.min(), diff.max())
            if (self.b, self.h, self.w) <= (2, 100, 100):
                print_nd_array(diff, round=10)
                print_nd_array(kernel_freqs)
                print_nd_array(kernel_freqs_np)
            raise

    def test_gather_fft(self):
        kernels = [
            (i + 1) * np.ones((self.b // size, self.h, self.w)) for i in range(size)
        ]
        kernel = None
        if rank == 0:
            # mpi4py.MPI.Exception: MPI_ERR_TRUNCATE: message truncated
            # means add dtype
            kernel = cp.asarray(np.stack(kernels), dtype="f")

        recv_kernel = cp.empty((self.b // size, self.h, self.w), dtype="f")
        comm.Scatter(kernel, recv_kernel, root=0)
        kernel_forward_plan = get_fft_plan(recv_kernel, axes=(-2, -1), value_type="R2C")
        kernel_freqs = get_fft(recv_kernel, kernel_forward_plan)

        recv_kernel_freqs = None
        if rank == 0:
            recv_kernel_freqs = cp.empty(
                (self.b, self.h, self.w // 2 + 1), dtype="complex64"
            )
        comm.Gather(kernel_freqs, recv_kernel_freqs, root=0)
        if rank == 0:
            kernel_freqs_np = np.fft.rfft2(
                np.concatenate(kernels), axes=(-2, -1), norm="ortho"
            )
            try:
                self.assertTrue(
                    np.allclose(recv_kernel_freqs, kernel_freqs_np, atol=1.0e-5)
                )
            except:
                diff = np.abs(cp.asnumpy(recv_kernel_freqs) - kernel_freqs_np)
                print(diff.min(), diff.max())
                if (self.b, self.h, self.w) <= (2, 100, 100):
                    print_nd_array(diff, round=10)
                    print_nd_array(recv_kernel_freqs)
                    print_nd_array(kernel_freqs_np)
                raise


if __name__ == "__main__":
    unittest.main()
