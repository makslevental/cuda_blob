import unittest

import cupy as cp
import numpy as np

from cuda_profiling import GPUTimer
from main import (
    get_fft,
    get_inverse_fft,
)
from kernels import componentwiseMatrixMul1vsBatchfloat2, componentwise_mult_depr, componentwise_mult
from util import print_nd_array


class MyTestCase(unittest.TestCase):
    b, h, w = 20, 4096, 4096

    def test_single_gpu_inverse_fft(self):
        with cp.cuda.Device(0):
            kernel = cp.arange(self.b * self.h * self.w).reshape(
                (self.b, self.h, self.w)
            )

            kernel_freqs = get_fft(kernel)
            out = get_inverse_fft(kernel_freqs)

            try:
                diff = np.abs(cp.asnumpy(kernel) - cp.asnumpy(out))
                self.assertTrue(diff.max() < 1e-4)
            except:
                if (self.b, self.h, self.w) < (10, 10, 10):
                    print_nd_array(kernel_freqs)
                    print_nd_array(out)
                else:
                    print(diff.min(), diff.max())
                raise

    def test_single_gpu_fft(self):
        with cp.cuda.Device(0):
            kernel = cp.arange(self.b * self.h * self.w).reshape(
                (self.b, self.h, self.w)
            )
            kernel_freqs = get_fft(kernel)

            kernel_np = np.arange(self.b * self.h * self.w).reshape(
                (self.b, self.h, self.w)
            )
            kernel_freqs_np = np.fft.rfft2(kernel_np, axes=(-2, -1), norm="ortho")

            try:
                diff = np.abs(cp.asnumpy(kernel_freqs) - kernel_freqs_np)
                self.assertTrue(diff.max() < 1e-4)
            except:
                if (self.b, self.h, self.w) < (10, 10, 10):
                    print_nd_array(kernel_freqs)
                    print_nd_array(kernel_freqs_np)
                else:
                    print(diff.min(), diff.max())
                raise

    def test_raw_kernel(self):
        with cp.cuda.Device(0):
            kernel = (
                cp.arange(self.b * self.h * self.w)
                .reshape((self.b, self.h, self.w))
                .astype(cp.complex64)
            )
            img = 2 * cp.ones((self.h, self.w)).astype(cp.complex64)
            out = cp.zeros_like(kernel).astype(cp.complex64)
            with GPUTimer("componentwiseMatrixMul1vsBatchfloat2"):
                n_blocks_h, n_blocks_w = self.h // 32, self.w // 32
                if n_blocks_w % 32:
                    n_blocks_w += 1
                if n_blocks_h % 32:
                    n_blocks_h += 1
                componentwiseMatrixMul1vsBatchfloat2(
                    (n_blocks_w, n_blocks_h, self.b),
                    (32, 32),
                    (img, kernel, out, self.b, self.h, self.w),
                )

            with GPUTimer("broadcast"):
                broadcast = img * kernel

            try:
                diff = np.abs(broadcast - out)
                self.assertTrue(diff.max() < 1e-4)
            except:
                if (self.b, self.h, self.w) < (10, 10, 10):
                    print_nd_array(broadcast)
                    print_nd_array(out)
                else:
                    print(diff.min(), diff.max())
                raise

    def test_numba_kernel_depr(self):
        with cp.cuda.Device(0):
            kernel = (
                cp.arange(self.b * self.h * self.w)
                .reshape((self.b, self.h, self.w))
                .astype(cp.complex64)
            )
            img = 2 * cp.ones((self.h, self.w)).astype(cp.complex64)
            out = cp.zeros_like(kernel).astype(cp.complex64)
            with GPUTimer("numba"):
                n_blocks_h, n_blocks_w = self.h // 32, self.w // 32
                if n_blocks_w % 32 or n_blocks_w == 0:
                    n_blocks_w += 1
                if n_blocks_h % 32 or n_blocks_h == 0:
                    n_blocks_h += 1

                componentwise_mult_depr[(n_blocks_w, n_blocks_h, self.b), (32, 32)](
                    img.ravel(), kernel.ravel(), out.ravel(), self.b, self.h, self.w
                )

            with GPUTimer("broadcast"):
                broadcast = img * kernel

            try:
                diff = np.abs(broadcast - out)
                self.assertTrue(diff.max() < 1e-4)
            except:
                if (self.b, self.h, self.w) < (10, 10, 10):
                    print_nd_array(broadcast)
                    print_nd_array(out)
                else:
                    print(diff.min(), diff.max())
                raise

    def test_numba_kernel(self):
        with cp.cuda.Device(0):
            kernel = (
                cp.arange(self.b * self.h * self.w)
                .reshape((self.b, self.h, self.w))
                .astype(cp.complex64)
            )
            img = 2 * cp.ones((self.h, self.w)).astype(cp.complex64)
            out = cp.zeros_like(kernel).astype(cp.complex64)
            with GPUTimer("numba"):
                n_blocks_h, n_blocks_w = self.h // 32, self.w // 32
                if n_blocks_w % 32 or n_blocks_w == 0:
                    n_blocks_w += 1
                if n_blocks_h % 32 or n_blocks_h == 0:
                    n_blocks_h += 1

                print(self.b, self.h, self.w)
                componentwise_mult[(n_blocks_w, n_blocks_h, self.b), (32, 32)](
                    img, kernel, out, self.b, self.h, self.w
                )

            with GPUTimer("broadcast"):
                broadcast = img * kernel

            try:
                diff = np.abs(broadcast - out)
                self.assertTrue(diff.max() < 1e-4)
            except:
                if (self.b, self.h, self.w) < (10, 10, 10):
                    print_nd_array(broadcast)
                    print_nd_array(out)
                else:
                    print(diff.min(), diff.max())
                raise

    # def test_mpi(self):
    #     comm = MPI.COMM_WORLD
    #     size = comm.Get_size()
    #     rank = comm.Get_rank()
    #
    #     # Allreduce
    #     sendbuf = cp.arange(10, dtype="i")
    #     recvbuf = cp.empty_like(sendbuf)
    #     comm.Allreduce(sendbuf, recvbuf)
    #     assert cp.allclose(recvbuf, sendbuf * size)
    #
    #     # Bcast
    #     if rank == 0:
    #         buf = cp.arange(100, dtype=cp.complex64)
    #     else:
    #         buf = cp.empty(100, dtype=cp.complex64)
    #     comm.Bcast(buf)
    #     assert cp.allclose(buf, cp.arange(100, dtype=cp.complex64))
    #
    #     # Send-Recv
    #     if rank == 0:
    #         buf = cp.arange(20, dtype=cp.float64)
    #         comm.Send(buf, dest=1, tag=88)
    #     else:
    #         buf = cp.empty(20, dtype=cp.float64)
    #         comm.Recv(buf, source=0, tag=88)
    #         assert cp.allclose(buf, cp.arange(20, dtype=cp.float64))


if __name__ == "__main__":
    unittest.main()
