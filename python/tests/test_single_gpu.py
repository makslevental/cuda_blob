import unittest

import cupy as cp
import numpy as np
from cupyx.scipy.fftpack import get_fft_plan

from src.profiling import GPUTimer
from src.ffts import get_fft, get_inverse_fft
from src.kernels import componentwise_mult_raw_kernel, componentwise_mult_numba_depr, componentwise_mult_numba
from src.util import print_nd_array


class MyTestCase(unittest.TestCase):
    b, h, w = 20, 4096, 4096
    print(f"single gpu tests with b {b}, h {h}, w {w}")

    def test_single_gpu_inverse_fft(self):
        with cp.cuda.Device(0):
            kernel = cp.arange(self.b * self.h * self.w).reshape(
                (self.b, self.h, self.w)
            )

            kernel_forward_plan = get_fft_plan(kernel, axes=(-2, -1), value_type="R2C")
            kernel_freqs = get_fft(kernel, kernel_forward_plan)
            kernel_inverse_plan = get_fft_plan(kernel_freqs, axes=(-2, -1), value_type="C2R")
            out = get_inverse_fft(kernel_freqs, kernel_inverse_plan)

            try:
                self.assertTrue(cp.allclose(kernel, out))
            except:
                if (self.b, self.h, self.w) < (10, 10, 10):
                    print_nd_array(kernel_freqs)
                    print_nd_array(out)
                else:
                    diff = np.abs(cp.asnumpy(kernel) - cp.asnumpy(out))
                    print(diff.min(), diff.max())
                raise

    def test_single_gpu_fft(self):
        with cp.cuda.Device(0):
            kernel = cp.arange(self.b * self.h * self.w).reshape(
                (self.b, self.h, self.w)
            )
            kernel_forward_plan = get_fft_plan(kernel, axes=(-2, -1), value_type="R2C")
            kernel_freqs = get_fft(kernel, kernel_forward_plan)

            kernel_np = np.arange(self.b * self.h * self.w).reshape(
                (self.b, self.h, self.w)
            )
            kernel_freqs_np = np.fft.rfft2(kernel_np, axes=(-2, -1), norm="ortho")

            try:
                self.assertTrue(cp.allclose(kernel_freqs, kernel_freqs_np))
            except:
                if (self.b, self.h, self.w) < (10, 10, 10):
                    print_nd_array(kernel_freqs)
                    print_nd_array(kernel_freqs_np)
                else:
                    diff = np.abs(cp.asnumpy(kernel_freqs) - kernel_freqs_np)
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
                componentwise_mult_raw_kernel(
                    (n_blocks_w, n_blocks_h, self.b),
                    (32, 32),
                    (img, kernel, out, self.b, self.h, self.w),
                )

            with GPUTimer("broadcast"):
                broadcast = img * kernel

            try:
                self.assertTrue(cp.allclose(broadcast, out))
            except:
                if (self.b, self.h, self.w) < (10, 10, 10):
                    print_nd_array(broadcast)
                    print_nd_array(out)
                else:
                    diff = np.abs(broadcast - out)
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

                componentwise_mult_numba_depr[
                    (n_blocks_w, n_blocks_h, self.b), (32, 32)
                ](img.ravel(), kernel.ravel(), out.ravel(), self.b, self.h, self.w)

            with GPUTimer("broadcast"):
                broadcast = img * kernel

            try:
                self.assertTrue(cp.allclose(broadcast, out))
            except:
                if (self.b, self.h, self.w) < (10, 10, 10):
                    print_nd_array(broadcast)
                    print_nd_array(out)
                else:
                    diff = np.abs(broadcast - out)
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

                componentwise_mult_numba[(n_blocks_w, n_blocks_h, self.b), (32, 32)](
                    img, kernel, out, self.b, self.h, self.w
                )

            with GPUTimer("broadcast"):
                broadcast = img * kernel

            try:
                self.assertTrue(cp.allclose(broadcast, out))
            except:
                if (self.b, self.h, self.w) < (10, 10, 10):
                    print_nd_array(broadcast)
                    print_nd_array(out)
                else:
                    diff = np.abs(broadcast - out)
                    print(diff.min(), diff.max())
                raise


if __name__ == "__main__":
    unittest.main()
