import cupy as cp
from numba import cuda

componentwiseMatrixMul1vsBatchfloat2 = cp.RawKernel(
    r"""
extern "C" __global__ void componentwiseMatrixMul1vsBatchfloat2(
    const float2* singleIn,
    const float2* batchIn,
    float2* out,
    int batch_size,
    int rows,
    int cols) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int batch = blockIdx.z;

    size_t yy_xx = row * cols + col;
    size_t zz_yy_xx = batch * (rows * cols) + yy_xx;

    if (batch < batch_size && row < rows && col < cols) {
        out[zz_yy_xx].x =
            singleIn[yy_xx].x * batchIn[zz_yy_xx].x - singleIn[yy_xx].y * batchIn[zz_yy_xx].y;
        out[zz_yy_xx].y =
            singleIn[yy_xx].x * batchIn[zz_yy_xx].y + singleIn[yy_xx].y * batchIn[zz_yy_xx].x;
    }
}""",
    "componentwiseMatrixMul1vsBatchfloat2",
)


@cuda.jit("void(complex64[:], complex64[:], complex64[:], int64, int64, int64)")
def componentwise_mult_depr(single_in, batch_in, out, batch_size, rows, cols):
    col = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    row = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    batch = cuda.blockIdx.z

    yy_xx = row * cols + col
    zz_yy_xx = batch * (rows * cols) + yy_xx
    if batch < batch_size and row < rows and col < cols:
        out[zz_yy_xx] = single_in[yy_xx] * batch_in[zz_yy_xx]


@cuda.jit(
    "void(complex64[:, :], complex64[:, :, :], complex64[:, :, :], uint64, uint64, uint64)"
)
def componentwise_mult(single_in, batch_in, out, batch_size, rows, cols):
    col = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    row = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    batch = cuda.blockIdx.z

    if batch < batch_size and row < rows and col < cols:
        out[batch, row, col] = single_in[row, col] * batch_in[batch, row, col]
        # out[zz_yy_xx].real = (
        #     single_in[yy_xx].real * batch_in[zz_yy_xx].real
        #     - single_in[yy_xx].imag * batch_in[zz_yy_xx].imag
        # )
        # out[zz_yy_xx].imag = (
        #     single_in[yy_xx].real * batch_in[zz_yy_xx].real
        #     + single_in[yy_xx].imag * batch_in[zz_yy_xx].imag
        # )
