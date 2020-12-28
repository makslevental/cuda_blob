#include <cassert>
#include <cufft.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <tiffio.h>

#include "helper_cuda.h"
#include "stacktrace.h"

#define DEBUG 0

__global__ void componentwiseMatrixMul1vsBatchfloat2(float2* singleIn,
                                                     float2* batchIn,
                                                     float2* out,
                                                     int batch_size,
                                                     int rows,
                                                     int cols) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int batch = blockIdx.z;

    auto yy_xx = row * cols + col;
    auto zz_yy_xx = batch * (rows * cols) + yy_xx;

    if (batch < batch_size && row < rows && col < cols) {
        out[zz_yy_xx].x =
            singleIn[yy_xx].x * batchIn[zz_yy_xx].x - singleIn[yy_xx].y * batchIn[zz_yy_xx].y;
        out[zz_yy_xx].y =
            singleIn[yy_xx].x * batchIn[zz_yy_xx].y + singleIn[yy_xx].y * batchIn[zz_yy_xx].x;
    }
}

void printDevice3Dfloat2(float2* dev_ptr, int batch_size, int width, int height) {
    float2* host_ptr = new float2[height * width * batch_size];
    checkCudaErrors(cudaMemcpy(
        host_ptr, dev_ptr, sizeof(float2) * height * width * batch_size, cudaMemcpyDeviceToHost));
    for (int k = 0; k < batch_size; k++) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                printf("%f + %f i    ",
                       host_ptr[k * (height * width) + i * width + j].x,
                       host_ptr[k * (height * width) + i * width + j].y);
            }
            std::cout << "\n";
        }
        printf("*********************\n");
    }
    delete[] host_ptr;
}

float** gaussianKernel(int width = 21, float sigma = 3.0) {
    float** kernel = new float*[width];
    auto mean = (width - 1) / 2;
    auto norm = 0.0;
    for (int y = 0; y < width; y++) {
        kernel[y] = new float[width];
        for (int x = 0; x < width; x++) {
            kernel[y][x] = (1 / (2 * M_PI * pow(sigma, 2))) *
                           exp(-(pow(x - mean, 2) + pow(y - mean, 2)) / (2 * pow(sigma, 2)));
            norm += kernel[y][x];
        }
    }
    for (int y = 0; y < width; y++) {
        for (int x = 0; x < width; x++) {
            kernel[y][x] /= norm;
        }
    }
#if DEBUG
    for (int y = 0; y < width; y++) {
        for (int x = 0; x < width; x++) {
            kernel[y][x] /= norm;
            printf("%f ", kernel[y][x]);
        }
        std::cout << "\n";
    }
#endif
    return kernel;
}

int main() {
    /* - - - load image using OpenCV - - - */
    auto fp = "/home/max/dev_projects/cuda_blob/data/S_000_1752450056/"
              "Tile_r1-c1_S_000_1752450056.tif";
    cv::Mat img_cv = imread(fp, cv::IMREAD_GRAYSCALE);
    if (img_cv.empty()) {
        std::cout << "Could not read the image: " << fp << std::endl;
        return 1;
    }
    img_cv.convertTo(img_cv, CV_32FC1);
    cv::resize(img_cv, img_cv, cv::Size(8192, 8192));
    auto img_width = img_cv.cols;
    auto img_height = img_cv.rows;
    auto batch_size = 15;
    printf("width %d, height %d\n", img_width, img_height);
    assert(img_cv.channels() == 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    checkCudaErrors(cudaDeviceSynchronize());
    cudaEventRecord(start);

    /* - - - Building the Kernel with 0-padding to the size of the image - - - */
    auto kernel_h = new float[batch_size * img_height * img_width];
    std::fill(kernel_h, kernel_h + batch_size * img_height * img_width, 0);
    int y, x;
    int zz_yy_xx;
    for (int k = 0; k < batch_size; k++) {
        auto radius = k;
        auto kernel = gaussianKernel(2 * radius + 1);
        for (int i = ((img_height / 2) - radius); i <= ((img_height / 2) + radius); i++) {
            for (int j = ((img_width / 2) - radius); j <= ((img_width / 2) + radius); j++) {
                y = i - ((img_height / 2) - radius);
                x = j - ((img_width / 2) - radius);
                zz_yy_xx = k * (img_height * img_width) + i * img_width + j;
                kernel_h[zz_yy_xx] = kernel[y][x];
            }
        }
    }
#if DEBUG
    printf("kernel before \n");
    for (int k = 0; k < batch_size; k++) {
        for (int i = 0; i < img_height; i++) {
            for (int j = 0; j < img_width; j++) {
                printf("%f    ", kernel_h[k * (img_height * img_width) + i * img_width + j]);
            }
            std::cout << "\n";
        }
        printf("*********************\n");
    }
#endif

    /* - - -  ffts - - - */
    // img fft
    cufftHandle img_forward_plan, img_inverse_plan;
    float* img_h = (float*)(img_cv.isContinuous() ? img_cv.data : img_cv.clone().data);
    float* img_d;
    float2* img_freqs_d;

#if DEBUG
    printf("img before \n");
    for (int i = 0; i < img_height; i++) {
        for (int j = 0; j < img_width; j++) {
            printf("%f    ", img_h[i * img_width + j]);
        }
        std::cout << "\n";
    }
#endif

    checkCudaErrors(cufftPlan2d(&img_forward_plan, img_width, img_height, CUFFT_R2C));
    checkCudaErrors(cufftPlan2d(&img_inverse_plan, img_width, img_height, CUFFT_C2R));
    checkCudaErrors(cudaMalloc(&img_d, sizeof(float) * img_height * img_width));
    checkCudaErrors(cudaMalloc(&img_freqs_d, sizeof(float2) * img_height * (img_width / 2 + 1)));
    checkCudaErrors(
        cudaMemcpy(img_d, img_h, sizeof(float) * img_height * img_width, cudaMemcpyHostToDevice));
    checkCudaErrors(cufftExecR2C(img_forward_plan, img_d, img_freqs_d));

#if DEBUG
    printf("img_freqs_d\n");
    printDevice3Dfloat2(img_freqs_d, 1, (img_width / 2 + 1), img_height);
#endif

    // https://forums.developer.nvidia.com/t/cufft-out-of-place-transform-destroys-input/15133
    // something weird going on here?
    // https://github.com/google/jax/issues/2874#issuecomment-622290366
    //#if DEBUG
    //    checkCudaErrors(cufftExecC2R(img_inverse_plan, img_freqs_d, img_d));
    //    checkCudaErrors(
    //        cudaMemcpy(img_h, img_d, sizeof(float) * img_height * img_width,
    //        cudaMemcpyDeviceToHost));
    //    printf("img after\n");
    //    for (int i = 0; i < img_height; i++) {
    //        for (int j = 0; j < img_width; j++) {
    //            printf("%f    ", (1.0 / (img_width * img_height)) * abs(img_h[i * img_width +
    //            j]));
    //        }
    //        std::cout << "\n";
    //    }
    //#endif

    // kernel batch fft
    cufftHandle kernel_forward_plan, kernel_inverse_plan;
    // dimension of fft
    int rank = 2;
    int n[2] = {img_height, img_width};
    // dist between batches
    int idist = img_height * img_width;
    int odist = img_height * (img_width / 2 + 1);
    // input/output sizes with pitches ("unpitched")
    int inembed[] = {img_height, img_width};
    int onembed[] = {img_height, img_width / 2 + 1};
    // stride between adjacent entries in row
    int istride = 1;
    int ostride = 1;
    float* kernel_d;
    float2* kernel_freqs_d;

    checkCudaErrors(cufftPlanMany(&kernel_forward_plan,
                                  rank,
                                  n,
                                  inembed,
                                  istride,
                                  idist,
                                  onembed,
                                  ostride,
                                  odist,
                                  CUFFT_R2C,
                                  batch_size));
    checkCudaErrors(cufftPlanMany(&kernel_inverse_plan,
                                  rank,
                                  n,
                                  onembed,
                                  ostride,
                                  odist,
                                  inembed,
                                  istride,
                                  idist,
                                  CUFFT_C2R,
                                  batch_size));

    checkCudaErrors(cudaMalloc(&kernel_d, sizeof(float) * img_height * img_width * batch_size));
    checkCudaErrors(cudaMemcpy(kernel_d,
                               kernel_h,
                               sizeof(float) * img_height * img_width * batch_size,
                               cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&kernel_freqs_d,
                               sizeof(float2) * img_height * (img_width / 2 + 1) * batch_size));
    checkCudaErrors(cufftExecR2C(kernel_forward_plan, kernel_d, kernel_freqs_d));
    checkCudaErrors(cudaGetLastError());

#if DEBUG
    printf("kernel_freqs_d\n");
    printDevice3Dfloat2(kernel_freqs_d, batch_size, (img_width / 2 + 1), img_height);
#endif

    // https://forums.developer.nvidia.com/t/cufft-out-of-place-transform-destroys-input/15133
    // something weird going on here?
    // https://github.com/google/jax/issues/2874#issuecomment-622290366
    // CUFFT has the same behavior as FFTW, it computes unnormalized FFTs.
    // https://stackoverflow.com/a/6460822
    //#if DEBUG
    //    checkCudaErrors(cufftExecC2R(kernel_inverse_plan, kernel_freqs_d, kernel_d));
    //    checkCudaErrors(cudaMemcpy(kernel_h,
    //                               kernel_d,
    //                               sizeof(float) * img_height * img_width * batch_size,
    //                               cudaMemcpyDeviceToHost));
    //    printf("kernel after \n");
    //    for (int k = 0; k < batch_size; k++) {
    //        for (int i = 0; i < img_height; i++) {
    //            for (int j = 0; j < img_width; j++) {
    //                printf("%f    ",
    //                       (1.0 / (img_width * img_height)) *
    //                           abs(kernel_h[k * (img_height * img_width) + i * img_width + j]));
    //            }
    //            std::cout << "\n";
    //        }
    //        printf("*********************\n");
    //    }
    //#endif

    auto num_threads = 32;
    dim3 dim_block(num_threads, num_threads);
    int n_blocks_w = (img_width / 2 + 1) / num_threads;
    if (((img_width / 2 + 1) % num_threads) != 0) n_blocks_w++;
    int n_blocks_h = img_height / num_threads;
    if ((img_height % num_threads) != 0) n_blocks_h++;
    dim3 dim_grid(n_blocks_w, n_blocks_h, batch_size);

    float2* filtered_d;
    checkCudaErrors(
        cudaMalloc(&filtered_d, sizeof(float2) * batch_size * (img_width / 2 + 1) * img_height));

    /* element-wise matrix-mul */
    componentwiseMatrixMul1vsBatchfloat2<<<dim_grid, dim_block>>>(
        img_freqs_d, kernel_freqs_d, filtered_d, batch_size, (img_width / 2 + 1), img_height);
    checkCudaErrors(cudaGetLastError());

    cudaEventRecord(stop);
    float milliseconds = 0;
    checkCudaErrors(cudaDeviceSynchronize());
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("copy running time %.10f\n", milliseconds);

#if DEBUG
    checkCudaErrors(cudaDeviceSynchronize());
    printf("filtered_d\n");
    printDevice3Dfloat2(filtered_d, batch_size, (img_width / 2 + 1), img_height);
#endif
}