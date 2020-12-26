#include <iostream>
#include <tiffio.h>
#include <opencv2/opencv.hpp>
#include <cufft.h>

#include "stacktrace.h"
#include "helper_cuda.h"
#include "util.h"

__global__ void KernelFFTShift2D(cufftDoubleComplex* IM, int im_height, int im_width);

__global__ void
ComponentwiseMatrixMul(cufftDoubleComplex* in1, cufftDoubleComplex* in2, cufftDoubleComplex* out, int row, int col);

__global__ void
ZeroPadding(cufftDoubleComplex* F, cufftDoubleComplex* FP, int newCols, int newRows, int oldCols, int oldRows);


__global__ void
ZeroPadding(cufftDoubleComplex* F, cufftDoubleComplex* FP, int newCols, int newRows, int oldCols, int oldRows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int ind = idx * newCols + idy;

    if (idx < newRows && idy < newCols) {
        if (idx < oldRows && idy < oldCols) {
            FP[ind].x = F[idx * oldCols + idy].x;
        } else if (idx > oldRows || idy > oldCols) {
            FP[ind].x = 0;
        }
    }
}


__global__ void
ComponentwiseMatrixMul(cufftDoubleComplex* in1, cufftDoubleComplex* in2, cufftDoubleComplex* out, int row, int col) {
    int indexRow = threadIdx.x + blockIdx.x * blockDim.x;
    int indexCol = threadIdx.y + blockIdx.y * blockDim.y;
    if (indexRow < row && indexCol < col) {
        out[indexRow * col + indexCol].x = in1[indexRow * col + indexCol].x * in2[indexRow * col + indexCol].x;
        out[indexRow * col + indexCol].y = in1[indexRow * col + indexCol].y * in2[indexRow * col + indexCol].y;
    }
}


__global__ void KernelFFTShift2D(cufftDoubleComplex* IM, int im_height, int im_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int ind = idy * im_width + idx;
    int x, y, indshift;
    cufftDoubleComplex v;

    if (idx < im_width && idy < im_height / 2) {
        if (idx < im_width / 2 && idy < im_height / 2) {
            x = idx + im_width / 2;
            y = idy + im_height / 2;
        } else if (idx >= im_width / 2 && idy < im_height / 2) {
            x = idx - im_width / 2;
            y = idy + im_height / 2;
        }

        indshift = y * im_width + x;
        v.x = IM[ind].x;
        v.y = IM[ind].y;

        IM[ind].x = IM[indshift].x;
        IM[ind].y = IM[indshift].y;

        IM[indshift].x = v.x;
        IM[indshift].y = v.y;
    }
}

int main(int argc, char* argv[]) {
    auto fp = "/home/max/dev_projects/cuda_blob/data/S_000_1752450056/Tile_r1-c1_S_000_1752450056.tif";
    cv::Mat cv_img = imread(fp, cv::IMREAD_GRAYSCALE);
    if (cv_img.empty()) {
        std::cout << "Could not read the image: " << fp << std::endl;
        return 1;
    }
    cv::resize(cv_img, cv_img, cv::Size(8092, 8092));

    auto im_width = cv_img.cols;
    auto im_height = cv_img.rows;
    printf("width %d, height %d", im_width, im_height);
    auto im_channels = cv_img.channels();
    assert(im_channels == 1);

    auto cufft_img = new cufftDoubleComplex* [im_height];
    for (int i = 0; i < im_height; i++) {
        cufft_img[i] = new cufftDoubleComplex[im_width];
        for (int j = 0; j < im_width; j++) {
            cufft_img[i][j].x = (double) cv_img.at<uchar>(j, i);
            cufft_img[i][j].y = 0;
        }
    }

    auto kernel_file = OpenFile("/home/max/dev_projects/cuda_blob/Kernel51.txt");

    /* - - - Building the Kernel with 0-padding - - - */
    auto cufft_padded_kernel = new cufftDoubleComplex* [im_height];
    for (int i = 0; i < im_height; i++) {
        cufft_padded_kernel[i] = new cufftDoubleComplex[im_width];
        for (int j = 0; j < im_width; j++) {
            if (
                    (i >= ((im_height / 2) - 2)) && // +/- 2 gives the cufft_padded_kernel width
                    (i <= ((im_height / 2) + 2)) &&
                    (j >= ((im_width / 2) - 2)) &&
                    (j <= ((im_width / 2) + 2))
                    ) {
                assert(kernel_file >> cufft_padded_kernel[i][j].x);
                cufft_padded_kernel[i][j].y = 0.0;
            } else {
                cufft_padded_kernel[i][j].x = 0.0;
                cufft_padded_kernel[i][j].y = 0.0;
            }
        }
    }

    cufftDoubleComplex* cufft_img_d;
    cufftDoubleComplex* cufft_padded_kernel_d;
    cufftDoubleComplex* cufft_ffted_img_d;
    cufftDoubleComplex* cufft_ffted_padded_kernel_d;

    checkCudaErrors(cudaMalloc((void**) &cufft_img_d, im_width * im_height * sizeof(cufftDoubleComplex)));
    checkCudaErrors(cudaMalloc((void**) &cufft_padded_kernel_d, im_width * im_height * sizeof(cufftDoubleComplex)));
    checkCudaErrors(cudaMalloc((void**) &cufft_ffted_img_d, im_width * im_height * sizeof(cufftDoubleComplex)));
    checkCudaErrors(
            cudaMalloc((void**) &cufft_ffted_padded_kernel_d, im_width * im_height * sizeof(cufftDoubleComplex)));

    /* --- Copying image and cufft_padded_kernel on device --- */
    for (int i = 0; i < im_height; ++i) {
        cudaMemcpy2D(
                cufft_img_d + i * im_width,
                sizeof(cufftDoubleComplex),
                cufft_img[i],
                sizeof(cufftDoubleComplex),
                sizeof(cufftDoubleComplex),
                im_width,
                cudaMemcpyHostToDevice
        );
    }

    for (int i = 0; i < im_height; ++i) {
        cudaMemcpy2D(
                cufft_padded_kernel_d + i * im_width,
                sizeof(cufftDoubleComplex),
                cufft_padded_kernel[i],
                sizeof(cufftDoubleComplex),
                sizeof(cufftDoubleComplex),
                im_width,
                cudaMemcpyHostToDevice
        );
    }

    auto num_threads = 32;
    dim3 dim_block(num_threads, num_threads);
    int n_blocks_w = im_width / num_threads;
    if ((im_width % num_threads) != 0)
        n_blocks_w++;
    int n_blocks_h = im_height / num_threads;
    if ((im_height % num_threads) != 0)
        n_blocks_h++;
    dim3 dim_grid(n_blocks_w, n_blocks_h);

    /* Creating plans */
    cufftHandle plan_img_Z2Z, plan_inv_Z2Z, plan_kernel_Z2Z;
    auto cufft_type = CUFFT_Z2Z;
    checkCudaErrors(cufftPlan2d(&plan_img_Z2Z, im_width, im_height, cufft_type));
    checkCudaErrors(cufftPlan2d(&plan_kernel_Z2Z, im_width, im_height, cufft_type));
    checkCudaErrors(cufftPlan2d(&plan_inv_Z2Z, im_width, im_height, cufft_type));

    /* - - - Fast Fourier Transform on image - - - */
    checkCudaErrors(cufftExecZ2Z(plan_img_Z2Z, cufft_img_d, cufft_ffted_img_d, CUFFT_FORWARD));
    KernelFFTShift2D<<<dim_grid, dim_block>>>(cufft_ffted_img_d, im_height, im_width);

    /* - - - Fast Fourier Transform on kernel - - - */
    checkCudaErrors(cufftExecZ2Z(plan_kernel_Z2Z, cufft_padded_kernel_d, cufft_ffted_padded_kernel_d, CUFFT_FORWARD));
    KernelFFTShift2D<<<dim_grid, dim_block>>>(cufft_ffted_padded_kernel_d, im_height, im_width);

    /* element-wise matrix-mul */
    ComponentwiseMatrixMul<<<dim_grid, dim_block>>>(
            cufft_ffted_img_d,
            cufft_ffted_padded_kernel_d,
            cufft_ffted_img_d,
            im_height,
            im_width
    );


    /* - - - Executing IFFT and shifting back - - - */
    KernelFFTShift2D<<<dim_grid, dim_block>>>(cufft_ffted_img_d, im_height, im_width);
    checkCudaErrors(cufftExecZ2Z(plan_inv_Z2Z, cufft_ffted_img_d, cufft_img_d, CUFFT_INVERSE));
    KernelFFTShift2D<<<dim_grid, dim_block>>>(cufft_img_d, im_height, im_width);

    /* - - - Generating output - - - */
    auto c = (cufftDoubleComplex*) malloc(im_width * im_height * sizeof(cufftDoubleComplex));
    cudaMemcpy(c, cufft_img_d, sizeof(cufftDoubleComplex) * im_height * im_width, cudaMemcpyDeviceToHost);

    long double max = c[0].x;
    for (int i = 0; i < im_height; i++) {
        for (int j = 0; j < im_width; j++) {
            if (c[i * im_width + j].x > max)
                max = c[i * im_width + j].x;
        }
    }
    cv_img.convertTo(cv_img, CV_64F);
    for (int i = 0; i < im_height; i++) {
        for (int j = 0; j < im_width; j++) {
            cv_img.at<double>(j, i) = floor((c[i * im_width + j].x / max) * 255);
        }
    }
    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
    compression_params.push_back(9);
    imwrite("/home/max/dev_projects/cuda_blob/data/output_image.jpg", cv_img, compression_params);

    return 0;
}