#include <cassert>
#include <cufft.h>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <tiffio.h>

#include "helper_cuda.h"
#include "stacktrace.h"
#include "util.h"

//__global__ void
// KernelFFTShift2D(cufftDoubleComplex* IM, int im_height, int im_width);
//
//__global__ void ComponentwiseMatrixMul(cufftDoubleComplex* in1,
//                                       cufftDoubleComplex* in2,
//                                       cufftDoubleComplex* out,
//                                       int row,
//                                       int col);
//
//__global__ void ZeroPadding(cufftDoubleComplex* F,
//                            cufftDoubleComplex* FP,
//                            int newCols,
//                            int newRows,
//                            int oldCols,
//                            int oldRows);
//
//__global__ void ZeroPadding(cufftDoubleComplex* F,
//                            cufftDoubleComplex* FP,
//                            int newCols,
//                            int newRows,
//                            int oldCols,
//                            int oldRows) {
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    int idy = blockIdx.y * blockDim.y + threadIdx.y;
//    int ind = idx * newCols + idy;
//
//    if (idx < newRows && idy < newCols) {
//        if (idx < oldRows && idy < oldCols) {
//            FP[ind].x = F[idx * oldCols + idy].x;
//        } else if (idx > oldRows || idy > oldCols) {
//            FP[ind].x = 0;
//        }
//    }
//}
//
//__global__ void ComponentwiseMatrixMul(cufftDoubleComplex* in1,
//                                       cufftDoubleComplex* in2,
//                                       cufftDoubleComplex* out,
//                                       int row,
//                                       int col) {
//    int indexRow = threadIdx.x + blockIdx.x * blockDim.x;
//    int indexCol = threadIdx.y + blockIdx.y * blockDim.y;
//    if (indexRow < row && indexCol < col) {
//        out[indexRow * col + indexCol].x =
//            in1[indexRow * col + indexCol].x * in2[indexRow * col +
//            indexCol].x;
//        out[indexRow * col + indexCol].y =
//            in1[indexRow * col + indexCol].y * in2[indexRow * col +
//            indexCol].y;
//    }
//}
//
//__global__ void
// KernelFFTShift2D(cufftDoubleComplex* IM, int im_height, int im_width) {
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    int idy = blockIdx.y * blockDim.y + threadIdx.y;
//    int ind = idy * im_width + idx;
//    int x, y, indshift;
//    cufftDoubleComplex v;
//
//    if (idx < im_width && idy < im_height / 2) {
//        if (idx < im_width / 2 && idy < im_height / 2) {
//            x = idx + im_width / 2;
//            y = idy + im_height / 2;
//        } else if (idx >= im_width / 2 && idy < im_height / 2) {
//            x = idx - im_width / 2;
//            y = idy + im_height / 2;
//        }
//
//        indshift = y * im_width + x;
//        v.x = IM[ind].x;
//        v.y = IM[ind].y;
//
//        IM[ind].x = IM[indshift].x;
//        IM[ind].y = IM[indshift].y;
//
//        IM[indshift].x = v.x;
//        IM[indshift].y = v.y;
//    }
//}
//
// int main(int argc, char* argv[]) {
//    auto fp = "/home/max/dev_projects/cuda_blob/data/S_000_1752450056/"
//              "Tile_r1-c1_S_000_1752450056.tif";
//    cv::Mat cv_img = imread(fp, cv::IMREAD_GRAYSCALE);
//    if (cv_img.empty()) {
//        std::cout << "Could not read the image: " << fp << std::endl;
//        return 1;
//    }
//    cv::resize(cv_img, cv_img, cv::Size(8092, 8092));
//
//    auto im_width = cv_img.cols;
//    auto im_height = cv_img.rows;
//    auto batch_size = 10;
//    printf("width %d, height %d", im_width, im_height);
//    auto im_channels = cv_img.channels();
//    assert(im_channels == 1);
//
//    auto cufft_img = new cufftDoubleComplex**[batch_size];
//    for (int k = 0; k < batch_size; k++) {
//        cufft_img[k] = new cufftDoubleComplex*[im_height];
//        for (int i = 0; i < im_height; i++) {
//            cufft_img[k][i] = new cufftDoubleComplex[im_width];
//            for (int j = 0; j < im_width; j++) {
//                cufft_img[k][i][j].x = (double)cv_img.at<uchar>(j, i);
//                cufft_img[k][i][j].y = 0;
//            }
//        }
//    }
//
//    auto kernel_file =
//        openFile("/home/max/dev_projects/cuda_blob/Kernel51.txt");
//
//    /* - - - Building the Kernel with 0-padding - - - */
//    auto cufft_padded_kernel = new cufftDoubleComplex**[batch_size];
//    for (int k = 0; k < batch_size; k++) {
//        cufft_padded_kernel[k] = new cufftDoubleComplex*[im_height];
//        for (int i = 0; i < im_height; i++) {
//            cufft_padded_kernel[k][i] = new cufftDoubleComplex[im_width];
//            for (int j = 0; j < im_width; j++) {
//                if ((i >= ((im_height / 2) - 2)) &&
//                    (i <= ((im_height / 2) + 2)) &&
//                    (j >= ((im_width / 2) - 2)) &&
//                    (j <= ((im_width / 2) + 2))) {
//                    assert(kernel_file >> cufft_padded_kernel[k][i][j].x);
//                    cufft_padded_kernel[k][i][j].y = 0.0;
//                } else {
//                    cufft_padded_kernel[k][i][j].x = 0.0;
//                    cufft_padded_kernel[k][i][j].y = 0.0;
//                }
//            }
//        }
//    }
//
//    cufftDoubleComplex* cufft_img_d;
//    cufftDoubleComplex* cufft_padded_kernel_d;
//    cufftDoubleComplex* cufft_ffted_img_d;
//    cufftDoubleComplex* cufft_ffted_padded_kernel_d;
//
//    checkCudaErrors(cudaMalloc((void**)&cufft_img_d,
//                               batch_size * im_width * im_height *
//                                   sizeof(cufftDoubleComplex)));
//    checkCudaErrors(cudaMalloc((void**)&cufft_padded_kernel_d,
//                               batch_size * im_width * im_height *
//                                   sizeof(cufftDoubleComplex)));
//    checkCudaErrors(cudaMalloc((void**)&cufft_ffted_img_d,
//                               batch_size * im_width * im_height *
//                                   sizeof(cufftDoubleComplex)));
//    checkCudaErrors(cudaMalloc((void**)&cufft_ffted_padded_kernel_d,
//                               batch_size * im_width * im_height *
//                                   sizeof(cufftDoubleComplex)));
//
//    /* --- Copying image and cufft_padded_kernel on device --- */
//    for (int i = 0; i < im_height; ++i) {
//        cudaMemcpy2D(cufft_img_d + i * im_width,
//                     sizeof(cufftDoubleComplex),
//                     cufft_img[i],
//                     sizeof(cufftDoubleComplex),
//                     sizeof(cufftDoubleComplex),
//                     im_width,
//                     cudaMemcpyHostToDevice);
//    }
//
//    for (int i = 0; i < im_height; ++i) {
//        cudaMemcpy2D(cufft_padded_kernel_d + i * im_width,
//                     sizeof(cufftDoubleComplex),
//                     cufft_padded_kernel[i],
//                     sizeof(cufftDoubleComplex),
//                     sizeof(cufftDoubleComplex),
//                     im_width,
//                     cudaMemcpyHostToDevice);
//    }
//
//    auto num_threads = 32;
//    dim3 dim_block(num_threads, num_threads);
//    int n_blocks_w = im_width / num_threads;
//    if ((im_width % num_threads) != 0) n_blocks_w++;
//    int n_blocks_h = im_height / num_threads;
//    if ((im_height % num_threads) != 0) n_blocks_h++;
//    dim3 dim_grid(n_blocks_w, n_blocks_h);
//
//    /* Creating plans */
//    cufftHandle plan_img_Z2Z, plan_inv_Z2Z, plan_kernel_Z2Z;
//    auto cufft_type = CUFFT_Z2Z;
//    checkCudaErrors(
//        cufftPlan2d(&plan_img_Z2Z, im_width, im_height, cufft_type));
//    checkCudaErrors(
//        cufftPlan2d(&plan_kernel_Z2Z, im_width, im_height, cufft_type));
//    checkCudaErrors(
//        cufftPlan2d(&plan_inv_Z2Z, im_width, im_height, cufft_type));
//
//    /* - - - Fast Fourier Transform on image - - - */
//    checkCudaErrors(cufftExecZ2Z(
//        plan_img_Z2Z, cufft_img_d, cufft_ffted_img_d, CUFFT_FORWARD));
//    KernelFFTShift2D<<<dim_grid, dim_block>>>(
//        cufft_ffted_img_d, im_height, im_width);
//
//    /* - - - Fast Fourier Transform on kernel - - - */
//    checkCudaErrors(cufftExecZ2Z(plan_kernel_Z2Z,
//                                 cufft_padded_kernel_d,
//                                 cufft_ffted_padded_kernel_d,
//                                 CUFFT_FORWARD));
//    KernelFFTShift2D<<<dim_grid, dim_block>>>(
//        cufft_ffted_padded_kernel_d, im_height, im_width);
//
//    /* element-wise matrix-mul */
//    ComponentwiseMatrixMul<<<dim_grid, dim_block>>>(cufft_ffted_img_d,
//                                                    cufft_ffted_padded_kernel_d,
//                                                    cufft_ffted_img_d,
//                                                    im_height,
//                                                    im_width);
//
//    /* - - - Executing IFFT and shifting back - - - */
//    KernelFFTShift2D<<<dim_grid, dim_block>>>(
//        cufft_ffted_img_d, im_height, im_width);
//    checkCudaErrors(cufftExecZ2Z(
//        plan_inv_Z2Z, cufft_ffted_img_d, cufft_img_d, CUFFT_INVERSE));
//    KernelFFTShift2D<<<dim_grid, dim_block>>>(cufft_img_d, im_height,
//    im_width);
//
//    /* - - - Generating output - - - */
//    auto c = (cufftDoubleComplex*)malloc(im_width * im_height *
//                                         sizeof(cufftDoubleComplex));
//    cudaMemcpy(c,
//               cufft_img_d,
//               sizeof(cufftDoubleComplex) * im_height * im_width,
//               cudaMemcpyDeviceToHost);
//
//    long double max = c[0].x;
//    for (int i = 0; i < im_height; i++) {
//        for (int j = 0; j < im_width; j++) {
//            if (c[i * im_width + j].x > max) max = c[i * im_width + j].x;
//        }
//    }
//    cv_img.convertTo(cv_img, CV_64F);
//    for (int i = 0; i < im_height; i++) {
//        for (int j = 0; j < im_width; j++) {
//            cv_img.at<double>(j, i) =
//                floor((c[i * im_width + j].x / max) * 255);
//        }
//    }
//    std::vector<int> compression_params;
//    compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
//    compression_params.push_back(9);
//    imwrite("/home/max/dev_projects/cuda_blob/data/output_image.jpg",
//            cv_img,
//            compression_params);
//
//    return 0;
//}

//#define NRANK 2
//#define BATCH 10
//
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <cufft.h>
//#include <iomanip>
//#include <iostream>
//#include <stdio.h>
//#include <vector>
//
// using namespace std;
//
// const size_t NX = 4;
// const size_t NY = 6;
//
// int main() {
//    // Input array (static) - host side
//    float h_in_data_static[NX][NY] = {
//        {0.7943, 0.6020, 0.7482, 0.9133, 0.9961, 0.9261},
//        {0.3112, 0.2630, 0.4505, 0.1524, 0.0782, 0.1782},
//        {0.5285, 0.6541, 0.0838, 0.8258, 0.4427, 0.3842},
//        {0.1656, 0.6892, 0.2290, 0.5383, 0.1067, 0.1712}};
//
//    // --------------------------------
//    // Input array (dynamic) - host side
//    float* h_in_data_dynamic = new float[NX * NY];
//
//    // Set the values
//    size_t h_ipitch;
//    for (int r = 0; r < NX; ++r) // this can be also done on GPU
//    {
//        for (int c = 0; c < NY; ++c) {
//            h_in_data_dynamic[NY * r + c] = h_in_data_static[r][c];
//        }
//    }
//    // --------------------------------
//    int owidth = (NY / 2) + 1;
//
//    // Output array - host side
//    float2* h_out_data_temp = new float2[NX * owidth];
//
//    // Input and Output array - device side
//    cufftHandle plan;
//    cufftReal* d_in_data;
//    cufftComplex* d_out_data;
//    int n[NRANK] = {NX, NY};
//
//    //  Copy input array from Host to Device
//    size_t ipitch;
//    cudaMallocPitch((void**)&d_in_data, &ipitch, NY * sizeof(cufftReal), NX);
//    cudaMemcpy2D(d_in_data,
//                 ipitch,
//                 h_in_data_dynamic,
//                 NY * sizeof(float),
//                 NY * sizeof(float),
//                 NX,
//                 cudaMemcpyHostToDevice);
//
//    //  Allocate memory for output array - device side
//    size_t opitch;
//    cudaMallocPitch(
//        (void**)&d_out_data, &opitch, owidth * sizeof(cufftComplex), NX);
//
//    //  Performe the fft
//    int rank = 2;                 // 2D fft
//    int istride = 1, ostride = 1; // Stride lengths
//    int idist = 1, odist = 1;     // Distance between batches
//    int inembed[] = {
//        NX,
//        static_cast<int>(ipitch / sizeof(cufftReal))}; // Input size with
//        pitch
//    int onembed[] = {
//        NX,
//        static_cast<int>(opitch /
//                         sizeof(cufftComplex))}; // Output size with pitch
//    int batch = 1;
//    cufftPlanMany(&plan,
//                  rank,
//                  n,
//                  inembed,
//                  istride,
//                  idist,
//                  onembed,
//                  ostride,
//                  odist,
//                  CUFFT_R2C,
//                  batch);
//    cufftExecR2C(plan, d_in_data, d_out_data);
//    cudaDeviceSynchronize();
//
//    // Copy d_in_data back from device to host
//    cudaMemcpy2D(h_out_data_temp,
//                 owidth * sizeof(float2),
//                 d_out_data,
//                 opitch,
//                 owidth * sizeof(cufftComplex),
//                 NX,
//                 cudaMemcpyDeviceToHost);
//
//    // Print the results
//    for (int i = 0; i < NX; i++) {
//        for (int j = 0; j < owidth; j++)
//            printf(" %f + %fi",
//                   h_out_data_temp[i * owidth + j].x,
//                   h_out_data_temp[i * owidth + j].y);
//        printf("\n");
//    }
//    cudaFree(d_in_data);
//
//    return 0;
//}

float** gaussianKernel(int width = 21, float sigma = 3.0) {
    float** kernel = new float*[width];
    auto mean = (width - 1) / 2;
    auto norm = 0.0;
    for (int y = 0; y < width; y++) {
        kernel[y] = new float[width];
        for (int x = 0; x < width; x++) {
            kernel[y][x] = (1 / (2 * M_PI * pow(sigma, 2))) *
                           exp(-(pow(x - mean, 2) + pow(y - mean, 2)) /
                               (2 * pow(sigma, 2)));
            norm += kernel[y][x];
        }
    }
    for (int y = 0; y < width; y++) {
        for (int x = 0; x < width; x++) {
            kernel[y][x] /= norm;
        }
    }
//    for (int y = 0; y < width; y++) {
//        for (int x = 0; x < width; x++) {
//            kernel[y][x] /= norm;
//            printf("%f ", kernel[y][x]);
//        }
//        std::cout << "\n";
//    }
    return kernel;
}

int main() {
    auto fp = "/home/max/dev_projects/cuda_blob/data/S_000_1752450056/"
              "Tile_r1-c1_S_000_1752450056.tif";
    cv::Mat cv_img = imread(fp, cv::IMREAD_GRAYSCALE);
    if (cv_img.empty()) {
        std::cout << "Could not read the image: " << fp << std::endl;
        return 1;
    }
    cv::resize(cv_img, cv_img, cv::Size(8192, 8192));
//    cv::resize(cv_img, cv_img, cv::Size(1024, 1024));

    auto im_width = cv_img.cols;
    auto im_height = cv_img.rows;
    auto batch_size = 20;
    printf("width %d, height %d\n", im_width, im_height);
    auto im_channels = cv_img.channels();
    assert(im_channels == 1);

    cufftHandle forward_plan, inverse_plan;

    int batch = batch_size;
    int rank = 2;

    int nRows = im_height;
    int nCols = im_width;
    int n[2] = {nRows, nCols};

    // dist between batches
    int idist = nRows * nCols;
    int odist = nRows * (nCols / 2 + 1);

    // input/output sizes with pitches ("unpitched")
    int inembed[] = {nRows, nCols};
    int onembed[] = {nRows, nCols / 2 + 1};

    int istride = 1;
    int ostride = 1;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float* h_in = (float*)malloc(sizeof(float) * batch * nRows * nCols);

    auto zz_yy_xx = 0;
    cudaEventRecord(start);
    for (int k = 0; k < batch_size; k++) {
        for (int i = 0; i < im_height; i++) {
            for (int j = 0; j < im_width; j++) {
                h_in[zz_yy_xx++] = (float)cv_img.at<uchar>(j, i);
            }
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("copy running time %f\n", milliseconds);

    /* - - - Building the Kernel with 0-padding - - - */
    auto radius = 3;
    auto kernel = gaussianKernel(2 * radius + 1);
    float* k_in = (float*)malloc(sizeof(float) * batch * nRows * nCols);
    for (int k = 0; k < batch_size; k++) {
        for (int i = 0; i < im_height; i++) {
            for (int j = 0; j < im_width; j++) {
                zz_yy_xx = k * (im_height * im_width) + i * im_width + j;
                if ((i >= ((im_height / 2) - radius)) &&
                    (i <= ((im_height / 2) + radius)) &&
                    (j >= ((im_width / 2) - radius)) &&
                    (j <= ((im_width / 2) + radius))) {
                    auto y = i - ((im_height / 2) - radius);
                    auto x = j - ((im_width / 2) - radius);
                    k_in[zz_yy_xx] = kernel[y][x];
                } else {
                    k_in[zz_yy_xx] = 0.0f;
                }
            }
        }
    }
//    for (int k = 0; k < batch_size; k++) {
//        for (int i = 0; i < im_height; i++) {
//            for (int j = 0; j < im_width; j++) {
//                printf("%f    ",
//                       k_in[k * (im_height * im_width) + i * im_width + j]);
//            }
//            std::cout << "\n";
//        }
//        printf("*********************\n");
//    }

    checkCudaErrors(cufftPlanMany(&forward_plan,
                                  rank,
                                  n,
                                  inembed,
                                  istride,
                                  idist,
                                  onembed,
                                  ostride,
                                  odist,
                                  CUFFT_R2C,
                                  batch));

    float2* h_freq =
        (float2*)malloc(sizeof(float2) * nRows * (nCols / 2 + 1) * batch);

    float* d_in;
    checkCudaErrors(cudaMalloc(&d_in, sizeof(float) * nRows * nCols * batch));
    float2* d_freq;
    checkCudaErrors(
        cudaMalloc(&d_freq, sizeof(float2) * nRows * (nCols / 2 + 1) * batch));

    checkCudaErrors(cudaMemcpy(d_in,
                               k_in,
                               sizeof(float) * nRows * nCols * batch,
                               cudaMemcpyHostToDevice));

    for (int i = 0; i < 10; i++) {
        cudaEventRecord(start);
        checkCudaErrors(cufftExecR2C(forward_plan, d_in, d_freq));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("fft running time %f\n", milliseconds);
    }

    checkCudaErrors(cudaMemcpy(h_freq,
                               d_freq,
                               sizeof(float2) * nRows * (nCols / 2 + 1) * batch,
                               cudaMemcpyDeviceToHost));

    //    for (int i = 0; i < nRows * (nCols / 2 + 1) * batch; i++)
    //        printf("Direct transform: %i %f %f\n", i, h_freq[i].x,
    //        h_freq[i].y);

    checkCudaErrors(cufftPlanMany(&inverse_plan,
                                  rank,
                                  n,
                                  onembed,
                                  ostride,
                                  odist,
                                  inembed,
                                  istride,
                                  idist,
                                  CUFFT_C2R,
                                  batch));

    checkCudaErrors(cufftExecC2R(inverse_plan, d_freq, d_in));

    checkCudaErrors(cudaMemcpy(k_in,
                               d_in,
                               sizeof(float) * nRows * nCols * batch,
                               cudaMemcpyDeviceToHost));

    // CUFFT has the same behavior as FFTW, it computes unnormalized FFTs.
    // https://stackoverflow.com/a/6460822
//    for (int k = 0; k < batch_size; k++) {
//        for (int i = 0; i < im_height; i++) {
//            for (int j = 0; j < im_width; j++) {
//                printf("%f    ",
//                       (1.0/(im_width*im_height))*abs(k_in[k * (im_height * im_width) + i * im_width + j]));
//            }
//            std::cout << "\n";
//        }
//        printf("*********************\n");
//    }
}