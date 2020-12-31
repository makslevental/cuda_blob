//
// Created by Maksim Levental on 12/28/20.
//

#include <cuComplex.h>
#include <cufft.h>
#include <cufftXt.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "helper_cuda.h"
#include "multiGPU.cuh"
#include "stacktrace.h"
#include "util.h"

void multiplyCoefficient(float2* signal,
                         cudaLibXtDesc* kernel,
                         int nGPUs,
                         int batchSize,
                         int imgHeight,
                         int imgWidth);

int runMultiGPU() {
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* - - - load image using OpenCV - - - */
    cv::Mat imgCv;
    CUDA_TIME([&]() {
        auto fp = "/home/max/dev_projects/cuda_blob/data/S_000_1752450056/"
                  "Tile_r1-c1_S_000_1752450056.tif";
        imgCv = imread(fp, cv::IMREAD_GRAYSCALE);
        if (imgCv.empty()) {
            std::cout << "Could not read the image: " << fp << std::endl;
            return 1;
        }
        imgCv.convertTo(imgCv, CV_32FC1);
    })
    printf("img load and convert to float time %.5fms\n", milliseconds);
    cv::resize(imgCv, imgCv, cv::Size(8192, 8192));
    int imgWidth = imgCv.cols;
    int imgHeight = imgCv.rows;
    int batchSize = 15;
    printf("width %d, height %d batch size %d\n", imgWidth, imgHeight, batchSize);
    assert(imgCv.channels() == 1);

    /* - - - Building the Kernel with 0-padding to the size of the image - - - */
    // The real-to-complex transform is implicitly a forward transform. For an in-place
    // real-to-complex transform where FFTW compatible output is desired, the input size must be
    // padded to ⌊N/2⌋ + 1 complex elements.
    // wtf?
    // something having to do with padding again???
    size_t totalSize = (size_t)batchSize * (size_t)imgHeight * ((size_t)imgWidth + 2);
    std::vector<float> kernelH(totalSize);
    std::fill(kernelH.begin(), kernelH.begin() + totalSize, 0);
    int y, x;
    size_t zzyyxx;
    CUDA_TIME([&]() {
        for (int k = 0; k < batchSize; k++) {
            auto radius = 1;
            auto kernel = gaussianKernel(2 * radius + 1);
            for (int i = ((imgHeight / 2) - radius); i <= ((imgHeight / 2) + radius); i++) {
                for (int j = ((imgWidth / 2) - radius); j <= ((imgWidth / 2) + radius); j++) {
                    y = i - ((imgHeight / 2) - radius);
                    x = j - ((imgWidth / 2) - radius);
                    zzyyxx = k * (imgHeight * imgWidth) + i * imgWidth + j;
                    kernelH[zzyyxx] = kernel[y][x];
                }
            }
        }
    })
    printf("kernel creation time %.5fms\n", milliseconds);

    ////////////////////// doesn't work
    //    print3Dfloat(reinterpret_cast<float*>(imgCv.data), 1, imgHeight, imgWidth);
    //    print3Dfloat(kernelH.data(), batchSize, imgHeight, imgWidth + 2);

    /* - - -  ffts - - - */
    // img fft
    cufftHandle imgForwardPlan, imgInversePlan;
    float* imgH = (float*)(imgCv.isContinuous() ? imgCv.data : imgCv.clone().data);
    float* imgD;
    float2* imgFreqsD;

    CUDA_TIME(
        [&]() { checkCudaErrors(cufftPlan2d(&imgForwardPlan, imgWidth, imgHeight, CUFFT_R2C)); })
    printf("img forward plan time %.5fms\n", milliseconds);
    CUDA_TIME(
        [&]() { checkCudaErrors(cufftPlan2d(&imgInversePlan, imgWidth, imgHeight, CUFFT_C2R)); })
    printf("img inverse plan time %.5fms\n", milliseconds);
    checkCudaErrors(cudaMalloc(&imgD, sizeof(float) * imgHeight * imgWidth));
    checkCudaErrors(cudaMalloc(&imgFreqsD, sizeof(float2) * imgHeight * (imgWidth / 2 + 1)));

    CUDA_TIME([&]() {
        checkCudaErrors(
            cudaMemcpy(imgD, imgH, sizeof(float) * imgHeight * imgWidth, cudaMemcpyHostToDevice));
    })
    printf("img copy to device time %.5fms\n", milliseconds);
    CUDA_TIME([&]() { checkCudaErrors(cufftExecR2C(imgForwardPlan, imgD, imgFreqsD)); })
    printf("img fft time %.5fms\n", milliseconds);

    // kernel fft
    static const int numGPUs = 2;
    int gpus[numGPUs] = {0, 1};

    cufftHandle kernelForwardPlan, kernelInversePlan;
    checkCudaErrors(cufftCreate(&kernelForwardPlan));
    checkCudaErrors(cufftCreate(&kernelInversePlan));
    checkCudaErrors(cufftXtSetGPUs(kernelForwardPlan, numGPUs, gpus));
    checkCudaErrors(cufftXtSetGPUs(kernelInversePlan, numGPUs, gpus));

    // dimension of fft
    int rank = 2;
    int n[2] = {imgHeight, imgWidth};
    // input/output sizes with pitches ("unpitched")
    int inEmbed[] = {imgHeight, imgWidth};
    int onEmbed[] = {imgHeight, imgWidth / 2 + 1};
    // dist between batches
    int iDist = imgHeight * imgWidth;
    int oDist = imgHeight * (imgWidth / 2 + 1);
    // stride between adjacent entries in row
    int iStride = 1;
    int oStride = 1;

    size_t workSize[2];
    CUDA_TIME([&]() {
        cufftMakePlanMany(kernelForwardPlan,
                          rank,
                          n,
                          inEmbed,
                          iStride,
                          iDist,
                          onEmbed,
                          oStride,
                          oDist,
                          CUFFT_R2C,
                          batchSize,
                          workSize);
        cufftMakePlanMany(kernelInversePlan,
                          rank,
                          n,
                          onEmbed,
                          oStride,
                          oDist,
                          inEmbed,
                          iStride,
                          iDist,
                          CUFFT_C2R,
                          batchSize,
                          workSize);
    })
    printf("kernel plan time %.5fms\n", milliseconds);
    cudaLibXtDesc* kernelFreqsDDesc;
    checkCudaErrors(cufftXtMalloc(kernelForwardPlan, &kernelFreqsDDesc, CUFFT_XT_FORMAT_INPLACE));
    CUDA_TIME([&]() {
        checkCudaErrors(cufftXtMemcpy(
            kernelForwardPlan, kernelFreqsDDesc, kernelH.data(), CUFFT_COPY_HOST_TO_DEVICE));
    })
    printf("kernel copy to device time %.5fms\n", milliseconds);
    CUDA_TIME([&]() {
        checkCudaErrors(
            cufftXtExecDescriptorR2C(kernelForwardPlan, kernelFreqsDDesc, kernelFreqsDDesc));
    })
    printf("kernel fft time %.5fms\n", milliseconds);

    //    std::vector<float2> hOut(batchSize * imgHeight * (imgWidth / 2 + 1));
    //    checkCudaErrors(cufftXtMemcpy(
    //        kernelForwardPlan, (void*)kernelFreqsD, kernelFreqsDDesc,
    //        CUFFT_COPY_DEVICE_TO_DEVICE));
    //    checkCudaErrors(cudaDeviceSynchronize());

    //    print3Dfloat2(hOut, batchSize, imgHeight, (imgWidth / 2 + 1));

    //    printf("\n\nValue of Library Descriptor\n");
    //    printf("Number of GPUs %d\n", kernelFreqsDDesc->descriptor->nGPUs);
    //    printf("Device id  %d %d\n",
    //           kernelFreqsDDesc->descriptor->GPUs[0],
    //           kernelFreqsDDesc->descriptor->GPUs[1]);
    //    printf("Data size on GPU %ld %ld\n",
    //           (long)(kernelFreqsDDesc->descriptor->size[0] / sizeof(cufftComplex)),
    //           (long)(kernelFreqsDDesc->descriptor->size[1] / sizeof(cufftComplex)));

    // Multiply the coefficients together and normalize the result
    CUDA_TIME([&]() {
        multiplyCoefficient(imgFreqsD, kernelFreqsDDesc, numGPUs, batchSize, imgHeight, imgWidth);
    })
    printf("filtering time %.5fms\n", milliseconds);

    CUDA_TIME([&]() {
        checkCudaErrors(
            cufftXtExecDescriptorC2R(kernelInversePlan, kernelFreqsDDesc, kernelFreqsDDesc));
    })
    printf("filtered inverse fft time %.5fms\n", milliseconds);
    std::vector<float> filteredH(totalSize);
    CUDA_TIME([&]() {
        checkCudaErrors(cufftXtMemcpy(
            kernelInversePlan, filteredH.data(), kernelFreqsDDesc, CUFFT_COPY_DEVICE_TO_HOST));
    })
    printf("filtered copy to host time %.5fms\n", milliseconds);

    //    print3Dfloat(filteredH.data(), batchSize, imgHeight, imgWidth + 2);

    checkCudaErrors(cufftXtFree(kernelFreqsDDesc));
    checkCudaErrors(cufftDestroy(kernelForwardPlan));
    checkCudaErrors(cudaDeviceReset());

    return 0;
}

////////////////////////////////////////////////////////////////////////////////
//  Launch Kernel on multiple GPU
////////////////////////////////////////////////////////////////////////////////
void multiplyCoefficient(float2* signal,
                         cudaLibXtDesc* kernel,
                         int nGPUs,
                         int batchSize,
                         int imgHeight,
                         int imgWidth) {
    auto numThreads = 32;
    dim3 dimBlock(numThreads, numThreads);
    int nBlocksW = (imgWidth / 2 + 1) / numThreads;
    if (((imgWidth / 2 + 1) % numThreads) != 0) nBlocksW++;
    int nBlocksH = imgHeight / numThreads;
    if ((imgHeight % numThreads) != 0) nBlocksH++;
    dim3 dimGrid(nBlocksW, nBlocksH, batchSize);

    int imgSize = sizeof(float2) * imgHeight * (imgWidth / 2 + 1);
    int origDevice;
    checkCudaErrors(cudaGetDevice(&origDevice));
    int device;
    for (int i = 0; i < nGPUs; i++) {
        device = kernel->descriptor->GPUs[i];
        checkCudaErrors(cudaSetDevice(device));
        float2* localSignal;
        if (device != origDevice) {
            checkCudaErrors(cudaMalloc(&localSignal, imgSize));
            checkCudaErrors(cudaMemcpyPeer(localSignal, device, signal, 0, imgSize));
        } else {
            localSignal = signal;
        }
        componentwiseMatrixMul1vsBatchfloat2<<<dimGrid, dimBlock>>>(
            localSignal,
            (float2*)kernel->descriptor->data[i],
            (float2*)kernel->descriptor->data[i],
            batchSize / nGPUs,
            imgHeight,
            (imgWidth / 2 + 1));
    }

    for (int i = 0; i < nGPUs; i++) {
        device = kernel->descriptor->GPUs[i];
        checkCudaErrors(cudaSetDevice(device));
        checkCudaErrors(cudaDeviceSynchronize());
        getLastCudaError("Kernel execution failed [ componentwiseMatrixMul1vsBatchfloat2 ]");
    }
    checkCudaErrors(cudaSetDevice(origDevice));
}

//#include <assert.h>
//#include <cufft.h>
//#include <stdio.h>
//#include <stdlib.h>
//
// const size_t sigSize = 1 << 29;
// typedef cufftComplex ctype;
// typedef cufftReal rtype;
//
// int runMultiGPU() {
//
//    cufftResult res;
//    rtype *devInData0, *devInData1;
//    ctype *devOutData0, *devOutData1;
//    cufftHandle handle0, handle1;
//    cudaStream_t stream0, stream1;
//
//    cudaSetDevice(0);
//    res = cufftPlan1d(&handle0, sigSize, CUFFT_R2C, 1);
//    assert(res == CUFFT_SUCCESS);
//    cudaStreamCreate(&stream0);
//    res = cufftSetStream(handle0, stream0);
//    assert(res == CUFFT_SUCCESS);
//    cudaMalloc(&devInData0, sizeof(rtype) * sigSize);
//    cudaMalloc(&devOutData0, sizeof(ctype) * (sigSize * 2 + 1));
//
//    cudaSetDevice(1);
//    res = cufftPlan1d(&handle1, sigSize, CUFFT_R2C, 1);
//    assert(res == CUFFT_SUCCESS);
//    cudaStreamCreate(&stream1);
//    res = cufftSetStream(handle1, stream1);
//    assert(res == CUFFT_SUCCESS);
//    cudaMalloc(&devInData1, sizeof(rtype) * sigSize);
//    cudaMalloc(&devOutData1, sizeof(ctype) * (sigSize * 2 + 1));
//
//    cudaSetDevice(0);
//    res = cufftExecR2C(handle0, devInData0, devOutData0);
//    assert(res == CUFFT_SUCCESS);
//    cudaDeviceSynchronize();
//
//    cudaSetDevice(1);
//    res = cufftExecR2C(handle1, devInData1, devOutData1);
//    assert(res == CUFFT_SUCCESS);
//    cudaDeviceSynchronize();
//
//    return 0;
//}
