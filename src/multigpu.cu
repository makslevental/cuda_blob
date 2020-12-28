//
// Created by Maksim Levental on 12/28/20.
//

#include <cuComplex.h>
#include <cufft.h>
#include <iostream>
#include <vector>

#include "cufftXt.h"
#include "helper_cuda.h"
#include "multigpu.cuh"

void print3Dfloat2(std::vector<float2> hOut, int nb, int nr, int nc) {
    for (int k = 0; k < nb; k++) {
        for (int i = 0; i < nr; i++) {
            for (int j = 0; j < nc; j++) {
                auto zzyyxx = (k * (nr * nc)) + (i * nc) + j;
                printf("%.2f + %.2fi    ", hOut[zzyyxx].x, hOut[zzyyxx].y);
            }
            printf("\n");
        }
        printf("\nbatch %d *************\n", k);
    }
}

void print3D(std::vector<float> hOut, int nb, int nr, int nc) {
    for (int k = 0; k < nb; k++) {
        for (int i = 0; i < nr; i++) {
            printf("[ ");
            for (int j = 0; j < nc; j++) {
                printf("%.2f,   ", hOut[k * (nr * nc) + i * nc + j]);
            }
            printf("],\n ");
        }
        printf("\nbatch %d *************\n", k);
    }
}

int runMultiGPU() {
    static const int numGPUs = 2;
    int gpus[numGPUs] = {0, 1};

    // The real-to-complex transform is implicitly a forward transform. For an in-place
    // real-to-complex transform where FFTW compatible output is desired, the input size must be
    // padded to ⌊N/2⌋ + 1 complex elements.
    // wtf?
    int nb = 16;
    int nr = 8;
    int nc = 8;

    // Fill with junk data
    std::vector<float> hIn(nb * nr * nc);
    //    for (int i = 0; i < nb * nr * nc; ++i) {
    //        hIn[i] = static_cast<float>(i);
    //    }
    for (int k = 0; k < nb; k++) {
        for (int i = 0; i < nr; i++) {
            for (int j = 0; j < nc; j++) {
                hIn[k * (nr * nc) + i * nc + j] = i * nc + j;
            }
        }
    }

    //    print3D(hIn, nb, nr, nc);

    cufftHandle plan;
    checkCudaErrors(cufftCreate(&plan));
    checkCudaErrors(cufftXtSetGPUs(plan, numGPUs, gpus));

    // dimension of fft
    int rank = 2;
    int n[2] = {nr, nc};
    // input/output sizes with pitches ("unpitched")
    int inEmbed[] = {nr, nc};
    int onEmbed[] = {nr, nc / 2 + 1};
    // dist between batches
    int iDist = nr * nc;
    int oDist = nr * (nc / 2 + 1);
    // stride between adjacent entries in row
    int iStride = 1;
    int oStride = 1;

    size_t workSize[2];
    cufftMakePlanMany(
        plan, rank, n, inEmbed, iStride, iDist, onEmbed, oStride, oDist, CUFFT_R2C, nb, workSize);

    cudaLibXtDesc* dX;
    checkCudaErrors(cufftXtMalloc(plan, &dX, CUFFT_XT_FORMAT_INPLACE));

    checkCudaErrors(cufftXtMemcpy(plan, dX, (void*)hIn.data(), CUFFT_COPY_HOST_TO_DEVICE));

    checkCudaErrors(cufftXtExecDescriptorR2C(plan, dX, dX));
    checkCudaErrors(cudaGetLastError());

    std::vector<float2> hOut(nb * nr * (nc / 2 + 1));
    checkCudaErrors(cufftXtMemcpy(plan, (void*)hOut.data(), dX, CUFFT_COPY_DEVICE_TO_HOST));
    checkCudaErrors(cudaDeviceSynchronize());

    print3Dfloat2(hOut, nb, nr, (nc / 2 + 1));

    checkCudaErrors(cufftXtFree(dX));
    checkCudaErrors(cufftDestroy(plan));

    checkCudaErrors(cudaDeviceReset());

    return 0;
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

//#include "cufft.h"
//#include "cufftXt.h"
//#include "prettyprint.h"
//#include <cassert>
//#include <cuComplex.h>
//#include <vector>
//
//#define CUDA_CHECK(x)  assert(x == cudaSuccess)
//#define CUFFT_CHECK(x) assert(x == CUFFT_SUCCESS)
//
// int runMultiGPU() {
//    static const int numGPUs = 2;
//    int gpus[numGPUs] = {0, 1};
//
//    int nr = 16;
//    int nc = 8;
//
//    // Fill with junk data
//    std::vector<cuFloatComplex> h_x(nr * nc);
//    for (int i = 0; i < nr * nc; ++i) {
//        h_x[i].x = static_cast<float>(i);
//    }
//
//    print3Dfloat2(h_x, 1, nr, nc);
//
//    cufftHandle plan;
//    CUFFT_CHECK(cufftCreate(&plan));
//    CUFFT_CHECK(cufftXtSetGPUs(plan, numGPUs, gpus));
//
//    std::vector<size_t> workSizes(numGPUs);
//    int n[] = {nr};
//
//    //    // dimension of fft
//    //    int rank = 2;
//    //    Array of size rank, describing the size of each dimension, n[0] being the size of the
//    //    outermost and n[rank-1] innermost (contiguous) dimension of a transform.
//    //    int n[2] = {nr, nc};
//    //    // dist between batches
//    //    int iDist = nr * nc;
//    //    int oDist = nr * (nc / 2 + 1);
//    //    // input/output sizes with pitches ("unpitched")
//    //    int inEmbed[] = {nr, nc};
//    //    int onEmbed[] = {nr, nc / 2 + 1};
//    //    // stride between adjacent entries in row
//    //    int iStride = 1;
//    //    int oStride = 1;
//
//    CUFFT_CHECK(cufftMakePlanMany(plan,
//                                  1, // rank
//                                  n, // n
//                                  n, // inembed
//                                  1, // istride
//                                  1, // idist
//                                  n, // onembed
//                                  1, // ostride
//                                  1, // odist
//                                  CUFFT_C2C,
//                                  nc,
//                                  workSizes.data()));
//
//    cudaLibXtDesc* d_x;
//    CUFFT_CHECK(cufftXtMalloc(plan, &d_x, CUFFT_XT_FORMAT_INPLACE));
//
//    CUFFT_CHECK(cufftXtMemcpy(plan, d_x, (void*)h_x.data(), CUFFT_COPY_HOST_TO_DEVICE));
//
//    CUFFT_CHECK(cufftXtExecDescriptorC2C(plan, d_x, d_x, CUFFT_FORWARD));
//
//    std::vector<float2> h_out(nr * nc);
//    CUFFT_CHECK(cufftXtMemcpy(plan, (void*)h_out.data(), d_x, CUFFT_COPY_DEVICE_TO_HOST));
//
//    print3Dfloat2(h_out, 1, nr, nc);
//
//    //    CUFFT_CHECK(cufftXtFree(d_x));
//    //    CUFFT_CHECK(cufftDestroy(plan));
//    //
//    //    CUDA_CHECK(cudaDeviceReset());
//
//    return 0;
//}