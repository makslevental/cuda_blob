//
// Created by Maksim Levental on 12/28/20.
//

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "helper_cuda.h"
#include "stacktrace.h"
#include "util.h"

std::ifstream openFile(const std::string& myFile) {
    std::ifstream file(myFile, std::ios::in);
    if (!file) {
        std::cerr << "Can't open file " + myFile + "!" << std::endl;
        exit(EXIT_FAILURE);
    }
    return file;
}

const char* depthToStr(int depth) {
    switch (depth) {
    case CV_8U:
        return "unsigned char";
    case CV_8S:
        return "char";
    case CV_16U:
        return "unsigned short";
    case CV_16S:
        return "short";
    case CV_32S:
        return "int";
    case CV_32F:
        return "float";
    case CV_64F:
        return "double";
    default:
        return "invalid type!";
    }
}

template <typename T> void printArray(const T(&a), uint N, std::ostream& o) {
    o << "{";
    for (int i = 0; i < N - 1; ++i) {
        o << a[i] << ", ";
    }
    o << a[N - 1] << "}\n";
}

//#define MPI_CHECK(call) \
//    if((call) != MPI_SUCCESS) { \
//        printf("MPI error calling \"%s\"\n", #call); \
//        MPI_Abort(MPI_COMM_WORLD, -1); }

// int testMpi() {
//    MPI_Init(nullptr, nullptr);
//    int world_size;
//    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
//    int world_rank;
//    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
//
//    double* x_h, * y_h; // host data
//    double* x_d, * y_d; // device data
//    const int N = 100,
//            nBytes = N * sizeof(double);
//
//    x_h = new double[N];
//    y_h = new double[N];
//
//    // allocate memory on device
//    for (int i = 0; i < N; i++) {
//        x_h[i] = (i % 137) + 1;
//    }
//
//    CHECKCUDAERRORS(cudaSetDevice(world_rank));
//    // copy data:  host --> device
//    if (world_rank == 0) {
//        print_array(x_h, N);
//        CHECKCUDAERRORS(cudaMalloc((void**) &x_d, nBytes));
//        CHECKCUDAERRORS(cudaMemcpy(x_d, x_h, nBytes, cudaMemcpyHostToDevice));
//        MPI_CHECK(MPI_Send(x_d, N, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD));
//        printf("successfully sent\n");
//    }
//
//    if (world_rank == 1) {
//        CHECKCUDAERRORS(cudaMalloc((void**) &y_d, nBytes));
//        MPI_Status status;
//        MPI_CHECK(MPI_Recv(y_d, N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status));
//        printf("successfully received\n");
//        CHECKCUDAERRORS(cudaMemcpy(y_h, y_d, nBytes, cudaMemcpyDeviceToHost));
//        print_array(y_h, N);
//    }
//
//    delete[] x_h;
//    delete[] y_h;
//    cudaFree(x_d);
//    cudaFree(y_d);
//    MPI_Finalize();
//    return 0;
//}

float** gaussianKernel(int width, float sigma) {
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

void print3Dfloat(float* hOut, int nb, int nr, int nc) {
    for (int k = 0; k < nb; k++) {
        for (int i = 0; i < nr; i++) {
            for (int j = 0; j < nc; j++) {
                auto zzyyxx = (k * (nr * nc)) + (i * nc) + j;
                printf("%.2f ", hOut[zzyyxx]);
            }
            printf("\n");
        }
        printf("\nbatch %d *************\n", k);
    }
}

__global__ void componentwiseMatrixMul1vsBatchfloat2(float2* singleIn,
                                                     float2* batchIn,
                                                     float2* out,
                                                     int batchSize,
                                                     int rows,
                                                     int cols) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int batch = blockIdx.z;

    auto yyxx = row * cols + col;
    auto zzyyxx = batch * (rows * cols) + yyxx;

    if (batch < batchSize && row < rows && col < cols) {
        out[zzyyxx].x = singleIn[yyxx].x * batchIn[zzyyxx].x - singleIn[yyxx].y * batchIn[zzyyxx].y;
        out[zzyyxx].y = singleIn[yyxx].x * batchIn[zzyyxx].y + singleIn[yyxx].y * batchIn[zzyyxx].x;
    }
}
