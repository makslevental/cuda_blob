//
// Created by Maksim Levental on 12/25/20.
//

#ifndef CUDA_BLOB_UTIL_H
#define CUDA_BLOB_UTIL_H

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <mpi.h>

std::ifstream OpenFile(const std::string& myFile) {
    std::ifstream file(myFile, std::ios::in);
    if (!file) {
        std::cerr << "Can't open file " + myFile + "!" << std::endl;
        exit(EXIT_FAILURE);
    }
    return file;

}

const char* DepthToStr(int depth) {
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

template<typename T>
void PrintArray(const T(& a), uint N, std::ostream& o = std::cout) {
    o << "{";
    for (std::size_t i = 0; i < N - 1; ++i) {
        o << a[i] << ", ";
    }
    o << a[N - 1] << "}\n";
}

//#define MPI_CHECK(call) \
//    if((call) != MPI_SUCCESS) { \
//        printf("MPI error calling \"%s\"\n", #call); \
//        MPI_Abort(MPI_COMM_WORLD, -1); }

//int test_mpi() {
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


#endif //CUDA_BLOB_UTIL_H
