//
// Created by Maksim Levental on 12/25/20.
//

#ifndef CUDA_BLOB_UTIL_H
#define CUDA_BLOB_UTIL_H

#include <iostream>
#include <vector>

std::ifstream openFile(const std::string& myFile);

const char* depthToStr(int depth);

template <typename T> void printArray(const T(&a), uint N, std::ostream& o = std::cout);

float** gaussianKernel(int width = 21, float sigma = 3.0);

void printDevice3Dfloat2(float2* dev_ptr, int batch_size, int width, int height);

void print3Dfloat2(std::vector<float2> hOut, int nb, int nr, int nc);

void print3Dfloat(float* hOut, int nb, int nr, int nc);

__global__ void componentwiseMatrixMul1vsBatchfloat2(float2* singleIn,
                                                     float2* batchIn,
                                                     float2* out,
                                                     int batchSize,
                                                     int rows,
                                                     int cols);
#endif // CUDA_BLOB_UTIL_H
