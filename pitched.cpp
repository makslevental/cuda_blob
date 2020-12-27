//
// Created by Maksim Levental on 12/26/20.
//

template <typename T>
inline T* getElementAddr(T* base, size_t pitch, int row, int col) {
    return (T*)((char*)base + row * pitch) + col;
}

template <typename T>
__global__ void readPitched2D(cudaPitchedPtr devPitchedPtr) {
    uint col = threadIdx.x + (blockDim.x * blockIdx.x);
    uint row = threadIdx.y + (blockDim.y * blockIdx.y);

    // https://stackoverflow.com/a/28767235
    // The reason they cast the array pointer into a char* is that pitch
    // returns a Byte size, not a number of elements (pitch could be non
    // multiple of element size).
    char* devPtr = (char*)devPitchedPtr.ptr;
    size_t pitch = devPitchedPtr.pitch;
    size_t w = devPitchedPtr.xsize;
    size_t h = devPitchedPtr.ysize;
    // T* pElement = (T*)((char*)BaseAddress + Row * pitch) + Column;
    if ((row < h) && (col < w)) {
        printf("row %d, col% d, val %f\n",
               row,
               col,
               ((T*)devPtr)[row * pitch / sizeof(T) + col]);
    }
}

template <typename T>
__global__ void writePitched2D(cudaPitchedPtr devPitchedPtr) {
    uint col = threadIdx.x + (blockDim.x * blockIdx.x);
    uint row = threadIdx.y + (blockDim.y * blockIdx.y);

    // https://stackoverflow.com/a/28767235
    // The reason they cast the array pointer into a char* is that pitch
    // returns a Byte size, not a number of elements (pitch could be non
    // multiple of element size).
    //
    // https://stackoverflow.com/a/21801712
    // This assumes the pitch parameter will be passed as a number of bytes,
    // which is the way cudaMallocPitch sets it.
    //
    char* devPtr = (char*)devPitchedPtr.ptr;
    size_t pitch = devPitchedPtr.pitch;
    size_t w = devPitchedPtr.xsize;
    size_t h = devPitchedPtr.ysize;

    // T* pElement = (T*)((char*)BaseAddress + Row * pitch) + Column;
    if ((row < h) && (col < w)) {
        // this is wrong because bitch is in bytes not float
        // ((T*)devPtr)[row * pitch + col] = (T)1.0;
        // either this or the following line
        ((T*)devPtr)[row * pitch / sizeof(T) + col] = (T)1.0;
        // *((T*)(devPtr + row * pitch) + col) = (T)1.0;
    }
}

inline std::tuple<dim3, dim3> getKernelDims(uint im_width, uint im_height) {
    // max 32*32 = 1024 thread per block
    auto num_threads = 32;
    dim3 dim_block(num_threads, num_threads);
    uint n_blocks_w = im_width / num_threads;
    if ((im_width % num_threads) != 0) n_blocks_w++;
    uint n_blocks_h = im_height / num_threads;
    if ((im_height % num_threads) != 0) n_blocks_h++;
    dim3 dim_grid(n_blocks_w, n_blocks_h);

    return std::make_tuple(dim_grid, dim_block);
}

int main() {
    size_t pitch;
    float* dev_memory;
    size_t im_width = 5;
    size_t im_height = 5;
    checkCudaErrors(cudaMallocPitch(
        &dev_memory,
        &pitch,
        im_width * sizeof(float), // actual width (will be padded
        // out to size `pitch` bytes)
        im_height                 // actual height (not padded
    ));
    printf("pitch %zu bytes\n", pitch);
    cudaPitchedPtr pitchedPtr{dev_memory, pitch, im_width, im_height};
    auto [dim_grid, dim_block] = getKernelDims(im_width, im_height);

    readPitched2D<float><<<dim_grid, dim_block>>>(pitchedPtr);
    writePitched2D<float><<<dim_grid, dim_block>>>(pitchedPtr);
    readPitched2D<float><<<dim_grid, dim_block>>>(pitchedPtr);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    auto host_memory = new float[im_height * im_width];

    // takes either linear memory or statically allocated 2d arrays
    checkCudaErrors(
        cudaMemcpy2D(host_memory,
            /*dest pitch in bytes*/ im_width * sizeof(float),
                     dev_memory,
            /*src pitch in bytes*/ pitch,
            /*actual width in bytes*/ im_width * sizeof(float),
                     im_height,
                     cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaDeviceSynchronize());

    for (int i = 0; i < im_height; i++) {
        for (int j = 0; j < im_width; j++) {
            printf("row %i column %i value %f \n",
                   i,
                   j,
                   host_memory[i * im_width + j]);
        }
    }
}
