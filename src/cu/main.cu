#pragma once
#include <cuda.h>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include <Windows.h>


//à bien inclure après glew
#include "cuda_gl_interop.h"
#include "device_functions.h"
#include "device_atomic_functions.h"



#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include "thrust/sort.h"
#include "thrust/copy.h"



#include "../h/Const.h"
#include "../h/Geometry.h"
#include "../cuh/ParticulesField.cuh"

//Function to retrieve the error message from cuda
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) { exit(code); getchar(); }
    }
}

//Function to retrieve the GPU information
#define getInfGPU() { gpuInfo();} 
inline void gpuInfo()
{
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "Free memory: " << free_mem / 1024.0 / 1024.0 << " MB" << std::endl;
    std::cout << "Total memory: " << total_mem / 1024.0 / 1024.0 << " MB" << std::endl;
}


//Function to create the grid 1D
__global__ void createGrid(ParticulesField* part, int* indexTri)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Calcul of the index of the cell in the uniform grid
    if (idx < WORKINGSET)
    {
        int3 pos;

        // Calcul of the position of the cell in the uniform grid
        pos.x = floor(part->GetPoint(idx).Position.getX() / h);
        pos.y = floor(part->GetPoint(idx).Position.getY() / h);
        pos.z = floor(part->GetPoint(idx).Position.getZ() / h);

        // Put the index of the cell in the uniform grid
        indexTri[idx] = pos.z * (int)(DATA_H / h) * (int)(DATA_W / h) + pos.y * (int)(DATA_W / h) + pos.x;
    }
}


__global__ void loop(ParticulesField* part, int* indexTri, int* start, int* end)
{


}


int main()
{
    //Creation of the particules field and the grid
    ParticulesField* starting_particules = new ParticulesField();
    int* grille = new int[WORKINGSET];
    int* start = new int[DATA_W * DATA_H * DATA_W];
    int* end = new int[DATA_W * DATA_H * DATA_W];
       
    //Allocation of the memory on the GPU
    ParticulesField* gpuPart = nullptr;
    int* gpuGrille = nullptr;
    int* gpuStart = nullptr;
    int* gpuEnd = nullptr;

    gpuErrchk(cudaMalloc((void**)&gpuPart, sizeof(ParticulesField)));
    gpuErrchk(cudaMalloc((void**)&gpuGrille, sizeof(int) * WORKINGSET));
    gpuErrchk(cudaMalloc((void**)&gpuStart, sizeof(int) * SizeCube));
    gpuErrchk(cudaMalloc((void**)&gpuEnd, sizeof(int) * SizeCube));
    gpuErrchk(cudaMemcpy(gpuPart, starting_particules, sizeof(ParticulesField), cudaMemcpyHostToDevice));


    //Dimension of the grid and the block
    dim3 grid(WORKINGSET/BlockSize);
    dim3 block(BlockSize);

    //First kernel to create the grid 1D
    std::cout << "Creation of the grid" << std::endl;
    createGrid << <grid, block >> > (gpuPart, gpuGrille);

    gpuErrchk(cudaDeviceSynchronize());
        
    //Thrust part to sort the grid
    thrust::device_ptr<int> dev_ptr(gpuGrille);
    thrust::sort(dev_ptr, dev_ptr + WORKINGSET);

    
    gpuErrchk(cudaDeviceSynchronize());

    

    gpuErrchk(cudaFree(gpuPart));
    gpuErrchk(cudaFree(gpuGrille));
    delete starting_particules;
    delete grille;
    

    return 0;
}



