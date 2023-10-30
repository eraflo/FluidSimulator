#pragma once
#include <cuda.h>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include <Windows.h>


//à bien inclure après glew
#include "cuda_gl_interop.h"
#include "device_atomic_functions.h"



#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include "thrust/sort.h"
#include "thrust/copy.h"



#include "h/Const.h"
#include "cuh/ParticulesField.cuh"

//Function to retrieve the error message from cuda
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
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
__global__ void createGrid(ParticulesField* part, GridIndexSorted* indexLinearTri)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Calcul of the index of the cell in the uniform grid for each particule
    if (idx < WORKINGSET)
    {
        float3 pos;

        // Calcul of the coordinate of the cell where the particule is located in the uniform grid (still in 3D)
        pos.x = floor(part->GetPoint(idx).Position.getX() / h);
        pos.y = floor(part->GetPoint(idx).Position.getY() / h);
        pos.z = floor(part->GetPoint(idx).Position.getZ() / h);

        // Calcul of the index of the cell in the uniform grid 1D
        indexLinearTri->indexSorted[idx] = pos.z * (int)(DATA_H / h) * (int)(DATA_W / h) + pos.y * (int)(DATA_W / h) + pos.x;
    }
}


__global__ void loop(ParticulesField* part, int* indexTri, int* start, int* end)
{


}


int main()
{
    //Creation of the particules field and the grid
    ParticulesField* starting_particules = new ParticulesField();
    GridIndexSorted* grid1D = new GridIndexSorted();
       
    //Allocation of the memory on the GPU
    ParticulesField* gpuPart = nullptr;
    GridIndexSorted* gpuGrid = nullptr;

    gpuErrchk(cudaMalloc((void**)&gpuPart, sizeof(ParticulesField)));
    gpuErrchk(cudaMalloc((void**)&gpuGrid, sizeof(GridIndexSorted)));
    gpuErrchk(cudaMemcpy(gpuPart, starting_particules, sizeof(ParticulesField), cudaMemcpyHostToDevice));

    // Display the particules field
    for (int i = 0; i < WORKINGSET; i++)
	{
		std::cout << starting_particules->GetPoint(i).Position.getX() << " " << starting_particules->GetPoint(i).Position.getY() << " " << starting_particules->GetPoint(i).Position.getZ() << std::endl;
	}

    //Dimension of the grid and the block
    dim3 grid(WORKINGSET/BlockSize);
    dim3 block(BlockSize);

    //First kernel to create the grid 1D
    std::cout << "Creation of the grid" << std::endl;
    createGrid << <grid, block >> > (gpuPart, gpuGrid);

    gpuErrchk(cudaDeviceSynchronize());

    // Copy the grid from the GPU to the CPU
    gpuErrchk(cudaMemcpy(grid1D, gpuGrid, sizeof(int) * WORKINGSET, cudaMemcpyDeviceToHost));

    // Display the grid
    for (int i = 0; i < WORKINGSET; i++)
	{
		std::cout << grid1D->indexSorted[i] << std::endl;
	}
        
    //Thrust part to sort the grid
    std::cout << "Sort of the grid" << std::endl;
    thrust::device_ptr<int> dev_ptr(gpuGrid->indexSorted);
    thrust::sort(dev_ptr, dev_ptr + WORKINGSET);

    // Display the grid
    gpuErrchk(cudaMemcpy(grid1D->indexSorted, gpuGrid->indexSorted, sizeof(int) * WORKINGSET, cudaMemcpyDeviceToHost));
	for (int i = 0; i < WORKINGSET; i++)
    {
    std::cout << grid1D->indexSorted[i] << std::endl;
    }

    
    gpuErrchk(cudaDeviceSynchronize());

    

    gpuErrchk(cudaFree(gpuPart));
    gpuErrchk(cudaFree(gpuGrid));
    delete starting_particules;
    delete grid1D;
    

    return 0;
}



