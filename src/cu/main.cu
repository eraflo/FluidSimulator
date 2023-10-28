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

//fonction pour récupérer en partie les erreurs GPU
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) { exit(code); getchar(); }
    }
}

//fonction pour récupérer les informations d'utilisation de la mémoire
#define getInfGPU() { gpuInfo();} 
inline void gpuInfo()
{
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "Free memory: " << free_mem / 1024.0 / 1024.0 << " MB" << std::endl;
    std::cout << "Total memory: " << total_mem / 1024.0 / 1024.0 << " MB" << std::endl;
}


//fonction pour créer notre grille 1D avec les indices de case où se trouvent chaque particule
__global__ void createGrid(ParticulesField* part, int* indexTri)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // calcul de l'indice de la case
    if (idx < WORKINGSET)
    {
        int3 pos;

        // calcul des coordonnées de la case dans la grille uniforme
        pos.x = floor(part->point[idx].Position.getX() / h);
        pos.y = floor(part->point[idx].Position.getY() / h);
        pos.z = floor(part->point[idx].Position.getZ() / h);

        // calcul de l'indice de la case dans la grille uniforme
        indexTri[idx] = pos.z * (int)(DATA_H / h) * (int)(DATA_W / h) + pos.y * (int)(DATA_W / h) + pos.x;
    }
}



//noyau Poly6
__device__ float Poly6(ParticulesField* part, int neighbor)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float result = 0.0f;

    // constante du noyau Poly6 : B3D = 315 / (64 * PI * h^9)
	float B3D = 315 / (64 * M_PI * pow(h, 9));

    // calcul de la distance entre la particule et son voisin pour chaque coordonnée de Xbarre
	float x = pow(part->point[idx].Position.getX() - part->point[neighbor].Position.getX(), 2);
	float y = pow(part->point[idx].Position.getY() - part->point[neighbor].Position.getY(), 2);
	float z = pow(part->point[idx].Position.getZ() - part->point[neighbor].Position.getZ(), 2);
	
    // calcul de la norme de la distance
    float r = x + y + z;

    // calcul du noyau Poly6
	if (r <= pow(h, 2) && r >= 0)
	{
		result = B3D * pow(pow(h, 2) - r, 3);
	}

	return result;
}

//noyau Spiky
__device__ float Spiky(ParticulesField* part, int neighbor)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float result = 0.0f;

    // constante du noyau Spiky : B3D = 15 / (M_PI * h^6)
	float B3D = 15 / (M_PI * pow(h, 6));

    // calcul de la distance entre la particule et son voisin pour chaque coordonnée de Xbarre
    float x = pow(part->point[idx].Position.getX() - part->point[neighbor].Position.getX(), 2);


	return result;
}

//calcul de la densité
__device__ void Density(ParticulesField* part, int* neighbor)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = 0; i < n_avg; i++)
    {
        int index = idx - threadIdx.x + i;
        if (neighbor[index] != 0)
        {
            part->density[idx] += part->masse[neighbor[index]] * Poly6(part, neighbor[index]);
        }
    }
}

//calcul de la force de pression
__device__ void Pressure(ParticulesField* part, int* neighbor)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    part->pressure[idx] = k0 * (part->density[idx] - rho0);

    for (int i = 0; i < n_avg; i++)
    {
        int index = idx - threadIdx.x + i;
        if (neighbor[index] != 0)
        {
            part->pressure[neighbor[index]] = k0 * (part->density[idx] - rho0);


            part->pressure_gradient[idx].setX(part->pressure_gradient[idx].getX() + part->masse[neighbor[index]] * ((part->pressure[idx] + part->pressure[neighbor[index]])
                / (2 * part->density[neighbor[index]])) * gradSpiky(part, neighbor[index]).getX());
            part->pressure_gradient[idx].setX(part->pressure_gradient[idx].getY() + part->masse[neighbor[index]] * ((part->pressure[idx] + part->pressure[neighbor[index]])
                / (2 * part->density[neighbor[index]])) * gradSpiky(part, neighbor[index]).getY());
            part->pressure_gradient[idx].setZ(part->pressure_gradient[idx].getZ() + part->masse[neighbor[index]] * ((part->pressure[idx] + part->pressure[neighbor[index]])
                / (2 * part->density[neighbor[index]])) * gradSpiky(part, neighbor[index]).getZ());
        }
    }
}


__global__ void loop(ParticulesField* part, int* indexTri, int* start, int* end)
{


}


int main()
{
    //création des particules, et des tableaux pour la recherche de voisinage
    ParticulesField* starting_particules = new ParticulesField();
    int* grille = new int[WORKINGSET];
    int* start = new int[DATA_W * DATA_H * DATA_W];
    int* end = new int[DATA_W * DATA_H * DATA_W];
       
    //Allocation de la mémoire GPU
    ParticulesField* gpuPart = nullptr;
    int* gpuGrille = nullptr;
    int* gpuStart = nullptr;
    int* gpuEnd = nullptr;

    gpuErrchk(cudaMalloc((void**)&gpuPart, sizeof(ParticulesField)));
    gpuErrchk(cudaMalloc((void**)&gpuGrille, sizeof(int) * WORKINGSET));
    gpuErrchk(cudaMalloc((void**)&gpuStart, sizeof(int) * SizeCube));
    gpuErrchk(cudaMalloc((void**)&gpuEnd, sizeof(int) * SizeCube));
    gpuErrchk(cudaMemcpy(gpuPart, starting_particules, sizeof(ParticulesField), cudaMemcpyHostToDevice));


    //Dimension de la grid
    dim3 grid(WORKINGSET/BlockSize);
    dim3 block(BlockSize);

    //Premier kernel, création de la grille 1D
    createGrid << <grid, block >> > (gpuPart, gpuGrille);

    gpuErrchk(cudaDeviceSynchronize());
        
    //Partie thrust pour trié la grille 1D
    thrust::device_ptr<int> dev_ptr(gpuGrille);
    thrust::sort(dev_ptr, dev_ptr + WORKINGSET);

    // 

    gpuErrchk(cudaDeviceSynchronize());

    

    gpuErrchk(cudaFree(gpuPart));
    gpuErrchk(cudaFree(gpuGrille));
    delete starting_particules;
    delete grille;
    

    return 0;
}



