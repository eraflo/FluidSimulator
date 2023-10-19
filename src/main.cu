#pragma once
#include <cuda.h>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include "../Dependencies/GLEW/include/GL/glew.h"
#include "../Dependencies/GLFW/include/GLFW/glfw3.h"
//à bien inclure après glew
#include "cuda_gl_interop.h"
#include "device_functions.h"
#include "device_atomic_functions.h"



#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include "thrust/sort.h"
#include "thrust/copy.h"

#include "Entête/Renderer.h"


#include "Entête/Texture.h"
#include "Entête/VertexBuffer.h"
#include "Entête/VertexBufferLayout.h"
#include "Entête/IndexBuffer.h"
#include "Entête/VertexArray.h"
#include "Entête/Shader.h"

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include <Windows.h>


#include <iostream>
#include <fstream>
#include <string>
#include <sstream>


#include <random>
#include <assert.h>
#include <math.h>
#include <iomanip>
#include <vector>
#include <algorithm>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

#include "test/TestClearColor.h"
#include "test/TestTexture2D.h"
#include "test/TestColor2D.h"
#include "test/TestColor3D.h"

#include "Entête/Const.h"
#include "Entête/Geometry.h"
#include "Entête/Particule.cuh"

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


//fonction pour utiliser la souris lorsque la partie graphique sera implémenté
glm::vec3* Mouse(GLFWwindow* window, double xpos, double ypos, float deltaTime, 
    float horizontalAngle, float verticalAngle, float mouseSpeed)
{
    glm::vec3* liste = new glm::vec3[3];
    glfwGetCursorPos(window, &xpos, &ypos);
    horizontalAngle += mouseSpeed * deltaTime * float(1024 / 2 - xpos);
    verticalAngle += mouseSpeed * deltaTime * float(768 / 2 - ypos);

    glm::vec3 direction(
        cos(verticalAngle) * sin(horizontalAngle),
        sin(verticalAngle),
        cos(verticalAngle) * cos(horizontalAngle)
    );


    glm::vec3 right = glm::vec3(
        sin(horizontalAngle - 3.14f / 2.0f),
        0,
        cos(horizontalAngle - 3.14f / 2.0f)
    );

    glm::vec3 up = glm::cross(right, direction);


    liste[0] = direction;
    liste[1] = right;
    liste[2] = up;

    return liste;
}


//fonction pour créer notre grille 1D avec les indices de case où se trouvent chaque particule
__global__ void createGrid(Particule* part, int* indexTri)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < WORKINGSET)
    {
        int3 pos;
        pos.x = floor(part->point[idx].Position.x / h);
        pos.y = floor(part->point[idx].Position.y / h);
        pos.z = floor(part->point[idx].Position.z / h);

        indexTri[idx] = pos.z * (int)(DATA_H / h) * (int)(DATA_W / h) + pos.y * (int)(DATA_W / h) + pos.x;
    }
}

//tentative de fonction pour le calcul du voisinage
/*
__device__ void Voisinage(Particule* part, int* indexTri, int* start, int* end)
{
   int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < WORKINGSET)
    {
        EndStart(indexTri, start, end);
        __syncthreads();
        
        for (int i = 0; i < WORKINGSET; i++)
        {
            int index = indexTri[i];
            if (index == idx) continue;
            if (index != my || index != my + 1 || index != my - 1) continue;
            float x = part[index].point.Position.x - part[idx].point.Position.x;
            float y = part[index].point.Position.y - part[idx].point.Position.y;
            float z = part[index].point.Position.z - part[idx].point.Position.z;
            float r = pow(x, 2) + pow(y, 2) + pow(z, 2);

            if (r <= pow(h, 2) && r >= 0)
            {
                int index1 = idx - threadIdx.x + part[idx].nbneigh;
                neighbor[index1] = indexTri[i];
                atomicAdd(&part[idx].nbneigh, 1);
            }
        }
        
    }
}*/
/* Tentative de fonction pour calculer les indices de début et de fin de des particules d'une case
__device__ void EndStart(int* indexTri, int* start, int* end)
{
    __shared__ int current;
    __shared__ int current_seq;
    __shared__ int i;
    if (threadIdx.x == 0) {
        current = indexTri[0];
        current_seq = 0;
        i = 1;
    }
    __syncthreads();
    if (threadIdx.x + blockIdx.x * blockDim.x < WORKINGSET) {
        if (indexTri[threadIdx.x + blockIdx.x * blockDim.x] != current) {
            end[current_seq] = threadIdx.x + blockIdx.x * blockDim.x - 1;
            current = indexTri[threadIdx.x + blockIdx.x * blockDim.x];
            current_seq++;
            start[current_seq] = threadIdx.x + blockIdx.x * blockDim.x;
        }
    }
    __syncthreads();
    if (threadIdx.x == blockDim.x - 1) {
        end[current_seq] = WORKINGSET - 1;
    }
}
*/


//noyau Poly6
__device__ float Poly6(Particule* part, int neighbor)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float W = 315 / (64 * M_PI * pow(h, 9));
    float mul = 0;
    float x = pow(part->point[idx].Position.x - part->point[neighbor].Position.x, 2);
    float y = pow(part->point[idx].Position.y - part->point[neighbor].Position.y, 2);
    float z = pow(part->point[idx].Position.z - part->point[neighbor].Position.z, 2);
    float r = x + y + z;
    if (r <= pow(h, 2) && r >= 0)
    {
        mul = pow(pow(h, 2) - pow(r, 2), 3);
    }
    return mul * W;
}

//gradient du noyau Spiky
__device__ Vec3 gradSpiky(Particule* part, int neighbor)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    Vec3 res = { 0.0f, 0.0f, 0.0f };

    float W = 15 / (M_PI * pow(h, 6));
    float x = pow(part->point[idx].Position.x - part->point[neighbor].Position.x, 2);
    float y = pow(part->point[idx].Position.y - part->point[neighbor].Position.y, 2);
    float z = pow(part->point[idx].Position.z - part->point[neighbor].Position.z, 2);
    float r = x + y + z;
    if (r <= pow(h, 2) && r >= 0)
    {
        float rl = sqrt(r);
        float hr = h - rl;
        float hr2 = pow(hr, 2);
        res.x = W * hr2 * (part->point[idx].Position.x / rl);
        res.y = W * hr2 * (part->point[idx].Position.y / rl);
        res.z = W * hr2 * (part->point[idx].Position.z / rl);
    }

    return res;
}

//calcul de la densité
__device__ void Density(Particule* part, int* neighbor)
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
__device__ void Pressure(Particule* part, int* neighbor)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    part->pressure[idx] = k0 * (part->density[idx] - rho0);

    for (int i = 0; i < n_avg; i++)
    {
        int index = idx - threadIdx.x + i;
        if (neighbor[index] != 0)
        {
            part->pressure[neighbor[index]] = k0 * (part->density[idx] - rho0);


            part->pressure_gradient[idx].x += part->masse[neighbor[index]] * ((part->pressure[idx] + part->pressure[neighbor[index]])
                / (2 * part->density[neighbor[index]])) * gradSpiky(part, neighbor[index]).x;
            part->pressure_gradient[idx].y += part->masse[neighbor[index]] * ((part->pressure[idx] + part->pressure[neighbor[index]])
                / (2 * part->density[neighbor[index]])) * gradSpiky(part, neighbor[index]).y;
            part->pressure_gradient[idx].z += part->masse[neighbor[index]] * ((part->pressure[idx] + part->pressure[neighbor[index]])
                / (2 * part->density[neighbor[index]])) * gradSpiky(part, neighbor[index]).z;
        }
    }
}

//ne fonctionne pas pour le moment, contiendra la logique à répéter pendant tout le programme
__global__ void loop(Particule* part, int* indexTri, int* start, int* end)
{
    //Voisinage(part, indexTri, start, end);
    //__syncthreads();
    //Density(part, neighbor);
    //__syncthreads();
    //Pressure(part, neighbor);

}


int main()
{
    GLFWwindow* window;
    // position
    glm::vec3 position = glm::vec3(0, 0, 5);
    
    //variables pour la partie graphique futur
    float horizontalAngle = 3.14f;
    float verticalAngle = 0.0f;
    float FoV = 45.0f;

    float speed = 3.0f; 
    float mouseSpeed = 0.005f;
    float deltaTime = 0;
       
    /* Initialize the library */
    if (!glfwInit())
        return -1;
    
    //Création de la fenêtre glfw
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_SAMPLES, 4); //antialliasing
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glEnable(GL_CULL_FACE);

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(WIDTH, HEIGHT, "Simulation de fluide", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }
    
    

    /* Make the window's context current */
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (glewInit() != GLEW_OK)
        std::cout << "Error!" << std::endl;
    {
        //création des particules, et des tableaux pour la recherche de voisinage
        Particule* starting_particules = new Particule();
        int* grille = new int[WORKINGSET];
        int* start = new int[DATA_W * DATA_H * DATA_W];
        int* end = new int[DATA_W * DATA_H * DATA_W];

        //Pour la représentation graphique
        double xpos, ypos;
       
        //Allocation de la mémoire GPU
        Particule* gpuPart = nullptr;
        int* gpuGrille = nullptr;
        int* gpuStart = nullptr;
        int* gpuEnd = nullptr;

        gpuErrchk(cudaMalloc((void**)&gpuPart, sizeof(Particule)));
        gpuErrchk(cudaMalloc((void**)&gpuGrille, sizeof(int) * WORKINGSET));
        gpuErrchk(cudaMalloc((void**)&gpuStart, sizeof(int) * SizeCube));
        gpuErrchk(cudaMalloc((void**)&gpuEnd, sizeof(int) * SizeCube));
        gpuErrchk(cudaMemcpy(gpuPart, starting_particules, sizeof(Particule), cudaMemcpyHostToDevice));


        //Dimension de la grid
        dim3 grid(WORKINGSET/BlockSize);
        dim3 block(BlockSize);

        //Premier kernel, création de la grille 1D
        createGrid << <grid, block >> > (gpuPart, gpuGrille);

        gpuErrchk(cudaDeviceSynchronize());
        
        //Partie thrust pour trié la grille 1D
        thrust::device_vector<int> d_vec(WORKINGSET);
        thrust::copy(gpuGrille, gpuGrille + WORKINGSET, d_vec.begin());

        thrust::sort_by_key(d_vec.begin(), d_vec.end(), gpuPart->id);
        //thrust::sort(d_vec.begin(), d_vec.end());

        //vrai prog
        //thrust::copy(d_vec.begin(), d_vec.end(), gpuGrille);

        //Pour l'affichage de la grille triée
        thrust::copy(d_vec.begin(), d_vec.end(), grille);

        cudaMemcpy(starting_particules, gpuPart, sizeof(Particule), cudaMemcpyDeviceToHost);

        
        //affichage grille triée
        for (int i = 0; i < 500; i++)
        {
            int index = starting_particules->id[i];
            std::cout << grille[i] 
                << " " << index
                << " " << starting_particules->point[index].Position.x
                << " " << starting_particules->point[index].Position.y
                << " " << starting_particules->point[index].Position.z
                << std::endl;
        }

        gpuErrchk(cudaDeviceSynchronize());

        //Ne fonctionne pas encore
        //loop << <grid, block >> > (gpuPart, gpuGrille, gpuStart, gpuEnd);

        //gpuErrchk(cudaDeviceSynchronize());

        /*
        gpuErrchk(cudaMemcpy(end, gpuEnd, sizeof(int) * SizeCube, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(start, gpuStart, sizeof(int) * SizeCube, cudaMemcpyDeviceToHost));

        for (int i = 0; i < 10; i++)
        {
            std::cout << end[i] << std::endl;
            std::cout << start[i] << std::endl;
        }
        */

        

        /* Loop until the user closes the window */
            while (!glfwWindowShouldClose(window))
            {
                
            /*
            double currentTime = glfwGetTime();
            float deltaTime = float(currentTime - deltaTime);

            glm::vec3* liste = Mouse(window, xpos, ypos, deltaTime, horizontalAngle,
                verticalAngle, mouseSpeed);



            // Move forward
            if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
                position += liste[0] * deltaTime * speed;
            }
            // Move backward
            if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
                position -= liste[0] * deltaTime * speed;
            }
            // Strafe right
            if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
                position += liste[1] * deltaTime * speed;
            }
            // Strafe left
            if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
                position -= liste[1] * deltaTime * speed;
            }
            glm::mat4 ProjectionMatrix = glm::perspective(glm::radians(FoV), 4.0f / 3.0f, 0.1f, 100.0f);
            // Camera matrix
            glm::mat4 ViewMatrix = glm::lookAt(
                position,           // Camera is here
                position + liste[0], // and looks here : at the same position, plus "direction"
                liste[2]                  // Head is up (set to 0,-1,0 to look upside-down)
            );


            glfwSetCursorPos(window, WIDTH / 2, HEIGHT / 2);*/
            glfwSwapBuffers(window);

            glfwPollEvents();
        }
        gpuErrchk(cudaFree(gpuPart));
        gpuErrchk(cudaFree(gpuGrille));
        delete starting_particules;
        delete grille;
    }


    glfwTerminate();
    return 0;
}



