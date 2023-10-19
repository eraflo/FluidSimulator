#pragma once
#include <math.h>

//dimensionnement fenêtre
#define WIDTH 1280
#define HEIGHT 720

//GPU
#define WORKINGSET 2000
#define BlockSize 256

//Simulation
#define n_avg 32
# define M_PI 3.14159265358979323846 
#define h 0.8
#define Temperature 20
#define R_boltz 8.31446261815324
#define M_mol 0.018
#define rho0 1
#define k0 (Temperature*R_boltz*M_mol)

//Cube
#define Ne 32
#define iters 10
#define DATA_W 8
#define DATA_H 8
#define DATA_D 8
#define IX(x, y, z) ((x) + (y) * (DATA_W/h) + (z) * (DATA_W/h) * (DATA_H/h))
#define SizeCube DATA_W*DATA_H*DATA_D
