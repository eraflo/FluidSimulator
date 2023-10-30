#pragma once
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../h/Const.h"
#include "Geometry.cuh"


class ParticulesField
{
	private:
		Vertex point[WORKINGSET];
		float rad[WORKINGSET];
		float density[WORKINGSET];
		float masse[WORKINGSET];
		Vec3 velocity[WORKINGSET];
		Vec3 pressure_gradient[WORKINGSET];
		Vec3 laplacien_viscosity[WORKINGSET];
		int neighbors[WORKINGSET][n_avg];
		int id[WORKINGSET];
		float timestep;

		__host__ float GenCoord(float min, float max);

	public:
		HOSTDEVICE ParticulesField();


		// Getters
		HOSTDEVICE Vertex GetPoint(int i);
		HOSTDEVICE Vec3 GetVelocity(int i);
		HOSTDEVICE float GetRad(int i);
		HOSTDEVICE float GetDensity(int i);
		HOSTDEVICE float GetMasse(int i);
		HOSTDEVICE Vec3 GetPressureGradient(int i);
		HOSTDEVICE Vec3 GetLaplacienViscosity(int i);
		HOSTDEVICE float GetTimestep();
		HOSTDEVICE int GetId(int i);

		// Fluid simulation functions

		// Neighbors search
		__device__ void NeighborsSearch();
		
		//Poly6 core (on 1 neighbor)
		__device__ float Poly6(int neighbor);

		//Spiky core gradient (on 1 neighbor)
		__device__ Vec3 SpikyGradient(int neighbor);

		//Viscosity core laplacien (on 1 neighbor)
		__device__ Vec3 ViscosityLaplacien(int neighbor);

		//Density calculation
		__device__ void Density(int* neighbor);

		//Pressure Force calculation
		__device__ void PressureForce(int* neighbor);

		//Viscosity Force calculation
		__device__ void ViscosityForce(int* neighbor);

};