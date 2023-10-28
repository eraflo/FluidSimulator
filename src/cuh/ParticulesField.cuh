#pragma once
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../h/Geometry.h"
#include "../h/Const.h"
#include <vector>


class ParticulesField
{
	private:
		Vertex point[WORKINGSET];
		Vec3 velocity[WORKINGSET];
		float rad[WORKINGSET];
		float density[WORKINGSET];
		float pressure[WORKINGSET];
		float viscosity[WORKINGSET];
		float masse[WORKINGSET];
		Vec3 pressure_gradient[WORKINGSET];
		Vec3 laplacien_viscosity[WORKINGSET];
		float timestep[WORKINGSET];
		int nbneigh[WORKINGSET];
		int id[WORKINGSET];

	public:
		__device__ ParticulesField();

		float GenCoord(float min, float max);

		// Getters
		HOSTDEVICE Vertex GetPoint(int i);
		HOSTDEVICE Vec3 GetVelocity(int i);
		HOSTDEVICE float GetRad(int i);
		HOSTDEVICE float GetDensity(int i);
		HOSTDEVICE float GetPressure(int i);
		HOSTDEVICE float GetViscosity(int i);
		HOSTDEVICE float GetMasse(int i);
		HOSTDEVICE Vec3 GetPressureGradient(int i);
		HOSTDEVICE Vec3 GetLaplacienViscosity(int i);
		HOSTDEVICE float GetTimestep(int i);
		HOSTDEVICE int GetNbneigh(int i);
		HOSTDEVICE int GetId(int i);

};