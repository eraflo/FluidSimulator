#pragma once
#include "../cuh/ParticulesField.cuh"


HOSTDEVICE ParticulesField::ParticulesField()
{
	for (int i = 0; i < WORKINGSET; i++)
	{
		Vec3 Pos = Vec3();
		Pos.setX(GenCoord(0, DATA_W));
		Pos.setY(GenCoord(0, DATA_H));
		Pos.setZ(GenCoord(0, DATA_D));

		Vec4 Col = { 0.0f, 0.0f, 0.0f, 1.0f };
		point[i].Color = Col;
		point[i].Position = Pos;
		rad[i] = 0.2f;
		velocity[i] = { 0, 0, 0 };
		density[i] = 0.0f;
		masse[i] = (float)pow(2.0f * rad[i], 3) * 900;
		pressure_gradient[i] = { 0, 0, 0 };
		laplacien_viscosity[i] = { 0, 0, 0 };
		id[i] = i;
		
		for(int j = 0; j < n_avg; j++)
		{
			neighbors[i][j] = NULL;
		}

	}
}

// Function to generate random float number between min and max at the initialization of the particules
__host__ float ParticulesField::GenCoord(float min, float max)
{
	float nRand;
	nRand = min + ((float)rand() * (max - min + 1) / (RAND_MAX - 1));

	// Limit the number of decimal to 2
	nRand = floorf(nRand * 100) / 100;
	return nRand;
}

HOSTDEVICE Vertex ParticulesField::GetPoint(int i)
{
	return point[i];
}

HOSTDEVICE Vec3 ParticulesField::GetVelocity(int i)
{
	return velocity[i];
}

HOSTDEVICE float ParticulesField::GetRad(int i)
{
	return rad[i];
}

HOSTDEVICE float ParticulesField::GetDensity(int i)
{
	return density[i];
}

HOSTDEVICE float ParticulesField::GetMasse(int i)
{
	return masse[i];
}

HOSTDEVICE Vec3 ParticulesField::GetPressureGradient(int i)
{
	return pressure_gradient[i];
}

HOSTDEVICE Vec3 ParticulesField::GetLaplacienViscosity(int i)
{
	return laplacien_viscosity[i];
}

HOSTDEVICE float ParticulesField::GetTimestep()
{
	return timestep;
}

HOSTDEVICE int ParticulesField::GetId(int i)
{
	return id[i];
}

// Function to get the neighbors of each particule
__device__ void ParticulesField::NeighborsSearch()
{

}

//Poly6 core
__device__ float ParticulesField::Poly6(int neighbor)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float result = 0.0f;

	// constant of Poly6 core : B3D = 315 / (64 * PI * h^9)
	float B3D = 315 / (64 * M_PI * pow(h, 9));

	// Calcul of the distance between the particule and its neighbor for each coordinate of Xbarre
	float x = pow(this->point[idx].Position.getX() - this->point[neighbor].Position.getX(), 2);
	float y = pow(this->point[idx].Position.getY() - this->point[neighbor].Position.getY(), 2);
	float z = pow(this->point[idx].Position.getZ() - this->point[neighbor].Position.getZ(), 2);

	// Calcul of the norm of the distance
	float r = x + y + z;

	// Calcul of the Poly6 core
	if (r <= h && r >= 0)
	{
		result = B3D * pow(pow(h, 2) - pow(r, 2), 3);
	}

	return result;
}

__device__ Vec3 ParticulesField::SpikyGradient(int neighbor)
{
	// Index of the particule
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	// Gradient of the Spiky core

	// B3D = 45 / (M_PI * h^6)
	float B3D = 45 / (M_PI * pow(h, 6));

	// Calcul of the distance between the particule and its neighbor for each coordinate of Xbarre
	float x = this->point[idx].Position.getX() - this->point[neighbor].Position.getX();
	float y = this->point[idx].Position.getY() - this->point[neighbor].Position.getY();
	float z = this->point[idx].Position.getZ() - this->point[neighbor].Position.getZ();

	// Calcul of the norm of the distance
	float r = x + y + z;
	float abs_r = sqrt(pow(r, 2));

	// Calcul of the Spiky core gradient
	Vec3 result = { 0, 0, 0 };
	result.setX(-B3D * x/abs_r * pow(h - abs_r, 2));
	result.setY(-B3D * y/abs_r * pow(h - abs_r, 2));
	result.setZ(-B3D * z/abs_r * pow(h - abs_r, 2));

	return result;
}

__device__ Vec3 ParticulesField::ViscosityLaplacien(int neighbor)
{
	// Index of the particule
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	// Constant of the Viscosity core : B3D = 45 / (2 * M_PI * h^6)
	float B3D = 45 / (2 * M_PI * pow(h, 6));

	// Calcul of the distance between the particule and its neighbor for each coordinate of Xbarre
	float x = this->point[idx].Position.getX() - this->point[neighbor].Position.getX();
	float y = this->point[idx].Position.getY() - this->point[neighbor].Position.getY();
	float z = this->point[idx].Position.getZ() - this->point[neighbor].Position.getZ();

	// Calcul of the norm of the distance
	float r = x + y + z;
	float abs_r = sqrt(pow(r, 2));

	// Calcul of the Viscosity core gradient
	Vec3 result = { 0, 0, 0 };
	result.setX(-B3D * (h - abs_r));
	result.setY(-B3D * (h - abs_r));
	result.setZ(-B3D * (h - abs_r));

	return result;
}

//Density calculation
__device__ void ParticulesField::Density(int* neighbor)
{
	// Index of the particule
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	// Calcul for each neighbor of the particule (Assuming that there is an average of n_avg (32) neighbor)
	for (int i = 0; i < n_avg; i++)
	{
		// Index of the neighbor 
		int index = idx - threadIdx.x + i;

		if (neighbor[index] != 0)
		{
			// Calcul of the density
			this->density[idx] += this->masse[neighbor[index]] * Poly6(neighbor[index]);
		}
	}
}

//Pressure calculation
__device__ void ParticulesField::PressureForce(int* neighbor)
{
	// Index of the particule
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	// Pressure for the actual particule
	float p = k0 * (this->density[idx] - rho0);

	for(int i = 0; i < n_avg; i++)
	{
		// Index of the neighbor 
		int index = idx - threadIdx.x + i;

		if (neighbor[index] != 0)
		{
			// Calcul of the pressure for the neighbor
			float p_neigh = k0 * (this->density[neighbor[index]] - rho0);
			Vec3 SpikyGrad = SpikyGradient(neighbor[index]);

			// Calcul of the pressure gradient
			this->pressure_gradient[idx].setX(this->pressure_gradient[idx].getX() +
										this->masse[neighbor[index]] *
										(p + p_neigh) / (2 * this->density[neighbor[index]])
										* SpikyGrad.getX()
			);

			this->pressure_gradient[idx].setY(this->pressure_gradient[idx].getY() +
										this->masse[neighbor[index]] *
										(p + p_neigh) / (2 * this->density[neighbor[index]])
										* SpikyGrad.getY()
			);

			this->pressure_gradient[idx].setZ(this->pressure_gradient[idx].getZ() +
										this->masse[neighbor[index]] *
										(p + p_neigh) / (2 * this->density[neighbor[index]])
										* SpikyGrad.getZ()
			);
			
		}
	}
}

__device__ void ParticulesField::ViscosityForce(int* neighbor)
{
	// Index of the particule
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = 0; i < n_avg; i++)
	{
		// Index of the neighbor 
		int index = idx - threadIdx.x + i;

		if (neighbor[index] != 0)
		{
			// Calcul of the viscosity gradient
			Vec3 laplacien = ViscosityLaplacien(neighbor[index]);

			this->laplacien_viscosity[idx].setX(this->laplacien_viscosity[idx].getX() +
										this->masse[neighbor[index]] *
										(this->velocity[neighbor[index]].getX() - this->velocity[idx].getX()) /
										this->density[neighbor[index]] *
										laplacien.getX()
			);

			this->laplacien_viscosity[idx].setY(this->laplacien_viscosity[idx].getY() +
										this->masse[neighbor[index]] *
										(this->velocity[neighbor[index]].getY() - this->velocity[idx].getY()) /
										this->density[neighbor[index]] *
										laplacien.getY()
			);

			this->laplacien_viscosity[idx].setZ(this->laplacien_viscosity[idx].getZ() +
										this->masse[neighbor[index]] *
										(this->velocity[neighbor[index]].getZ() - this->velocity[idx].getZ()) /
										this->density[neighbor[index]] *
										laplacien.getZ()
			);
		}
	}

	this->laplacien_viscosity[idx].setX(this->laplacien_viscosity[idx].getX() * v);
	this->laplacien_viscosity[idx].setY(this->laplacien_viscosity[idx].getY() * v);
	this->laplacien_viscosity[idx].setZ(this->laplacien_viscosity[idx].getZ() * v);
}







