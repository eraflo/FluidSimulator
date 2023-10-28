#include "../cuh/ParticulesField.cuh"


__device__ ParticulesField::ParticulesField()
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
		rad[i] = 0.2;
		velocity[i] = { 0, 0, 0 };
		density[i] = 0.0f;
		pressure[i] = ((Temperature * R_boltz) / M_mol) * (density[i] - rho0);
		viscosity[i] = 0.5f;
		masse[i] = pow(2.0f * rad[i], 3) * 900;
		pressure_gradient[i] = { 1, 1, 1 };
		laplacien_viscosity[i] = { 1, 1, 1 };
		nbneigh[i] = 0;
		id[i] = i;
	}
}

// Function to generate random float number between min and max at the initialization of the particules
float ParticulesField::GenCoord(float min, float max)
{
	float nRand;
	nRand = min + ((float)rand() * (max - min + 1) / (RAND_MAX - 1));
	return nRand;
}

HOSTDEVICE Vertex ParticulesField::GetPoint(int i)
{
	return point[i];
}
