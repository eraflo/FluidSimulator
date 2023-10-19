#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Geometry.h"
#include "Const.h"
#include <vector>
#include "../../Dependencies/GLEW/include/GL/glew.h"


class Particule
{
public:
	Vertex point[WORKINGSET];
	Vec3 velocity[WORKINGSET];
	GLfloat rad[WORKINGSET];
	GLfloat density[WORKINGSET];
	GLfloat pressure[WORKINGSET];
	GLfloat viscosity[WORKINGSET];
	GLfloat masse[WORKINGSET];
	Vec3 pressure_gradient[WORKINGSET];
	Vec3 laplacien_viscosity[WORKINGSET];
	GLfloat timestep[WORKINGSET];
	GLint nbneigh[WORKINGSET];
	GLint id[WORKINGSET];
	Particule() 
	{
		for (int i = 0; i < WORKINGSET; i++)
		{
			Vec3 Pos = { GenCoord(0, DATA_W), GenCoord(0, DATA_H), GenCoord(0, DATA_D) };
			Vec4 Col = { 0.0f, 0.0f, 0.0f, 1.0f };
			point[i].Color = Col;
			point[i].Position = Pos;
			rad[i] = 0.2;
			velocity[i].x = 0.0f;
			velocity[i].y = 0.0f;
			velocity[i].z = 0.0f;
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
	float GenCoord(float min, float max)
	{
		float nRand;
		nRand = min + ((float)rand() * (max - min + 1) / (RAND_MAX - 1));
		return nRand;
	}
};