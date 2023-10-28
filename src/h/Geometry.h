#pragma once
#include "Const.h"
#include <vector>

class Vec3
{
	private:
		float x, y, z;
	public:
		// constructor
		__host__ __device__ Vec3() : x(0), y(0), z(0) {}
		__host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
		// Accessors
		__host__ __device__ float getX() const { return this->x; }
		__host__ __device__ float getY() const { return y; }
		__host__ __device__ float getZ() const { return z; }
		__host__ __device__ void setX(float x) { this->x = x; }
		__host__ __device__ void setY(float y) { this->y = y; }
		__host__ __device__ void setZ(float z) { this->z = z; }
};

class Vec4
{
	public:
		// constructor
		__host__ __device__ Vec4() : x(0), y(0), z(0), w(0) {}
		__host__ __device__ Vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
	private:
		float x, y, z, w;
};

struct Vertex
{
	Vec3 Position;
	Vec4 Color;
};

