#pragma once
#include "Const.h"
#include <vector>

class Vec3
{
	private:
		float x, y, z;
	public:
		// constructor
		HOSTDEVICE Vec3() : x(0), y(0), z(0) {}
		HOSTDEVICE Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
		// Accessors
		HOSTDEVICE float getX() const { return this->x; }
		HOSTDEVICE float getY() const { return y; }
		HOSTDEVICE float getZ() const { return z; }
		HOSTDEVICE void setX(float x) { this->x = x; }
		HOSTDEVICE void setY(float y) { this->y = y; }
		HOSTDEVICE void setZ(float z) { this->z = z; }
};

class Vec4
{
	public:
		// constructor
		HOSTDEVICE Vec4() : x(0), y(0), z(0), w(0) {}
		HOSTDEVICE Vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
	private:
		float x, y, z, w;
};

struct Vertex
{
	Vec3 Position;
	Vec4 Color;
};

