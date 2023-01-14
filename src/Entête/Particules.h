#pragma once
#include <iostream>
#include <math.h>

struct Point
{
	float x, y, z;
};

class Particules
{
private:
	Point* center;
public:
	void draw();
	Point* getCenter();
	void setCenter(Point*);
};

