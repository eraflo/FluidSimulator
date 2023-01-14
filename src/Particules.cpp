#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Dependencies/GLEW/include/GL/glew.h"
#include "../Dependencies/GLFW/include/GLFW/glfw3.h"
#include "cuda_gl_interop.h"
#include "Entête/Particules.h"
#define M_PI 3.14159265358979323846



void Particules::draw()
{
    int i, j;
    double r = 2;
    int lats = 20;
    int longs = 20;
    for (i = 0; i <= lats; i++) {
        double lat0 = M_PI * (-0.5 + (double)(i - 1) / lats);
        double z0 = sin(lat0);
        double zr0 = cos(lat0);

        double lat1 = M_PI * (-0.5 + (double)i / lats);
        double z1 = sin(lat1);
        double zr1 = cos(lat1);

        glBegin(GL_QUAD_STRIP);
        for (j = 0; j <= longs; j++) {
            double lng = 2 * M_PI * (double)(j - 1) / longs;
            double x = cos(lng);
            double y = sin(lng);

            glNormal3f(x * zr0, y * zr0, z0);
            glVertex3f(r * x * zr0, r * y * zr0, r * z0);
            glNormal3f(x * zr1, y * zr1, z1);
            glVertex3f(r * x * zr1, r * y * zr1, r * z1);
        }
        glEnd();
    }
}

Point* Particules::getCenter()
{
    return this->center;
}

void Particules::setCenter(Point* cent)
{
    this->center = cent;
}
