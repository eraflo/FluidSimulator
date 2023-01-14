#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Dependencies/GLEW/include/GL/glew.h"
#include "../Dependencies/GLFW/include/GLFW/glfw3.h"
//à bien inclure après glew
#include "cuda_gl_interop.h"


#include "Entête/Renderer.h"


#include "Entête/Texture.h"
#include "Entête/VertexBuffer.h"
#include "Entête/VertexBufferLayout.h"
#include "Entête/IndexBuffer.h"
#include "Entête/VertexArray.h"
#include "Entête/Shader.h"
#include "../src/Entête/Particules.h"

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include <Windows.h>


#include <iostream>
#include <fstream>
#include <string>
#include <sstream>


#include <random>
#include <assert.h>
#include <iomanip>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

#include "test/TestClearColor.h"

#define WIDTH 1280
#define HEIGHT 720
#define BlockSize 256



int main()
{
    GLFWwindow* window;
    
    Particules* particule = new Particules();
    Point* center = new Point();
    center->x = 0.0f;
    center->y = 0.0f;
    center->z = 0.0f;
    particule->setCenter(center);
    //size_t N = 256 * 1024;


    /* Initialize the library */
    if (!glfwInit())
        return -1;

    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_SAMPLES, 4); //antialliasing
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(WIDTH, HEIGHT, "Simulation de fluide", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (glewInit() != GLEW_OK)
        std::cout << "Error!" << std::endl;
    {
        

        GLCall(glEnable(GL_BLEND));
        GLCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

        
        Renderer renderer;

        //Interface
        ImGui::CreateContext();
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init(glsl_version);
        ImGui::StyleColorsDark();

        test::TestClearColor test;

        /* Loop until the user closes the window */
        while (!glfwWindowShouldClose(window))
        {
            renderer.Clear();

            test.OnUpdate(0.0f);
            test.OnRender();

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            
            test.OnImGuiRender();

            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            glfwSwapBuffers(window);

            glfwPollEvents();
        }
    }

    //Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    return 0;
}



