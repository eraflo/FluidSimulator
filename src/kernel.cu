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
#include "test/TestTexture2D.h"
#include "test/TestColor2D.h"
#include "test/TestColor3D.h"

#include "Entête/Const.h"

/*

int main()
{
    GLFWwindow* window;
    
    //size_t N = 256 * 1024;


    if (!glfwInit())
        return -1;

    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_SAMPLES, 4); //antialliasing
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Simulation de fluide", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

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

        //création d'un menu
        test::Test* currentTest = nullptr;
        test::TestMenu* testMenu = new test::TestMenu(currentTest);
        currentTest = testMenu;

        testMenu->RegisterTest<test::TestColor2D>("2D Color");
        testMenu->RegisterTest<test::TestColor3D>("3D Color");
        testMenu->RegisterTest<test::TestTexture2D>("2D Texture");

        

        while (!glfwWindowShouldClose(window))
        {
            GLCall(glClearColor(0.0f, 0.0f, 0.0f, 1.0f));
            renderer.Clear();


            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            if (currentTest)
            {
                currentTest->OnUpdate(0.0f);
                currentTest->OnRender();
                ImGui::Begin("Test");

                if (currentTest != testMenu && ImGui::Button("<-"))
                {
                    delete currentTest;
                    currentTest = testMenu;
                }
                currentTest->OnImGuiRender();
                
                ImGui::End();
            }
            

            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            glfwSwapBuffers(window);

            glfwPollEvents();
        }
        delete currentTest;
        if (currentTest != testMenu)
        {
            delete testMenu;
        }
    }

    

    //Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    return 0;
}



*/