#include "TestColor2D.h"

#include "../../src/Entête/Renderer.h"
#include "imgui/imgui.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"


#include "../../src/Entête/Const.h"

namespace test
{
	test::TestColor2D::TestColor2D()
		: m_TranslationA(200, 200, 0),
			m_TranslationB(400, 200, 0),
			m_Proj(glm::ortho(0.0f, (float)WIDTH, 0.0f, (float)HEIGHT, -1.0f, 1.0f)), 
			m_View(glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, 0)))
	{


		float positions[] = {
			-50.0f, -50.0f, 0.18f, 0.6f, 0.96f, 1.0f,
			50.0f,  -50.0f, 0.18f, 0.6f, 0.96f, 1.0f,
			50.0f,   50.0f, 0.18f, 0.6f, 0.96f, 1.0f,
			-50.0f,  50.0f, 0.18f, 0.6f, 0.96f, 1.0f,

			50.0f,	50.0f, 1.0, 0.93f, 0.24f, 1.0f,
			150.0f, 50.0f, 1.0, 0.93f, 0.24f, 1.0f,
			150.0f, 150.0f, 1.0, 0.93f, 0.24f, 1.0f,
			50.0f,  150.0f, 1.0, 0.93f, 0.24f, 1.0f
		};

		unsigned int indices[] = {
			0, 1, 2, 2, 3, 0,
			4, 5, 6, 6, 7, 4
		};

		//Blend mode pour garder la transparence
		GLCall(glEnable(GL_BLEND));
		GLCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

		//ajout des éléments à render dans un vertex buffer
		m_VAO = std::make_unique<VertexArray>();
		m_VBO = std::make_unique<VertexBuffer>(positions, 8 * 6 * sizeof(float));
		VertexBufferLayout layout;
		layout.Push<float>(2);
		layout.Push<float>(4);

		m_VAO->AddBuffer(*m_VBO, layout);
		m_IndexBuffer = std::make_unique<IndexBuffer>(indices, 12);




		//création du shader
		m_Shader = std::make_unique<Shader>("res/shaders/ColorShader.shader");
		m_Shader->Bind();

	}

	test::TestColor2D::~TestColor2D()
	{
	}

	void test::TestColor2D::OnUpdate(float deltaTime)
	{
	}

	void test::TestColor2D::OnRender()
	{
		GLCall(glClearColor(0.0f, 0.0f, 0.0f, 1.0f));
		GLCall(glClear(GL_COLOR_BUFFER_BIT));

		Renderer renderer;

		{
			glm::mat4 model = glm::translate(glm::mat4(1.0f), m_TranslationA);
			glm::mat4 mvp = m_Proj * m_View * model;
			m_Shader->Bind();
			m_Shader->SetUniformMat4f("u_MVP", mvp);
			renderer.Draw(*m_VAO, *m_IndexBuffer, *m_Shader);
		}

	}

	void test::TestColor2D::OnImGuiRender()
	{
		ImGui::SliderFloat3("Translation A", &m_TranslationA.x, 0.0f, (float)WIDTH);
		//ImGui::SliderFloat3("Translation B", &m_TranslationB.x, 0.0f, (float)WIDTH);
		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
	}

}
