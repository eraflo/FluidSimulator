#include "TestColor3D.h"

#include "../../src/Entête/Renderer.h"
#include "imgui/imgui.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"


#include "../../src/Entête/Const.h"
#include "../../src/Entête/Geometry.h"
#include <array>



namespace test
{

	test::TestColor3D::TestColor3D()
		: m_TranslationA(200, 200, 0),
			m_TranslationB(400, 200, 0),
			m_Proj(glm::ortho(0.0f, (float)WIDTH, 0.0f, (float)HEIGHT, -1.0f, 1.0f)), 
			m_View(glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, 0)))
	{

		const size_t MaxQuadCount = 1000;
		const size_t MaxVertexCount = MaxQuadCount * 4;
		const size_t MaxIndexCount = MaxQuadCount * 6;


		uint32_t indices[MaxIndexCount];
		uint32_t offset = 0;
		for (size_t i = 0; i < MaxIndexCount; i += 6)
		{
			indices[i] = 0 + offset;
			indices[i + 1] = 1 + offset;
			indices[i + 2] = 2 + offset;

			indices[i + 3] = 2 + offset;
			indices[i + 4] = 3 + offset;
			indices[i + 5] = 0 + offset;
			
			offset += 4;
		}

		/*
		unsigned int indices[] = {
			0,1,2, 0,2,3, 0,3,4, 0,4,5, 0,5,6, 0,6,1,
			1,6,7, 1,7,2, 7,4,3, 7,3,2, 4,7,6, 4,6,5
		};*/

		//GLCall(glEnable(GL_DEPTH_TEST));

		//Blend mode pour garder la transparence
		GLCall(glEnable(GL_BLEND));
		GLCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

		//ajout des éléments à render dans un vertex buffer
		m_VAO = std::make_unique<VertexArray>();
		m_VBO = std::make_unique<VertexBuffer>(nullptr, sizeof(Vertex) * MaxQuadCount);
		VertexBufferLayout layout;
		layout.Push<float>(2);
		layout.Push<float>(4);


		m_VAO->AddBuffer(*m_VBO, layout);
		m_IndexBuffer = std::make_unique<IndexBuffer>(indices, MaxIndexCount);


		//création du shader
		m_Shader = std::make_unique<Shader>("res/shaders/ColorShader.shader");
		m_Shader->Bind();

	}

	test::TestColor3D::~TestColor3D()
	{
		GLCall(glDisable(GL_DEPTH_TEST));
	}

	static Vertex* CreateQuad(Vertex* target, float x, float y, float z)
	{
		float size = 100.0f;

		target->Position = { x, y, z };
		target->Color = { 0.18f, 0.6f, 0.96f, 1.0f };
		target++;

		target->Position = { x + size, y, z };
		target->Color = { 0.18f, 0.6f, 0.96f, 1.0f };
		target++;

		target->Position = { x + size, y + size, z };
		target->Color = { 0.18f, 0.6f, 0.96f, 1.0f };
		target++;

		target->Position = { x, y + size, z };
		target->Color = { 0.18f, 0.6f, 0.96f, 1.0f };
		target++;

		return target;
	}



	void test::TestColor3D::OnUpdate(float deltaTime)
	{
		float positions[] = {
			 50.0f,   50.0f,  50.0f, 0.18f, 0.6f, 0.96f, 1.0f,
			-50.0f,   50.0f,  50.0f, 0.18f, 0.6f, 0.96f, 1.0f,
			-50.0f,  -50.0f,  50.0f, 0.18f, 0.6f, 0.96f, 1.0f,
			 50.0f,  -50.0f,  50.0f, 0.18f, 0.6f, 0.96f, 1.0f,
			 50.0f,  -50.0f,  -50.0f, 0.18f, 0.6f, 0.96f, 1.0f,
			 50.0f,   50.0f,  -50.0f, 0.18f, 0.6f, 0.96f, 1.0f,
			-50.0f,   50.0f,  -50.0f, 0.18f, 0.6f, 0.96f, 1.0f,
			-50.0f,  -50.0f,  -50.0f, 0.18f, 0.6f, 0.96f, 1.0f,
		};


		std::array<Vertex, 1000> vertices;
		Vertex* buffer = vertices.data();

		for (int y = 0; y < 50; y += 10)
		{
			for (int x = 0; x < 50; x += 10) {
				buffer = CreateQuad(buffer, x, y, 0.0f);
				indexCount += 6;
			}
		}


		buffer = CreateQuad(buffer, m_QuadPosition[0], m_QuadPosition[1], 0.0f);
		indexCount += 6;


		GLCall(glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(Vertex), vertices.data()));
	}

	void test::TestColor3D::OnRender()
	{
		GLCall(glClearColor(0.0f, 0.0f, 0.0f, 1.0f));
		//this->OnUpdate(1 / ImGui::GetIO().Framerate);
		GLCall(glClear(GL_COLOR_BUFFER_BIT));

		Renderer renderer;


		{
			glm::mat4 m_Model = glm::translate(glm::mat4(1.0f), m_TranslationA);
			glm::mat4 mvp = m_Proj * m_View * m_Model;
			m_Shader->Bind();
			m_Shader->SetUniformMat4f("u_MVP", mvp);
			renderer.Draw(*m_VAO, *m_IndexBuffer, *m_Shader);
		}
		
	}

	void test::TestColor3D::OnImGuiRender()
	{
		ImGui::SliderFloat3("Translation A", &m_TranslationA.x, 0.0f, (float)WIDTH);
		ImGui::DragFloat2("Translation Quad 1", m_QuadPosition, 0.1f);
		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
	}

}
