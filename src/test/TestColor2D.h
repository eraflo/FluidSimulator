#pragma once

#include "Test.h"

#include "../../src/Entête/VertexBuffer.h"
#include "../../src/Entête/VertexBufferLayout.h"
#include "../../src/Entête/VertexArray.h"
#include "../../src/Entête/Texture.h"

#include <memory>

namespace test
{
	class TestColor2D : public Test
	{
	public:
		TestColor2D();
		~TestColor2D();

		void OnUpdate(float deltaTime) override;
		void OnRender() override;
		void OnImGuiRender() override;

	private:
		std::unique_ptr<VertexArray> m_VAO;
		std::unique_ptr<VertexBuffer> m_VBO;
		std::unique_ptr<IndexBuffer> m_IndexBuffer;
		std::unique_ptr<Shader> m_Shader;

		glm::vec3 m_TranslationA;
		glm::vec3 m_TranslationB;
		glm::mat4 m_Proj, m_View;
	};

}