#pragma once
#include <iostream>
#include <string>
#include <unordered_map>

#include "glm/glm.hpp"

//Stockage du texte dans le fichier shader pour chaque shader
struct ShaderProgramSources {
	std::string VertexSource;
	std::string FragmentSource;
};



class Shader
{
private:
	std::string m_FilePath;
	unsigned int m_RendererID;

	int GetUniformLocation(const std::string& name);

	ShaderProgramSources ParseShader(const std::string& filepath);
	unsigned int CompileShader(unsigned int type, const std::string& source);
	unsigned int CreateShader(const std::string& vertexShader, const std::string& fragmentShader);


	//caching for uniforms
	std::unordered_map<std::string, int> m_UniformLocationCache;

public:
	Shader(const std::string& filepath);
	~Shader(void);


	void Bind(void) const;
	void Unbind(void) const;

	// Set uniforms
	void SetUniform4f(const std::string& name, float v0, float v1, float v2, float v3);
	void SetUniformi(const std::string& name, int value);
	void SetUniform1f(const std::string& name, float value);
	void SetUniformMat4f(const std::string& name, const glm::mat4& matrix);

};