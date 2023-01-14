#include "Entête/VertexBuffer.h"

#include "Entête/Renderer.h"


VertexBuffer::VertexBuffer(const void* data, unsigned int size)
{
    GLCall(glGenBuffers(1, &m_RendererID)); //donne un id (unique) au buffer
    GLCall(glBindBuffer(GL_ARRAY_BUFFER, m_RendererID)); //selection du buffer pour l'utiliser
    GLCall(glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW)); //taille du buffer
}

VertexBuffer::~VertexBuffer()
{
    GLCall(glDeleteBuffers(1, &m_RendererID));
}

void VertexBuffer::Bind() const
{
    GLCall(glBindBuffer(GL_ARRAY_BUFFER, m_RendererID)); //selection du buffer pour l'utiliser
}


void VertexBuffer::Unbind() const
{
    GLCall(glBindBuffer(GL_ARRAY_BUFFER, 0)); //selection du buffer pour l'utiliser
}