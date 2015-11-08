#ifndef GDEV_GLES3_HPP
#define GDEV_GLES3_HPP

#include "IGraphicsDataFactory.hpp"
#include "IGraphicsCommandQueue.hpp"
#include "boo/IGraphicsContext.hpp"
#include <vector>
#include <unordered_set>

namespace boo
{

class GLDataFactory : public IGraphicsDataFactory
{
    IGraphicsContext* m_parent;
    IGraphicsData* m_deferredData = nullptr;
    std::unordered_set<IGraphicsData*> m_committedData;
public:
    GLDataFactory(IGraphicsContext* parent);
    ~GLDataFactory() {}

    Platform platform() const {return PlatformOGL;}
    const char* platformName() const {return "OpenGL";}

    IGraphicsBufferS* newStaticBuffer(BufferUse use, const void* data, size_t stride, size_t count);
    IGraphicsBufferS* newStaticBuffer(BufferUse use, std::unique_ptr<uint8_t[]>&& data, size_t stride, size_t count);
    IGraphicsBufferD* newDynamicBuffer(BufferUse use, size_t stride, size_t count);

    ITextureS* newStaticTexture(size_t width, size_t height, size_t mips, TextureFormat fmt,
                                const void* data, size_t sz);
    ITextureS* newStaticTexture(size_t width, size_t height, size_t mips, TextureFormat fmt,
                                std::unique_ptr<uint8_t[]>&& data, size_t sz);
    ITextureD* newDynamicTexture(size_t width, size_t height, TextureFormat fmt);
    ITextureR* newRenderTexture(size_t width, size_t height, size_t samples);

    IVertexFormat* newVertexFormat(size_t elementCount, const VertexElementDescriptor* elements);

    IShaderPipeline* newShaderPipeline(const char* vertSource, const char* fragSource,
                                       size_t texCount, const char** texNames,
                                       BlendFactor srcFac, BlendFactor dstFac,
                                       bool depthTest, bool depthWrite, bool backfaceCulling);

    IShaderDataBinding*
    newShaderDataBinding(IShaderPipeline* pipeline,
                         IVertexFormat* vtxFormat,
                         IGraphicsBuffer* vbo, IGraphicsBuffer* ebo,
                         size_t ubufCount, IGraphicsBuffer** ubufs,
                         size_t texCount, ITexture** texs);

    void reset();
    IGraphicsData* commit();
    void destroyData(IGraphicsData*);
    void destroyAllData();
};

}

#endif // GDEV_GLES3_HPP
