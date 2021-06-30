#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <vector>

#include "boo/BooObject.hpp"
#include "boo/System.hpp"

#ifdef __SWITCH__
#include <ctype.h>
#endif

namespace boo {
struct IGraphicsCommandQueue;

/** Supported buffer uses */
enum class BufferUse { Null, Vertex, Index, Uniform };

/** Typeless graphics buffer */
struct IGraphicsBuffer : IObj {
  bool dynamic() const { return m_dynamic; }

protected:
  bool m_dynamic;
  explicit IGraphicsBuffer(bool dynamic) : m_dynamic(dynamic) {}
};

/** Static resource buffer for verts, indices, uniform constants */
struct IGraphicsBufferS : IGraphicsBuffer {
protected:
  IGraphicsBufferS() : IGraphicsBuffer(false) {}
};

/** Dynamic resource buffer for verts, indices, uniform constants */
struct IGraphicsBufferD : IGraphicsBuffer {
  virtual void load(const void* data, size_t sz) = 0;
  virtual void* map(size_t sz) = 0;
  virtual void unmap() = 0;

protected:
  IGraphicsBufferD() : IGraphicsBuffer(true) {}
};

/** Texture access types */
enum class TextureType { Static, StaticArray, Dynamic, Render, CubeRender };

/** Supported texture formats */
enum class TextureFormat { RGBA8, I8, I16, DXT1, DXT3, DXT5, BPTC };

/** Supported texture clamp modes */
enum class TextureClampMode { Invalid = -1, Repeat, ClampToWhite, ClampToBlack, ClampToEdge, ClampToEdgeNearest };

/** Typeless texture */
struct ITexture : IObj {
  TextureType type() const { return m_type; }

  /* Only applies on GL and Vulkan. Use shader semantics on other platforms */
  virtual void setClampMode(TextureClampMode mode) {}

protected:
  TextureType m_type;
  explicit ITexture(TextureType type) : m_type(type) {}
};

/** Static resource buffer for textures */
struct ITextureS : ITexture {
protected:
  ITextureS() : ITexture(TextureType::Static) {}
};

/** Static-array resource buffer for array textures */
struct ITextureSA : ITexture {
protected:
  ITextureSA() : ITexture(TextureType::StaticArray) {}
};

/** Dynamic resource buffer for textures */
struct ITextureD : ITexture {
  virtual void load(const void* data, size_t sz) = 0;
  virtual void* map(size_t sz) = 0;
  virtual void unmap() = 0;

protected:
  ITextureD() : ITexture(TextureType::Dynamic) {}
};

/** Resource buffer for render-target textures */
struct ITextureR : ITexture {
protected:
  ITextureR() : ITexture(TextureType::Render) {}
};

/** Resource buffer for cube render-target textures */
struct ITextureCubeR : ITexture {
protected:
  ITextureCubeR() : ITexture(TextureType::CubeRender) {}
};

/** Types of vertex attributes */
enum class VertexSemantic {
  None = 0,
  Position3,
  Position4,
  Normal3,
  Normal4,
  Color,
  ColorUNorm,
  UV2,
  UV4,
  Weight,
  ModelView,
  Position2,
  SemanticMask = 0xf,
  Instanced = 0x10
};
ENABLE_BITWISE_ENUM(VertexSemantic)

/** Used to create IVertexFormat */
struct VertexElementDescriptor {
  VertexSemantic semantic{};
  int semanticIdx = 0;
  VertexElementDescriptor() = default;
  VertexElementDescriptor(VertexSemantic s, int idx = 0) : semantic(s), semanticIdx(idx) {}
};

/** Structure for passing vertex format info for pipeline construction */
struct VertexFormatInfo {
  size_t elementCount = 0;
  const VertexElementDescriptor* elements = nullptr;

  VertexFormatInfo() = default;

  VertexFormatInfo(size_t sz, const VertexElementDescriptor* elem) : elementCount(sz), elements(elem) {}

  template <typename T>
  VertexFormatInfo(const T& tp) : elementCount(std::extent_v<T>), elements(tp) {}

  VertexFormatInfo(std::initializer_list<VertexElementDescriptor> l)
  : elementCount(l.size()), elements(std::move(l.begin())) {}
};

/** Opaque token for referencing a shader stage usable in a graphics pipeline */
struct IShaderStage : IObj {};

/** Opaque token for referencing a complete graphics pipeline state necessary
 *  to rasterize geometry (shaders and blending modes mainly) */
struct IShaderPipeline : IObj {
  virtual bool isReady() const = 0;
};

/** Opaque token serving as indirection table for shader resources
 *  and IShaderPipeline reference. Each renderable surface-material holds one
 *  as a reference */
struct IShaderDataBinding : IObj {};

/** Used wherever distinction of pipeline stages is needed */
enum class PipelineStage { Null, Vertex, Fragment, Geometry, Control, Evaluation };

/** Used by platform shader pipeline constructors */
enum class Primitive { Triangles, TriStrips, Patches };

/** Used by platform shader pipeline constructors */
enum class CullMode { None, Backface, Frontface };

/** Used by platform shader pipeline constructors */
enum class ZTest {
  None,
  LEqual, /* Flipped on Vulkan, D3D, Metal */
  Greater,
  GEqual,
  Equal
};

/** Used by platform shader pipeline constructors */
enum class BlendFactor {
  Zero,
  One,
  SrcColor,
  InvSrcColor,
  DstColor,
  InvDstColor,
  SrcAlpha,
  InvSrcAlpha,
  DstAlpha,
  InvDstAlpha,
  SrcColor1,
  InvSrcColor1,

  /* Special value that activates DstColor - SrcColor blending */
  Subtract
};

/** Structure for passing additional pipeline construction information */
struct AdditionalPipelineInfo {
  BlendFactor srcFac = BlendFactor::One;
  BlendFactor dstFac = BlendFactor::Zero;
  Primitive prim = Primitive::TriStrips;
  ZTest depthTest = ZTest::LEqual;
  bool depthWrite = true;
  bool colorWrite = true;
  bool alphaWrite = false;
  CullMode culling = CullMode::Backface;
  uint32_t patchSize = 0;
  bool overwriteAlpha = false;
  bool depthAttachment = true;
};

/** Factory object for creating batches of resources as an IGraphicsData token */
struct IGraphicsDataFactory {
  virtual ~IGraphicsDataFactory() = default;

  enum class Platform { Null, OpenGL, D3D11, Metal, Vulkan, GX, NX };
  virtual Platform platform() const = 0;
  virtual const SystemChar* platformName() const = 0;

  struct Context {
    virtual Platform platform() const = 0;
    virtual const SystemChar* platformName() const = 0;

    virtual ObjToken<IGraphicsBufferS> newStaticBuffer(BufferUse use, const void* data, size_t stride,
                                                       size_t count) = 0;
    virtual ObjToken<IGraphicsBufferD> newDynamicBuffer(BufferUse use, size_t stride, size_t count) = 0;

    virtual ObjToken<ITextureS> newStaticTexture(size_t width, size_t height, size_t mips, TextureFormat fmt,
                                                 TextureClampMode clampMode, const void* data, size_t sz) = 0;
    virtual ObjToken<ITextureSA> newStaticArrayTexture(size_t width, size_t height, size_t layers, size_t mips,
                                                       TextureFormat fmt, TextureClampMode clampMode, const void* data,
                                                       size_t sz) = 0;
    virtual ObjToken<ITextureD> newDynamicTexture(size_t width, size_t height, TextureFormat fmt,
                                                  TextureClampMode clampMode) = 0;
    virtual ObjToken<ITextureR> newRenderTexture(size_t width, size_t height, TextureClampMode clampMode,
                                                 size_t colorBindingCount, size_t depthBindingCount) = 0;
    virtual ObjToken<ITextureCubeR> newCubeRenderTexture(size_t width, size_t mips) = 0;

    virtual ObjToken<IShaderStage> newShaderStage(const uint8_t* data, size_t size, PipelineStage stage) = 0;

    ObjToken<IShaderStage> newShaderStage(const std::vector<uint8_t>& data, PipelineStage stage) {
      return newShaderStage(data.data(), data.size(), stage);
    }

    virtual ObjToken<IShaderPipeline> newShaderPipeline(ObjToken<IShaderStage> vertex, ObjToken<IShaderStage> fragment,
                                                        ObjToken<IShaderStage> geometry, ObjToken<IShaderStage> control,
                                                        ObjToken<IShaderStage> evaluation,
                                                        const VertexFormatInfo& vtxFmt,
                                                        const AdditionalPipelineInfo& additionalInfo,
                                                        bool asynchronous = true) = 0;

    ObjToken<IShaderPipeline> newShaderPipeline(ObjToken<IShaderStage> vertex, ObjToken<IShaderStage> fragment,
                                                const VertexFormatInfo& vtxFmt,
                                                const AdditionalPipelineInfo& additionalInfo,
                                                bool asynchronous = true) {
      return newShaderPipeline(std::move(vertex), std::move(fragment), {}, {}, {}, vtxFmt, additionalInfo,
                               asynchronous);
    }

    virtual ObjToken<IShaderDataBinding> newShaderDataBinding(
        const ObjToken<IShaderPipeline>& pipeline, const ObjToken<IGraphicsBuffer>& vbo,
        const ObjToken<IGraphicsBuffer>& instVbo, const ObjToken<IGraphicsBuffer>& ibo, size_t ubufCount,
        const ObjToken<IGraphicsBuffer>* ubufs, const PipelineStage* ubufStages, const size_t* ubufOffs,
        const size_t* ubufSizes, size_t texCount, const ObjToken<ITexture>* texs, const int* texBindIdx,
        const bool* depthBind, size_t baseVert = 0, size_t baseInst = 0) = 0;

    ObjToken<IShaderDataBinding> newShaderDataBinding(const ObjToken<IShaderPipeline>& pipeline,
                                                      const ObjToken<IGraphicsBuffer>& vbo,
                                                      const ObjToken<IGraphicsBuffer>& instVbo,
                                                      const ObjToken<IGraphicsBuffer>& ibo, size_t ubufCount,
                                                      const ObjToken<IGraphicsBuffer>* ubufs,
                                                      const PipelineStage* ubufStages, size_t texCount,
                                                      const ObjToken<ITexture>* texs, const int* texBindIdx,
                                                      const bool* depthBind, size_t baseVert = 0, size_t baseInst = 0) {
      return newShaderDataBinding(pipeline, vbo, instVbo, ibo, ubufCount, ubufs, ubufStages, nullptr, nullptr, texCount,
                                  texs, texBindIdx, depthBind, baseVert, baseInst);
    }
  };

  virtual void commitTransaction(const std::function<bool(Context& ctx)>& __BooTraceArgs) = 0;
  virtual void lazyCommitTransaction(const std::function<bool(Context& ctx)>& f __BooTraceArgs) {
    commitTransaction(f __BooTraceArgsUse);
  }
  virtual ObjToken<IGraphicsBufferD> newPoolBuffer(BufferUse use, size_t stride, size_t count __BooTraceArgs) = 0;
  virtual void setDisplayGamma(float gamma) = 0;
  virtual bool isTessellationSupported(uint32_t& maxPatchSizeOut) = 0;
  virtual void waitUntilShadersReady() = 0;
  virtual bool areShadersReady() = 0;
};

using GraphicsDataFactoryContext = IGraphicsDataFactory::Context;
using FactoryCommitFunc = std::function<bool(GraphicsDataFactoryContext& ctx)>;

} // namespace boo
