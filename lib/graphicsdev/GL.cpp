#include "boo/graphicsdev/GL.hpp"
#include "boo/graphicsdev/glew.h"
#include "boo/IGraphicsContext.hpp"
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <LogVisor/LogVisor.hpp>

namespace boo
{
static LogVisor::LogModule Log("boo::GL");

struct GLData : IGraphicsData
{
    std::vector<std::unique_ptr<class GLShaderPipeline>> m_SPs;
    std::vector<std::unique_ptr<struct GLShaderDataBinding>> m_SBinds;
    std::vector<std::unique_ptr<class GLGraphicsBufferS>> m_SBufs;
    std::vector<std::unique_ptr<class GLGraphicsBufferD>> m_DBufs;
    std::vector<std::unique_ptr<class GLTextureS>> m_STexs;
    std::vector<std::unique_ptr<class GLTextureD>> m_DTexs;
    std::vector<std::unique_ptr<class GLTextureR>> m_RTexs;
    std::vector<std::unique_ptr<struct GLVertexFormat>> m_VFmts;
};

static const GLenum USE_TABLE[] =
{
    GL_INVALID_ENUM,
    GL_ARRAY_BUFFER,
    GL_ELEMENT_ARRAY_BUFFER,
    GL_UNIFORM_BUFFER
};

class GLGraphicsBufferS : public IGraphicsBufferS
{
    friend class GLDataFactory;
    friend struct GLCommandQueue;
    GLuint m_buf;
    GLenum m_target;
    GLGraphicsBufferS(BufferUse use, const void* data, size_t sz)
    {
        m_target = USE_TABLE[int(use)];
        glGenBuffers(1, &m_buf);
        glBindBuffer(m_target, m_buf);
        glBufferData(m_target, sz, data, GL_STATIC_DRAW);
    }
public:
    ~GLGraphicsBufferS() {glDeleteBuffers(1, &m_buf);}

    void bindVertex() const
    {glBindBuffer(GL_ARRAY_BUFFER, m_buf);}
    void bindIndex() const
    {glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_buf);}
    void bindUniform(size_t idx) const
    {glBindBufferBase(GL_UNIFORM_BUFFER, idx, m_buf);}
};

class GLGraphicsBufferD : public IGraphicsBufferD
{
    friend class GLDataFactory;
    friend struct GLCommandQueue;
    struct GLCommandQueue* m_q;
    GLuint m_bufs[3];
    GLenum m_target;
    void* m_mappedBuf = nullptr;
    size_t m_mappedSize = 0;
    GLGraphicsBufferD(GLCommandQueue* q, BufferUse use)
    : m_q(q)
    {
        m_target = USE_TABLE[int(use)];
        glGenBuffers(3, m_bufs);
    }
public:
    ~GLGraphicsBufferD() {glDeleteBuffers(3, m_bufs);}

    void load(const void* data, size_t sz);
    void* map(size_t sz);
    void unmap();

    void bindVertex() const;
    void bindIndex() const;
    void bindUniform(size_t idx) const;
};

IGraphicsBufferS*
GLDataFactory::newStaticBuffer(BufferUse use, const void* data, size_t stride, size_t count)
{
    GLGraphicsBufferS* retval = new GLGraphicsBufferS(use, data, stride * count);
    static_cast<GLData*>(m_deferredData)->m_SBufs.emplace_back(retval);
    return retval;
}

IGraphicsBufferS*
GLDataFactory::newStaticBuffer(BufferUse use, std::unique_ptr<uint8_t[]>&& data, size_t stride, size_t count)
{
    std::unique_ptr<uint8_t[]> d = std::move(data);
    GLGraphicsBufferS* retval = new GLGraphicsBufferS(use, d.get(), stride * count);
    static_cast<GLData*>(m_deferredData)->m_SBufs.emplace_back(retval);
    return retval;
}

class GLTextureS : public ITextureS
{
    friend class GLDataFactory;
    GLuint m_tex;
    GLTextureS(size_t width, size_t height, size_t mips,
               TextureFormat fmt, const void* data, size_t sz)
    {
        const uint8_t* dataIt = static_cast<const uint8_t*>(data);
        glGenTextures(1, &m_tex);
        glBindTexture(GL_TEXTURE_2D, m_tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        if (mips > 1)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        else
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        if (fmt == TextureFormat::RGBA8)
        {
            for (size_t i=0 ; i<mips ; ++i)
            {
                glTexImage2D(GL_TEXTURE_2D, i, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, dataIt);
                dataIt += width * height * 4;
                width /= 2;
                height /= 2;
            }
        }
    }
public:
    ~GLTextureS() {glDeleteTextures(1, &m_tex);}

    void bind(size_t idx) const
    {
        glActiveTexture(GL_TEXTURE0 + idx);
        glBindTexture(GL_TEXTURE_2D, m_tex);
    }
};

class GLTextureD : public ITextureD
{
    friend class GLDataFactory;
    friend struct GLCommandQueue;
    struct GLCommandQueue* m_q;
    GLuint m_texs[3];
    void* m_mappedBuf = nullptr;
    size_t m_mappedSize = 0;
    size_t m_width = 0;
    size_t m_height = 0;
    GLTextureD(GLCommandQueue* q, size_t width, size_t height, TextureFormat fmt);
public:
    ~GLTextureD();

    void load(const void* data, size_t sz);
    void* map(size_t sz);
    void unmap();

    void bind(size_t idx) const;
};

class GLTextureR : public ITextureR
{
    friend class GLDataFactory;
    friend struct GLCommandQueue;
    struct GLCommandQueue* m_q;
    GLuint m_texs[2];
    GLuint m_fbo = 0;
    size_t m_width = 0;
    size_t m_height = 0;
    size_t m_samples = 0;
    GLTextureR(GLCommandQueue* q, size_t width, size_t height, size_t samples);
public:
    ~GLTextureR();

    void bind(size_t idx) const
    {
        glActiveTexture(GL_TEXTURE0 + idx);
        glBindTexture(GL_TEXTURE_2D, m_texs[0]);
    }
    
    void resize(size_t width, size_t height)
    {
        m_width = width;
        m_height = height;
        glBindTexture(GL_TEXTURE_2D, m_texs[0]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, m_texs[1]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, width, height, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, nullptr);
    }
};

ITextureS*
GLDataFactory::newStaticTexture(size_t width, size_t height, size_t mips, TextureFormat fmt,
                                   const void* data, size_t sz)
{
    GLTextureS* retval = new GLTextureS(width, height, mips, fmt, data, sz);
    static_cast<GLData*>(m_deferredData)->m_STexs.emplace_back(retval);
    return retval;
}

ITextureS*
GLDataFactory::newStaticTexture(size_t width, size_t height, size_t mips, TextureFormat fmt,
                                std::unique_ptr<uint8_t[]>&& data, size_t sz)
{
    std::unique_ptr<uint8_t[]> d = std::move(data);
    GLTextureS* retval = new GLTextureS(width, height, mips, fmt, d.get(), sz);
    static_cast<GLData*>(m_deferredData)->m_STexs.emplace_back(retval);
    return retval;
}

class GLShaderPipeline : public IShaderPipeline
{
    friend class GLDataFactory;
    friend struct GLShaderDataBinding;
    GLuint m_vert = 0;
    GLuint m_frag = 0;
    GLuint m_prog = 0;
    GLenum m_sfactor = GL_ONE;
    GLenum m_dfactor = GL_ZERO;
    bool m_depthTest = true;
    bool m_depthWrite = true;
    bool m_backfaceCulling = true;
    std::vector<GLint> m_uniLocs;
    bool initObjects()
    {
        m_vert = glCreateShader(GL_VERTEX_SHADER);
        m_frag = glCreateShader(GL_FRAGMENT_SHADER);
        m_prog = glCreateProgram();
        if (!m_vert || !m_frag || !m_prog)
        {
            glDeleteShader(m_vert);
            m_vert = 0;
            glDeleteShader(m_frag);
            m_frag = 0;
            glDeleteProgram(m_prog);
            m_prog = 0;
            return false;
        }
        glAttachShader(m_prog, m_vert);
        glAttachShader(m_prog, m_frag);
        return true;
    }
    void clearObjects()
    {
        if (m_vert)
            glDeleteShader(m_vert);
        if (m_frag)
            glDeleteShader(m_frag);
        if (m_prog)
            glDeleteProgram(m_prog);
    }
    GLShaderPipeline() = default;
public:
    operator bool() const {return m_prog != 0;}
    ~GLShaderPipeline() {clearObjects();}
    GLShaderPipeline& operator=(const GLShaderPipeline&) = delete;
    GLShaderPipeline(const GLShaderPipeline&) = delete;
    GLShaderPipeline& operator=(GLShaderPipeline&& other)
    {
        m_vert = other.m_vert;
        other.m_vert = 0;
        m_frag = other.m_frag;
        other.m_frag = 0;
        m_prog = other.m_prog;
        other.m_prog = 0;
        m_sfactor = other.m_sfactor;
        m_dfactor = other.m_dfactor;
        m_depthTest = other.m_depthTest;
        m_depthWrite = other.m_depthWrite;
        m_backfaceCulling = other.m_backfaceCulling;
        m_uniLocs = std::move(other.m_uniLocs);
        return *this;
    }
    GLShaderPipeline(GLShaderPipeline&& other) {*this = std::move(other);}

    GLuint bind() const
    {
        glUseProgram(m_prog);

        if (m_dfactor != GL_ZERO)
        {
            glEnable(GL_BLEND);
            glBlendFunc(m_sfactor, m_dfactor);
        }
        else
            glDisable(GL_BLEND);

        if (m_depthTest)
            glEnable(GL_DEPTH_TEST);
        else
            glDisable(GL_DEPTH_TEST);
        glDepthMask(m_depthWrite);

        if (m_backfaceCulling)
            glEnable(GL_CULL_FACE);
        else
            glDisable(GL_CULL_FACE);

        return m_prog;
    }
};

static const GLenum BLEND_FACTOR_TABLE[] =
{
    GL_ZERO,
    GL_ONE,
    GL_SRC_COLOR,
    GL_ONE_MINUS_SRC_COLOR,
    GL_DST_COLOR,
    GL_ONE_MINUS_DST_COLOR,
    GL_SRC_ALPHA,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_DST_ALPHA,
    GL_ONE_MINUS_DST_ALPHA
};

IShaderPipeline* GLDataFactory::newShaderPipeline
(const char* vertSource, const char* fragSource,
 size_t texCount, const char* texArrayName,
 size_t uniformBlockCount, const char** uniformBlockNames,
 BlendFactor srcFac, BlendFactor dstFac,
 bool depthTest, bool depthWrite, bool backfaceCulling)
{
    GLShaderPipeline shader;
    if (!shader.initObjects())
    {
        Log.report(LogVisor::Error, "unable to create shader objects\n");
        return nullptr;
    }
    shader.m_sfactor = BLEND_FACTOR_TABLE[int(srcFac)];
    shader.m_dfactor = BLEND_FACTOR_TABLE[int(dstFac)];
    shader.m_depthTest = depthTest;
    shader.m_depthWrite = depthWrite;
    shader.m_backfaceCulling = backfaceCulling;

    glShaderSource(shader.m_vert, 1, &vertSource, nullptr);
    glCompileShader(shader.m_vert);
    GLint status;
    glGetShaderiv(shader.m_vert, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE)
    {
        GLint logLen;
        glGetShaderiv(shader.m_vert, GL_INFO_LOG_LENGTH, &logLen);
        char* log = (char*)malloc(logLen);
        glGetShaderInfoLog(shader.m_vert, logLen, nullptr, log);
        Log.report(LogVisor::Error, "unable to compile vert source\n%s\n%s\n", log, vertSource);
        free(log);
        return nullptr;
    }

    glShaderSource(shader.m_frag, 1, &fragSource, nullptr);
    glCompileShader(shader.m_frag);
    glGetShaderiv(shader.m_frag, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE)
    {
        GLint logLen;
        glGetShaderiv(shader.m_frag, GL_INFO_LOG_LENGTH, &logLen);
        char* log = (char*)malloc(logLen);
        glGetShaderInfoLog(shader.m_frag, logLen, nullptr, log);
        Log.report(LogVisor::Error, "unable to compile frag source\n%s\n%s\n", log, fragSource);
        free(log);
        return nullptr;
    }

    glLinkProgram(shader.m_prog);
    glGetProgramiv(shader.m_prog, GL_LINK_STATUS, &status);
    if (status != GL_TRUE)
    {
        GLint logLen;
        glGetProgramiv(shader.m_prog, GL_INFO_LOG_LENGTH, &logLen);
        char* log = (char*)malloc(logLen);
        glGetProgramInfoLog(shader.m_prog, logLen, nullptr, log);
        Log.report(LogVisor::Error, "unable to link shader program\n%s\n", log);
        free(log);
        return nullptr;
    }

    glUseProgram(shader.m_prog);

    if (uniformBlockCount)
    {
        shader.m_uniLocs.reserve(uniformBlockCount);
        for (size_t i=0 ; i<uniformBlockCount ; ++i)
        {
            GLint uniLoc = glGetUniformBlockIndex(shader.m_prog, uniformBlockNames[i]);
            if (uniLoc < 0)
                Log.report(LogVisor::FatalError, "unable to find uniform block '%s'", uniformBlockNames[i]);
            shader.m_uniLocs.push_back(uniLoc);
        }
    }

    if (texCount && texArrayName)
    {
        GLint texLoc = glGetUniformLocation(shader.m_prog, texArrayName);
        if (texLoc < 0)
            Log.report(LogVisor::FatalError, "unable to find sampler variable '%s'", texArrayName);
        if (texCount > m_texUnis.size())
            for (size_t i=m_texUnis.size() ; i<texCount ; ++i)
                m_texUnis.push_back(i);
        glUniform1iv(texLoc, m_texUnis.size(), m_texUnis.data());
    }

    GLShaderPipeline* retval = new GLShaderPipeline(std::move(shader));
    static_cast<GLData*>(m_deferredData)->m_SPs.emplace_back(retval);
    return retval;
}

struct GLVertexFormat : IVertexFormat
{
    GLCommandQueue* m_q;
    GLuint m_vao = 0;
    size_t m_elementCount;
    std::unique_ptr<VertexElementDescriptor[]> m_elements;
    GLVertexFormat(GLCommandQueue* q, size_t elementCount,
                      const VertexElementDescriptor* elements);
    ~GLVertexFormat();
    void bind() const {glBindVertexArray(m_vao);}
};

struct GLShaderDataBinding : IShaderDataBinding
{
    const GLShaderPipeline* m_pipeline;
    const GLVertexFormat* m_vtxFormat;
    size_t m_ubufCount;
    std::unique_ptr<IGraphicsBuffer*[]> m_ubufs;
    size_t m_texCount;
    std::unique_ptr<ITexture*[]> m_texs;
    GLShaderDataBinding(IShaderPipeline* pipeline,
                        IVertexFormat* vtxFormat,
                        size_t ubufCount, IGraphicsBuffer** ubufs,
                        size_t texCount, ITexture** texs)
    : m_pipeline(static_cast<GLShaderPipeline*>(pipeline)),
      m_vtxFormat(static_cast<GLVertexFormat*>(vtxFormat)),
      m_ubufCount(ubufCount),
      m_ubufs(new IGraphicsBuffer*[ubufCount]),
      m_texCount(texCount),
      m_texs(new ITexture*[texCount])
    {
        for (size_t i=0 ; i<ubufCount ; ++i)
            m_ubufs[i] = ubufs[i];
        for (size_t i=0 ; i<texCount ; ++i)
            m_texs[i] = texs[i];
    }
    void bind() const
    {
        GLuint prog = m_pipeline->bind();
        m_vtxFormat->bind();
        for (size_t i=0 ; i<m_ubufCount ; ++i)
        {
            IGraphicsBuffer* ubuf = m_ubufs[i];
            if (ubuf->dynamic())
                static_cast<GLGraphicsBufferD*>(ubuf)->bindUniform(i);
            else
                static_cast<GLGraphicsBufferS*>(ubuf)->bindUniform(i);
            glUniformBlockBinding(prog, m_pipeline->m_uniLocs.at(i), i);
        }
        for (size_t i=0 ; i<m_texCount ; ++i)
        {
            ITexture* tex = m_texs[i];
            if (tex->type() == TextureType::Dynamic)
                static_cast<GLTextureD*>(tex)->bind(i);
            else if (tex->type() == TextureType::Static)
                static_cast<GLTextureS*>(tex)->bind(i);
        }
    }
};

IShaderDataBinding*
GLDataFactory::newShaderDataBinding(IShaderPipeline* pipeline,
                                       IVertexFormat* vtxFormat,
                                       IGraphicsBuffer*, IGraphicsBuffer*,
                                       size_t ubufCount, IGraphicsBuffer** ubufs,
                                       size_t texCount, ITexture** texs)
{
    GLShaderDataBinding* retval =
    new GLShaderDataBinding(pipeline, vtxFormat, ubufCount, ubufs, texCount, texs);
    static_cast<GLData*>(m_deferredData)->m_SBinds.emplace_back(retval);
    return retval;
}

GLDataFactory::GLDataFactory(IGraphicsContext* parent)
: m_parent(parent), m_deferredData(new struct GLData()) {}

void GLDataFactory::reset()
{
    delete static_cast<GLData*>(m_deferredData);
    m_deferredData = new struct GLData();
}

IGraphicsData* GLDataFactory::commit()
{
    IGraphicsData* retval = m_deferredData;
    m_deferredData = new struct GLData();
    m_committedData.insert(retval);
    /* Let's go ahead and flush to ensure our data gets to the GPU
       While this isn't strictly required, some drivers might behave
       differently */
    glFlush();
    return retval;
}
    
void GLDataFactory::destroyData(IGraphicsData* d)
{
    GLData* data = static_cast<GLData*>(d);
    m_committedData.erase(data);
    delete data;
}

void GLDataFactory::destroyAllData()
{
    for (IGraphicsData* data : m_committedData)
        delete static_cast<GLData*>(data);
    m_committedData.clear();
}

static const GLint SEMANTIC_COUNT_TABLE[] =
{
    3,
    3,
    4,
    2,
    4
};

static const size_t SEMANTIC_SIZE_TABLE[] =
{
    12,
    12,
    4,
    8,
    16
};

static const GLenum SEMANTIC_TYPE_TABLE[] =
{
    GL_FLOAT,
    GL_FLOAT,
    GL_UNSIGNED_BYTE,
    GL_FLOAT,
    GL_FLOAT
};

struct GLCommandQueue : IGraphicsCommandQueue
{
    Platform platform() const {return IGraphicsDataFactory::Platform::OGL;}
    const SystemChar* platformName() const {return _S("OGL");}
    IGraphicsContext* m_parent = nullptr;

    struct Command
    {
        enum class Op
        {
            SetShaderDataBinding,
            SetRenderTarget,
            SetViewport,
            SetClearColor,
            ClearTarget,
            SetDrawPrimitive,
            Draw,
            DrawIndexed,
            DrawInstances,
            DrawInstancesIndexed,
            Present
        } m_op;
        union
        {
            const IShaderDataBinding* binding;
            const ITextureR* target;
            const ITextureR* source;
            SWindowRect rect;
            float rgba[4];
            GLbitfield flags;
            GLenum prim;
            struct
            {
                size_t start;
                size_t count;
                size_t instCount;
            };
        };
        Command(Op op) : m_op(op) {}
    };
    std::vector<Command> m_cmdBufs[3];
    size_t m_fillBuf = 0;
    size_t m_completeBuf = 0;
    size_t m_drawBuf = 0;
    bool m_running = true;

    std::mutex m_mt;
    std::condition_variable m_cv;
    std::mutex m_initmt;
    std::condition_variable m_initcv;
    std::unique_lock<std::mutex> m_initlk;
    std::thread m_thr;

    struct RenderTextureResize
    {
        GLTextureR* tex;
        size_t width;
        size_t height;
    };

    /* These members are locked for multithreaded access */
    std::vector<RenderTextureResize> m_pendingResizes;
    std::vector<GLVertexFormat*> m_pendingFmtAdds;
    std::vector<GLuint> m_pendingFmtDels;
    std::vector<GLTextureR*> m_pendingFboAdds;
    std::vector<GLuint> m_pendingFboDels;

    static void ConfigureVertexFormat(GLVertexFormat* fmt)
    {
        glGenVertexArrays(1, &fmt->m_vao);

        size_t stride = 0;
        for (size_t i=0 ; i<fmt->m_elementCount ; ++i)
        {
            const VertexElementDescriptor* desc = &fmt->m_elements[i];
            stride += SEMANTIC_SIZE_TABLE[int(desc->semantic)];
        }

        size_t offset = 0;
        glBindVertexArray(fmt->m_vao);
        const IGraphicsBuffer* lastVBO = nullptr;
        const IGraphicsBuffer* lastEBO = nullptr;
        for (size_t i=0 ; i<fmt->m_elementCount ; ++i)
        {
            const VertexElementDescriptor* desc = &fmt->m_elements[i];
            if (desc->vertBuffer != lastVBO)
            {
                lastVBO = desc->vertBuffer;
                if (lastVBO->dynamic())
                    static_cast<const GLGraphicsBufferD*>(lastVBO)->bindVertex();
                else
                    static_cast<const GLGraphicsBufferS*>(lastVBO)->bindVertex();
            }
            if (desc->indexBuffer != lastEBO)
            {
                lastEBO = desc->indexBuffer;
                if (lastEBO->dynamic())
                    static_cast<const GLGraphicsBufferD*>(lastEBO)->bindIndex();
                else
                    static_cast<const GLGraphicsBufferS*>(lastEBO)->bindIndex();
            }
            glEnableVertexAttribArray(i);
            glVertexAttribPointer(i, SEMANTIC_COUNT_TABLE[int(desc->semantic)],
                    SEMANTIC_TYPE_TABLE[int(desc->semantic)], GL_TRUE, stride, (void*)offset);
            offset += SEMANTIC_SIZE_TABLE[int(desc->semantic)];
        }
    }

    static void ConfigureFBO(GLTextureR* tex)
    {
        glGenFramebuffers(1, &tex->m_fbo);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, tex->m_fbo);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex->m_texs[0], 0);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, tex->m_texs[1], 0);
    }

    static void RenderingWorker(GLCommandQueue* self)
    {
        {
            std::unique_lock<std::mutex> lk(self->m_initmt);
            self->m_parent->makeCurrent();
            if (glewInit() != GLEW_OK)
                Log.report(LogVisor::FatalError, "unable to init glew");
            self->m_parent->postInit();
        }
        self->m_initcv.notify_one();
        while (self->m_running)
        {
            {
                std::unique_lock<std::mutex> lk(self->m_mt);
                self->m_cv.wait(lk);
                if (!self->m_running)
                    break;
                self->m_drawBuf = self->m_completeBuf;

                if (self->m_pendingResizes.size())
                {
                    for (const RenderTextureResize& resize : self->m_pendingResizes)
                        resize.tex->resize(resize.width, resize.height);
                    self->m_pendingResizes.clear();
                }

                if (self->m_pendingFmtAdds.size())
                {
                    for (GLVertexFormat* fmt : self->m_pendingFmtAdds)
                        ConfigureVertexFormat(fmt);
                    self->m_pendingFmtAdds.clear();
                }

                if (self->m_pendingFmtDels.size())
                {
                    for (GLuint fmt : self->m_pendingFmtDels)
                        glDeleteVertexArrays(1, &fmt);
                    self->m_pendingFmtDels.clear();
                }

                if (self->m_pendingFboAdds.size())
                {
                    for (GLTextureR* tex : self->m_pendingFboAdds)
                        ConfigureFBO(tex);
                    self->m_pendingFboAdds.clear();
                }

                if (self->m_pendingFboDels.size())
                {
                    for (GLuint fbo : self->m_pendingFboDels)
                        glDeleteFramebuffers(1, &fbo);
                    self->m_pendingFboDels.clear();
                }
            }
            std::vector<Command>& cmds = self->m_cmdBufs[self->m_drawBuf];
            GLenum prim = GL_TRIANGLES;
            for (const Command& cmd : cmds)
            {
                switch (cmd.m_op)
                {
                case Command::Op::SetShaderDataBinding:
                    static_cast<const GLShaderDataBinding*>(cmd.binding)->bind();
                    break;
                case Command::Op::SetRenderTarget:
                {
                    const GLTextureR* tex = static_cast<const GLTextureR*>(cmd.target);
                    if (!tex)
                        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
                    else
                        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, tex->m_fbo);
                    break;
                }
                case Command::Op::SetViewport:
                    glViewport(cmd.rect.location[0], cmd.rect.location[1],
                               cmd.rect.size[0], cmd.rect.size[1]);
                    break;
                case Command::Op::SetClearColor:
                    glClearColor(cmd.rgba[0], cmd.rgba[1], cmd.rgba[2], cmd.rgba[3]);
                    break;
                case Command::Op::ClearTarget:
                    glClear(cmd.flags);
                    break;
                case Command::Op::SetDrawPrimitive:
                    prim = cmd.prim;
                    break;
                case Command::Op::Draw:
                    glDrawArrays(prim, cmd.start, cmd.count);
                    break;
                case Command::Op::DrawIndexed:
                    glDrawElements(prim, cmd.count, GL_UNSIGNED_INT, (void*)cmd.start);
                    break;
                case Command::Op::DrawInstances:
                    glDrawArraysInstanced(prim, cmd.start, cmd.count, cmd.instCount);
                    break;
                case Command::Op::DrawInstancesIndexed:
                    glDrawElementsInstanced(prim, cmd.count, GL_UNSIGNED_INT, (void*)cmd.start, cmd.instCount);
                    break;
                case Command::Op::Present:
                {
                    const GLTextureR* tex = static_cast<const GLTextureR*>(cmd.source);
                    if (tex)
                    {
                        glBindFramebuffer(GL_READ_FRAMEBUFFER, tex->m_fbo);
                        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
                        glBlitFramebuffer(0, 0, tex->m_width, tex->m_height, 0, 0,
                                          tex->m_width, tex->m_height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
                    }
                    self->m_parent->present();
                    break;
                }
                default: break;
                }
            }
            cmds.clear();
        }
    }

    GLCommandQueue(IGraphicsContext* parent)
    : m_parent(parent),
      m_initlk(m_initmt),
      m_thr(RenderingWorker, this)
    {
        m_initcv.wait(m_initlk);
        m_initlk.unlock();
    }

    ~GLCommandQueue()
    {
        m_running = false;
        m_cv.notify_one();
        m_thr.join();
    }

    void setShaderDataBinding(IShaderDataBinding* binding)
    {
        std::vector<Command>& cmds = m_cmdBufs[m_fillBuf];
        cmds.emplace_back(Command::Op::SetShaderDataBinding);
        cmds.back().binding = binding;
    }

    void setRenderTarget(ITextureR* target)
    {
        std::vector<Command>& cmds = m_cmdBufs[m_fillBuf];
        cmds.emplace_back(Command::Op::SetRenderTarget);
        cmds.back().target = target;
    }

    void setViewport(const SWindowRect& rect)
    {
        std::vector<Command>& cmds = m_cmdBufs[m_fillBuf];
        cmds.emplace_back(Command::Op::SetViewport);
        cmds.back().rect = rect;
    }
    
    void resizeRenderTexture(ITextureR* tex, size_t width, size_t height)
    {
        std::unique_lock<std::mutex> lk(m_mt);
        GLTextureR* texgl = static_cast<GLTextureR*>(tex);
        m_pendingResizes.push_back({texgl, width, height});
    }

    void flushBufferUpdates()
    {
        glFlush();
    }

    void setClearColor(const float rgba[4])
    {
        std::vector<Command>& cmds = m_cmdBufs[m_fillBuf];
        cmds.emplace_back(Command::Op::SetClearColor);
        cmds.back().rgba[0] = rgba[0];
        cmds.back().rgba[1] = rgba[1];
        cmds.back().rgba[2] = rgba[2];
        cmds.back().rgba[3] = rgba[3];
    }

    void clearTarget(bool render=true, bool depth=true)
    {
        std::vector<Command>& cmds = m_cmdBufs[m_fillBuf];
        cmds.emplace_back(Command::Op::ClearTarget);
        cmds.back().flags = 0;
        if (render)
            cmds.back().flags |= GL_COLOR_BUFFER_BIT;
        if (depth)
            cmds.back().flags |= GL_DEPTH_BUFFER_BIT;
    }

    void setDrawPrimitive(Primitive prim)
    {
        std::vector<Command>& cmds = m_cmdBufs[m_fillBuf];
        cmds.emplace_back(Command::Op::SetDrawPrimitive);
        if (prim == Primitive::Triangles)
            cmds.back().prim = GL_TRIANGLES;
        else if (prim == Primitive::TriStrips)
            cmds.back().prim = GL_TRIANGLE_STRIP;
    }

    void draw(size_t start, size_t count)
    {
        std::vector<Command>& cmds = m_cmdBufs[m_fillBuf];
        cmds.emplace_back(Command::Op::Draw);
        cmds.back().start = start;
        cmds.back().count = count;
    }

    void drawIndexed(size_t start, size_t count)
    {
        std::vector<Command>& cmds = m_cmdBufs[m_fillBuf];
        cmds.emplace_back(Command::Op::DrawIndexed);
        cmds.back().start = start;
        cmds.back().count = count;
    }

    void drawInstances(size_t start, size_t count, size_t instCount)
    {
        std::vector<Command>& cmds = m_cmdBufs[m_fillBuf];
        cmds.emplace_back(Command::Op::DrawInstances);
        cmds.back().start = start;
        cmds.back().count = count;
        cmds.back().instCount = instCount;
    }

    void drawInstancesIndexed(size_t start, size_t count, size_t instCount)
    {
        std::vector<Command>& cmds = m_cmdBufs[m_fillBuf];
        cmds.emplace_back(Command::Op::DrawInstancesIndexed);
        cmds.back().start = start;
        cmds.back().count = count;
        cmds.back().instCount = instCount;
    }

    void resolveDisplay(ITextureR* source)
    {
        std::vector<Command>& cmds = m_cmdBufs[m_fillBuf];
        cmds.emplace_back(Command::Op::Present);
        cmds.back().source = source;
    }
    
    void addVertexFormat(GLVertexFormat* fmt)
    {
        std::unique_lock<std::mutex> lk(m_mt);
        m_pendingFmtAdds.push_back(fmt);
    }

    void delVertexFormat(GLVertexFormat* fmt)
    {
        std::unique_lock<std::mutex> lk(m_mt);
        m_pendingFmtDels.push_back(fmt->m_vao);
    }

    void addFBO(GLTextureR* tex)
    {
        std::unique_lock<std::mutex> lk(m_mt);
        m_pendingFboAdds.push_back(tex);
    }

    void delFBO(GLTextureR* tex)
    {
        std::unique_lock<std::mutex> lk(m_mt);
        m_pendingFboDels.push_back(tex->m_fbo);
    }

    void execute()
    {
        std::unique_lock<std::mutex> lk(m_mt);
        m_completeBuf = m_fillBuf;
        for (size_t i=0 ; i<3 ; ++i)
        {
            if (i == m_completeBuf || i == m_drawBuf)
                continue;
            m_fillBuf = i;
            break;
        }
        lk.unlock();
        m_cv.notify_one();
        m_cmdBufs[m_fillBuf].clear();
    }
};

void GLGraphicsBufferD::load(const void* data, size_t sz)
{
    glBindBuffer(m_target, m_bufs[m_q->m_fillBuf]);
    glBufferData(m_target, sz, data, GL_DYNAMIC_DRAW);
}
void* GLGraphicsBufferD::map(size_t sz)
{
    if (m_mappedBuf)
        free(m_mappedBuf);
    m_mappedBuf = malloc(sz);
    m_mappedSize = sz;
    return m_mappedBuf;
}
void GLGraphicsBufferD::unmap()
{
    glBindBuffer(m_target, m_bufs[m_q->m_fillBuf]);
    glBufferData(m_target, m_mappedSize, m_mappedBuf, GL_DYNAMIC_DRAW);
    free(m_mappedBuf);
    m_mappedBuf = nullptr;
}
void GLGraphicsBufferD::bindVertex() const
{glBindBuffer(GL_ARRAY_BUFFER, m_bufs[m_q->m_drawBuf]);}
void GLGraphicsBufferD::bindIndex() const
{glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_bufs[m_q->m_drawBuf]);}
void GLGraphicsBufferD::bindUniform(size_t idx) const
{glBindBufferBase(GL_UNIFORM_BUFFER, idx, m_bufs[m_q->m_drawBuf]);}

IGraphicsBufferD*
GLDataFactory::newDynamicBuffer(BufferUse use, size_t stride, size_t count)
{
    GLCommandQueue* q = static_cast<GLCommandQueue*>(m_parent->getCommandQueue());
    GLGraphicsBufferD* retval = new GLGraphicsBufferD(q, use);
    static_cast<GLData*>(m_deferredData)->m_DBufs.emplace_back(retval);
    return retval;
}

GLTextureD::GLTextureD(GLCommandQueue* q, size_t width, size_t height, TextureFormat fmt)
: m_q(q), m_width(width), m_height(height)
{
    glGenTextures(3, m_texs);
    glBindTexture(GL_TEXTURE_2D, m_texs[0]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, m_texs[1]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, m_texs[2]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
}
GLTextureD::~GLTextureD() {glDeleteTextures(3, m_texs);}

void GLTextureD::load(const void* data, size_t sz)
{
    glBindTexture(GL_TEXTURE_2D, m_texs[m_q->m_fillBuf]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
}
void* GLTextureD::map(size_t sz)
{
    if (m_mappedBuf)
        free(m_mappedBuf);
    m_mappedBuf = malloc(sz);
    m_mappedSize = sz;
    return m_mappedBuf;
}
void GLTextureD::unmap()
{
    glBindTexture(GL_TEXTURE_2D, m_texs[m_q->m_fillBuf]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, m_mappedBuf);
    free(m_mappedBuf);
    m_mappedBuf = nullptr;
}

void GLTextureD::bind(size_t idx) const
{
    glActiveTexture(GL_TEXTURE0 + idx);
    glBindTexture(GL_TEXTURE_2D, m_texs[0]);
}

ITextureD*
GLDataFactory::newDynamicTexture(size_t width, size_t height, TextureFormat fmt)
{
    GLCommandQueue* q = static_cast<GLCommandQueue*>(m_parent->getCommandQueue());
    GLTextureD* retval = new GLTextureD(q, width, height, fmt);
    static_cast<GLData*>(m_deferredData)->m_DTexs.emplace_back(retval);
    return retval;
}

GLTextureR::GLTextureR(GLCommandQueue* q, size_t width, size_t height, size_t samples)
: m_q(q), m_width(width), m_height(height), m_samples(samples)
{
    glGenTextures(2, m_texs);
    glBindTexture(GL_TEXTURE_2D, m_texs[0]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, m_texs[1]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, width, height, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, nullptr);
    m_q->addFBO(this);
}
GLTextureR::~GLTextureR() {glDeleteTextures(2, m_texs); m_q->delFBO(this);}

ITextureR*
GLDataFactory::newRenderTexture(size_t width, size_t height, size_t samples)
{
    GLCommandQueue* q = static_cast<GLCommandQueue*>(m_parent->getCommandQueue());
    GLTextureR* retval = new GLTextureR(q, width, height, samples);
    q->resizeRenderTexture(retval, width, height);
    static_cast<GLData*>(m_deferredData)->m_RTexs.emplace_back(retval);
    return retval;
}

GLVertexFormat::GLVertexFormat(GLCommandQueue* q, size_t elementCount,
                                     const VertexElementDescriptor* elements)
: m_q(q),
  m_elementCount(elementCount),
  m_elements(new VertexElementDescriptor[elementCount])
{
    for (size_t i=0 ; i<elementCount ; ++i)
        m_elements[i] = elements[i];
    m_q->addVertexFormat(this);
}
GLVertexFormat::~GLVertexFormat() {m_q->delVertexFormat(this);}

IVertexFormat* GLDataFactory::newVertexFormat
(size_t elementCount, const VertexElementDescriptor* elements)
{
    GLCommandQueue* q = static_cast<GLCommandQueue*>(m_parent->getCommandQueue());
    GLVertexFormat* retval = new struct GLVertexFormat(q, elementCount, elements);
    static_cast<GLData*>(m_deferredData)->m_VFmts.emplace_back(retval);
    return retval;
}

IGraphicsCommandQueue* _NewGLCommandQueue(IGraphicsContext* parent)
{
    return new struct GLCommandQueue(parent);
}

}
