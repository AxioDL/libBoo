#pragma once
#if BOO_HAS_VULKAN

#include <mutex>
#include <queue>
#include <unordered_map>
#include <vector>

#include "boo/BooObject.hpp"
#include "boo/IGraphicsContext.hpp"
#include "boo/IWindow.hpp"
#include "boo/System.hpp"
#include "boo/graphicsdev/IGraphicsDataFactory.hpp"
#include "boo/graphicsdev/VulkanDispatchTable.hpp"

/* Forward-declare handle type for Vulkan Memory Allocator */
struct VmaAllocator_T;

namespace boo {
struct BaseGraphicsData;

struct VulkanContext {
  struct LayerProperties {
    VkLayerProperties properties;
    std::vector<VkExtensionProperties> extensions;
  };

#ifndef NDEBUG
  PFN_vkDestroyDebugReportCallbackEXT m_destroyDebugReportCallback = nullptr;
  VkDebugReportCallbackEXT m_debugReportCallback = VK_NULL_HANDLE;
#endif
  std::vector<LayerProperties> m_instanceLayerProperties;
  std::vector<const char*> m_layerNames;
  std::vector<const char*> m_instanceExtensionNames;
  VkInstance m_instance = VK_NULL_HANDLE;
  std::vector<const char*> m_deviceExtensionNames;
  std::vector<VkPhysicalDevice> m_gpus;
  VkPhysicalDeviceFeatures m_features;
  VkPhysicalDeviceProperties m_gpuProps;
  VkPhysicalDeviceMemoryProperties m_memoryProperties;
  VkDevice m_dev = VK_NULL_HANDLE;
  VmaAllocator_T* m_allocator = VK_NULL_HANDLE;
  uint32_t m_queueCount;
  uint32_t m_graphicsQueueFamilyIndex = UINT32_MAX;
  std::vector<VkQueueFamilyProperties> m_queueProps;
  VkQueue m_queue = VK_NULL_HANDLE;
  std::mutex m_queueLock;
  VkDescriptorSetLayout m_descSetLayout = VK_NULL_HANDLE;
  VkPipelineLayout m_pipelinelayout = VK_NULL_HANDLE;
  VkRenderPass m_pass = VK_NULL_HANDLE;
  VkRenderPass m_passOneSample = VK_NULL_HANDLE;
  VkRenderPass m_passColorOnly = VK_NULL_HANDLE;
  VkCommandPool m_loadPool = VK_NULL_HANDLE;
  VkCommandBuffer m_loadCmdBuf = VK_NULL_HANDLE;
  VkFormat m_displayFormat;
  VkFormat m_internalFormat;

  struct Window {
    struct SwapChain {
      VkFormat m_format = VK_FORMAT_UNDEFINED;
      VkSwapchainKHR m_swapChain = VK_NULL_HANDLE;
      struct Buffer {
        VkImage m_image = VK_NULL_HANDLE;
        VkImageView m_colorView = VK_NULL_HANDLE;
        VkFramebuffer m_framebuffer = VK_NULL_HANDLE;
        VkRenderPassBeginInfo m_passBeginInfo = {};
        void setImage(VulkanContext* ctx, VkImage image, uint32_t width, uint32_t height);
        void destroy(VkDevice dev);
      };
      std::vector<Buffer> m_bufs;
      uint32_t m_backBuf = 0;
      void destroy(VkDevice dev) {
        for (Buffer& buf : m_bufs)
          buf.destroy(dev);
        m_bufs.clear();
        if (m_swapChain) {
          vk::DestroySwapchainKHR(dev, m_swapChain, nullptr);
          m_swapChain = VK_NULL_HANDLE;
        }
        m_backBuf = 0;
      }
    } m_swapChains[2];
    uint32_t m_activeSwapChain = 0;

#if _WIN32
    HWND m_hwnd = 0;
    bool m_fs = false;
    LONG_PTR m_fsStyle;
    LONG_PTR m_fsExStyle;
    RECT m_fsRect;
    int m_fsCountDown = 0;
#endif
  };
  std::unordered_map<const boo::IWindow*, std::unique_ptr<Window>> m_windows;

  VkSampleCountFlags m_sampleCountColor = VK_SAMPLE_COUNT_1_BIT;
  VkSampleCountFlags m_sampleCountDepth = VK_SAMPLE_COUNT_1_BIT;
  float m_anisotropy = 1.f;
  bool m_deepColor = false;

  std::unordered_map<uint32_t, VkSampler> m_samplers;

  bool initVulkan(std::string_view appName, PFN_vkGetInstanceProcAddr getVkProc);
  bool enumerateDevices();
  void initDevice();
  void destroyDevice();
  void initSwapChain(Window& windowCtx, VkSurfaceKHR surface, VkFormat format, VkColorSpaceKHR colorspace);

  struct SwapChainResize {
    Window& m_windowCtx;
    VkSurfaceKHR m_surface;
    VkFormat m_format;
    VkColorSpaceKHR m_colorspace;
    SWindowRect m_rect;
    SwapChainResize(Window& windowCtx, VkSurfaceKHR surface, VkFormat format, VkColorSpaceKHR colorspace,
                    const SWindowRect& rect)
    : m_windowCtx(windowCtx), m_surface(surface), m_format(format), m_colorspace(colorspace), m_rect(rect) {}
  };
  std::queue<SwapChainResize> m_deferredResizes;
  std::mutex m_resizeLock;
  void resizeSwapChain(Window& windowCtx, VkSurfaceKHR surface, VkFormat format, VkColorSpaceKHR colorspace,
                       const SWindowRect& rect);
  bool _resizeSwapChains();
};
extern VulkanContext g_VulkanContext;

class VulkanDataFactory : public IGraphicsDataFactory {
public:
  class Context final : public IGraphicsDataFactory::Context {
    friend class VulkanDataFactoryImpl;
    VulkanDataFactory& m_parent;
    boo::ObjToken<BaseGraphicsData> m_data;

  public:
    Context(VulkanDataFactory& parent __BooTraceArgs);
    ~Context();
    Platform platform() const { return Platform::Vulkan; }
    const SystemChar* platformName() const { return _SYS_STR("Vulkan"); }

    boo::ObjToken<IGraphicsBufferS> newStaticBuffer(BufferUse use, const void* data, size_t stride, size_t count);
    boo::ObjToken<IGraphicsBufferD> newDynamicBuffer(BufferUse use, size_t stride, size_t count);

    boo::ObjToken<ITextureS> newStaticTexture(size_t width, size_t height, size_t mips, TextureFormat fmt,
                                              TextureClampMode clampMode, const void* data, size_t sz);
    boo::ObjToken<ITextureSA> newStaticArrayTexture(size_t width, size_t height, size_t layers, size_t mips,
                                                    TextureFormat fmt, TextureClampMode clampMode, const void* data,
                                                    size_t sz);
    boo::ObjToken<ITextureD> newDynamicTexture(size_t width, size_t height, TextureFormat fmt,
                                               TextureClampMode clampMode);
    boo::ObjToken<ITextureR> newRenderTexture(size_t width, size_t height, TextureClampMode clampMode,
                                              size_t colorBindCount, size_t depthBindCount);
    ObjToken<ITextureCubeR> newCubeRenderTexture(size_t width, size_t mips);

    ObjToken<IShaderStage> newShaderStage(const uint8_t* data, size_t size, PipelineStage stage);

    ObjToken<IShaderPipeline> newShaderPipeline(ObjToken<IShaderStage> vertex, ObjToken<IShaderStage> fragment,
                                                ObjToken<IShaderStage> geometry, ObjToken<IShaderStage> control,
                                                ObjToken<IShaderStage> evaluation, const VertexFormatInfo& vtxFmt,
                                                const AdditionalPipelineInfo& additionalInfo, bool asynchronous = true);

    boo::ObjToken<IShaderDataBinding> newShaderDataBinding(
        const boo::ObjToken<IShaderPipeline>& pipeline, const boo::ObjToken<IGraphicsBuffer>& vbo,
        const boo::ObjToken<IGraphicsBuffer>& instVbo, const boo::ObjToken<IGraphicsBuffer>& ibo, size_t ubufCount,
        const boo::ObjToken<IGraphicsBuffer>* ubufs, const PipelineStage* ubufStages, const size_t* ubufOffs,
        const size_t* ubufSizes, size_t texCount, const boo::ObjToken<ITexture>* texs, const int* bindIdxs,
        const bool* bindDepth, size_t baseVert = 0, size_t baseInst = 0);
  };

  static std::vector<uint8_t> CompileGLSL(const char* source, PipelineStage stage);
};

} // namespace boo

#endif
