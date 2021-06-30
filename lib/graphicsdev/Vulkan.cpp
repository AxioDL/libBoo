#include "boo/graphicsdev/Vulkan.hpp"

#include <array>
#include <cmath>
#include <vector>
#include <optional>

#include <glslang/Public/ShaderLang.h>
#include <StandAlone/ResourceLimits.h>
#include <SPIRV/GlslangToSpv.h>
#include <SPIRV/disassemble.h>
#include <optick.h>

#include "boo/IGraphicsContext.hpp"
#include "boo/graphicsdev/GLSLMacros.hpp"
#include "boo/graphicsdev/IGraphicsCommandQueue.hpp"
#include "lib/graphicsdev/Common.hpp"

#define AMD_PAL_HACK 1

#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#include "vk_mem_alloc.h"

#include <logvisor/logvisor.hpp>

#define BOO_VK_MAX_DESCRIPTOR_SETS 65536

#undef min
#undef max
#undef None

static const char* GammaVS = "#version 330\n" BOO_GLSL_BINDING_HEAD
                             "layout(location=0) in vec4 posIn;\n"
                             "layout(location=1) in vec4 uvIn;\n"
                             "\n"
                             "struct VertToFrag\n"
                             "{\n"
                             "    vec2 uv;\n"
                             "};\n"
                             "\n"
                             "SBINDING(0) out VertToFrag vtf;\n"
                             "void main()\n"
                             "{\n"
                             "    vtf.uv = uvIn.xy;\n"
                             "    gl_Position = posIn;\n"
                             "}\n";

static const char* GammaFS = "#version 330\n" BOO_GLSL_BINDING_HEAD
                             "struct VertToFrag\n"
                             "{\n"
                             "    vec2 uv;\n"
                             "};\n"
                             "\n"
                             "SBINDING(0) in VertToFrag vtf;\n"
                             "layout(location=0) out vec4 colorOut;\n"
                             "TBINDING0 uniform sampler2D screenTex;\n"
                             "TBINDING1 uniform sampler2D gammaLUT;\n"
                             "void main()\n"
                             "{\n"
                             "    ivec4 tex = ivec4(texture(screenTex, vtf.uv) * 65535.0);\n"
                             "    for (int i=0 ; i<3 ; ++i)\n"
                             "        colorOut[i] = texelFetch(gammaLUT, ivec2(tex[i] % 256, tex[i] / 256), 0).r;\n"
                             "}\n";

namespace boo {
static logvisor::Module Log("boo::Vulkan");
VulkanContext g_VulkanContext;
class VulkanDataFactoryImpl;
struct VulkanCommandQueue;
struct VulkanDescriptorPool;

class VulkanDataFactoryImpl final : public VulkanDataFactory, public GraphicsDataFactoryHead {
  friend struct VulkanCommandQueue;
  friend class VulkanDataFactory::Context;
  friend struct VulkanData;
  friend struct VulkanPool;
  friend struct VulkanDescriptorPool;
  friend struct VulkanShaderDataBinding;

  IGraphicsContext* m_parent;
  VulkanContext* m_ctx;
  VulkanDescriptorPool* m_descPoolHead = nullptr;

  PipelineCompileQueue<class VulkanShaderPipeline> m_pipelineQueue;
  std::optional<Context> m_lazyContext;
  void flushContext(Context&);

  float m_gamma = 1.f;
  ObjToken<IShaderPipeline> m_gammaShader;
  ObjToken<ITextureD> m_gammaLUT;
  ObjToken<IGraphicsBufferS> m_gammaVBO;
  ObjToken<IShaderDataBinding> m_gammaBinding;
  void SetupGammaResources() {
    commitTransaction([this](IGraphicsDataFactory::Context& ctx) {
      auto vertexSiprv = VulkanDataFactory::CompileGLSL(GammaVS, PipelineStage::Vertex);
      auto vertexShader = ctx.newShaderStage(vertexSiprv, PipelineStage::Vertex);
      auto fragmentSiprv = VulkanDataFactory::CompileGLSL(GammaFS, PipelineStage::Fragment);
      auto fragmentShader = ctx.newShaderStage(fragmentSiprv, PipelineStage::Fragment);
      const VertexElementDescriptor vfmt[] = {{VertexSemantic::Position4}, {VertexSemantic::UV4}};
      AdditionalPipelineInfo info = {
          BlendFactor::One, BlendFactor::Zero, Primitive::TriStrips, ZTest::None, false, true, false, CullMode::None};
      m_gammaShader = ctx.newShaderPipeline(vertexShader, fragmentShader, vfmt, info, false);
      m_gammaLUT = ctx.newDynamicTexture(256, 256, TextureFormat::I16, TextureClampMode::ClampToEdge);
      setDisplayGamma(1.f);
      const struct Vert {
        float pos[4];
        float uv[4];
      } verts[4] = {{{-1.f, -1.f, 0.f, 1.f}, {0.f, 0.f, 0.f, 0.f}},
                    {{1.f, -1.f, 0.f, 1.f}, {1.f, 0.f, 0.f, 0.f}},
                    {{-1.f, 1.f, 0.f, 1.f}, {0.f, 1.f, 0.f, 0.f}},
                    {{1.f, 1.f, 0.f, 1.f}, {1.f, 1.f, 0.f, 0.f}}};
      m_gammaVBO = ctx.newStaticBuffer(BufferUse::Vertex, verts, 32, 4);
      ObjToken<ITexture> texs[] = {{}, m_gammaLUT.get()};
      m_gammaBinding = ctx.newShaderDataBinding(m_gammaShader, m_gammaVBO.get(), {}, {}, 0, nullptr, nullptr, 2, texs,
                                                nullptr, nullptr);
      return true;
    } BooTrace);
  }

  void DestroyGammaResources() {
    m_gammaBinding.reset();
    m_gammaVBO.reset();
    m_gammaLUT.reset();
    m_gammaShader.reset();
  }

public:
  VulkanDataFactoryImpl(IGraphicsContext* parent, VulkanContext* ctx);
  ~VulkanDataFactoryImpl() { assert(m_descPoolHead == nullptr && "Dangling descriptor pools detected"); }

  Platform platform() const override { return Platform::Vulkan; }
  const SystemChar* platformName() const override { return _SYS_STR("Vulkan"); }

  boo::ObjToken<VulkanDescriptorPool> allocateDescriptorSets(VkDescriptorSet* out);

  void commitTransaction(const FactoryCommitFunc& __BooTraceArgs) override;
  void lazyCommitTransaction(const FactoryCommitFunc& __BooTraceArgs) override;

  boo::ObjToken<IGraphicsBufferD> newPoolBuffer(BufferUse use, size_t stride, size_t count __BooTraceArgs) override;

  void setDisplayGamma(float gamma) override {
    m_gamma = gamma;
    UpdateGammaLUT(m_gammaLUT.get(), gamma);
  }

  bool isTessellationSupported(uint32_t& maxPatchSizeOut) override {
    maxPatchSizeOut = 0;
    if (!m_ctx->m_features.tessellationShader)
      return false;
    maxPatchSizeOut = m_ctx->m_gpuProps.limits.maxTessellationPatchSize;
    return true;
  }

  void waitUntilShadersReady() override {
    m_pipelineQueue.waitUntilReady();
  }

  bool areShadersReady() override {
    return m_pipelineQueue.isReady();
  }
};

static void ThrowIfFailed(VkResult res) {
  if (res != VK_SUCCESS)
    Log.report(logvisor::Fatal, FMT_STRING("{}\n"), res);
}

static VKAPI_ATTR VkBool32 VKAPI_CALL dbgFunc(VkDebugReportFlagsEXT msgFlags, VkDebugReportObjectTypeEXT objType,
                                              uint64_t srcObject, size_t location, int32_t msgCode,
                                              const char* pLayerPrefix, const char* pMsg, void* pUserData) {
  if (msgFlags & VK_DEBUG_REPORT_ERROR_BIT_EXT) {
    Log.report(logvisor::Error, FMT_STRING("[{}] Code {} : {}"), pLayerPrefix, msgCode, pMsg);
  } else if (msgFlags & VK_DEBUG_REPORT_WARNING_BIT_EXT) {
    Log.report(logvisor::Warning, FMT_STRING("[{}] Code {} : {}"), pLayerPrefix, msgCode, pMsg);
  } else if (msgFlags & VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT) {
    Log.report(logvisor::Warning, FMT_STRING("[{}] Code {} : {}"), pLayerPrefix, msgCode, pMsg);
  } else if (msgFlags & VK_DEBUG_REPORT_INFORMATION_BIT_EXT) {
    Log.report(logvisor::Info, FMT_STRING("[{}] Code {} : {}"), pLayerPrefix, msgCode, pMsg);
  } else if (msgFlags & VK_DEBUG_REPORT_DEBUG_BIT_EXT) {
    Log.report(logvisor::Info, FMT_STRING("[{}] Code {} : {}"), pLayerPrefix, msgCode, pMsg);
  }

  /*
   * false indicates that layer should not bail-out of an
   * API call that had validation failures. This may mean that the
   * app dies inside the driver due to invalid parameter(s).
   * That's what would happen without validation layers, so we'll
   * keep that behavior here.
   */
  return VK_FALSE;
}

static void SetImageLayout(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspectMask,
                           VkImageLayout old_image_layout, VkImageLayout new_image_layout, uint32_t mipCount,
                           uint32_t layerCount, uint32_t baseMipLevel = 0) {
  VkImageMemoryBarrier imageMemoryBarrier = {};
  imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  imageMemoryBarrier.pNext = nullptr;
  imageMemoryBarrier.srcAccessMask = 0;
  imageMemoryBarrier.dstAccessMask = 0;
  imageMemoryBarrier.oldLayout = old_image_layout;
  imageMemoryBarrier.newLayout = new_image_layout;
  imageMemoryBarrier.image = image;
  imageMemoryBarrier.subresourceRange.aspectMask = aspectMask;
  imageMemoryBarrier.subresourceRange.baseMipLevel = baseMipLevel;
  imageMemoryBarrier.subresourceRange.levelCount = mipCount;
  imageMemoryBarrier.subresourceRange.layerCount = layerCount;
  imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

  VkPipelineStageFlags src_stages = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
  VkPipelineStageFlags dest_stages = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;

  switch (old_image_layout) {
  case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
    imageMemoryBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
    src_stages = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
    break;
  case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
    imageMemoryBarrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    src_stages = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
    break;
  case VK_IMAGE_LAYOUT_PREINITIALIZED:
    imageMemoryBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    break;
  case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
    imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    src_stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
    break;
  case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
    imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    src_stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
    break;
  case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
    imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    src_stages = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
    break;
  default:
    break;
  }

  switch (new_image_layout) {
  case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
    imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    dest_stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
    break;
  case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
    imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    dest_stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
    break;
  case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
    imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dest_stages = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
    break;
  case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
    imageMemoryBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
    dest_stages = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
    break;
  case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
    imageMemoryBarrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dest_stages = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
    break;
  case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
    imageMemoryBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    dest_stages = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
    break;
  default:
    break;
  }

  vk::CmdPipelineBarrier(cmd, src_stages, dest_stages, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);
}

static VkResult InitGlobalExtensionProperties(VulkanContext::LayerProperties& layerProps) {
  VkExtensionProperties* instance_extensions;
  uint32_t instance_extension_count;
  VkResult res;
  char* layer_name = nullptr;

  layer_name = layerProps.properties.layerName;

  do {
    res = vk::EnumerateInstanceExtensionProperties(layer_name, &instance_extension_count, nullptr);
    if (res)
      return res;

    if (instance_extension_count == 0) {
      return VK_SUCCESS;
    }

    layerProps.extensions.resize(instance_extension_count);
    instance_extensions = layerProps.extensions.data();
    res = vk::EnumerateInstanceExtensionProperties(layer_name, &instance_extension_count, instance_extensions);
  } while (res == VK_INCOMPLETE);

  return res;
}

/*
 * Return 1 (true) if all layer names specified in check_names
 * can be found in given layer properties.
 */
static void demo_check_layers(const std::vector<VulkanContext::LayerProperties>& layerProps,
                              const std::vector<const char*>& layerNames) {
  uint32_t check_count = layerNames.size();
  uint32_t layer_count = layerProps.size();
  for (uint32_t i = 0; i < check_count; i++) {
    VkBool32 found = 0;
    for (uint32_t j = 0; j < layer_count; j++) {
      if (!strcmp(layerNames[i], layerProps[j].properties.layerName)) {
        found = 1;
      }
    }
    if (!found) {
      Log.report(logvisor::Fatal, FMT_STRING("Cannot find layer: {}"), layerNames[i]);
    }
  }
}

bool VulkanContext::initVulkan(std::string_view appName, PFN_vkGetInstanceProcAddr getVkProc) {
  vk::init_dispatch_table_top(getVkProc);

  if (!glslang::InitializeProcess()) {
    Log.report(logvisor::Error, FMT_STRING("unable to initialize glslang"));
    return false;
  }

  uint32_t instanceLayerCount;
  VkLayerProperties* vkProps = nullptr;
  VkResult res;

  /*
   * It's possible, though very rare, that the number of
   * instance layers could change. For example, installing something
   * could include new layers that the loader would pick up
   * between the initial query for the count and the
   * request for VkLayerProperties. The loader indicates that
   * by returning a VK_INCOMPLETE status and will update the
   * the count parameter.
   * The count parameter will be updated with the number of
   * entries loaded into the data pointer - in case the number
   * of layers went down or is smaller than the size given.
   */
#ifdef _WIN32
  char* vkSdkPath = getenv("VK_SDK_PATH");
  if (vkSdkPath) {
    std::string str = "VK_LAYER_PATH=";
    str += vkSdkPath;
    str += "\\Bin";
    _putenv(str.c_str());
  }
#else
  setenv("VK_LAYER_PATH", "/usr/share/vulkan/explicit_layer.d", 1);
#endif
  do {
    ThrowIfFailed(vk::EnumerateInstanceLayerProperties(&instanceLayerCount, nullptr));

    if (instanceLayerCount == 0)
      break;

    vkProps = (VkLayerProperties*)realloc(vkProps, instanceLayerCount * sizeof(VkLayerProperties));

    res = vk::EnumerateInstanceLayerProperties(&instanceLayerCount, vkProps);
  } while (res == VK_INCOMPLETE);

  /*
   * Now gather the extension list for each instance layer.
   */
  for (uint32_t i = 0; i < instanceLayerCount; ++i) {
    LayerProperties layerProps;
    layerProps.properties = vkProps[i];
    ThrowIfFailed(InitGlobalExtensionProperties(layerProps));
    m_instanceLayerProperties.push_back(layerProps);
  }
  free(vkProps);

  /* need platform surface extensions */
  m_instanceExtensionNames.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
#ifdef _WIN32
  m_instanceExtensionNames.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#else
  m_instanceExtensionNames.push_back(VK_KHR_XCB_SURFACE_EXTENSION_NAME);
#endif

#ifndef NDEBUG
  m_layerNames.push_back("VK_LAYER_KHRONOS_validation");
  // m_layerNames.push_back("VK_LAYER_RENDERDOC_Capture");
  // m_layerNames.push_back("VK_LAYER_LUNARG_api_dump");
#endif

  demo_check_layers(m_instanceLayerProperties, m_layerNames);

#ifndef NDEBUG
  /* Enable debug callback extension */
  m_instanceExtensionNames.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
#endif

  /* create the instance */
  VkApplicationInfo appInfo = {};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pNext = nullptr;
  appInfo.pApplicationName = appName.data();
  appInfo.applicationVersion = 1;
  appInfo.pEngineName = "Boo";
  appInfo.engineVersion = 1;
  appInfo.apiVersion = VK_API_VERSION_1_1;

  VkInstanceCreateInfo instInfo = {};
  instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instInfo.pNext = nullptr;
  instInfo.flags = 0;
  instInfo.pApplicationInfo = &appInfo;
  instInfo.enabledLayerCount = m_layerNames.size();
  instInfo.ppEnabledLayerNames = m_layerNames.size() ? m_layerNames.data() : nullptr;
  instInfo.enabledExtensionCount = m_instanceExtensionNames.size();
  instInfo.ppEnabledExtensionNames = m_instanceExtensionNames.data();

  VkResult instRes = vk::CreateInstance(&instInfo, nullptr, &m_instance);
  if (instRes != VK_SUCCESS) {
    Log.report(logvisor::Error,
               FMT_STRING("The Vulkan runtime is installed, but there are no supported "
                   "hardware vendor interfaces present"));
    return false;
  }

#ifndef NDEBUG
  PFN_vkCreateDebugReportCallbackEXT createDebugReportCallback =
      (PFN_vkCreateDebugReportCallbackEXT)vk::GetInstanceProcAddr(m_instance, "vkCreateDebugReportCallbackEXT");
  if (!createDebugReportCallback)
    Log.report(logvisor::Fatal, FMT_STRING("GetInstanceProcAddr: Unable to find vkCreateDebugReportCallbackEXT function."));

  m_destroyDebugReportCallback =
      (PFN_vkDestroyDebugReportCallbackEXT)vk::GetInstanceProcAddr(m_instance, "vkDestroyDebugReportCallbackEXT");
  if (!m_destroyDebugReportCallback)
    Log.report(logvisor::Fatal, FMT_STRING("GetInstanceProcAddr: Unable to find vkDestroyDebugReportCallbackEXT function."));

  VkDebugReportCallbackCreateInfoEXT debugCreateInfo = {};
  debugCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CREATE_INFO_EXT;
  debugCreateInfo.pNext = nullptr;
  debugCreateInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;
  debugCreateInfo.pfnCallback = dbgFunc;
  debugCreateInfo.pUserData = nullptr;
  ThrowIfFailed(createDebugReportCallback(m_instance, &debugCreateInfo, nullptr, &m_debugReportCallback));
#endif

  vk::init_dispatch_table_middle(m_instance, false);

  return true;
}

bool VulkanContext::enumerateDevices() {
  uint32_t gpuCount = 1;
  ThrowIfFailed(vk::EnumeratePhysicalDevices(m_instance, &gpuCount, nullptr));
  if (!gpuCount)
    return false;
  m_gpus.resize(gpuCount);

  ThrowIfFailed(vk::EnumeratePhysicalDevices(m_instance, &gpuCount, m_gpus.data()));
  if (!gpuCount)
    return false;

  vk::GetPhysicalDeviceQueueFamilyProperties(m_gpus[0], &m_queueCount, nullptr);
  if (!m_queueCount)
    return false;

  m_queueProps.resize(m_queueCount);
  vk::GetPhysicalDeviceQueueFamilyProperties(m_gpus[0], &m_queueCount, m_queueProps.data());
  if (!m_queueCount)
    return false;

  /* This is as good a place as any to do this */
  vk::GetPhysicalDeviceMemoryProperties(m_gpus[0], &m_memoryProperties);
  vk::GetPhysicalDeviceProperties(m_gpus[0], &m_gpuProps);

  return true;
}

void VulkanContext::initDevice() {
  if (m_graphicsQueueFamilyIndex == UINT32_MAX)
    Log.report(logvisor::Fatal, FMT_STRING("VulkanContext::m_graphicsQueueFamilyIndex hasn't been initialized"));

  /* create the device and queues */
  VkDeviceQueueCreateInfo queueInfo = {};
  float queuePriorities[1] = {0.0};
  queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueInfo.pNext = nullptr;
  queueInfo.queueCount = 1;
  queueInfo.pQueuePriorities = queuePriorities;
  queueInfo.queueFamilyIndex = m_graphicsQueueFamilyIndex;

  vk::GetPhysicalDeviceFeatures(m_gpus[0], &m_features);
  VkPhysicalDeviceFeatures features = {};
  if (m_features.samplerAnisotropy)
    features.samplerAnisotropy = VK_TRUE;
  if (!m_features.textureCompressionBC)
    Log.report(logvisor::Fatal, FMT_STRING("Vulkan device does not support DXT-format textures"));
  features.textureCompressionBC = VK_TRUE;
  VkShaderStageFlagBits tessellationDescriptorBit = VkShaderStageFlagBits(0);
  if (m_features.tessellationShader) {
    tessellationDescriptorBit = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
    features.tessellationShader = VK_TRUE;
  }
  if (!m_features.dualSrcBlend)
    Log.report(logvisor::Fatal, FMT_STRING("Vulkan device does not support dual-source blending"));
  features.dualSrcBlend = VK_TRUE;

  uint32_t extCount = 0;
  vk::EnumerateDeviceExtensionProperties(m_gpus[0], nullptr, &extCount, nullptr);
  std::vector<VkExtensionProperties> extensions(extCount);
  vk::EnumerateDeviceExtensionProperties(m_gpus[0], nullptr, &extCount, extensions.data());
  bool hasSwapchain = false;
  bool hasDebugMarker = false;
  bool hasGetMemReq2 = false;
  bool hasDedicatedAllocation = false;
  for (const VkExtensionProperties& ext : extensions) {
    if (!hasSwapchain && !strcmp(ext.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME))
      hasSwapchain = true;
    else if (!hasDebugMarker && !strcmp(ext.extensionName, VK_EXT_DEBUG_MARKER_EXTENSION_NAME))
      hasDebugMarker = true;
    else if (!hasGetMemReq2 && !strcmp(ext.extensionName, VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME))
      hasGetMemReq2 = true;
    else if (!hasDedicatedAllocation && !strcmp(ext.extensionName, VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME))
      hasDedicatedAllocation = true;
  }

  if (!hasSwapchain)
    Log.report(logvisor::Fatal, FMT_STRING("Vulkan device does not support swapchains"));

  /* need swapchain device extension */
  m_deviceExtensionNames.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
#ifdef BOO_GRAPHICS_DEBUG_GROUPS
  if (hasDebugMarker) {
    /* Enable debug marker extension if enabled in the build system */
    m_deviceExtensionNames.push_back(VK_EXT_DEBUG_MARKER_EXTENSION_NAME);
  }
#endif

  VmaAllocatorCreateFlags allocFlags = VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT;
  if (hasGetMemReq2 && hasDedicatedAllocation) {
    m_deviceExtensionNames.push_back(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
    m_deviceExtensionNames.push_back(VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);
    allocFlags |= VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT;
  }

  VkDeviceCreateInfo deviceInfo = {};
  deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceInfo.pNext = nullptr;
  deviceInfo.queueCreateInfoCount = 1;
  deviceInfo.pQueueCreateInfos = &queueInfo;
  deviceInfo.enabledLayerCount = m_layerNames.size();
  deviceInfo.ppEnabledLayerNames = deviceInfo.enabledLayerCount ? m_layerNames.data() : nullptr;
  deviceInfo.enabledExtensionCount = m_deviceExtensionNames.size();
  deviceInfo.ppEnabledExtensionNames = deviceInfo.enabledExtensionCount ? m_deviceExtensionNames.data() : nullptr;
  deviceInfo.pEnabledFeatures = &features;

  ThrowIfFailed(vk::CreateDevice(m_gpus[0], &deviceInfo, nullptr, &m_dev));

  vk::init_dispatch_table_bottom(m_instance, m_dev);

  /* allocator */
  VmaVulkanFunctions vulkanFunctions = {};
  vulkanFunctions.vkGetPhysicalDeviceProperties = vk::GetPhysicalDeviceProperties;
  vulkanFunctions.vkGetPhysicalDeviceMemoryProperties = vk::GetPhysicalDeviceMemoryProperties;
  vulkanFunctions.vkAllocateMemory = vk::AllocateMemory;
  vulkanFunctions.vkFreeMemory = vk::FreeMemory;
  vulkanFunctions.vkMapMemory = vk::MapMemory;
  vulkanFunctions.vkUnmapMemory = vk::UnmapMemory;
  vulkanFunctions.vkBindBufferMemory = vk::BindBufferMemory;
  vulkanFunctions.vkBindImageMemory = vk::BindImageMemory;
  vulkanFunctions.vkGetBufferMemoryRequirements = vk::GetBufferMemoryRequirements;
  vulkanFunctions.vkGetImageMemoryRequirements = vk::GetImageMemoryRequirements;
  vulkanFunctions.vkCreateBuffer = vk::CreateBuffer;
  vulkanFunctions.vkDestroyBuffer = vk::DestroyBuffer;
  vulkanFunctions.vkCreateImage = vk::CreateImage;
  vulkanFunctions.vkDestroyImage = vk::DestroyImage;
  if (hasGetMemReq2 && hasDedicatedAllocation) {
    vulkanFunctions.vkGetBufferMemoryRequirements2KHR = reinterpret_cast<PFN_vkGetBufferMemoryRequirements2KHR>(
        vk::GetDeviceProcAddr(m_dev, "vkGetBufferMemoryRequirements2KHR"));
    vulkanFunctions.vkGetImageMemoryRequirements2KHR = reinterpret_cast<PFN_vkGetImageMemoryRequirements2KHR>(
        vk::GetDeviceProcAddr(m_dev, "vkGetImageMemoryRequirements2KHR"));
  }
  VmaAllocatorCreateInfo allocatorInfo = {};
  allocatorInfo.flags = allocFlags;
  allocatorInfo.physicalDevice = m_gpus[0];
  allocatorInfo.device = m_dev;
  allocatorInfo.pVulkanFunctions = &vulkanFunctions;
  ThrowIfFailed(vmaCreateAllocator(&allocatorInfo, &m_allocator));

  // Going to need a command buffer to send the memory barriers in
  // set_image_layout but we couldn't have created one before we knew
  // what our graphics_queue_family_index is, but now that we have it,
  // create the command buffer

  VkCommandPoolCreateInfo cmdPoolInfo = {};
  cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  cmdPoolInfo.pNext = nullptr;
  cmdPoolInfo.queueFamilyIndex = m_graphicsQueueFamilyIndex;
  cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  ThrowIfFailed(vk::CreateCommandPool(m_dev, &cmdPoolInfo, nullptr, &m_loadPool));

  VkCommandBufferAllocateInfo cmd = {};
  cmd.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cmd.pNext = nullptr;
  cmd.commandPool = m_loadPool;
  cmd.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmd.commandBufferCount = 1;
  ThrowIfFailed(vk::AllocateCommandBuffers(m_dev, &cmd, &m_loadCmdBuf));

  vk::GetDeviceQueue(m_dev, m_graphicsQueueFamilyIndex, 0, &m_queue);
  OPTICK_GPU_INIT_VULKAN(&m_dev, &m_gpus[0], &m_queue, &m_graphicsQueueFamilyIndex, 1, nullptr);

  /* Begin load command buffer here */
  VkCommandBufferBeginInfo cmdBufBeginInfo = {};
  cmdBufBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  cmdBufBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  ThrowIfFailed(vk::BeginCommandBuffer(m_loadCmdBuf, &cmdBufBeginInfo));

  m_sampleCountColor = flp2(std::min(m_gpuProps.limits.framebufferColorSampleCounts, m_sampleCountColor));
  m_sampleCountDepth = flp2(std::min(m_gpuProps.limits.framebufferDepthSampleCounts, m_sampleCountDepth));

  if (m_features.samplerAnisotropy)
    m_anisotropy = std::min(m_gpuProps.limits.maxSamplerAnisotropy, m_anisotropy);
  else
    m_anisotropy = 1;

  VkDescriptorSetLayoutBinding layoutBindings[BOO_GLSL_MAX_UNIFORM_COUNT + BOO_GLSL_MAX_TEXTURE_COUNT];
  for (int i = 0; i < BOO_GLSL_MAX_UNIFORM_COUNT; ++i) {
    layoutBindings[i].binding = i;
    layoutBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    layoutBindings[i].descriptorCount = 1;
    layoutBindings[i].stageFlags =
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | tessellationDescriptorBit;
    layoutBindings[i].pImmutableSamplers = nullptr;
  }
  for (int i = BOO_GLSL_MAX_UNIFORM_COUNT; i < BOO_GLSL_MAX_UNIFORM_COUNT + BOO_GLSL_MAX_TEXTURE_COUNT; ++i) {
    layoutBindings[i].binding = i;
    layoutBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    layoutBindings[i].descriptorCount = 1;
    layoutBindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | tessellationDescriptorBit;
    layoutBindings[i].pImmutableSamplers = nullptr;
  }

  VkDescriptorSetLayoutCreateInfo descriptorLayout = {};
  descriptorLayout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptorLayout.pNext = nullptr;
  descriptorLayout.bindingCount = BOO_GLSL_MAX_UNIFORM_COUNT + BOO_GLSL_MAX_TEXTURE_COUNT;
  descriptorLayout.pBindings = layoutBindings;

  ThrowIfFailed(vk::CreateDescriptorSetLayout(m_dev, &descriptorLayout, nullptr, &m_descSetLayout));

  VkPipelineLayoutCreateInfo pipelineLayout = {};
  pipelineLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayout.setLayoutCount = 1;
  pipelineLayout.pSetLayouts = &m_descSetLayout;
  ThrowIfFailed(vk::CreatePipelineLayout(m_dev, &pipelineLayout, nullptr, &m_pipelinelayout));

  std::string gpuName = m_gpuProps.deviceName;
  Log.report(logvisor::Info, FMT_STRING("Initialized {}"), gpuName);
  Log.report(logvisor::Info, FMT_STRING("Vulkan version {}.{}.{}"), m_gpuProps.apiVersion >> 22,
             (m_gpuProps.apiVersion >> 12) & 0b1111111111, m_gpuProps.apiVersion & 0b111111111111);
  Log.report(logvisor::Info, FMT_STRING("Driver version {}.{}.{}"), m_gpuProps.driverVersion >> 22,
             (m_gpuProps.driverVersion >> 12) & 0b1111111111, m_gpuProps.driverVersion & 0b111111111111);
}

void VulkanContext::destroyDevice() {
  for (auto& s : m_samplers)
    vk::DestroySampler(m_dev, s.second, nullptr);
  m_samplers.clear();

  if (m_passColorOnly) {
    vk::DestroyRenderPass(m_dev, m_passColorOnly, nullptr);
    m_passColorOnly = VK_NULL_HANDLE;
  }

  if (m_passOneSample) {
    vk::DestroyRenderPass(m_dev, m_passOneSample, nullptr);
    m_passOneSample = VK_NULL_HANDLE;
  }

  if (m_pass) {
    vk::DestroyRenderPass(m_dev, m_pass, nullptr);
    m_pass = VK_NULL_HANDLE;
  }

  if (m_pipelinelayout) {
    vk::DestroyPipelineLayout(m_dev, m_pipelinelayout, nullptr);
    m_pipelinelayout = VK_NULL_HANDLE;
  }

  if (m_descSetLayout) {
    vk::DestroyDescriptorSetLayout(m_dev, m_descSetLayout, nullptr);
    m_descSetLayout = VK_NULL_HANDLE;
  }

  if (m_loadPool) {
    vk::DestroyCommandPool(m_dev, m_loadPool, nullptr);
    m_loadPool = VK_NULL_HANDLE;
  }

  if (m_allocator) {
    vmaDestroyAllocator(m_allocator);
    m_allocator = VK_NULL_HANDLE;
  }

#ifndef NDEBUG
  if (m_debugReportCallback) {
    m_destroyDebugReportCallback(m_instance, m_debugReportCallback, nullptr);
    m_debugReportCallback = VK_NULL_HANDLE;
  }
#endif

  if (m_dev) {
    vk::DestroyDevice(m_dev, nullptr);
    m_dev = VK_NULL_HANDLE;
  }

  if (m_instance) {
    vk::DestroyInstance(m_instance, nullptr);
    m_instance = VK_NULL_HANDLE;
  }
}

void VulkanContext::Window::SwapChain::Buffer::setImage(VulkanContext* ctx, VkImage image, uint32_t width,
                                                        uint32_t height) {
  m_image = image;
  if (m_colorView)
    vk::DestroyImageView(ctx->m_dev, m_colorView, nullptr);
  if (m_framebuffer)
    vk::DestroyFramebuffer(ctx->m_dev, m_framebuffer, nullptr);

  /* Create resource views */
  VkImageViewCreateInfo viewCreateInfo = {};
  viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewCreateInfo.pNext = nullptr;
  viewCreateInfo.image = m_image;
  viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  viewCreateInfo.format = ctx->m_displayFormat;
  viewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_R;
  viewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_G;
  viewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_B;
  viewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_A;
  viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  viewCreateInfo.subresourceRange.baseMipLevel = 0;
  viewCreateInfo.subresourceRange.levelCount = 1;
  viewCreateInfo.subresourceRange.baseArrayLayer = 0;
  viewCreateInfo.subresourceRange.layerCount = 1;
  ThrowIfFailed(vk::CreateImageView(ctx->m_dev, &viewCreateInfo, nullptr, &m_colorView));

  /* framebuffer */
  VkFramebufferCreateInfo fbCreateInfo = {};
  fbCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  fbCreateInfo.pNext = nullptr;
  fbCreateInfo.renderPass = ctx->m_passColorOnly;
  fbCreateInfo.attachmentCount = 1;
  fbCreateInfo.width = width;
  fbCreateInfo.height = height;
  fbCreateInfo.layers = 1;
  fbCreateInfo.pAttachments = &m_colorView;
  ThrowIfFailed(vk::CreateFramebuffer(ctx->m_dev, &fbCreateInfo, nullptr, &m_framebuffer));

  m_passBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  m_passBeginInfo.pNext = nullptr;
  m_passBeginInfo.renderPass = ctx->m_passColorOnly;
  m_passBeginInfo.framebuffer = m_framebuffer;
  m_passBeginInfo.renderArea.offset.x = 0;
  m_passBeginInfo.renderArea.offset.y = 0;
  m_passBeginInfo.renderArea.extent.width = width;
  m_passBeginInfo.renderArea.extent.height = height;
  m_passBeginInfo.clearValueCount = 0;
  m_passBeginInfo.pClearValues = nullptr;
}

void VulkanContext::Window::SwapChain::Buffer::destroy(VkDevice dev) {
  if (m_colorView)
    vk::DestroyImageView(dev, m_colorView, nullptr);
  if (m_framebuffer)
    vk::DestroyFramebuffer(dev, m_framebuffer, nullptr);
}

void VulkanContext::initSwapChain(VulkanContext::Window& windowCtx, VkSurfaceKHR surface, VkFormat format,
                                  VkColorSpaceKHR colorspace) {
  m_internalFormat = m_displayFormat = format;
  if (m_deepColor)
    m_internalFormat = VK_FORMAT_R16G16B16A16_UNORM;

  /* bootstrap render passes if needed */
  if (!m_pass) {
    VkAttachmentDescription attachments[2] = {};

    /* color attachment */
    attachments[0].format = m_internalFormat;
    attachments[0].samples = VkSampleCountFlagBits(m_sampleCountColor);
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkAttachmentReference colorAttachmentRef = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    /* depth attachment */
    attachments[1].format = VK_FORMAT_D32_SFLOAT;
    attachments[1].samples = VkSampleCountFlagBits(m_sampleCountDepth);
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkAttachmentReference depthAttachmentRef = {1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    /* render subpass */
    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;

    /* render pass */
    VkRenderPassCreateInfo renderPass = {};
    renderPass.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPass.attachmentCount = 2;
    renderPass.pAttachments = attachments;
    renderPass.subpassCount = 1;
    renderPass.pSubpasses = &subpass;
    ThrowIfFailed(vk::CreateRenderPass(m_dev, &renderPass, nullptr, &m_pass));

    /* render pass one sample */
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    ThrowIfFailed(vk::CreateRenderPass(m_dev, &renderPass, nullptr, &m_passOneSample));

    /* render pass color only */
    attachments[0].format = m_displayFormat;
    renderPass.attachmentCount = 1;
    subpass.pDepthStencilAttachment = nullptr;
    ThrowIfFailed(vk::CreateRenderPass(m_dev, &renderPass, nullptr, &m_passColorOnly));
  }

  VkSurfaceCapabilitiesKHR surfCapabilities;
  ThrowIfFailed(vk::GetPhysicalDeviceSurfaceCapabilitiesKHR(m_gpus[0], surface, &surfCapabilities));

  uint32_t presentModeCount;
  ThrowIfFailed(vk::GetPhysicalDeviceSurfacePresentModesKHR(m_gpus[0], surface, &presentModeCount, nullptr));
  std::unique_ptr<VkPresentModeKHR[]> presentModes(new VkPresentModeKHR[presentModeCount]);

  ThrowIfFailed(vk::GetPhysicalDeviceSurfacePresentModesKHR(m_gpus[0], surface, &presentModeCount, presentModes.get()));

  VkExtent2D swapChainExtent;
  // width and height are either both -1, or both not -1.
  if (surfCapabilities.currentExtent.width == (uint32_t)-1) {
    // If the surface size is undefined, the size is set to
    // the size of the images requested.
    swapChainExtent.width = 50;
    swapChainExtent.height = 50;
  } else {
    // If the surface size is defined, the swap chain size must match
    swapChainExtent = surfCapabilities.currentExtent;
  }

  // If mailbox mode is available, use it, as is the lowest-latency non-
  // tearing mode.  If not, try IMMEDIATE which will usually be available,
  // and is fastest (though it tears).  If not, fall back to FIFO which is
  // always available.
  VkPresentModeKHR swapchainPresentMode = VK_PRESENT_MODE_FIFO_KHR;
  for (size_t i = 0; i < presentModeCount; ++i) {
    if (presentModes[i] == VK_PRESENT_MODE_MAILBOX_KHR) {
      swapchainPresentMode = VK_PRESENT_MODE_MAILBOX_KHR;
      break;
    }
    if ((swapchainPresentMode != VK_PRESENT_MODE_MAILBOX_KHR) && (presentModes[i] == VK_PRESENT_MODE_IMMEDIATE_KHR)) {
      swapchainPresentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
    }
  }

  // Determine the number of VkImage's to use in the swap chain (we desire to
  // own only 1 image at a time, besides the images being displayed and
  // queued for display):
  uint32_t desiredNumberOfSwapChainImages = surfCapabilities.minImageCount + 1;
  if ((surfCapabilities.maxImageCount > 0) && (desiredNumberOfSwapChainImages > surfCapabilities.maxImageCount)) {
    // Application must settle for fewer images than desired:
    desiredNumberOfSwapChainImages = surfCapabilities.maxImageCount;
  }

  VkSurfaceTransformFlagBitsKHR preTransform;
  if (surfCapabilities.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
    preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
  else
    preTransform = surfCapabilities.currentTransform;

  VkSwapchainCreateInfoKHR swapChainInfo = {};
  swapChainInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  swapChainInfo.pNext = nullptr;
  swapChainInfo.surface = surface;
  swapChainInfo.minImageCount = desiredNumberOfSwapChainImages;
  swapChainInfo.imageFormat = format;
  swapChainInfo.imageExtent.width = swapChainExtent.width;
  swapChainInfo.imageExtent.height = swapChainExtent.height;
  swapChainInfo.preTransform = preTransform;
  swapChainInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  swapChainInfo.imageArrayLayers = 1;
  swapChainInfo.presentMode = swapchainPresentMode;
  swapChainInfo.oldSwapchain = nullptr;
  swapChainInfo.clipped = true;
  swapChainInfo.imageColorSpace = colorspace;
  swapChainInfo.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  swapChainInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  swapChainInfo.queueFamilyIndexCount = 0;
  swapChainInfo.pQueueFamilyIndices = nullptr;

  Window::SwapChain& sc = windowCtx.m_swapChains[windowCtx.m_activeSwapChain];
  ThrowIfFailed(vk::CreateSwapchainKHR(m_dev, &swapChainInfo, nullptr, &sc.m_swapChain));
  sc.m_format = format;

  uint32_t swapchainImageCount;
  ThrowIfFailed(vk::GetSwapchainImagesKHR(m_dev, sc.m_swapChain, &swapchainImageCount, nullptr));

  std::unique_ptr<VkImage[]> swapchainImages(new VkImage[swapchainImageCount]);
  ThrowIfFailed(vk::GetSwapchainImagesKHR(m_dev, sc.m_swapChain, &swapchainImageCount, swapchainImages.get()));

  /* images */
  sc.m_bufs.resize(swapchainImageCount);
  for (uint32_t i = 0; i < swapchainImageCount; ++i) {
    Window::SwapChain::Buffer& buf = sc.m_bufs[i];
    buf.setImage(this, swapchainImages[i], swapChainExtent.width, swapChainExtent.height);
  }
}

void VulkanContext::resizeSwapChain(VulkanContext::Window& windowCtx, VkSurfaceKHR surface, VkFormat format,
                                    VkColorSpaceKHR colorspace, const SWindowRect& rect) {
  std::unique_lock<std::mutex> lk(m_resizeLock);
  m_deferredResizes.emplace(windowCtx, surface, format, colorspace, rect);
}

bool VulkanContext::_resizeSwapChains() {
  std::unique_lock<std::mutex> lk(m_resizeLock);
  if (m_deferredResizes.empty())
    return false;

  while (m_deferredResizes.size()) {
    SwapChainResize& resize = m_deferredResizes.front();

    VkSurfaceCapabilitiesKHR surfCapabilities;
    ThrowIfFailed(vk::GetPhysicalDeviceSurfaceCapabilitiesKHR(m_gpus[0], resize.m_surface, &surfCapabilities));

    uint32_t presentModeCount;
    ThrowIfFailed(vk::GetPhysicalDeviceSurfacePresentModesKHR(m_gpus[0], resize.m_surface, &presentModeCount, nullptr));
    std::unique_ptr<VkPresentModeKHR[]> presentModes(new VkPresentModeKHR[presentModeCount]);

    ThrowIfFailed(vk::GetPhysicalDeviceSurfacePresentModesKHR(m_gpus[0], resize.m_surface, &presentModeCount,
                                                              presentModes.get()));

    VkExtent2D swapChainExtent;
    // width and height are either both -1, or both not -1.
    if (surfCapabilities.currentExtent.width == (uint32_t)-1) {
      // If the surface size is undefined, the size is set to
      // the size of the images requested.
      swapChainExtent.width = 50;
      swapChainExtent.height = 50;
    } else {
      // If the surface size is defined, the swap chain size must match
      swapChainExtent = surfCapabilities.currentExtent;
    }

    // If mailbox mode is available, use it, as is the lowest-latency non-
    // tearing mode.  If not, try IMMEDIATE which will usually be available,
    // and is fastest (though it tears).  If not, fall back to FIFO which is
    // always available.
    VkPresentModeKHR swapchainPresentMode = VK_PRESENT_MODE_FIFO_KHR;
    for (size_t i = 0; i < presentModeCount; ++i) {
      if (presentModes[i] == VK_PRESENT_MODE_MAILBOX_KHR) {
        swapchainPresentMode = VK_PRESENT_MODE_MAILBOX_KHR;
        break;
      }
      if ((swapchainPresentMode != VK_PRESENT_MODE_MAILBOX_KHR) && (presentModes[i] == VK_PRESENT_MODE_IMMEDIATE_KHR)) {
        swapchainPresentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
      }
    }

    // Determine the number of VkImage's to use in the swap chain (we desire to
    // own only 1 image at a time, besides the images being displayed and
    // queued for display):
    uint32_t desiredNumberOfSwapChainImages = surfCapabilities.minImageCount + 1;
    if ((surfCapabilities.maxImageCount > 0) && (desiredNumberOfSwapChainImages > surfCapabilities.maxImageCount)) {
      // Application must settle for fewer images than desired:
      desiredNumberOfSwapChainImages = surfCapabilities.maxImageCount;
    }

    VkSurfaceTransformFlagBitsKHR preTransform;
    if (surfCapabilities.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
      preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    else
      preTransform = surfCapabilities.currentTransform;

    Window::SwapChain& oldSc = resize.m_windowCtx.m_swapChains[resize.m_windowCtx.m_activeSwapChain];

    VkSwapchainCreateInfoKHR swapChainInfo = {};
    swapChainInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapChainInfo.pNext = nullptr;
    swapChainInfo.surface = resize.m_surface;
    swapChainInfo.minImageCount = desiredNumberOfSwapChainImages;
    swapChainInfo.imageFormat = resize.m_format;
    swapChainInfo.imageExtent.width = swapChainExtent.width;
    swapChainInfo.imageExtent.height = swapChainExtent.height;
    swapChainInfo.preTransform = preTransform;
    swapChainInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapChainInfo.imageArrayLayers = 1;
    swapChainInfo.presentMode = swapchainPresentMode;
    swapChainInfo.oldSwapchain = oldSc.m_swapChain;
    swapChainInfo.clipped = true;
    swapChainInfo.imageColorSpace = resize.m_colorspace;
    swapChainInfo.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    swapChainInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapChainInfo.queueFamilyIndexCount = 0;
    swapChainInfo.pQueueFamilyIndices = nullptr;

    resize.m_windowCtx.m_activeSwapChain ^= 1;
    Window::SwapChain& sc = resize.m_windowCtx.m_swapChains[resize.m_windowCtx.m_activeSwapChain];
    sc.destroy(m_dev);
    ThrowIfFailed(vk::CreateSwapchainKHR(m_dev, &swapChainInfo, nullptr, &sc.m_swapChain));
    sc.m_format = resize.m_format;

    uint32_t swapchainImageCount;
    ThrowIfFailed(vk::GetSwapchainImagesKHR(m_dev, sc.m_swapChain, &swapchainImageCount, nullptr));

    std::unique_ptr<VkImage[]> swapchainImages(new VkImage[swapchainImageCount]);
    ThrowIfFailed(vk::GetSwapchainImagesKHR(m_dev, sc.m_swapChain, &swapchainImageCount, swapchainImages.get()));

    /* images */
    sc.m_bufs.resize(swapchainImageCount);
    for (uint32_t i = 0; i < swapchainImageCount; ++i) {
      Window::SwapChain::Buffer& buf = sc.m_bufs[i];
      buf.setImage(this, swapchainImages[i], swapChainExtent.width, swapChainExtent.height);
    }

    m_deferredResizes.pop();
  }

  return true;
}

struct VulkanDescriptorPool : ListNode<VulkanDescriptorPool, VulkanDataFactoryImpl*> {
  VkDescriptorPool m_descPool;
  int m_allocatedSets = 0;

  VulkanDescriptorPool(VulkanDataFactoryImpl* factory)
  : ListNode<VulkanDescriptorPool, VulkanDataFactoryImpl*>(factory) {
    OPTICK_EVENT();
    VkDescriptorPoolSize poolSizes[2] = {};
    VkDescriptorPoolCreateInfo descriptorPoolInfo = {};
    descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolInfo.pNext = nullptr;
    descriptorPoolInfo.maxSets = BOO_VK_MAX_DESCRIPTOR_SETS;
    descriptorPoolInfo.poolSizeCount = 2;
    descriptorPoolInfo.pPoolSizes = poolSizes;

    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = BOO_GLSL_MAX_UNIFORM_COUNT * BOO_VK_MAX_DESCRIPTOR_SETS;

    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = BOO_GLSL_MAX_TEXTURE_COUNT * BOO_VK_MAX_DESCRIPTOR_SETS;

    ThrowIfFailed(vk::CreateDescriptorPool(factory->m_ctx->m_dev, &descriptorPoolInfo, nullptr, &m_descPool));
  }

  ~VulkanDescriptorPool() {
    OPTICK_EVENT();
    vk::DestroyDescriptorPool(m_head->m_ctx->m_dev, m_descPool, nullptr);
  }

  static std::unique_lock<std::recursive_mutex> _getHeadLock(VulkanDataFactoryImpl* factory) {
    return std::unique_lock<std::recursive_mutex>{factory->m_dataMutex};
  }
  static VulkanDescriptorPool*& _getHeadPtr(VulkanDataFactoryImpl* factory) { return factory->m_descPoolHead; }
};

boo::ObjToken<VulkanDescriptorPool> VulkanDataFactoryImpl::allocateDescriptorSets(VkDescriptorSet* out) {
  OPTICK_EVENT();
  std::lock_guard<std::recursive_mutex> lk(m_dataMutex);
  boo::ObjToken<VulkanDescriptorPool> pool;
  if (!m_descPoolHead || m_descPoolHead->m_allocatedSets == BOO_VK_MAX_DESCRIPTOR_SETS)
    pool = new VulkanDescriptorPool(this);
  else
    pool = m_descPoolHead;

  VkDescriptorSetLayout layouts[] = {m_ctx->m_descSetLayout, m_ctx->m_descSetLayout};
  VkDescriptorSetAllocateInfo descAllocInfo;
  descAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  descAllocInfo.pNext = nullptr;
  descAllocInfo.descriptorPool = pool->m_descPool;
  descAllocInfo.descriptorSetCount = 2;
  descAllocInfo.pSetLayouts = layouts;
  ThrowIfFailed(vk::AllocateDescriptorSets(m_ctx->m_dev, &descAllocInfo, out));
  pool->m_allocatedSets += 2;

  return pool;
}

struct AllocatedBuffer {
  VkBuffer m_buffer = VK_NULL_HANDLE;
  VmaAllocation m_allocation;

  void* _create(VulkanContext* ctx, const VkBufferCreateInfo* pBufferCreateInfo, VmaMemoryUsage usage) {
    OPTICK_EVENT();
    assert(m_buffer == VK_NULL_HANDLE && "create may only be called once");
    VmaAllocationCreateInfo bufAllocInfo = {};
    bufAllocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    bufAllocInfo.usage = usage;

    VmaAllocationInfo allocInfo;
    ThrowIfFailed(
        vmaCreateBuffer(ctx->m_allocator, pBufferCreateInfo, &bufAllocInfo, &m_buffer, &m_allocation, &allocInfo));
    return allocInfo.pMappedData;
  }

  void* createCPU(VulkanContext* ctx, const VkBufferCreateInfo* pBufferCreateInfo) {
    return _create(ctx, pBufferCreateInfo, VMA_MEMORY_USAGE_CPU_ONLY);
  }

  void* createCPUtoGPU(VulkanContext* ctx, const VkBufferCreateInfo* pBufferCreateInfo) {
    return _create(ctx, pBufferCreateInfo, VMA_MEMORY_USAGE_CPU_TO_GPU);
  }

  void destroy(VulkanContext* ctx) {
    if (m_buffer) {
      OPTICK_EVENT();
      vmaDestroyBuffer(ctx->m_allocator, m_buffer, m_allocation);
      m_buffer = VK_NULL_HANDLE;
    }
  }
};

struct AllocatedImage {
  VkImage m_image = VK_NULL_HANDLE;
  VmaAllocation m_allocation;
  VkImageLayout m_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkImageLayout m_committedLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  void _create(VulkanContext* ctx, const VkImageCreateInfo* pImageCreateInfo, VmaAllocationCreateFlags flags) {
    assert(m_image == VK_NULL_HANDLE && "create may only be called once");
    VmaAllocationCreateInfo bufAllocInfo = {};
    bufAllocInfo.flags = flags;
    bufAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    ThrowIfFailed(vmaCreateImage(ctx->m_allocator, pImageCreateInfo, &bufAllocInfo, &m_image, &m_allocation, nullptr));
  }

  void create(VulkanContext* ctx, const VkImageCreateInfo* pImageCreateInf) { _create(ctx, pImageCreateInf, 0); }

  void createFB(VulkanContext* ctx, const VkImageCreateInfo* pImageCreateInf) {
    _create(ctx, pImageCreateInf, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
  }

  void destroy(VulkanContext* ctx) {
    if (m_image) {
      vmaDestroyImage(ctx->m_allocator, m_image, m_allocation);
      m_image = VK_NULL_HANDLE;
      m_layout = VK_IMAGE_LAYOUT_UNDEFINED;
      m_committedLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    }
  }

  void toLayout(VkCommandBuffer cmdBuf, VkImageAspectFlags aspect, VkImageLayout layout,
                uint32_t mipCount = 1, uint32_t layerCount = 1, uint32_t baseMipLevel = 0) {
    if (layout != m_layout) {
      SetImageLayout(cmdBuf, m_image, aspect, m_layout, layout, mipCount, layerCount, baseMipLevel);
      m_layout = layout;
    }
  }

  void commitLayout() { m_committedLayout = m_layout; }
  void rollbackLayout() { m_layout = m_committedLayout; }
};

struct VulkanData : BaseGraphicsData {
  VulkanContext* m_ctx;

  /* Vertex, Index, Uniform */
  AllocatedBuffer m_constantBuffers[3];
  AllocatedBuffer m_texStagingBuffer;

  explicit VulkanData(VulkanDataFactoryImpl& head __BooTraceArgs)
  : BaseGraphicsData(head __BooTraceArgsUse), m_ctx(head.m_ctx) {}
  ~VulkanData() {
    for (int i = 0; i < 3; ++i)
      m_constantBuffers[i].destroy(m_ctx);
    m_texStagingBuffer.destroy(m_ctx);
  }
};

struct VulkanPool : BaseGraphicsPool {
  VulkanContext* m_ctx;
  AllocatedBuffer m_constantBuffer;

  explicit VulkanPool(VulkanDataFactoryImpl& head __BooTraceArgs)
  : BaseGraphicsPool(head __BooTraceArgsUse), m_ctx(head.m_ctx) {}
  ~VulkanPool() { m_constantBuffer.destroy(m_ctx); }
};

static const VkBufferUsageFlagBits USE_TABLE[] = {VkBufferUsageFlagBits(0), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                                  VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT};

class VulkanGraphicsBufferS : public GraphicsDataNode<IGraphicsBufferS> {
  friend class VulkanDataFactory;
  friend struct VulkanCommandQueue;
  VulkanContext* m_ctx;
  size_t m_sz;
  std::unique_ptr<uint8_t[]> m_stagingBuf;
  VulkanGraphicsBufferS(const boo::ObjToken<BaseGraphicsData>& parent, BufferUse use, VulkanContext* ctx,
                        const void* data, size_t stride, size_t count)
  : GraphicsDataNode<IGraphicsBufferS>(parent)
  , m_ctx(ctx)
  , m_sz(stride * count)
  , m_stagingBuf(new uint8_t[m_sz])
  , m_use(use) {
    memmove(m_stagingBuf.get(), data, m_sz);
    m_bufferInfo.range = m_sz;
  }

public:
  size_t size() const { return m_sz; }
  VkDescriptorBufferInfo m_bufferInfo;
  BufferUse m_use;

  VkDeviceSize sizeForGPU(VulkanContext* ctx, VkDeviceSize offset) {
    m_bufferInfo.offset = offset;
    offset += m_sz;

    if (m_use == BufferUse::Uniform) {
      size_t minOffset = std::max(VkDeviceSize(256), ctx->m_gpuProps.limits.minUniformBufferOffsetAlignment);
      offset = (offset + minOffset - 1) & ~(minOffset - 1);
    }

    return offset;
  }

  void placeForGPU(VkBuffer bufObj, uint8_t* buf) {
    m_bufferInfo.buffer = bufObj;
    memmove(buf + m_bufferInfo.offset, m_stagingBuf.get(), m_sz);
    m_stagingBuf.reset();
  }
};

template <class DataCls>
class VulkanGraphicsBufferD : public GraphicsDataNode<IGraphicsBufferD, DataCls> {
  friend class VulkanDataFactory;
  friend class VulkanDataFactoryImpl;
  friend struct VulkanCommandQueue;
  VulkanContext* m_ctx;
  size_t m_cpuSz;
  std::unique_ptr<uint8_t[]> m_cpuBuf;
  int m_validSlots = 0;
  VulkanGraphicsBufferD(const boo::ObjToken<DataCls>& parent, BufferUse use, VulkanContext* ctx, size_t stride,
                        size_t count)
  : GraphicsDataNode<IGraphicsBufferD, DataCls>(parent)
  , m_ctx(ctx)
  , m_cpuSz(stride * count)
  , m_cpuBuf(new uint8_t[m_cpuSz])
  , m_use(use) {
    m_bufferInfo[0].range = m_cpuSz;
    m_bufferInfo[1].range = m_cpuSz;
  }
  void update(int b);

public:
  VkDescriptorBufferInfo m_bufferInfo[2];
  uint8_t* m_bufferPtrs[2] = {};
  BufferUse m_use;
  void load(const void* data, size_t sz);
  void* map(size_t sz);
  void unmap();

  VkDeviceSize sizeForGPU(VulkanContext* ctx, VkDeviceSize offset) {
    for (int i = 0; i < 2; ++i) {
      m_bufferInfo[i].offset = offset;
      offset += m_cpuSz;

      if (m_use == BufferUse::Uniform) {
        size_t minOffset = std::max(VkDeviceSize(256), ctx->m_gpuProps.limits.minUniformBufferOffsetAlignment);
        offset = (offset + minOffset - 1) & ~(minOffset - 1);
      }
    }

    return offset;
  }

  void placeForGPU(VkBuffer bufObj, uint8_t* buf) {
    m_bufferInfo[0].buffer = bufObj;
    m_bufferInfo[1].buffer = bufObj;
    m_bufferPtrs[0] = buf + m_bufferInfo[0].offset;
    m_bufferPtrs[1] = buf + m_bufferInfo[1].offset;
  }
};

static void MakeSampler(VulkanContext* ctx, VkSampler& sampOut, TextureClampMode mode, int mips) {
  uint32_t key = (uint32_t(mode) << 16) | mips;
  auto search = ctx->m_samplers.find(key);
  if (search != ctx->m_samplers.end()) {
    sampOut = search->second;
    return;
  }

  /* Create linear sampler */
  VkSamplerCreateInfo samplerInfo = {};
  samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerInfo.pNext = nullptr;
  samplerInfo.magFilter = VK_FILTER_LINEAR;
  samplerInfo.minFilter = VK_FILTER_LINEAR;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerInfo.anisotropyEnable = ctx->m_features.samplerAnisotropy;
  samplerInfo.maxAnisotropy = ctx->m_anisotropy;
  samplerInfo.maxLod = mips - 1;
  switch (mode) {
  case TextureClampMode::Repeat:
  default:
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    break;
  case TextureClampMode::ClampToWhite:
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    break;
  case TextureClampMode::ClampToBlack:
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    break;
  case TextureClampMode::ClampToEdge:
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    break;
  case TextureClampMode::ClampToEdgeNearest:
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    break;
  }
  ThrowIfFailed(vk::CreateSampler(ctx->m_dev, &samplerInfo, nullptr, &sampOut));
  ctx->m_samplers[key] = sampOut;
}

class VulkanTextureS : public GraphicsDataNode<ITextureS> {
  friend class VulkanDataFactory;
  VulkanContext* m_ctx;
  TextureFormat m_fmt;
  size_t m_sz;
  size_t m_width, m_height, m_mips;
  TextureClampMode m_clampMode;
  VkFormat m_vkFmt;
  int m_pixelPitchNum = 1;
  int m_pixelPitchDenom = 1;

  VulkanTextureS(const boo::ObjToken<BaseGraphicsData>& parent, VulkanContext* ctx, size_t width, size_t height,
                 size_t mips, TextureFormat fmt, TextureClampMode clampMode, const void* data, size_t sz)
  : GraphicsDataNode<ITextureS>(parent)
  , m_ctx(ctx)
  , m_fmt(fmt)
  , m_sz(sz)
  , m_width(width)
  , m_height(height)
  , m_mips(mips)
  , m_clampMode(clampMode) {
    VkFormat pfmt = VK_FORMAT_UNDEFINED;
    switch (fmt) {
    case TextureFormat::RGBA8:
      pfmt = VK_FORMAT_R8G8B8A8_UNORM;
      m_pixelPitchNum = 4;
      break;
    case TextureFormat::I8:
      pfmt = VK_FORMAT_R8_UNORM;
      break;
    case TextureFormat::I16:
      pfmt = VK_FORMAT_R16_UNORM;
      m_pixelPitchNum = 2;
      break;
    case TextureFormat::DXT1:
      pfmt = VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
      m_pixelPitchNum = 1;
      m_pixelPitchDenom = 2;
      break;
    case TextureFormat::DXT3:
      pfmt = VK_FORMAT_BC2_UNORM_BLOCK;
      m_pixelPitchNum = 1;
      m_pixelPitchDenom = 1;
      break;
    case TextureFormat::DXT5:
      pfmt = VK_FORMAT_BC3_UNORM_BLOCK;
      m_pixelPitchNum = 1;
      m_pixelPitchDenom = 1;
      break;
    case TextureFormat::BPTC:
      pfmt = VK_FORMAT_BC7_UNORM_BLOCK;
      m_pixelPitchNum = 1;
      m_pixelPitchDenom = 1;
      break;
    default:
      Log.report(logvisor::Fatal, FMT_STRING("unsupported tex format"));
    }
    m_vkFmt = pfmt;

    /* create cpu image buffer */
    VkBufferCreateInfo bufCreateInfo = {};
    bufCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufCreateInfo.size = sz;
    bufCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    void* mappedData = m_cpuBuf.createCPU(ctx, &bufCreateInfo);
    memmove(mappedData, data, sz);
  }

public:
  AllocatedBuffer m_cpuBuf;
  AllocatedImage m_gpuTex;
  VkImageView m_gpuView = VK_NULL_HANDLE;
  VkSampler m_sampler = VK_NULL_HANDLE;
  VkDescriptorImageInfo m_descInfo;
  ~VulkanTextureS() {
    vk::DestroyImageView(m_ctx->m_dev, m_gpuView, nullptr);
    m_gpuTex.destroy(m_ctx);
    m_cpuBuf.destroy(m_ctx);
  }

  void _setClampMode(TextureClampMode mode) {
    MakeSampler(m_ctx, m_sampler, mode, m_mips);
    m_descInfo.sampler = m_sampler;
  }

  void setClampMode(TextureClampMode mode) {
    if (m_clampMode == mode)
      return;
    m_clampMode = mode;
    _setClampMode(mode);
  }

  void deleteUploadObjects() { m_cpuBuf.destroy(m_ctx); }

  void placeForGPU(VulkanContext* ctx) {
    /* create gpu image */
    VkImageCreateInfo texCreateInfo = {};
    texCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    texCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    texCreateInfo.format = m_vkFmt;
    texCreateInfo.mipLevels = m_mips;
    texCreateInfo.arrayLayers = 1;
    texCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    texCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    texCreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    texCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    texCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    texCreateInfo.extent = {uint32_t(m_width), uint32_t(m_height), 1};
    texCreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    m_gpuTex.create(m_ctx, &texCreateInfo);

    _setClampMode(m_clampMode);
    m_descInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    /* create image view */
    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.pNext = nullptr;
    viewInfo.image = m_gpuTex.m_image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = m_vkFmt;
    viewInfo.components.r = VK_COMPONENT_SWIZZLE_R;
    viewInfo.components.g = VK_COMPONENT_SWIZZLE_G;
    viewInfo.components.b = VK_COMPONENT_SWIZZLE_B;
    viewInfo.components.a = VK_COMPONENT_SWIZZLE_A;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = m_mips;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    ThrowIfFailed(vk::CreateImageView(ctx->m_dev, &viewInfo, nullptr, &m_gpuView));
    m_descInfo.imageView = m_gpuView;

    /* Since we're going to blit to the texture image, set its layout to
     * DESTINATION_OPTIMAL */
    m_gpuTex.toLayout(ctx->m_loadCmdBuf, VK_IMAGE_ASPECT_COLOR_BIT,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, m_mips);

    VkBufferImageCopy copyRegions[16] = {};
    size_t width = m_width;
    size_t height = m_height;
    size_t regionCount = std::min(size_t(16), m_mips);
    size_t offset = 0;
    for (size_t i = 0; i < regionCount; ++i) {
      size_t regionPitch = width * height * m_pixelPitchNum / m_pixelPitchDenom;

      copyRegions[i].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      copyRegions[i].imageSubresource.mipLevel = i;
      copyRegions[i].imageSubresource.baseArrayLayer = 0;
      copyRegions[i].imageSubresource.layerCount = 1;
      copyRegions[i].imageExtent.width = width;
      copyRegions[i].imageExtent.height = height;
      copyRegions[i].imageExtent.depth = 1;
      copyRegions[i].bufferOffset = offset;

      if (width > 1)
        width /= 2;
      if (height > 1)
        height /= 2;
      offset += regionPitch;
    }

    /* Put the copy command into the command buffer */
    vk::CmdCopyBufferToImage(ctx->m_loadCmdBuf, m_cpuBuf.m_buffer, m_gpuTex.m_image,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, regionCount, copyRegions);

    /* Set the layout for the texture image from DESTINATION_OPTIMAL to
     * SHADER_READ_ONLY */
    m_gpuTex.toLayout(ctx->m_loadCmdBuf, VK_IMAGE_ASPECT_COLOR_BIT,
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, m_mips);
  }

  TextureFormat format() const { return m_fmt; }
};

class VulkanTextureSA : public GraphicsDataNode<ITextureSA> {
  friend class VulkanDataFactory;
  VulkanContext* m_ctx;
  TextureFormat m_fmt;
  size_t m_sz;
  size_t m_width, m_height, m_layers, m_mips;
  TextureClampMode m_clampMode = TextureClampMode::Invalid;
  VkFormat m_vkFmt;
  int m_pixelPitchNum = 1;
  int m_pixelPitchDenom = 1;

  VulkanTextureSA(const boo::ObjToken<BaseGraphicsData>& parent, VulkanContext* ctx, size_t width, size_t height,
                  size_t layers, size_t mips, TextureFormat fmt, TextureClampMode clampMode, const void* data,
                  size_t sz)
  : GraphicsDataNode<ITextureSA>(parent)
  , m_ctx(ctx)
  , m_fmt(fmt)
  , m_sz(sz)
  , m_width(width)
  , m_height(height)
  , m_layers(layers)
  , m_mips(mips)
  , m_clampMode(clampMode) {
    VkFormat pfmt = VK_FORMAT_UNDEFINED;
    switch (fmt) {
    case TextureFormat::RGBA8:
      pfmt = VK_FORMAT_R8G8B8A8_UNORM;
      m_pixelPitchNum = 4;
      break;
    case TextureFormat::I8:
      pfmt = VK_FORMAT_R8_UNORM;
      break;
    case TextureFormat::I16:
      pfmt = VK_FORMAT_R16_UNORM;
      m_pixelPitchNum = 2;
      break;
    default:
      Log.report(logvisor::Fatal, FMT_STRING("unsupported tex format"));
    }
    m_vkFmt = pfmt;

    /* create cpu image buffer */
    VkBufferCreateInfo bufCreateInfo = {};
    bufCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufCreateInfo.size = sz;
    bufCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    void* mappedData = m_cpuBuf.createCPU(ctx, &bufCreateInfo);
    memmove(mappedData, data, sz);
  }

public:
  AllocatedBuffer m_cpuBuf;
  AllocatedImage m_gpuTex;
  VkImageView m_gpuView = VK_NULL_HANDLE;
  VkSampler m_sampler = VK_NULL_HANDLE;
  VkDescriptorImageInfo m_descInfo;
  ~VulkanTextureSA() {
    vk::DestroyImageView(m_ctx->m_dev, m_gpuView, nullptr);
    m_gpuTex.destroy(m_ctx);
    m_cpuBuf.destroy(m_ctx);
  }

  void _setClampMode(TextureClampMode mode) {
    MakeSampler(m_ctx, m_sampler, mode, m_mips);
    m_descInfo.sampler = m_sampler;
  }

  void setClampMode(TextureClampMode mode) {
    if (m_clampMode == mode)
      return;
    m_clampMode = mode;
    _setClampMode(mode);
  }

  void deleteUploadObjects() { m_cpuBuf.destroy(m_ctx); }

  void placeForGPU(VulkanContext* ctx) {
    /* create gpu image */
    VkImageCreateInfo texCreateInfo = {};
    texCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    texCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    texCreateInfo.format = m_vkFmt;
    texCreateInfo.mipLevels = m_mips;
    texCreateInfo.arrayLayers = m_layers;
    texCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    texCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    texCreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    texCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    texCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    texCreateInfo.extent = {uint32_t(m_width), uint32_t(m_height), 1};
    texCreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    m_gpuTex.create(m_ctx, &texCreateInfo);

    _setClampMode(m_clampMode);
    m_descInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    /* create image view */
    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.pNext = nullptr;
    viewInfo.image = m_gpuTex.m_image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    viewInfo.format = m_vkFmt;
    viewInfo.components.r = VK_COMPONENT_SWIZZLE_R;
    viewInfo.components.g = VK_COMPONENT_SWIZZLE_G;
    viewInfo.components.b = VK_COMPONENT_SWIZZLE_B;
    viewInfo.components.a = VK_COMPONENT_SWIZZLE_A;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = m_mips;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = m_layers;

    ThrowIfFailed(vk::CreateImageView(ctx->m_dev, &viewInfo, nullptr, &m_gpuView));
    m_descInfo.imageView = m_gpuView;

    /* Since we're going to blit to the texture image, set its layout to
     * DESTINATION_OPTIMAL */
    m_gpuTex.toLayout(ctx->m_loadCmdBuf, VK_IMAGE_ASPECT_COLOR_BIT,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, m_mips, m_layers);

    VkBufferImageCopy copyRegions[16] = {};
    size_t width = m_width;
    size_t height = m_height;
    size_t regionCount = std::min(size_t(16), m_mips);
    size_t offset = 0;
    for (size_t i = 0; i < regionCount; ++i) {
      size_t regionPitch = width * height * m_layers * m_pixelPitchNum / m_pixelPitchDenom;

      copyRegions[i].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      copyRegions[i].imageSubresource.mipLevel = i;
      copyRegions[i].imageSubresource.baseArrayLayer = 0;
      copyRegions[i].imageSubresource.layerCount = m_layers;
      copyRegions[i].imageExtent.width = width;
      copyRegions[i].imageExtent.height = height;
      copyRegions[i].imageExtent.depth = 1;
      copyRegions[i].bufferOffset = offset;

      if (width > 1)
        width /= 2;
      if (height > 1)
        height /= 2;
      offset += regionPitch;
    }

    /* Put the copy command into the command buffer */
    vk::CmdCopyBufferToImage(ctx->m_loadCmdBuf, m_cpuBuf.m_buffer, m_gpuTex.m_image,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, regionCount, copyRegions);

    /* Set the layout for the texture image from DESTINATION_OPTIMAL to
     * SHADER_READ_ONLY */
    m_gpuTex.toLayout(ctx->m_loadCmdBuf, VK_IMAGE_ASPECT_COLOR_BIT,
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, m_mips, m_layers);
  }

  TextureFormat format() const { return m_fmt; }
  size_t layers() const { return m_layers; }
};

class VulkanTextureD : public GraphicsDataNode<ITextureD> {
  friend class VulkanDataFactory;
  friend struct VulkanCommandQueue;
  size_t m_width;
  size_t m_height;
  TextureFormat m_fmt;
  TextureClampMode m_clampMode;
  VulkanCommandQueue* m_q;
  std::unique_ptr<uint8_t[]> m_stagingBuf;
  size_t m_cpuSz;
  VkDeviceSize m_cpuOffsets[2];
  VkFormat m_vkFmt;
  int m_validSlots = 0;
  VulkanTextureD(const boo::ObjToken<BaseGraphicsData>& parent, VulkanCommandQueue* q, size_t width, size_t height,
                 TextureFormat fmt, TextureClampMode clampMode)
  : GraphicsDataNode<ITextureD>(parent), m_width(width), m_height(height), m_fmt(fmt), m_clampMode(clampMode), m_q(q) {
    VkFormat pfmt = VK_FORMAT_UNDEFINED;
    switch (fmt) {
    case TextureFormat::RGBA8:
      pfmt = VK_FORMAT_R8G8B8A8_UNORM;
      m_cpuSz = width * height * 4;
      break;
    case TextureFormat::I8:
      pfmt = VK_FORMAT_R8_UNORM;
      m_cpuSz = width * height;
      break;
    case TextureFormat::I16:
      pfmt = VK_FORMAT_R16_UNORM;
      m_cpuSz = width * height * 2;
      break;
    default:
      Log.report(logvisor::Fatal, FMT_STRING("unsupported tex format"));
    }
    m_vkFmt = pfmt;
    m_stagingBuf.reset(new uint8_t[m_cpuSz]);
  }
  void update(int b);

public:
  VkBuffer m_cpuBuf = VK_NULL_HANDLE; /* Owned externally */
  uint8_t* m_cpuBufPtrs[2] = {};
  AllocatedImage m_gpuTex[2];
  VkImageView m_gpuView[2];
  VkSampler m_sampler = VK_NULL_HANDLE;
  VkDescriptorImageInfo m_descInfo[2];
  ~VulkanTextureD();

  void _setClampMode(TextureClampMode mode);
  void setClampMode(TextureClampMode mode);
  void load(const void* data, size_t sz);
  void* map(size_t sz);
  void unmap();

  VkDeviceSize sizeForGPU(VulkanContext* ctx, VkDeviceSize offset) {
    for (int i = 0; i < 2; ++i) {
      m_cpuOffsets[i] = offset;
      offset += m_cpuSz;
    }

    return offset;
  }

  void placeForGPU(VulkanContext* ctx, VkBuffer bufObj, uint8_t* buf) {
    m_cpuBuf = bufObj;
    m_cpuBufPtrs[0] = buf + m_cpuOffsets[0];
    m_cpuBufPtrs[1] = buf + m_cpuOffsets[1];

    /* Create images */
    VkImageCreateInfo texCreateInfo = {};
    texCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    texCreateInfo.pNext = nullptr;
    texCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    texCreateInfo.format = m_vkFmt;
    texCreateInfo.extent.width = m_width;
    texCreateInfo.extent.height = m_height;
    texCreateInfo.extent.depth = 1;
    texCreateInfo.mipLevels = 1;
    texCreateInfo.arrayLayers = 1;
    texCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    texCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    texCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    texCreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    texCreateInfo.queueFamilyIndexCount = 0;
    texCreateInfo.pQueueFamilyIndices = nullptr;
    texCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    texCreateInfo.flags = 0;

    _setClampMode(m_clampMode);
    for (int i = 0; i < 2; ++i) {
      /* create gpu image */
      m_gpuTex[i].create(ctx, &texCreateInfo);
      m_descInfo[i].sampler = m_sampler;
      m_descInfo[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.pNext = nullptr;
    viewInfo.image = nullptr;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = m_vkFmt;
    viewInfo.components.r = VK_COMPONENT_SWIZZLE_R;
    viewInfo.components.g = VK_COMPONENT_SWIZZLE_G;
    viewInfo.components.b = VK_COMPONENT_SWIZZLE_B;
    viewInfo.components.a = VK_COMPONENT_SWIZZLE_A;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    for (int i = 0; i < 2; ++i) {
      /* create image view */
      viewInfo.image = m_gpuTex[i].m_image;
      ThrowIfFailed(vk::CreateImageView(ctx->m_dev, &viewInfo, nullptr, &m_gpuView[i]));

      m_descInfo[i].imageView = m_gpuView[i];
    }
  }

  TextureFormat format() const { return m_fmt; }
};

#define MAX_BIND_TEXS 4

class VulkanTextureR : public GraphicsDataNode<ITextureR> {
  friend class VulkanDataFactory;
  friend struct VulkanCommandQueue;
  VulkanCommandQueue* m_q;
  size_t m_width = 0;
  size_t m_height = 0;
  VkSampleCountFlags m_samplesColor, m_samplesDepth;

  size_t m_colorBindCount;
  size_t m_depthBindCount;

  void Setup(VulkanContext* ctx) {
    /* no-ops on first call */
    doDestroy();

    /* color target */
    VkImageCreateInfo texCreateInfo = {};
    texCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    texCreateInfo.pNext = nullptr;
    texCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    texCreateInfo.format = ctx->m_internalFormat;
    texCreateInfo.extent.width = m_width;
    texCreateInfo.extent.height = m_height;
    texCreateInfo.extent.depth = 1;
    texCreateInfo.mipLevels = 1;
    texCreateInfo.arrayLayers = 1;
    texCreateInfo.samples = VkSampleCountFlagBits(m_samplesColor);
    texCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    texCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    texCreateInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    texCreateInfo.queueFamilyIndexCount = 0;
    texCreateInfo.pQueueFamilyIndices = nullptr;
    texCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    texCreateInfo.flags = 0;
    m_colorTex.createFB(ctx, &texCreateInfo);

    /* depth target */
    texCreateInfo.samples = VkSampleCountFlagBits(m_samplesDepth);
    texCreateInfo.format = VK_FORMAT_D32_SFLOAT;
    texCreateInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    m_depthTex.createFB(ctx, &texCreateInfo);

    texCreateInfo.samples = VkSampleCountFlagBits(1);

    for (size_t i = 0; i < m_colorBindCount; ++i) {
      texCreateInfo.format = ctx->m_internalFormat;
      texCreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      m_colorBindTex[i].createFB(ctx, &texCreateInfo);

      m_colorBindDescInfo[i].sampler = m_sampler;
      m_colorBindDescInfo[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    for (size_t i = 0; i < m_depthBindCount; ++i) {
      texCreateInfo.format = VK_FORMAT_D32_SFLOAT;
      texCreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      m_depthBindTex[i].createFB(ctx, &texCreateInfo);

      m_depthBindDescInfo[i].sampler = m_sampler;
      m_depthBindDescInfo[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    /* Create resource views */
    VkImageViewCreateInfo viewCreateInfo = {};
    viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCreateInfo.pNext = nullptr;
    viewCreateInfo.image = m_colorTex.m_image;
    viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewCreateInfo.format = ctx->m_internalFormat;
    viewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_R;
    viewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_G;
    viewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_B;
    viewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_A;
    viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewCreateInfo.subresourceRange.baseMipLevel = 0;
    viewCreateInfo.subresourceRange.levelCount = 1;
    viewCreateInfo.subresourceRange.baseArrayLayer = 0;
    viewCreateInfo.subresourceRange.layerCount = 1;
    ThrowIfFailed(vk::CreateImageView(ctx->m_dev, &viewCreateInfo, nullptr, &m_colorView));

    viewCreateInfo.image = m_depthTex.m_image;
    viewCreateInfo.format = VK_FORMAT_D32_SFLOAT;
    viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    ThrowIfFailed(vk::CreateImageView(ctx->m_dev, &viewCreateInfo, nullptr, &m_depthView));

    for (size_t i = 0; i < m_colorBindCount; ++i) {
      viewCreateInfo.image = m_colorBindTex[i].m_image;
      viewCreateInfo.format = ctx->m_internalFormat;
      viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      ThrowIfFailed(vk::CreateImageView(ctx->m_dev, &viewCreateInfo, nullptr, &m_colorBindView[i]));
      m_colorBindDescInfo[i].imageView = m_colorBindView[i];
    }

    for (size_t i = 0; i < m_depthBindCount; ++i) {
      viewCreateInfo.image = m_depthBindTex[i].m_image;
      viewCreateInfo.format = VK_FORMAT_D32_SFLOAT;
      viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
      ThrowIfFailed(vk::CreateImageView(ctx->m_dev, &viewCreateInfo, nullptr, &m_depthBindView[i]));
      m_depthBindDescInfo[i].imageView = m_depthBindView[i];
    }

    /* framebuffer */
    VkFramebufferCreateInfo fbCreateInfo = {};
    fbCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbCreateInfo.pNext = nullptr;
    fbCreateInfo.renderPass = ctx->m_pass;
    fbCreateInfo.attachmentCount = 2;
    fbCreateInfo.width = m_width;
    fbCreateInfo.height = m_height;
    fbCreateInfo.layers = 1;
    VkImageView attachments[2] = {m_colorView, m_depthView};
    fbCreateInfo.pAttachments = attachments;
    ThrowIfFailed(vk::CreateFramebuffer(ctx->m_dev, &fbCreateInfo, nullptr, &m_framebuffer));

    m_passBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    m_passBeginInfo.pNext = nullptr;
    m_passBeginInfo.renderPass = ctx->m_pass;
    m_passBeginInfo.framebuffer = m_framebuffer;
    m_passBeginInfo.renderArea.offset.x = 0;
    m_passBeginInfo.renderArea.offset.y = 0;
    m_passBeginInfo.renderArea.extent.width = m_width;
    m_passBeginInfo.renderArea.extent.height = m_height;
    m_passBeginInfo.clearValueCount = 0;
    m_passBeginInfo.pClearValues = nullptr;
  }

  VulkanTextureR(const boo::ObjToken<BaseGraphicsData>& parent, VulkanCommandQueue* q, size_t width, size_t height,
                 TextureClampMode clampMode, size_t colorBindCount, size_t depthBindCount);

public:
  AllocatedImage m_colorTex;
  VkImageView m_colorView = VK_NULL_HANDLE;

  AllocatedImage m_depthTex;
  VkImageView m_depthView = VK_NULL_HANDLE;

  AllocatedImage m_colorBindTex[MAX_BIND_TEXS] = {};
  VkImageView m_colorBindView[MAX_BIND_TEXS] = {};
  VkDescriptorImageInfo m_colorBindDescInfo[MAX_BIND_TEXS] = {};

  AllocatedImage m_depthBindTex[MAX_BIND_TEXS] = {};
  VkImageView m_depthBindView[MAX_BIND_TEXS] = {};
  VkDescriptorImageInfo m_depthBindDescInfo[MAX_BIND_TEXS] = {};

  VkFramebuffer m_framebuffer = VK_NULL_HANDLE;
  VkRenderPassBeginInfo m_passBeginInfo = {};

  VkSampler m_sampler = VK_NULL_HANDLE;

  void setClampMode(TextureClampMode mode);
  void doDestroy();
  ~VulkanTextureR();

  void resize(VulkanContext* ctx, size_t width, size_t height) {
    if (width < 1)
      width = 1;
    if (height < 1)
      height = 1;
    m_width = width;
    m_height = height;
    Setup(ctx);
  }

  void initializeBindLayouts(VulkanContext* ctx) {
    for (size_t i = 0; i < m_colorBindCount; ++i)
      m_colorBindTex[i].toLayout(ctx->m_loadCmdBuf, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    for (size_t i = 0; i < m_depthBindCount; ++i)
      m_depthBindTex[i].toLayout(ctx->m_loadCmdBuf, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }

  void toColorTransferSrcLayout(VkCommandBuffer cmdBuf) {
    m_colorTex.toLayout(cmdBuf, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
  }

  void toDepthTransferSrcLayout(VkCommandBuffer cmdBuf) {
    m_depthTex.toLayout(cmdBuf, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
  }

  void toColorAttachmentLayout(VkCommandBuffer cmdBuf) {
    m_colorTex.toLayout(cmdBuf, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  }

  void toDepthAttachmentLayout(VkCommandBuffer cmdBuf) {
    m_depthTex.toLayout(cmdBuf, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
  }

  void toAttachmentLayout(VkCommandBuffer cmdBuf) {
    toColorAttachmentLayout(cmdBuf);
    toDepthAttachmentLayout(cmdBuf);
  }

  void toColorBindTransferDstLayout(VkCommandBuffer cmdBuf, int bindIdx) {
    m_colorBindTex[bindIdx].toLayout(cmdBuf, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  }

  void toDepthBindTransferDstLayout(VkCommandBuffer cmdBuf, int bindIdx) {
    m_depthBindTex[bindIdx].toLayout(cmdBuf, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  }

  void toColorBindShaderReadLayout(VkCommandBuffer cmdBuf, int bindIdx) {
    m_colorBindTex[bindIdx].toLayout(cmdBuf, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }

  void toDepthBindShaderReadLayout(VkCommandBuffer cmdBuf, int bindIdx) {
    m_depthBindTex[bindIdx].toLayout(cmdBuf, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }

  void commitLayouts() {
    m_colorTex.commitLayout();
    m_depthTex.commitLayout();
    for (size_t i = 0; i < m_colorBindCount; ++i)
      m_colorBindTex[i].commitLayout();
    for (size_t i = 0; i < m_depthBindCount; ++i)
      m_depthBindTex[i].commitLayout();
  }

  void rollbackLayouts() {
    m_colorTex.rollbackLayout();
    m_depthTex.rollbackLayout();
    for (size_t i = 0; i < m_colorBindCount; ++i)
      m_colorBindTex[i].rollbackLayout();
    for (size_t i = 0; i < m_depthBindCount; ++i)
      m_depthBindTex[i].rollbackLayout();
  }
};

class VulkanTextureCubeR : public GraphicsDataNode<ITextureCubeR> {
  friend class VulkanDataFactory;
  friend struct VulkanCommandQueue;
  VulkanCommandQueue* m_q;
  size_t m_width;
  size_t m_mipCount = 0;

  void Setup(VulkanContext* ctx) {
    /* no-ops on first call */
    doDestroy();

    setClampMode(TextureClampMode::Repeat);

    /* color target */
    VkImageCreateInfo texCreateInfo = {};
    texCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    texCreateInfo.pNext = nullptr;
    texCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    texCreateInfo.format = ctx->m_internalFormat;
    texCreateInfo.extent.width = m_width;
    texCreateInfo.extent.height = m_width;
    texCreateInfo.extent.depth = 1;
    texCreateInfo.mipLevels = m_mipCount;
    texCreateInfo.arrayLayers = 6;
    texCreateInfo.samples = VkSampleCountFlagBits(1);
    texCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    texCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    texCreateInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                          VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    texCreateInfo.queueFamilyIndexCount = 0;
    texCreateInfo.pQueueFamilyIndices = nullptr;
    texCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    texCreateInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    m_colorTex.createFB(ctx, &texCreateInfo);

    /* depth target */
    texCreateInfo.mipLevels = 1;
    texCreateInfo.samples = VkSampleCountFlagBits(1);
    texCreateInfo.format = VK_FORMAT_D32_SFLOAT;
    texCreateInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    m_depthTex.createFB(ctx, &texCreateInfo);

    m_colorBindDescInfo.sampler = m_sampler;
    m_colorBindDescInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    /* Create resource views */
    VkImageViewCreateInfo viewCreateInfo = {};
    viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCreateInfo.pNext = nullptr;
    viewCreateInfo.image = m_colorTex.m_image;
    viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
    viewCreateInfo.format = ctx->m_internalFormat;
    viewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_R;
    viewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_G;
    viewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_B;
    viewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_A;
    viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewCreateInfo.subresourceRange.baseMipLevel = 0;
    viewCreateInfo.subresourceRange.levelCount = m_mipCount;
    viewCreateInfo.subresourceRange.baseArrayLayer = 0;
    viewCreateInfo.subresourceRange.layerCount = 6;
    ThrowIfFailed(vk::CreateImageView(ctx->m_dev, &viewCreateInfo, nullptr, &m_colorBindView));
    m_colorBindDescInfo.imageView = m_colorBindView;

    viewCreateInfo.image = m_colorTex.m_image;
    viewCreateInfo.subresourceRange.levelCount = 1;
    viewCreateInfo.subresourceRange.layerCount = 1;
    for (int i = 0; i < 6; ++i) {
      viewCreateInfo.subresourceRange.baseArrayLayer = i;
      ThrowIfFailed(vk::CreateImageView(ctx->m_dev, &viewCreateInfo, nullptr, &m_colorView[i]));
    }

    viewCreateInfo.image = m_depthTex.m_image;
    viewCreateInfo.format = VK_FORMAT_D32_SFLOAT;
    viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    for (int i = 0; i < 6; ++i) {
      viewCreateInfo.subresourceRange.baseArrayLayer = i;
      ThrowIfFailed(vk::CreateImageView(ctx->m_dev, &viewCreateInfo, nullptr, &m_depthView[i]));
    }

    /* framebuffer */
    VkFramebufferCreateInfo fbCreateInfo = {};
    fbCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbCreateInfo.pNext = nullptr;
    fbCreateInfo.renderPass = ctx->m_passOneSample;
    fbCreateInfo.attachmentCount = 2;
    fbCreateInfo.width = m_width;
    fbCreateInfo.height = m_width;
    fbCreateInfo.layers = 1;
    VkImageView attachments[2] = {};
    fbCreateInfo.pAttachments = attachments;
    for (int i = 0; i < 6; ++i) {
      attachments[0] = m_colorView[i];
      attachments[1] = m_depthView[i];
      ThrowIfFailed(vk::CreateFramebuffer(ctx->m_dev, &fbCreateInfo, nullptr, &m_framebuffer[i]));

      auto& pbInfo = m_passBeginInfo[i];
      pbInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
      pbInfo.pNext = nullptr;
      pbInfo.renderPass = ctx->m_passOneSample;
      pbInfo.framebuffer = m_framebuffer[i];
      pbInfo.renderArea.offset.x = 0;
      pbInfo.renderArea.offset.y = 0;
      pbInfo.renderArea.extent.width = m_width;
      pbInfo.renderArea.extent.height = m_width;
      pbInfo.clearValueCount = 0;
      pbInfo.pClearValues = nullptr;
    }
  }

  VulkanTextureCubeR(const boo::ObjToken<BaseGraphicsData>& parent, VulkanCommandQueue* q, size_t width, size_t mips);

public:
  AllocatedImage m_colorTex;
  VkImageView m_colorView[6] = {};
  VkImageLayout m_mipsLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  AllocatedImage m_depthTex;
  VkImageView m_depthView[6] = {};

  VkImageView m_colorBindView = VK_NULL_HANDLE;
  VkDescriptorImageInfo m_colorBindDescInfo = {};

  VkFramebuffer m_framebuffer[6] = {};
  VkRenderPassBeginInfo m_passBeginInfo[6] = {};

  VkSampler m_sampler = VK_NULL_HANDLE;

  void setClampMode(TextureClampMode mode);
  void doDestroy();
  ~VulkanTextureCubeR();

  void resize(VulkanContext* ctx, size_t width, size_t mips) {
    if (width < 1)
      width = 1;
    m_width = width;
    m_mipCount = mips;
    Setup(ctx);
  }

  void toColorTransferSrcLayout(VkCommandBuffer cmdBuf) {
    m_colorTex.toLayout(cmdBuf, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 1, 6);
  }

  void toColorAttachmentLayout(VkCommandBuffer cmdBuf) {
    m_colorTex.toLayout(cmdBuf, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 1, 6);
  }

  void toDepthAttachmentLayout(VkCommandBuffer cmdBuf) {
    m_depthTex.toLayout(cmdBuf, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1, 6);
  }

  void toAttachmentLayout(VkCommandBuffer cmdBuf) {
    toColorAttachmentLayout(cmdBuf);
    toDepthAttachmentLayout(cmdBuf);
  }

  void toColorShaderReadLayout(VkCommandBuffer cmdBuf) {
    m_colorTex.toLayout(cmdBuf, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, m_mipCount, 6);
  }

  void commitLayouts() {
    m_colorTex.commitLayout();
    m_depthTex.commitLayout();
  }

  void rollbackLayouts() {
    m_colorTex.rollbackLayout();
    m_depthTex.rollbackLayout();
  }
};

static const size_t SEMANTIC_SIZE_TABLE[] = {0, 12, 16, 12, 16, 16, 4, 8, 16, 16, 16, 8};

static const VkFormat SEMANTIC_TYPE_TABLE[] = {
    VK_FORMAT_UNDEFINED,           VK_FORMAT_R32G32B32_SFLOAT,    VK_FORMAT_R32G32B32A32_SFLOAT,
    VK_FORMAT_R32G32B32_SFLOAT,    VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT,
    VK_FORMAT_R8G8B8A8_UNORM,      VK_FORMAT_R32G32_SFLOAT,       VK_FORMAT_R32G32B32A32_SFLOAT,
    VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32_SFLOAT,
};

struct VulkanVertexFormat {
  VkVertexInputBindingDescription m_bindings[2];
  std::unique_ptr<VkVertexInputAttributeDescription[]> m_attributes;
  VkPipelineVertexInputStateCreateInfo m_info;
  size_t m_stride = 0;
  size_t m_instStride = 0;

  VulkanVertexFormat(const VertexFormatInfo& info)
  : m_attributes(new VkVertexInputAttributeDescription[info.elementCount]) {
    m_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    m_info.pNext = nullptr;
    m_info.flags = 0;
    m_info.vertexBindingDescriptionCount = 0;
    m_info.pVertexBindingDescriptions = m_bindings;
    m_info.vertexAttributeDescriptionCount = info.elementCount;
    m_info.pVertexAttributeDescriptions = m_attributes.get();

    for (size_t i = 0; i < info.elementCount; ++i) {
      const VertexElementDescriptor* elemin = &info.elements[i];
      VkVertexInputAttributeDescription& attribute = m_attributes[i];
      int semantic = int(elemin->semantic & boo::VertexSemantic::SemanticMask);
      attribute.location = i;
      attribute.format = SEMANTIC_TYPE_TABLE[semantic];
      if (True(elemin->semantic & boo::VertexSemantic::Instanced)) {
        attribute.binding = 1;
        attribute.offset = m_instStride;
        m_instStride += SEMANTIC_SIZE_TABLE[semantic];
      } else {
        attribute.binding = 0;
        attribute.offset = m_stride;
        m_stride += SEMANTIC_SIZE_TABLE[semantic];
      }
    }

    if (m_stride) {
      m_bindings[0].binding = 0;
      m_bindings[0].stride = m_stride;
      m_bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
      ++m_info.vertexBindingDescriptionCount;
    }
    if (m_instStride) {
      m_bindings[m_info.vertexBindingDescriptionCount].binding = 1;
      m_bindings[m_info.vertexBindingDescriptionCount].stride = m_instStride;
      m_bindings[m_info.vertexBindingDescriptionCount].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;
      ++m_info.vertexBindingDescriptionCount;
    }
  }
};

static const VkPrimitiveTopology PRIMITIVE_TABLE[] = {
    VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, VK_PRIMITIVE_TOPOLOGY_PATCH_LIST};

static const VkBlendFactor BLEND_FACTOR_TABLE[] = {VK_BLEND_FACTOR_ZERO,       VK_BLEND_FACTOR_ONE,
                                                   VK_BLEND_FACTOR_SRC_COLOR,  VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
                                                   VK_BLEND_FACTOR_DST_COLOR,  VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR,
                                                   VK_BLEND_FACTOR_SRC_ALPHA,  VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                                                   VK_BLEND_FACTOR_DST_ALPHA,  VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
                                                   VK_BLEND_FACTOR_SRC1_COLOR, VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR};

class VulkanShaderStage : public GraphicsDataNode<IShaderStage> {
  friend class VulkanDataFactory;
  VulkanContext* m_ctx;
  VkShaderModule m_module;
  VulkanShaderStage(const boo::ObjToken<BaseGraphicsData>& parent, VulkanContext* ctx, const uint8_t* data, size_t size,
                    PipelineStage stage)
  : GraphicsDataNode<IShaderStage>(parent), m_ctx(ctx) {
    VkShaderModuleCreateInfo smCreateInfo = {};
    smCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smCreateInfo.pNext = nullptr;
    smCreateInfo.flags = 0;

    smCreateInfo.codeSize = size;
    smCreateInfo.pCode = (uint32_t*)data;
    ThrowIfFailed(vk::CreateShaderModule(m_ctx->m_dev, &smCreateInfo, nullptr, &m_module));
  }

public:
  ~VulkanShaderStage() { vk::DestroyShaderModule(m_ctx->m_dev, m_module, nullptr); }
  VkShaderModule shader() const { return m_module; }
};

class VulkanShaderPipeline : public GraphicsDataNode<IShaderPipeline> {
protected:
  friend class VulkanDataFactory;
  friend class VulkanDataFactoryImpl;
  friend struct VulkanShaderDataBinding;
  VulkanContext* m_ctx;
  VkPipelineCache m_pipelineCache;
  mutable VulkanVertexFormat m_vtxFmt;
  mutable ObjToken<IShaderStage> m_vertex;
  mutable ObjToken<IShaderStage> m_fragment;
  mutable ObjToken<IShaderStage> m_geometry;
  mutable ObjToken<IShaderStage> m_control;
  mutable ObjToken<IShaderStage> m_evaluation;
  BlendFactor m_srcFac;
  BlendFactor m_dstFac;
  Primitive m_prim;
  ZTest m_depthTest;
  bool m_depthWrite;
  bool m_colorWrite;
  bool m_alphaWrite;
  bool m_overwriteAlpha;
  CullMode m_culling;
  uint32_t m_patchSize;
  bool m_asynchronous;
  mutable std::atomic<VkPipeline> m_pipeline = VK_NULL_HANDLE;

  VulkanShaderPipeline(const boo::ObjToken<BaseGraphicsData>& parent, VulkanContext* ctx, ObjToken<IShaderStage> vertex,
                       ObjToken<IShaderStage> fragment, ObjToken<IShaderStage> geometry, ObjToken<IShaderStage> control,
                       ObjToken<IShaderStage> evaluation, VkPipelineCache pipelineCache, const VertexFormatInfo& vtxFmt,
                       const AdditionalPipelineInfo& info, bool asynchronous)
  : GraphicsDataNode<IShaderPipeline>(parent)
  , m_ctx(ctx)
  , m_pipelineCache(pipelineCache)
  , m_vtxFmt(vtxFmt)
  , m_vertex(vertex)
  , m_fragment(fragment)
  , m_geometry(geometry)
  , m_control(control)
  , m_evaluation(evaluation)
  , m_srcFac(info.srcFac)
  , m_dstFac(info.dstFac)
  , m_prim(info.prim)
  , m_depthTest(info.depthTest)
  , m_depthWrite(info.depthWrite)
  , m_colorWrite(info.colorWrite)
  , m_alphaWrite(info.alphaWrite)
  , m_overwriteAlpha(info.overwriteAlpha)
  , m_culling(info.culling)
  , m_patchSize(info.patchSize)
  , m_asynchronous(asynchronous) {
    if (control && evaluation)
      m_prim = Primitive::Patches;
  }

public:
  ~VulkanShaderPipeline() {
    if (m_pipeline)
      vk::DestroyPipeline(m_ctx->m_dev, m_pipeline, nullptr);
    if (m_pipelineCache)
      vk::DestroyPipelineCache(m_ctx->m_dev, m_pipelineCache, nullptr);
  }
  VulkanShaderPipeline& operator=(const VulkanShaderPipeline&) = delete;
  VulkanShaderPipeline(const VulkanShaderPipeline&) = delete;
  VkPipeline bind(VkRenderPass rPass = 0) const {
    compile(rPass);
    while (m_pipeline == VK_NULL_HANDLE) {}
    return m_pipeline;
  }

  mutable std::atomic_bool m_startCompile = {};
  void compile(VkRenderPass rPass = 0) const {
    bool falseCmp = false;
    if (m_startCompile.compare_exchange_strong(falseCmp, true)) {
      if (!m_pipeline) {
        if (!rPass)
          rPass = m_ctx->m_pass;

        VkCullModeFlagBits cullMode;
        switch (m_culling) {
        case CullMode::None:
        default:
          cullMode = VK_CULL_MODE_NONE;
          break;
        case CullMode::Backface:
          cullMode = VK_CULL_MODE_BACK_BIT;
          break;
        case CullMode::Frontface:
          cullMode = VK_CULL_MODE_FRONT_BIT;
          break;
        }

        std::vector<VkDynamicState> dynamicStateEnables;

        VkPipelineShaderStageCreateInfo stages[5] = {};
        uint32_t numStages = 0;

        if (m_vertex) {
          stages[numStages].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
          stages[numStages].pNext = nullptr;
          stages[numStages].flags = 0;
          stages[numStages].stage = VK_SHADER_STAGE_VERTEX_BIT;
          stages[numStages].module = m_vertex.cast<VulkanShaderStage>()->shader();
          stages[numStages].pName = "main";
          stages[numStages++].pSpecializationInfo = nullptr;
        }

        if (m_fragment) {
          stages[numStages].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
          stages[numStages].pNext = nullptr;
          stages[numStages].flags = 0;
          stages[numStages].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
          stages[numStages].module = m_fragment.cast<VulkanShaderStage>()->shader();
          stages[numStages].pName = "main";
          stages[numStages++].pSpecializationInfo = nullptr;
        }

        if (m_geometry) {
          stages[numStages].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
          stages[numStages].pNext = nullptr;
          stages[numStages].flags = 0;
          stages[numStages].stage = VK_SHADER_STAGE_GEOMETRY_BIT;
          stages[numStages].module = m_geometry.cast<VulkanShaderStage>()->shader();
          stages[numStages].pName = "main";
          stages[numStages++].pSpecializationInfo = nullptr;
        }

        if (m_control) {
          stages[numStages].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
          stages[numStages].pNext = nullptr;
          stages[numStages].flags = 0;
          stages[numStages].stage = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
          stages[numStages].module = m_control.cast<VulkanShaderStage>()->shader();
          stages[numStages].pName = "main";
          stages[numStages++].pSpecializationInfo = nullptr;
        }

        if (m_evaluation) {
          stages[numStages].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
          stages[numStages].pNext = nullptr;
          stages[numStages].flags = 0;
          stages[numStages].stage = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
          stages[numStages].module = m_evaluation.cast<VulkanShaderStage>()->shader();
          stages[numStages].pName = "main";
          stages[numStages++].pSpecializationInfo = nullptr;
        }

        VkPipelineInputAssemblyStateCreateInfo assemblyInfo = {};
        assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        assemblyInfo.pNext = nullptr;
        assemblyInfo.flags = 0;
        assemblyInfo.topology = PRIMITIVE_TABLE[int(m_prim)];
        assemblyInfo.primitiveRestartEnable = m_prim == Primitive::TriStrips;

        VkPipelineTessellationStateCreateInfo tessInfo = {};
        tessInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO;
        tessInfo.pNext = nullptr;
        tessInfo.flags = 0;
        tessInfo.patchControlPoints = m_patchSize;

        VkPipelineViewportStateCreateInfo viewportInfo = {};
        viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportInfo.pNext = nullptr;
        viewportInfo.flags = 0;
        viewportInfo.viewportCount = 1;
        viewportInfo.pViewports = nullptr;
        viewportInfo.scissorCount = 1;
        viewportInfo.pScissors = nullptr;
        dynamicStateEnables.push_back(VK_DYNAMIC_STATE_VIEWPORT);
        dynamicStateEnables.push_back(VK_DYNAMIC_STATE_SCISSOR);
#if AMD_PAL_HACK
        dynamicStateEnables.push_back(VK_DYNAMIC_STATE_BLEND_CONSTANTS);
#endif

        VkPipelineRasterizationStateCreateInfo rasterizationInfo = {};
        rasterizationInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizationInfo.pNext = nullptr;
        rasterizationInfo.flags = 0;
        rasterizationInfo.depthClampEnable = VK_FALSE;
        rasterizationInfo.rasterizerDiscardEnable = VK_FALSE;
        rasterizationInfo.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizationInfo.cullMode = cullMode;
        rasterizationInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizationInfo.depthBiasEnable = VK_FALSE;
        rasterizationInfo.lineWidth = 1.f;

        VkPipelineMultisampleStateCreateInfo multisampleInfo = {};
        multisampleInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampleInfo.pNext = nullptr;
        multisampleInfo.flags = 0;
        multisampleInfo.rasterizationSamples = VkSampleCountFlagBits(m_ctx->m_sampleCountColor);

        VkPipelineDepthStencilStateCreateInfo depthStencilInfo = {};
        depthStencilInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencilInfo.pNext = nullptr;
        depthStencilInfo.flags = 0;
        depthStencilInfo.depthTestEnable = m_depthTest != ZTest::None;
        depthStencilInfo.depthWriteEnable = m_depthWrite;
        depthStencilInfo.front.compareOp = VK_COMPARE_OP_ALWAYS;
        depthStencilInfo.back.compareOp = VK_COMPARE_OP_ALWAYS;

        switch (m_depthTest) {
        case ZTest::None:
        default:
          depthStencilInfo.depthCompareOp = VK_COMPARE_OP_ALWAYS;
          break;
        case ZTest::LEqual:
          depthStencilInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
          break;
        case ZTest::Greater:
          depthStencilInfo.depthCompareOp = VK_COMPARE_OP_GREATER;
          break;
        case ZTest::Equal:
          depthStencilInfo.depthCompareOp = VK_COMPARE_OP_EQUAL;
          break;
        case ZTest::GEqual:
          depthStencilInfo.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
          break;
        }

        VkPipelineColorBlendAttachmentState colorAttachment = {};
        colorAttachment.blendEnable = m_dstFac != BlendFactor::Zero;
        if (m_srcFac == BlendFactor::Subtract || m_dstFac == BlendFactor::Subtract) {
          colorAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
          colorAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
          colorAttachment.colorBlendOp = VK_BLEND_OP_REVERSE_SUBTRACT;
          if (m_overwriteAlpha) {
            colorAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            colorAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            colorAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
          } else {
            colorAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            colorAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            colorAttachment.alphaBlendOp = VK_BLEND_OP_REVERSE_SUBTRACT;
          }
        } else {
          colorAttachment.srcColorBlendFactor = BLEND_FACTOR_TABLE[int(m_srcFac)];
          colorAttachment.dstColorBlendFactor = BLEND_FACTOR_TABLE[int(m_dstFac)];
          colorAttachment.colorBlendOp = VK_BLEND_OP_ADD;
          if (m_overwriteAlpha) {
            colorAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            colorAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
          } else {
            colorAttachment.srcAlphaBlendFactor = BLEND_FACTOR_TABLE[int(m_srcFac)];
            colorAttachment.dstAlphaBlendFactor = BLEND_FACTOR_TABLE[int(m_dstFac)];
          }
          colorAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
        }
        colorAttachment.colorWriteMask =
            (m_colorWrite ? (VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT) : 0) |
            (m_alphaWrite ? VK_COLOR_COMPONENT_A_BIT : 0);

        VkPipelineColorBlendStateCreateInfo colorBlendInfo = {};
        colorBlendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlendInfo.pNext = nullptr;
        colorBlendInfo.flags = 0;
        colorBlendInfo.logicOpEnable = VK_FALSE;
        colorBlendInfo.attachmentCount = 1;
        colorBlendInfo.pAttachments = &colorAttachment;

        VkPipelineDynamicStateCreateInfo dynamicState = {};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.pNext = nullptr;
        dynamicState.pDynamicStates = dynamicStateEnables.data();
        dynamicState.dynamicStateCount = uint32_t(dynamicStateEnables.size());

        VkGraphicsPipelineCreateInfo pipelineCreateInfo = {};
        pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineCreateInfo.pNext = nullptr;
        pipelineCreateInfo.flags = 0;
        pipelineCreateInfo.stageCount = numStages;
        pipelineCreateInfo.pStages = stages;
        pipelineCreateInfo.pVertexInputState = &m_vtxFmt.m_info;
        pipelineCreateInfo.pInputAssemblyState = &assemblyInfo;
        pipelineCreateInfo.pTessellationState = &tessInfo;
        pipelineCreateInfo.pViewportState = &viewportInfo;
        pipelineCreateInfo.pRasterizationState = &rasterizationInfo;
        pipelineCreateInfo.pMultisampleState = &multisampleInfo;
        pipelineCreateInfo.pDepthStencilState = &depthStencilInfo;
        pipelineCreateInfo.pColorBlendState = &colorBlendInfo;
        pipelineCreateInfo.pDynamicState = &dynamicState;
        pipelineCreateInfo.layout = m_ctx->m_pipelinelayout;
        pipelineCreateInfo.renderPass = rPass;

        VkPipeline p;
        ThrowIfFailed(
            vk::CreateGraphicsPipelines(m_ctx->m_dev, m_pipelineCache, 1, &pipelineCreateInfo, nullptr, &p));
        m_pipeline = p;

        m_vertex.reset();
        m_fragment.reset();
        m_geometry.reset();
        m_control.reset();
        m_evaluation.reset();
      }
    }
  }

  bool isReady() const { return m_pipeline != VK_NULL_HANDLE; }
};

static const VkDescriptorBufferInfo* GetBufferGPUResource(const IGraphicsBuffer* buf, int idx) {
  if (buf->dynamic()) {
    const VulkanGraphicsBufferD<BaseGraphicsData>* cbuf =
        static_cast<const VulkanGraphicsBufferD<BaseGraphicsData>*>(buf);
    return &cbuf->m_bufferInfo[idx];
  } else {
    const VulkanGraphicsBufferS* cbuf = static_cast<const VulkanGraphicsBufferS*>(buf);
    return &cbuf->m_bufferInfo;
  }
}

static const VkDescriptorImageInfo* GetTextureGPUResource(const ITexture* tex, int idx, int bindIdx, bool depth) {
  switch (tex->type()) {
  case TextureType::Dynamic: {
    const VulkanTextureD* ctex = static_cast<const VulkanTextureD*>(tex);
    return &ctex->m_descInfo[idx];
  }
  case TextureType::Static: {
    const VulkanTextureS* ctex = static_cast<const VulkanTextureS*>(tex);
    return &ctex->m_descInfo;
  }
  case TextureType::StaticArray: {
    const VulkanTextureSA* ctex = static_cast<const VulkanTextureSA*>(tex);
    return &ctex->m_descInfo;
  }
  case TextureType::Render: {
    const VulkanTextureR* ctex = static_cast<const VulkanTextureR*>(tex);
    return depth ? &ctex->m_depthBindDescInfo[bindIdx] : &ctex->m_colorBindDescInfo[bindIdx];
  }
  case TextureType::CubeRender: {
    const VulkanTextureCubeR* ctex = static_cast<const VulkanTextureCubeR*>(tex);
    return &ctex->m_colorBindDescInfo;
  }
  default:
    break;
  }
  return nullptr;
}

struct VulkanShaderDataBinding : GraphicsDataNode<IShaderDataBinding> {
  VulkanContext* m_ctx;
  boo::ObjToken<IShaderPipeline> m_pipeline;
  boo::ObjToken<IGraphicsBuffer> m_vbuf;
  boo::ObjToken<IGraphicsBuffer> m_instVbuf;
  boo::ObjToken<IGraphicsBuffer> m_ibuf;
  std::vector<boo::ObjToken<IGraphicsBuffer>> m_ubufs;
  std::vector<std::array<VkDescriptorBufferInfo, 2>> m_ubufOffs;
  VkImageView m_knownViewHandles[2][BOO_GLSL_MAX_TEXTURE_COUNT] = {};
  struct BindTex {
    boo::ObjToken<ITexture> tex;
    int idx;
    bool depth;
  };
  std::vector<BindTex> m_texs;

  VkBuffer m_vboBufs[2][2] = {{}, {}};
  VkDeviceSize m_vboOffs[2][2] = {{}, {}};
  VkBuffer m_iboBufs[2] = {};
  VkDeviceSize m_iboOffs[2] = {};
  boo::ObjToken<VulkanDescriptorPool> m_descPool;
  VkDescriptorSet m_descSets[2] = {};

  size_t m_vertOffset;
  size_t m_instOffset;

#ifndef NDEBUG
  /* Debugging aids */
  bool m_committed = false;
#endif

  VulkanShaderDataBinding(const boo::ObjToken<BaseGraphicsData>& d, VulkanDataFactoryImpl& factory,
                          const boo::ObjToken<IShaderPipeline>& pipeline, const boo::ObjToken<IGraphicsBuffer>& vbuf,
                          const boo::ObjToken<IGraphicsBuffer>& instVbuf, const boo::ObjToken<IGraphicsBuffer>& ibuf,
                          size_t ubufCount, const boo::ObjToken<IGraphicsBuffer>* ubufs, const size_t* ubufOffs,
                          const size_t* ubufSizes, size_t texCount, const boo::ObjToken<ITexture>* texs,
                          const int* bindIdxs, const bool* depthBinds, size_t baseVert, size_t baseInst)
  : GraphicsDataNode<IShaderDataBinding>(d)
  , m_ctx(factory.m_ctx)
  , m_pipeline(pipeline)
  , m_vbuf(vbuf)
  , m_instVbuf(instVbuf)
  , m_ibuf(ibuf) {
    VulkanShaderPipeline* cpipeline = m_pipeline.cast<VulkanShaderPipeline>();
    VulkanVertexFormat& vtxFmt = cpipeline->m_vtxFmt;
    m_vertOffset = baseVert * vtxFmt.m_stride;
    m_instOffset = baseInst * vtxFmt.m_instStride;

    if (ubufOffs && ubufSizes) {
      m_ubufOffs.reserve(ubufCount);
      for (size_t i = 0; i < ubufCount; ++i) {
#ifndef NDEBUG
        if (ubufOffs[i] % 256) {
          Log.report(logvisor::Fatal, FMT_STRING("non-256-byte-aligned uniform-offset {} provided to newShaderDataBinding"),
                     i);
        }
#endif
        std::array<VkDescriptorBufferInfo, 2> fillArr;
        fillArr.fill({VK_NULL_HANDLE, ubufOffs[i], (ubufSizes[i] + 255) & ~255});
        m_ubufOffs.push_back(fillArr);
      }
    }
    m_ubufs.reserve(ubufCount);
    for (size_t i = 0; i < ubufCount; ++i) {
#ifndef NDEBUG
      if (!ubufs[i]) {
        Log.report(logvisor::Fatal, FMT_STRING("null uniform-buffer {} provided to newShaderDataBinding"), i);
      }
#endif
      m_ubufs.push_back(ubufs[i]);
    }
    m_texs.reserve(texCount);
    for (size_t i = 0; i < texCount; ++i) {
      m_texs.push_back({texs[i], bindIdxs ? bindIdxs[i] : 0, depthBinds ? depthBinds[i] : false});
    }

    size_t totalDescs = ubufCount + texCount;
    if (totalDescs > 0)
      m_descPool = factory.allocateDescriptorSets(m_descSets);
  }

  void commit(VulkanContext* ctx) {
    OPTICK_EVENT();
    VkWriteDescriptorSet writes[(BOO_GLSL_MAX_UNIFORM_COUNT + BOO_GLSL_MAX_TEXTURE_COUNT) * 2] = {};
    size_t totalWrites = 0;
    for (int b = 0; b < 2; ++b) {
      if (m_vbuf) {
        const VkDescriptorBufferInfo* vbufInfo = GetBufferGPUResource(m_vbuf.get(), b);
        m_vboBufs[b][0] = vbufInfo->buffer;
        m_vboOffs[b][0] = vbufInfo->offset + m_vertOffset;
      }
      if (m_instVbuf) {
        const VkDescriptorBufferInfo* vbufInfo = GetBufferGPUResource(m_instVbuf.get(), b);
        m_vboBufs[b][1] = vbufInfo->buffer;
        m_vboOffs[b][1] = vbufInfo->offset + m_instOffset;
      }
      if (m_ibuf) {
        const VkDescriptorBufferInfo* ibufInfo = GetBufferGPUResource(m_ibuf.get(), b);
        m_iboBufs[b] = ibufInfo->buffer;
        m_iboOffs[b] = ibufInfo->offset;
      }

      size_t binding = 0;
      if (m_ubufOffs.size()) {
        for (size_t i = 0; i < BOO_GLSL_MAX_UNIFORM_COUNT; ++i) {
          if (i < m_ubufs.size()) {
            VkDescriptorBufferInfo& modInfo = m_ubufOffs[i][b];
            if (modInfo.range) {
              writes[totalWrites].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
              writes[totalWrites].pNext = nullptr;
              writes[totalWrites].dstSet = m_descSets[b];
              writes[totalWrites].descriptorCount = 1;
              writes[totalWrites].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
              const VkDescriptorBufferInfo* origInfo = GetBufferGPUResource(m_ubufs[i].get(), b);
              modInfo.buffer = origInfo->buffer;
              modInfo.offset += origInfo->offset;
              writes[totalWrites].pBufferInfo = &modInfo;
              writes[totalWrites].dstArrayElement = 0;
              writes[totalWrites].dstBinding = binding;
              ++totalWrites;
            }
          }
          ++binding;
        }
      } else {
        for (size_t i = 0; i < BOO_GLSL_MAX_UNIFORM_COUNT; ++i) {
          if (i < m_ubufs.size()) {
            writes[totalWrites].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[totalWrites].pNext = nullptr;
            writes[totalWrites].dstSet = m_descSets[b];
            writes[totalWrites].descriptorCount = 1;
            writes[totalWrites].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[totalWrites].pBufferInfo = GetBufferGPUResource(m_ubufs[i].get(), b);
            writes[totalWrites].dstArrayElement = 0;
            writes[totalWrites].dstBinding = binding;
            ++totalWrites;
          }
          ++binding;
        }
      }

      for (size_t i = 0; i < BOO_GLSL_MAX_TEXTURE_COUNT; ++i) {
        if (i < m_texs.size() && m_texs[i].tex) {
          writes[totalWrites].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
          writes[totalWrites].pNext = nullptr;
          writes[totalWrites].dstSet = m_descSets[b];
          writes[totalWrites].descriptorCount = 1;
          writes[totalWrites].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
          writes[totalWrites].pImageInfo =
              GetTextureGPUResource(m_texs[i].tex.get(), b, m_texs[i].idx, m_texs[i].depth);
          writes[totalWrites].dstArrayElement = 0;
          writes[totalWrites].dstBinding = binding;
          m_knownViewHandles[b][i] = writes[totalWrites].pImageInfo->imageView;
          ++totalWrites;
        }
        ++binding;
      }
    }
    if (totalWrites) {
      OPTICK_EVENT("vk::UpdateDescriptorSets");
      vk::UpdateDescriptorSets(ctx->m_dev, totalWrites, writes, 0, nullptr);
    }

#ifndef NDEBUG
    m_committed = true;
#endif
  }

  void bind(VkCommandBuffer cmdBuf, int b, VkRenderPass rPass = 0) {
#ifndef NDEBUG
    if (!m_committed)
      Log.report(logvisor::Fatal, FMT_STRING("attempted to use uncommitted VulkanShaderDataBinding"));
#endif

    /* Ensure resized texture bindings are re-bound */
    size_t binding = BOO_GLSL_MAX_UNIFORM_COUNT;
    VkWriteDescriptorSet writes[BOO_GLSL_MAX_TEXTURE_COUNT] = {};
    size_t totalWrites = 0;
    for (size_t i = 0; i < BOO_GLSL_MAX_TEXTURE_COUNT; ++i) {
      if (i < m_texs.size() && m_texs[i].tex) {
        const VkDescriptorImageInfo* resComp =
            GetTextureGPUResource(m_texs[i].tex.get(), b, m_texs[i].idx, m_texs[i].depth);
        if (resComp->imageView != m_knownViewHandles[b][i]) {
          writes[totalWrites].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
          writes[totalWrites].pNext = nullptr;
          writes[totalWrites].dstSet = m_descSets[b];
          writes[totalWrites].descriptorCount = 1;
          writes[totalWrites].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
          writes[totalWrites].pImageInfo = resComp;
          writes[totalWrites].dstArrayElement = 0;
          writes[totalWrites].dstBinding = binding;
          ++totalWrites;
          m_knownViewHandles[b][i] = resComp->imageView;
        }
      }
      ++binding;
    }
    if (totalWrites)
      vk::UpdateDescriptorSets(m_ctx->m_dev, totalWrites, writes, 0, nullptr);

    vk::CmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline.cast<VulkanShaderPipeline>()->bind(rPass));
    if (m_descSets[b])
      vk::CmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_ctx->m_pipelinelayout, 0, 1, &m_descSets[b],
                                0, nullptr);

    if (m_vbuf && m_instVbuf)
      vk::CmdBindVertexBuffers(cmdBuf, 0, 2, m_vboBufs[b], m_vboOffs[b]);
    else if (m_vbuf)
      vk::CmdBindVertexBuffers(cmdBuf, 0, 1, m_vboBufs[b], m_vboOffs[b]);
    else if (m_instVbuf)
      vk::CmdBindVertexBuffers(cmdBuf, 1, 1, &m_vboBufs[b][1], &m_vboOffs[b][1]);

    if (m_ibuf)
      vk::CmdBindIndexBuffer(cmdBuf, m_iboBufs[b], m_iboOffs[b], VK_INDEX_TYPE_UINT32);

#if AMD_PAL_HACK
    /* AMD GCN architecture is prone to hanging after binding a new pipeline without also refreshing the
     * device context registers (i.e. viewport, scissor, line width, blend constants). Blend Constants
     * are the simplest register to set within the PAL codebase. */
    float dummy[4] = {};
    vk::CmdSetBlendConstants(cmdBuf, dummy);
#endif
  }
};

struct VulkanCommandQueue final : IGraphicsCommandQueue {
  Platform platform() const { return IGraphicsDataFactory::Platform::Vulkan; }
  const SystemChar* platformName() const { return _SYS_STR("Vulkan"); }
  VulkanContext* m_ctx;
  VulkanContext::Window* m_windowCtx;
  IGraphicsContext* m_parent;

  VkCommandPool m_cmdPool;
  VkCommandBuffer m_cmdBufs[2];
  VkSemaphore m_swapChainReadySem = VK_NULL_HANDLE;
  VkSemaphore m_drawCompleteSem = VK_NULL_HANDLE;
  VkFence m_drawCompleteFence;

  VkCommandPool m_dynamicCmdPool;
  VkCommandBuffer m_dynamicCmdBufs[2];
  VkFence m_dynamicBufFence;

  bool m_running = true;
  bool m_dynamicNeedsReset = false;
  bool m_submitted = false;

  int m_fillBuf = 0;
  int m_drawBuf = 0;

  std::vector<boo::ObjToken<boo::IObj>> m_drawResTokens[2];

  void resetCommandBuffer() {
    ThrowIfFailed(vk::ResetCommandBuffer(m_cmdBufs[m_fillBuf], 0));
    VkCommandBufferBeginInfo cmdBufBeginInfo = {};
    cmdBufBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cmdBufBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    ThrowIfFailed(vk::BeginCommandBuffer(m_cmdBufs[m_fillBuf], &cmdBufBeginInfo));
    OPTICK_GPU_CONTEXT(m_cmdBufs[m_fillBuf]);
  }

  void resetDynamicCommandBuffer() {
    ThrowIfFailed(vk::ResetCommandBuffer(m_dynamicCmdBufs[m_fillBuf], 0));
    VkCommandBufferBeginInfo cmdBufBeginInfo = {};
    cmdBufBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cmdBufBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    ThrowIfFailed(vk::BeginCommandBuffer(m_dynamicCmdBufs[m_fillBuf], &cmdBufBeginInfo));
    OPTICK_GPU_CONTEXT(m_dynamicCmdBufs[m_fillBuf]);
    m_dynamicNeedsReset = false;
  }

  void stallDynamicUpload() {
    if (m_dynamicNeedsReset) {
      ThrowIfFailed(vk::WaitForFences(m_ctx->m_dev, 1, &m_dynamicBufFence, VK_FALSE, -1));
      resetDynamicCommandBuffer();
    }
  }

  VulkanCommandQueue(VulkanContext* ctx, VulkanContext::Window* windowCtx, IGraphicsContext* parent)
  : m_ctx(ctx), m_windowCtx(windowCtx), m_parent(parent) {
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = m_ctx->m_graphicsQueueFamilyIndex;
    ThrowIfFailed(vk::CreateCommandPool(ctx->m_dev, &poolInfo, nullptr, &m_cmdPool));
    ThrowIfFailed(vk::CreateCommandPool(ctx->m_dev, &poolInfo, nullptr, &m_dynamicCmdPool));

    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_cmdPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 2;

    VkCommandBufferBeginInfo cmdBufBeginInfo = {};
    cmdBufBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cmdBufBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    ThrowIfFailed(vk::AllocateCommandBuffers(m_ctx->m_dev, &allocInfo, m_cmdBufs));
    ThrowIfFailed(vk::BeginCommandBuffer(m_cmdBufs[0], &cmdBufBeginInfo));
    OPTICK_GPU_CONTEXT(m_cmdBufs[0]);

    allocInfo.commandPool = m_dynamicCmdPool;
    ThrowIfFailed(vk::AllocateCommandBuffers(m_ctx->m_dev, &allocInfo, m_dynamicCmdBufs));
    ThrowIfFailed(vk::BeginCommandBuffer(m_dynamicCmdBufs[0], &cmdBufBeginInfo));
    OPTICK_GPU_CONTEXT(m_dynamicCmdBufs[0]);

    VkSemaphoreCreateInfo semInfo = {};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    ThrowIfFailed(vk::CreateSemaphore(ctx->m_dev, &semInfo, nullptr, &m_swapChainReadySem));
    ThrowIfFailed(vk::CreateSemaphore(ctx->m_dev, &semInfo, nullptr, &m_drawCompleteSem));

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    ThrowIfFailed(vk::CreateFence(m_ctx->m_dev, &fenceInfo, nullptr, &m_drawCompleteFence));
    ThrowIfFailed(vk::CreateFence(m_ctx->m_dev, &fenceInfo, nullptr, &m_dynamicBufFence));
  }

  void startRenderer() { static_cast<VulkanDataFactoryImpl*>(m_parent->getDataFactory())->SetupGammaResources(); }

  void stopRenderer() {
    m_running = false;
    if (m_submitted && vk::GetFenceStatus(m_ctx->m_dev, m_drawCompleteFence) == VK_NOT_READY)
      vk::WaitForFences(m_ctx->m_dev, 1, &m_drawCompleteFence, VK_FALSE, -1);
    stallDynamicUpload();
    static_cast<VulkanDataFactoryImpl*>(m_parent->getDataFactory())->DestroyGammaResources();
    m_drawResTokens[0].clear();
    m_drawResTokens[1].clear();
    m_boundTarget.reset();
    m_resolveDispSource.reset();
  }

  ~VulkanCommandQueue() {
    if (m_running)
      stopRenderer();

    vk::DestroyFence(m_ctx->m_dev, m_dynamicBufFence, nullptr);
    vk::DestroyFence(m_ctx->m_dev, m_drawCompleteFence, nullptr);
    vk::DestroySemaphore(m_ctx->m_dev, m_drawCompleteSem, nullptr);
    vk::DestroySemaphore(m_ctx->m_dev, m_swapChainReadySem, nullptr);
    vk::DestroyCommandPool(m_ctx->m_dev, m_dynamicCmdPool, nullptr);
    vk::DestroyCommandPool(m_ctx->m_dev, m_cmdPool, nullptr);
  }

  void setShaderDataBinding(const boo::ObjToken<IShaderDataBinding>& binding) {
    VulkanShaderDataBinding* cbind = binding.cast<VulkanShaderDataBinding>();
    cbind->bind(m_cmdBufs[m_fillBuf], m_fillBuf);
    m_drawResTokens[m_fillBuf].push_back(binding.get());
  }

  boo::ObjToken<ITexture> m_boundTarget;
  void setRenderTarget(const boo::ObjToken<ITextureR>& target) {
    VulkanTextureR* ctarget = target.cast<VulkanTextureR>();
    VkCommandBuffer cmdBuf = m_cmdBufs[m_fillBuf];

    if (m_boundTarget.get() != ctarget) {
      if (m_boundTarget)
        vk::CmdEndRenderPass(cmdBuf);

      ctarget->toAttachmentLayout(cmdBuf);

      m_boundTarget = target.get();
      m_drawResTokens[m_fillBuf].push_back(target.get());
      vk::CmdBeginRenderPass(cmdBuf, &ctarget->m_passBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
    }
  }

  static constexpr int CubeFaceRemap[] = {0, 1, 3, 2, 4, 5};
  int m_boundFace = 0;
  void setRenderTarget(const ObjToken<ITextureCubeR>& target, int face) {
    VulkanTextureCubeR* ctarget = target.cast<VulkanTextureCubeR>();
    VkCommandBuffer cmdBuf = m_cmdBufs[m_fillBuf];
    face = CubeFaceRemap[face];

    if (m_boundTarget.get() != ctarget || m_boundFace != face) {
      if (m_boundTarget)
        vk::CmdEndRenderPass(cmdBuf);

      ctarget->toAttachmentLayout(cmdBuf);

      m_boundTarget = target.get();
      m_boundFace = face;
      m_drawResTokens[m_fillBuf].push_back(target.get());
      vk::CmdBeginRenderPass(cmdBuf, &ctarget->m_passBeginInfo[face], VK_SUBPASS_CONTENTS_INLINE);
    }
  }

  void setViewport(const SWindowRect& rect, float znear, float zfar) {
    if (m_boundTarget) {
      size_t texHeight = 0;
      switch (m_boundTarget->type()) {
      case TextureType::Render: texHeight = m_boundTarget.cast<VulkanTextureR>()->m_height; break;
      case TextureType::CubeRender: texHeight = m_boundTarget.cast<VulkanTextureCubeR>()->m_width; break;
      default: break;
      }
      VkViewport vp = {float(rect.location[0]),
                       float(std::max(0, int(texHeight) - rect.location[1] - rect.size[1])),
                       float(rect.size[0]),
                       float(rect.size[1]),
                       znear,
                       zfar};
      vk::CmdSetViewport(m_cmdBufs[m_fillBuf], 0, 1, &vp);
    }
  }

  void setScissor(const SWindowRect& rect) {
    if (m_boundTarget) {
      size_t texHeight = 0;
      switch (m_boundTarget->type()) {
      case TextureType::Render: texHeight = m_boundTarget.cast<VulkanTextureR>()->m_height; break;
      case TextureType::CubeRender: texHeight = m_boundTarget.cast<VulkanTextureCubeR>()->m_width; break;
      default: break;
      }
      VkRect2D vkrect = {
          {int32_t(rect.location[0]), int32_t(std::max(0, int(texHeight) - rect.location[1] - rect.size[1]))},
          {uint32_t(rect.size[0]), uint32_t(rect.size[1])}};
      vk::CmdSetScissor(m_cmdBufs[m_fillBuf], 0, 1, &vkrect);
    }
  }

  std::unordered_map<VulkanTextureR*, std::pair<size_t, size_t>> m_texResizes;
  void resizeRenderTexture(const boo::ObjToken<ITextureR>& tex, size_t width, size_t height) {
    VulkanTextureR* ctex = tex.cast<VulkanTextureR>();
    m_texResizes[ctex] = std::make_pair(width, height);
    m_drawResTokens[m_fillBuf].push_back(tex.get());
  }

  std::unordered_map<VulkanTextureCubeR*, std::pair<size_t, size_t>> m_cubeTexResizes;
  void resizeRenderTexture(const boo::ObjToken<ITextureCubeR>& tex, size_t width, size_t mips) {
    VulkanTextureCubeR* ctex = tex.cast<VulkanTextureCubeR>();
    m_cubeTexResizes[ctex] = std::make_pair(width, mips);
    m_drawResTokens[m_fillBuf].push_back(tex.get());
  }

  void generateMipmaps(const ObjToken<ITextureCubeR>& tex) {
    SCOPED_GRAPHICS_DEBUG_GROUP(this, "VulkanCommandQueue::generateMipmaps", {1.f, 0.f, 0.f, 1.f});
    VulkanTextureCubeR* ctex = tex.cast<VulkanTextureCubeR>();
    VkCommandBuffer cmdBuf = m_cmdBufs[m_fillBuf];
    if (m_boundTarget) {
      vk::CmdEndRenderPass(cmdBuf);
      m_boundTarget.reset();
    }

    ctex->toColorTransferSrcLayout(cmdBuf);

    size_t tmpWidth = ctex->m_width;
    for (size_t i = 1; i < ctex->m_mipCount; i++) {
      VkImageBlit blit = {};

      blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      blit.srcSubresource.layerCount = 6;
      blit.srcSubresource.mipLevel = i-1;
      blit.srcOffsets[1].x = int32_t(tmpWidth);
      blit.srcOffsets[1].y = int32_t(tmpWidth);
      blit.srcOffsets[1].z = 1;

      tmpWidth >>= 1;

      blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      blit.dstSubresource.layerCount = 6;
      blit.dstSubresource.mipLevel = i;
      blit.dstOffsets[1].x = int32_t(tmpWidth);
      blit.dstOffsets[1].y = int32_t(tmpWidth);
      blit.dstOffsets[1].z = 1;

      SetImageLayout(cmdBuf, ctex->m_colorTex.m_image, VK_IMAGE_ASPECT_COLOR_BIT,
                     ctex->m_mipsLayout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, 6, i);

      vk::CmdBlitImage(cmdBuf, ctex->m_colorTex.m_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       ctex->m_colorTex.m_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

      SetImageLayout(cmdBuf, ctex->m_colorTex.m_image, VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 1, 6, i);
    }

    ctex->toColorShaderReadLayout(cmdBuf);
    ctex->m_mipsLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  }

  void schedulePostFrameHandler(std::function<void(void)>&& func) { func(); }

  float m_clearColor[4] = {0.0, 0.0, 0.0, 0.0};
  void setClearColor(const float rgba[4]) {
    m_clearColor[0] = rgba[0];
    m_clearColor[1] = rgba[1];
    m_clearColor[2] = rgba[2];
    m_clearColor[3] = rgba[3];
  }

  void clearTarget(bool render = true, bool depth = true) {
    if (!m_boundTarget)
      return;
    VkClearAttachment clr[2] = {};
    VkClearRect rect = {};
    rect.layerCount = 1;
    switch (m_boundTarget->type()) {
    case TextureType::Render: {
      VulkanTextureR* ctex = m_boundTarget.cast<VulkanTextureR>();
      rect.rect.extent.width = ctex->m_width;
      rect.rect.extent.height = ctex->m_height;
      break;
    }
    case TextureType::CubeRender: {
      VulkanTextureCubeR* ctex = m_boundTarget.cast<VulkanTextureCubeR>();
      rect.rect.extent.width = ctex->m_width;
      rect.rect.extent.height = ctex->m_width;
      break;
    }
    default: break;
    }

    if (render && depth) {
      clr[0].clearValue.color.float32[0] = m_clearColor[0];
      clr[0].clearValue.color.float32[1] = m_clearColor[1];
      clr[0].clearValue.color.float32[2] = m_clearColor[2];
      clr[0].clearValue.color.float32[3] = m_clearColor[3];
      clr[0].aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      clr[1].aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
      clr[1].clearValue.depthStencil.depth = 1.f;
      vk::CmdClearAttachments(m_cmdBufs[m_fillBuf], 2, clr, 1, &rect);
    } else if (render) {
      clr[0].clearValue.color.float32[0] = m_clearColor[0];
      clr[0].clearValue.color.float32[1] = m_clearColor[1];
      clr[0].clearValue.color.float32[2] = m_clearColor[2];
      clr[0].clearValue.color.float32[3] = m_clearColor[3];
      clr[0].aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      vk::CmdClearAttachments(m_cmdBufs[m_fillBuf], 1, clr, 1, &rect);
    } else if (depth) {
      clr[0].aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
      clr[0].clearValue.depthStencil.depth = 1.f;
      vk::CmdClearAttachments(m_cmdBufs[m_fillBuf], 1, clr, 1, &rect);
    }
  }

  void draw(size_t start, size_t count) { vk::CmdDraw(m_cmdBufs[m_fillBuf], count, 1, start, 0); }

  void drawIndexed(size_t start, size_t count, size_t baseVertex) { vk::CmdDrawIndexed(m_cmdBufs[m_fillBuf], count, 1, start, baseVertex, 0); }

  void drawInstances(size_t start, size_t count, size_t instCount, size_t startInst) {
    vk::CmdDraw(m_cmdBufs[m_fillBuf], count, instCount, start, startInst);
  }

  void drawInstancesIndexed(size_t start, size_t count, size_t instCount, size_t startInst) {
    vk::CmdDrawIndexed(m_cmdBufs[m_fillBuf], count, instCount, start, 0, startInst);
  }

  boo::ObjToken<ITextureR> m_resolveDispSource;
  void resolveDisplay(const boo::ObjToken<ITextureR>& source) { m_resolveDispSource = source; }

  bool _resolveDisplay() {
    if (!m_resolveDispSource)
      return false;
    VulkanContext::Window::SwapChain& sc = m_windowCtx->m_swapChains[m_windowCtx->m_activeSwapChain];
    if (!sc.m_swapChain)
      return false;

    VkCommandBuffer cmdBuf = m_cmdBufs[m_drawBuf];
    VulkanTextureR* csource = m_resolveDispSource.cast<VulkanTextureR>();
#ifndef NDEBUG
    if (!csource->m_colorBindCount)
      Log.report(logvisor::Fatal, FMT_STRING("texture provided to resolveDisplay() must have at least 1 color binding"));
#endif

    VkResult res =
        vk::AcquireNextImageKHR(m_ctx->m_dev, sc.m_swapChain, UINT64_MAX, m_swapChainReadySem, nullptr, &sc.m_backBuf);
    if (res == VK_ERROR_OUT_OF_DATE_KHR) {
      return false;
    }
    ThrowIfFailed(res);
    VulkanContext::Window::SwapChain::Buffer& dest = sc.m_bufs[sc.m_backBuf];

    VulkanDataFactoryImpl* dataFactory = static_cast<VulkanDataFactoryImpl*>(m_parent->getDataFactory());
    if (dataFactory->m_gamma != 1.f || m_ctx->m_internalFormat != m_ctx->m_displayFormat) {
      SWindowRect rect(0, 0, csource->m_width, csource->m_height);
      _resolveBindTexture(cmdBuf, csource, rect, true, 0, true, false);
      VulkanShaderDataBinding* gammaBinding = dataFactory->m_gammaBinding.cast<VulkanShaderDataBinding>();

      SetImageLayout(cmdBuf, dest.m_image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 1, 1);

      vk::CmdBeginRenderPass(cmdBuf, &dest.m_passBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

      gammaBinding->m_texs[0].tex = m_resolveDispSource.get();
      gammaBinding->bind(cmdBuf, m_drawBuf, m_ctx->m_passColorOnly);
      vk::CmdDraw(cmdBuf, 4, 1, 0, 0);
      gammaBinding->m_texs[0].tex.reset();

      vk::CmdEndRenderPass(cmdBuf);
      m_boundTarget.reset();

      SetImageLayout(cmdBuf, dest.m_image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, 1, 1);
    } else {
      SetImageLayout(cmdBuf, dest.m_image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, 1);

      csource->toColorTransferSrcLayout(cmdBuf);

      if (csource->m_samplesColor > 1) {
        VkImageResolve resolveInfo = {};
        resolveInfo.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        resolveInfo.srcSubresource.mipLevel = 0;
        resolveInfo.srcSubresource.baseArrayLayer = 0;
        resolveInfo.srcSubresource.layerCount = 1;
        resolveInfo.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        resolveInfo.dstSubresource.mipLevel = 0;
        resolveInfo.dstSubresource.baseArrayLayer = 0;
        resolveInfo.dstSubresource.layerCount = 1;
        resolveInfo.extent.width = csource->m_width;
        resolveInfo.extent.height = csource->m_height;
        resolveInfo.extent.depth = 1;
        vk::CmdResolveImage(cmdBuf, csource->m_colorTex.m_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dest.m_image,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &resolveInfo);
      } else {
        VkImageCopy copyInfo = {};
        copyInfo.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyInfo.srcSubresource.mipLevel = 0;
        copyInfo.srcSubresource.baseArrayLayer = 0;
        copyInfo.srcSubresource.layerCount = 1;
        copyInfo.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyInfo.dstSubresource.mipLevel = 0;
        copyInfo.dstSubresource.baseArrayLayer = 0;
        copyInfo.dstSubresource.layerCount = 1;
        copyInfo.extent.width = csource->m_width;
        copyInfo.extent.height = csource->m_height;
        copyInfo.extent.depth = 1;
        vk::CmdCopyImage(cmdBuf, csource->m_colorTex.m_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dest.m_image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyInfo);
      }

      SetImageLayout(cmdBuf, dest.m_image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, 1, 1);
    }

    m_resolveDispSource.reset();
    return true;
  }

  void _resolveBindTexture(VkCommandBuffer cmdBuf, VulkanTextureR* ctexture, const SWindowRect& rect, bool tlOrigin,
                           int bindIdx, bool color, bool depth) {
    if (color && ctexture->m_colorBindCount) {
      if (ctexture->m_samplesColor <= 1) {
        VkImageCopy copyInfo = {};
        SWindowRect intersectRect = rect.intersect(SWindowRect(0, 0, ctexture->m_width, ctexture->m_height));
        copyInfo.srcOffset.y = tlOrigin ? intersectRect.location[1]
                                        : (ctexture->m_height - intersectRect.size[1] - intersectRect.location[1]);
        copyInfo.srcOffset.x = intersectRect.location[0];
        copyInfo.dstOffset = copyInfo.srcOffset;
        copyInfo.extent.width = intersectRect.size[0];
        copyInfo.extent.height = intersectRect.size[1];
        copyInfo.extent.depth = 1;
        copyInfo.dstSubresource.mipLevel = 0;
        copyInfo.dstSubresource.baseArrayLayer = 0;
        copyInfo.dstSubresource.layerCount = 1;
        copyInfo.srcSubresource.mipLevel = 0;
        copyInfo.srcSubresource.baseArrayLayer = 0;
        copyInfo.srcSubresource.layerCount = 1;

        ctexture->toColorTransferSrcLayout(cmdBuf);

        ctexture->toColorBindTransferDstLayout(cmdBuf, bindIdx);

        copyInfo.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyInfo.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

        vk::CmdCopyImage(cmdBuf, ctexture->m_colorTex.m_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         ctexture->m_colorBindTex[bindIdx].m_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyInfo);

        if (ctexture == m_boundTarget.get())
          ctexture->toColorAttachmentLayout(cmdBuf);

        ctexture->toColorBindShaderReadLayout(cmdBuf, bindIdx);
      } else {
        VkImageResolve resolveInfo = {};
        SWindowRect intersectRect = rect.intersect(SWindowRect(0, 0, ctexture->m_width, ctexture->m_height));
        resolveInfo.srcOffset.y = tlOrigin ? intersectRect.location[1]
                                           : (ctexture->m_height - intersectRect.size[1] - intersectRect.location[1]);
        resolveInfo.srcOffset.x = intersectRect.location[0];
        resolveInfo.dstOffset = resolveInfo.srcOffset;
        resolveInfo.extent.width = intersectRect.size[0];
        resolveInfo.extent.height = intersectRect.size[1];
        resolveInfo.extent.depth = 1;
        resolveInfo.dstSubresource.mipLevel = 0;
        resolveInfo.dstSubresource.baseArrayLayer = 0;
        resolveInfo.dstSubresource.layerCount = 1;
        resolveInfo.srcSubresource.mipLevel = 0;
        resolveInfo.srcSubresource.baseArrayLayer = 0;
        resolveInfo.srcSubresource.layerCount = 1;

        ctexture->toColorTransferSrcLayout(cmdBuf);

        ctexture->toColorBindTransferDstLayout(cmdBuf, bindIdx);

        resolveInfo.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        resolveInfo.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

        vk::CmdResolveImage(cmdBuf, ctexture->m_colorTex.m_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            ctexture->m_colorBindTex[bindIdx].m_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                            &resolveInfo);

        if (ctexture == m_boundTarget.get())
          ctexture->toColorAttachmentLayout(cmdBuf);

        ctexture->toColorBindShaderReadLayout(cmdBuf, bindIdx);
      }
    }

    if (depth && ctexture->m_depthBindCount) {
      if (ctexture->m_samplesDepth <= 1) {
        VkImageCopy copyInfo = {};
        SWindowRect intersectRect = rect.intersect(SWindowRect(0, 0, ctexture->m_width, ctexture->m_height));
        copyInfo.srcOffset.y = tlOrigin ? intersectRect.location[1]
                                        : (ctexture->m_height - intersectRect.size[1] - intersectRect.location[1]);
        copyInfo.srcOffset.x = intersectRect.location[0];
        copyInfo.dstOffset = copyInfo.srcOffset;
        copyInfo.extent.width = intersectRect.size[0];
        copyInfo.extent.height = intersectRect.size[1];
        copyInfo.extent.depth = 1;
        copyInfo.dstSubresource.mipLevel = 0;
        copyInfo.dstSubresource.baseArrayLayer = 0;
        copyInfo.dstSubresource.layerCount = 1;
        copyInfo.srcSubresource.mipLevel = 0;
        copyInfo.srcSubresource.baseArrayLayer = 0;
        copyInfo.srcSubresource.layerCount = 1;

        ctexture->toDepthTransferSrcLayout(cmdBuf);

        ctexture->toDepthBindTransferDstLayout(cmdBuf, bindIdx);

        copyInfo.srcSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        copyInfo.dstSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

        vk::CmdCopyImage(cmdBuf, ctexture->m_depthTex.m_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         ctexture->m_depthBindTex[bindIdx].m_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyInfo);

        if (ctexture == m_boundTarget.get())
          ctexture->toDepthAttachmentLayout(cmdBuf);

        ctexture->toDepthBindShaderReadLayout(cmdBuf, bindIdx);
      } else {
        VkImageResolve resolveInfo = {};
        SWindowRect intersectRect = rect.intersect(SWindowRect(0, 0, ctexture->m_width, ctexture->m_height));
        resolveInfo.srcOffset.y = tlOrigin ? intersectRect.location[1]
                                           : (ctexture->m_height - intersectRect.size[1] - intersectRect.location[1]);
        resolveInfo.srcOffset.x = intersectRect.location[0];
        resolveInfo.dstOffset = resolveInfo.srcOffset;
        resolveInfo.extent.width = intersectRect.size[0];
        resolveInfo.extent.height = intersectRect.size[1];
        resolveInfo.extent.depth = 1;
        resolveInfo.dstSubresource.mipLevel = 0;
        resolveInfo.dstSubresource.baseArrayLayer = 0;
        resolveInfo.dstSubresource.layerCount = 1;
        resolveInfo.srcSubresource.mipLevel = 0;
        resolveInfo.srcSubresource.baseArrayLayer = 0;
        resolveInfo.srcSubresource.layerCount = 1;

        ctexture->toDepthTransferSrcLayout(cmdBuf);

        ctexture->toDepthBindTransferDstLayout(cmdBuf, bindIdx);

        resolveInfo.srcSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        resolveInfo.dstSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

        vk::CmdResolveImage(cmdBuf, ctexture->m_depthTex.m_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            ctexture->m_depthBindTex[bindIdx].m_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                            &resolveInfo);

        if (ctexture == m_boundTarget.get())
          ctexture->toDepthAttachmentLayout(cmdBuf);

        ctexture->toDepthBindShaderReadLayout(cmdBuf, bindIdx);
      }
    }
  }

  void resolveBindTexture(const boo::ObjToken<ITextureR>& texture, const SWindowRect& rect, bool tlOrigin, int bindIdx,
                          bool color, bool depth, bool clearDepth) {
    VkCommandBuffer cmdBuf = m_cmdBufs[m_fillBuf];
    VulkanTextureR* ctexture = texture.cast<VulkanTextureR>();

    vk::CmdEndRenderPass(cmdBuf);
    _resolveBindTexture(cmdBuf, ctexture, rect, tlOrigin, bindIdx, color, depth);
    VkRenderPassBeginInfo* pbInfo = nullptr;
    switch (m_boundTarget->type()) {
    case TextureType::Render: pbInfo = &m_boundTarget.cast<VulkanTextureR>()->m_passBeginInfo; break;
    case TextureType::CubeRender: pbInfo = &m_boundTarget.cast<VulkanTextureCubeR>()->m_passBeginInfo[m_boundFace]; break;
    default: break;
    }
    vk::CmdBeginRenderPass(cmdBuf, pbInfo, VK_SUBPASS_CONTENTS_INLINE);

    if (clearDepth) {
      VkClearAttachment clr = {};
      VkClearRect rect = {};
      rect.layerCount = 1;
      rect.rect.extent.width = ctexture->m_width;
      rect.rect.extent.height = ctexture->m_height;

      clr.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
      clr.clearValue.depthStencil.depth = 1.f;
      vk::CmdClearAttachments(cmdBuf, 1, &clr, 1, &rect);
    }
  }

  void _commitImageLayouts();
  void _rollbackImageLayouts();

  void execute();

#ifdef BOO_GRAPHICS_DEBUG_GROUPS
  void pushDebugGroup(const char* name, const std::array<float, 4>& color) {
    if (vk::CmdDebugMarkerBeginEXT) {
      VkCommandBuffer cmdBuf = m_cmdBufs[m_fillBuf];
      VkDebugMarkerMarkerInfoEXT markerInfo = {};
      markerInfo.sType = VK_STRUCTURE_TYPE_DEBUG_MARKER_MARKER_INFO_EXT;
      markerInfo.pMarkerName = name;
      std::copy(color.begin(), color.end(), markerInfo.color);
      vk::CmdDebugMarkerBeginEXT(cmdBuf, &markerInfo);
    }
  }

  void popDebugGroup() {
    if (vk::CmdDebugMarkerEndEXT) {
      VkCommandBuffer cmdBuf = m_cmdBufs[m_fillBuf];
      vk::CmdDebugMarkerEndEXT(cmdBuf);
    }
  }
#endif
};

void VulkanTextureR::doDestroy() {
  if (m_framebuffer) {
    vk::DestroyFramebuffer(m_q->m_ctx->m_dev, m_framebuffer, nullptr);
    m_framebuffer = VK_NULL_HANDLE;
  }
  if (m_colorView) {
    vk::DestroyImageView(m_q->m_ctx->m_dev, m_colorView, nullptr);
    m_colorView = VK_NULL_HANDLE;
  }
  m_colorTex.destroy(m_q->m_ctx);
  if (m_depthView) {
    vk::DestroyImageView(m_q->m_ctx->m_dev, m_depthView, nullptr);
    m_depthView = VK_NULL_HANDLE;
  }
  m_depthTex.destroy(m_q->m_ctx);
  for (size_t i = 0; i < MAX_BIND_TEXS; ++i)
    if (m_colorBindView[i]) {
      vk::DestroyImageView(m_q->m_ctx->m_dev, m_colorBindView[i], nullptr);
      m_colorBindView[i] = VK_NULL_HANDLE;
    }
  for (size_t i = 0; i < MAX_BIND_TEXS; ++i)
    m_colorBindTex[i].destroy(m_q->m_ctx);
  for (size_t i = 0; i < MAX_BIND_TEXS; ++i)
    if (m_depthBindView[i]) {
      vk::DestroyImageView(m_q->m_ctx->m_dev, m_depthBindView[i], nullptr);
      m_depthBindView[i] = VK_NULL_HANDLE;
    }
  for (size_t i = 0; i < MAX_BIND_TEXS; ++i)
    m_depthBindTex[i].destroy(m_q->m_ctx);
}

void VulkanTextureR::setClampMode(TextureClampMode mode) {
  MakeSampler(m_q->m_ctx, m_sampler, mode, 1);
  for (size_t i = 0; i < m_colorBindCount; ++i)
    m_colorBindDescInfo[i].sampler = m_sampler;
  for (size_t i = 0; i < m_depthBindCount; ++i)
    m_depthBindDescInfo[i].sampler = m_sampler;
}

VulkanTextureR::VulkanTextureR(const boo::ObjToken<BaseGraphicsData>& parent, VulkanCommandQueue* q, size_t width,
                               size_t height, TextureClampMode clampMode, size_t colorBindCount, size_t depthBindCount)
: GraphicsDataNode<ITextureR>(parent)
, m_q(q)
, m_width(width)
, m_height(height)
, m_samplesColor(q->m_ctx->m_sampleCountColor)
, m_samplesDepth(q->m_ctx->m_sampleCountDepth)
, m_colorBindCount(colorBindCount)
, m_depthBindCount(depthBindCount) {
  if (colorBindCount > MAX_BIND_TEXS)
    Log.report(logvisor::Fatal, FMT_STRING("too many color bindings for render texture"));
  if (depthBindCount > MAX_BIND_TEXS)
    Log.report(logvisor::Fatal, FMT_STRING("too many depth bindings for render texture"));

  if (m_samplesColor == 0)
    m_samplesColor = 1;
  if (m_samplesDepth == 0)
    m_samplesDepth = 1;
  setClampMode(clampMode);
  Setup(q->m_ctx);
}

VulkanTextureR::~VulkanTextureR() {
  vk::DestroyFramebuffer(m_q->m_ctx->m_dev, m_framebuffer, nullptr);
  vk::DestroyImageView(m_q->m_ctx->m_dev, m_colorView, nullptr);
  m_colorTex.destroy(m_q->m_ctx);
  vk::DestroyImageView(m_q->m_ctx->m_dev, m_depthView, nullptr);
  m_depthTex.destroy(m_q->m_ctx);
  for (size_t i = 0; i < MAX_BIND_TEXS; ++i)
    if (m_colorBindView[i])
      vk::DestroyImageView(m_q->m_ctx->m_dev, m_colorBindView[i], nullptr);
  for (size_t i = 0; i < MAX_BIND_TEXS; ++i)
    m_colorBindTex[i].destroy(m_q->m_ctx);
  for (size_t i = 0; i < MAX_BIND_TEXS; ++i)
    if (m_depthBindView[i])
      vk::DestroyImageView(m_q->m_ctx->m_dev, m_depthBindView[i], nullptr);
  for (size_t i = 0; i < MAX_BIND_TEXS; ++i)
    m_depthBindTex[i].destroy(m_q->m_ctx);
}

void VulkanTextureCubeR::doDestroy() {
  if (m_framebuffer[0]) {
    for (int i = 0; i < 6; ++i)
      vk::DestroyFramebuffer(m_q->m_ctx->m_dev, m_framebuffer[i], nullptr);
    m_framebuffer[0] = VK_NULL_HANDLE;
  }
  if (m_colorBindView) {
    vk::DestroyImageView(m_q->m_ctx->m_dev, m_colorBindView, nullptr);
    m_colorBindView = VK_NULL_HANDLE;
  }
  if (m_colorView[0]) {
    for (int i = 0; i < 6; ++i)
      vk::DestroyImageView(m_q->m_ctx->m_dev, m_colorView[i], nullptr);
    m_colorView[0] = VK_NULL_HANDLE;
  }
  m_colorTex.destroy(m_q->m_ctx);
  if (m_depthView[0]) {
    for (int i = 0; i < 6; ++i)
      vk::DestroyImageView(m_q->m_ctx->m_dev, m_depthView[i], nullptr);
    m_depthView[0] = VK_NULL_HANDLE;
  }
  m_depthTex.destroy(m_q->m_ctx);
}

void VulkanTextureCubeR::setClampMode(TextureClampMode mode) {
  MakeSampler(m_q->m_ctx, m_sampler, mode, m_mipCount);
  m_colorBindDescInfo.sampler = m_sampler;
}

VulkanTextureCubeR::VulkanTextureCubeR(const boo::ObjToken<BaseGraphicsData>& parent,
                                       VulkanCommandQueue* q, size_t width, size_t mips)
: GraphicsDataNode<ITextureCubeR>(parent)
, m_q(q)
, m_width(width)
, m_mipCount(mips) {
  Setup(q->m_ctx);
}

VulkanTextureCubeR::~VulkanTextureCubeR() {
  for (int i = 0; i < 6; ++i)
    vk::DestroyFramebuffer(m_q->m_ctx->m_dev, m_framebuffer[i], nullptr);
  vk::DestroyImageView(m_q->m_ctx->m_dev, m_colorBindView, nullptr);
  for (int i = 0; i < 6; ++i)
    vk::DestroyImageView(m_q->m_ctx->m_dev, m_colorView[i], nullptr);
  m_colorTex.destroy(m_q->m_ctx);
  for (int i = 0; i < 6; ++i)
    vk::DestroyImageView(m_q->m_ctx->m_dev, m_depthView[i], nullptr);
  m_depthTex.destroy(m_q->m_ctx);
}

template <class DataCls>
void VulkanGraphicsBufferD<DataCls>::update(int b) {
  int slot = 1 << b;
  if ((slot & m_validSlots) == 0) {
    OPTICK_EVENT();
    memmove(m_bufferPtrs[b], m_cpuBuf.get(), m_cpuSz);
    m_validSlots |= slot;
  }
}

template <class DataCls>
void VulkanGraphicsBufferD<DataCls>::load(const void* data, size_t sz) {
  OPTICK_EVENT();
  size_t bufSz = std::min(sz, m_cpuSz);
  memmove(m_cpuBuf.get(), data, bufSz);
  m_validSlots = 0;
}
template <class DataCls>
void* VulkanGraphicsBufferD<DataCls>::map(size_t sz) {
  if (sz > m_cpuSz)
    return nullptr;
  return m_cpuBuf.get();
}
template <class DataCls>
void VulkanGraphicsBufferD<DataCls>::unmap() {
  m_validSlots = 0;
}

VulkanTextureD::~VulkanTextureD() {
  vk::DestroyImageView(m_q->m_ctx->m_dev, m_gpuView[0], nullptr);
  vk::DestroyImageView(m_q->m_ctx->m_dev, m_gpuView[1], nullptr);
  m_gpuTex[0].destroy(m_q->m_ctx);
  m_gpuTex[1].destroy(m_q->m_ctx);
}

void VulkanTextureD::update(int b) {
  int slot = 1 << b;
  if ((slot & m_validSlots) == 0) {
    m_q->stallDynamicUpload();
    VkCommandBuffer cmdBuf = m_q->m_dynamicCmdBufs[b];

    /* copy staging data */
    memmove(m_cpuBufPtrs[b], m_stagingBuf.get(), m_cpuSz);

    m_gpuTex[b].toLayout(cmdBuf, VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    /* Put the copy command into the command buffer */
    VkBufferImageCopy copyRegion = {};
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.mipLevel = 0;
    copyRegion.imageSubresource.baseArrayLayer = 0;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageExtent.width = m_width;
    copyRegion.imageExtent.height = m_height;
    copyRegion.imageExtent.depth = 1;
    copyRegion.bufferOffset = m_cpuOffsets[b];

    vk::CmdCopyBufferToImage(cmdBuf, m_cpuBuf, m_gpuTex[b].m_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                             &copyRegion);

    /* Set the layout for the texture image from DESTINATION_OPTIMAL to
     * SHADER_READ_ONLY */
    m_gpuTex[b].toLayout(cmdBuf, VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    m_validSlots |= slot;
  }
}
void VulkanTextureD::_setClampMode(TextureClampMode mode) {
  MakeSampler(m_q->m_ctx, m_sampler, mode, 1);
  for (int i = 0; i < 2; ++i)
    m_descInfo[i].sampler = m_sampler;
}
void VulkanTextureD::setClampMode(TextureClampMode mode) {
  if (m_clampMode == mode)
    return;
  m_clampMode = mode;
  _setClampMode(mode);
}
void VulkanTextureD::load(const void* data, size_t sz) {
  size_t bufSz = std::min(sz, m_cpuSz);
  memmove(m_stagingBuf.get(), data, bufSz);
  m_validSlots = 0;
}
void* VulkanTextureD::map(size_t sz) {
  if (sz > m_cpuSz)
    return nullptr;
  return m_stagingBuf.get();
}
void VulkanTextureD::unmap() { m_validSlots = 0; }

VulkanDataFactoryImpl::VulkanDataFactoryImpl(IGraphicsContext* parent, VulkanContext* ctx)
: m_parent(parent), m_ctx(ctx) {}

VulkanDataFactory::Context::Context(VulkanDataFactory& parent __BooTraceArgs)
: m_parent(parent), m_data(new VulkanData(static_cast<VulkanDataFactoryImpl&>(parent) __BooTraceArgsUse)) {}
VulkanDataFactory::Context::~Context() {}

boo::ObjToken<IGraphicsBufferS> VulkanDataFactory::Context::newStaticBuffer(BufferUse use, const void* data,
                                                                            size_t stride, size_t count) {
  OPTICK_EVENT();
  VulkanDataFactoryImpl& factory = static_cast<VulkanDataFactoryImpl&>(m_parent);
  return {new VulkanGraphicsBufferS(m_data, use, factory.m_ctx, data, stride, count)};
}

boo::ObjToken<IGraphicsBufferD> VulkanDataFactory::Context::newDynamicBuffer(BufferUse use, size_t stride,
                                                                             size_t count) {
  OPTICK_EVENT();
  VulkanDataFactoryImpl& factory = static_cast<VulkanDataFactoryImpl&>(m_parent);
  return {new VulkanGraphicsBufferD<BaseGraphicsData>(m_data, use, factory.m_ctx, stride, count)};
}

boo::ObjToken<ITextureS> VulkanDataFactory::Context::newStaticTexture(size_t width, size_t height, size_t mips,
                                                                      TextureFormat fmt, TextureClampMode clampMode,
                                                                      const void* data, size_t sz) {
  OPTICK_EVENT();
  VulkanDataFactoryImpl& factory = static_cast<VulkanDataFactoryImpl&>(m_parent);
  return {new VulkanTextureS(m_data, factory.m_ctx, width, height, mips, fmt, clampMode, data, sz)};
}

boo::ObjToken<ITextureSA> VulkanDataFactory::Context::newStaticArrayTexture(size_t width, size_t height, size_t layers,
                                                                            size_t mips, TextureFormat fmt,
                                                                            TextureClampMode clampMode,
                                                                            const void* data, size_t sz) {
  OPTICK_EVENT();
  VulkanDataFactoryImpl& factory = static_cast<VulkanDataFactoryImpl&>(m_parent);
  return {new VulkanTextureSA(m_data, factory.m_ctx, width, height, layers, mips, fmt, clampMode, data, sz)};
}

boo::ObjToken<ITextureD> VulkanDataFactory::Context::newDynamicTexture(size_t width, size_t height, TextureFormat fmt,
                                                                       TextureClampMode clampMode) {
  OPTICK_EVENT();
  VulkanDataFactoryImpl& factory = static_cast<VulkanDataFactoryImpl&>(m_parent);
  VulkanCommandQueue* q = static_cast<VulkanCommandQueue*>(factory.m_parent->getCommandQueue());
  return {new VulkanTextureD(m_data, q, width, height, fmt, clampMode)};
}

boo::ObjToken<ITextureR> VulkanDataFactory::Context::newRenderTexture(size_t width, size_t height,
                                                                      TextureClampMode clampMode, size_t colorBindCount,
                                                                      size_t depthBindCount) {
  OPTICK_EVENT();
  VulkanDataFactoryImpl& factory = static_cast<VulkanDataFactoryImpl&>(m_parent);
  VulkanCommandQueue* q = static_cast<VulkanCommandQueue*>(factory.m_parent->getCommandQueue());
  return {new VulkanTextureR(m_data, q, width, height, clampMode, colorBindCount, depthBindCount)};
}

ObjToken<ITextureCubeR> VulkanDataFactory::Context::newCubeRenderTexture(size_t width, size_t mips) {
  OPTICK_EVENT();
  VulkanDataFactoryImpl& factory = static_cast<VulkanDataFactoryImpl&>(m_parent);
  VulkanCommandQueue* q = static_cast<VulkanCommandQueue*>(factory.m_parent->getCommandQueue());
  return {new VulkanTextureCubeR(m_data, q, width, mips)};
}

ObjToken<IShaderStage> VulkanDataFactory::Context::newShaderStage(const uint8_t* data, size_t size,
                                                                  PipelineStage stage) {
  OPTICK_EVENT();
  VulkanDataFactoryImpl& factory = static_cast<VulkanDataFactoryImpl&>(m_parent);

  if (stage == PipelineStage::Control || stage == PipelineStage::Evaluation) {
    if (!factory.m_ctx->m_features.tessellationShader)
      Log.report(logvisor::Fatal, FMT_STRING("Device does not support tessellation shaders"));
  }

  return {new VulkanShaderStage(m_data, factory.m_ctx, data, size, stage)};
}

ObjToken<IShaderPipeline> VulkanDataFactory::Context::newShaderPipeline(
    ObjToken<IShaderStage> vertex, ObjToken<IShaderStage> fragment, ObjToken<IShaderStage> geometry,
    ObjToken<IShaderStage> control, ObjToken<IShaderStage> evaluation, const VertexFormatInfo& vtxFmt,
    const AdditionalPipelineInfo& additionalInfo, bool asynchronous) {
  OPTICK_EVENT();
  VulkanDataFactoryImpl& factory = static_cast<VulkanDataFactoryImpl&>(m_parent);

  if (control || evaluation) {
    if (!factory.m_ctx->m_features.tessellationShader)
      Log.report(logvisor::Fatal, FMT_STRING("Device does not support tessellation shaders"));
    if (additionalInfo.patchSize > factory.m_ctx->m_gpuProps.limits.maxTessellationPatchSize)
      Log.report(logvisor::Fatal, FMT_STRING("Device supports {} patch vertices, {} requested"),
                 int(factory.m_ctx->m_gpuProps.limits.maxTessellationPatchSize), int(additionalInfo.patchSize));
  }

  return {new VulkanShaderPipeline(m_data, factory.m_ctx, vertex, fragment, geometry, control, evaluation,
                                   VK_NULL_HANDLE, vtxFmt, additionalInfo, asynchronous)};
}

boo::ObjToken<IShaderDataBinding> VulkanDataFactory::Context::newShaderDataBinding(
    const boo::ObjToken<IShaderPipeline>& pipeline, const boo::ObjToken<IGraphicsBuffer>& vbuf,
    const boo::ObjToken<IGraphicsBuffer>& instVbuf, const boo::ObjToken<IGraphicsBuffer>& ibuf, size_t ubufCount,
    const boo::ObjToken<IGraphicsBuffer>* ubufs, const PipelineStage* /*ubufStages*/, const size_t* ubufOffs,
    const size_t* ubufSizes, size_t texCount, const boo::ObjToken<ITexture>* texs, const int* bindIdxs,
    const bool* bindDepth, size_t baseVert, size_t baseInst) {
  OPTICK_EVENT();
  VulkanDataFactoryImpl& factory = static_cast<VulkanDataFactoryImpl&>(m_parent);
  return {new VulkanShaderDataBinding(m_data, factory, pipeline, vbuf, instVbuf, ibuf, ubufCount, ubufs, ubufOffs,
                                      ubufSizes, texCount, texs, bindIdxs, bindDepth, baseVert, baseInst)};
}

void VulkanDataFactoryImpl::commitTransaction(
    const std::function<bool(IGraphicsDataFactory::Context&)>& trans __BooTraceArgs) {

  Context ctx(*this __BooTraceArgsUse);
  if (!trans(ctx)) {
    return;
  }
  if (m_lazyContext) {
    flushContext(*m_lazyContext);
    m_lazyContext.reset();
  }
  flushContext(ctx);
}

void VulkanDataFactoryImpl::lazyCommitTransaction(
    const std::function<bool(IGraphicsDataFactory::Context&)>& trans __BooTraceArgs) {

  if (!m_lazyContext) {
    m_lazyContext.emplace(*this __BooTraceArgsUse);
  }
  trans(*m_lazyContext);
}

void VulkanDataFactoryImpl::flushContext(Context& ctx) {
  VulkanData* data = ctx.m_data.cast<VulkanData>();

  /* Start asynchronous shader compiles */
  if (data->m_SPs)
    for (IShaderPipeline& p : *data->m_SPs) {
      auto& cp = static_cast<VulkanShaderPipeline&>(p);
      if (cp.m_asynchronous)
        m_pipelineQueue.addPipeline({&p});
    }

  /* size up resources */
  VkDeviceSize constantMemSizes[3] = {};
  VkDeviceSize texMemSize = 0;

  if (data->m_SBufs)
    for (IGraphicsBufferS& buf : *data->m_SBufs) {
      auto& cbuf = static_cast<VulkanGraphicsBufferS&>(buf);
      if (cbuf.m_use == BufferUse::Null)
        continue;
      VkDeviceSize& sz = constantMemSizes[int(cbuf.m_use) - 1];
      sz = cbuf.sizeForGPU(m_ctx, sz);
    }

  if (data->m_DBufs)
    for (IGraphicsBufferD& buf : *data->m_DBufs) {
      auto& cbuf = static_cast<VulkanGraphicsBufferD<BaseGraphicsData>&>(buf);
      if (cbuf.m_use == BufferUse::Null)
        continue;
      VkDeviceSize& sz = constantMemSizes[int(cbuf.m_use) - 1];
      sz = cbuf.sizeForGPU(m_ctx, sz);
    }

  if (data->m_DTexs)
    for (ITextureD& tex : *data->m_DTexs) {
      auto& ctex = static_cast<VulkanTextureD&>(tex);
      texMemSize = ctex.sizeForGPU(m_ctx, texMemSize);
    }

  std::unique_lock<std::mutex> qlk(m_ctx->m_queueLock);

  /* allocate memory and place buffers */
  for (int i = 0; i < 3; ++i) {
    if (constantMemSizes[i]) {
      AllocatedBuffer& poolBuf = data->m_constantBuffers[i];

      VkBufferCreateInfo createInfo = {};
      createInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
      createInfo.size = constantMemSizes[i];
      createInfo.usage = USE_TABLE[i + 1];
      createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      uint8_t* mappedData = reinterpret_cast<uint8_t*>(poolBuf.createCPUtoGPU(m_ctx, &createInfo));

      if (data->m_SBufs)
        for (IGraphicsBufferS& buf : *data->m_SBufs) {
          auto& cbuf = static_cast<VulkanGraphicsBufferS&>(buf);
          if (int(cbuf.m_use) - 1 != i)
            continue;
          cbuf.placeForGPU(poolBuf.m_buffer, mappedData);
        }

      if (data->m_DBufs)
        for (IGraphicsBufferD& buf : *data->m_DBufs) {
          auto& cbuf = static_cast<VulkanGraphicsBufferD<BaseGraphicsData>&>(buf);
          if (int(cbuf.m_use) - 1 != i)
            continue;
          cbuf.placeForGPU(poolBuf.m_buffer, mappedData);
        }
    }
  }

  /* place static textures */
  if (data->m_STexs) {
    OPTICK_EVENT("m_STexs place static textures");
    for (ITextureS& tex : *data->m_STexs)
      static_cast<VulkanTextureS&>(tex).placeForGPU(m_ctx);
  }

  if (data->m_SATexs) {
    OPTICK_EVENT("m_SATexs place static textures");
    for (ITextureSA& tex : *data->m_SATexs)
      static_cast<VulkanTextureSA&>(tex).placeForGPU(m_ctx);
  }

  /* allocate memory and place dynamic textures */
  if (texMemSize) {
    OPTICK_EVENT("place dynamic textures");
    AllocatedBuffer& poolBuf = data->m_texStagingBuffer;

    VkBufferCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    createInfo.size = texMemSize;
    createInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    uint8_t* mappedData = reinterpret_cast<uint8_t*>(poolBuf.createCPUtoGPU(m_ctx, &createInfo));

    if (data->m_DTexs)
      for (ITextureD& tex : *data->m_DTexs)
        static_cast<VulkanTextureD&>(tex).placeForGPU(m_ctx, poolBuf.m_buffer, mappedData);
  }

  /* initialize bind texture layout */
  if (data->m_RTexs)
    for (ITextureR& tex : *data->m_RTexs)
      static_cast<VulkanTextureR&>(tex).initializeBindLayouts(m_ctx);

  {
    /* Execute static uploads */
    OPTICK_EVENT("vk::EndCommandBuffer");
    ThrowIfFailed(vk::EndCommandBuffer(m_ctx->m_loadCmdBuf));
  }

  {
    /* Take exclusive lock here and submit queue */
    OPTICK_EVENT("vk::QueueWaitIdle");
    ThrowIfFailed(vk::QueueWaitIdle(m_ctx->m_queue));
  }

  {
    OPTICK_EVENT("vk::QueueSubmit");
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &m_ctx->m_loadCmdBuf;
    ThrowIfFailed(vk::QueueSubmit(m_ctx->m_queue, 1, &submitInfo, VK_NULL_HANDLE));
  }

  /* Commit data bindings (create descriptor sets) */
  if (data->m_SBinds) {
    OPTICK_EVENT("Commit data bindings");
    for (IShaderDataBinding& bind : *data->m_SBinds)
      static_cast<VulkanShaderDataBinding&>(bind).commit(m_ctx);
  }

  {
    /* Wait for uploads to complete */
    OPTICK_EVENT("vk::QueueWaitIdle");
    ThrowIfFailed(vk::QueueWaitIdle(m_ctx->m_queue));
    qlk.unlock();
  }

  {
    /* Reset command buffer */
    OPTICK_EVENT("vk::ResetCommandBuffer");
    ThrowIfFailed(vk::ResetCommandBuffer(m_ctx->m_loadCmdBuf, 0));
  }

  {
    OPTICK_EVENT("vk::BeginCommandBuffer");
    VkCommandBufferBeginInfo cmdBufBeginInfo = {};
    cmdBufBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cmdBufBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    ThrowIfFailed(vk::BeginCommandBuffer(m_ctx->m_loadCmdBuf, &cmdBufBeginInfo));
  }

  /* Delete upload objects */
  if (data->m_STexs) {
    OPTICK_EVENT("m_STexs deleteUploadObjects");
    for (ITextureS& tex : *data->m_STexs)
      static_cast<VulkanTextureS&>(tex).deleteUploadObjects();
  }

  if (data->m_SATexs) {
    OPTICK_EVENT("m_SATexs deleteUploadObjects");
    for (ITextureSA& tex : *data->m_SATexs)
      static_cast<VulkanTextureSA&>(tex).deleteUploadObjects();
  }
}

boo::ObjToken<IGraphicsBufferD> VulkanDataFactoryImpl::newPoolBuffer(BufferUse use, size_t stride,
                                                                     size_t count __BooTraceArgs) {
  boo::ObjToken<BaseGraphicsPool> pool(new VulkanPool(*this __BooTraceArgsUse));
  VulkanPool* cpool = pool.cast<VulkanPool>();
  VulkanGraphicsBufferD<BaseGraphicsPool>* retval =
      new VulkanGraphicsBufferD<BaseGraphicsPool>(pool, use, m_ctx, stride, count);

  VkDeviceSize size = retval->sizeForGPU(m_ctx, 0);

  /* allocate memory */
  if (size) {
    AllocatedBuffer& poolBuf = cpool->m_constantBuffer;
    VkBufferCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    createInfo.size = size;
    createInfo.usage = USE_TABLE[int(use)];
    createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    uint8_t* mappedData = reinterpret_cast<uint8_t*>(poolBuf.createCPUtoGPU(m_ctx, &createInfo));
    retval->placeForGPU(poolBuf.m_buffer, mappedData);
  }

  return {retval};
}

void VulkanCommandQueue::_commitImageLayouts() {
  VulkanDataFactoryImpl* gfxF = static_cast<VulkanDataFactoryImpl*>(m_parent->getDataFactory());
  if (gfxF->m_dataHead) {
    for (BaseGraphicsData& d : *gfxF->m_dataHead) {
      if (d.m_RTexs)
        for (ITextureR& t : *d.m_RTexs)
          static_cast<VulkanTextureR&>(t).commitLayouts();
      if (d.m_CubeRTexs)
        for (ITextureCubeR& t : *d.m_CubeRTexs)
          static_cast<VulkanTextureCubeR&>(t).commitLayouts();
    }
  }
}

void VulkanCommandQueue::_rollbackImageLayouts() {
  VulkanDataFactoryImpl* gfxF = static_cast<VulkanDataFactoryImpl*>(m_parent->getDataFactory());
  if (gfxF->m_dataHead) {
    for (BaseGraphicsData& d : *gfxF->m_dataHead) {
      if (d.m_RTexs)
        for (ITextureR& t : *d.m_RTexs)
          static_cast<VulkanTextureR&>(t).rollbackLayouts();
      if (d.m_CubeRTexs)
        for (ITextureCubeR& t : *d.m_CubeRTexs)
          static_cast<VulkanTextureCubeR&>(t).rollbackLayouts();
    }
  }
}

void VulkanCommandQueue::execute() {
  if (!m_running)
    return;

  SCOPED_GRAPHICS_DEBUG_GROUP(this, "VulkanCommandQueue::execute", {1.f, 0.f, 0.f, 1.f});

  /* Stage dynamic uploads */
  {
    OPTICK_EVENT("Stage dynamic uploads");
    VulkanDataFactoryImpl* gfxF = static_cast<VulkanDataFactoryImpl*>(m_parent->getDataFactory());
    std::unique_lock<std::recursive_mutex> datalk(gfxF->m_dataMutex);
    if (gfxF->m_dataHead) {
      for (BaseGraphicsData& d : *gfxF->m_dataHead) {
        if (d.m_DBufs)
          for (IGraphicsBufferD& b : *d.m_DBufs)
            static_cast<VulkanGraphicsBufferD<BaseGraphicsData>&>(b).update(m_fillBuf);
        if (d.m_DTexs)
          for (ITextureD& t : *d.m_DTexs)
            static_cast<VulkanTextureD&>(t).update(m_fillBuf);
      }
    }
    if (gfxF->m_poolHead) {
      for (BaseGraphicsPool& p : *gfxF->m_poolHead) {
        if (p.m_DBufs)
          for (IGraphicsBufferD& b : *p.m_DBufs)
            static_cast<VulkanGraphicsBufferD<BaseGraphicsData>&>(b).update(m_fillBuf);
      }
    }
    datalk.unlock();
  }

  /* Perform dynamic uploads */
  {
    OPTICK_EVENT("Perform dynamic uploads");
    std::unique_lock<std::mutex> lk(m_ctx->m_queueLock);
    if (!m_dynamicNeedsReset) {
      vk::EndCommandBuffer(m_dynamicCmdBufs[m_fillBuf]);

      vk::WaitForFences(m_ctx->m_dev, 1, &m_dynamicBufFence, VK_FALSE, -1);
      vk::ResetFences(m_ctx->m_dev, 1, &m_dynamicBufFence);

      VkSubmitInfo submitInfo = {};
      submitInfo.pNext = nullptr;
      submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      submitInfo.waitSemaphoreCount = 0;
      submitInfo.pWaitSemaphores = nullptr;
      submitInfo.pWaitDstStageMask = nullptr;
      submitInfo.commandBufferCount = 1;
      submitInfo.pCommandBuffers = &m_dynamicCmdBufs[m_fillBuf];
      submitInfo.signalSemaphoreCount = 0;
      submitInfo.pSignalSemaphores = nullptr;
      ThrowIfFailed(vk::QueueSubmit(m_ctx->m_queue, 1, &submitInfo, m_dynamicBufFence));
    }
  }

  vk::CmdEndRenderPass(m_cmdBufs[m_fillBuf]);
  m_boundTarget.reset();

  {
    OPTICK_EVENT("Check on fence");
    /* Check on fence */
    if (m_submitted && vk::GetFenceStatus(m_ctx->m_dev, m_drawCompleteFence) == VK_NOT_READY) {
      /* Abandon this list (renderer too slow) */
      resetCommandBuffer();
      _rollbackImageLayouts();
      m_dynamicNeedsReset = true;
      m_resolveDispSource = nullptr;

      /* Clear dead data */
      m_drawResTokens[m_fillBuf].clear();
      return;
    }
    m_submitted = false;
  }
  {
    OPTICK_EVENT("vk::ResetFences");
    vk::ResetFences(m_ctx->m_dev, 1, &m_drawCompleteFence);
  }

  /* Perform texture and swap-chain resizes */
  if (m_ctx->_resizeSwapChains() || m_texResizes.size() || m_cubeTexResizes.size()) {
    for (const auto& resize : m_texResizes)
      resize.first->resize(m_ctx, resize.second.first, resize.second.second);
    m_texResizes.clear();
    for (const auto& resize : m_cubeTexResizes)
      resize.first->resize(m_ctx, resize.second.first, resize.second.second);
    m_cubeTexResizes.clear();
    resetCommandBuffer();
    _rollbackImageLayouts();
    m_dynamicNeedsReset = true;
    m_resolveDispSource = nullptr;
    return;
  }

  /* Clear dead data */
  m_drawResTokens[m_drawBuf].clear();

  m_drawBuf = m_fillBuf;
  m_fillBuf ^= 1;

  /* Queue the command buffer for execution */
  VkPipelineStageFlags pipeStageFlags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  VkSubmitInfo submitInfo = {};
  submitInfo.pNext = nullptr;
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.waitSemaphoreCount = 0;
  submitInfo.pWaitSemaphores = nullptr;
  submitInfo.pWaitDstStageMask = &pipeStageFlags;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &m_cmdBufs[m_drawBuf];
  submitInfo.signalSemaphoreCount = 0;
  submitInfo.pSignalSemaphores = nullptr;
  if (_resolveDisplay()) {
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &m_swapChainReadySem;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &m_drawCompleteSem;
  }

  {
    OPTICK_EVENT("vk::QueueSubmit");
    ThrowIfFailed(vk::EndCommandBuffer(m_cmdBufs[m_drawBuf]));
    ThrowIfFailed(vk::QueueSubmit(m_ctx->m_queue, 1, &submitInfo, m_drawCompleteFence));
  }
  m_submitted = true;

  if (submitInfo.signalSemaphoreCount) {
    VulkanContext::Window::SwapChain& thisSc = m_windowCtx->m_swapChains[m_windowCtx->m_activeSwapChain];

    VkPresentInfoKHR present;
    present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.pNext = nullptr;
    present.swapchainCount = 1;
    present.pSwapchains = &thisSc.m_swapChain;
    present.pImageIndices = &thisSc.m_backBuf;
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &m_drawCompleteSem;
    present.pResults = nullptr;

    OPTICK_GPU_FLIP(&thisSc.m_swapChain);

    VkResult res = vk::QueuePresentKHR(m_ctx->m_queue, &present);
    if (res == VK_ERROR_OUT_OF_DATE_KHR) {
      // ignore, resize deferred
    } else {
      ThrowIfFailed(res);
    }
  }

  resetCommandBuffer();
  resetDynamicCommandBuffer();
  _commitImageLayouts();
}

std::unique_ptr<IGraphicsCommandQueue> _NewVulkanCommandQueue(VulkanContext* ctx, VulkanContext::Window* windowCtx,
                                                              IGraphicsContext* parent) {
  return std::make_unique<VulkanCommandQueue>(ctx, windowCtx, parent);
}

std::unique_ptr<IGraphicsDataFactory> _NewVulkanDataFactory(IGraphicsContext* parent, VulkanContext* ctx) {
  return std::make_unique<VulkanDataFactoryImpl>(parent, ctx);
}

static const EShLanguage ShaderTypes[] = {EShLangVertex,   EShLangVertex,      EShLangFragment,
                                          EShLangGeometry, EShLangTessControl, EShLangTessEvaluation};

std::vector<uint8_t> VulkanDataFactory::CompileGLSL(const char* source, PipelineStage stage) {
  EShLanguage lang = ShaderTypes[int(stage)];
  const EShMessages messages = EShMessages(EShMsgSpvRules | EShMsgVulkanRules);
  glslang::TShader shader(lang);
  shader.setStrings(&source, 1);
  if (!shader.parse(&glslang::DefaultTBuiltInResource, 110, false, messages)) {
    fmt::print(FMT_STRING("{}\n"), source);
    Log.report(logvisor::Fatal, FMT_STRING("unable to compile shader\n{}"), shader.getInfoLog());
  }

  glslang::TProgram prog;
  prog.addShader(&shader);
  if (!prog.link(messages)) {
    Log.report(logvisor::Fatal, FMT_STRING("unable to link shader program\n{}"), prog.getInfoLog());
  }

  std::vector<unsigned int> out;
  glslang::GlslangToSpv(*prog.getIntermediate(lang), out);
  // spv::Disassemble(std::cerr, out);

  std::vector<uint8_t> ret(out.size() * 4);
  memcpy(ret.data(), out.data(), ret.size());
  return ret;
}

} // namespace boo
