#include "boo/IWindow.hpp"
#include "boo/IGraphicsContext.hpp"
#include "boo/IApplication.hpp"
#include "boo/graphicsdev/GL.hpp"

#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <GL/glx.h>

#define XK_MISCELLANY
#define XK_XKB_KEYS
#define XK_LATIN1
#include <X11/keysymdef.h>
#include <xkbcommon/xkbcommon.h>
#include <X11/extensions/XInput2.h>
#include <X11/Xatom.h>
#include <LogVisor/LogVisor.hpp>

#include "XlibCommon.hpp"

#define REF_DPMM 3.78138
#define FS_ATOM "_NET_WM_STATE_FULLSCREEN"

#define MWM_HINTS_FUNCTIONS   (1L << 0)
#define MWM_HINTS_DECORATIONS (1L << 1)

#define MWM_DECOR_BORDER      (1L<<1)
#define MWM_DECOR_RESIZEH     (1L<<2)
#define MWM_DECOR_TITLE       (1L<<3)
#define MWM_DECOR_MENU        (1L<<4)
#define MWM_DECOR_MINIMIZE    (1L<<5)
#define MWM_DECOR_MAXIMIZE    (1L<<6)

#define MWM_FUNC_RESIZE       (1L<<1)
#define MWM_FUNC_MOVE         (1L<<2)
#define MWM_FUNC_MINIMIZE     (1L<<3)
#define MWM_FUNC_MAXIMIZE     (1L<<4)
#define MWM_FUNC_CLOSE        (1L<<5)

#undef None

typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);
static glXCreateContextAttribsARBProc glXCreateContextAttribsARB = 0;
typedef int (*glXWaitVideoSyncSGIProc)(int divisor, int remainder, unsigned int* count);
static glXWaitVideoSyncSGIProc glXWaitVideoSyncSGI = 0;
static const int ContextAttribs[] =
{
    GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
    GLX_CONTEXT_MINOR_VERSION_ARB, 3,
    GLX_CONTEXT_FLAGS_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
    //GLX_CONTEXT_FLAGS_ARB, GLX_CONTEXT_DEBUG_BIT_ARB,
    //GLX_CONTEXT_FLAGS_ARB, GLX_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB,
    0
};

namespace boo
{
static LogVisor::LogModule Log("boo::WindowXCB");
IGraphicsCommandQueue* _NewGLCommandQueue(IGraphicsContext* parent);
void _XlibUpdateLastGlxCtx(GLXContext lastGlxCtx);
void GLXExtensionCheck();
void GLXEnableVSync(Display* disp, GLXWindow drawable);

extern int XINPUT_OPCODE;

static uint32_t translateKeysym(KeySym sym, ESpecialKey& specialSym, EModifierKey& modifierSym)
{
    specialSym = ESpecialKey::None;
    modifierSym = EModifierKey::None;
    if (sym >= XK_F1 && sym <= XK_F12)
        specialSym = ESpecialKey(int(ESpecialKey::F1) + sym - XK_F1);
    else if (sym == XK_Escape)
        specialSym = ESpecialKey::Esc;
    else if (sym == XK_Return)
        specialSym = ESpecialKey::Enter;
    else if (sym == XK_BackSpace)
        specialSym = ESpecialKey::Backspace;
    else if (sym == XK_Insert)
        specialSym = ESpecialKey::Insert;
    else if (sym == XK_Delete)
        specialSym = ESpecialKey::Delete;
    else if (sym == XK_Home)
        specialSym = ESpecialKey::Home;
    else if (sym == XK_End)
        specialSym = ESpecialKey::End;
    else if (sym == XK_Page_Up)
        specialSym = ESpecialKey::PgUp;
    else if (sym == XK_Page_Down)
        specialSym = ESpecialKey::PgDown;
    else if (sym == XK_Left)
        specialSym = ESpecialKey::Left;
    else if (sym == XK_Right)
        specialSym = ESpecialKey::Right;
    else if (sym == XK_Up)
        specialSym = ESpecialKey::Up;
    else if (sym == XK_Down)
        specialSym = ESpecialKey::Down;
    else if (sym == XK_Shift_L || sym == XK_Shift_R)
        modifierSym = EModifierKey::Shift;
    else if (sym == XK_Control_L || sym == XK_Control_R)
        modifierSym = EModifierKey::Ctrl;
    else if (sym == XK_Alt_L || sym == XK_Alt_R)
        modifierSym = EModifierKey::Alt;
    else
        return xkb_keysym_to_utf32(sym);
    return 0;
}

static EModifierKey translateModifiers(unsigned state)
{
    EModifierKey retval = EModifierKey::None;
    if (state & ShiftMask)
        retval |= EModifierKey::Shift;
    if (state & ControlMask)
        retval |= EModifierKey::Ctrl;
    if (state & Mod1Mask)
        retval |= EModifierKey::Alt;
    return retval;
}

static EMouseButton translateButton(unsigned detail)
{
    switch (detail)
    {
    case 1:
        return EMouseButton::Primary;
    case 3:
        return EMouseButton::Secondary;
    case 2:
        return EMouseButton::Middle;
    case 8:
        return EMouseButton::Aux1;
    case 9:
        return EMouseButton::Aux2;
    default: break;
    }
    return EMouseButton::None;
}

struct XCBAtoms
{
    Atom m_wmProtocols = 0;
    Atom m_wmDeleteWindow = 0;
    Atom m_netwmState = 0;
    Atom m_netwmStateFullscreen = 0;
    Atom m_netwmStateAdd = 0;
    Atom m_netwmStateRemove = 0;
    Atom m_motifWmHints = 0;
    XCBAtoms(Display* disp)
    {
        m_wmProtocols = XInternAtom(disp, "WM_PROTOCOLS", True);
        m_wmDeleteWindow = XInternAtom(disp, "WM_DELETE_WINDOW", True);
        m_netwmState = XInternAtom(disp, "_NET_WM_STATE", False);
        m_netwmStateFullscreen = XInternAtom(disp, "_NET_WM_STATE_FULLSCREEN", False);
        m_netwmStateAdd = XInternAtom(disp, "_NET_WM_STATE_ADD", False);
        m_netwmStateRemove = XInternAtom(disp, "_NET_WM_STATE_REMOVE", False);
        m_motifWmHints = XInternAtom(disp, "_MOTIF_WM_HINTS", True);
    }
};
static XCBAtoms* S_ATOMS = NULL;

static void genFrameDefault(Screen* screen, int& xOut, int& yOut, int& wOut, int& hOut)
{
    float width = screen->width * 2.0 / 3.0;
    float height = screen->height * 2.0 / 3.0;
    xOut = (screen->width - width) / 2.0;
    yOut = (screen->height - height) / 2.0;
    wOut = width;
    hOut = height;
}
    
struct GraphicsContextGLX : IGraphicsContext
{
    EGraphicsAPI m_api;
    EPixelFormat m_pf;
    IWindow* m_parentWindow;
    Display* m_xDisp = nullptr;
    GLXContext m_lastCtx = 0;

    GLXFBConfig m_fbconfig = 0;
    int m_visualid = 0;
    GLXWindow m_glxWindow = 0;
    GLXContext m_glxCtx = 0;

    IGraphicsCommandQueue* m_commandQueue = nullptr;
    IGraphicsDataFactory* m_dataFactory = nullptr;
    GLXContext m_mainCtx = 0;
    GLXContext m_loadCtx = 0;

    std::thread m_vsyncThread;
    bool m_vsyncRunning;

public:
    IWindowCallback* m_callback;

    GraphicsContextGLX(EGraphicsAPI api, IWindow* parentWindow,
                       Display* display, int defaultScreen,
                       GLXContext lastCtx, uint32_t& visualIdOut)
    : m_api(api),
      m_pf(EPixelFormat::RGBA8_Z24),
      m_parentWindow(parentWindow),
      m_xDisp(display),
      m_lastCtx(lastCtx)
    {
        m_dataFactory = new class GLDataFactory(this);

        /* Query framebuffer configurations */
        GLXFBConfig* fbConfigs = nullptr;
        int numFBConfigs = 0;
        fbConfigs = glXGetFBConfigs(display, defaultScreen, &numFBConfigs);
        if (!fbConfigs || numFBConfigs == 0)
        {
            Log.report(LogVisor::FatalError, "glXGetFBConfigs failed");
            return;
        }

        for (int i=0 ; i<numFBConfigs ; ++i)
        {
            GLXFBConfig config = fbConfigs[i];
            int visualId, depthSize, colorSize, doubleBuffer;
            glXGetFBConfigAttrib(display, config, GLX_VISUAL_ID, &visualId);
            glXGetFBConfigAttrib(display, config, GLX_DEPTH_SIZE, &depthSize);
            glXGetFBConfigAttrib(display, config, GLX_BUFFER_SIZE, &colorSize);
            glXGetFBConfigAttrib(display, config, GLX_DOUBLEBUFFER, &doubleBuffer);

            /* Double-buffer only */
            if (!doubleBuffer)
                continue;

            if (m_pf == EPixelFormat::RGBA8 && colorSize >= 32)
            {
                m_fbconfig = config;
                m_visualid = visualId;
                break;
            }
            else if (m_pf == EPixelFormat::RGBA8_Z24 && colorSize >= 32 && depthSize >= 24)
            {
                m_fbconfig = config;
                m_visualid = visualId;
                break;
            }
            else if (m_pf == EPixelFormat::RGBAF32 && colorSize >= 128)
            {
                m_fbconfig = config;
                m_visualid = visualId;
                break;
            }
            else if (m_pf == EPixelFormat::RGBAF32_Z24 && colorSize >= 128 && depthSize >= 24)
            {
                m_fbconfig = config;
                m_visualid = visualId;
                break;
            }
        }
        XFree(fbConfigs);

        if (!m_fbconfig)
        {
            Log.report(LogVisor::FatalError, "unable to find suitable pixel format");
            return;
        }

        visualIdOut = m_visualid;
    }

    void destroy()
    {
        if (m_glxCtx)
        {
            glXDestroyContext(m_xDisp, m_glxCtx);
            m_glxCtx = nullptr;
        }
        if (m_glxWindow)
        {
            glXDestroyWindow(m_xDisp, m_glxWindow);
            m_glxWindow = 0;
        }
        if (m_loadCtx)
        {
            glXDestroyContext(m_xDisp, m_loadCtx);
            m_loadCtx = nullptr;
        }
        if (m_vsyncRunning)
        {
            m_vsyncRunning = false;
            m_vsyncThread.join();
        }
    }

    ~GraphicsContextGLX() {destroy();}

    void _setCallback(IWindowCallback* cb)
    {
        m_callback = cb;
    }

    EGraphicsAPI getAPI() const
    {
        return m_api;
    }

    EPixelFormat getPixelFormat() const
    {
        return m_pf;
    }

    void setPixelFormat(EPixelFormat pf)
    {
        if (pf > EPixelFormat::RGBAF32_Z24)
            return;
        m_pf = pf;
    }

    std::mutex m_vsyncmt;
    std::condition_variable m_vsynccv;

    void initializeContext()
    {
        if (!glXCreateContextAttribsARB)
        {
            glXCreateContextAttribsARB = (glXCreateContextAttribsARBProc)
                    glXGetProcAddressARB((const GLubyte*)"glXCreateContextAttribsARB");
            if (!glXCreateContextAttribsARB)
                Log.report(LogVisor::FatalError, "unable to resolve glXCreateContextAttribsARB");
        }
        if (!glXWaitVideoSyncSGI)
        {
            glXWaitVideoSyncSGI = (glXWaitVideoSyncSGIProc)
                    glXGetProcAddressARB((const GLubyte*)"glXWaitVideoSyncSGI");
            if (!glXWaitVideoSyncSGI)
                Log.report(LogVisor::FatalError, "unable to resolve glXWaitVideoSyncSGI");
        }

        m_glxCtx = glXCreateContextAttribsARB(m_xDisp, m_fbconfig, m_lastCtx, True, ContextAttribs);
        if (!m_glxCtx)
            Log.report(LogVisor::FatalError, "unable to make new GLX context");
        m_glxWindow = glXCreateWindow(m_xDisp, m_fbconfig, m_parentWindow->getPlatformHandle(), nullptr);
        if (!m_glxWindow)
            Log.report(LogVisor::FatalError, "unable to make new GLX window");
        _XlibUpdateLastGlxCtx(m_glxCtx);

        /* Spawn vsync thread */
        m_vsyncRunning = true;
        std::mutex initmt;
        std::condition_variable initcv;
        std::unique_lock<std::mutex> outerLk(initmt);
        m_vsyncThread = std::thread([&]()
        {
            Display* vsyncDisp;
            GLXContext vsyncCtx;
            {
                std::unique_lock<std::mutex> innerLk(initmt);

                vsyncDisp = XOpenDisplay(0);
                if (!vsyncDisp)
                    Log.report(LogVisor::FatalError, "unable to open new vsync display");
                XLockDisplay(vsyncDisp);

                static int attributeList[] = { GLX_RGBA, GLX_DOUBLEBUFFER, GLX_RED_SIZE, 1, GLX_GREEN_SIZE, 1, GLX_BLUE_SIZE, 1, 0 };
                XVisualInfo *vi = glXChooseVisual(vsyncDisp, DefaultScreen(vsyncDisp),attributeList);

                vsyncCtx = glXCreateContext(vsyncDisp, vi, nullptr, True);
                if (!vsyncCtx)
                    Log.report(LogVisor::FatalError, "unable to make new vsync GLX context");

                if (!glXMakeCurrent(vsyncDisp, DefaultRootWindow(vsyncDisp), vsyncCtx))
                    Log.report(LogVisor::FatalError, "unable to make vsync context current");
            }
            initcv.notify_one();

            while (m_vsyncRunning)
            {
                unsigned int sync;
                int err = glXWaitVideoSyncSGI(1, 0, &sync);
                if (err)
                    Log.report(LogVisor::FatalError, "wait err");
                m_vsynccv.notify_one();
            }

            glXMakeCurrent(vsyncDisp, 0, nullptr);
            glXDestroyContext(vsyncDisp, vsyncCtx);
            XUnlockDisplay(vsyncDisp);
            XCloseDisplay(vsyncDisp);
        });
        initcv.wait(outerLk);

        XUnlockDisplay(m_xDisp);
        m_commandQueue = _NewGLCommandQueue(this);
        XLockDisplay(m_xDisp);
    }

    void makeCurrent()
    {
        XLockDisplay(m_xDisp);
        if (!glXMakeContextCurrent(m_xDisp, m_glxWindow, m_glxWindow, m_glxCtx))
            Log.report(LogVisor::FatalError, "unable to make GLX context current");
        XUnlockDisplay(m_xDisp);
    }

    void postInit()
    {
        GLXExtensionCheck();
        XLockDisplay(m_xDisp);
        GLXEnableVSync(m_xDisp, m_glxWindow);
        XUnlockDisplay(m_xDisp);
    }

    IGraphicsCommandQueue* getCommandQueue()
    {
        return m_commandQueue;
    }

    IGraphicsDataFactory* getDataFactory()
    {
        return m_dataFactory;
    }

    IGraphicsDataFactory* getMainContextDataFactory()
    {
        XLockDisplay(m_xDisp);
        if (!m_mainCtx)
        {
            m_mainCtx = glXCreateContextAttribsARB(m_xDisp, m_fbconfig, m_glxCtx, True, ContextAttribs);
            if (!m_mainCtx)
                Log.report(LogVisor::FatalError, "unable to make main GLX context");
        }
        if (!glXMakeContextCurrent(m_xDisp, m_glxWindow, m_glxWindow, m_mainCtx))
            Log.report(LogVisor::FatalError, "unable to make main GLX context current");
        XUnlockDisplay(m_xDisp);
        return getDataFactory();
    }

    IGraphicsDataFactory* getLoadContextDataFactory()
    {
        XLockDisplay(m_xDisp);
        if (!m_loadCtx)
        {
            m_loadCtx = glXCreateContextAttribsARB(m_xDisp, m_fbconfig, m_glxCtx, True, ContextAttribs);
            if (!m_loadCtx)
                Log.report(LogVisor::FatalError, "unable to make load GLX context");
        }
        if (!glXMakeContextCurrent(m_xDisp, m_glxWindow, m_glxWindow, m_loadCtx))
            Log.report(LogVisor::FatalError, "unable to make load GLX context current");
        XUnlockDisplay(m_xDisp);
        return getDataFactory();
    }

    void present()
    {
        XLockDisplay(m_xDisp);
        glXSwapBuffers(m_xDisp, m_glxWindow);
        XUnlockDisplay(m_xDisp);
    }

};

class WindowXlib : public IWindow
{
    Display* m_xDisp;
    IWindowCallback* m_callback;
    Colormap m_colormapId;
    Window m_windowId;
    GraphicsContextGLX m_gfxCtx;
    uint32_t m_visualId;

    /* Last known input device id (0xffff if not yet set) */
    int m_lastInputID = 0xffff;
    ETouchType m_touchType = ETouchType::None;

    /* Scroll valuators */
    int m_hScrollValuator = -1;
    int m_vScrollValuator = -1;
    double m_hScrollLast = 0.0;
    double m_vScrollLast = 0.0;

    /* Cached window rectangle (to avoid repeated X queries) */
    int m_wx, m_wy, m_ww, m_wh;
    float m_pixelFactor;
    bool m_inFs = false;

    /* Cached window style */
    EWindowStyle m_styleFlags;

    /* Current cursor enum */
    EMouseCursor m_cursor = EMouseCursor::None;
    bool m_cursorWait = false;
    static Cursor GetXCursor(EMouseCursor cur)
    {
        switch (cur)
        {
        case EMouseCursor::Pointer:
            return X_CURSORS.m_pointer;
        case EMouseCursor::HorizontalArrow:
            return X_CURSORS.m_hArrow;
        case EMouseCursor::VerticalArrow:
            return X_CURSORS.m_vArrow;
        case EMouseCursor::IBeam:
            return X_CURSORS.m_ibeam;
        default: break;
        }
        return X_CURSORS.m_pointer;
    }

public:
    
    WindowXlib(const std::string& title,
              Display* display, int defaultScreen,
              GLXContext lastCtx)
    : m_xDisp(display), m_callback(nullptr),
      m_gfxCtx(IGraphicsContext::EGraphicsAPI::OpenGL3_3,
               this, display, defaultScreen,
               lastCtx, m_visualId)
    {
        if (!S_ATOMS)
            S_ATOMS = new XCBAtoms(display);

        /* Default screen */
        Screen* screen = ScreenOfDisplay(display, defaultScreen);
        m_pixelFactor = screen->width / (float)screen->mwidth / REF_DPMM;

        XVisualInfo visTemplate;
        visTemplate.screen = defaultScreen;
        int numVisuals;
        XVisualInfo* visualList = XGetVisualInfo(display, VisualScreenMask, &visTemplate, &numVisuals);
        Visual* selectedVisual = nullptr;
        for (int i=0 ; i<numVisuals ; ++i)
        {
            if (visualList[i].visualid == m_visualId)
            {
                selectedVisual = visualList[i].visual;
                break;
            }
        }
        XFree(visualList);

        /* Create colormap */
        m_colormapId = XCreateColormap(m_xDisp, screen->root, selectedVisual, AllocNone);

        /* Create window */
        int x, y, w, h;
        genFrameDefault(screen, x, y, w, h);
        XSetWindowAttributes swa;
        swa.colormap = m_colormapId;
        swa.border_pixmap = 0;
        swa.event_mask = FocusChangeMask | KeyPressMask | KeyReleaseMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask | ExposureMask | StructureNotifyMask | LeaveWindowMask | EnterWindowMask;

        m_windowId = XCreateWindow(display, screen->root, x, y, w, h, 10,
                                   CopyFromParent, CopyFromParent, selectedVisual,
                                   CWBorderPixel | CWEventMask | CWColormap, &swa);


        /* The XInput 2.1 extension enables per-pixel smooth scrolling trackpads */
        XIEventMask mask = {XIAllMasterDevices, XIMaskLen(XI_LASTEVENT)};
        mask.mask = (unsigned char*)malloc(mask.mask_len);
        memset(mask.mask, 0, mask.mask_len);
        /* XISetMask(mask.mask, XI_Motion); Can't do this without losing mouse move events :( */
        XISetMask(mask.mask, XI_TouchBegin);
        XISetMask(mask.mask, XI_TouchUpdate);
        XISetMask(mask.mask, XI_TouchEnd);
        XISelectEvents(m_xDisp, m_windowId, &mask, 1);
        free(mask.mask);


        /* Register netwm extension atom for window closing */
        XSetWMProtocols(m_xDisp, m_windowId, &S_ATOMS->m_wmDeleteWindow, 1);

        /* Set the title of the window */
        const unsigned char* c_title = (unsigned char*)title.c_str();
        XChangeProperty(m_xDisp, m_windowId, XA_WM_NAME, XA_STRING, 8, PropModeReplace, c_title, title.length());

        /* Set the title of the window icon */
        XChangeProperty(m_xDisp, m_windowId, XA_WM_ICON_NAME, XA_STRING, 8, PropModeReplace, c_title, title.length());

        /* Initialize context */
        XMapWindow(m_xDisp, m_windowId);
        setStyle(EWindowStyle::Default);
        setCursor(EMouseCursor::Pointer);
        XFlush(m_xDisp);

        m_gfxCtx.initializeContext();
    }
    
    ~WindowXlib()
    {
        XLockDisplay(m_xDisp);
        m_gfxCtx.destroy();
        XUnmapWindow(m_xDisp, m_windowId);
        XDestroyWindow(m_xDisp, m_windowId);
        XFreeColormap(m_xDisp, m_colormapId);
        XUnlockDisplay(m_xDisp);
        APP->_deletedWindow(this);
    }
    
    void setCallback(IWindowCallback* cb)
    {
        m_callback = cb;
    }
    
    void showWindow()
    {
        XLockDisplay(m_xDisp);
        XMapWindow(m_xDisp, m_windowId);
        XUnlockDisplay(m_xDisp);
    }
    
    void hideWindow()
    {
        XLockDisplay(m_xDisp);
        XUnmapWindow(m_xDisp, m_windowId);
        XUnlockDisplay(m_xDisp);
    }
    
    std::string getTitle()
    {
        unsigned long nitems;
        Atom     actualType;
        int      actualFormat;
        unsigned long     bytes;
        unsigned char* string = nullptr;
        XLockDisplay(m_xDisp);
        int ret = XGetWindowProperty(m_xDisp, m_windowId, XA_WM_NAME, 0, ~0l, False,
                                     XA_STRING, &actualType, &actualFormat, &nitems, &bytes, &string);
        XUnlockDisplay(m_xDisp);
        if (ret == Success)
        {
            std::string retval((const char*)string);
            XFree(string);
            return retval;
        }
        return std::string();
    }
    
    void setTitle(const std::string& title)
    {
        const unsigned char* c_title = (unsigned char*)title.c_str();
        XLockDisplay(m_xDisp);
        XChangeProperty(m_xDisp, m_windowId, XA_WM_NAME, XA_STRING, 8,
                        PropModeReplace, c_title, title.length());
        XUnlockDisplay(m_xDisp);
    }

    void setCursor(EMouseCursor cursor)
    {
        if (cursor == m_cursor && !m_cursorWait)
            return;
        m_cursor = cursor;
        XLockDisplay(m_xDisp);
        XDefineCursor(m_xDisp, m_windowId, GetXCursor(cursor));
        XUnlockDisplay(m_xDisp);
    }

    void setWaitCursor(bool wait)
    {
        if (wait && !m_cursorWait)
        {
            XLockDisplay(m_xDisp);
            XDefineCursor(m_xDisp, m_windowId, X_CURSORS.m_wait);
            XUnlockDisplay(m_xDisp);
            m_cursorWait = true;
        }
        else if (!wait && m_cursorWait)
        {
            setCursor(m_cursor);
            m_cursorWait = false;
        }
    }
    
    void setWindowFrameDefault()
    {
        int x, y, w, h;
        Screen* screen = DefaultScreenOfDisplay(m_xDisp);
        genFrameDefault(screen, x, y, w, h);
        XWindowChanges values = {(int)x, (int)y, (int)w, (int)h};
        XLockDisplay(m_xDisp);
        XConfigureWindow(m_xDisp, m_windowId, CWX|CWY|CWWidth|CWHeight, &values);
        XUnlockDisplay(m_xDisp);
    }
    
    void getWindowFrame(float& xOut, float& yOut, float& wOut, float& hOut) const
    {
        XWindowAttributes attrs;
        XLockDisplay(m_xDisp);
        XGetWindowAttributes(m_xDisp, m_windowId, &attrs);
        XUnlockDisplay(m_xDisp);
        xOut = attrs.x;
        yOut = attrs.y;
        wOut = attrs.width;
        hOut = attrs.height;
    }

    void getWindowFrame(int& xOut, int& yOut, int& wOut, int& hOut) const
    {
        XWindowAttributes attrs;
        XLockDisplay(m_xDisp);
        XGetWindowAttributes(m_xDisp, m_windowId, &attrs);
        XUnlockDisplay(m_xDisp);
        xOut = attrs.x;
        yOut = attrs.y;
        wOut = attrs.width;
        hOut = attrs.height;
    }
    
    void setWindowFrame(float x, float y, float w, float h)
    {
        XWindowChanges values = {(int)x, (int)y, (int)w, (int)h};
        XLockDisplay(m_xDisp);
        XConfigureWindow(m_xDisp, m_windowId, CWX|CWY|CWWidth|CWHeight, &values);
        XUnlockDisplay(m_xDisp);
    }

    void setWindowFrame(int x, int y, int w, int h)
    {
        XWindowChanges values = {x, y, w, h};
        XLockDisplay(m_xDisp);
        XConfigureWindow(m_xDisp, m_windowId, CWX|CWY|CWWidth|CWHeight, &values);
        XUnlockDisplay(m_xDisp);
    }
    
    float getVirtualPixelFactor() const
    {
        return m_pixelFactor;
    }

    bool isFullscreen() const
    {
        return m_inFs;
        unsigned long nitems;
        Atom     actualType;
        int      actualFormat;
        unsigned long     bytes;
        Atom* vals = nullptr;
        bool fullscreen = false;
        XLockDisplay(m_xDisp);
        int ret = XGetWindowProperty(m_xDisp, m_windowId, S_ATOMS->m_netwmState, 0, ~0l, False,
                                     XA_ATOM, &actualType, &actualFormat, &nitems, &bytes, (unsigned char**)&vals);
        XUnlockDisplay(m_xDisp);
        if (ret == Success)
        {
            for (int i=0 ; i<nitems ; ++i)
            {
                if (vals[i] == S_ATOMS->m_netwmStateFullscreen)
                {
                    fullscreen = true;
                    break;
                }
            }
            XFree(vals);
            return fullscreen;
        }

        return false;
    }

    void setStyle(EWindowStyle style)
    {
        struct
        {
            unsigned long flags;
            unsigned long functions;
            unsigned long decorations;
            long inputMode;
            unsigned long status;
        } wmHints = {0};

        if (S_ATOMS->m_motifWmHints)
        {
            wmHints.flags = MWM_HINTS_DECORATIONS | MWM_HINTS_FUNCTIONS;
            if ((style & EWindowStyle::Titlebar) != EWindowStyle::None)
            {
                wmHints.decorations |= MWM_DECOR_BORDER | MWM_DECOR_TITLE | MWM_DECOR_MINIMIZE | MWM_DECOR_MENU;
                wmHints.functions  |= MWM_FUNC_MOVE | MWM_FUNC_MINIMIZE;
            }
            if ((style & EWindowStyle::Resize) != EWindowStyle::None)
            {
                wmHints.decorations |= MWM_DECOR_MAXIMIZE | MWM_DECOR_RESIZEH;
                wmHints.functions  |= MWM_FUNC_RESIZE | MWM_FUNC_MAXIMIZE;
            }

            if ((style & EWindowStyle::Close) != EWindowStyle::None)
                wmHints.functions |= MWM_FUNC_CLOSE;

            XLockDisplay(m_xDisp);
            XChangeProperty(m_xDisp, m_windowId, S_ATOMS->m_motifWmHints, S_ATOMS->m_motifWmHints, 32, PropModeReplace, (unsigned char*)&wmHints, 5);
            XUnlockDisplay(m_xDisp);
        }

        m_styleFlags = style;
    }

    EWindowStyle getStyle() const
    {
        return m_styleFlags;
    }

    void setFullscreen(bool fs)
    {
        if (fs == m_inFs)
            return;

        XEvent fsEvent = {0};
        fsEvent.xclient.type = ClientMessage;
        fsEvent.xclient.serial = 0;
        fsEvent.xclient.send_event = True;
        fsEvent.xclient.window = m_windowId;
        fsEvent.xclient.message_type = S_ATOMS->m_netwmState;
        fsEvent.xclient.format = 32;
        fsEvent.xclient.data.l[0] = fs;
        fsEvent.xclient.data.l[1] = S_ATOMS->m_netwmStateFullscreen;
        fsEvent.xclient.data.l[2] = 0;
        XLockDisplay(m_xDisp);
        XSendEvent(m_xDisp, DefaultRootWindow(m_xDisp), False,
                   StructureNotifyMask | SubstructureRedirectMask, (XEvent*)&fsEvent);
        XUnlockDisplay(m_xDisp);

        m_inFs = fs;
    }

    void waitForRetrace()
    {
        std::unique_lock<std::mutex> lk(m_gfxCtx.m_vsyncmt);
        m_gfxCtx.m_vsynccv.wait(lk);
    }

    uintptr_t getPlatformHandle() const
    {
        return (uintptr_t)m_windowId;
    }

    void _pointingDeviceChanged(int deviceId)
    {
        int nDevices;
        XIDeviceInfo* devices = XIQueryDevice(m_xDisp, deviceId, &nDevices);

        for (int i=0 ; i<nDevices ; ++i)
        {
            XIDeviceInfo* device = &devices[i];

            /* First iterate classes for scrollables */
            int hScroll = -1;
            int vScroll = -1;
            m_hScrollLast = 0.0;
            m_vScrollLast = 0.0;
            m_hScrollValuator = -1;
            m_vScrollValuator = -1;
            for (int j=0 ; j<device->num_classes ; ++j)
            {
                XIAnyClassInfo* dclass = device->classes[j];
                if (dclass->type == XIScrollClass)
                {
                    XIScrollClassInfo* scrollClass = (XIScrollClassInfo*)dclass;
                    if (scrollClass->scroll_type == XIScrollTypeVertical)
                        vScroll = scrollClass->number;
                    else if (scrollClass->scroll_type == XIScrollTypeHorizontal)
                        hScroll = scrollClass->number;
                }
            }

            /* Next iterate for touch and scroll valuators */
            for (int j=0 ; j<device->num_classes ; ++j)
            {
                XIAnyClassInfo* dclass = device->classes[j];
                if (dclass->type == XIValuatorClass)
                {
                    XIValuatorClassInfo* valClass = (XIValuatorClassInfo*)dclass;
                    if (valClass->number == vScroll)
                    {
                        m_vScrollLast = valClass->value;
                        m_vScrollValuator = vScroll;
                    }
                    else if (valClass->number == hScroll)
                    {
                        m_hScrollLast = valClass->value;
                        m_hScrollValuator = hScroll;
                    }
                }
                else if (dclass->type == XITouchClass)
                {
                    XITouchClassInfo* touchClass = (XITouchClassInfo*)dclass;
                    if (touchClass->mode == XIDirectTouch)
                        m_touchType = ETouchType::Display;
                    else if (touchClass->mode == XIDependentTouch)
                        m_touchType = ETouchType::Trackpad;
                    else
                        m_touchType = ETouchType::None;
                }
            }
        }

        XIFreeDeviceInfo(devices);
        m_lastInputID = deviceId;
    }

    SWindowCoord MakeButtonEventCoord(XEvent* event) const
    {
        int x = event->xbutton.x;
        int y = m_wh-event->xbutton.y;
        return
        {
            {x, y},
            {int(x / m_pixelFactor), int(y / m_pixelFactor)},
            {x / float(m_ww), y / float(m_wh)}
        };
    }

    SWindowCoord MakeMotionEventCoord(XEvent* event) const
    {
        int x = event->xmotion.x;
        int y = m_wh-event->xmotion.y;
        return
        {
            {x, y},
            {int(x / m_pixelFactor), int(y / m_pixelFactor)},
            {x / float(m_ww), y / float(m_wh)}
        };
    }

    SWindowCoord MakeCrossingEventCoord(XEvent* event) const
    {
        int x = event->xcrossing.x;
        int y = m_wh-event->xcrossing.y;
        return
        {
            {x, y},
            {int(x / m_pixelFactor), int(y / m_pixelFactor)},
            {x / float(m_ww), y / float(m_wh)}
        };
    }

    void _incomingEvent(void* e)
    {
        XEvent* event = (XEvent*)e;
        switch (event->type)
        {
        case ClientMessage:
        {
            if (event->xclient.data.l[0] == S_ATOMS->m_wmDeleteWindow && m_callback)
            {
                m_callback->destroyed();
                m_callback = nullptr;
            }
            return;
        }
        case Expose:
        {
            Window nw;
            XWindowAttributes wxa;
            int x, y;
            XTranslateCoordinates(m_xDisp, m_windowId, DefaultRootWindow(m_xDisp), event->xexpose.x, event->xexpose.y, &x, &y, &nw);
            XGetWindowAttributes(m_xDisp, m_windowId, &wxa);
            m_wx = x - wxa.x;
            m_wy = y - wxa.y;
            m_ww = event->xexpose.width;
            m_wh = event->xexpose.height;
            if (m_callback)
            {
                SWindowRect rect =
                { {m_wx, m_wy}, {m_ww, m_wh} };
                XUnlockDisplay(m_xDisp);
                m_callback->resized(rect);
                XLockDisplay(m_xDisp);
            }
            return;
        }
        case ConfigureNotify:
        {
            Window nw;
            XWindowAttributes wxa;
            int x, y;
            XTranslateCoordinates(m_xDisp, m_windowId, DefaultRootWindow(m_xDisp), event->xconfigure.x, event->xconfigure.y, &x, &y, &nw);
            XGetWindowAttributes(m_xDisp, m_windowId, &wxa);
            m_wx = x - wxa.x;
            m_wy = y - wxa.y;
            m_ww = event->xconfigure.width;
            m_wh = event->xconfigure.height;

            if (m_callback)
            {
                SWindowRect rect =
                { {m_wx, m_wy}, {m_ww, m_wh} };
                m_callback->windowMoved(rect);
            }
            return;
        }
        case KeyPress:
        {
            if (m_callback)
            {
                ESpecialKey specialKey;
                EModifierKey modifierKey;
                uint32_t charCode = translateKeysym(XLookupKeysym(&event->xkey, 0),
                                                    specialKey, modifierKey);
                EModifierKey modifierMask = translateModifiers(event->xkey.state);
                if (charCode)
                    m_callback->charKeyDown(charCode, modifierMask, false);
                else if (specialKey != ESpecialKey::None)
                    m_callback->specialKeyDown(specialKey, modifierMask, false);
                else if (modifierKey != EModifierKey::None)
                    m_callback->modKeyDown(modifierKey, false);
            }
            return;
        }
        case KeyRelease:
        {
            if (m_callback)
            {
                ESpecialKey specialKey;
                EModifierKey modifierKey;
                uint32_t charCode = translateKeysym(XLookupKeysym(&event->xkey, 0),
                                                    specialKey, modifierKey);
                EModifierKey modifierMask = translateModifiers(event->xkey.state);
                if (charCode)
                    m_callback->charKeyUp(charCode, modifierMask);
                else if (specialKey != ESpecialKey::None)
                    m_callback->specialKeyUp(specialKey, modifierMask);
                else if (modifierKey != EModifierKey::None)
                    m_callback->modKeyUp(modifierKey);
            }
            return;
        }
        case ButtonPress:
        {
            if (m_callback)
            {
                getWindowFrame(m_wx, m_wy, m_ww, m_wh);
                EMouseButton button = translateButton(event->xbutton.button);
                if (button != EMouseButton::None)
                {
                    EModifierKey modifierMask = translateModifiers(event->xbutton.state);
                    m_callback->mouseDown(MakeButtonEventCoord(event), (EMouseButton)button,
                                          (EModifierKey)modifierMask);
                }

                /* Also handle legacy scroll events here */
                if (event->xbutton.button >= 4 && event->xbutton.button <= 7 &&
                    m_hScrollValuator == -1 && m_vScrollValuator == -1)
                {
                    SScrollDelta scrollDelta =
                    {
                        {0.0, 0.0},
                        false
                    };
                    if (event->xbutton.button == 4)
                        scrollDelta.delta[1] = 1.0;
                    else if (event->xbutton.button == 5)
                        scrollDelta.delta[1] = -1.0;
                    else if (event->xbutton.button == 6)
                        scrollDelta.delta[0] = 1.0;
                    else if (event->xbutton.button == 7)
                        scrollDelta.delta[0] = -1.0;
                    m_callback->scroll(MakeButtonEventCoord(event), scrollDelta);
                }
            }
            return;
        }
        case ButtonRelease:
        {
            if (m_callback)
            {
                getWindowFrame(m_wx, m_wy, m_ww, m_wh);
                EMouseButton button = translateButton(event->xbutton.button);
                if (button != EMouseButton::None)
                {
                    EModifierKey modifierMask = translateModifiers(event->xbutton.state);
                    m_callback->mouseUp(MakeButtonEventCoord(event), (EMouseButton)button,
                                        (EModifierKey)modifierMask);
                }
            }
            return;
        }
        case FocusIn:
        {
            if (m_callback)
                m_callback->focusGained();
            return;
        }
        case FocusOut:
        {
            if (m_callback)
                m_callback->focusLost();
            return;
        }
        case MotionNotify:
        {
            if (m_callback)
            {
                getWindowFrame(m_wx, m_wy, m_ww, m_wh);
                m_callback->mouseMove(MakeMotionEventCoord(event));
            }
            return;
        }
        case EnterNotify:
        {
            if (m_callback)
            {
                getWindowFrame(m_wx, m_wy, m_ww, m_wh);
                m_callback->mouseEnter(MakeCrossingEventCoord(event));
            }
            return;
        }
        case LeaveNotify:
        {
            if (m_callback)
            {
                getWindowFrame(m_wx, m_wy, m_ww, m_wh);
                m_callback->mouseLeave(MakeCrossingEventCoord(event));
            }
            return;
        }
        case GenericEvent:
        {
            if (event->xgeneric.extension == XINPUT_OPCODE)
            {
                getWindowFrame(m_wx, m_wy, m_ww, m_wh);
                switch (event->xgeneric.evtype)
                {
                case XI_Motion:
                {
                    fprintf(stderr, "motion\n");

                    XIDeviceEvent* ev = (XIDeviceEvent*)event;
                    if (m_lastInputID != ev->deviceid)
                        _pointingDeviceChanged(ev->deviceid);

                    int cv = 0;
                    double newScroll[2] = {m_hScrollLast, m_vScrollLast};
                    bool didScroll = false;
                    for (int i=0 ; i<ev->valuators.mask_len*8 ; ++i)
                    {
                        if (XIMaskIsSet(ev->valuators.mask, i))
                        {
                            if (i == m_hScrollValuator)
                            {
                                newScroll[0] = ev->valuators.values[cv];
                                didScroll = true;
                            }
                            else if (i == m_vScrollValuator)
                            {
                                newScroll[1] = ev->valuators.values[cv];
                                didScroll = true;
                            }
                            ++cv;
                        }
                    }

                    SScrollDelta scrollDelta =
                    {
                        {newScroll[0] - m_hScrollLast, newScroll[1] - m_vScrollLast},
                        true
                    };

                    m_hScrollLast = newScroll[0];
                    m_vScrollLast = newScroll[1];

                    if (m_callback && didScroll)
                    {
                        int event_x = int(ev->event_x) >> 16;
                        int event_y = m_wh - (int(ev->event_y) >> 16);
                        SWindowCoord coord =
                        {
                            {event_x, event_y},
                            {int(event_x / m_pixelFactor), int(event_y / m_pixelFactor)},
                            {event_x / float(m_ww), event_y / float(m_wh)}
                        };
                        m_callback->scroll(coord, scrollDelta);
                    }
                    return;
                }
                case XI_TouchBegin:
                {
                    XIDeviceEvent* ev = (XIDeviceEvent*)event;
                    if (m_lastInputID != ev->deviceid)
                        _pointingDeviceChanged(ev->deviceid);

                    int cv = 0;
                    double vals[32] = {};
                    for (int i=0 ; i<ev->valuators.mask_len*8 && i<32 ; ++i)
                    {
                        if (XIMaskIsSet(ev->valuators.mask, i))
                        {
                            vals[i] = ev->valuators.values[cv];
                            ++cv;
                        }
                    }

                    STouchCoord coord =
                    {
                        {vals[0], vals[1]}
                    };

                    if (m_callback)
                        m_callback->touchDown(coord, ev->detail);
                    return;
                }
                case XI_TouchUpdate:
                {
                    XIDeviceEvent* ev = (XIDeviceEvent*)event;
                    if (m_lastInputID != ev->deviceid)
                        _pointingDeviceChanged(ev->deviceid);

                    int cv = 0;
                    double vals[32] = {};
                    for (int i=0 ; i<ev->valuators.mask_len*8 && i<32 ; ++i)
                    {
                        if (XIMaskIsSet(ev->valuators.mask, i))
                        {
                            vals[i] = ev->valuators.values[cv];
                            ++cv;
                        }
                    }

                    STouchCoord coord =
                    {
                        {vals[0], vals[1]}
                    };

                    if (m_callback)
                        m_callback->touchMove(coord, ev->detail);
                    return;
                }
                case XI_TouchEnd:
                {
                    XIDeviceEvent* ev = (XIDeviceEvent*)event;
                    if (m_lastInputID != ev->deviceid)
                        _pointingDeviceChanged(ev->deviceid);

                    int cv = 0;
                    double vals[32] = {};
                    for (int i=0 ; i<ev->valuators.mask_len*8 && i<32 ; ++i)
                    {
                        if (XIMaskIsSet(ev->valuators.mask, i))
                        {
                            vals[i] = ev->valuators.values[cv];
                            ++cv;
                        }
                    }

                    STouchCoord coord =
                    {
                        {vals[0], vals[1]}
                    };

                    if (m_callback)
                        m_callback->touchUp(coord, ev->detail);
                    return;
                }
                }
            }
        }
        }
    }
    
    ETouchType getTouchType() const
    {
        return m_touchType;
    }
    
    IGraphicsCommandQueue* getCommandQueue()
    {
        return m_gfxCtx.getCommandQueue();
    }

    IGraphicsDataFactory* getDataFactory()
    {
        return m_gfxCtx.getDataFactory();
    }

    IGraphicsDataFactory* getMainContextDataFactory()
    {
        return m_gfxCtx.getMainContextDataFactory();
    }

    IGraphicsDataFactory* getLoadContextDataFactory()
    {
        return m_gfxCtx.getLoadContextDataFactory();
    }

    bool _isWindowMapped()
    {
        XWindowAttributes attr;
        XLockDisplay(m_xDisp);
        XGetWindowAttributes(m_xDisp, m_windowId, &attr);
        XUnlockDisplay(m_xDisp);
        return attr.map_state != IsUnmapped;
    }
};

IWindow* _WindowXlibNew(const std::string& title,
                       Display* display, int defaultScreen,
                       GLXContext lastCtx)
{
    XLockDisplay(display);
    IWindow* ret = new WindowXlib(title, display, defaultScreen, lastCtx);
    XUnlockDisplay(display);
    return ret;
}
    
}
