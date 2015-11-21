#ifndef BOO_SYSTEM_HPP
#define BOO_SYSTEM_HPP

#include <string>

#define ENABLE_BITWISE_ENUM(type)\
inline type operator|(type a, type b)\
{\
    using T = std::underlying_type_t<type>;\
    return type(static_cast<T>(a) | static_cast<T>(b));\
}\
inline type operator&(type a, type b)\
{\
    using T = std::underlying_type_t<type>;\
    return type(static_cast<T>(a) & static_cast<T>(b));\
}\
inline type& operator|=(type& a, const type& b)\
{\
    using T = std::underlying_type_t<type>;\
    a = type(static_cast<T>(a) | static_cast<T>(b));\
    return a;\
}\
inline type& operator&=(type& a, const type& b)\
{\
    using T = std::underlying_type_t<type>;\
    a = type(static_cast<T>(a) & static_cast<T>(b));\
    return a;\
}\
inline type operator~(const type& key)\
{\
    using T = std::underlying_type_t<type>;\
    return type(~static_cast<T>(key));\
}

namespace boo
{

#ifdef _WIN32
    using SystemString = std::wstring;
    using SystemChar = wchar_t;
#   ifndef _S
#   define _S(val) L ## val
#   endif
#else
    using SystemString = std::string;
    using SystemChar = char;
#   ifndef _S
#   define _S(val) val
#   endif
#endif

}

#ifdef _WIN32
#include <wrl/client.h>
template <class T>
using ComPtr = Microsoft::WRL::ComPtr<T>;
#endif

#endif
