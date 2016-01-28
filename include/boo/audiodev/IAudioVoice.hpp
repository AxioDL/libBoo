#ifndef BOO_IAUDIOVOICE_HPP
#define BOO_IAUDIOVOICE_HPP

#include <stddef.h>
#include <stdint.h>

namespace boo
{

enum class AudioChannelSet
{
    Stereo,
    Quad,
    Surround51,
    Surround71
};

enum class AudioChannel
{
    FrontLeft,
    FrontRight,
    RearLeft,
    RearRight,
    FrontCenter,
    LFE,
    SideLeft,
    SideRight
};

struct ChannelMap
{
    unsigned m_channelCount = 0;
    AudioChannel m_channels[8] = {};
};

static inline unsigned ChannelCount(AudioChannelSet layout)
{
    switch (layout)
    {
    case AudioChannelSet::Stereo:
        return 2;
    case AudioChannelSet::Quad:
        return 4;
    case AudioChannelSet::Surround51:
        return 6;
    case AudioChannelSet::Surround71:
        return 8;
    }
    return 0;
}

struct IAudioVoice
{
    /** Get voice's actual channel-map based on client request and HW capabilities */
    virtual const ChannelMap& channelMap() const=0;

    /** Called by client in response to IAudioVoiceCallback::needsNextBuffer()
     *  Supplying channel-interleaved sample data */
    virtual void bufferSampleData(const int16_t* data, size_t frames)=0;
};

}

#endif // BOO_IAUDIOVOICE_HPP
