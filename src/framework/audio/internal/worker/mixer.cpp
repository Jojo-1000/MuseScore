/*
 * SPDX-License-Identifier: GPL-3.0-only
 * MuseScore-CLA-applies
 *
 * MuseScore
 * Music Composition & Notation
 *
 * Copyright (C) 2021 MuseScore BVBA and others
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#include "mixer.h"

#include "async/async.h"
#include "log.h"

#include <limits>

#include "concurrency/taskscheduler.h"

#include "internal/audiosanitizer.h"
#include "internal/audiothread.h"
#include "internal/dsp/audiomathutils.h"
#include "audioerrors.h"

using namespace mu;
using namespace mu::audio;
using namespace mu::async;

Mixer::Mixer()
{
    ONLY_AUDIO_WORKER_THREAD;
}

Mixer::~Mixer()
{
    ONLY_AUDIO_WORKER_THREAD;
}

IAudioSourcePtr Mixer::mixedSource()
{
    ONLY_AUDIO_WORKER_THREAD;
    return shared_from_this();
}

RetVal<MixerChannelPtr> Mixer::addChannel(const TrackId trackId, IAudioSourcePtr source)
{
    ONLY_AUDIO_WORKER_THREAD;

    RetVal<MixerChannelPtr> result;

    if (!source) {
        result.val = nullptr;
        result.ret = make_ret(Err::InvalidAudioSource);
        return result;
    }

    m_trackChannels.emplace(trackId, std::make_shared<MixerChannel>(trackId, std::move(source), m_sampleRate));

    result.val = m_trackChannels[trackId];
    result.ret = make_ret(Ret::Code::Ok);

    return result;
}

RetVal<MixerChannelPtr> Mixer::addAuxChannel(const TrackId trackId)
{
    ONLY_AUDIO_WORKER_THREAD;

    m_auxChannels.emplace(trackId, std::make_shared<MixerChannel>(trackId, m_sampleRate));

    RetVal<MixerChannelPtr> result;
    result.val = m_auxChannels[trackId];
    result.ret = make_ret(Ret::Code::Ok);

    return result;
}

Ret Mixer::removeChannel(const TrackId trackId)
{
    ONLY_AUDIO_WORKER_THREAD;

    auto search = m_trackChannels.find(trackId);

    if (search != m_trackChannels.end() && search->second) {
        m_trackChannels.erase(trackId);
        return make_ret(Ret::Code::Ok);
    }

    search = m_auxChannels.find(trackId);

    if (search != m_auxChannels.end() && search->second) {
        m_auxChannels.erase(trackId);
        return make_ret(Ret::Code::Ok);
    }

    return make_ret(Err::InvalidTrackId);
}

void Mixer::setAudioChannelsCount(const audioch_t count)
{
    ONLY_AUDIO_WORKER_THREAD;

    m_audioChannelsCount = count;
}

void Mixer::setSampleRate(unsigned int sampleRate)
{
    ONLY_AUDIO_WORKER_THREAD;

    m_limiter = std::make_unique<dsp::Limiter>(sampleRate);

    AbstractAudioSource::setSampleRate(sampleRate);

    for (auto& channel : m_trackChannels) {
        channel.second->setSampleRate(sampleRate);
    }
}

unsigned int Mixer::audioChannelsCount() const
{
    ONLY_AUDIO_WORKER_THREAD;

    return m_audioChannelsCount;
}

samples_t Mixer::process(float* outBuffer, size_t bufferSize, samples_t samplesPerChannel)
{
    ONLY_AUDIO_WORKER_THREAD;

    for (IClockPtr clock : m_clocks) {
        clock->forward((samplesPerChannel * 1000000) / m_sampleRate);
    }

    const size_t outBufferSize = samplesPerChannel * m_audioChannelsCount;
    IF_ASSERT_FAILED(outBufferSize <= bufferSize) {
        return 0;
    }

    std::fill(outBuffer, outBuffer + outBufferSize, 0.f);

    // Use 2 channels in between because fx assumes that
    const audioch_t intermediateChannels = 2;
    const size_t writeCacheSize = samplesPerChannel * intermediateChannels;
    if (m_writeCacheBuff.size() != writeCacheSize) {
        m_writeCacheBuff.resize(outBufferSize, 0.f);
    }
    std::fill(m_writeCacheBuff.begin(), m_writeCacheBuff.end(), 0.f);

    samples_t masterChannelSampleCount = 0;

    std::vector<std::pair<std::future<std::vector<float> >, audioch_t> > futureList;

    for (const auto& pair : m_trackChannels) {
        MixerChannelPtr channel = pair.second;
        const audioch_t audioChannels = channel->audioChannelsCount();
        std::future<std::vector<float> > future = TaskScheduler::instance()->submit([outBufferSize, samplesPerChannel,
                                                                                     channel, audioChannels]() -> std::vector<float> {
            // Buffers are kept for each thread instance, but potentially need to be resized if number of samples change
            thread_local std::vector<float> buffer;
            thread_local std::vector<float> silent_buffer;

            silent_buffer.resize(samplesPerChannel * audioChannels, 0.f);
            buffer = silent_buffer;

            if (channel) {
                channel->process(buffer.data(), outBufferSize, samplesPerChannel);
            }

            return buffer;
        });

        futureList.emplace_back(std::move(future), audioChannels);
    }

    for (size_t i = 0; i < futureList.size(); ++i) {
        mixOutputFromChannel(m_writeCacheBuff.data(), writeCacheSize, intermediateChannels, futureList[i].first.get().data(), samplesPerChannel, futureList[i].second);

        masterChannelSampleCount = std::max(samplesPerChannel, masterChannelSampleCount);
    }

    if (m_masterParams.muted || masterChannelSampleCount == 0) {
        for (audioch_t audioChNum = 0; audioChNum < m_audioChannelsCount; ++audioChNum) {
            notifyAboutAudioSignalChanges(audioChNum, 0);
        }
        return 0;
    }

    completeOutput(m_writeCacheBuff.data(), samplesPerChannel, intermediateChannels);

    for (IFxProcessorPtr& fxProcessor : m_masterFxProcessors) {
        if (fxProcessor->active()) {
            fxProcessor->process(m_writeCacheBuff.data(), writeCacheSize, samplesPerChannel);
        }
    }

    audioch_t minChannels = std::min(intermediateChannels, m_audioChannelsCount);
    for (samples_t s = 0; s < samplesPerChannel; ++s) {
        // TODO: Mix stereo -> mono or stereo -> surround
        for (audioch_t audioChNum = 0; audioChNum < minChannels; ++audioChNum) {
            int outIdx = s * m_audioChannelsCount + audioChNum;
            int inIdx = s * intermediateChannels + audioChNum;

            outBuffer[outIdx] += m_writeCacheBuff[inIdx];
        }
    }

    return masterChannelSampleCount;
}

void Mixer::setIsActive(bool arg)
{
    ONLY_AUDIO_WORKER_THREAD;

    AbstractAudioSource::setIsActive(arg);

    for (const auto& channel : m_trackChannels) {
        channel.second->setIsActive(arg);
    }
}

void Mixer::addClock(IClockPtr clock)
{
    ONLY_AUDIO_WORKER_THREAD;

    m_clocks.insert(std::move(clock));
}

void Mixer::removeClock(IClockPtr clock)
{
    ONLY_AUDIO_WORKER_THREAD;

    m_clocks.erase(clock);
}

AudioOutputParams Mixer::masterOutputParams() const
{
    ONLY_AUDIO_WORKER_THREAD;

    return m_masterParams;
}

void Mixer::setMasterOutputParams(const AudioOutputParams& params)
{
    ONLY_AUDIO_WORKER_THREAD;

    if (m_masterParams == params) {
        return;
    }

    m_masterFxProcessors.clear();
    m_masterFxProcessors = fxResolver()->resolveMasterFxList(params.fxChain);

    for (IFxProcessorPtr& fx : m_masterFxProcessors) {
        fx->setSampleRate(m_sampleRate);
        fx->paramsChanged().onReceive(this, [this](const AudioFxParams& fxParams) {
            m_masterParams.fxChain.insert_or_assign(fxParams.chainOrder, fxParams);
            m_masterOutputParamsChanged.send(m_masterParams);
        });
    }

    AudioOutputParams resultParams = params;

    auto findFxProcessor = [this](const std::pair<AudioFxChainOrder, AudioFxParams>& params) -> IFxProcessorPtr {
        for (IFxProcessorPtr& fx : m_masterFxProcessors) {
            if (fx->params().chainOrder != params.first) {
                continue;
            }

            if (fx->params().resourceMeta == params.second.resourceMeta) {
                return fx;
            }
        }

        return nullptr;
    };

    for (auto it = resultParams.fxChain.begin(); it != resultParams.fxChain.end();) {
        if (IFxProcessorPtr fx = findFxProcessor(*it)) {
            fx->setActive(it->second.active);
            ++it;
        } else {
            it = resultParams.fxChain.erase(it);
        }
    }

    m_masterParams = resultParams;
    m_masterOutputParamsChanged.send(resultParams);
}

void Mixer::clearMasterOutputParams()
{
    setMasterOutputParams(AudioOutputParams());
}

Channel<AudioOutputParams> Mixer::masterOutputParamsChanged() const
{
    return m_masterOutputParamsChanged;
}

async::Channel<audioch_t, AudioSignalVal> Mixer::masterAudioSignalChanges() const
{
    return m_audioSignalNotifier.audioSignalChanges;
}

void Mixer::mixOutputFromChannel(float* outBuffer, size_t outBufferSize, audioch_t outChannels, float* inBuffer, unsigned int samplesCount,
                                 audioch_t inChannels)
{
    IF_ASSERT_FAILED(outBuffer && inBuffer) {
        return;
    }
    IF_ASSERT_FAILED(outBufferSize >= outChannels * samplesCount) {
        return;
    }

    if (m_masterParams.muted) {
        return;
    }

    audioch_t minChannels = std::min(outChannels, inChannels);
    for (samples_t s = 0; s < samplesCount; ++s) {
        // TODO: Mix stereo -> mono or stereo -> surround
        for (audioch_t audioChNum = 0; audioChNum < minChannels; ++audioChNum) {
            int outIdx = s * outChannels + audioChNum;
            int inIdx = s * inChannels + audioChNum;

            outBuffer[outIdx] += inBuffer[inIdx];
        }
    }
}

void Mixer::completeOutput(float* buffer, samples_t samplesPerChannel, audioch_t channels)
{
    IF_ASSERT_FAILED(buffer) {
        return;
    }

    float totalSquaredSum = 0.f;
    float volume = dsp::linearFromDecibels(m_masterParams.volume);

    for (audioch_t audioChNum = 0; audioChNum < channels; ++audioChNum) {
        float singleChannelSquaredSum = 0.f;

        gain_t totalGain = dsp::balanceGain(m_masterParams.balance, audioChNum) * volume;

        for (samples_t s = 0; s < samplesPerChannel; ++s) {
            int idx = s * channels + audioChNum;

            float resultSample = buffer[idx] * totalGain;
            buffer[idx] = resultSample;

            float squaredSample = resultSample * resultSample;
            totalSquaredSum += squaredSample;
            singleChannelSquaredSum += squaredSample;
        }

        float rms = dsp::samplesRootMeanSquare(singleChannelSquaredSum, samplesPerChannel);
        notifyAboutAudioSignalChanges(audioChNum, rms);
    }

    if (!m_limiter->isActive()) {
        return;
    }

    float totalRms = dsp::samplesRootMeanSquare(totalSquaredSum, samplesPerChannel * m_audioChannelsCount);
    m_limiter->process(totalRms, buffer, channels, samplesPerChannel);
}

void Mixer::notifyAboutAudioSignalChanges(const audioch_t audioChannelNumber, const float linearRms) const
{
    m_audioSignalNotifier.updateSignalValues(audioChannelNumber, linearRms, dsp::dbFromSample(linearRms));
}
