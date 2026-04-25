#pragma once

#include <algorithm>
#include <atomic>
#include <string>

#include <spdlog/spdlog.h>

#include "isaaclab/devices/keyboard/keyboard.h"

enum class VelocityCommandInputMode
{
    Controller,
    Keyboard,
};

class VelocityCommandSource
{
public:
    static VelocityCommandSource& instance()
    {
        static VelocityCommandSource source;
        return source;
    }

    void set_mode(VelocityCommandInputMode mode)
    {
        mode_.store(mode, std::memory_order_relaxed);
        vx_.store(0.0f, std::memory_order_relaxed);
    }

    void set_keyboard_limits(float vx_min, float vx_max)
    {
        vx_min_.store(vx_min, std::memory_order_relaxed);
        vx_max_.store(vx_max, std::memory_order_relaxed);
    }

    VelocityCommandInputMode mode() const
    {
        return mode_.load(std::memory_order_relaxed);
    }

    bool uses_keyboard() const
    {
        return mode() == VelocityCommandInputMode::Keyboard;
    }

    void update_from_keyboard(const Keyboard& keyboard)
    {
        if (!uses_keyboard() || !keyboard.on_pressed)
        {
            return;
        }

        float vx = vx_.load(std::memory_order_relaxed);
        const std::string key = keyboard.key();

        if (key == "up")
        {
            vx += step_size_;
        }
        else if (key == "down")
        {
            vx -= step_size_;
        }
        else if (key == "r" || key == "R")
        {
            vx = 0.0f;
        }
        else
        {
            return;
        }

        vx = std::clamp(
            vx,
            vx_min_.load(std::memory_order_relaxed),
            vx_max_.load(std::memory_order_relaxed)
        );

        vx_.store(vx, std::memory_order_relaxed);
        spdlog::info("Keyboard velocity command: vx={:.2f}", vx);
    }

    float vx() const
    {
        return vx_.load(std::memory_order_relaxed);
    }

    static constexpr float step_size()
    {
        return step_size_;
    }

private:
    VelocityCommandSource() = default;

    static constexpr float step_size_ = 0.1f;

    std::atomic<VelocityCommandInputMode> mode_{VelocityCommandInputMode::Controller};
    std::atomic<float> vx_{0.0f};
    std::atomic<float> vx_min_{-0.5f};
    std::atomic<float> vx_max_{1.0f};
};
