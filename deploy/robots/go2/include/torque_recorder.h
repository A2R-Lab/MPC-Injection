#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <vector>

#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

class TorqueRecorder
{
public:
    using LowStateMessage = unitree_go::msg::dds_::LowState_;

    TorqueRecorder(std::filesystem::path output_path, std::vector<int> joint_ids_map);

    void subscribe();
    void start();
    void capture_lowstate(const LowStateMessage& lowstate);
    void record_command(const std::array<float, 3>& command);
    bool stop_and_write();

private:
    struct LowStateSample
    {
        std::array<float, 12> tau_est;
        uint32_t tick;
        int64_t host_monotonic_ns;
    };

    struct CommandSample
    {
        std::array<float, 3> command;
        int64_t host_monotonic_ns;
    };

    static int64_t monotonic_ns();

    const std::filesystem::path output_path_;
    const std::vector<int> joint_ids_map_;
    unitree::robot::ChannelSubscriber<LowStateMessage> lowstate_subscriber_;
    std::atomic<bool> recording_ {false};
    std::mutex mutex_;
    std::vector<LowStateSample> lowstate_samples_;
    std::vector<CommandSample> command_samples_;
    bool subscribed_ = false;
};
