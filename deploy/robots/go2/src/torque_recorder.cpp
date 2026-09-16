#include "torque_recorder.h"

#include <chrono>
#include <stdexcept>

#include <spdlog/spdlog.h>

#include "cnpy.h"

TorqueRecorder::TorqueRecorder(std::filesystem::path output_path, std::vector<int> joint_ids_map)
    : output_path_(std::move(output_path)),
      joint_ids_map_(std::move(joint_ids_map)),
      lowstate_subscriber_("rt/lowstate")
{
    if (joint_ids_map_.size() != 12)
    {
        throw std::runtime_error("Go2 torque recorder requires a 12-joint SDK-to-training map.");
    }
}

void TorqueRecorder::subscribe()
{
    if (subscribed_)
    {
        return;
    }

    lowstate_subscriber_.InitChannel(
        [this](const void* message)
        {
            capture_lowstate(*static_cast<const LowStateMessage*>(message));
        }
    );
    subscribed_ = true;
}

void TorqueRecorder::start()
{
    recording_.store(false, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        lowstate_samples_.clear();
        command_samples_.clear();
    }
    recording_.store(true, std::memory_order_release);
}

void TorqueRecorder::capture_lowstate(const LowStateMessage& lowstate)
{
    const int64_t timestamp_ns = monotonic_ns();
    if (!recording_.load(std::memory_order_acquire))
    {
        return;
    }

    LowStateSample sample;
    sample.tick = lowstate.tick();
    sample.host_monotonic_ns = timestamp_ns;
    for (size_t joint = 0; joint < sample.tau_est.size(); ++joint)
    {
        sample.tau_est[joint] = lowstate.motor_state()[joint_ids_map_[joint]].tau_est();
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (recording_.load(std::memory_order_acquire))
    {
        lowstate_samples_.push_back(sample);
    }
}

void TorqueRecorder::record_command(const std::array<float, 3>& command)
{
    const int64_t timestamp_ns = monotonic_ns();
    if (!recording_.load(std::memory_order_acquire))
    {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (recording_.load(std::memory_order_acquire))
    {
        command_samples_.push_back({command, timestamp_ns});
    }
}

bool TorqueRecorder::stop_and_write()
{
    recording_.store(false, std::memory_order_release);

    std::vector<LowStateSample> lowstate_samples;
    std::vector<CommandSample> command_samples;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        lowstate_samples = lowstate_samples_;
        command_samples = command_samples_;
    }

    if (lowstate_samples.empty())
    {
        spdlog::info("No Policy/Velocity torque samples were collected; not writing {}.", output_path_.string());
        return false;
    }

    std::vector<float> tau_est(12 * lowstate_samples.size());
    std::vector<uint32_t> ticks(lowstate_samples.size());
    std::vector<int64_t> timestamps(lowstate_samples.size());
    for (size_t sample_index = 0; sample_index < lowstate_samples.size(); ++sample_index)
    {
        const auto& sample = lowstate_samples[sample_index];
        ticks[sample_index] = sample.tick;
        timestamps[sample_index] = sample.host_monotonic_ns;
        for (size_t joint = 0; joint < sample.tau_est.size(); ++joint)
        {
            tau_est[joint * lowstate_samples.size() + sample_index] = sample.tau_est[joint];
        }
    }

    std::vector<float> commands(3 * command_samples.size());
    std::vector<int64_t> command_timestamps(command_samples.size());
    for (size_t sample_index = 0; sample_index < command_samples.size(); ++sample_index)
    {
        const auto& sample = command_samples[sample_index];
        command_timestamps[sample_index] = sample.host_monotonic_ns;
        for (size_t axis = 0; axis < sample.command.size(); ++axis)
        {
            commands[axis * command_samples.size() + sample_index] = sample.command[axis];
        }
    }

    const uint8_t is_hardware_tau_est = 1;
    const std::string archive_path = output_path_.string();
    cnpy::npz_save(archive_path, "tau_applied", tau_est.data(), {12, lowstate_samples.size()});
    cnpy::npz_save(archive_path, "lowstate_tick", ticks.data(), {ticks.size()}, "a");
    cnpy::npz_save(archive_path, "host_monotonic_ns", timestamps.data(), {timestamps.size()}, "a");
    cnpy::npz_save(archive_path, "commanded_velocity", commands.data(), {3, command_samples.size()}, "a");
    cnpy::npz_save(archive_path, "command_host_monotonic_ns", command_timestamps.data(), {command_timestamps.size()}, "a");
    cnpy::npz_save(archive_path, "is_hardware_tau_est", &is_hardware_tau_est, {1}, "a");

    spdlog::info(
        "Saved {} Policy/Velocity tau_est samples and {} policy commands to {}.",
        lowstate_samples.size(),
        command_samples.size(),
        archive_path
    );
    return true;
}

int64_t TorqueRecorder::monotonic_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}
