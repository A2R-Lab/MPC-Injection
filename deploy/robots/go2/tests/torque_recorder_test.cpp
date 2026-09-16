#include "FSM/CtrlFSM.h"
#include "torque_recorder.h"

#include <array>
#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

#include "cnpy.h"

namespace
{

bool expect(bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << message << '\n';
        return false;
    }
    return true;
}

class ExitTrackingState : public BaseState
{
public:
    ExitTrackingState() : BaseState(999, "ExitTracking") {}

    void exit() override
    {
        exited = true;
    }

    bool exited = false;
};

} // namespace

int main(int argc, char** argv)
{
    const bool remove_output = argc == 1;
    const std::filesystem::path output_path = remove_output
        ? std::filesystem::temp_directory_path() / "go2_torque_recorder_test.npz"
        : argv[1];
    std::filesystem::remove(output_path);

    TorqueRecorder recorder(output_path, {3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8});
    TorqueRecorder::LowStateMessage lowstate;
    lowstate.tick(42);
    for (size_t joint = 0; joint < 12; ++joint)
    {
        lowstate.motor_state()[joint].tau_est(static_cast<float>(joint + 1));
    }

    recorder.capture_lowstate(lowstate);
    recorder.start();
    recorder.capture_lowstate(lowstate);
    recorder.record_command({0.2f, -0.1f, 0.3f});
    if (!expect(recorder.stop_and_write(), "normal finalization should write a Policy sample"))
    {
        return 1;
    }

    const auto archive = cnpy::npz_load(output_path.string());
    const auto tau = archive.at("tau_applied").as_vec<float>();
    const auto ticks = archive.at("lowstate_tick").as_vec<uint32_t>();
    const auto timestamps = archive.at("host_monotonic_ns").as_vec<int64_t>();
    const auto commands = archive.at("commanded_velocity").as_vec<float>();
    const auto command_timestamps = archive.at("command_host_monotonic_ns").as_vec<int64_t>();
    const auto hardware = archive.at("is_hardware_tau_est").as_vec<uint8_t>();

    const std::vector<float> expected_tau = {4, 5, 6, 1, 2, 3, 10, 11, 12, 7, 8, 9};
    if (!expect(archive.at("tau_applied").shape == std::vector<size_t>({12, 1}), "tau_applied should have shape (12, N)")) return 1;
    if (!expect(archive.at("commanded_velocity").shape == std::vector<size_t>({3, 1}), "commanded_velocity should have shape (3, M)")) return 1;
    if (!expect(tau == expected_tau, "tau_est should use the SDK-to-training map")) return 1;
    if (!expect(ticks.size() == 1 && ticks[0] == 42, "low-state tick should align with tau_est")) return 1;
    if (!expect(timestamps.size() == 1 && timestamps[0] > 0, "low-state timestamp should be recorded")) return 1;
    if (!expect(commands == std::vector<float>({0.2f, -0.1f, 0.3f}), "policy command should be recorded")) return 1;
    if (!expect(command_timestamps.size() == 1 && command_timestamps[0] > 0, "command timestamp should be recorded")) return 1;
    if (!expect(hardware == std::vector<uint8_t>({1}), "archive should identify hardware tau_est")) return 1;

    auto state = std::make_shared<ExitTrackingState>();
    CtrlFSM fsm(state);
    fsm.start();
    fsm.stop();
    if (!expect(state->exited, "orderly shutdown should exit the active state")) return 1;

    if (remove_output)
    {
        std::filesystem::remove(output_path);
    }
    return 0;
}
