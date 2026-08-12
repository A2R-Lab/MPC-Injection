#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "velocity_command_source.h"

#include <unitree/idl/ros2/String_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <string>
#include <thread>

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

namespace
{

volatile std::sig_atomic_t shutdown_requested = 0;

void request_shutdown(int)
{
    shutdown_requested = 1;
}

std::string service_name_from_motion_mode(const std::string& form, const std::string& name)
{
    if (name.empty())
    {
        return "none";
    }

    if (form == "0")
    {
        if (name == "normal") return "sport_mode";
        if (name == "ai") return "ai_sport";
        if (name == "advanced") return "advanced_sport";
    }
    else
    {
        if (name == "ai-w") return "wheeled_sport(go2W)";
        if (name == "normal-w") return "wheeled_sport(b2W)";
    }

    return name;
}

void release_unitree_motion_service()
{
    unitree::robot::b2::MotionSwitcherClient motion_switcher;
    motion_switcher.SetTimeout(5.0f);
    motion_switcher.Init();

    constexpr int max_attempts = 8;
    for (int attempt = 1; attempt <= max_attempts; ++attempt)
    {
        std::string form;
        std::string name;
        const int32_t check_ret = motion_switcher.CheckMode(form, name);
        if (check_ret != 0)
        {
            spdlog::warn(
                "MotionSwitcher CheckMode failed on attempt {}/{} with error code {}.",
                attempt,
                max_attempts,
                check_ret
            );
        }
        else if (name.empty())
        {
            spdlog::info("No Unitree high-level motion service is active.");
            return;
        }
        else
        {
            spdlog::warn(
                "Active Unitree motion service detected: {} (form='{}', mode='{}'). Releasing it before low-level control.",
                service_name_from_motion_mode(form, name),
                form,
                name
            );
        }

        const int32_t release_ret = motion_switcher.ReleaseMode();
        if (release_ret == 0)
        {
            spdlog::info("MotionSwitcher ReleaseMode succeeded.");
        }
        else
        {
            spdlog::warn("MotionSwitcher ReleaseMode failed with error code {}.", release_ret);
        }

        std::this_thread::sleep_for(std::chrono::seconds(3));
    }

    std::string form;
    std::string name;
    const int32_t check_ret = motion_switcher.CheckMode(form, name);
    if (check_ret == 0 && name.empty())
    {
        spdlog::info("No Unitree high-level motion service is active.");
        return;
    }

    if (check_ret != 0)
    {
        spdlog::critical(
            "Could not verify that Unitree's high-level motion service is released. CheckMode error code: {}.",
            check_ret
        );
    }
    else
    {
        spdlog::critical(
            "Unitree high-level motion service is still active after release attempts: {} (form='{}', mode='{}').",
            service_name_from_motion_mode(form, name),
            form,
            name
        );
    }

    std::exit(1);
}

void verify_lowcmd_channel_is_free()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::go2::subscription::LowCmd>();
    std::this_thread::sleep_for(std::chrono::milliseconds(1200));
    if (!lowcmd_sub->isTimeout())
    {
        spdlog::critical(
            "Another process is still publishing on rt/lowcmd after releasing Unitree's motion service. "
            "Stop the other low-level controller before launching go2_ctrl."
        );
        std::exit(1);
    }
}

void stop_lidar_rotation()
{
    unitree::robot::ChannelPublisher<std_msgs::msg::dds_::String_> lidar_switch("rt/utlidar/switch");
    lidar_switch.InitChannel();

    std_msgs::msg::dds_::String_ command;
    command.data("OFF");

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    bool wrote = false;
    for (int attempt = 0; attempt < 3; ++attempt)
    {
        wrote = lidar_switch.Write(command) || wrote;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (wrote)
    {
        spdlog::info("Sent OFF command to Unitree LiDAR switch topic; LiDAR rotation should stop.");
    }
    else
    {
        spdlog::warn("Failed to publish OFF command to Unitree LiDAR switch topic.");
    }
}

VelocityCommandInputMode select_input_mode()
{
    while (true)
    {
        std::cout << "Select velocity command input ([c]ontroller / [k]eyboard): ";

        std::string choice;
        if (!std::getline(std::cin, choice))
        {
            std::cin.clear();
            return VelocityCommandInputMode::Controller;
        }

        choice.erase(
            std::remove_if(
                choice.begin(),
                choice.end(),
                [](unsigned char ch) { return std::isspace(ch); }
            ),
            choice.end()
        );

        std::transform(
            choice.begin(),
            choice.end(),
            choice.begin(),
            [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); }
        );

        if (choice == "c" || choice == "controller")
        {
            return VelocityCommandInputMode::Controller;
        }
        if (choice == "k" || choice == "keyboard")
        {
            return VelocityCommandInputMode::Keyboard;
        }

        std::cout << "Please enter 'c' or 'k'.\n";
    }
}

} // namespace

void init_fsm_state()
{
    release_unitree_motion_service();
    verify_lowcmd_channel_is_free();

    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);
    std::signal(SIGINT, request_shutdown);
    if (param::torque_output_enabled)
    {
        spdlog::info("Hardware tau_est recording enabled: {}", param::torque_output.string());
    }
    else
    {
        spdlog::info("Hardware torque recording disabled.");
    }
    const auto input_mode = select_input_mode();

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     Go2 Controller \n";

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());
    stop_lidar_rotation();

    init_fsm_state();

    VelocityCommandSource::instance().set_mode(input_mode);
    if (input_mode == VelocityCommandInputMode::Keyboard)
    {
        FSMState::keyboard = std::make_shared<Keyboard>();
    }

    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    std::cout << "Press [L2 + up] to enter FixStand mode.\n";
    std::cout << "And then press [R2 + A] to start controlling the robot.\n";
    if (input_mode == VelocityCommandInputMode::Keyboard)
    {
        std::cout << "Keyboard mode enabled: [Up]/[Down] adjust vx in steps of "
                  << VelocityCommandSource::step_size() << ". Press [R] to reset vx.\n";
        std::cout << "Press [L2 + B] on the controller to enter passive mode.\n";
    }
    else
    {
        std::cout << "Controller mode enabled: use the controller sticks for velocity commands.\n";
        std::cout << "Press [L2 + B] to enter passive mode.\n";
    }

    while (!shutdown_requested)
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    std::cout << "\nCtrl+C received; stopping controller.\n";
    fsm->stop();
    if (param::torque_output_enabled && !State_RLBase::torque_recording_started.load())
    {
        spdlog::info("No Policy/Velocity torque samples were collected; not writing {}.", param::torque_output.string());
    }
    
    return 0;
}
