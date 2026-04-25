#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "velocity_command_source.h"

#include <algorithm>
#include <cctype>
#include <string>

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

namespace
{

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
    auto lowcmd_sub = std::make_shared<unitree::robot::go2::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
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
    const auto input_mode = select_input_mode();

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     Go2 Controller \n";

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

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

    while (true)
    {
        sleep(1);
    }
    
    return 0;
}
