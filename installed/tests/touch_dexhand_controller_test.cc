
#include <ros/ros.h>
#include <iostream>
#include <thread>
#include <csignal>

#include "../touch_hand_controller.h"
#include "../touch_dexhand_ros1.h"
#include "kuavo_assets/include/package_path.h"

using namespace dexhand;

bool running = true;

void signal_handler(int signal) {
    // Handle the signal
    running = false; // Example action
}

int normal_main(int argc, char** argv) {    
    using namespace eef_controller;
    std::string asserts_path = ocs2::kuavo_assets::getPath();
    std::string config_path = asserts_path + "/config/gesture/preset_gestures.json";

    auto hand_controller = TouchDexhandContrller::Create(config_path);
    if(!hand_controller->init()) {
        std::cout << "[ERROR] Failed to init touch dexhand controller.\n";
        return -1;
    }

    signal(SIGINT, signal_handler);

    std::thread status_thread([&]() {
        while (running) {
            auto finger_status = hand_controller->get_finger_status();
            auto touch_status = hand_controller->get_touch_status();
            std::cout <<"[Left Hand]:" << finger_status[0];
            std::cout <<"[Right Hand]:" << finger_status[1];
            // std::cout <<  touch_status[1]->at(1) << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
        std::cout << "status thread exit \n";
    });

    FingerArray close_speed = {0, 30, 30, 30, 30, 30};
    FingerArray open_speed = {0, -30, -30, -30, -30, -30};
    while (running) {
        // Close both hands
        if (!running) break;
        std::cout << "\033[33mDual Hand Position --> [Close]\033[0m \n";
        hand_controller->send_position(kDualHandClosePositions);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        // Open both hands
        if (!running) break;
        std::cout << "\033[33mDual Hand Position --> [Open]\033[0m \n";
        hand_controller->send_right_position(kOpenFingerPositions);
        hand_controller->send_left_position(kOpenFingerPositions);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        
        if (!running) break;
        std::cout << "\033[32mLeft Hand Speed --> [Close]\033[0m \n";
        hand_controller->send_left_speed(close_speed);
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        
        if (!running) break;
        std::cout << "\033[32mLeft Hand Speed --> [Open]\033[0m \n";
        hand_controller->send_left_speed(open_speed);
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));

        if (!running) break;
        std::cout << "\033[32mRight Hand Speed --> [Close]\033[0m \n";
        hand_controller->send_right_speed(close_speed);
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        
        if (!running) break;
        std::cout << "\033[32mRight Hand Speed --> [Open]\033[0m \n";
        hand_controller->send_right_speed(open_speed);
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }

    if(status_thread.joinable()) {
        status_thread.join();
    }

    return 0;
}

int ros1_main(int argc, char** argv) {
    ros::init(argc, argv, "touch_dexhand_test_node");
    ros::NodeHandle nh;
    using namespace eef_controller;

    TouchDexHandNode node;
    
    std::string asserts_path = ocs2::kuavo_assets::getPath();
    std::string config_path = asserts_path + "/config/gesture/preset_gestures.json";

    if (!node.init(nh, config_path)) {
        std::cout << "[ERROR] Failed to init touch dexhand node.\n";
        return -1;
    }

    std::cout << "\n\n----- Touch Dexhand ROS Node ----- \n";
    std::cout << "Topics:\n";
    std::cout << " - /dexhand/command \n";
    std::cout << " - /dexhand/right/command \n";
    std::cout << " - /dexhand/left/command \n";
    std::cout << " - /dexhand/state \n";
    std::cout << " - /dexhand/touch_state \n";
    std::cout << "Services:\n";
    std::cout << "----- Touch Dexhand ROS Node ----- \n";
    std::cout << "/dexhand/change_force_level \n";
    std::cout << "/dexhand/left/enable_touch_sensor \n";
    std::cout << "/dexhand/right/enable_touch_sensor \n";
    std::cout << "/gesture/execute \n";
    std::cout << "/gesture/execute_state \n";
    std::cout << "/gesture/list \n";
    ros::spin();
    
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--ros") {
        return ros1_main(argc, argv);
    } else {
        return normal_main(argc, argv);
    }
}