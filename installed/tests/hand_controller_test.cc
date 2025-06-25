#include <iostream>
#include <thread>
#include "hand_controller.h"
#include "gesture_types.h"

using namespace eef_controller;
int main(int argc, char **argv)
{
    std::string asserts_path = KUAVO_ASSETS_PATH;
    // Override KUAVO_ASSETS_PATH if first argument is provided
    if (argc > 1 && argv[1] != nullptr) {
        asserts_path = std::string(argv[1]);
    }
    std::string config_path = asserts_path + "/config/gesture/preset_gestures.json";
    std::cout << "config_path: " << config_path << std::endl;
    BrainCoControllerPtr bc_ptr = BrainCoController::Create(config_path);

    bc_ptr->init();

    std::vector<GestureInfoMsg> gestures = bc_ptr->list_gestures();

    std::cout << "@@@@ gestures: \n";
    for(auto &gesture: gestures)
        std::cout << " " << gesture.gesture_name << std::endl;
    std::cout << "-------------------------------------------\n";         

    /***  test: abort! ***/
    {
        std::cout << "------------------ test: abort! ------------\n";         
        std::string err_msg;
        bc_ptr->execute_gesture(HandSide::RIGHT, "ok", err_msg);
        std::cout << "execute_gesture: `ok`, err_msg:" << err_msg <<"\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
        
        /*** test: abort 'ok' ***/
        bc_ptr->execute_gesture(HandSide::RIGHT, "666", err_msg);  
        std::cout << "execute_gesture: `666`, err_msg:" << err_msg <<"\n";

        GestureExecuteInfoVec gs_number_123 {
            {HandSide::RIGHT, "number_1"},
            {HandSide::RIGHT, "number_2"},
            {HandSide::RIGHT, "number_3"},
        };

        bc_ptr->execute_gestures(gs_number_123, err_msg);
        std::cout << "execute_gestures: number 1~3, err_msg:" << err_msg <<"\n";

        /*** test: abort batch gesture ***/
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        bc_ptr->execute_gesture(HandSide::RIGHT, "rock-and-roll", err_msg); 
    }

    /*** test: batch gesture ***/
    {
        std::cout << "------------------ test: batch gesture ------------\n";         
        GestureExecuteInfoVec gests {
            {HandSide::RIGHT, "number_1"},
            {HandSide::RIGHT, "number_2"},
            {HandSide::RIGHT, "number_3"},
            {HandSide::RIGHT, "number_4"},
            {HandSide::RIGHT, "number_5"},
            {HandSide::RIGHT, "number_6"},
            {HandSide::RIGHT, "number_7"},
            {HandSide::RIGHT, "number_8"},
            {HandSide::RIGHT, "number_9"},
        };
        
        std::string err_msg;
        bc_ptr->execute_gestures(gests, err_msg);
        std::cout << "execute_gestures number 1~9, err_msg:" << err_msg <<"\n";

        int count_100ms = 100; 
        while(count_100ms > 0) {
            std::cout <<"gesture executing: " << std::boolalpha << bc_ptr->is_gesture_executing() << "\n";
            count_100ms --;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    /*** test: execute all gestures ***/
    {
        std::cout << "------------------ test: execute all gestures ------------\n";         
        GestureExecuteInfoVec gests;
        for(auto &gs: gestures)
            gests.push_back({HandSide::RIGHT, gs.gesture_name});

        std::string err_msg;
        bc_ptr->execute_gestures(gests, err_msg);
        std::cout << "execute all preset gestures, err_msg:" << err_msg <<"\n";
        std::this_thread::sleep_for(std::chrono::seconds(20)); /* waiting for all gestures to finish */
    }

    bc_ptr->close();
    std::cout << "brainco closed." << std::endl;
    return 0;
}