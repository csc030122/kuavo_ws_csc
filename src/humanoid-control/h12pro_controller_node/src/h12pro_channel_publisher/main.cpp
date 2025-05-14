#include <ros/ros.h>
#include <std_msgs/String.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <errno.h>
#include <unistd.h>
#include <vector>
#include <signal.h>  // for signal handling

#include "drivers_sbus.h"

#include <h12pro_controller_node/h12proRemoteControllerChannel.h>

// Signal handler for graceful shutdown
void signalHandler(int signum) {
    ROS_INFO("Interrupt signal (%d) received. Shutting down...", signum);
    // Perform any necessary cleanup here
    ros::shutdown();
}

int limitChannelValue(int value, int min, int max) {
    return std::max(min, std::min(value, max));
}

int main(int argc, char **argv) {
    // Register signal handler
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    ros::init(argc, argv, "h12pro_channel_publisher_node");
    ros::NodeHandle nh;

    while (initSbus() == -1) {
        ROS_ERROR("Sbus init failed, retrying...");
        ros::Duration(1.0).sleep();  // 暂停 1 秒
    }

    ros::Publisher pub_channel = nh.advertise<h12pro_controller_node::h12proRemoteControllerChannel>("h12pro_channel", 1);
    h12pro_controller_node::h12proRemoteControllerChannel channel_msg;
    channel_msg.channels.resize(12);
    ros::Rate rate(250);
    
    while (ros::ok()) {
        recSbusData();
        channel_msg.channels[0] = SbusRxData.channel_1;
        channel_msg.channels[1] = limitChannelValue(SbusRxData.channel_2, 902, 1102);   // 限制站立、下蹲高度（向上提为902，向下蹲为1102）
        channel_msg.channels[2] = SbusRxData.channel_3;
        channel_msg.channels[3] = limitChannelValue(SbusRxData.channel_4, 802, 1202);   // 限制左右平移步伐大小（向左走为802，向右走为1202）
        channel_msg.channels[4] = SbusRxData.channel_5;
        channel_msg.channels[5] = SbusRxData.channel_6;
        channel_msg.channels[6] = SbusRxData.channel_7;
        channel_msg.channels[7] = SbusRxData.channel_8;
        channel_msg.channels[8] = SbusRxData.channel_9;
        channel_msg.channels[9] = SbusRxData.channel_10;
        channel_msg.channels[10] = SbusRxData.channel_11;
        channel_msg.channels[11] = SbusRxData.channel_12;
        channel_msg.sbus_state = SbusRxData.sbus_state;
        pub_channel.publish(channel_msg);
        rate.sleep();
    }

    return 0;
}
