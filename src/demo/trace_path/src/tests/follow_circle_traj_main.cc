#include <ros/ros.h>
#include <iostream>

#include "path_generator.h"
#include "circle_path_generator.h"

#include "path_tracer.h"
#include "mpc_path_tracer.h"

using namespace trace_path;
int main (int argc, char **argv)
{
    ros::init (argc, argv, "trace_path_circle_demo_node");
    ros::NodeHandle nh;
    
    std::cout <<" \n***********************************\n\n* Follow Circle Traj\n";
    std::cout <<" \n***********************************\n";

    // MPC 
    
    // 参考路径中, 点的间隙 = speed * dt  (单位：m)
    const double kRadius = 1.0;  // 1.0 m
    CircleParams circle_params = {.radius = kRadius, .speed = 0.25,.dt = 0.1};

    // 创建路径生成器
    auto circle_gen = PathGeneratorFactory::Create<CirclePathGenerator>(circle_params);

    auto mpc_path_tracer = PathTracerFactory::Create<MpcPathTracer>(nh);
    mpc_path_tracer->set_max_linear_velocity(0.20);     // x 速度 0.2 m/s
    mpc_path_tracer->set_max_angular_velocity(M_PI/8);  // yaw 速度 0.2 m/s

    /* 圆形 */
    mpc_path_tracer->Follow(circle_gen);

    return 0;
}