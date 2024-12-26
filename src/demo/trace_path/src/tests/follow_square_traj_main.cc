#include <ros/ros.h>
#include <iostream>

#include "path_generator.h"
#include "square_path_generator.h"

#include "path_tracer.h"
#include "mpc_path_tracer.h"

using namespace trace_path;
int main (int argc, char **argv)
{
    ros::init (argc, argv, "trace_path_square_demo_node");
    ros::NodeHandle nh;
    
    std::cout <<" \n***********************************\n\n* Follow Square Traj\n";
    std::cout <<" \n***********************************\n";

    // MPC 
    
    // 参考路径中, 点的间隙 = speed * dt  (单位：m)
    SquareParams square_params = {.side_length = 2.0, .speed = 0.25, .dt = 0.1}; 

    // 创建路径生成器
    auto square_gen = PathGeneratorFactory::Create<SquarePathGenerator>(square_params);

    auto mpc_path_tracer = PathTracerFactory::Create<MpcPathTracer>(nh);
    mpc_path_tracer->set_max_linear_velocity(0.20); // 降低速度 0.2 m/s
    mpc_path_tracer->set_max_angular_velocity(M_PI/8);
    
    /* 正方形 */
    mpc_path_tracer->Follow(square_gen);


    return 0;
}