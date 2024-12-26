#include <ros/ros.h>
#include <iostream>

#include "path_generator.h"
#include "s_curve_path_generator.h"

#include "path_tracer.h"
#include "mpc_path_tracer.h"

using namespace trace_path;
int main (int argc, char **argv)
{
    ros::init (argc, argv, "trace_path_s_curve_demo_node");
    ros::NodeHandle nh;
    
    std::cout <<" \n***********************************\n\n* Follow S Curve Traj\n";
    std::cout <<" \n***********************************\n";

    // MPC 
    
    // 参考路径中, 点的间隙 = speed * dt  (单位：m)
    SCurveParams s_curve_params = {.length = 2.5, .amplitude = 1.25, .speed = 0.25, .dt = 0.1};
    // SCurveParams s_curve_params = {.length = 3.5, .amplitude = 1.25, .speed = 0.25, .dt = 0.1};

    // 创建路径生成器
    auto s_curve_gen = PathGeneratorFactory::Create<SCurvePathGenerator>(s_curve_params);

    auto mpc_path_tracer = PathTracerFactory::Create<MpcPathTracer>(nh);
    mpc_path_tracer->set_max_linear_velocity(0.20); // 降低速度 0.2 m/s
    mpc_path_tracer->set_max_angular_velocity(M_PI/8);
    
    /* "S" 曲线  */
    mpc_path_tracer->Follow(s_curve_gen); // "S" 曲线 



    return 0;
}