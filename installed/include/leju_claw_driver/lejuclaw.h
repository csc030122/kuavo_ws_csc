#ifndef LEJUCLAW_CPP_IMPL_H
#define LEJUCLAW_CPP_IMPL_H
#define POSITION_TOLERANCE 1
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <algorithm>
#include <variant>
#include <yaml-cpp/yaml.h>
#include <cmath>
#include <fstream>
#include <pwd.h>
#include <unistd.h>
#include <iostream>
#include <iterator>
#include <time.h>
#include <chrono>
#include "ruiwoSDK.h"
#include <algorithm>

class LeJuClaw
{
//#define ENABLE_JOINT_DEBUG  // 取消注释此行以启用关节调试打印
 public:
    
     enum class PawMoveState  : int8_t {
        ERROR = -1,                       // 出现错误
        LEFT_REACHED_RIGHT_REACHED = 0,   // 所有夹爪到位
        LEFT_REACHED_RIGHT_GRABBED = 1,   // 左夹爪到位，右夹爪抓取到物品
        LEFT_GRABBED_RIGHT_REACHED = 2,   // 右夹爪到位，左夹爪抓取到物品
        LEFT_GRABBED_RIGHT_GRABBED = 3,   // 所有夹爪均夹取
        
    };

    enum class State {
        None,
        Enabled,
        Disabled
    };

    struct MotorStateData {
        uint8_t id;
        State   state;
        MotorStateData():id(0x0), state(State::None) {}
        MotorStateData(uint8_t id_, State state_):id(id_), state(state_) {}
    };    
    using MotorStateDataVec = std::vector<MotorStateData>;
public:
    LeJuClaw(std::string unused = "");
    ~LeJuClaw();
    int initialize(bool init_bmlib);
    void enable();
    void disable();
    void set_zero();
    void go_to_zero();
    void go_to_zero_with_current_control();
    bool find_claw_limit_velocity_control(bool is_open_direction, float kp, float kd, float alpha,
                            float max_current, float stall_current_threshold, 
                            float stall_velocity_threshold, float dt, 
                            int stable_time_ms, int timeout_ms);
    void set_positions(const std::vector<uint8_t> &index, const std::vector<double> &positions, const std::vector<double> &torque, const std::vector<double> &velocity);
    void set_torque(const std::vector<uint8_t> &index, const std::vector<double> &torque);
    void set_velocity(const std::vector<uint8_t> &index, const std::vector<double> &velocity);
    void set_joint_state(int index, const std::vector<float> &state);
    PawMoveState move_paw(const std::vector<double> &positions, const std::vector<double> &velocity,const std::vector<double> &torque);
    std::vector<std::vector<float>> get_joint_state();
    std::vector<double> get_positions();
    std::vector<double> get_torque();
    std::vector<double> get_velocity();
    void close();
    MotorStateDataVec get_motor_state();
    std::vector<int> disable_joint_ids;
    
    void set_claw_kpkd(int kp, int kd);
    float lowPassFilter(float input, float prevOutput, float alpha);
    void clear_all_torque();

private:
    // 控制参数常量
    static constexpr float DEFAULT_KP = 7.00f;                     // 比例增益系数，用于位置控制
    static constexpr float DEFAULT_KD = 2.50f;                     // 微分增益系数，用于阻尼控制
    static constexpr float DEFAULT_ALPHA = 0.20f;                  // 低通滤波器系数，用于信号平滑
    static constexpr float DEFAULT_MAX_CURRENT = 1.80f;            // 最大电流限制，单位 A
    static constexpr float DEFAULT_MIN_ERROR = 0.10f;              // 最小误差阈值，用于判断到位精度
    static constexpr float DEFAULT_DT = 0.002f;                    // 控制周期，单位 s
    // 卡死检测参数
    static constexpr float STUCK_DETECTION_DELAY_MS = 50.0f;       // 卡死检测延迟时间，单位 ms
    static constexpr float STUCK_DETECTION_TIME_MS = 50.0f;        // 卡死检测持续时间，单位 ms
    static constexpr float STUCK_POSITION_THRESHOLD = 0.002f;      // 卡死位置阈值，单位 rad
    static constexpr float IMPACT_CURRENT = 3.0f;                  // 冲击电流阈值，单位 A
    static constexpr float IMPACT_DURATION_MS = 200.0f;            // 冲击持续时间，单位 ms
    static constexpr float IMPACT_INTERVAL_MS = 100.0f;            // 冲击间隔时间，单位 ms
    static constexpr float LIMIT_RANGE_PERCENT = 5.0f;             // 限位范围百分比，在此范围内才执行3A反冲，单位 %
    // 稳定检测参数
    static constexpr float STABLE_POSITION_THRESHOLD = 0.005f;     // 稳定位置阈值，单位 rad
    static constexpr float STABLE_VELOCITY_THRESHOLD = 0.1f;       // 稳定速度阈值，单位 rad/s
    static constexpr float STABLE_DETECTION_TIME_MS = 50.0f;       // 稳定检测时间，单位 ms
    
    // 初始化寻找零点参数
    static constexpr float ZERO_CONTROL_KP = 0.0f;                  // 零点控制比例增益，零点寻找时使用
    static constexpr float ZERO_CONTROL_KD = 1.0f;                  // 零点控制微分增益，零点寻找时使用
    static constexpr float ZERO_CONTROL_ALPHA = 1.50f;              // 零点控制低通滤波系数
    static constexpr float ZERO_CONTROL_MAX_CURRENT = 1.80f;        // 零点控制最大电流限制，单位 A
    static constexpr float ZERO_CONTROL_DT = 0.001f;                // 零点控制周期，单位 s
    static constexpr int ZERO_STABLE_TIME_MS = 2000;                // 零点稳定时间，单位 ms
    static constexpr int ZERO_FIND_TIMEOUT_MS = 6000;               // 零点寻找超时时间，单位 ms
    static constexpr float STALL_CURRENT_THRESHOLD = 1.50f;         // 堵转电流阈值，超过阈值后认为到达限位，单位 A
    static constexpr float STALL_VELOCITY_THRESHOLD = 0.80f;        // 堵转速度阈值，超过阈值后认为到达限位，单位 rad/s
    static constexpr int ZERO_WAIT_MS = 500;                        // 零点等待时间，单位 ms
    static constexpr float OPEN_LIMIT_ADJUSTMENT = -10.0f;          // 开限位调整值，百分比，单位 rad
    static constexpr float CLOSE_LIMIT_ADJUSTMENT = 0.0f;           // 关限位调整值，百分比，单位 rad
    static constexpr float TARGET_VELOCITY = 65.0f;                 // 限位寻找目标速度，单位 rad/s
    // 关爪限位寻找参数
    static constexpr float CLOSE_STUCK_DETECTION_THRESHOLD = 0.02f; // 关爪卡死检测阈值，单位 rad
    static constexpr int CLOSE_NORMAL_MODE_DELAY_MS = 1000;         // 关爪正常模式延迟时间，单位 ms
    static constexpr int CLOSE_IMPACT_DURATION_MS = 200;            // 关爪冲击持续时间，单位 ms
    static constexpr int CLOSE_IMPACT_INTERVAL_MS = 100;            // 关爪冲击间隔时间，单位 ms
    static constexpr int CLOSE_MAX_ATTEMPTS = 5;                    // 关爪最大尝试次数
    
    std::vector<float> create_zero_vector(size_t size);
    void send_torque_with_lock(const std::vector<float>& torques);
    void send_claw_torque_only(const std::vector<float>& torques);  // 发送夹爪电流，左右独立控
    
    private:
    void control_thread();
    // void get_parameter();
    void get_config(const std::string &config_file);
    std::string get_home_path();
    void interpolate_move(const std::vector<float> &start_positions, const std::vector<float> &target_positions, const std::vector<float> &speeds, float dt);
    std::vector<std::vector<float>> interpolate_positions_with_speed(const std::vector<float> &a, const std::vector<float> &b, const std::vector<float> &speeds, float dt);
    void send_positions(const std::vector<int> &index, const std::vector<float> &pos, const std::vector<float> &torque, const std::vector<float> &velocity);
    std::vector<int> get_joint_addresses(const YAML::Node &config, const std::string &joint_type, int count);
    std::vector<bool> get_joint_online_status(const YAML::Node &config, const std::string &joint_type, int count);
    std::vector<std::vector<int>> get_joint_parameters(const YAML::Node &config, const std::string &joint_type, int count);
    // 关节ID
    std::vector<int> Claw_joint_address = {0x0F, 0x10};
    
    static inline const std::vector<std::vector<int>> Claw_parameter = {{0, 10, 2, 0, 0, 0, 0},
                                                                        {0, 10, 2, 0, 0, 0, 0}};

                                                                           
    // 反转电机ID 可能会用上?
    static inline  std::vector<int> Negtive_joint_address_list = {};
    
    // // ptm:力控模式 servo:伺服模式
    static inline  std::string Control_mode = "ptm";
    
    // 3A电流使用统计
    static inline int three_amp_current_usage_count = 0;

    
    std::vector<bool> Claw_joint_online;

    std::vector<std::vector<int>> Joint_parameter_list;
    std::vector<int> Joint_address_list;
    std::vector<bool> Joint_online_list;

    // RUIWOTools ruiwo;
    bool thread_running;
    bool thread_end;
    std::thread control_thread_;
    std::mutex sendpos_lock;
    std::mutex recvpos_lock;
    std::mutex sendvel_lock;
    std::mutex recvvel_lock;
    std::mutex sendtor_lock;
    std::mutex recvtor_lock;
    std::mutex state_lock;
    std::mutex update_lock;
    std::mutex can_lock;

    bool target_update;
    std::vector<float> target_positions;
    std::vector<float> target_velocity;
    std::vector<float> target_torque;
    std::vector<int> target_pos_kp;
    std::vector<int> target_pos_kd;
    std::vector<int> target_vel_kp;
    std::vector<int> target_vel_kd;
    std::vector<int> target_vel_ki;

    std::vector<float> old_target_positions;
    std::vector<float> current_positions;
    std::vector<float> current_torque;
    std::vector<float> current_velocity;
    std::vector<std::vector<float>> joint_status;
    std::vector<float> joint_start_positions;  // 行程起点位置
    std::vector<float> joint_end_positions;    // 行程终点位置
};

#endif // LEJUCLAW_CPP_H