#pragma once

#ifndef HUMANOID_PARAM_INTERFACE_H
#define HUMANOID_PARAM_INTERFACE_H

#include "humanoid_param_interface/common/kuavo_settings.h"
#include "humanoid_param_interface/common/common.h"
#include "humanoid_param_interface/common/utils.h"
#include "humanoid_param_interface/common/json_config_reader.hpp"

namespace HighlyDynamic
{
using vector_t = Eigen::Matrix<double, Eigen::Dynamic, 1>;
class HumanoidParamInterface
    {
    public:
        static HumanoidParamInterface &getInstance(RobotVersion robot_version, bool real, double dt = 0.002);
        static HumanoidParamInterface *getInstancePtr(RobotVersion robot_version, bool real, double dt = 0.002);

        virtual ~HumanoidParamInterface();

        HumanoidParamInterface(const HumanoidParamInterface &) = delete;
        HumanoidParamInterface &operator=(const HumanoidParamInterface &) = delete;
        inline double getTimeStep() const { return dt_; }
        inline const KuavoSettings &getKuavoSettings() const { return kuavo_settings_; }
        JSONConfigReader *getRobotConfig() const { return robot_config_; }
        RobotVersion getRobotVersion() const { return robot_version_; }
    private:
        HumanoidParamInterface(RobotVersion robot_version, bool real, double dt = 0.001);
        static std::shared_ptr<HumanoidParamInterface> instance;
        RobotVersion robot_version_;

    private:
        double dt_;
        bool real_;
        KuavoSettings kuavo_settings_;
        JSONConfigReader *robot_config_;
    };

}
#endif