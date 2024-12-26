#!/bin/bash

# 接收源码目录作为参数
SRC_DIR=$1
# SRC_DIR="/home/fandes/workspace/kuavo-ros-control/src/kuavo_assets"
# 获取机器人版本
if [ -z "${ROBOT_VERSION}" ]; then
    echo -e "\033[31m\nError: ROBOT_VERSION 机器人版本号环境变量未设置！\033[0m" >&2
    echo -e "\033[31m请参考README.md文档, 确认你的机器人版本，通过以下方式设置机器人版本号:\033[0m" >&2
    echo -e "\033[32m1. 在当前终端执行(临时设置): \n
        export ROBOT_VERSION=40\033[0m" >&2 
    echo -e "\033[32m\n2. 将其添加到你的 ~/.bashrc 或者 ~/.zshrc 终端配置文件中:
    如执行: \n
        echo 'export ROBOT_VERSION=40' >> ~/.bashrc \n
    添加到 ~/.bashrc 文件(bash终端)末尾，重启终端后生效\033[0m\n" >&2
    exit 1 
fi

# 允许的机器人版本号列表
allowed_versions=("30" "31" "32" "40" "41" "42" "43")
if ! [[ " ${allowed_versions[@]} " =~ " ${ROBOT_VERSION} " ]]; then
    echo -e "\033[31m\nError: 机器人版本号(环境变量中 ROBOT_VERSION 的值) = '${ROBOT_VERSION}' 无效 \033[0m" >&2
    echo -e "\033[31m请参考readme.md文档，确认你的机器人版本号\n目前可用的版本号有: \n[${allowed_versions[*]}] \033[0m\n" >&2
    exit 1
fi

ROBOT_VERSION=${ROBOT_VERSION}  # 如果环境变量未设置，默认为1

echo $ROBOT_VERSION
# 进入脚本所在目录
cd "$(dirname "$0")"

# 定义URDF文件列表
URDF_FILES=(
    "biped_s${ROBOT_VERSION}.urdf"
    "drake/biped_v3_all_joint.urdf"
    "drake/biped_v3.urdf"
    "drake/biped_v3_full.urdf"
    # 可以继续添加其他URDF文件
)

# 需要修改的link名称
LINK_NAMES=(
    "base_link"
    "torso"
    "torso"
    "torso"
)


# 构建基础URDF路径
BASE_URDF_PATH="${SRC_DIR}/models/biped_s${ROBOT_VERSION}/urdf"

CONFIG_DIR=~/.config/lejuconfig
MASS_FILE=${CONFIG_DIR}/TotalMassV${ROBOT_VERSION}
MASS_BAK=${CONFIG_DIR}/.TotalMassV${ROBOT_VERSION}.bak
OCS2_DIR=/var/ocs2/full_v${ROBOT_VERSION}*

# 确保配置目录存在
mkdir -p ${CONFIG_DIR}

# 如果totalMass不存在，创建文件
if [ ! -f ${MASS_FILE} ]; then
    # 对每个URDF文件执行get_urdf_mass.sh
    for URDF_FILE in "${URDF_FILES[@]}"; do
        FULL_PATH="${BASE_URDF_PATH}/${URDF_FILE}"
        echo $FULL_PATH
        if [ -f "$FULL_PATH" ]; then
            origin_mass=$(./get_urdf_mass.sh "$FULL_PATH")
            echo "get total mass :$origin_mass"
            echo $origin_mass > ${MASS_FILE}
            echo $origin_mass > ${MASS_BAK}
            break  # 只需要从第一个存在的URDF文件获取质量即可
        fi
    done
    rm -rf ${OCS2_DIR}
    echo "Created new TotalMass file"
fi

# 检查备份文件
if [ ! -f ${MASS_BAK} ]; then
    # 没有备份文件，创建备份并删除ocs2目录
    cp ${MASS_FILE} ${MASS_BAK}
    rm -rf ${OCS2_DIR}
    echo "Created backup mass file"
else
    # 比较当前文件和备份文件
    CURRENT_MASS=$(cat ${MASS_FILE})
    BACKUP_MASS=$(cat ${MASS_BAK})
    
    if [ "$CURRENT_MASS" != "$BACKUP_MASS" ]; then
        rm -rf ${OCS2_DIR}
        cp ${MASS_FILE} ${MASS_BAK}
        echo -e "\033[33m\nNote: Total Mass changed, removing ${OCS2_DIR} cppad directory which will be rebuilt on next launch\033[0m" >&2
    fi
fi

# 获取总质量并修改所有URDF
TOTAL_MASS=$(cat ${MASS_FILE})
echo "由配置文件 ${MASS_FILE} 指定总质量为: $TOTAL_MASS kg" >&2
# 遍历URDF_FILES，并同步取出链接名称
for index in "${!URDF_FILES[@]}"; do
    URDF_FILE="${URDF_FILES[$index]}"
    link_name="${LINK_NAMES[$index]}"  # 获取对应的链接名称
    FULL_PATH="${BASE_URDF_PATH}/${URDF_FILE}"

    if [ -f "$FULL_PATH" ]; then
        # 调用修改质量的脚本，同时传入链接名称
        ./modify_torso_mass.sh "$FULL_PATH" "${TOTAL_MASS}" "$link_name"  # 传入link_name作为参数
        echo " Updated mass of ${FULL_PATH} to ${TOTAL_MASS}" >&2
    else
        echo "Warning: ${URDF_FILE} not found" >&2
    fi
done


# 修改mujoco xml文件
BASE_XML_PATH="${SRC_DIR}/models/biped_s${ROBOT_VERSION}/xml"
XML_FILE_PATH="${BASE_XML_PATH}/biped_s${ROBOT_VERSION}.xml"
if [ -f "$XML_FILE_PATH" ]; then
    # 调用修改质量的脚本，同时传入链接名称
    ./modify_torso_mass_xml.sh "$XML_FILE_PATH" "${TOTAL_MASS}"  # 传入link_name作为参数
    echo "Updated xml mass for ${XML_FILE_PATH}"
else
    echo "Warning: ${XML_FILE_PATH} not found" >&2
fi

chmod 0777 ${MASS_FILE} ${MASS_BAK}
