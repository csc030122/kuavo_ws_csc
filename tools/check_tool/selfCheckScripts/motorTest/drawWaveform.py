import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import re

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    data = np.loadtxt(file_path)
    valid_mask = data[:, 0] != 0
    timestamps = data[valid_mask, 0]
    values = data[valid_mask, 1]
    if len(timestamps) == 0:
        raise ValueError("文件中无有效数据（时间戳全为0）")
    return timestamps, values

def plot_input_response(input_file, response_file, index, motor_id):
    try:
        input_ts, input_val = load_data(input_file)
        resp_ts, resp_val = load_data(response_file)
        
        plt.figure(figsize=(12, 6))
        plt.plot(input_ts, input_val, 'b-', label='Input')
        plt.plot(resp_ts, resp_val, 'r--', label='Response')
        plt.title(f'Motor {motor_id}')
        plt.xlabel('Time (ms)' if input_ts.dtype == np.int64 else 'Time (s)')
        plt.ylabel('Signal Value')
        plt.grid(True)
        plt.legend()
        
        # 修改保存路径
        os.makedirs('./image', exist_ok=True)
        plt.savefig(f'./image/motor_{motor_id}.png')
        print(f"成功绘制: 第 {index} 组 Motor {motor_id}，数据点数量: {len(input_ts)}")
        plt.close()
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
    except ValueError as e:
        print(f"警告: {e}")
    except Exception as e:
        print(f"处理数据时出错: {e}")

def main():
    # 修改数据目录为/file
    data_dir = './file'
    
    input_pattern = re.compile(r'inputData_(\d+)_(\d+)\.txt')
    response_pattern = re.compile(r'responseData_(\d+)_(\d+)\.txt')
    
    input_files = {}
    response_files = {}
    
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        input_match = input_pattern.search(file)
        if input_match:
            index = int(input_match.group(1))
            motor_id = int(input_match.group(2))
            input_files[(index, motor_id)] = file_path
        
        response_match = response_pattern.search(file)
        if response_match:
            index = int(response_match.group(1))
            motor_id = int(response_match.group(2))
            response_files[(index, motor_id)] = file_path
    
    all_pairs = set(input_files.keys()).union(set(response_files.keys()))
    
    for (index, motor_id) in all_pairs:
        key = (index, motor_id)
        input_file = input_files.get(key)
        response_file = response_files.get(key)
        
        if not input_file or not response_file:
            print(f"警告: Motor {motor_id} 文件不完整，跳过")
            continue
        
        plot_input_response(input_file, response_file, index, motor_id)

if __name__ == "__main__":
    main()
