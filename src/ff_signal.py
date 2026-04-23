# 信号分析时首先运行此脚本
# 适用于仅有自由场信号的算例
#
# 本脚本用于处理仅有自由场（Free Field）信号的算例：
# 1. 读取自由场时域信号数据（_FF.csv）
# 2. 频域分析：对信号进行FFT变换，得到幅值和SPL频谱
# 3. 频域输出：保存频域数据到 _FreqDomain.csv（含频率、幅值、SPL）
# 4. 周期SPL计算：计算每个周期的声压级，输出到 _SPLs.csv

import numpy as np
import pandas as pd
from analyzer.utils import rfft, SPLs

if __name__ == "__main__":
    file_path = r"Case01"
    filename_prefix_list = [
        "Case01_Rotor"
    ]   
    OBS_Numbers = 12
    OBS_Position_list = np.array([
        [0, 3, 0], 
        [0, 4, 0],
        [0, 5, 0],
        [0, 3, -0.25],
        [0, 4, -0.25],
        [0, 5, -0.25],
        [0, 3, -0.5],
        [0, 4, -0.5],
        [0, 5, -0.5],
        [0, 3, -0.75],
        [0, 4, -0.75],
        [0, 5, -0.75],
        ])
    if OBS_Numbers != len(OBS_Position_list):
        raise ValueError("OBS_Numbers must be equal to the length of OBS_Position_list")

    cycles = 5

    # 组合 OBS_Numbers 和 Filename_list 中的元素形成要读取的文件名
    Sound_FreeField_file_list = [
        f"{filename_prefix_list[i]}_OBS{j+1:04d}_FF.csv" for i in range(len(filename_prefix_list)) for j in range(OBS_Numbers)
    ]
    print(Sound_FreeField_file_list)

    # 组合 Filename_list 和 OBS_Position_list 中的元素形成输入的 OBS_Position 列表
    OBS_Position_list = np.array([
        OBS_Position_list[i] for j in range(len(filename_prefix_list)) for i in range(OBS_Numbers) 
    ])
    print(OBS_Position_list)

    if len(Sound_FreeField_file_list) != len(OBS_Position_list):
        raise ValueError("Sound_FreeField_file_list and OBS_Position_list must have the same length")
    
    out_prefix = [f"{Sound_FreeField_file_list[i].replace('_FF.csv', '')}" for i in range(len(Sound_FreeField_file_list))]
    SPLs_df = pd.DataFrame(columns=out_prefix)
    SPLs_thickness_df = pd.DataFrame(columns=out_prefix)
    SPLs_load_df = pd.DataFrame(columns=out_prefix)
    for i in range(len(Sound_FreeField_file_list)):
        # 读取 Free Field 数据
        Sound_FF = pd.read_csv(f"{file_path}/{Sound_FreeField_file_list[i]}", sep=",", header=0)
        time_signal_FF = np.array([Sound_FF['Time'], Sound_FF['Total']])
        time_signal_FF_thickness = np.array([Sound_FF['Time'], Sound_FF['Thickness']])
        time_signal_FF_load = np.array([Sound_FF['Time'], Sound_FF['Load']])
        # 计算 Free Field 数据的频域并导出
        freq_FF, _, amp_fft_FF, SPL_fft_FF = rfft(time_signal_FF)
        _, _, amp_fft_FF_thickness, SPL_fft_FF_thickness = rfft(time_signal_FF_thickness)
        _, _, amp_fft_FF_load, SPL_fft_FF_load = rfft(time_signal_FF_load)
        freq_domain_data = np.array([freq_FF, amp_fft_FF, SPL_fft_FF, amp_fft_FF_thickness, SPL_fft_FF_thickness, amp_fft_FF_load, SPL_fft_FF_load]).T
        freq_domain_data = pd.DataFrame(freq_domain_data, columns=['Frequency(Hz)', 'amp_Total(Pa)', 'SPL_Total(dB)', 'amp_Thickness(Pa)', 'SPL_Thickness(dB)', 'amp_Load(Pa)', 'SPL_Load(dB)'])
        freq_domain_data.to_csv(f"{file_path}/{out_prefix[i]}_FreqDomain.csv", index=False)
        print(f"Export freq domain data: {file_path}\\{out_prefix[i]}_FreqDomain.csv")
        # 计算 Free Field 时域数据的声压级
        SPL_array = SPLs(time_signal_FF, cycles)
        SPL_array_thickness = SPLs(time_signal_FF_thickness, cycles)
        SPL_array_load = SPLs(time_signal_FF_load, cycles)
        SPLs_df[out_prefix[i]] = SPL_array
        SPLs_thickness_df[out_prefix[i]] = SPL_array_thickness
        SPLs_load_df[out_prefix[i]] = SPL_array_load
    # 将所有 OBS 的声压级数据
    SPLs_filename_list = [f"{filename_prefix_list[i].split('_')[0]}_SPLs.csv" for i in range(len(filename_prefix_list))]
    for i in range(len(SPLs_filename_list)):
        # 处理Total数据
        data_total = SPLs_df.iloc[:, i*OBS_Numbers:(i+1)*OBS_Numbers]
        data_total.columns = [f"OBS{j+1}_Total" for j in range(OBS_Numbers)]

        # 处理Thickness数据
        data_thickness = SPLs_thickness_df.iloc[:, i*OBS_Numbers:(i+1)*OBS_Numbers]
        data_thickness.columns = [f"OBS{j+1}_Thickness" for j in range(OBS_Numbers)]

        # 处理Load数据
        data_load = SPLs_load_df.iloc[:, i*OBS_Numbers:(i+1)*OBS_Numbers]
        data_load.columns = [f"OBS{j+1}_Load" for j in range(OBS_Numbers)]

        # 合并数据
        data = pd.concat([data_total, data_thickness, data_load], axis=1)
        # 在第一列插入一个名为 Cycles 的列，值为 1到 cycles
        data.insert(0, 'Cycles', range(1, cycles+1))
        # 输出结果，小数点后保留5位
        data = data.round(5)
        data.to_csv(f"{file_path}\\{SPLs_filename_list[i]}", index=False)
        print(f"Export SPLs data: {file_path}\\{SPLs_filename_list[i]}")
