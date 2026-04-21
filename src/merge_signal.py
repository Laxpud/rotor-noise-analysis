# 信号分析时首先运行此脚本
# 适用于有自由场和表面反射信号的算例
#
# 本脚本用于将自由场（Free Field）和表面反射（Surface Reflection）信号进行合并处理：
# 1. 信号合并：将FF和SR信号叠加并去除直流分量，得到合并信号
# 2. 频域分析：对FF、SR、merged三种信号分别进行FFT变换
# 3. 时域输出：保存三种信号的时域数据到 _TimeDomain.csv
# 4. 频域输出：保存三种信号的频域数据（SPL和幅值）到 _FreqDomain.csv
# 5. 周期SPL计算：计算合并信号每个周期的声压级，输出到 _SPLs_merged.csv

import numpy as np
import pandas as pd
from analyzer.utils import SPLs, rfft

def main(
    file_path, 
    Sound_FreeField_file_list, 
    Sound_SurfaceReflection_file_list, 
    OBS_Position_list, 
    cycles
    ):
    """
    
    """
    print(f"File_path: {file_path}")

    if len(Sound_FreeField_file_list) != len(Sound_SurfaceReflection_file_list):
        raise ValueError("Sound_FreeField_file_list and Sound_SurfaceReflection_file_list must have the same length.")
    if len(Sound_FreeField_file_list) != len(OBS_Position_list):
        raise ValueError("Sound_FreeField_file_list and OBS_Position_list must have the same length.")
    
    print(f'Total files number: {len(Sound_FreeField_file_list)}')
    # 创建输出文件名的前缀
    out_prefix = [f"{Sound_FreeField_file_list[i].replace('_FF.csv', '')}" for i in range(len(Sound_FreeField_file_list))]
    # 创建DataFrame来存储各个分量的SPLs
    SPLs_merged_df = pd.DataFrame(columns=out_prefix)
    SPLs_FF_total_df = pd.DataFrame(columns=out_prefix)
    SPLs_FF_thickness_df = pd.DataFrame(columns=out_prefix)
    SPLs_FF_load_df = pd.DataFrame(columns=out_prefix)
    SPLs_SR_total_df = pd.DataFrame(columns=out_prefix)
    SPLs_SR_thickness_df = pd.DataFrame(columns=out_prefix)
    SPLs_SR_load_df = pd.DataFrame(columns=out_prefix)
    SPLs_merged_thickness_df = pd.DataFrame(columns=out_prefix)
    SPLs_merged_load_df = pd.DataFrame(columns=out_prefix)
    for i in range(len(Sound_FreeField_file_list)):
        print(f"\nFile index: {i}")
        OBS_Position = OBS_Position_list[i]
        print(f"OBS_Position: {OBS_Position}")
        Sound_FF = pd.read_csv(f"{file_path}\\{Sound_FreeField_file_list[i]}", header=0)
        print(f"Read sound_FF_file: {Sound_FreeField_file_list[i]}")
        Sound_SR = pd.read_csv(f"{file_path}\\{Sound_SurfaceReflection_file_list[i]}", header=0)
        print(f"Read sound_SR_file: {Sound_SurfaceReflection_file_list[i]}")

        # 将dataframe转换为nparray
        time_signal_FF = np.array([Sound_FF['Time'], Sound_FF['Total']])
        time_signal_SR = np.array([Sound_SR['Time'], Sound_SR['Total']])

        # 读取Thickness和Load分量
        time_signal_FF_thickness = np.array([Sound_FF['Time'], Sound_FF['Thickness']])
        time_signal_FF_load = np.array([Sound_FF['Time'], Sound_FF['Load']])
        time_signal_SR_thickness = np.array([Sound_SR['Time'], Sound_SR['Thickness']])
        time_signal_SR_load = np.array([Sound_SR['Time'], Sound_SR['Load']])

        print("Merge free field and surface reflection")
        time_signal_merged = np.array([
            time_signal_SR[0],
            time_signal_FF[1]+time_signal_SR[1]-np.mean(time_signal_FF[1]+time_signal_SR[1])
            ])

        # 合并Thickness和Load分量
        merged_thickness = time_signal_FF_thickness[1] + time_signal_SR_thickness[1] - np.mean(time_signal_FF_thickness[1] + time_signal_SR_thickness[1])
        time_signal_merged_thickness = np.array([time_signal_SR[0], merged_thickness])

        merged_load = time_signal_FF_load[1] + time_signal_SR_load[1] - np.mean(time_signal_FF_load[1] + time_signal_SR_load[1])
        time_signal_merged_load = np.array([time_signal_SR[0], merged_load])
        # 获得time_signal_merged的频域
        if len(time_signal_merged[0]) != len(time_signal_FF[0]) or len(time_signal_merged[0]) != len(time_signal_SR[0]): 
            raise ValueError("Time signal of free field and merged do not match.")
        N = len(time_signal_merged[0])
        
        # 计算所有数据的频域
        freq_FF, _, amp_fft_FF, SPL_fft_FF = rfft(time_signal_FF)
        freq_SR, _, amp_fft_SR, SPL_fft_SR = rfft(time_signal_SR)
        freq_merged, _, amp_fft_merged, SPL_fft_merged = rfft(time_signal_merged)

        # 计算Thickness分量的频域
        _, _, amp_fft_FF_thickness, SPL_fft_FF_thickness = rfft(time_signal_FF_thickness)
        _, _, amp_fft_SR_thickness, SPL_fft_SR_thickness = rfft(time_signal_SR_thickness)
        _, _, amp_fft_merged_thickness, SPL_fft_merged_thickness = rfft(time_signal_merged_thickness)

        # 计算Load分量的频域
        _, _, amp_fft_FF_load, SPL_fft_FF_load = rfft(time_signal_FF_load)
        _, _, amp_fft_SR_load, SPL_fft_SR_load = rfft(time_signal_SR_load)
        _, _, amp_fft_merged_load, SPL_fft_merged_load = rfft(time_signal_merged_load)
    
        # 比对所有频域数据的频率是否一致
        if not np.allclose(freq_FF, freq_SR):
            raise ValueError("Frequency of free field and surface reflection do not match.")
        if not np.allclose(freq_FF, freq_merged):
            raise ValueError("Frequency of free field and merged do not match.") 

        # 将频域结果输出到一个csv文件
        freq_domain_data = np.array([
            freq_FF,
            SPL_fft_FF,
            SPL_fft_SR,
            SPL_fft_merged,
            amp_fft_FF,
            amp_fft_SR,
            amp_fft_merged,
            # Thickness相关列
            SPL_fft_FF_thickness,
            SPL_fft_SR_thickness,
            SPL_fft_merged_thickness,
            amp_fft_FF_thickness,
            amp_fft_SR_thickness,
            amp_fft_merged_thickness,
            # Load相关列
            SPL_fft_FF_load,
            SPL_fft_SR_load,
            SPL_fft_merged_load,
            amp_fft_FF_load,
            amp_fft_SR_load,
            amp_fft_merged_load,
            ]).T
        freq_domain_data = pd.DataFrame(freq_domain_data, columns=[
            'Frequency(Hz)', 'SPL_FF_Total(dB)', 'SPL_SR_Total(dB)', 'SPL_merged_Total(dB)',
            'amp_FF_Total(Pa)', 'amp_SR_Total(Pa)', 'amp_merged_Total(Pa)',
            # Thickness相关列
            'SPL_FF_Thickness(dB)', 'SPL_SR_Thickness(dB)', 'SPL_merged_Thickness(dB)',
            'amp_FF_Thickness(Pa)', 'amp_SR_Thickness(Pa)', 'amp_merged_Thickness(Pa)',
            # Load相关列
            'SPL_FF_Load(dB)', 'SPL_SR_Load(dB)', 'SPL_merged_Load(dB)',
            'amp_FF_Load(Pa)', 'amp_SR_Load(Pa)', 'amp_merged_Load(Pa)'
        ])
        freq_domain_data.to_csv(f"{file_path}\\{out_prefix[i]}_FreqDomain.csv", index=False)
        print(f"Export freq domain data: {file_path}\\{out_prefix[i]}_FreqDomain.csv")
        
        # 将时域结果输出到一个csv中
        time_domain_data = np.array([
            time_signal_FF[0],
            time_signal_FF[1],
            time_signal_SR[1],
            time_signal_merged[1],
            # 新增Thickness和Load列
            time_signal_FF_thickness[1],
            time_signal_FF_load[1],
            time_signal_SR_thickness[1],
            time_signal_SR_load[1],
            time_signal_merged_thickness[1],
            time_signal_merged_load[1]
            ]).T
        time_domain_data = pd.DataFrame(time_domain_data, columns=[
            'Time(ms)', 'FF_Total(Pa)', 'SR_Total(Pa)', 'merged_Total(Pa)',
            'FF_Thickness(Pa)', 'FF_Load(Pa)',
            'SR_Thickness(Pa)', 'SR_Load(Pa)',
            'merged_Thickness(Pa)', 'merged_Load(Pa)'
        ])
        time_domain_data.to_csv(f"{file_path}\\{out_prefix[i]}_TimeDomain.csv", index=False)
        print(f"Export time domain data: {file_path}\\{out_prefix[i]}_TimeDomain.csv")
        
        # 计算每个周期的声压级
        SPLs_merged_df[out_prefix[i]] = SPLs(time_signal_merged, cycles)
        SPL_FF_total = SPLs(time_signal_FF, cycles)
        SPL_FF_thickness = SPLs(time_signal_FF_thickness, cycles)
        SPL_FF_load = SPLs(time_signal_FF_load, cycles)
        SPL_SR_total = SPLs(time_signal_SR, cycles)
        SPL_SR_thickness = SPLs(time_signal_SR_thickness, cycles)
        SPL_SR_load = SPLs(time_signal_SR_load, cycles)
        SPL_merged_thickness = SPLs(time_signal_merged_thickness, cycles)
        SPL_merged_load = SPLs(time_signal_merged_load, cycles)

        SPLs_FF_total_df[out_prefix[i]] = SPL_FF_total
        SPLs_FF_thickness_df[out_prefix[i]] = SPL_FF_thickness
        SPLs_FF_load_df[out_prefix[i]] = SPL_FF_load
        SPLs_SR_total_df[out_prefix[i]] = SPL_SR_total
        SPLs_SR_thickness_df[out_prefix[i]] = SPL_SR_thickness
        SPLs_SR_load_df[out_prefix[i]] = SPL_SR_load
        SPLs_merged_thickness_df[out_prefix[i]] = SPL_merged_thickness
        SPLs_merged_load_df[out_prefix[i]] = SPL_merged_load
        
    print("\n", SPLs_merged_df)
    return (time_signal_merged, freq_domain_data, time_domain_data,
            SPLs_merged_df, SPLs_FF_total_df, SPLs_FF_thickness_df, SPLs_FF_load_df,
            SPLs_SR_total_df, SPLs_SR_thickness_df, SPLs_SR_load_df,
            SPLs_merged_thickness_df, SPLs_merged_load_df)

if __name__ == "__main__":
        
    file_path = r"Case05"
    file_prefix_list = [
        "Case05_Rotor",
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
        f"{file_prefix_list[i]}_OBS{j+1:04d}_FF.csv" for i in range(len(file_prefix_list)) for j in range(OBS_Numbers)
    ]
    print(Sound_FreeField_file_list)
    Sound_SurfaceReflection_file_list = [
        f"{file_prefix_list[i]}_OBS{j+1:04d}_SR.csv" for i in range(len(file_prefix_list)) for j in range(OBS_Numbers)
    ]
    print(Sound_SurfaceReflection_file_list)
    # 组合 Filename_list 和 OBS_Position_list 中的元素形成输入的 OBS_Position 列表
    OBS_Position_list = np.array([
        OBS_Position_list[i] for j in range(len(file_prefix_list)) for i in range(OBS_Numbers) 
    ])
    print(OBS_Position_list)

    # 调用主函数
    (time_signal_merged, freq_domain_data, time_domain_data,
     SPLs_merged_df, SPLs_FF_total_df, SPLs_FF_thickness_df, SPLs_FF_load_df,
     SPLs_SR_total_df, SPLs_SR_thickness_df, SPLs_SR_load_df,
     SPLs_merged_thickness_df, SPLs_merged_load_df) = main(
        file_path,
        Sound_FreeField_file_list,
        Sound_SurfaceReflection_file_list,
        OBS_Position_list,
        cycles
        )
    # 按 Filename_list 输出 SPLs_merged
    SPLs_merged_filename_list = [f"{file_prefix_list[i].split('_')[0]}_SPLs_merged.csv" for i in range(len(file_prefix_list))]
    # 将 SPLs_merged_df 中的数据分组并输出，分组数量为SPLs_merged_filename_list中的元素数量
    for i in range(len(SPLs_merged_filename_list)):
        # 处理merged_Total数据
        data_merged_total = SPLs_merged_df.iloc[:, i*OBS_Numbers:(i+1)*OBS_Numbers]
        data_merged_total.columns = [f"OBS{j+1}_merged_Total" for j in range(OBS_Numbers)]

        # 处理FF分量数据
        data_FF_total = SPLs_FF_total_df.iloc[:, i*OBS_Numbers:(i+1)*OBS_Numbers]
        data_FF_total.columns = [f"OBS{j+1}_FF_Total" for j in range(OBS_Numbers)]
        data_FF_thickness = SPLs_FF_thickness_df.iloc[:, i*OBS_Numbers:(i+1)*OBS_Numbers]
        data_FF_thickness.columns = [f"OBS{j+1}_FF_Thickness" for j in range(OBS_Numbers)]
        data_FF_load = SPLs_FF_load_df.iloc[:, i*OBS_Numbers:(i+1)*OBS_Numbers]
        data_FF_load.columns = [f"OBS{j+1}_FF_Load" for j in range(OBS_Numbers)]

        # 处理SR分量数据
        data_SR_total = SPLs_SR_total_df.iloc[:, i*OBS_Numbers:(i+1)*OBS_Numbers]
        data_SR_total.columns = [f"OBS{j+1}_SR_Total" for j in range(OBS_Numbers)]
        data_SR_thickness = SPLs_SR_thickness_df.iloc[:, i*OBS_Numbers:(i+1)*OBS_Numbers]
        data_SR_thickness.columns = [f"OBS{j+1}_SR_Thickness" for j in range(OBS_Numbers)]
        data_SR_load = SPLs_SR_load_df.iloc[:, i*OBS_Numbers:(i+1)*OBS_Numbers]
        data_SR_load.columns = [f"OBS{j+1}_SR_Load" for j in range(OBS_Numbers)]

        # 处理merged其他分量数据
        data_merged_thickness = SPLs_merged_thickness_df.iloc[:, i*OBS_Numbers:(i+1)*OBS_Numbers]
        data_merged_thickness.columns = [f"OBS{j+1}_merged_Thickness" for j in range(OBS_Numbers)]
        data_merged_load = SPLs_merged_load_df.iloc[:, i*OBS_Numbers:(i+1)*OBS_Numbers]
        data_merged_load.columns = [f"OBS{j+1}_merged_Load" for j in range(OBS_Numbers)]

        # 合并所有数据
        data = pd.concat([
            data_merged_total,
            data_FF_total, data_FF_thickness, data_FF_load,
            data_SR_total, data_SR_thickness, data_SR_load,
            data_merged_thickness, data_merged_load
        ], axis=1)
        # 在第一列插入一个名为 Cycles 的列，值为 1到 cycles
        data.insert(0, 'Cycles', range(1, cycles+1))
        # 输出结果，小数点后保留5位
        data = data.round(5)
        data.to_csv(f"{file_path}\\{SPLs_merged_filename_list[i]}", index=False)
        print(f"Export SPLs_merged data: {file_path}\\{SPLs_merged_filename_list[i]}")
