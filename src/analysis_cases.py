# 算例具备频域数据（_FreqDomain.csv）后再运行此脚本
# 适用于有自由场和表面反射信号的算例
#
# 本脚本用于对旋翼气动噪声进行频域分析，主要包含两大功能：
# 1. 峰值频率分析（PeakFrequenceAnalyze）：
#    - 从频域数据中提取峰值频率（谐波分量）
#    - 分析主谐波的频率及其在自由场（FF）和表面反射（SR）条件下的SPL值
#    - 输出结果保存为 _Harmonics.csv 文件
#
# 2. 频带能量贡献分析（BandCountributionAnalyze）：
#    - 将全频段划分为低频、中频、高频三个频带
#    - 计算各频带的能量分布及占总能量的比例
#    - 识别主导频带（dominant band）
#    - 输出详细频带数据（_BandContribution.csv）和汇总数据（_BandContribution_Summary.csv）

import numpy as np
import pandas as pd

from analyzer.analyzer import PeakFrequencyAnalyzer
from analyzer.analyzer import BandContributionAnalyzer

def PeakFrequenceAnalyze(file_path, filename_prefix):
    for i in range(len(filename_prefix)):
        # 读取频率域线性幅值数据
        freq_domain_data = pd.read_csv(f"{file_path}\\{filename_prefix[i]}_FreqDomain.csv", header=0, sep=',')
        freq = freq_domain_data['Frequency(Hz)']
        amp_merged = freq_domain_data['amp_merged(Pa)']
        SPL_FF = freq_domain_data['SPL_FF(dB)']
        SPL_SR = freq_domain_data['SPL_SR(dB)']
        SPL_merged = freq_domain_data['SPL_merged(dB)']   

        # 峰值分析
        analyzer = PeakFrequencyAnalyzer(freq.values)
        result = analyzer.analyze_spectrum(amp_merged.values, prominence=0.005)
        harmonic_indices = result['harmonic_indices']
        harmonic_freq = freq[harmonic_indices]
        harmonic_SPL_FF = SPL_FF[harmonic_indices]
        harmonic_SPL_SR = SPL_SR[harmonic_indices]
        harmonic_SPL_merged = SPL_merged[harmonic_indices]
        
        # 输出到文件
        output_data = pd.DataFrame({
            'Harmonic Order': np.arange(1, len(harmonic_freq) + 1),
            'Frequency(Hz)': harmonic_freq,
            'SPL_FF(dB)': harmonic_SPL_FF,
            'SPL_SR(dB)': harmonic_SPL_SR,
            'SPL_merged(dB)': harmonic_SPL_merged,
        })
        output_data.to_csv(f"{file_path}\\{filename_prefix[i]}_Harmonics.csv", index=False)

def BandCountributionAnalyze(file_path, filename_prefix, group_prefixes=None):
    summary_list = []
    
    for i in range(len(filename_prefix)):
        # 读取频率域线性幅值数据
        freq_domain_data = pd.read_csv(f"{file_path}\\{filename_prefix[i]}_FreqDomain.csv", header=0, sep=',')
        freq = freq_domain_data['Frequency(Hz)']
        amp_merged = freq_domain_data['amp_merged(Pa)']

        # 带宽分析
        analyzer = BandContributionAnalyzer(freq.values)
        result = analyzer.analyze_band_contribution(amp_merged.values)
        
        if not result:
            continue
            
        # 1. 保存详细频带数据
        band_energies = result['band_energies']
        detail_df = pd.DataFrame({
            'Center Frequency(Hz)': [b['center_freq'] for b in band_energies],
            'Energy': [b['energy'] for b in band_energies],
            'SPL(dB)': [b['energy_dB'] for b in band_energies],
            'Energy Ratio': [b['energy_ratio'] for b in band_energies]
        })
        detail_df.to_csv(f"{file_path}\\{filename_prefix[i]}_BandContribution.csv", index=False)
        
        # 2. 收集汇总数据
        total_energy = result['total_energy']
        energy_dist = result['energy_distribution']
        dominant_band = result['dominant_band']
        
        summary_item = {
            'Filename': filename_prefix[i],
            'Total Energy': total_energy,
            'Dominant Freq(Hz)': dominant_band['center_freq'],
            'Dominant Ratio': dominant_band['energy_ratio'],
            'Low Freq Energy': energy_dist['low_freq'],
            'Mid Freq Energy': energy_dist['mid_freq'],
            'High Freq Energy': energy_dist['high_freq'],
            'Low Freq Ratio': energy_dist['low_freq'] / total_energy if total_energy > 0 else 0,
            'Mid Freq Ratio': energy_dist['mid_freq'] / total_energy if total_energy > 0 else 0,
            'High Freq Ratio': energy_dist['high_freq'] / total_energy if total_energy > 0 else 0
        }
        summary_list.append(summary_item)
    
    # 保存汇总数据
    if summary_list:
        summary_df = pd.DataFrame(summary_list)
        # 调整列顺序
        columns_order = [
            'Filename', 'Total Energy', 'Dominant Freq(Hz)', 'Dominant Ratio',
            'Low Freq Energy', 'Mid Freq Energy', 'High Freq Energy',
            'Low Freq Ratio', 'Mid Freq Ratio', 'High Freq Ratio'
        ]
        summary_df = summary_df[columns_order]
        
        if group_prefixes:
            # 按前缀分组保存
            for prefix in group_prefixes:
                # 筛选属于当前前缀的数据
                # 假设 filename_prefix 是 "{prefix}_OBS..." 格式
                group_df = summary_df[summary_df['Filename'].str.startswith(prefix)]
                if not group_df.empty:
                    group_df.to_csv(f"{file_path}\\{prefix}_BandContribution_Summary.csv", index=False)
        else:
            # 如果没有提供前缀，保存为一个总文件
            summary_df.to_csv(f"{file_path}\\Case04_BandContribution_Summary.csv", index=False)
        

if __name__ == "__main__":

    file_path = r"Case03"
    Filename_list = [
        "Case03_Rotor",
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
    
    # 组合 OBS_Numbers 和 Filename_list 中的元素形成要读取的文件名
    filename_prefix = [
        f"{Filename_list[i]}_OBS{j+1:04d}" for i in range(len(Filename_list)) for j in range(OBS_Numbers)
    ]

    PeakFrequenceAnalyze(file_path, filename_prefix)
    BandCountributionAnalyze(file_path, filename_prefix, group_prefixes=Filename_list)