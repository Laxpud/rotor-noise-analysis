"""
频谱分析 -- 自由场+地面反射：峰值频率/谐波识别以及倍频程带能量贡献分析。

本模块基于 FF+SR 预处理阶段预先计算的频域数据（``_FreqDomain.csv``）运行，
提供两个分析流程：

* ``run_peak_analysis``  -- 使用合并 Total 谱识别谐波（峰值）频率，并收集
  FF、SR 和合并分量的 SPL 值。
* ``run_band_analysis``  -- 将合并分量谱划分为倍频程/分数倍频程带，并计算
  能量贡献指标。

输入
------
* ``{prefix}_FreqDomain.csv`` -- 来自 FF+SR 预处理阶段，包含 FF、SR 和合并
  幅值/SPL 列。

输出
-------
* ``{prefix}_Harmonics.csv``              -- 谐波阶次 SPL 表（FF、SR、合并）。
* ``{prefix}_BandContribution.csv``       -- 各频带的能量和 SPL 详情。
* ``{prefix}_BandContribution_Summary.csv`` -- 按组汇总的综合摘要。

使用方式
-----------
* 直接运行  : ``python spectral_ffsr.py``
* 命令行调度:
  - ``python main.py peak ...``
  - ``python main.py band ...``
"""

import os
import sys

# 确保上级 ``src`` 包在导入路径中。
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
from spectral import PeakFrequencyAnalyzer, BandContributionAnalyzer


def run_peak_analysis(file_path, filename_prefix):
    """从 FF+SR 谱中识别谐波（峰值）频率及其 SPL 值。

    对于每个观测点，扫描合并 Total 幅值谱以寻找峰值。然后使用检测到的谐波
    索引提取所有九种信号/分量组合（FF/SR/合并 × Total/Thickness/Load）的
    SPL 值。

    Parameters
    ----------
    file_path : str
        包含 ``_FreqDomain.csv`` 文件并接收输出文件的目录。
    filename_prefix : list of str
        观测点文件名前缀列表。

    Returns
    -------
    None
        为每个观测点写入 ``{prefix}_Harmonics.csv``。

    Notes
    -----
    谐波检测基于合并 Total 幅值，它代表 FF 和 SR 的相干叠加。突出度阈值固定为 0.005。
    """
    for i in range(len(filename_prefix)):
        # ---- 加载频域数据 ----
        data = pd.read_csv(f"{file_path}\\{filename_prefix[i]}_FreqDomain.csv", header=0, sep=',')
        freq = data['Frequency(Hz)']
        amp_merged_Total = data['amp_merged_Total(Pa)']
        SPL_FF_Total = data['SPL_FF_Total(dB)']
        SPL_SR_Total = data['SPL_SR_Total(dB)']
        SPL_merged_Total = data['SPL_merged_Total(dB)']
        SPL_FF_Thickness = data['SPL_FF_Thickness(dB)']
        SPL_SR_Thickness = data['SPL_SR_Thickness(dB)']
        SPL_merged_Thickness = data['SPL_merged_Thickness(dB)']
        SPL_FF_Load = data['SPL_FF_Load(dB)']
        SPL_SR_Load = data['SPL_SR_Load(dB)']
        SPL_merged_Load = data['SPL_merged_Load(dB)']

        # ---- 在合并 Total 幅值上分析峰值 ----
        analyzer = PeakFrequencyAnalyzer(freq.values)
        result = analyzer.analyze_spectrum(amp_merged_Total.values, prominence=0.005)
        harmonic_indices = result['harmonic_indices']
        # 提取所有信号/分量组合的谐波频率数据。
        harmonic_freq = freq[harmonic_indices]
        harmonic_SPL_FF_Total = SPL_FF_Total[harmonic_indices]
        harmonic_SPL_SR_Total = SPL_SR_Total[harmonic_indices]
        harmonic_SPL_merged_Total = SPL_merged_Total[harmonic_indices]
        harmonic_SPL_FF_Thickness = SPL_FF_Thickness[harmonic_indices]
        harmonic_SPL_SR_Thickness = SPL_SR_Thickness[harmonic_indices]
        harmonic_SPL_merged_Thickness = SPL_merged_Thickness[harmonic_indices]
        harmonic_SPL_FF_Load = SPL_FF_Load[harmonic_indices]
        harmonic_SPL_SR_Load = SPL_SR_Load[harmonic_indices]
        harmonic_SPL_merged_Load = SPL_merged_Load[harmonic_indices]

        # ---- 保存谐波表 ----
        output_data = pd.DataFrame({
            'Harmonic Order': np.arange(1, len(harmonic_freq) + 1),
            'Frequency(Hz)': harmonic_freq,
            'SPL_FF_Total(dB)': harmonic_SPL_FF_Total,
            'SPL_SR_Total(dB)': harmonic_SPL_SR_Total,
            'SPL_merged_Total(dB)': harmonic_SPL_merged_Total,
            'SPL_FF_Thickness(dB)': harmonic_SPL_FF_Thickness,
            'SPL_SR_Thickness(dB)': harmonic_SPL_SR_Thickness,
            'SPL_merged_Thickness(dB)': harmonic_SPL_merged_Thickness,
            'SPL_FF_Load(dB)': harmonic_SPL_FF_Load,
            'SPL_SR_Load(dB)': harmonic_SPL_SR_Load,
            'SPL_merged_Load(dB)': harmonic_SPL_merged_Load,
        })
        output_data.to_csv(f"{file_path}\\{filename_prefix[i]}_Harmonics.csv", index=False)
        print(f"Export data to {file_path}\\{filename_prefix[i]}_Harmonics.csv")


def run_band_analysis(file_path, filename_prefix, group_prefixes=None):
    """从 FF+SR 谱中计算倍频程/分数倍频程带能量贡献。

    对于每个观测点，将各噪声分量（Total、Thickness、Load）的合并幅值谱划分
    到频率带中。计算各频带能量、SPL 和能量比指标。可选地按组前缀生成跨观测
    点的综合摘要。

    Parameters
    ----------
    file_path : str
        包含 ``_FreqDomain.csv`` 文件并接收输出文件的目录。
    filename_prefix : list of str
        观测点文件名前缀列表。
    group_prefixes : list of str, optional
        如果给定，则为每个前缀写入按组汇总的 CSV 文件。

    Returns
    -------
    None
        写入每个观测点的 ``_BandContribution.csv`` 和汇总 CSV 文件。

    Notes
    -----
    频带分析使用合并（相干叠加）幅值来反映 FF+SR 组合信号。频带划分由
    ``BandContributionAnalyzer`` 内部处理。
    """
    summary_list = []

    for i in range(len(filename_prefix)):
        # ---- 加载频域数据 ----
        data = pd.read_csv(f"{file_path}\\{filename_prefix[i]}_FreqDomain.csv", header=0, sep=',')
        freq = data['Frequency(Hz)']
        # 使用合并幅值进行频带分析。
        amp_merged_Total = data['amp_merged_Total(Pa)']
        amp_merged_Thickness = data['amp_merged_Thickness(Pa)']
        amp_merged_Load = data['amp_merged_Load(Pa)']

        def process_component(amp, component_name):
            """处理单个合并噪声分量的频带贡献。

            Parameters
            ----------
            amp : array-like
                该分量的合并幅值谱。
            component_name : str
                用于列名的标签（例如 'Total'）。

            Returns
            -------
            detail_data : dict or None
                各频带详情（中心频率、能量、SPL、比例）。
            summary_item : dict or None
                单行摘要（总能量、主导频带、频率区域分解）。
            """
            analyzer = BandContributionAnalyzer(freq.values)
            result = analyzer.analyze_band_contribution(amp.values)
            if not result:
                return None, None

            # 收集各频带详情。
            band_energies = result['band_energies']
            detail_data = {
                'Center Frequency(Hz)': [b['center_freq'] for b in band_energies],
                f'Energy_{component_name}': [b['energy'] for b in band_energies],
                f'SPL_{component_name}(dB)': [b['energy_dB'] for b in band_energies],
                f'Energy Ratio_{component_name}': [b['energy_ratio'] for b in band_energies]
            }

            # 该分量的摘要指标。
            total_energy = result['total_energy']
            energy_dist = result['energy_distribution']
            dominant_band = result['dominant_band']

            summary_item = {
                'Filename': filename_prefix[i],
                f'Total Energy_{component_name}': total_energy,
                f'Dominant Freq_{component_name}(Hz)': dominant_band['center_freq'],
                f'Dominant Ratio_{component_name}': dominant_band['energy_ratio'],
                f'Low Freq Energy_{component_name}': energy_dist['low_freq'],
                f'Mid Freq Energy_{component_name}': energy_dist['mid_freq'],
                f'High Freq Energy_{component_name}': energy_dist['high_freq'],
                f'Low Freq Ratio_{component_name}': energy_dist['low_freq'] / total_energy if total_energy > 0 else 0,
                f'Mid Freq Ratio_{component_name}': energy_dist['mid_freq'] / total_energy if total_energy > 0 else 0,
                f'High Freq Ratio_{component_name}': energy_dist['high_freq'] / total_energy if total_energy > 0 else 0
            }
            return detail_data, summary_item

        # ---- 遍历三种合并噪声分量 ----
        all_detail_data = {}
        combined_summary = {'Filename': filename_prefix[i]}
        for component, amp in [('Total', amp_merged_Total), ('Thickness', amp_merged_Thickness), ('Load', amp_merged_Load)]:
            detail_data, summary_item = process_component(amp, component)
            if detail_data and summary_item:
                all_detail_data.update(detail_data)
                combined_summary.update(summary_item)

        # ---- 保存每个观测点的频带详情 ----
        if all_detail_data:
            detail_df = pd.DataFrame(all_detail_data)
            columns = ['Center Frequency(Hz)'] + [col for col in detail_df.columns if col != 'Center Frequency(Hz)']
            detail_df = detail_df[columns]
            detail_df.to_csv(f"{file_path}\\{filename_prefix[i]}_BandContribution.csv", index=False)
            print(f"Export data to {file_path}\\{filename_prefix[i]}_BandContribution.csv")

        if combined_summary:
            summary_list.append(combined_summary)

    # ---- 构建并保存跨观测点摘要 ----
    if summary_list:
        summary_df = pd.DataFrame(summary_list)
        columns_order = ['Filename']
        for component in ['Total', 'Thickness', 'Load']:
            columns_order.extend([
                f'Total Energy_{component}', f'Dominant Freq_{component}(Hz)', f'Dominant Ratio_{component}',
                f'Low Freq Energy_{component}', f'Mid Freq Energy_{component}', f'High Freq Energy_{component}',
                f'Low Freq Ratio_{component}', f'Mid Freq Ratio_{component}', f'High Freq Ratio_{component}'
            ])
        columns_order = [col for col in columns_order if col in summary_df.columns]
        summary_df = summary_df[columns_order]

        if group_prefixes:
            # 写入按组汇总的文件。
            for prefix in group_prefixes:
                group_df = summary_df[summary_df['Filename'].str.startswith(prefix)]
                if not group_df.empty:
                    group_df.to_csv(f"{file_path}\\{prefix}_BandContribution_Summary.csv", index=False)
                    print(f"Export data to {file_path}\\{prefix}_BandContribution_Summary.csv")

        else:
            summary_df.to_csv(f"{file_path}\\BandContribution_Summary.csv", index=False)
            print(f"Export data to {file_path}\\BandContribution_Summary.csv")


if __name__ == "__main__":
    # ---- 直接执行示例 ----
    file_path = r"Case05"
    Filename_list = ["Case05_Rotor"]
    OBS_Numbers = 12
    OBS_Position_list = np.array([
        [0, 3, 0], [0, 4, 0], [0, 5, 0],
        [0, 3, -0.25], [0, 4, -0.25], [0, 5, -0.25],
        [0, 3, -0.5], [0, 4, -0.5], [0, 5, -0.5],
        [0, 3, -0.75], [0, 4, -0.75], [0, 5, -0.75],
    ])
    if OBS_Numbers != len(OBS_Position_list):
        raise ValueError("OBS_Numbers must be equal to the length of OBS_Position_list")
    filename_prefix = [f"{Filename_list[0]}_OBS{j + 1:04d}" for j in range(OBS_Numbers)]
    run_peak_analysis(file_path, filename_prefix)
    run_band_analysis(file_path, filename_prefix, group_prefixes=Filename_list)
