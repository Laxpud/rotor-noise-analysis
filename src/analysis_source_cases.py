# 算例具备频域数据（_FreqDomain.csv）后再运行此脚本
# 适用于有自由场和表面反射信号的算例
#
# 本脚本用于对旋翼气动噪声进行源项频域贡献量化分析，主要包含以下功能：
# 1. 谐频与宽频噪声分离
# 2. 各谐频上的噪声源贡献分析（厚度、定常载荷、非定常载荷）
# 3. 1/3倍频程频带内的噪声源贡献分析
# 4. 谐频能量在各频带内的占比分析
#
# 支持三种信号条件的分析：
# - FF：自由场信号
# - SR：表面反射信号
# - merged：自由场+表面反射的合成信号
#
# 输出文件：
# 1. *_SourceContribution_Detail.csv：频率点级的详细数据
# 2. *_BandSourceContribution.csv：1/3倍频程频带的汇总数据
# 3. *_SourceContribution_Summary.csv：全频段的统计汇总数据
# 4. *_HarmonicContribution.csv：各阶谐频的噪声源贡献数据

import numpy as np
import pandas as pd
from typing import Optional, List, Dict

from analyzer.source_analyzer import SourceContributionAnalyzer


def SourceContributionAnalyze(
    file_path: str,
    filename_prefix: List[str],
    group_prefixes: Optional[List[str]] = None,
    fundamental_freq: Optional[float] = None,
    harmonic_bandwidth_ratio: float = 0.03,
    max_harmonic_order: int = 30,
    band_type: str = 'octave',
    band_fraction: int = 3,
    band_f_low: float = 10,
    band_f_high: float = 20000
):
    """
    源项频域贡献量化分析主函数（支持自由场+表面反射场景）

    参数:
        file_path: 数据文件所在目录路径
        filename_prefix: 要处理的文件名前缀列表（不包含_FreqDomain.csv后缀）
        group_prefixes: 分组保存汇总数据的前缀列表（可选）
        fundamental_freq: 基频（叶片通过频率），如不提供则自动识别（使用merged_Total分量）
        harmonic_bandwidth_ratio: 谐频带宽比（相对于谐频频率）
        max_harmonic_order: 最大谐频阶数
        band_type: 频带类型，'octave'（倍频程）或'custom'（自定义）
        band_fraction: 倍频程分数，1=1倍频程，3=1/3倍频程，12=1/12倍频程
        band_f_low: 最低分析频率
        band_f_high: 最高分析频率
    """
    summary_list = []

    # 需要处理的信号类型
    signal_types = ['FF', 'SR', 'merged']

    for prefix in filename_prefix:
        print(f"Processing {prefix}...")

        # 读取频率域线性幅值数据
        freq_domain_data = pd.read_csv(f"{file_path}\\{prefix}_FreqDomain.csv", header=0, sep=',')
        freq = freq_domain_data['Frequency(Hz)'].values

        # 首先使用merged_Total分量识别谐频，确保三种信号类型使用相同的谐频参考
        if fundamental_freq is None:
            amp_merged_Total = freq_domain_data['amp_merged_Total(Pa)'].values
            from analyzer.analyzer import PeakFrequencyAnalyzer
            peak_analyzer = PeakFrequencyAnalyzer(freq)
            peak_result = peak_analyzer.analyze_spectrum(amp_merged_Total, prominence=0.005)
            fundamental_freq = peak_result['fundamental_freq']
            print(f"Automatically identified fundamental frequency: {fundamental_freq:.2f} Hz")

        # 存储所有信号类型的分析结果
        all_detail_data = {'Frequency(Hz)': freq}
        all_band_data = {}
        all_harmonic_data = {}
        all_global_stats = {}

        # 分别处理每种信号类型
        for signal_type in signal_types:
            print(f"  Processing {signal_type} component...")

            # 获取该信号类型的各分量幅值
            amp_Thickness = freq_domain_data[f'amp_{signal_type}_Thickness(Pa)'].values
            amp_Load = freq_domain_data[f'amp_{signal_type}_Load(Pa)'].values
            amp_Total = freq_domain_data[f'amp_{signal_type}_Total(Pa)'].values

            # 初始化源项贡献分析器
            analyzer = SourceContributionAnalyzer(freq)

            # 执行分析（使用统一的基频）
            results = analyzer.analyze(
                thickness_spectrum=amp_Thickness,
                load_spectrum=amp_Load,
                total_spectrum=amp_Total,
                fundamental_freq=fundamental_freq,
                harmonic_bandwidth_ratio=harmonic_bandwidth_ratio,
                max_harmonic_order=max_harmonic_order,
                band_type=band_type,
                band_fraction=band_fraction,
                band_f_low=band_f_low,
                band_f_high=band_f_high
            )

            # 收集详细频率点数据
            detail_df = results['detail_data']
            for col in detail_df.columns:
                if col != 'Frequency(Hz)':
                    all_detail_data[f'{signal_type}_{col}'] = detail_df[col]

            # 收集频带数据
            band_results = results['band_results']
            if band_results:
                band_df = pd.DataFrame(band_results)
                if not all_band_data:
                    # 第一次添加，保留频率相关列
                    all_band_data['Center Frequency(Hz)'] = band_df['center_freq']
                    all_band_data['Lower Bound(Hz)'] = band_df['lower_bound']
                    all_band_data['Upper Bound(Hz)'] = band_df['upper_bound']
                # 添加当前信号类型的列
                for col in band_df.columns:
                    if col not in ['center_freq', 'lower_bound', 'upper_bound', 'n_points']:
                        all_band_data[f'{signal_type}_{col}'] = band_df[col]

            # 收集谐频数据
            harmonic_results = results['harmonic_results']
            if harmonic_results:
                harmonic_df = pd.DataFrame(harmonic_results)
                if not all_harmonic_data:
                    # 第一次添加，保留谐频相关列
                    all_harmonic_data['Harmonic Order'] = harmonic_df['harmonic_order']
                    all_harmonic_data['Nominal Frequency(Hz)'] = harmonic_df['nominal_freq']
                    all_harmonic_data['Actual Frequency(Hz)'] = harmonic_df['actual_freq']
                # 添加当前信号类型的列
                for col in harmonic_df.columns:
                    if col not in ['harmonic_order', 'nominal_freq', 'actual_freq']:
                        all_harmonic_data[f'{signal_type}_{col}'] = harmonic_df[col]

            # 收集全局统计数据
            global_stats = results['global_stats']
            for key, value in global_stats.items():
                all_global_stats[f'{signal_type}_{key}'] = value

        # 1. 输出详细频率点数据
        detail_df = pd.DataFrame(all_detail_data)
        detail_df.to_csv(f"{file_path}\\{prefix}_SourceContribution_Detail.csv", index=False)

        # 2. 输出频带贡献数据
        if all_band_data:
            band_df = pd.DataFrame(all_band_data)
            # 重命名列，更友好
            column_mapping = {
                'thickness_energy': 'Thickness Energy',
                'thickness_ratio': 'Thickness Energy Ratio',
                'thickness_spl': 'Thickness SPL(dB)',
                'steady_load_energy': 'Steady Load Energy',
                'steady_load_ratio': 'Steady Load Energy Ratio',
                'steady_load_spl': 'Steady Load SPL(dB)',
                'unsteady_load_energy': 'Unsteady Load Energy',
                'unsteady_load_ratio': 'Unsteady Load Energy Ratio',
                'unsteady_load_spl': 'Unsteady Load SPL(dB)',
                'total_energy': 'Total Energy',
                'total_ratio': 'Total Energy Ratio',
                'total_spl': 'Total SPL(dB)',
                'harmonic_energy': 'Harmonic Energy',
                'harmonic_ratio': 'Harmonic Energy Ratio',
                'harmonic_spl': 'Harmonic SPL(dB)',
                'broadband_energy': 'Broadband Energy',
                'broadband_ratio': 'Broadband Energy Ratio',
                'broadband_spl': 'Broadband SPL(dB)',
                'harmonic_ratio_in_band': 'Harmonic Ratio in Band'
            }
            # 应用重命名
            renamed_columns = {}
            for col in band_df.columns:
                for signal_type in signal_types:
                    prefix_str = f'{signal_type}_'
                    if col.startswith(prefix_str):
                        base_col = col[len(prefix_str):]
                        if base_col in column_mapping:
                            renamed_columns[col] = f'{signal_type}_{column_mapping[base_col]}'
                        else:
                            renamed_columns[col] = col
            band_df = band_df.rename(columns=renamed_columns)
            band_df.to_csv(f"{file_path}\\{prefix}_BandSourceContribution.csv", index=False)

        # 3. 输出谐频贡献数据
        if all_harmonic_data:
            harmonic_df = pd.DataFrame(all_harmonic_data)
            # 重命名列
            column_mapping = {
                'thickness_amp': 'Thickness Amplitude(Pa)',
                'thickness_spl': 'Thickness SPL(dB)',
                'thickness_ratio': 'Thickness Energy Ratio',
                'steady_load_amp': 'Steady Load Amplitude(Pa)',
                'steady_load_spl': 'Steady Load SPL(dB)',
                'steady_load_ratio': 'Steady Load Energy Ratio',
                'unsteady_load_amp': 'Unsteady Load Amplitude(Pa)',
                'unsteady_load_spl': 'Unsteady Load SPL(dB)',
                'unsteady_load_ratio': 'Unsteady Load Energy Ratio',
                'total_amp': 'Total Amplitude(Pa)',
                'total_spl': 'Total SPL(dB)',
                'harmonic_amp': 'Harmonic Amplitude(Pa)',
                'harmonic_spl': 'Harmonic SPL(dB)',
                'broadband_amp': 'Broadband Amplitude(Pa)',
                'broadband_spl': 'Broadband SPL(dB)'
            }
            # 应用重命名
            renamed_columns = {}
            for col in harmonic_df.columns:
                for signal_type in signal_types:
                    prefix_str = f'{signal_type}_'
                    if col.startswith(prefix_str):
                        base_col = col[len(prefix_str):]
                        if base_col in column_mapping:
                            renamed_columns[col] = f'{signal_type}_{column_mapping[base_col]}'
                        else:
                            renamed_columns[col] = col
            harmonic_df = harmonic_df.rename(columns=renamed_columns)
            harmonic_df.to_csv(f"{file_path}\\{prefix}_HarmonicContribution.csv", index=False)

        # 4. 准备汇总数据
        summary_item = {
            'Filename': prefix,
            'Fundamental Frequency(Hz)': fundamental_freq,
            'Number of Harmonics': len(results['harmonic_freqs']),
        }
        # 添加全局统计项
        summary_item.update(all_global_stats)
        summary_list.append(summary_item)

    # 保存汇总数据
    if summary_list:
        summary_df = pd.DataFrame(summary_list)
        # 调整列顺序
        base_columns = ['Filename', 'Fundamental Frequency(Hz)', 'Number of Harmonics']

        # 按信号类型和类别分组
        source_columns = []
        for signal_type in signal_types:
            for source in ['thickness', 'steady_load', 'unsteady_load', 'total', 'harmonic', 'broadband']:
                source_columns.extend([
                    f'{signal_type}_{source}_total_energy',
                    f'{signal_type}_{source}_total_ratio',
                    f'{signal_type}_{source}_total_spl',
                    f'{signal_type}_{source}_low_freq_energy',
                    f'{signal_type}_{source}_low_freq_ratio',
                    f'{signal_type}_{source}_mid_freq_energy',
                    f'{signal_type}_{source}_mid_freq_ratio',
                    f'{signal_type}_{source}_high_freq_energy',
                    f'{signal_type}_{source}_high_freq_ratio'
                ])
                if source == 'harmonic':
                    source_columns.append(f'{signal_type}_harmonic_total_ratio')

        # 确保列存在
        source_columns = [col for col in source_columns if col in summary_df.columns]
        columns_order = base_columns + source_columns
        summary_df = summary_df[columns_order]

        # 重命名列
        column_mapping = {
            'thickness_total_energy': 'Thickness Total Energy',
            'thickness_total_ratio': 'Thickness Total Energy Ratio',
            'thickness_total_spl': 'Thickness Total SPL(dB)',
            'thickness_low_freq_energy': 'Thickness Low Frequency Energy',
            'thickness_low_freq_ratio': 'Thickness Low Frequency Ratio',
            'thickness_mid_freq_energy': 'Thickness Mid Frequency Energy',
            'thickness_mid_freq_ratio': 'Thickness Mid Frequency Ratio',
            'thickness_high_freq_energy': 'Thickness High Frequency Energy',
            'thickness_high_freq_ratio': 'Thickness High Frequency Ratio',
            'steady_load_total_energy': 'Steady Load Total Energy',
            'steady_load_total_ratio': 'Steady Load Total Energy Ratio',
            'steady_load_total_spl': 'Steady Load Total SPL(dB)',
            'steady_load_low_freq_energy': 'Steady Load Low Frequency Energy',
            'steady_load_low_freq_ratio': 'Steady Load Low Frequency Ratio',
            'steady_load_mid_freq_energy': 'Steady Load Mid Frequency Energy',
            'steady_load_mid_freq_ratio': 'Steady Load Mid Frequency Ratio',
            'steady_load_high_freq_energy': 'Steady Load High Frequency Energy',
            'steady_load_high_freq_ratio': 'Steady Load High Frequency Ratio',
            'unsteady_load_total_energy': 'Unsteady Load Total Energy',
            'unsteady_load_total_ratio': 'Unsteady Load Total Energy Ratio',
            'unsteady_load_total_spl': 'Unsteady Load Total SPL(dB)',
            'unsteady_load_low_freq_energy': 'Unsteady Load Low Frequency Energy',
            'unsteady_load_low_freq_ratio': 'Unsteady Load Low Frequency Ratio',
            'unsteady_load_mid_freq_energy': 'Unsteady Load Mid Frequency Energy',
            'unsteady_load_mid_freq_ratio': 'Unsteady Load Mid Frequency Ratio',
            'unsteady_load_high_freq_energy': 'Unsteady Load High Frequency Energy',
            'unsteady_load_high_freq_ratio': 'Unsteady Load High Frequency Ratio',
            'total_total_energy': 'Total Energy',
            'total_total_ratio': 'Total Energy Ratio',
            'total_total_spl': 'Total SPL(dB)',
            'total_low_freq_energy': 'Total Low Frequency Energy',
            'total_low_freq_ratio': 'Total Low Frequency Ratio',
            'total_mid_freq_energy': 'Total Mid Frequency Energy',
            'total_mid_freq_ratio': 'Total Mid Frequency Ratio',
            'total_high_freq_energy': 'Total High Frequency Energy',
            'total_high_freq_ratio': 'Total High Frequency Ratio',
            'harmonic_total_energy': 'Harmonic Total Energy',
            'harmonic_total_ratio': 'Harmonic Total Energy Ratio',
            'harmonic_total_spl': 'Harmonic Total SPL(dB)',
            'harmonic_low_freq_energy': 'Harmonic Low Frequency Energy',
            'harmonic_low_freq_ratio': 'Harmonic Low Frequency Ratio',
            'harmonic_mid_freq_energy': 'Harmonic Mid Frequency Energy',
            'harmonic_mid_freq_ratio': 'Harmonic Mid Frequency Ratio',
            'harmonic_high_freq_energy': 'Harmonic High Frequency Energy',
            'harmonic_high_freq_ratio': 'Harmonic High Frequency Ratio',
            'broadband_total_energy': 'Broadband Total Energy',
            'broadband_total_ratio': 'Broadband Total Energy Ratio',
            'broadband_total_spl': 'Broadband Total SPL(dB)',
            'broadband_low_freq_energy': 'Broadband Low Frequency Energy',
            'broadband_low_freq_ratio': 'Broadband Low Frequency Ratio',
            'broadband_mid_freq_energy': 'Broadband Mid Frequency Energy',
            'broadband_mid_freq_ratio': 'Broadband Mid Frequency Ratio',
            'broadband_high_freq_energy': 'Broadband High Frequency Energy',
            'broadband_high_freq_ratio': 'Broadband High Frequency Ratio',
            'harmonic_total_ratio': 'Total Harmonic Energy Ratio'
        }
        # 应用重命名
        renamed_columns = {}
        for col in summary_df.columns:
            for signal_type in signal_types:
                prefix_str = f'{signal_type}_'
                if col.startswith(prefix_str):
                    base_col = col[len(prefix_str):]
                    if base_col in column_mapping:
                        renamed_columns[col] = f'{signal_type}_{column_mapping[base_col]}'
                    else:
                        renamed_columns[col] = col
        summary_df = summary_df.rename(columns=renamed_columns)

        if group_prefixes:
            # 按前缀分组保存
            for group_prefix in group_prefixes:
                # 筛选属于当前前缀的数据
                group_df = summary_df[summary_df['Filename'].str.startswith(group_prefix)]
                if not group_df.empty:
                    group_df.to_csv(f"{file_path}\\{group_prefix}_SourceContribution_Summary.csv", index=False)
        else:
            # 如果没有提供前缀，保存为一个总文件
            summary_df.to_csv(f"{file_path}\\SourceContribution_Summary.csv", index=False)

    print("Analysis completed!")


if __name__ == "__main__":
    # 示例配置 - 请根据实际情况修改
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

    # 可选：指定基频（叶片通过频率），如不指定则自动识别
    # fundamental_freq = 25.0  # 示例值，请根据实际情况修改

    # 运行分析
    SourceContributionAnalyze(
        file_path=file_path,
        filename_prefix=filename_prefix,
        group_prefixes=Filename_list,
        # fundamental_freq=fundamental_freq
    )
