"""
谐频/宽频统计分析 -- 自由场 + 表面反射。

本模块基于预先计算的频域数据（``_FreqDomain.csv``，FFSR 格式，19 列）运行，
对载荷噪声（Load）分量的 FF、SR、merged 三种信号分别进行谐频/宽频分离，
并输出组合了三种信号的多层次统计信息。

输入
------
* ``{prefix}_FreqDomain.csv`` -- 来自预处理阶段的输出（19 列 FFSR 格式）。

输出
-------
* ``{prefix}_HarmonicBroadband_Summary.csv``        -- 总体统计摘要 (含 FF/SR/merged)
* ``{prefix}_HarmonicBroadband_Band.csv``            -- 分频段统计 (含 FF/SR/merged)
* ``{prefix}_HarmonicBroadband_HarmonicDetail.csv``  -- 逐谐频点明细 (含 FF/SR/merged)
* ``{prefix}_HarmonicBroadband_Detail.csv``          -- 全频段逐点明细 (含 FF/SR/merged)
* ``{group}_HarmonicBroadband_Summary.csv``          -- 按组汇总的跨观测点摘要

使用方式
-----------
* 直接运行  : ``python harmonic_broadband_ffsr.py``
* 命令行调度: ``python main.py harmonic ... --has-reflection``
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
from decomposition import HarmonicBroadbandAnalyzer


def run_harmonic_broadband_analysis(file_path, filename_prefix, group_prefixes=None,
                                     fundamental_freq=None, max_harmonic_order=30,
                                     harmonic_bandwidth_ratio=0.03,
                                     band_fraction=3, f_low=10.0, f_high=20000.0):
    """对 FF+SR 场景执行谐频/宽频统计分析.

    对每个观测点，分别对 FF Load、SR Load、merged Load 三个分量进行
    谐频/宽频分离和统计，谐频检测统一使用 merged Total 谱。

    Parameters
    ----------
    file_path : str
        包含 ``_FreqDomain.csv`` 文件并接收输出文件的目录。
    filename_prefix : list of str
        观测点文件名前缀列表。
    group_prefixes : list of str, optional
        分组名称列表，用于生成跨观测点 Summary 聚合。
    fundamental_freq : float, optional
        叶片通过频率 (Hz)。为 None 则从 merged Total 谱自动检测。
    max_harmonic_order : int
        最大谐频阶数，默认 30。
    harmonic_bandwidth_ratio : float
        谐频提取带宽比，默认 0.03。
    band_fraction : int
        Octave band 分数，默认 3 (1/3 octave)。
    f_low : float
        Octave band 中心频率下限 (Hz)，默认 10。
    f_high : float
        Octave band 中心频率上限 (Hz)，默认 20000。

    Returns
    -------
    None
    """
    summary_rows = []
    # 三种信号的列名后缀 -> tag 映射
    signal_configs = [
        ('FF', 'FF_'),
        ('SR', 'SR_'),
        ('merged', 'merged_'),
    ]

    for i in range(len(filename_prefix)):
        prefix = filename_prefix[i]

        # ---- 读取频域数据 ----
        csv_path = f"{file_path}\\{prefix}_FreqDomain.csv"
        data = pd.read_csv(csv_path, header=0, sep=',')
        freq = data['Frequency(Hz)'].values
        amp_merged_Total = data['amp_merged_Total(Pa)'].values

        # 共用同一个 analyzer 实例（频率轴相同）
        analyzer = HarmonicBroadbandAnalyzer(freq)

        # 先用 merged Total 检测谐频（只检测一次，三种信号共用谐频列表）
        harm_info = analyzer.detect_harmonics(
            amp_merged_Total, bpf=fundamental_freq, max_order=max_harmonic_order
        )
        harmonic_freqs = harm_info['harmonic_freqs']
        harmonic_indices = harm_info['harmonic_indices']
        detected_freq = harm_info['fundamental_freq']

        if detected_freq is None or len(harmonic_indices) == 0:
            print(f"Warning: Unable to detect fundamental frequency for {prefix}, skipping.")
            continue

        summary_row = {'Filename': prefix}
        all_band_dfs = []
        all_harmonic_dfs = []
        all_detail_dfs = []

        for signal_name, tag in signal_configs:
            amp_load = data[f'amp_{signal_name}_Load(Pa)'].values

            # 分离谐频/宽频
            harmonic_amp, broadband_amp = analyzer.separate(
                amp_load, harmonic_freqs, harmonic_bandwidth_ratio
            )

            # 计算各层次统计
            summary = analyzer.compute_summary(
                tag, amp_load, harmonic_amp, broadband_amp,
                detected_freq, len(harmonic_indices)
            )
            summary_row.update(summary)

            band_df = analyzer.compute_band_detail(
                amp_load, harmonic_amp, broadband_amp, f_low, f_high, band_fraction
            )
            # 给 band_df 列名加 tag 前缀（除前三列外）
            if not band_df.empty:
                band_df = _prefix_columns(band_df, tag, skip=3)
                all_band_dfs.append(band_df)

            harmonic_df = analyzer.compute_harmonic_detail(
                amp_load, harmonic_indices, detected_freq, harmonic_bandwidth_ratio
            )
            if not harmonic_df.empty:
                harmonic_df = _prefix_columns(harmonic_df, tag, skip=3)
                all_harmonic_dfs.append(harmonic_df)

            detail_df = analyzer.compute_full_detail(amp_load, harmonic_amp, broadband_amp)
            if not detail_df.empty:
                detail_df = _prefix_columns(detail_df, tag, skip=1)
                all_detail_dfs.append(detail_df)

        summary_rows.append(summary_row)

        # 保存单观测点 Summary
        per_point_summary = pd.DataFrame([summary_row])
        per_point_summary.to_csv(
            f"{file_path}\\{prefix}_HarmonicBroadband_Summary.csv", index=False
        )
        print(f"Export data to {file_path}\\{prefix}_HarmonicBroadband_Summary.csv")

        # ---- 合并 & 保存多信号输出 ----
        # Band: 按 Center Frequency 合并
        if all_band_dfs:
            merged_band = _merge_dfs_on(all_band_dfs, ['Center Frequency(Hz)',
                                         'Lower Bound(Hz)', 'Upper Bound(Hz)'])
            out = f"{file_path}\\{prefix}_HarmonicBroadband_Band.csv"
            merged_band.to_csv(out, index=False)
            print(f"Export data to {out}")

        # Harmonic: 按 Harmonic Order 合并
        if all_harmonic_dfs:
            merged_harmonic = _merge_dfs_on(all_harmonic_dfs, ['Harmonic Order',
                                             'Nominal Frequency(Hz)', 'Actual Frequency(Hz)'])
            out = f"{file_path}\\{prefix}_HarmonicBroadband_HarmonicDetail.csv"
            merged_harmonic.to_csv(out, index=False)
            print(f"Export data to {out}")

        # Detail: 按 Frequency 合并
        if all_detail_dfs:
            merged_detail = _merge_dfs_on(all_detail_dfs, ['Frequency(Hz)'])
            out = f"{file_path}\\{prefix}_HarmonicBroadband_Detail.csv"
            merged_detail.to_csv(out, index=False)
            print(f"Export data to {out}")

    # ---- 全局汇总 Summary ----
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        cols = ['Filename'] + [c for c in summary_df.columns if c != 'Filename']
        summary_df = summary_df[cols]

        if group_prefixes:
            for gp in group_prefixes:
                group_df = summary_df[summary_df['Filename'].str.startswith(gp)]
                if not group_df.empty:
                    out = f"{file_path}\\{gp}_HarmonicBroadband_Summary.csv"
                    group_df.to_csv(out, index=False)
                    print(f"Export data to {out}")
        else:
            out = f"{file_path}\\HarmonicBroadband_Summary.csv"
            summary_df.to_csv(out, index=False)
            print(f"Export data to {out}")


def _prefix_columns(df, tag, skip=0):
    """给 DataFrame 的列名（跳过前 skip 列）添加 tag 前缀."""
    df = df.copy()
    new_cols = list(df.columns[:skip]) + [f'{tag}{c}' for c in df.columns[skip:]]
    df.columns = new_cols
    return df


def _merge_dfs_on(dfs, key_cols):
    """按 key_cols 合并多个 DataFrame."""
    result = dfs[0]
    for df in dfs[1:]:
        result = result.merge(df, on=key_cols, how='outer')
    return result


if __name__ == "__main__":
    file_path = r"Case05"
    Filename_list = ["Case05_Rotor"]
    OBS_Numbers = 12
    filename_prefix = [f"{Filename_list[0]}_OBS{j + 1:04d}" for j in range(OBS_Numbers)]
    run_harmonic_broadband_analysis(file_path, filename_prefix,
                                     group_prefixes=Filename_list,
                                     fundamental_freq=46.97,
                                     max_harmonic_order=50)
