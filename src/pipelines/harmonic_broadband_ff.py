"""
谐频/宽频统计分析 -- 仅自由场。

本模块基于预先计算的频域数据（``_FreqDomain.csv``）运行，对载荷噪声
（Load）分量进行谐频/宽频分离，并输出多层次的统计信息。

输入
------
* ``{prefix}_FreqDomain.csv`` -- 来自预处理阶段的输出（7 列 FF 格式）。

输出
-------
* ``{prefix}_HarmonicBroadband_Summary.csv``        -- 总体统计摘要
* ``{prefix}_HarmonicBroadband_Band.csv``            -- 分频段 (octave band) 统计
* ``{prefix}_HarmonicBroadband_HarmonicDetail.csv``  -- 逐谐频点明细
* ``{prefix}_HarmonicBroadband_Detail.csv``          -- 全频段逐点明细
* ``{group}_HarmonicBroadband_Summary.csv``          -- 按组汇总的跨观测点摘要

使用方式
-----------
* 直接运行  : ``python harmonic_broadband_ff.py``
* 命令行调度: ``python main.py harmonic ...``
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
    """对 FF 场景执行谐频/宽频统计分析.

    对每个观测点，从 FreqDomain CSV 读取 Load 分量幅值谱，用 Total 分量
    检测谐频，分离谐频/宽频，输出 4 种统计 CSV。

    Parameters
    ----------
    file_path : str
        包含 ``_FreqDomain.csv`` 文件并接收输出文件的目录。
    filename_prefix : list of str
        观测点文件名前缀列表。
    group_prefixes : list of str, optional
        分组名称列表，用于生成跨观测点 Summary 聚合。
    fundamental_freq : float, optional
        叶片通过频率 (Hz)。为 None 则从 Total 谱自动检测。
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

    for i in range(len(filename_prefix)):
        prefix = filename_prefix[i]

        # ---- 读取频域数据 ----
        csv_path = f"{file_path}\\{prefix}_FreqDomain.csv"
        data = pd.read_csv(csv_path, header=0, sep=',')
        freq = data['Frequency(Hz)'].values
        amp_Total = data['amp_Total(Pa)'].values
        amp_Load = data['amp_Load(Pa)'].values

        # ---- 执行分析 ----
        analyzer = HarmonicBroadbandAnalyzer(freq)
        result = analyzer.analyze(
            total_spectrum=amp_Total, load_spectrum=amp_Load,
            bpf=fundamental_freq, max_order=max_harmonic_order,
            bandwidth_ratio=harmonic_bandwidth_ratio,
            band_fraction=band_fraction, f_low=f_low, f_high=f_high,
            tag=""
        )

        if result['fundamental_freq'] is None:
            print(f"Warning: Unable to detect fundamental frequency for {prefix}, skipping.")
            continue

        # ---- 保存输出 CSV ----
        # 1. Summary
        summary_row = {'Filename': prefix}
        summary_row.update(result['summary'])
        summary_rows.append(summary_row)

        # 保存单观测点 Summary
        per_point_summary = pd.DataFrame([summary_row])
        per_point_summary.to_csv(
            f"{file_path}\\{prefix}_HarmonicBroadband_Summary.csv", index=False
        )
        print(f"Export data to {file_path}\\{prefix}_HarmonicBroadband_Summary.csv")

        # 2. Band detail
        if not result['band_df'].empty:
            band_path = f"{file_path}\\{prefix}_HarmonicBroadband_Band.csv"
            result['band_df'].to_csv(band_path, index=False)
            print(f"Export data to {band_path}")

        # 3. Harmonic detail
        if not result['harmonic_df'].empty:
            harmonic_path = f"{file_path}\\{prefix}_HarmonicBroadband_HarmonicDetail.csv"
            result['harmonic_df'].to_csv(harmonic_path, index=False)
            print(f"Export data to {harmonic_path}")

        # 4. Full detail
        if not result['detail_df'].empty:
            detail_path = f"{file_path}\\{prefix}_HarmonicBroadband_Detail.csv"
            result['detail_df'].to_csv(detail_path, index=False)
            print(f"Export data to {detail_path}")

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


if __name__ == "__main__":
    file_path = r"Case01"
    Filename_list = ["Case01_Rotor"]
    OBS_Numbers = 12
    filename_prefix = [f"{Filename_list[0]}_OBS{j + 1:04d}" for j in range(OBS_Numbers)]
    run_harmonic_broadband_analysis(file_path, filename_prefix,
                                     group_prefixes=Filename_list,
                                     fundamental_freq=46.97,
                                     max_harmonic_order=50)
