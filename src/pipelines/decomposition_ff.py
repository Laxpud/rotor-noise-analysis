"""
噪声源分解分析（仅自由场，相位约束方法）。

本模块使用相位约束频域方法将自由场总噪声分解为源贡献（厚度噪声、稳态载荷、
非稳态载荷）。同时分离谐波和宽频分量。该分析同时使用频域数据
（``_FreqDomain.csv``）和时域数据（``_FF.csv``），以保留复值 FFT 系数。

输出包括：
* 每个频率的源贡献详情。
* 倍频程/分数倍频程带聚合贡献。
* 每个谐波的贡献分解。
* 跨观测点综合汇总表。

输入
------
* ``{prefix}_FreqDomain.csv`` -- 频域幅值/SPL（来自预处理阶段）。
* ``{prefix}_FF.csv``         -- 时域压力（Thickness、Load、Total）。

输出
-------
* ``{prefix}_SourceContribution_Detail.csv``    -- 每个频率的分解详情。
* ``{prefix}_BandSourceContribution.csv``      -- 各频带的贡献。
* ``{prefix}_HarmonicContribution.csv``         -- 各谐波的贡献。
* ``{prefix}_SourceContribution_Summary.csv``   -- 跨观测点汇总。

使用方式
-----------
* 直接运行  : ``python decomposition_ff.py``
* 命令行调度: ``python main.py source ...``
"""

import os
import sys

# 确保上级 ``src`` 包在导入路径中。
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
from typing import Optional, List
import os

from decomposition import SourceContributionAnalyzer
from signal_utils import rfft


def _load_time_domain(file_path: str, prefix: str):
    """加载时域 FF 数据并计算归一化复 FFT 系数。

    读取 ``{prefix}_FF.csv`` 文件，堆叠时间与信号数组，执行实 FFT，并将复系数
    乘以 ``2.0 / N`` 得到单边幅值/相位谱（直流分量减半）。

    Parameters
    ----------
    file_path : str
        包含时域 CSV 文件的目录。
    prefix : str
        观测点文件名前缀（例如 ``"Case01_Rotor_OBS0001"``）。

    Returns
    -------
    freq : ndarray
        实 FFT 的频率 bin。
    thick_complex : ndarray
        厚度噪声的归一化复谱。
    load_complex : ndarray
        载荷噪声的归一化复谱。
    total_complex : ndarray
        总噪声的归一化复谱。

    Notes
    -----
    缩放系数 ``2.0 / N`` 将原始 FFT 输出转换为单边幅值。直流 bin（索引 0）
    在双边约定下已经正确，因此再减半一次。
    """
    time_file = os.path.join(file_path, f"{prefix}_FF.csv")
    time_data = pd.read_csv(time_file)
    # 堆叠 [time; signal] 供下游辅助函数使用。
    time_thick = np.vstack([time_data['Time'].values, time_data['Thickness'].values])
    time_load = np.vstack([time_data['Time'].values, time_data['Load'].values])
    time_total = np.vstack([time_data['Time'].values, time_data['Total'].values])
    # 实 FFT（输出复值，此处无需相位解缠）。
    freq, thick_complex, _, _ = rfft(time_thick, return_phase=False)
    _, load_complex, _, _ = rfft(time_load, return_phase=False)
    _, total_complex, _, _ = rfft(time_total, return_phase=False)
    N = len(time_data)
    # 缩放为单边幅值谱。
    thick_complex = thick_complex * (2.0 / N); thick_complex[0] = thick_complex[0] / 2.0
    load_complex = load_complex * (2.0 / N); load_complex[0] = load_complex[0] / 2.0
    total_complex = total_complex * (2.0 / N); total_complex[0] = total_complex[0] / 2.0
    return freq, thick_complex, load_complex, total_complex


def run_decomposition_analysis(
    file_path: str,
    filename_prefix: List[str],
    group_prefixes: Optional[List[str]] = None,
    fundamental_freq: Optional[float] = None,
    harmonic_bandwidth_ratio: float = 0.03,
    max_harmonic_order: int = 30,
    band_type: str = "octave",
    band_fraction: int = 3,
    band_f_low: float = 10,
    band_f_high: float = 20000,
    check_phase_consistency: bool = False
):
    """对仅 FF 数据运行基于相位约束的源分解。

    对于每个观测点前缀，该函数：
    1. 加载频域数据和时域复谱。
    2. 如果未提供基频（叶片通过频率），则自动检测。
    3. 运行 ``SourceContributionAnalyzer`` 以分离厚度、稳态载荷、非稳态载荷、
       谐波和宽频贡献。
    4. 写入详情、频带、谐波和汇总 CSV 输出文件。

    Parameters
    ----------
    file_path : str
        包含输入文件并接收输出 CSV 文件的目录。
    filename_prefix : list of str
        观测点文件名前缀列表。
    group_prefixes : list of str, optional
        如果给定，则写入按组汇总的 CSV 文件。
    fundamental_freq : float, optional
        基频（叶片通过频率），单位为 Hz。如果为 ``None``，则从 Total 幅值谱中
        自动检测。
    harmonic_bandwidth_ratio : float, optional
        用于谐波提取的带宽与基频之比。默认值为 0.03。
    max_harmonic_order : int, optional
        要提取的最大谐波阶次。默认值为 30。
    band_type : str, optional
        频带类型，例如 ``"octave"`` 或 ``"fractional"``。
    band_fraction : int, optional
        分数倍频程带的分母（例如 3 表示 1/3 倍频程）。
    band_f_low : float, optional
        频带分析的下限频率（Hz）。默认值为 10。
    band_f_high : float, optional
        频带分析的上限频率（Hz）。默认值为 20000。
    check_phase_consistency : bool, optional
        如果为 True，则计算并输出厚度和载荷分量之间的相位一致性统计量。

    Returns
    -------
    None
        结果写入 ``file_path`` 下的 CSV 文件。

    Notes
    -----
    核心分解逻辑位于 ``SourceContributionAnalyzer`` 中。相位一致性检查有助于
    诊断两个源机制之间是否存在非物理的抵消现象。
    """
    summary_list = []
    for prefix in filename_prefix:
        print(f"Processing {prefix}...")
        # ---- 加载数据 ----
        freq_domain_data = pd.read_csv(os.path.join(file_path, f"{prefix}_FreqDomain.csv"), header=0, sep=",")
        freq = freq_domain_data["Frequency(Hz)"].values
        # 从时域恢复复 FFT 系数。
        _, thick_complex, load_complex, total_complex = _load_time_domain(file_path, prefix)

        # ---- 如果未提供则自动检测基频 ----
        if fundamental_freq is None:
            from spectral import PeakFrequencyAnalyzer
            amp_Total = freq_domain_data["amp_Total(Pa)"].values
            peak_analyzer = PeakFrequencyAnalyzer(freq)
            peak_result = peak_analyzer.analyze_spectrum(amp_Total, prominence=0.005)
            fundamental_freq = peak_result["fundamental_freq"]
            print(f"Automatically identified fundamental frequency: {fundamental_freq:.2f} Hz")

        # ---- 运行源分解 ----
        analyzer = SourceContributionAnalyzer(freq)
        results = analyzer.analyze(
            thickness_complex=thick_complex, load_complex=load_complex,
            total_complex=total_complex, fundamental_freq=fundamental_freq,
            harmonic_bandwidth_ratio=harmonic_bandwidth_ratio,
            max_harmonic_order=max_harmonic_order,
            band_type=band_type, band_fraction=band_fraction,
            band_f_low=band_f_low, band_f_high=band_f_high,
            check_phase_consistency=check_phase_consistency)

        # ---- 如果要求则报告相位一致性 ----
        if check_phase_consistency and results.get('phase_stats') is not None:
            ps = results['phase_stats']
            print(f"  Phase stats: mean diff={ps['mean_phase_diff_deg']:.2f}°, "
                  f"var={ps['overall_phase_diff_variance_deg']:.4f}°²")

        # ---- 输出 1：每个频率的详情 ----
        detail_df = results["detail_data"]
        detail_df.to_csv(f"{file_path}\\{prefix}_SourceContribution_Detail.csv", index=False)
        print(f"Export data to {file_path}\\{prefix}_SourceContribution_Detail.csv")

        # ---- 输出 2：按频带聚合的贡献 ----
        band_results = results["band_results"]
        if band_results:
            band_df = pd.DataFrame(band_results)
            # 构建有序的列列表：几何列在前，然后是源 × 指标。
            output_columns = ["center_freq", "lower_bound", "upper_bound"]
            for src in ["thickness", "steady_load", "unsteady_load", "total", "harmonic", "broadband"]:
                for suf in ["energy", "ratio", "spl"]:
                    output_columns.append(f"{src}_{suf}")
            output_columns.append("harmonic_ratio_in_band")
            output_columns = [c for c in output_columns if c in band_df.columns]
            band_df = band_df[output_columns]
            # 转换为可读列名。
            col_map = {
                "center_freq": "Center Frequency(Hz)", "lower_bound": "Lower Bound(Hz)",
                "upper_bound": "Upper Bound(Hz)",
                "thickness_energy": "Thickness Energy", "thickness_ratio": "Thickness Energy Ratio",
                "thickness_spl": "Thickness SPL(dB)",
                "steady_load_energy": "Steady Load Energy", "steady_load_ratio": "Steady Load Energy Ratio",
                "steady_load_spl": "Steady Load SPL(dB)",
                "unsteady_load_energy": "Unsteady Load Energy", "unsteady_load_ratio": "Unsteady Load Energy Ratio",
                "unsteady_load_spl": "Unsteady Load SPL(dB)",
                "total_energy": "Total Energy", "total_ratio": "Total Energy Ratio", "total_spl": "Total SPL(dB)",
                "harmonic_energy": "Harmonic Energy", "harmonic_ratio": "Harmonic Energy Ratio",
                "harmonic_spl": "Harmonic SPL(dB)",
                "broadband_energy": "Broadband Energy", "broadband_ratio": "Broadband Energy Ratio",
                "broadband_spl": "Broadband SPL(dB)", "harmonic_ratio_in_band": "Harmonic Ratio in Band",
            }
            band_df = band_df.rename(columns=col_map)
            band_df.to_csv(f"{file_path}\\{prefix}_BandSourceContribution.csv", index=False)
            print(f"Export data to {file_path}\\{prefix}_BandSourceContribution.csv")

        # ---- 输出 3：各谐波贡献 ----
        harmonic_results = results["harmonic_results"]
        if harmonic_results:
            harmonic_df = pd.DataFrame(harmonic_results)
            # 构建有序的列列表。
            out_cols = ["harmonic_order", "nominal_freq", "actual_freq"]
            for src in ["thickness", "steady_load", "unsteady_load", "total", "harmonic", "broadband"]:
                for suf in ["amp", "spl", "ratio"]:
                    out_cols.append(f"{src}_{suf}")
            out_cols = [c for c in out_cols if c in harmonic_df.columns]
            harmonic_df = harmonic_df[out_cols]
            # 转换为可读列名。
            col_map_h = {
                "harmonic_order": "Harmonic Order", "nominal_freq": "Nominal Frequency(Hz)",
                "actual_freq": "Actual Frequency(Hz)",
                "thickness_amp": "Thickness Amplitude(Pa)", "thickness_spl": "Thickness SPL(dB)",
                "thickness_ratio": "Thickness Energy Ratio",
                "steady_load_amp": "Steady Load Amplitude(Pa)", "steady_load_spl": "Steady Load SPL(dB)",
                "steady_load_ratio": "Steady Load Energy Ratio",
                "unsteady_load_amp": "Unsteady Load Amplitude(Pa)", "unsteady_load_spl": "Unsteady Load SPL(dB)",
                "unsteady_load_ratio": "Unsteady Load Energy Ratio",
                "total_amp": "Total Amplitude(Pa)", "total_spl": "Total SPL(dB)",
                "harmonic_amp": "Harmonic Amplitude(Pa)", "harmonic_spl": "Harmonic SPL(dB)",
                "broadband_amp": "Broadband Amplitude(Pa)", "broadband_spl": "Broadband SPL(dB)",
            }
            harmonic_df = harmonic_df.rename(columns=col_map_h)
            harmonic_df.to_csv(f"{file_path}\\{prefix}_HarmonicContribution.csv", index=False)
            print(f"Export data to {file_path}\\{prefix}_HarmonicContribution.csv")

        # ---- 构建每个观测点的汇总行 ----
        global_stats = results["global_stats"]
        summary_item = {"Filename": prefix, "Fundamental Frequency(Hz)": results["fundamental_freq"],
                        "Number of Harmonics": len(results["harmonic_freqs"])}
        summary_item.update(global_stats)
        summary_list.append(summary_item)

    # ---- 输出 4：跨观测点汇总 ----
    if summary_list:
        summary_df = pd.DataFrame(summary_list)
        base_cols = ["Filename", "Fundamental Frequency(Hz)", "Number of Harmonics"]
        # 源级统计列。
        src_cols = []
        for src in ["thickness", "steady_load", "unsteady_load", "total", "harmonic", "broadband"]:
            src_cols.extend([f"{src}_total_energy", f"{src}_total_ratio", f"{src}_total_spl",
                             f"{src}_low_freq_energy", f"{src}_low_freq_ratio",
                             f"{src}_mid_freq_energy", f"{src}_mid_freq_ratio",
                             f"{src}_high_freq_energy", f"{src}_high_freq_ratio"])
        if 'harmonic_to_broadband_ratio' in summary_df.columns:
            src_cols.append('harmonic_to_broadband_ratio')
        src_cols = [c for c in src_cols if c in summary_df.columns]
        summary_df = summary_df[base_cols + src_cols]

        # 汇总表的可读列名。
        col_map_s = {
            "thickness_total_energy": "Thickness Total Energy", "thickness_total_ratio": "Thickness Total Energy Ratio",
            "thickness_total_spl": "Thickness Total SPL(dB)", "thickness_low_freq_energy": "Thickness Low Frequency Energy",
            "thickness_low_freq_ratio": "Thickness Low Frequency Ratio", "thickness_mid_freq_energy": "Thickness Mid Frequency Energy",
            "thickness_mid_freq_ratio": "Thickness Mid Frequency Ratio", "thickness_high_freq_energy": "Thickness High Frequency Energy",
            "thickness_high_freq_ratio": "Thickness High Frequency Ratio",
            "steady_load_total_energy": "Steady Load Total Energy", "steady_load_total_ratio": "Steady Load Total Energy Ratio",
            "steady_load_total_spl": "Steady Load Total SPL(dB)", "steady_load_low_freq_energy": "Steady Load Low Frequency Energy",
            "steady_load_low_freq_ratio": "Steady Load Low Frequency Ratio", "steady_load_mid_freq_energy": "Steady Load Mid Frequency Energy",
            "steady_load_mid_freq_ratio": "Steady Load Mid Frequency Ratio", "steady_load_high_freq_energy": "Steady Load High Frequency Energy",
            "steady_load_high_freq_ratio": "Steady Load High Frequency Ratio",
            "unsteady_load_total_energy": "Unsteady Load Total Energy", "unsteady_load_total_ratio": "Unsteady Load Total Energy Ratio",
            "unsteady_load_total_spl": "Unsteady Load Total SPL(dB)", "unsteady_load_low_freq_energy": "Unsteady Load Low Frequency Energy",
            "unsteady_load_low_freq_ratio": "Unsteady Load Low Frequency Ratio", "unsteady_load_mid_freq_energy": "Unsteady Load Mid Frequency Energy",
            "unsteady_load_mid_freq_ratio": "Unsteady Load Mid Frequency Ratio", "unsteady_load_high_freq_energy": "Unsteady Load High Frequency Energy",
            "unsteady_load_high_freq_ratio": "Unsteady Load High Frequency Ratio",
            "total_total_energy": "Total Energy", "total_total_ratio": "Total Energy Ratio", "total_total_spl": "Total SPL(dB)",
            "total_low_freq_energy": "Total Low Frequency Energy", "total_low_freq_ratio": "Total Low Frequency Ratio",
            "total_mid_freq_energy": "Total Mid Frequency Energy", "total_mid_freq_ratio": "Total Mid Frequency Ratio",
            "total_high_freq_energy": "Total High Frequency Energy", "total_high_freq_ratio": "Total High Frequency Ratio",
            "harmonic_total_energy": "Harmonic Total Energy", "harmonic_total_ratio": "Harmonic to Total Energy Ratio",
            "harmonic_total_spl": "Harmonic Total SPL(dB)", "harmonic_low_freq_energy": "Harmonic Low Frequency Energy",
            "harmonic_low_freq_ratio": "Harmonic Low Frequency Ratio", "harmonic_mid_freq_energy": "Harmonic Mid Frequency Energy",
            "harmonic_mid_freq_ratio": "Harmonic Mid Frequency Ratio", "harmonic_high_freq_energy": "Harmonic High Frequency Energy",
            "harmonic_high_freq_ratio": "Harmonic High Frequency Ratio",
            "broadband_total_energy": "Broadband Total Energy", "broadband_total_ratio": "Broadband Total Energy Ratio",
            "broadband_total_spl": "Broadband Total SPL(dB)", "broadband_low_freq_energy": "Broadband Low Frequency Energy",
            "broadband_low_freq_ratio": "Broadband Low Frequency Ratio", "broadband_mid_freq_energy": "Broadband Mid Frequency Energy",
            "broadband_mid_freq_ratio": "Broadband Mid Frequency Ratio", "broadband_high_freq_energy": "Broadband High Frequency Energy",
            "broadband_high_freq_ratio": "Broadband High Frequency Ratio",
            "harmonic_to_broadband_ratio": "Harmonic to (Harmonic+Broadband) Ratio",
        }
        summary_df = summary_df.rename(columns=col_map_s)

        # 写入按组汇总或单个聚合汇总。
        if group_prefixes:
            for gp in group_prefixes:
                gdf = summary_df[summary_df["Filename"].str.startswith(gp)]
                if not gdf.empty:
                    gdf.to_csv(f"{file_path}\\{gp}_SourceContribution_Summary.csv", index=False)
                    print(f"Export data to {file_path}\\{gp}_SourceContribution_Summary.csv")
        else:
            summary_df.to_csv(f"{file_path}\\SourceContribution_Summary.csv", index=False)
            print(f"Export data to {file_path}\\SourceContribution_Summary.csv")

    print("Analysis completed!")


if __name__ == "__main__":
    # ---- 直接执行示例 ----
    file_path = r"Case01"
    Filename_list = ["Case01_Rotor"]
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
    fundamental_freq = 46.9698
    check_phase_consistency = True

    run_decomposition_analysis(
        file_path=file_path, filename_prefix=filename_prefix,
        group_prefixes=Filename_list, fundamental_freq=fundamental_freq,
        max_harmonic_order=50, check_phase_consistency=check_phase_consistency)
