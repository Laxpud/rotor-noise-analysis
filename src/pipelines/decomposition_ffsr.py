"""
噪声源分解分析（自由场+地面反射，相位约束方法）。

本模块使用相位约束频域方法将自由场、地面反射和合并（相干叠加）总噪声分解
为源贡献（厚度噪声、稳态载荷、非稳态载荷）。同时分离谐波和宽频分量。

对于每个观测点，分解过程运行三次 —— 分别针对 FF 信号、SR 信号和合并
（FF+SR）信号 —— 然后将输出合并到带有信号类型前缀的统一表格中。

输入
------
* ``{prefix}_FreqDomain.csv`` -- 频域数据（来自 FF+SR 预处理阶段）。
* ``{prefix}_FF.csv``         -- 自由场时域压力。
* ``{prefix}_SR.csv``         -- 地面反射时域压力。

输出
-------
* ``{prefix}_SourceContribution_Detail.csv``    -- 每个频率的分解详情（所有信号类型）。
* ``{prefix}_BandSourceContribution.csv``      -- 各频带的贡献（所有信号类型）。
* ``{prefix}_HarmonicContribution.csv``         -- 各谐波的贡献（所有信号类型）。
* ``{prefix}_SourceContribution_Summary.csv``   -- 跨观测点汇总。

使用方式
-----------
* 直接运行  : ``python decomposition_ffsr.py``
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


def _load_and_process_time_domain(file_path: str, prefix: str, signal_type: str):
    """加载单个时域 CSV 文件（FF 或 SR）并计算 Thickness、Load 和 Total 的
    归一化复 FFT 系数。

    Parameters
    ----------
    file_path : str
        包含时域 CSV 文件的目录。
    prefix : str
        观测点文件名前缀（例如 ``"Case04_Rotor_OBS0001"``）。
    signal_type : str
        信号类型后缀：``"FF"`` 或 ``"SR"``。

    Returns
    -------
    thick_complex : ndarray
        厚度噪声的归一化复谱。
    load_complex : ndarray
        载荷噪声的归一化复谱。
    total_complex : ndarray
        总噪声的归一化复谱。

    Notes
    -----
    缩放系数 ``2.0 / N`` 将原始 FFT 输出转换为单边幅值（直流 bin 减半以匹配
    索引 0 处的双边约定）。
    """
    time_file = os.path.join(file_path, f"{prefix}_{signal_type}.csv")
    time_data = pd.read_csv(time_file)
    # 堆叠 [time; signal] 供 rfft 辅助函数使用。
    time_thick = np.vstack([time_data['Time'].values, time_data['Thickness'].values])
    time_load = np.vstack([time_data['Time'].values, time_data['Load'].values])
    time_total = np.vstack([time_data['Time'].values, time_data['Total'].values])
    # 实 FFT（复值输出，无相位解缠）。
    _, thick_complex, _, _ = rfft(time_thick, return_phase=False)
    _, load_complex, _, _ = rfft(time_load, return_phase=False)
    _, total_complex, _, _ = rfft(time_total, return_phase=False)
    N = len(time_data)
    # 缩放为单边幅值谱。
    thick_complex = thick_complex * (2.0 / N); thick_complex[0] = thick_complex[0] / 2.0
    load_complex = load_complex * (2.0 / N); load_complex[0] = load_complex[0] / 2.0
    total_complex = total_complex * (2.0 / N); total_complex[0] = total_complex[0] / 2.0
    return thick_complex, load_complex, total_complex


def run_decomposition_analysis(
    file_path: str, filename_prefix: List[str],
    group_prefixes: Optional[List[str]] = None,
    fundamental_freq: Optional[float] = None,
    harmonic_bandwidth_ratio: float = 0.03, max_harmonic_order: int = 30,
    band_type: str = "octave", band_fraction: int = 3,
    band_f_low: float = 10, band_f_high: float = 20000,
    check_phase_consistency: bool = False
):
    """对 FF+SR 数据运行基于相位约束的源分解。

    对于每个观测点，该函数：
    1. 加载 FF 和 SR 的复 FFT 谱；生成合并（相干叠加）谱。
    2. 如果未提供叶片通过频率，则自动检测。
    3. 分别对 FF、SR 和合并信号运行 ``SourceContributionAnalyzer``。
    4. 将结果合并到带有信号类型前缀的统一输出表格中。

    Parameters
    ----------
    file_path : str
        包含输入文件并接收输出 CSV 文件的目录。
    filename_prefix : list of str
        观测点文件名前缀列表。
    group_prefixes : list of str, optional
        如果给定，则写入按组汇总的 CSV 文件。
    fundamental_freq : float, optional
        基频（叶片通过频率），单位为 Hz。如果为 ``None``，则自动检测。
    harmonic_bandwidth_ratio : float, optional
        用于谐波提取的带宽与基频之比。默认值为 0.03。
    max_harmonic_order : int, optional
        要提取的最大谐波阶次。默认值为 30。
    band_type : str, optional
        频带类型（``"octave"`` 或 ``"fractional"``）。
    band_fraction : int, optional
        分数倍频程带的分母。默认值为 3（1/3 倍频程）。
    band_f_low : float, optional
        频带分析的下限频率（Hz）。默认值为 10。
    band_f_high : float, optional
        频带分析的上限频率（Hz）。默认值为 20000。
    check_phase_consistency : bool, optional
        如果为 True，则输出每种信号类型的相位一致性统计量。

    Returns
    -------
    None
        结果写入 ``file_path`` 下的 CSV 文件。

    Notes
    -----
    合并谱通过 FF 和 SR 的 FFT 系数进行复加法得到，代表相干叠加。
    """
    summary_list = []
    signal_types = ["FF", "SR", "merged"]

    for prefix in filename_prefix:
        print(f"Processing {prefix}...")
        # ---- 加载频域数据 ----
        freq_domain_data = pd.read_csv(os.path.join(file_path, f"{prefix}_FreqDomain.csv"), header=0, sep=",")
        freq = freq_domain_data["Frequency(Hz)"].values

        # ---- 加载并计算 FF 和 SR 的复谱 ----
        thick_ff, load_ff, total_ff = _load_and_process_time_domain(file_path, prefix, "FF")
        thick_sr, load_sr, total_sr = _load_and_process_time_domain(file_path, prefix, "SR")
        # 合并 = 相干叠加（复加法）。
        thick_merged = thick_ff + thick_sr
        load_merged = load_ff + load_sr
        total_merged = total_ff + total_sr

        # 按信号类型组织谱数据，供分析循环使用。
        signal_data = {"FF": {"thickness": thick_ff, "load": load_ff, "total": total_ff},
                       "SR": {"thickness": thick_sr, "load": load_sr, "total": total_sr},
                       "merged": {"thickness": thick_merged, "load": load_merged, "total": total_merged}}

        # ---- 如果需要则自动检测基频 ----
        if fundamental_freq is None:
            from spectral import PeakFrequencyAnalyzer
            amp_merged_Total = freq_domain_data["amp_merged_Total(Pa)"].values
            peak_analyzer = PeakFrequencyAnalyzer(freq)
            peak_result = peak_analyzer.analyze_spectrum(amp_merged_Total, prominence=0.005)
            fundamental_freq = peak_result["fundamental_freq"]
            print(f"Automatically identified fundamental frequency: {fundamental_freq:.2f} Hz")

        # 用于跨信号类型合并输出的累加器。
        all_detail_data = {"Frequency(Hz)": freq}
        all_band_data = {}
        all_harmonic_data = {}
        all_global_stats = {}

        # ---- 对每种信号类型运行分解 ----
        for signal_type in signal_types:
            print(f"  Processing {signal_type} component...")
            thick_complex = signal_data[signal_type]["thickness"]
            load_complex = signal_data[signal_type]["load"]
            total_complex = signal_data[signal_type]["total"]

            analyzer = SourceContributionAnalyzer(freq)
            results = analyzer.analyze(
                thickness_complex=thick_complex, load_complex=load_complex,
                total_complex=total_complex, fundamental_freq=fundamental_freq,
                harmonic_bandwidth_ratio=harmonic_bandwidth_ratio,
                max_harmonic_order=max_harmonic_order,
                band_type=band_type, band_fraction=band_fraction,
                band_f_low=band_f_low, band_f_high=band_f_high,
                check_phase_consistency=check_phase_consistency)

            # ---- 可选的相位一致性报告 ----
            if check_phase_consistency and results.get('phase_stats') is not None:
                ps = results['phase_stats']
                print(f"    Phase stats for {signal_type}: mean={ps['mean_phase_diff_deg']:.2f}°")

            # ---- 收集每个频率的详情（带前缀） ----
            detail_df = results["detail_data"]
            for col in detail_df.columns:
                if col != "Frequency(Hz)":
                    all_detail_data[f"{signal_type}_{col}"] = detail_df[col]

            # ---- 收集频带结果（带前缀） ----
            band_results = results["band_results"]
            if band_results:
                band_df = pd.DataFrame(band_results)
                # 仅存储一次几何列（从第一个信号类型获取）。
                if not all_band_data:
                    all_band_data["Center Frequency(Hz)"] = band_df["center_freq"]
                    all_band_data["Lower Bound(Hz)"] = band_df["lower_bound"]
                    all_band_data["Upper Bound(Hz)"] = band_df["upper_bound"]
                for col in band_df.columns:
                    if col not in ["center_freq", "lower_bound", "upper_bound", "n_points"]:
                        all_band_data[f"{signal_type}_{col}"] = band_df[col]

            # ---- 收集谐波结果（带前缀） ----
            harmonic_results = results["harmonic_results"]
            if harmonic_results:
                harmonic_df = pd.DataFrame(harmonic_results)
                # 仅存储一次关键列。
                if not all_harmonic_data:
                    all_harmonic_data["Harmonic Order"] = harmonic_df["harmonic_order"]
                    all_harmonic_data["Nominal Frequency(Hz)"] = harmonic_df["nominal_freq"]
                    all_harmonic_data["Actual Frequency(Hz)"] = harmonic_df["actual_freq"]
                for col in harmonic_df.columns:
                    if col not in ["harmonic_order", "nominal_freq", "actual_freq"]:
                        all_harmonic_data[f"{signal_type}_{col}"] = harmonic_df[col]

            # ---- 收集全局统计量（带前缀） ----
            global_stats = results["global_stats"]
            for key, value in global_stats.items():
                all_global_stats[f"{signal_type}_{key}"] = value

        # ---- 输出 1：每个频率的详情（所有信号类型） ----
        pd.DataFrame(all_detail_data).to_csv(f"{file_path}\\{prefix}_SourceContribution_Detail.csv", index=False)
        print(f"Export data to {file_path}\\{prefix}_SourceContribution_Detail.csv")

        # ---- 输出 2：按频带聚合的贡献（所有信号类型） ----
        if all_band_data:
            col_map = {"thickness_energy": "Thickness Energy", "thickness_ratio": "Thickness Energy Ratio",
                       "thickness_spl": "Thickness SPL(dB)", "steady_load_energy": "Steady Load Energy",
                       "steady_load_ratio": "Steady Load Energy Ratio", "steady_load_spl": "Steady Load SPL(dB)",
                       "unsteady_load_energy": "Unsteady Load Energy", "unsteady_load_ratio": "Unsteady Load Energy Ratio",
                       "unsteady_load_spl": "Unsteady Load SPL(dB)", "total_energy": "Total Energy",
                       "total_ratio": "Total Energy Ratio", "total_spl": "Total SPL(dB)",
                       "harmonic_energy": "Harmonic Energy", "harmonic_ratio": "Harmonic Energy Ratio",
                       "harmonic_spl": "Harmonic SPL(dB)", "broadband_energy": "Broadband Energy",
                       "broadband_ratio": "Broadband Energy Ratio", "broadband_spl": "Broadband SPL(dB)",
                       "harmonic_ratio_in_band": "Harmonic Ratio in Band"}
            band_df = pd.DataFrame(all_band_data)
            # 为列添加信号类型前缀并应用可读列名映射。
            renamed = {}
            for col in band_df.columns:
                for st in signal_types:
                    if col.startswith(f"{st}_"):
                        base = col[len(st) + 1:]
                        renamed[col] = f"{st}_{col_map.get(base, base)}"
            band_df = band_df.rename(columns=renamed)
            band_df.to_csv(f"{file_path}\\{prefix}_BandSourceContribution.csv", index=False)
            print(f"Export data to {file_path}\\{prefix}_BandSourceContribution.csv")

        # ---- 输出 3：各谐波贡献（所有信号类型） ----
        if all_harmonic_data:
            col_map_h = {"thickness_amp": "Thickness Amplitude(Pa)", "thickness_spl": "Thickness SPL(dB)",
                         "thickness_ratio": "Thickness Energy Ratio", "steady_load_amp": "Steady Load Amplitude(Pa)",
                         "steady_load_spl": "Steady Load SPL(dB)", "steady_load_ratio": "Steady Load Energy Ratio",
                         "unsteady_load_amp": "Unsteady Load Amplitude(Pa)", "unsteady_load_spl": "Unsteady Load SPL(dB)",
                         "unsteady_load_ratio": "Unsteady Load Energy Ratio", "total_amp": "Total Amplitude(Pa)",
                         "total_spl": "Total SPL(dB)", "harmonic_amp": "Harmonic Amplitude(Pa)",
                         "harmonic_spl": "Harmonic SPL(dB)", "broadband_amp": "Broadband Amplitude(Pa)",
                         "broadband_spl": "Broadband SPL(dB)"}
            harmonic_df = pd.DataFrame(all_harmonic_data)
            # 为列添加信号类型前缀并应用可读列名映射。
            renamed = {}
            for col in harmonic_df.columns:
                for st in signal_types:
                    if col.startswith(f"{st}_"):
                        base = col[len(st) + 1:]
                        renamed[col] = f"{st}_{col_map_h.get(base, base)}"
            harmonic_df = harmonic_df.rename(columns=renamed)
            harmonic_df.to_csv(f"{file_path}\\{prefix}_HarmonicContribution.csv", index=False)
            print(f"Export data to {file_path}\\{prefix}_HarmonicContribution.csv")

        # ---- 构建每个观测点的汇总行 ----
        summary_item = {"Filename": prefix, "Fundamental Frequency(Hz)": fundamental_freq,
                        "Number of Harmonics": len(results["harmonic_freqs"])}
        summary_item.update(all_global_stats)
        summary_list.append(summary_item)

    # ---- 输出 4：跨观测点汇总 ----
    if summary_list:
        summary_df = pd.DataFrame(summary_list)
        base_cols = ["Filename", "Fundamental Frequency(Hz)", "Number of Harmonics"]
        # 收集所有源级统计列（按信号类型）。
        src_cols = []
        for st in signal_types:
            for src in ["thickness", "steady_load", "unsteady_load", "total", "harmonic", "broadband"]:
                src_cols.extend([f"{st}_{src}_total_energy", f"{st}_{src}_total_ratio", f"{st}_{src}_total_spl",
                                 f"{st}_{src}_low_freq_energy", f"{st}_{src}_low_freq_ratio",
                                 f"{st}_{src}_mid_freq_energy", f"{st}_{src}_mid_freq_ratio",
                                 f"{st}_{src}_high_freq_energy", f"{st}_{src}_high_freq_ratio"])
                if src == "harmonic":
                    src_cols.append(f"{st}_harmonic_total_ratio")
                    src_cols.append(f"{st}_harmonic_to_broadband_ratio")
        src_cols = [c for c in src_cols if c in summary_df.columns]
        summary_df = summary_df[base_cols + src_cols]
        # 写入汇总表（列名已携带信号类型前缀）。
        summary_df.to_csv(f"{file_path}\\SourceContribution_Summary.csv", index=False)
        print(f"Export data to {file_path}\\SourceContribution_Summary.csv")
        if group_prefixes:
            for gp in group_prefixes:
                gdf = summary_df[summary_df["Filename"].str.startswith(gp)]
                if not gdf.empty:
                    gdf.to_csv(f"{file_path}\\{gp}_SourceContribution_Summary.csv", index=False)
                    print(f"Export data to {file_path}\\{gp}_SourceContribution_Summary.csv")

    print("Analysis completed!")


if __name__ == "__main__":
    # ---- 直接执行示例 ----
    file_path = r"Case04"
    Filename_list = ["Case04_Rotor"]
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
        max_harmonic_order=45, check_phase_consistency=check_phase_consistency)
