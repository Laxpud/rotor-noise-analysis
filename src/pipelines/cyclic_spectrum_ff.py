"""
循环谱分析（仅自由场）-- 基于循环平稳理论对稳态和非稳态载荷贡献进行定量评估。

本模块将 ``CyclicSpectrumAnalyzer`` 应用于自由场载荷噪声时程数据。沿叶片
通过频率（BPF）谐波计算循环谱密度（SCD），并由此可生成以下产品（由
``output`` 参数控制）：

* ``'scd'``       -- 三维循环谱密度幅值（``*_SCD_3D.npz``）。
* ``'coherence'`` -- 循环相干矩阵（``*_CyclicCoherence.csv``）。
* ``'ics'``       -- 积分循环谱（``*_IntegratedCyclicSpectrum.csv``）。
* ``'spectrum'``  -- 重构的稳态/非稳态 PSD 谱（``*_SteadySpectrum.csv``）。
* ``'summary'``   -- 全局贡献指标（``*_CyclicSummary.csv``）。

``'all'`` 输出所有产品；默认值为 ``['ics', 'spectrum', 'summary']``。

输入
------
* ``{prefix}_FF.csv``          -- 时域载荷压力（自由场）。
* ``{prefix}_FreqDomain.csv``  -- （仅在 ``bpf is None`` 时用于自动检测 BPF）。

输出
-------
由 ``output`` 参数控制；参见上述列表。

使用方式
-----------
* 直接运行  : ``python cyclic_spectrum_ff.py``
* 命令行调度: ``python main.py cyclic ...``
"""

import os
import sys

# 确保上级 ``src`` 包在导入路径中。
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
from typing import Optional, List, Union
import os

from decomposition import CyclicSpectrumAnalyzer

P_REF = 20e-6


def _resolve_output(output: Optional[List[str]]) -> List[str]:
    """将用户指定的输出选择解析为具体的列表。

    Parameters
    ----------
    output : list of str or None
        输出类型字符串（例如 ``['scd', 'spectrum']``）。``'all'`` 展开为所有
        可用的输出类型。``None`` 返回默认集合。

    Returns
    -------
    list of str
        具体的输出类型标识符列表。

    Notes
    -----
    可用标识符：``'scd'``、``'coherence'``、``'ics'``、``'spectrum'``、
    ``'summary'``。
    """
    if output is None:
        return ['ics', 'spectrum', 'summary']
    if 'all' in output:
        return ['scd', 'coherence', 'ics', 'spectrum', 'summary']
    return output


def run_cyclic_analysis(
    file_path: str,
    filename_prefix: List[str],
    group_prefixes: Optional[List[str]] = None,
    bpf: Optional[float] = None,
    max_harmonic_order: int = 30,
    output: Optional[List[str]] = None
):
    """对一组观测点的 FF 载荷噪声运行循环谱分析。

    对于每个观测点：
    1. 从 ``{prefix}_FF.csv`` 读取载荷时程数据。
    2. 确定采样率和叶片通过频率（自动或用户指定）。
    3. 通过 ``CyclicSpectrumAnalyzer`` 计算循环谱密度（SCD）。
    4. 输出请求的产品（SCD、相干性、ICS、谱、汇总）。

    Parameters
    ----------
    file_path : str
        包含输入文件并接收输出文件的目录。
    filename_prefix : list of str
        观测点文件名前缀列表。
    group_prefixes : list of str, optional
        （保留以保持一致性；本流程中未使用。）
    bpf : float, optional
        叶片通过（基频）频率，单位为 Hz。如果为 ``None``，则自动检测。
    max_harmonic_order : int, optional
        用于 SCD 计算的最大循环频率谐波阶次。默认值为 30。
    output : list of str, optional
        要生成的输出产品。默认值：``['ics', 'spectrum', 'summary']``。

    Returns
    -------
    None
        结果写入 ``file_path`` 下的文件。

    Notes
    -----
    SCD 幅值 ``|S_x^alpha(f)|`` 量化了由循环频率 ``alpha`` 分隔的频率分量之间
    的相关程度。对 ``f`` 积分得到积分循环谱，重构 alpha=0 和 alpha!=0 的贡献
    分别得到稳态和非稳态 PSD。
    """
    output = _resolve_output(output)

    for prefix in filename_prefix:
        print(f"Processing {prefix}...")

        # ---- 1. 加载时域载荷压力 ----
        time_file = os.path.join(file_path, f"{prefix}_FF.csv")
        time_data = pd.read_csv(time_file)
        # 堆叠 [time; Load_signal] 供分析器使用。
        time_load = np.vstack([time_data['Time'].values, time_data['Load'].values])

        # ---- 2. 从时间步长确定采样率 ----
        sample_rate = 1000.0 / np.mean(np.diff(time_data['Time'].values))

        # ---- 3. 如果未提供则自动检测叶片通过频率 ----
        if bpf is None:
            from spectral import PeakFrequencyAnalyzer
            # 使用预先计算的频域 Total 幅值进行峰值检测。
            fd_file = os.path.join(file_path, f"{prefix}_FreqDomain.csv")
            fd_data = pd.read_csv(fd_file)
            freq = fd_data["Frequency(Hz)"].values
            amp = fd_data["amp_Total(Pa)"].values
            analyzer = PeakFrequencyAnalyzer(freq)
            result = analyzer.analyze_spectrum(amp, prominence=0.005)
            bpf = result["fundamental_freq"]
            print(f"  Auto-identified BPF: {bpf:.2f} Hz")

        # ---- 4. 创建分析器并计算循环谱密度 ----
        csa = CyclicSpectrumAnalyzer(time_load, sample_rate, bpf)
        csa.compute_scd(max_harmonic_order=max_harmonic_order)

        # ---- 5. 输出选定的产品 ----

        if 'scd' in output:
            # 三维循环谱密度幅值：SCD(f, alpha)。
            scd_export = csa.get_scd_3d_export()
            np.savez_compressed(f"{file_path}\\{prefix}_SCD_3D.npz",
                                f=scd_export['f'], alpha=scd_export['alpha'],
                                scd_magnitude=scd_export['scd_magnitude'])
            print(f"  Saved SCD 3D data")

        if 'coherence' in output:
            # 循环相干性：|S_x^alpha(f)|^2 / (S_x^0(f) * S_x^0(f+alpha/2))。
            coh = csa.compute_cyclic_coherence()
            K_max = coh.shape[0] - 1
            col_names = [f"{csa.freq[i]:.1f}" for i in range(len(csa.freq))]
            coh_df = pd.DataFrame(coh, columns=col_names)
            coh_df.insert(0, 'alpha(Hz)', [k * bpf for k in range(K_max + 1)])
            coh_df.to_csv(f"{file_path}\\{prefix}_CyclicCoherence.csv", index=False)
            print(f"  Saved cyclic coherence matrix")

        if 'ics' in output:
            # 积分循环谱：I(alpha) = int |SCD(f, alpha)|^2 df。
            alphas, integrated = csa.compute_integrated_cyclic_spectrum()
            ics_df = pd.DataFrame({'alpha(Hz)': alphas, 'I(alpha)': integrated})
            ics_df.to_csv(f"{file_path}\\{prefix}_IntegratedCyclicSpectrum.csv", index=False)
            print(f"  Saved integrated cyclic spectrum")

        if 'spectrum' in output:
            # 重构的稳态（alpha=0）和非稳态（alpha!=0）PSD。
            steady_psd, unsteady_psd = csa.reconstruct_steady_spectrum()
            sp_df = pd.DataFrame({
                'Frequency(Hz)': csa.freq,
                'Steady_PSD(Pa2)': steady_psd,
                'Unsteady_PSD(Pa2)': unsteady_psd,
                'Steady_SPL(dB)': 10 * np.log10(steady_psd / (P_REF ** 2) + 1e-12),
                'Unsteady_SPL(dB)': 10 * np.log10(unsteady_psd / (P_REF ** 2) + 1e-12),
            })
            sp_df.to_csv(f"{file_path}\\{prefix}_SteadySpectrum.csv", index=False)
            print(f"  Saved steady/unsteady spectrum")

        if 'summary' in output:
            # 全局指标：能量比、SPL 和频率区域分解。
            metrics = csa.compute_metrics()
            summary_row = {'Filename': prefix, 'BPF(Hz)': bpf}
            for key in ['steady_ratio', 'unsteady_ratio', 'steady_spl', 'unsteady_spl',
                        'total_energy', 'steady_energy', 'unsteady_energy']:
                if key in metrics:
                    summary_row[key] = metrics[key]
            for band in ['low', 'mid', 'high']:
                for key in ['steady_ratio', 'unsteady_ratio']:
                    if f'{band}_{key}' in metrics:
                        summary_row[f'{band}_{key}'] = metrics[f'{band}_{key}']
            summary_df = pd.DataFrame([summary_row])
            summary_df.to_csv(f"{file_path}\\{prefix}_CyclicSummary.csv", index=False)
            print(f"  Saved cyclic summary: steady_ratio={metrics['steady_ratio']:.3f}")


if __name__ == "__main__":
    # ---- 直接执行示例（输出所有产品） ----
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

    run_cyclic_analysis(
        file_path=file_path,
        filename_prefix=filename_prefix,
        group_prefixes=Filename_list,
        bpf=46.9698,
        max_harmonic_order=45,
        output=['all']  # 用户选择的输出。
    )
