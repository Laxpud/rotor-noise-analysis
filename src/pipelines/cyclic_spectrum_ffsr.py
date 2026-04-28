"""
循环谱分析（自由场+地面反射）-- 基于循环平稳理论分别评估 FF、SR 和合并信号
的稳态/非稳态载荷贡献。

本模块将 ``CyclicSpectrumAnalyzer`` 应用于自由场（FF）、地面反射（SR）和合并
（FF+SR）信号的载荷噪声时程数据。对于每种信号类型，计算循环谱密度（SCD），
并且可以输出与 FF-only 流程相同的产品集合（由 ``output`` 参数控制）。

每种信号类型的输出产品：
* ``'scd'``       -- 三维 SCD 幅值（``*_<type>_SCD_3D.npz``）。
* ``'coherence'`` -- 循环相干矩阵（``*_<type>_CyclicCoherence.csv``）。
* ``'ics'``       -- 积分循环谱（``*_<type>_IntegratedCyclicSpectrum.csv``）。
* ``'spectrum'``  -- 稳态/非稳态 PSD 谱（``*_<type>_SteadySpectrum.csv``）。
* ``'summary'``   -- 全局贡献指标（``*_<type>_CyclicSummary.csv``）。

``'all'`` 输出所有产品；默认值为 ``['ics', 'spectrum', 'summary']``。

输入
------
* ``{prefix}_FF.csv``         -- 自由场时域载荷压力。
* ``{prefix}_SR.csv``         -- 地面反射时域载荷压力。
* ``{prefix}_FreqDomain.csv`` -- （仅在 ``bpf is None`` 时用于自动检测 BPF）。

输出
-------
由 ``output`` 参数控制；每种信号类型各生成一组文件。

使用方式
-----------
* 直接运行  : ``python cyclic_spectrum_ffsr.py``
* 命令行调度: ``python main.py cyclic ...``
"""

import os
import sys

# 确保上级 ``src`` 包在导入路径中。
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
from typing import Optional, List
import os

from decomposition import CyclicSpectrumAnalyzer

P_REF = 20e-6
SIGNAL_TYPES = ["FF", "SR", "merged"]


def _resolve_output(output: Optional[List[str]]) -> List[str]:
    """将用户指定的输出选择解析为具体的列表。

    Parameters
    ----------
    output : list of str or None
        输出类型字符串。``'all'`` 展开为所有可用类型。``None`` 返回默认集合。

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
    """对 FF、SR 和合并载荷噪声信号运行循环谱分析。

    对于每个观测点，三种信号类型（FF、SR、合并）的载荷时程数据独立地通过
    ``CyclicSpectrumAnalyzer`` 进行处理。所有请求的输出产品都会为每种信号类型
    生成，文件名中分别附加 ``_FF``、``_SR`` 或 ``_merged`` 后缀。

    Parameters
    ----------
    file_path : str
        包含输入文件并接收输出文件的目录。
    filename_prefix : list of str
        观测点文件名前缀列表。
    group_prefixes : list of str, optional
        （保留以保持一致性；本流程中未使用。）
    bpf : float, optional
        叶片通过频率，单位为 Hz。如果为 ``None``，则从合并 Total 幅值谱中
        自动检测。
    max_harmonic_order : int, optional
        最大循环频率谐波阶次。默认值为 30。
    output : list of str, optional
        要生成的输出产品。默认值：``['ics', 'spectrum', 'summary']``。

    Returns
    -------
    None
        结果写入 ``file_path`` 下的文件，文件名带有信号类型后缀。

    Notes
    -----
    合并信号在分析前通过 FF 和 SR 载荷压力时程的直接相加得到。
    """
    output = _resolve_output(output)

    for prefix in filename_prefix:
        print(f"Processing {prefix}...")

        # ---- 1. 加载 FF 时域数据并确定采样率 ----
        ff_file = os.path.join(file_path, f"{prefix}_FF.csv")
        ff_data = pd.read_csv(ff_file)
        sample_rate = 1000.0 / np.mean(np.diff(ff_data['Time'].values))

        # 堆叠 [time; Load_signal] 供 FF 使用。
        ff_load = np.vstack([ff_data['Time'].values, ff_data['Load'].values])

        # ---- 2. 加载 SR 时域数据 ----
        sr_file = os.path.join(file_path, f"{prefix}_SR.csv")
        sr_data = pd.read_csv(sr_file)
        sr_load = np.vstack([sr_data['Time'].values, sr_data['Load'].values])

        # ---- 3. 合成合并信号（时域直接相加） ----
        merged_load = np.vstack([ff_data['Time'].values,
                                 ff_data['Load'].values + sr_data['Load'].values])

        # 组织信号供按类型分析循环使用。
        signal_loads = {"FF": ff_load, "SR": sr_load, "merged": merged_load}

        # ---- 4. 如果需要则自动检测叶片通过频率 ----
        if bpf is None:
            from spectral import PeakFrequencyAnalyzer
            fd_file = os.path.join(file_path, f"{prefix}_FreqDomain.csv")
            fd_data = pd.read_csv(fd_file)
            freq = fd_data["Frequency(Hz)"].values
            amp = fd_data["amp_merged_Total(Pa)"].values
            analyzer = PeakFrequencyAnalyzer(freq)
            result = analyzer.analyze_spectrum(amp, prominence=0.005)
            bpf = result["fundamental_freq"]
            print(f"  Auto-identified BPF: {bpf:.2f} Hz")

        # ---- 5. 对每种信号类型运行循环分析 ----
        for signal_type in SIGNAL_TYPES:
            print(f"  Processing {signal_type} component...")
            time_load = signal_loads[signal_type]

            # 创建分析器并计算循环谱密度。
            csa = CyclicSpectrumAnalyzer(time_load, sample_rate, bpf)
            csa.compute_scd(max_harmonic_order=max_harmonic_order)

            # 输出文件名中使用的后缀。
            suffix = f"_{signal_type}"

            # ---- 输出产品（每种信号类型） ----

            if 'scd' in output:
                # 三维循环谱密度幅值。
                scd_export = csa.get_scd_3d_export()
                np.savez_compressed(f"{file_path}\\{prefix}{suffix}_SCD_3D.npz",
                                    f=scd_export['f'], alpha=scd_export['alpha'],
                                    scd_magnitude=scd_export['scd_magnitude'])

            if 'coherence' in output:
                # 循环相干矩阵。
                coh = csa.compute_cyclic_coherence()
                K_max = coh.shape[0] - 1
                col_names = [f"{csa.freq[i]:.1f}" for i in range(len(csa.freq))]
                coh_df = pd.DataFrame(coh, columns=col_names)
                coh_df.insert(0, 'alpha(Hz)', [k * bpf for k in range(K_max + 1)])
                coh_df.to_csv(f"{file_path}\\{prefix}{suffix}_CyclicCoherence.csv", index=False)
                print(f"Export data to {file_path}\\{prefix}{suffix}_CyclicCoherence.csv")

            if 'ics' in output:
                # 积分循环谱。
                alphas, integrated = csa.compute_integrated_cyclic_spectrum()
                ics_df = pd.DataFrame({'alpha(Hz)': alphas, 'I(alpha)': integrated})
                ics_df.to_csv(f"{file_path}\\{prefix}{suffix}_IntegratedCyclicSpectrum.csv", index=False)
                print(f"Export data to {file_path}\\{prefix}{suffix}_IntegratedCyclicSpectrum.csv")

            if 'spectrum' in output:
                # 重构的稳态/非稳态 PSD。
                steady_psd, unsteady_psd = csa.reconstruct_steady_spectrum()
                sp_df = pd.DataFrame({
                    'Frequency(Hz)': csa.freq,
                    'Steady_PSD(Pa2)': steady_psd,
                    'Unsteady_PSD(Pa2)': unsteady_psd,
                    'Steady_SPL(dB)': 10 * np.log10(steady_psd / (P_REF ** 2) + 1e-12),
                    'Unsteady_SPL(dB)': 10 * np.log10(unsteady_psd / (P_REF ** 2) + 1e-12),
                })
                sp_df.to_csv(f"{file_path}\\{prefix}{suffix}_SteadySpectrum.csv", index=False)
                print(f"Export data to {file_path}\\{prefix}{suffix}_SteadySpectrum.csv")

            if 'summary' in output:
                # 全局贡献指标。
                metrics = csa.compute_metrics()
                summary_row = {'Filename': f"{prefix}{suffix}", 'BPF(Hz)': bpf}
                for key in ['steady_ratio', 'unsteady_ratio', 'steady_spl', 'unsteady_spl',
                            'total_energy', 'steady_energy', 'unsteady_energy']:
                    if key in metrics:
                        summary_row[key] = metrics[key]
                for band in ['low', 'mid', 'high']:
                    for key in ['steady_ratio', 'unsteady_ratio']:
                        if f'{band}_{key}' in metrics:
                            summary_row[f'{band}_{key}'] = metrics[f'{band}_{key}']
                pd.DataFrame([summary_row]).to_csv(
                    f"{file_path}\\{prefix}{suffix}_CyclicSummary.csv", index=False)
                print(f"Export data to {file_path}\\{prefix}{suffix}_CyclicSummary.csv")
                print(f"    steady_ratio={metrics['steady_ratio']:.3f}")

        print(f"  Done {prefix}")


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

    run_cyclic_analysis(
        file_path=file_path,
        filename_prefix=filename_prefix,
        group_prefixes=Filename_list,
        bpf=46.9698,
        max_harmonic_order=45,
        output=['all']  # 用户选择的输出。
    )
