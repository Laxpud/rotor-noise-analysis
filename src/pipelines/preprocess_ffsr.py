"""
自由场+地面反射（FF+SR）信号预处理：将时域压力数据转换到频域，包含合并
（线性叠加）分量。

本模块同时读取自由场（``_FF.csv``）和地面反射（``_SR.csv``）时域 CSV
文件。对于每个观测点：

1. 分别计算 FF 和 SR 的厚度、载荷和总噪声的实 FFT 幅值/SPL 谱。
2. 生成"合并"谱，其幅值为 FF 与 SR 幅值的直接和（相干叠加）。
3. 计算 FF 和 SR 分量的逐周期总体 SPL 值。

输入
------
* ``{prefix}_FF.csv`` -- 包含 ``Time``、``Thickness``、``Load``、``Total`` 列的 CSV 文件。
* ``{prefix}_SR.csv`` -- 包含 ``Time``、``Thickness``、``Load``、``Total`` 列的 CSV 文件。

输出
-------
* ``{prefix}_merged.csv``     -- FF 和 SR 线性叠加后的时域压力历程。
* ``{prefix}_FreqDomain.csv`` -- FF、SR 和合并分量的频率、幅值和 SPL。
* ``{prefix}_SPLs.csv``       -- 各分量的逐周期总体 SPL（FF、SR 和 merged）。

使用方式
-----------
* 直接运行  : ``python preprocess_ffsr.py``
* 命令行调度: ``python main.py <command> ...``
"""

import os
import sys

# 确保上级 ``src`` 包在导入路径中。
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
from signal_utils import rfft, SPLs


TIME_DOMAIN_COLUMNS = ["Time", "Thickness", "Load", "Total"]


def _build_merged_time_domain(ff_data, sr_data, prefix):
    """生成 FF+SR 融合后的时域压力历程。

    Parameters
    ----------
    ff_data : pandas.DataFrame
        自由场时域数据，包含 ``Time``、``Thickness``、``Load``、``Total`` 列。
    sr_data : pandas.DataFrame
        表面反射时域数据，列格式与 ``ff_data`` 一致。
    prefix : str
        当前观测点文件名前缀，用于错误信息。

    Returns
    -------
    pandas.DataFrame
        与输入同列格式的融合时域数据。
    """
    missing = {
        "FF": [col for col in TIME_DOMAIN_COLUMNS if col not in ff_data.columns],
        "SR": [col for col in TIME_DOMAIN_COLUMNS if col not in sr_data.columns],
    }
    missing = {key: cols for key, cols in missing.items() if cols}
    if missing:
        raise ValueError(
            f"{prefix}: missing required time-domain columns: {missing}"
        )

    if len(ff_data) != len(sr_data):
        raise ValueError(
            f"{prefix}: FF and SR time histories have different lengths."
        )

    ff_time = ff_data["Time"].values
    sr_time = sr_data["Time"].values
    if not np.allclose(ff_time, sr_time):
        raise ValueError(f"{prefix}: FF and SR time axes do not match.")

    return pd.DataFrame({
        "Time": ff_time,
        "Thickness": ff_data["Thickness"].values + sr_data["Thickness"].values,
        "Load": ff_data["Load"].values + sr_data["Load"].values,
        "Total": ff_data["Total"].values + sr_data["Total"].values,
    })


def run_preprocess_ffsr(
    file_path, filename_prefix, cycles=15, export_merged_time=True
):
    """对给定的观测点前缀列表执行 FF+SR 预处理。

    对于每个观测点前缀，该函数：
    1. 读取 FF 和 SR 时域 CSV 文件。
    2. 生成 FF+SR 融合后的时域压力历程。
    3. 计算每种信号类型的实 FFT 幅值/SPL 谱。
    4. 生成合并（相干叠加）谱。
    5. 计算 FF、SR 和 merged 的逐周期总体 SPL。
    6. 写入 ``_merged.csv``、``_FreqDomain.csv`` 和 ``_SPLs.csv``。

    Parameters
    ----------
    file_path : str
        包含输入 CSV 文件并接收输出 CSV 文件的目录。
    filename_prefix : list of str
        文件名前缀列表（例如 ``["Case04_Rotor_OBS0001"]``）。
    cycles : int, optional
        用于逐周期 SPL 的旋翼周期数。默认值为 15。
    export_merged_time : bool, optional
        是否输出 ``{prefix}_merged.csv`` 时域融合文件。默认输出。

    Notes
    -----
    合并幅值通过*直接幅值相加*（相干叠加）获得，即
    ``amp_merged = amp_FF + amp_SR``。对应的 SPL 由合并幅值导出。
    """
    for prefix in filename_prefix:
        # -------- 自由场（FF） --------
        ff_data = pd.read_csv(f"{file_path}\\{prefix}_FF.csv", header=0, sep=',')
        ff_thick = np.vstack([ff_data['Time'].values, ff_data['Thickness'].values])
        ff_load = np.vstack([ff_data['Time'].values, ff_data['Load'].values])
        ff_total = np.vstack([ff_data['Time'].values, ff_data['Total'].values])

        # FF 分量的实 FFT。
        freq, _, amp_ff_thick, spl_ff_thick = rfft(ff_thick)
        _, _, amp_ff_load, spl_ff_load = rfft(ff_load)
        _, _, amp_ff_total, spl_ff_total = rfft(ff_total)

        # -------- 地面反射（SR） --------
        sr_data = pd.read_csv(f"{file_path}\\{prefix}_SR.csv", header=0, sep=',')
        sr_thick = np.vstack([sr_data['Time'].values, sr_data['Thickness'].values])
        sr_load = np.vstack([sr_data['Time'].values, sr_data['Load'].values])
        sr_total = np.vstack([sr_data['Time'].values, sr_data['Total'].values])

        # -------- 时域融合（线性叠加） --------
        merged_data = _build_merged_time_domain(ff_data, sr_data, prefix)
        merged_thick = np.vstack([
            merged_data['Time'].values,
            merged_data['Thickness'].values,
        ])
        merged_load = np.vstack([
            merged_data['Time'].values,
            merged_data['Load'].values,
        ])
        merged_total = np.vstack([
            merged_data['Time'].values,
            merged_data['Total'].values,
        ])

        if export_merged_time:
            merged_data.to_csv(f"{file_path}\\{prefix}_merged.csv", index=False)
            print(f"Export data to {file_path}\\{prefix}_merged.csv")

        # SR 分量的实 FFT。
        _, _, amp_sr_thick, spl_sr_thick = rfft(sr_thick)
        _, _, amp_sr_load, spl_sr_load = rfft(sr_load)
        _, _, amp_sr_total, spl_sr_total = rfft(sr_total)

        # -------- 合并（相干幅值求和） --------
        amp_merged_thick = amp_ff_thick + amp_sr_thick
        amp_merged_load = amp_ff_load + amp_sr_load
        amp_merged_total = amp_ff_total + amp_sr_total

        # 由合并幅值计算 SPL（参考值 20 muPa）。
        spl_merged_thick = 20 * np.log10(amp_merged_thick / 20e-6)
        spl_merged_load = 20 * np.log10(amp_merged_load / 20e-6)
        spl_merged_total = 20 * np.log10(amp_merged_total / 20e-6)

        # -------- 保存频域数据 --------
        freq_data = pd.DataFrame({
            'Frequency(Hz)': freq,
            'amp_FF_Total(Pa)': amp_ff_total, 'SPL_FF_Total(dB)': spl_ff_total,
            'amp_FF_Thickness(Pa)': amp_ff_thick, 'SPL_FF_Thickness(dB)': spl_ff_thick,
            'amp_FF_Load(Pa)': amp_ff_load, 'SPL_FF_Load(dB)': spl_ff_load,
            'amp_SR_Total(Pa)': amp_sr_total, 'SPL_SR_Total(dB)': spl_sr_total,
            'amp_SR_Thickness(Pa)': amp_sr_thick, 'SPL_SR_Thickness(dB)': spl_sr_thick,
            'amp_SR_Load(Pa)': amp_sr_load, 'SPL_SR_Load(dB)': spl_sr_load,
            'amp_merged_Total(Pa)': amp_merged_total, 'SPL_merged_Total(dB)': spl_merged_total,
            'amp_merged_Thickness(Pa)': amp_merged_thick, 'SPL_merged_Thickness(dB)': spl_merged_thick,
            'amp_merged_Load(Pa)': amp_merged_load, 'SPL_merged_Load(dB)': spl_merged_load,
        })
        freq_data.to_csv(f"{file_path}\\{prefix}_FreqDomain.csv", index=False)
        print(f"Export data to {file_path}\\{prefix}_FreqDomain.csv")

        # -------- 逐周期 SPL（每个旋翼转数的总体声压级） --------
        spls_ff_thick = SPLs(ff_thick, cycles)
        spls_ff_load = SPLs(ff_load, cycles)
        spls_ff_total = SPLs(ff_total, cycles)
        spls_sr_thick = SPLs(sr_thick, cycles)
        spls_sr_load = SPLs(sr_load, cycles)
        spls_sr_total = SPLs(sr_total, cycles)
        spls_merged_thick = SPLs(merged_thick, cycles)
        spls_merged_load = SPLs(merged_load, cycles)
        spls_merged_total = SPLs(merged_total, cycles)
        spls_data = pd.DataFrame({
            'Cycle': range(1, cycles + 1),
            'SPL_FF_Thickness(dB)': spls_ff_thick,
            'SPL_FF_Load(dB)': spls_ff_load,
            'SPL_FF_Total(dB)': spls_ff_total,
            'SPL_SR_Thickness(dB)': spls_sr_thick,
            'SPL_SR_Load(dB)': spls_sr_load,
            'SPL_SR_Total(dB)': spls_sr_total,
            'SPL_merged_Thickness(dB)': spls_merged_thick,
            'SPL_merged_Load(dB)': spls_merged_load,
            'SPL_merged_Total(dB)': spls_merged_total,
        })
        spls_data.to_csv(f"{file_path}\\{prefix}_SPLs.csv", index=False)
        print(f"Export data to {file_path}\\{prefix}_SPLs.csv")


if __name__ == "__main__":
    # ---- 直接执行示例 ----
    file_path = r"data\Case03"
    Filename_list = ["Case03_Rotor"]
    OBS_Numbers = 12
    filename_prefix = [f"{Filename_list[0]}_OBS{j + 1:04d}" for j in range(OBS_Numbers)]
    run_preprocess_ffsr(file_path, filename_prefix)
