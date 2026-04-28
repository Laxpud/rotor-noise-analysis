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
* ``{prefix}_FreqDomain.csv`` -- FF、SR 和合并分量的频率、幅值和 SPL。
* ``{prefix}_SPLs.csv``       -- 各分量的逐周期总体 SPL（仅 FF 和 SR）。

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


def run_preprocess_ffsr(file_path, filename_prefix, cycles=5):
    """对给定的观测点前缀列表执行 FF+SR 预处理。

    对于每个观测点前缀，该函数：
    1. 读取 FF 和 SR 时域 CSV 文件。
    2. 计算每种信号类型的实 FFT 幅值/SPL 谱。
    3. 生成合并（相干叠加）谱。
    4. 计算 FF 和 SR 的逐周期总体 SPL。
    5. 写入 ``_FreqDomain.csv`` 和 ``_SPLs.csv``。

    Parameters
    ----------
    file_path : str
        包含输入 CSV 文件并接收输出 CSV 文件的目录。
    filename_prefix : list of str
        文件名前缀列表（例如 ``["Case04_Rotor_OBS0001"]``）。
    cycles : int, optional
        用于逐周期 SPL 的旋翼周期数。默认值为 5。

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
        spls_data = pd.DataFrame({
            'Cycle': range(1, cycles + 1),
            'SPL_FF_Thickness(dB)': spls_ff_thick,
            'SPL_FF_Load(dB)': spls_ff_load,
            'SPL_FF_Total(dB)': spls_ff_total,
            'SPL_SR_Thickness(dB)': spls_sr_thick,
            'SPL_SR_Load(dB)': spls_sr_load,
            'SPL_SR_Total(dB)': spls_sr_total,
        })
        spls_data.to_csv(f"{file_path}\\{prefix}_SPLs.csv", index=False)
        print(f"Export data to {file_path}\\{prefix}_SPLs.csv")


if __name__ == "__main__":
    # ---- 直接执行示例 ----
    file_path = r"Case04"
    Filename_list = ["Case04_Rotor"]
    OBS_Numbers = 12
    filename_prefix = [f"{Filename_list[0]}_OBS{j + 1:04d}" for j in range(OBS_Numbers)]
    run_preprocess_ffsr(file_path, filename_prefix)
