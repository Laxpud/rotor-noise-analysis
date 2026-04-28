"""
自由场（FF）信号预处理：将时域压力数据转换到频域。

本模块读取包含压力分量信号（厚度噪声、载荷噪声、总噪声）的自由场时域
CSV 文件，并通过实 FFT 将其转换到频域。同时计算每个分量在用户指定
旋翼转数内的逐周期 SPL 值。

输入
------
* ``{prefix}_FF.csv`` -- 包含 ``Time``、``Thickness``、``Load``、``Total`` 列的 CSV 文件。

输出
-------
* ``{prefix}_FreqDomain.csv`` -- 各分量的频率、幅值和 SPL 谱。
* ``{prefix}_SPLs.csv``       -- 各分量逐周期的总体 SPL 值。

使用方式
-----------
* 直接运行  : ``python preprocess_ff.py``
* 命令行调度: ``python main.py <command> ...``
"""

import os
import sys

# 确保上级 ``src`` 包在导入路径中。
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
from signal_utils import rfft, SPLs


def run_preprocess_ff(file_path, filename_prefix, cycles=5):
    """对给定的观测点前缀列表执行 FF 预处理。

    对于每个观测点前缀，该函数：
    1. 读取时域 FF CSV 文件。
    2. 计算厚度、载荷和总噪声的实 FFT 幅值/SPL 谱。
    3. 在 ``cycles`` 个旋翼转数内计算逐周期（总体）SPL 值。
    4. 写入两个 CSV 文件：``_FreqDomain.csv`` 和 ``_SPLs.csv``。

    Parameters
    ----------
    file_path : str
        包含输入 CSV 文件并接收输出 CSV 文件的目录。
    filename_prefix : list of str
        文件名前缀列表（例如 ``["Case01_Rotor_OBS0001"]``）。
    cycles : int, optional
        用于逐周期 SPL 计算的旋翼周期数。默认值为 5。

    Notes
    -----
    该函数期望 ``{prefix}_FF.csv`` 文件存在于 ``file_path`` 目录下。
    """
    for prefix in filename_prefix:
        # ---- 1. 加载时域数据 ----
        time_data = pd.read_csv(f"{file_path}\\{prefix}_FF.csv", header=0, sep=',')
        # 将时间轴与信号值堆叠，供 rfft/SPLs 辅助函数使用。
        time_thick = np.vstack([time_data['Time'].values, time_data['Thickness'].values])
        time_load = np.vstack([time_data['Time'].values, time_data['Load'].values])
        time_total = np.vstack([time_data['Time'].values, time_data['Total'].values])

        # ---- 2. 实 FFT → 频率、幅值、SPL 谱 ----
        freq, _, amp_thick, spl_thick = rfft(time_thick)
        _, _, amp_load, spl_load = rfft(time_load)
        _, _, amp_total, spl_total = rfft(time_total)

        # ---- 3. 保存频域数据 ----
        freq_data = pd.DataFrame({
            'Frequency(Hz)': freq,
            'amp_Total(Pa)': amp_total, 'SPL_Total(dB)': spl_total,
            'amp_Thickness(Pa)': amp_thick, 'SPL_Thickness(dB)': spl_thick,
            'amp_Load(Pa)': amp_load, 'SPL_Load(dB)': spl_load,
        })
        freq_data.to_csv(f"{file_path}\\{prefix}_FreqDomain.csv", index=False)
        print(f"Export data to {file_path}\\{prefix}_FreqDomain.csv")

        # ---- 4. 逐周期 SPL（每个旋翼转数的总体声压级） ----
        spls_thick = SPLs(time_thick, cycles)
        spls_load = SPLs(time_load, cycles)
        spls_total = SPLs(time_total, cycles)
        spls_data = pd.DataFrame({
            'Cycle': range(1, cycles + 1),
            'SPL_Thickness(dB)': spls_thick,
            'SPL_Load(dB)': spls_load,
            'SPL_Total(dB)': spls_total,
        })
        spls_data.to_csv(f"{file_path}\\{prefix}_SPLs.csv", index=False)
        print(f"Export data to {file_path}\\{prefix}_SPLs.csv")


if __name__ == "__main__":
    # ---- 直接执行示例 ----
    file_path = r"Case01"
    Filename_list = ["Case01_Rotor"]
    OBS_Numbers = 12
    filename_prefix = [f"{Filename_list[0]}_OBS{j + 1:04d}" for j in range(OBS_Numbers)]
    run_preprocess_ff(file_path, filename_prefix)
