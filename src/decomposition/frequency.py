"""
谐频/宽频分离模块 (Harmonic/Broadband Frequency Separation).

本模块提供从声压频谱中将总噪声分解为谐频分量（由旋转机械产生的
离散频率噪声）与宽频分量（湍流等随机过程产生的连续谱噪声）的功能。

核心算法
--------
以已知的谐频频率（如 BPF 及其倍频）为中心，在每一谐频周围取一个
窄带（带宽 = 谐频 * bandwidth_ratio），将带内能量归为谐频分量，
带外能量归为宽频分量。分离后验证能量守恒：原频谱总能量应等于
谐频能量 + 宽频能量（允许微小数值误差）。

使用示例
--------
>>> separator = FrequencySeparator(freqs)
>>> harmonic, broadband = separator.separate_by_harmonic_extraction(
...     spectrum, harmonic_freqs=np.array([120, 240, 360]))
"""

import numpy as np
from typing import Tuple


class FrequencySeparator:
    """谐频/宽频分离器 -- 从频谱中提取谐频分量与宽频分量.

    通过谐频提取法 (harmonic extraction) 将总频谱分解为谐频部分
    和宽频部分。在每一谐频周围设定一个以 bandwidth_ratio 控制的
    窄带，带内能量归入谐频，带外能量归入宽频。

    Parameters
    ----------
    frequencies : np.ndarray
        频谱对应的频率轴 (Hz)，一维数组。

    Attributes
    ----------
    freqs : np.ndarray
        存储的频率轴。

    Notes
    -----
    - bandwidth_ratio 的选择影响分离精度：过大会将宽频泄露计入谐频，
      过小则可能遗漏谐频旁瓣能量。默认值 0.03 (3%) 对多数旋翼噪声
      场景适用。
    - 分离后自动验证能量守恒，若误差 > 1% 会打印警告。
    """

    def __init__(self, frequencies: np.ndarray):
        """初始化频率分离器.

        Parameters
        ----------
        frequencies : np.ndarray
            频谱对应的频率轴 (Hz)，一维数组。
        """
        self.freqs = frequencies

    def separate_by_harmonic_extraction(
        self,
        spectrum: np.ndarray,
        harmonic_freqs: np.ndarray,
        bandwidth_ratio: float = 0.03
    ) -> Tuple[np.ndarray, np.ndarray]:
        """通过谐频提取法将频谱分离为谐频分量和宽频分量.

        算法步骤
        --------
        1. 初始化：harmonic_spectrum 全零，broadband_spectrum 为原频谱副本。
        2. 遍历每个谐频：
           a. 计算带宽 = freq * bandwidth_ratio
           b. 确定频率窗口 [freq - bw/2, freq + bw/2]
           c. 找到窗口内的所有频率索引
           d. 将窗口内的频谱值复制到 harmonic_spectrum
           e. 将 broadband_spectrum 对应位置置零
        3. 验证能量守恒：|E_original - (E_harmonic + E_broadband)| / E_original < 1%

        Parameters
        ----------
        spectrum : np.ndarray
            输入的幅值频谱（Pa），一维实数数组。
        harmonic_freqs : np.ndarray
            已知的谐频频率列表（Hz），如 BPF 的整数倍。
        bandwidth_ratio : float, optional
            带宽与中心频率的比值，默认 0.03（即 3% 相对带宽）。

        Returns
        -------
        harmonic_spectrum : np.ndarray
            谐频分量的幅值频谱 -- 仅在谐频带内有值，其余为零。
        broadband_spectrum : np.ndarray
            宽频分量的幅值频谱 -- 谐频带内置零，其余保留原值。

        Notes
        -----
        - 多个谐频窗口可能重叠（当 bandwidth_ratio 较大时），此时
          重叠区域的能量会被最后一个遍历到的谐频覆盖。
        - 能量守恒使用 sum(amplitude^2) 作为能量度量，适用于幅值谱。
        """
        # 初始化：谐频谱全零，宽频谱为原谱副本
        harmonic_spectrum = np.zeros_like(spectrum)
        broadband_spectrum = spectrum.copy()

        # 遍历每个谐频，提取带内能量
        for freq in harmonic_freqs:
            # 计算以 freq 为中心的窄带边界
            bandwidth = freq * bandwidth_ratio
            lower = freq - bandwidth / 2
            upper = freq + bandwidth / 2

            # 找到落在当前频带内的所有频率索引
            indices = np.where((self.freqs >= lower) & (self.freqs <= upper))[0]

            if len(indices) > 0:
                # 将带内幅值复制到谐频谱
                harmonic_spectrum[indices] = spectrum[indices]
                # 从宽频谱中移除带内能量
                broadband_spectrum[indices] = 0

        # --- 能量守恒验证 ---
        # 使用幅值平方和作为能量度量 (Parseval 关系)
        original_energy = np.sum(spectrum ** 2)
        separated_energy = (
            np.sum(harmonic_spectrum ** 2) + np.sum(broadband_spectrum ** 2)
        )
        energy_error = (
            abs(separated_energy - original_energy) / original_energy
            if original_energy > 0 else 0
        )

        # 如果能量误差超过 1%，打印警告
        if energy_error > 0.01:
            print(
                f"Warning: Energy conservation error {energy_error:.2%} "
                f"in frequency separation"
            )

        return harmonic_spectrum, broadband_spectrum
