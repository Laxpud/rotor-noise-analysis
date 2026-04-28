'''
声学时域信号处理工具模块。

提供根据时域压力信号计算声压级 (SPL) 以及执行实数 FFT 以获取
频域谱的函数。

所有 SPL 计算均使用标准空气参考声压：P_REF = 20 uPa。

Functions
---------
SPL : 根据时域信号计算总声压级
SPLs : 计算各周期的声压级
rfft : 计算实数 FFT，返回频率轴、复频谱、幅值谱和 SPL 频谱
'''

import numpy as np

# 空气参考声压 (20 微帕)
P_REF = 20e-6


def SPL(time_signal, p_ref=P_REF):
    '''
    计算总声压级 (Sound Pressure Level, SPL)，单位为 dB。

    SPL 定义为：
        SPL = 20 * log10(均方根压力 / p_ref)

    Parameters
    ----------
    time_signal : ndarray
        形状为 (2, N) 的数组。第 0 行为时间（秒），
        第 1 行为压力幅值（帕斯卡）。
    p_ref : float, optional
        参考声压，单位为帕斯卡。默认为 20e-6 (20 uPa)。

    Returns
    -------
    spl : float
        声压级，单位为 dB。
    '''
    # 从信号行计算 RMS 压力并转换为 dB
    return 20 * np.log10(np.sqrt(np.mean(time_signal[1, :] ** 2))) - 20 * np.log10(p_ref)


def SPLs(time_signal, cycles, p_ref=P_REF):
    '''
    计算给定周期数下每个周期的声压级。

    将时域信号均匀划分为 ``cycles`` 个片段，对每个片段独立计算 SPL。

    Parameters
    ----------
    time_signal : ndarray
        形状为 (2, N) 的数组。第 0 行为时间，第 1 行为
        压力幅值（帕斯卡）。
    cycles : int
        将信号划分的周期数（片段数）。
    p_ref : float, optional
        参考声压，单位为帕斯卡。默认为 20e-6 (20 uPa)。

    Returns
    -------
    spls : ndarray
        形状为 (cycles,) 的数组，包含各周期的 SPL 值（dB）。
    '''
    result = np.zeros(cycles)
    N = len(time_signal[0])
    for i in range(cycles):
        # 提取信号的第 i 个片段
        seg = time_signal[1, i * N // cycles : (i + 1) * N // cycles]
        # 计算该片段的 SPL
        result[i] = 20 * np.log10(np.sqrt(np.mean(seg ** 2))) - 20 * np.log10(p_ref)
    return result


def rfft(time_signal, p_ref=P_REF, return_phase=False, unwrap_phase=True):
    '''
    对时域信号执行实数 FFT 并返回频域谱。

    使用 `numpy.fft.rfft` 以提高纯实数输入的计算效率。幅值谱经过缩放，
    使每个分量（除 DC 外）代表对应正弦分量的幅值。在幅值计算前，
    对 FFT 结果添加一个小的常数 (1e-10)，以避免在 SPL 转换时出现
    零值取对数的问题。

    Parameters
    ----------
    time_signal : ndarray
        形状为 (2, N) 的数组。第 0 行为时间（**毫秒**），
        第 1 行为压力幅值（帕斯卡）。
    p_ref : float, optional
        参考声压，单位为帕斯卡。默认为 20e-6 (20 uPa)。
    return_phase : bool, optional
        若为 True，同时返回相位谱。默认为 False。
    unwrap_phase : bool, optional
        若为 True（且 ``return_phase`` 为 True），对相位角进行解包裹
        以消除不连续性。默认为 True。

    Returns
    -------
    freq : ndarray
        频率轴，单位为 Hz（仅正频率部分）。
    fft_complex : ndarray
        复数 FFT 结果（仅正频率部分）。
    amp : ndarray
        缩放至峰值幅值的幅值谱。DC 分量（索引 0）
        减半以代表均值而非峰峰值。
    SPL_fft : ndarray
        SPL 频谱，单位为 dB，由 ``20 * log10(amp / p_ref)`` 计算得到。
    phase : ndarray, optional
        相位谱，单位为弧度。仅当 ``return_phase`` 为 True 时返回。
    '''
    N = len(time_signal[0])
    # 将时间间隔由毫秒转换为秒
    time_interval = np.mean(np.diff(time_signal[0])) / 1000
    # 实数 FFT 的频率轴（仅正频率部分）
    freq = np.fft.rfftfreq(N, d=time_interval)
    # 实数 FFT；添加微小常数以防止计算 SPL 时出现 log(0)
    fft = np.fft.rfft(time_signal[1]) + 1e-10
    # 缩放至峰值幅值：单边谱乘以 2/N
    amp = 2.0 / N * np.abs(fft)
    # DC 分量在单边谱中不应加倍，因此将其减半
    amp[0] = amp[0] / 2.0
    # 将幅值谱转换为 dB SPL
    SPL_fft = 20 * np.log10(amp / p_ref)

    if return_phase:
        # 从复数 FFT 中提取相位角
        phase = np.angle(fft)
        if unwrap_phase:
            # 消除 2*pi 跳变以获得连续的相位表示
            phase = np.unwrap(phase)
        return freq, fft, amp, SPL_fft, phase

    return freq, fft, amp, SPL_fft
