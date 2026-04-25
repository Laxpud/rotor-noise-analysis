import numpy as np

def SPL(time_signal, p_ref=20e-6):
    """
    计算输入信号的声压级，返回一个声压级值。

    参数:
    time_signal (np.array): 输入的时间信号的二维数组，第一列为时间，第二列为幅值。
    p_ref (float): 参考压力值 (Pa)。

    返回:
    SPL (float): 输入信号的声压级 (dB)。
    """    
    # 计算声压级
    SPL = 20 * np.log10(np.sqrt(np.mean(time_signal[1, :]**2))) - 20 * np.log10(p_ref)
    return SPL
    
def SPLs(time_signal, cycles, p_ref=20e-6):
    """
    计算输入信号的每个周期的声压级，返回一个包含每个周期声压级的数组。

    参数:
    time_signal (np.array): 输入的时间信号的二维数组，第一列为时间，第二列为幅值。
    p_ref (float): 参考压力值 (Pa)。
    cycles (int): 周期数。

    返回:
    SPL (np.array): 每个周期的声压级 (dB)，数组长度为 cycles。
    """
    # 计算每个周期的声压级
    SPLs = np.zeros(cycles)
    for i in range(cycles):
        SPLs[i] = 20 * np.log10(np.sqrt(np.mean(time_signal[1, i*len(time_signal[0])//cycles:(i+1)*len(time_signal[0])//cycles]**2))) - 20 * np.log10(p_ref)
    return SPLs

def rfft(time_signal, p_ref=20e-6, return_phase=False, unwrap_phase=True):
    """
    计算输入信号的傅里叶变换，返回频率、复数傅里叶变换、幅度谱、声压级，可选返回相位谱。

    参数:
    time_signal (np.array): 输入的时间信号的二维数组，第一列为时间，第二列为幅值。
    p_ref (float): 参考压力值 (Pa)。
    return_phase (bool): 是否返回相位谱，默认False。
    unwrap_phase (bool): 是否对相位谱进行解缠绕处理，默认True。

    返回:
    freq (np.array): 频率数组。
    fft (np.array): 复数傅里叶变换数组。
    amp_fft (np.array): 幅度谱数组。
    SPL_fft (np.array): 声压级数组 (dB)。
    phase_fft (np.array, 可选): 相位谱数组（弧度），仅当return_phase=True时返回。
    """
    N = len(time_signal[0])
    time_interval = np.mean(np.diff(time_signal[0]))/1000
    freq = np.fft.rfftfreq(N, d=time_interval)
    fft = np.fft.rfft(time_signal[1]) + 1e-10
    # 计算rfft返回的幅度谱，注意直流分量需要除以直接除以N，交流分量需要乘以2.0再除以N
    amp_fft = 2.0/N * np.abs(fft)
    amp_fft[0] = amp_fft[0] / 2.0
    SPL_fft = 20 * np.log10(amp_fft / p_ref)

    if return_phase:
        phase_fft = np.angle(fft)
        if unwrap_phase:
            phase_fft = np.unwrap(phase_fft)
        return freq, fft, amp_fft, SPL_fft, phase_fft

    return freq, fft, amp_fft, SPL_fft
