'''
峰值频率检测与谐波识别模块。

提供 `PeakFrequencyAnalyzer` 类，该类封装了 SciPy 的
`find_peaks` 函数，并附加了从频域频谱中检测基频及其
谐波的逻辑。

谐波识别算法的流程为：
  1. 对所有检测到的峰值两两配对，寻找比值接近整数
     （在可配置容差范围内）的峰值对。
  2. 根据这些比值推断可能的基频。
  3. 以峰值幅值之和进行投票/加权，选出最可能的基频。
  4. 在完整频率轴上搜索所识别基频的整数倍谐波。
'''

import numpy as np
from scipy import signal


class PeakFrequencyAnalyzer:
    '''
    分析频域频谱以检测峰值并识别基频和谐波频率。

    该类使用频率轴数组进行初始化，提供查找频谱峰值以及
    将峰值分类为基频及其关联谐波的方法。

    Parameters
    ----------
    freqs : array_like
        与频谱各频率单元对应的一维频率值数组（Hz）。

    Notes
    -----
    谐波识别算法采用两两比较策略：对每一对峰值，检查其频率比值
    是否接近整数。若是，则将较小的频率（或推断出的子谐波）作为
    候选基频。选择累积峰值幅值权重最高的基频作为最终结果。

    该算法至少需要两个检测到的峰值。若找到的峰值不足两个，
    ``identify_fundamental_and_harmonics`` 将返回
    ``(None, [])``。
    '''

    def __init__(self, freqs):
        '''将频率轴存储为 NumPy 数组。'''
        self.freqs = np.asarray(freqs)

    def find_peaks(self, spectrum, height=None, distance=None, prominence=None,
                   width=None, rel_height=0.5):
        '''
        使用 SciPy 的 `signal.find_peaks` 检测频谱中的峰值。

        Parameters
        ----------
        spectrum : ndarray
            一维频谱幅值（或 SPL 值）数组。
        height : float or tuple, optional
            所需峰值高度。直接传递给 `scipy.signal.find_peaks`。
        distance : float, optional
            峰值之间所需的最小水平距离。
        prominence : float, optional
            所需峰值突出度。
        width : float or tuple, optional
            所需峰值宽度。
        rel_height : float, optional
            测量峰值宽度时的相对高度。默认为 0.5。

        Returns
        -------
        peak_indices : ndarray
            检测到的峰值在频率/频谱数组中的索引。
        peak_properties : dict
            `scipy.signal.find_peaks` 返回的峰值属性字典。
        '''
        peak_indices, peak_properties = signal.find_peaks(
            spectrum, height=height, distance=distance,
            prominence=prominence, width=width, rel_height=rel_height
        )
        return peak_indices, peak_properties

    def identify_fundamental_and_harmonics(self, peak_freqs, peak_magnitudes,
                                           all_freqs=None, all_magnitudes=None,
                                           tolerance=0.05, max_harmonic_order=30):
        '''
        从一组峰值中识别基频及其谐波。

        算法分两个阶段执行：

        **阶段 1 — 候选生成。** 对每一对峰值频率
        ``(f1, f2)``（其中 ``f2 > f1``），计算比值 ``r = f2 / f1``。
        若 ``r`` 接近某个整数 ``n``（在 ``tolerance`` 相对误差范围内），
        则推断候选基频 ``f_candidate = min(f1, f2/n)``。
        候选基频的权重为两个峰值幅值之和。

        **阶段 2 — 投票与谐波搜索。** 将候选基频按四舍五入后的
        基频值分组，选择总权重最高的组。然后在完整频率网格上
        搜索选定基频整数倍的谐波。

        Parameters
        ----------
        peak_freqs : ndarray
            检测到的峰值频率（Hz）。
        peak_magnitudes : ndarray
            检测到的峰值幅值（或 SPL 值）。
        all_freqs : ndarray, optional
            用于搜索谐波的完整频率轴。若为 None，则跳过谐波搜索，
            仅返回基频。
        all_magnitudes : ndarray, optional
            完整频谱幅值（在搜索逻辑中当前未使用）。
        tolerance : float, optional
            接受整数比值的相对容差。默认为 0.05（5%）。
        max_harmonic_order : int, optional
            考虑的最大谐波阶数。默认为 30。

        Returns
        -------
        fundamental_freq : float or None
            识别出的基频，单位为 Hz。若无法确定基频则返回 None。
        harmonic_indices : list of int
            与识别出的谐波对应的 ``all_freqs`` 索引列表。
            若 ``all_freqs`` 为 None 或无匹配谐波则为空列表。
        '''
        # 推断基频至少需要两个峰值
        if len(peak_freqs) < 2:
            return None, []

        peak_freqs = np.asarray(peak_freqs)
        peak_magnitudes = np.asarray(peak_magnitudes)
        possible_fundamentals = []

        # ---- 阶段 1: 通过两两比较生成候选基频 ----
        for i, f1 in enumerate(peak_freqs):
            for j, f2 in enumerate(peak_freqs[i + 1:], i + 1):
                # 构造保证了 f2 > f1；检查比值是否接近整数
                ratio = f2 / f1
                nearest_int = round(ratio)
                # 仅考虑有意义的谐波关系（阶数 >= 2）
                if 2 <= nearest_int <= max_harmonic_order:
                    # 检查相对于目标整数比值的相对误差
                    if abs(ratio - nearest_int) / nearest_int < tolerance:
                        # 推断基频：若 f1 为基频，则 f2/n 应等于 f1。
                        # 取两者中较小的候选值。
                        candidate = min(f1, f2 / nearest_int)
                        # 以两个峰值幅值之和作为候选权重
                        weight = peak_magnitudes[i] + peak_magnitudes[j]
                        possible_fundamentals.append((candidate, weight, nearest_int))

        if not possible_fundamentals:
            return None, []

        # ---- 阶段 2: 投票选出最可能的基频 ----
        fundamental_votes = {}
        for f, weight, order in possible_fundamentals:
            # 按 0.1 Hz 精度四舍五入后分组
            f_rounded = round(f, 1)
            fundamental_votes[f_rounded] = fundamental_votes.get(f_rounded, 0) + weight

        # 选择总权重最高的基频
        fundamental_freq = max(fundamental_votes, key=fundamental_votes.get)

        # 若未提供完整频率轴，跳过谐波搜索
        if fundamental_freq is None or all_freqs is None:
            return fundamental_freq, []

        # ---- 谐波搜索: 扫描基频的整数倍 ----
        harmonic_indices = []
        current_order = 1
        max_freq = np.max(all_freqs)

        while True:
            target_freq = fundamental_freq * current_order
            # 当目标频率超过最大可用频率时停止
            if target_freq > max_freq:
                break
            if len(all_freqs) > 0:
                # 找到最接近目标谐波的频率单元
                freq_diff = np.abs(all_freqs - target_freq)
                min_idx = np.argmin(freq_diff)
                # 仅在允许容差范围内接受
                relative_diff = freq_diff[min_idx] / target_freq if target_freq > 0 else float('inf')
                if relative_diff < tolerance:
                    harmonic_indices.append(min_idx)
            current_order += 1

        return fundamental_freq, harmonic_indices

    def analyze_spectrum(self, spectrum, **find_peaks_kwargs):
        '''
        对频谱执行完整的峰值分析：检测峰值，然后识别基频及其谐波。

        这是一个便捷方法，将 ``find_peaks`` 和
        ``identify_fundamental_and_harmonics`` 串联起来，
        并将结果打包为单个字典返回。

        Parameters
        ----------
        spectrum : ndarray
            待分析的一维频谱幅值（或 SPL 值）数组。
        **find_peaks_kwargs
            转发给 ``find_peaks`` 的额外关键字参数
            （如 ``height``、``distance``、``prominence``）。

        Returns
        -------
        results : dict
            包含以下键的字典：

            - ``'peak_freqs'`` (ndarray): 检测到的峰值频率。
            - ``'peak_magnitudes'`` (ndarray): 检测到的峰值幅值。
            - ``'peak_indices'`` (ndarray): 检测到的峰值索引。
            - ``'num_peaks'`` (int): 检测到的峰值总数。
            - ``'fundamental_freq'`` (float or None): 识别出的基频。
            - ``'harmonic_indices'`` (list of int): 谐波峰值的索引。
            - ``'harmonic_freqs'`` (ndarray): 谐波峰值的频率。
            - ``'harmonic_magnitudes'`` (ndarray): 谐波峰值的幅值。
            - ``'peak_properties'`` (dict): SciPy 返回的完整峰值属性。
        '''
        # 步骤 1: 使用 SciPy 检测峰值
        peak_indices, peak_properties = self.find_peaks(spectrum, **find_peaks_kwargs)
        peak_freqs = self.freqs[peak_indices]
        # 提取峰值高度；优先使用 'peak_heights' 属性（若可用），
        # 否则回退到峰值索引处的原始频谱值
        peak_magnitudes = (peak_properties.get('peak_heights', spectrum[peak_indices])
                           if 'peak_heights' in peak_properties else spectrum[peak_indices])

        # 步骤 2: 识别基频和谐波
        fundamental_freq, harmonic_indices = self.identify_fundamental_and_harmonics(
            peak_freqs, peak_magnitudes,
            all_freqs=self.freqs, all_magnitudes=spectrum,
            tolerance=0.05, max_harmonic_order=30
        )

        # 步骤 3: 打包结果
        return {
            'peak_freqs': peak_freqs,
            'peak_magnitudes': peak_magnitudes,
            'peak_indices': peak_indices,
            'num_peaks': len(peak_freqs),
            'fundamental_freq': fundamental_freq,
            'harmonic_indices': harmonic_indices,
            'harmonic_freqs': self.freqs[harmonic_indices] if harmonic_indices else [],
            'harmonic_magnitudes': spectrum[harmonic_indices] if harmonic_indices else [],
            'peak_properties': peak_properties
        }
