'''
声学频谱的频带能量分析模块。

提供 `BandContributionAnalyzer` 类，该类将频域频谱
划分为倍频程或分数倍频程频带，并计算各频带的能量贡献。

支持 1/1、1/3 和 1/12 倍频程频带定义，以及一套专为
直升机旋翼噪声分析定制的自定义频带预设。
'''

import numpy as np
from signal_utils import P_REF


class BandContributionAnalyzer:
    '''
    分析频谱能量在各频带中的分布。

    该类创建频带定义（倍频程或自定义），并计算给定频谱中
    各频带的能量、能量占比和 dB 能量值。

    Parameters
    ----------
    freqs : array_like
        与频谱各频率单元对应的一维频率值数组（Hz）。
    reference_pressure : float, optional
        用于 dB 计算的参考声压，单位为帕斯卡。
        默认为 ``P_REF`` (20 uPa)。

    Notes
    -----
    频带能量定义为各频带截止频率范围内频谱幅值平方之和。
    能量 dB 值为：
        ``10 * log10(energy / p_ref^2)``
    对数内部添加了一个小常数 (1e-12)，以防止频带能量
    可忽略不计时出现零值取对数的问题。
    '''

    def __init__(self, freqs, reference_pressure=P_REF):
        '''存储频率轴和参考声压。'''
        self.freqs = np.asarray(freqs)
        self.reference_pressure = reference_pressure

    def create_octave_bands(self, fraction=3, f_low=10, f_high=20000):
        '''
        创建倍频程或分数倍频程频带定义列表。

        频带的上下截止频率根据指定的分数由中心频率计算得到：
          - 1/1 倍频程:  lower = fc / sqrt(2)   , upper = fc * sqrt(2)
          - 1/3 倍频程:  lower = fc / 2^(1/6)   , upper = fc * 2^(1/6)
          - 1/12 倍频程: lower = fc / 2^(1/24)  , upper = fc * 2^(1/24)

        Parameters
        ----------
        fraction : int, optional
            频带分数：1 表示全倍频程，3 表示 1/3 倍频程，
            12 表示 1/12 倍频程。默认为 3。
        f_low : float, optional
            包含的最小中心频率（Hz）。默认为 10。
        f_high : float, optional
            包含的最大中心频率（Hz）。默认为 20000。

        Returns
        -------
        bands : list of dict
            每个字典包含：
            - ``'center'`` (float): 中心频率（Hz）
            - ``'lower'`` (float): 下截止频率（Hz）
            - ``'upper'`` (float): 上截止频率（Hz）
            - ``'bandwidth'`` (float): 频带宽度 = upper - lower（Hz）

        Raises
        ------
        ValueError
            若 ``fraction`` 不是 1、3 或 12。
        '''
        # 标准 1/3 倍频程中心频率 (ISO 266)
        base_freqs = [
            1, 1.25, 1.6, 2, 2.5, 3.15, 4, 5, 6.3, 8, 10, 12.5,
            16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200,
            250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000,
            2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000
        ]
        # 筛选在指定范围内的中心频率
        center_freqs = [f for f in base_freqs if f_low <= f <= f_high]
        bands = []
        for fc in center_freqs:
            # 根据分数计算上下截止频率
            if fraction == 1:
                # 全倍频程：带宽为一个倍频程
                f_lower, f_upper = fc / np.sqrt(2), fc * np.sqrt(2)
            elif fraction == 3:
                # 1/3 倍频程：每个倍频程三个频带
                f_lower, f_upper = fc / (2 ** (1 / 6)), fc * (2 ** (1 / 6))
            elif fraction == 12:
                # 1/12 倍频程：每个倍频程十二个频带
                f_lower, f_upper = fc / (2 ** (1 / 24)), fc * (2 ** (1 / 24))
            else:
                raise ValueError("fraction must be 1, 3, or 12")
            bands.append({'center': fc, 'lower': f_lower, 'upper': f_upper,
                          'bandwidth': f_upper - f_lower})
        return bands

    def calculate_band_energy(self, spectrum, bands, p_ref=None):
        '''
        计算各频带的能量贡献。

        能量定义为频带截止范围内所有频率单元的频谱幅值平方之和。

        Parameters
        ----------
        spectrum : ndarray
            一维频谱幅值数组（幅值，非 dB）。
        bands : list of dict
            `create_octave_bands` 返回的频带定义列表。
        p_ref : float, optional
            参考声压，单位为帕斯卡。若为 None，则使用实例的
            ``reference_pressure``。

        Returns
        -------
        band_energies : list of dict
            每个字典包含：
            - ``'center_freq'`` (float): 频带中心频率（Hz）
            - ``'energy'`` (float): 频带内幅值平方之和
            - ``'energy_dB'`` (float): 相对于 p_ref 的能量级（dB）
            - ``'energy_ratio'`` (float): 占总能量的比例
            - ``'indices'`` (ndarray): 该频带内的频率单元索引
        '''
        if p_ref is None:
            p_ref = self.reference_pressure
        band_energies = []
        # 整个频谱的总能量，用于计算比例
        total_energy = np.sum(spectrum ** 2)
        for band in bands:
            # 找到频带上下截止频率范围内的所有频率单元
            indices = np.where((self.freqs >= band['lower']) & (self.freqs <= band['upper']))[0]
            if len(indices) > 0:
                # 能量 = 该频带内幅值平方之和
                energy = np.sum(spectrum[indices] ** 2)
                # 转换为 dB；添加 1e-12 以防止 log(0)
                energy_dB = 10 * np.log10(energy / (p_ref ** 2) + 1e-12)
                # 占总能量的比例
                energy_ratio = energy / total_energy if total_energy > 0 else 0
                band_energies.append({
                    'center_freq': band['center'],
                    'energy': energy, 'energy_dB': energy_dB,
                    'energy_ratio': energy_ratio, 'indices': indices
                })
        return band_energies

    def analyze_band_contribution(self, spectrum, method='octave', fraction=3,
                                  f_low=10, p_ref=None):
        '''
        对频谱执行完整的频带贡献分析。

        支持两种方法：
          - ``'octave'``: 使用标准倍频程或分数倍频程频带。
          - ``'custom'``: 使用一套专为旋翼噪声分析定制的固定频带
            （中心频率为 50, 125, 250, 500, 1000, 2000, 4000 Hz）。

        Parameters
        ----------
        spectrum : ndarray
            一维频谱幅值数组（幅值，非 dB）。
        method : str, optional
            频带定义方法：``'octave'`` 或 ``'custom'``。
            默认为 ``'octave'``。
        fraction : int, optional
            ``'octave'`` 方法使用的频带分数（1、3 或 12）。
            默认为 3（1/3 倍频程）。
        f_low : float, optional
            ``'octave'`` 方法的最小中心频率（Hz）。
            默认为 10。
        p_ref : float, optional
            参考声压，单位为帕斯卡。若为 None，则使用实例的
            ``reference_pressure``。

        Returns
        -------
        results : dict
            包含以下键的字典：
            - ``'band_type'`` (str): 描述所用频带类型的标签
            - ``'band_energies'`` (list of dict): 各频带的能量详情
            - ``'dominant_band'`` (dict): 能量最大的频带信息
            - ``'total_energy'`` (float): 总频谱能量
            - ``'energy_distribution'`` (dict): 按低频
              (< 250 Hz)、中频 (250–2000 Hz) 和高频 (>= 2000 Hz)
              分组的能量分布
        '''
        if p_ref is None:
            p_ref = self.reference_pressure
        if method == 'octave':
            # 构建倍频程/分数倍频程频带并计算能量
            bands = self.create_octave_bands(fraction=fraction, f_low=f_low)
            band_energies = self.calculate_band_energy(spectrum, bands, p_ref=p_ref)
            band_type = f'1/{fraction} octave'
        else:
            # 旋翼噪声分析的自定义频带集
            custom_bands = [
                {'center': 50, 'lower': 20, 'upper': 100},
                {'center': 125, 'lower': 100, 'upper': 160},
                {'center': 250, 'lower': 160, 'upper': 315},
                {'center': 500, 'lower': 315, 'upper': 630},
                {'center': 1000, 'lower': 630, 'upper': 1250},
                {'center': 2000, 'lower': 1250, 'upper': 2500},
                {'center': 4000, 'lower': 2500, 'upper': 5000},
            ]
            band_energies = self.calculate_band_energy(spectrum, custom_bands, p_ref=p_ref)
            band_type = 'custom'
        # 若无频带捕获到能量，返回空字典
        if not band_energies:
            return {}
        # 提取各频带的能量和比例用于汇总
        energies = [b['energy'] for b in band_energies]
        ratios = [b['energy_ratio'] for b in band_energies]
        # 识别能量最大的频带
        max_idx = np.argmax(energies)
        return {
            'band_type': band_type,
            'band_energies': band_energies,
            'dominant_band': {
                'center_freq': band_energies[max_idx]['center_freq'],
                'energy_ratio': ratios[max_idx],
                'energy_dB': band_energies[max_idx]['energy_dB']
            },
            # 总频谱能量，供参考
            'total_energy': np.sum(spectrum ** 2),
            # 按低/中/高频范围分组的能量
            'energy_distribution': {
                'low_freq': sum(e['energy'] for e in band_energies if e['center_freq'] < 250),
                'mid_freq': sum(e['energy'] for e in band_energies if 250 <= e['center_freq'] < 2000),
                'high_freq': sum(e['energy'] for e in band_energies if e['center_freq'] >= 2000)
            }
        }
