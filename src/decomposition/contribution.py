"""
源项贡献分析模块 (Source Contribution Analysis).

本模块提供完整的噪声源贡献分析流水线，将旋翼气动噪声分解为
不同物理源项（厚度噪声、定常载荷、非定常载荷）在不同频段的贡献。

分析流水线概述
--------------
SourceContributionAnalyzer.analyze() 执行以下步骤：
1. 谐频识别: 从总频谱中自动检测或使用给定的 BPF 谐频
2. 谐频/宽频分离: 将总噪声分为谐频（离散频率）和宽频（连续谱）
3. 相位约束分离: 将载荷噪声分解为定常和非定常分量（需要厚度噪声相位参考）
4. 频带贡献分析: 按倍频程或其他自定义频带统计各源项的贡献
5. 全局统计: 计算全频段和各子频段的能量占比
6. 谐频点分析: 在每个 BPF 谐频处详细分析各源项的幅值、SPL 和比例
7. 详细数据导出: 生成包含完整频谱信息的 DataFrame
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from spectral.peaks import PeakFrequencyAnalyzer
from spectral.bands import BandContributionAnalyzer
from signal_utils import P_REF
from .frequency import FrequencySeparator
from .phase_constraint import PhaseConstraintSeparator


class SourceBandAnalyzer(BandContributionAnalyzer):
    """噪声源频带贡献分析器 -- 多噪声源的频带能量分布分析.

    继承自 BandContributionAnalyzer，增加了对多个噪声源（如厚度、
    定常载荷、非定常载荷、谐频、宽频等）在相同频带上同时分析的能力。

    Parameters
    ----------
    frequencies : np.ndarray
        频谱对应的频率轴 (Hz)，一维数组。
    reference_pressure : float, optional
        参考声压 (Pa)，用于 SPL 计算，默认 P_REF (20e-6 Pa)。
    """

    def __init__(self, frequencies: np.ndarray, reference_pressure: float = P_REF):
        """初始化频带分析器.

        Parameters
        ----------
        frequencies : np.ndarray
            频率轴 (Hz)。
        reference_pressure : float, optional
            参考声压 (Pa)，默认 P_REF。
        """
        super().__init__(frequencies, reference_pressure)

    def analyze_source_contribution(
        self,
        spectra_dict: Dict[str, np.ndarray],
        band_type: str = 'octave',
        fraction: int = 3,
        f_low: float = 10,
        f_high: float = 20000,
        custom_bands: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """分析多个噪声源在各频带中的贡献.

        对每个频带，计算各噪声源的能量、能量占比和 SPL。
        如果频谱中包含 'harmonic' 和 'broadband' 两个源，
        还会额外计算谐频在该频带内的占比。

        Parameters
        ----------
        spectra_dict : Dict[str, np.ndarray]
            噪声源名称到幅值频谱的映射，如
            {'thickness': ..., 'steady_load': ..., ...}。
        band_type : str, optional
            频带类型: 'octave'（倍频程）或 'custom'（自定义）。默认 'octave'。
        fraction : int, optional
            倍频程分数，如 3 表示 1/3 倍频程。默认 3。
        f_low : float, optional
            最低频率 (Hz)。默认 10。
        f_high : float, optional
            最高频率 (Hz)。默认 20000。
        custom_bands : List[Dict], optional
            自定义频带列表，每个元素需包含 'lower', 'upper', 'center' 键。
            仅在 band_type='custom' 时使用。

        Returns
        -------
        results : List[Dict]
            频带分析结果列表，每个元素对应一个频带，包含：
            - 频带信息: center_freq, lower_bound, upper_bound, n_points
            - 各源项: {source_name}_energy, {source_name}_ratio, {source_name}_spl
            - 谐频占比: harmonic_ratio_in_band（仅在存在 harmonic/broadband 时）
        """
        # 确定频带定义
        if band_type == 'custom' and custom_bands is not None:
            bands = custom_bands
        else:
            bands = self.create_octave_bands(
                fraction=fraction, f_low=f_low, f_high=f_high
            )

        results = []
        for band in bands:
            # 找出落在当前频带内的所有频率索引
            indices = np.where(
                (self.freqs >= band['lower']) & (self.freqs <= band['upper'])
            )[0]

            if len(indices) == 0:
                continue  # 跳过没有数据的频带

            band_result = {
                'center_freq': band['center'],
                'lower_bound': band['lower'],
                'upper_bound': band['upper'],
                'n_points': len(indices),
            }

            # 计算各噪声源在当前频带内的能量（sum of amplitude^2）
            source_energies = {}
            total_energy = 0.0

            for source_name, spectrum in spectra_dict.items():
                energy = np.sum(spectrum[indices] ** 2)
                source_energies[source_name] = energy
                total_energy += energy

            # 为每个噪声源计算占比和 SPL
            for source_name, energy in source_energies.items():
                ratio = energy / total_energy if total_energy > 0 else 0.0
                spl = (
                    10 * np.log10(energy / (self.reference_pressure ** 2) + 1e-12)
                    if energy > 0 else -np.inf
                )
                band_result[f'{source_name}_energy'] = energy
                band_result[f'{source_name}_ratio'] = ratio
                band_result[f'{source_name}_spl'] = spl

            # 如果同时存在 harmonic 和 broadband，计算谐频在此频带内的占比
            if 'harmonic' in source_energies and 'broadband' in source_energies:
                h_energy = source_energies['harmonic']
                b_energy = source_energies['broadband']
                band_result['harmonic_ratio_in_band'] = (
                    h_energy / (h_energy + b_energy)
                    if (h_energy + b_energy) > 0 else 0.0
                )

            results.append(band_result)

        return results

    def calculate_global_statistics(
        self, spectra_dict: Dict[str, np.ndarray]
    ) -> Dict:
        """计算多噪声源的全频段和分频段统计.

        统计内容包括：
        - 各源项在全频段的总能量、总能量占比、总 SPL
        - 各源项在 low (0-250 Hz)、mid (250-2000 Hz)、high (>2000 Hz)
          三个频段的分频段能量和占比
        - harmonic/broadband 全局比值（若存在这两个源）

        Parameters
        ----------
        spectra_dict : Dict[str, np.ndarray]
            噪声源名称到幅值频谱的映射。

        Returns
        -------
        global_stats : Dict
            全局统计字典，键名格式：
            - '{source}_total_energy', '{source}_total_ratio', '{source}_total_spl'
            - '{source}_{band}_energy', '{source}_{band}_ratio'
            - 'harmonic_to_broadband_ratio'
        """
        global_stats = {}
        total_energy = 0.0
        source_energies = {}

        # ---- 全频段统计 ----
        for source_name, spectrum in spectra_dict.items():
            energy = np.sum(spectrum ** 2)
            source_energies[source_name] = energy
            total_energy += energy

        for source_name, energy in source_energies.items():
            ratio = energy / total_energy if total_energy > 0 else 0.0
            spl = (
                10 * np.log10(energy / (self.reference_pressure ** 2) + 1e-12)
                if energy > 0 else -np.inf
            )
            global_stats[f'{source_name}_total_energy'] = energy
            global_stats[f'{source_name}_total_ratio'] = ratio
            global_stats[f'{source_name}_total_spl'] = spl

        # ---- 分频段统计（low / mid / high） ----
        freq_bands = [
            ('low', 0, 250),
            ('mid', 250, 2000),
            ('high', 2000, np.inf),
        ]

        for band_name, fl, fh in freq_bands:
            # 找到当前频段内的频率索引
            indices = np.where((self.freqs >= fl) & (self.freqs < fh))[0]

            if len(indices) == 0:
                continue

            # 第一遍：计算频段总能量和各源项能量
            band_total = 0.0
            for source_name, spectrum in spectra_dict.items():
                band_energy = np.sum(spectrum[indices] ** 2)
                band_total += band_energy
                global_stats[f'{source_name}_{band_name}_energy'] = band_energy

            # 第二遍：计算各源项在该频段的占比
            for source_name, spectrum in spectra_dict.items():
                band_energy = global_stats[f'{source_name}_{band_name}_energy']
                ratio = band_energy / band_total if band_total > 0 else 0.0
                global_stats[f'{source_name}_{band_name}_ratio'] = ratio

        # ---- 谐频/宽频全局比值 ----
        if 'harmonic' in source_energies and 'broadband' in source_energies:
            h_total = source_energies['harmonic']
            b_total = source_energies['broadband']
            global_stats['harmonic_to_broadband_ratio'] = (
                h_total / (h_total + b_total)
                if (h_total + b_total) > 0 else 0.0
            )

        return global_stats


class SourceContributionAnalyzer:
    """源项贡献分析器 -- 完整的噪声源分解与分析流水线.

    集成以下分析能力：
    1. 谐频检测（PeakFrequencyAnalyzer）
    2. 谐频/宽频分离（FrequencySeparator）
    3. 定常/非定常载荷分离（PhaseConstraintSeparator）
    4. 频带贡献分析（SourceBandAnalyzer）
    5. 全局统计与谐频点详细分析

    Parameters
    ----------
    frequencies : np.ndarray
        频谱对应的频率轴 (Hz)，一维数组。
    reference_pressure : float, optional
        参考声压 (Pa)，用于 SPL 计算，默认 P_REF (20e-6 Pa)。

    Attributes
    ----------
    freqs : np.ndarray
        频率轴。
    ref_pressure : float
        参考声压。
    freq_separator : FrequencySeparator
        谐频/宽频分离器实例。
    phase_separator : PhaseConstraintSeparator
        相位约束分离器实例。
    band_analyzer : SourceBandAnalyzer
        频带贡献分析器实例。
    peak_analyzer : PeakFrequencyAnalyzer
        峰值频率分析器实例（用于自动检测基频）。

    Notes
    -----
    - analyze() 方法接受复频谱输入（厚度和载荷的 FFT 结果），
      因为相位约束法需要相位信息。
    - 总频谱 = 厚度 + 载荷（线性叠加），用于谐频检测和谐频/宽频分离。
    - 分析结果以字典形式返回，可直接用于绘图和报告生成。
    """

    def __init__(
        self,
        frequencies: np.ndarray,
        reference_pressure: float = P_REF
    ):
        """初始化源项贡献分析器.

        创建所有子分析器实例：频率分离器、相位约束分离器、
        频带分析器和峰值检测器。

        Parameters
        ----------
        frequencies : np.ndarray
            频率轴 (Hz)。
        reference_pressure : float, optional
            参考声压 (Pa)，默认 P_REF。
        """
        self.freqs = frequencies
        self.ref_pressure = reference_pressure

        # 子分析器实例化
        self.freq_separator = FrequencySeparator(frequencies)
        self.phase_separator = PhaseConstraintSeparator(frequencies)
        self.band_analyzer = SourceBandAnalyzer(frequencies, reference_pressure)
        self.peak_analyzer = PeakFrequencyAnalyzer(frequencies)

    def analyze(
        self,
        thickness_complex: np.ndarray,
        load_complex: np.ndarray,
        total_complex: Optional[np.ndarray] = None,
        fundamental_freq: Optional[float] = None,
        harmonic_bandwidth_ratio: float = 0.03,
        max_harmonic_order: int = 30,
        band_type: str = 'octave',
        band_fraction: int = 3,
        band_f_low: float = 10,
        band_f_high: float = 20000,
        check_phase_consistency: bool = False
    ) -> Dict:
        """执行完整的噪声源分解与分析流水线.

        流水线步骤
        -----------
        1. 谐频识别: 若未提供 fundamental_freq，则通过峰值检测自动提取。
        2. 谐频/宽频分离: 将总频谱分解为谐频分量和宽频分量。
        3. 相位约束分离: 将载荷复频谱分解为定常和非定常分量。
        4. 收集频谱: 汇总所有分解后的幅值频谱到一个字典。
        5. 频带贡献分析: 计算各源项在各频段的能量分布。
        6. 全局统计: 计算各源项的全频段和分频段统计量。
        7. 谐频点分析: 在每个 BPF 谐频处详细分析各源项的贡献。
        8. 详细数据导出: 生成包含完整频谱信息的 DataFrame。

        Parameters
        ----------
        thickness_complex : np.ndarray
            厚度噪声的复频谱（FFT 输出），一维复数数组。
        load_complex : np.ndarray
            载荷噪声的复频谱（FFT 输出），一维复数数组。
        total_complex : np.ndarray, optional
            总噪声的复频谱。若为 None，则计算 thickness + load。
        fundamental_freq : float, optional
            基频 (BPF, Hz)。若为 None，则通过峰值检测自动提取。
        harmonic_bandwidth_ratio : float, optional
            谐频带宽比，默认 0.03。
        max_harmonic_order : int, optional
            最高谐频阶次，默认 30。
        band_type : str, optional
            频带分析类型 ('octave' 或 'custom')，默认 'octave'。
        band_fraction : int, optional
            倍频程分数，默认 3（1/3 倍频程）。
        band_f_low : float, optional
            频带分析最低频率 (Hz)，默认 10。
        band_f_high : float, optional
            频带分析最高频率 (Hz)，默认 20000。
        check_phase_consistency : bool, optional
            是否启用相位一致性检查，默认 False。

        Returns
        -------
        result : Dict
            综合分析结果字典，包含以下键：
            - 'frequencies': 频率轴 (Hz)
            - 'spectra': 各源项的幅值频谱字典
            - 'fundamental_freq': 基频 (Hz)
            - 'harmonic_freqs': 谐频频率列表 (Hz)
            - 'band_results': 频带分析结果列表
            - 'global_stats': 全局统计字典
            - 'harmonic_results': 各阶谐频的详细分析
            - 'detail_data': pandas DataFrame，完整频谱数据
            - 'phase_stats': 相位统计（仅当 check_phase_consistency=True）
        """
        # 取厚度和载荷的幅值频谱（后续分析和能量计算使用）
        thickness_spectrum = np.abs(thickness_complex)
        load_spectrum = np.abs(load_complex)

        # 若未提供总频谱，则按线性叠加计算（厚度 + 载荷 = 总噪声）
        if total_complex is None:
            total_complex = thickness_complex + load_complex
        total_spectrum = np.abs(total_complex)

        # ============================================================
        # Step 1: 谐频识别
        # ============================================================
        if fundamental_freq is None:
            # 自动检测：通过峰值分析从总频谱中提取基频和谐频
            peak_result = self.peak_analyzer.analyze_spectrum(total_spectrum)
            fundamental_freq = peak_result['fundamental_freq']
            harmonic_freqs = peak_result['harmonic_freqs']
        else:
            # 手动指定：生成 k*BPF 的谐频数组（k=1,2,...,max_order）
            harmonic_freqs = (
                fundamental_freq * np.arange(1, max_harmonic_order + 1)
            )
            # 截断超出频率范围的谐频
            harmonic_freqs = harmonic_freqs[
                harmonic_freqs <= self.freqs.max()
            ]

        # ============================================================
        # Step 2: 谐频/宽频分离
        # ============================================================
        harmonic_spectrum, broadband_spectrum = (
            self.freq_separator.separate_by_harmonic_extraction(
                total_spectrum, harmonic_freqs, harmonic_bandwidth_ratio
            )
        )

        # ============================================================
        # Step 3: 相位约束法分离定常/非定常载荷
        # ============================================================
        steady_load, unsteady_load, phase_stats = (
            self.phase_separator.separate(
                load_complex, thickness_complex, fundamental_freq,
                max_harmonic_order, harmonic_bandwidth_ratio,
                check_phase_consistency
            )
        )

        # ============================================================
        # Step 4: 收集所有分解后的频谱到一个字典
        #         键名为源项名称，值为对应的幅值频谱
        # ============================================================
        spectra_dict = {
            'thickness': thickness_spectrum,   # 厚度噪声
            'steady_load': steady_load,        # 定常载荷噪声
            'unsteady_load': unsteady_load,    # 非定常载荷噪声
            'total': total_spectrum,           # 总噪声
            'harmonic': harmonic_spectrum,     # 谐频分量
            'broadband': broadband_spectrum,   # 宽频分量
        }

        # ============================================================
        # Step 5: 频带贡献分析
        #         按倍频程（默认 1/3 oct）统计各源项的频带分布
        # ============================================================
        band_results = self.band_analyzer.analyze_source_contribution(
            spectra_dict, band_type, band_fraction,
            band_f_low, band_f_high
        )

        # ============================================================
        # Step 6: 全局统计
        #         全频段和分频段 (low/mid/high) 的能量占比及 SPL
        # ============================================================
        global_stats = self.band_analyzer.calculate_global_statistics(
            spectra_dict
        )

        # ============================================================
        # Step 7: 谐频点详细分析
        #         在每个 BPF 谐频处，分析各源项的幅值、SPL 和比例
        # ============================================================
        harmonic_results = self._analyze_harmonic_points(
            spectra_dict, harmonic_freqs, harmonic_bandwidth_ratio
        )

        # ============================================================
        # Step 8: 详细数据导出
        #         生成包含所有频率点详细信息的 DataFrame
        # ============================================================
        detail_data = self._create_detail_data(spectra_dict)

        # ============================================================
        # 汇总返回
        # ============================================================
        return {
            'frequencies': self.freqs,
            'spectra': spectra_dict,
            'fundamental_freq': fundamental_freq,
            'harmonic_freqs': harmonic_freqs,
            'band_results': band_results,
            'global_stats': global_stats,
            'harmonic_results': harmonic_results,
            'detail_data': detail_data,
            'phase_stats': phase_stats if check_phase_consistency else None,
        }

    def _analyze_harmonic_points(
        self,
        spectra_dict: Dict[str, np.ndarray],
        harmonic_freqs: np.ndarray,
        bandwidth_ratio: float = 0.03
    ) -> List[Dict]:
        """分析每个谐频点处各噪声源的贡献.

        对每个谐频，在带宽内找到总噪声最大的频率点（实际峰值位置），
        然后计算各源项在该点的幅值、SPL 和能量占比。

        Parameters
        ----------
        spectra_dict : Dict[str, np.ndarray]
            源项名称到幅值频谱的映射。
        harmonic_freqs : np.ndarray
            谐频频率列表 (Hz)。
        bandwidth_ratio : float, optional
            用于确定峰值搜索窗口的带宽比，默认 0.03。

        Returns
        -------
        results : List[Dict]
            谐频点分析结果列表，每个元素包含：
            - harmonic_order: 谐频阶次 (1, 2, ...)
            - nominal_freq: 名义谐频 (Hz)
            - actual_freq: 实际峰值频率 (Hz)
            - {source}_amp: 各源项幅值 (Pa)
            - {source}_spl: 各源项 SPL (dB)
            - thickness_ratio, steady_load_ratio, unsteady_load_ratio
        """
        results = []

        for order, freq in enumerate(harmonic_freqs, 1):
            # 确定谐频周围的搜索窗口
            bw = freq * bandwidth_ratio
            indices = np.where(
                (self.freqs >= freq - bw / 2) & (self.freqs <= freq + bw / 2)
            )[0]

            if len(indices) == 0:
                continue

            # 在窗口内找到总噪声最大的频率 bin（实际峰值位置）
            max_idx = indices[
                np.argmax(spectra_dict['total'][indices])
            ]
            actual_freq = self.freqs[max_idx]

            hr = {
                'harmonic_order': order,
                'nominal_freq': freq,
                'actual_freq': actual_freq,
            }

            # 提取各源项在该频率点的幅值和 SPL
            for source_name, spectrum in spectra_dict.items():
                amp = spectrum[max_idx]
                spl = (
                    20 * np.log10(amp / self.ref_pressure + 1e-12)
                    if amp > 0 else -np.inf
                )
                hr[f'{source_name}_amp'] = amp
                hr[f'{source_name}_spl'] = spl

            # 计算厚度/定常载荷/非定常载荷三者的能量占比
            # 能量 = 幅值^2（对于单一频率点）
            t_a = spectra_dict['thickness'][max_idx]
            s_a = spectra_dict['steady_load'][max_idx]
            u_a = spectra_dict['unsteady_load'][max_idx]
            total_amp_sq = t_a ** 2 + s_a ** 2 + u_a ** 2

            if total_amp_sq > 0:
                hr['thickness_ratio'] = (t_a ** 2) / total_amp_sq
                hr['steady_load_ratio'] = (s_a ** 2) / total_amp_sq
                hr['unsteady_load_ratio'] = (u_a ** 2) / total_amp_sq
            else:
                hr['thickness_ratio'] = 0.0
                hr['steady_load_ratio'] = 0.0
                hr['unsteady_load_ratio'] = 0.0

            results.append(hr)

        return results

    def _create_detail_data(
        self, spectra_dict: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """创建包含完整频谱详细信息的 DataFrame.

        为每个频率点生成所有源项的幅值、SPL 和能量数据列。

        Parameters
        ----------
        spectra_dict : Dict[str, np.ndarray]
            源项名称到幅值频谱的映射。

        Returns
        -------
        df : pd.DataFrame
            包含以下列的 DataFrame:
            - 'Frequency(Hz)': 频率
            - '{source}_amp(Pa)': 各源项幅值
            - '{source}_SPL(dB)': 各源项声压级
            - '{source}_energy': 各源项能量 (Pa²)
        """
        data = {'Frequency(Hz)': self.freqs}

        for source_name, spectrum in spectra_dict.items():
            # 幅值 (Pa)
            data[f'{source_name}_amp(Pa)'] = spectrum
            # SPL (dB re {ref_pressure} Pa)
            data[f'{source_name}_SPL(dB)'] = (
                20 * np.log10(spectrum / self.ref_pressure + 1e-12)
            )
            # 能量 (Pa²) = 幅值的平方
            data[f'{source_name}_energy'] = spectrum ** 2

        return pd.DataFrame(data)
