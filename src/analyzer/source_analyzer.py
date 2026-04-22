"""
源项频域贡献量化分析模块
实现谐频与宽频分离、定常与非定常载荷分离、源项贡献分析功能
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.signal import find_peaks
from .analyzer import PeakFrequencyAnalyzer, BandContributionAnalyzer
from .utils import rfft
P_REF = 20e-6  # 和utils中保持一致的参考声压（空气声学标准）


class FrequencySeparator:
    """
    频率分离器：实现谐频噪声与宽频噪声的分离
    """

    def __init__(self, frequencies: np.ndarray):
        """
        初始化频率分离器

        参数:
            frequencies: 频率数组，单位Hz
        """
        self.freqs = frequencies
        self.n_freqs = len(frequencies)

    def separate_by_harmonic_extraction(
        self,
        spectrum: np.ndarray,
        harmonic_freqs: np.ndarray,
        bandwidth_ratio: float = 0.03
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        基于谐频提取法分离谐频和宽频成分

        参数:
            spectrum: 原始频谱幅值数组
            harmonic_freqs: 识别出的谐频频率数组
            bandwidth_ratio: 谐频带宽比，相对于谐频频率，默认3%

        返回:
            harmonic_spectrum: 谐频成分频谱
            broadband_spectrum: 宽频成分频谱
        """
        # 初始化频谱
        harmonic_spectrum = np.zeros_like(spectrum)
        broadband_spectrum = spectrum.copy()

        # 逐个处理每个谐频
        for freq in harmonic_freqs:
            # 计算谐频带宽
            bandwidth = freq * bandwidth_ratio
            lower_bound = freq - bandwidth / 2
            upper_bound = freq + bandwidth / 2

            # 找到频带内的索引
            indices = np.where((self.freqs >= lower_bound) & (self.freqs <= upper_bound))[0]

            if len(indices) > 0:
                # 提取谐频成分
                harmonic_spectrum[indices] = spectrum[indices]
                # 从宽频中扣除谐频成分
                broadband_spectrum[indices] = 0

        # 能量守恒验证
        original_energy = np.sum(spectrum ** 2)
        separated_energy = np.sum(harmonic_spectrum ** 2) + np.sum(broadband_spectrum ** 2)
        energy_error = abs(separated_energy - original_energy) / original_energy if original_energy > 0 else 0

        if energy_error > 0.01:  # 允许1%的误差
            print(f"Warning: Energy conservation error {energy_error:.2%} in frequency separation")

        return harmonic_spectrum, broadband_spectrum


class LoadNoiseSeparator:
    """
    载荷噪声分离器：实现定常载荷噪声与非定常载荷噪声的分离
    """

    def __init__(self, frequencies: np.ndarray):
        """
        初始化载荷噪声分离器

        参数:
            frequencies: 频率数组，单位Hz
        """
        self.freqs = frequencies
        self.n_freqs = len(frequencies)

    def identify_harmonic_bands(
        self,
        fundamental_freq: float,
        max_harmonic_order: int = 30,
        bandwidth_ratio: float = 0.03
    ) -> List[np.ndarray]:
        """
        识别各阶谐频的频率带索引

        参数:
            fundamental_freq: 基频频率
            max_harmonic_order: 最大谐频阶数
            bandwidth_ratio: 谐频带宽比

        返回:
            harmonic_indices_list: 各阶谐频的索引列表
        """
        harmonic_indices_list = []

        for order in range(1, max_harmonic_order + 1):
            harmonic_freq = fundamental_freq * order
            bandwidth = harmonic_freq * bandwidth_ratio
            lower_bound = harmonic_freq - bandwidth / 2
            upper_bound = harmonic_freq + bandwidth / 2

            indices = np.where((self.freqs >= lower_bound) & (self.freqs <= upper_bound))[0]
            if len(indices) > 0:
                harmonic_indices_list.append(indices)

        return harmonic_indices_list

    def calculate_correlation(
        self,
        load_spectrum: np.ndarray,
        thickness_spectrum: np.ndarray,
        harmonic_indices_list: List[np.ndarray]
    ) -> float:
        """
        计算谐频范围内载荷噪声与厚度噪声的频谱相关性

        参数:
            load_spectrum: 载荷噪声频谱
            thickness_spectrum: 厚度噪声频谱
            harmonic_indices_list: 谐频索引列表

        返回:
            correlation: 相关系数
        """
        # 收集所有谐频点的数据
        load_harmonic = []
        thickness_harmonic = []

        for indices in harmonic_indices_list:
            load_harmonic.extend(load_spectrum[indices])
            thickness_harmonic.extend(thickness_spectrum[indices])

        if len(load_harmonic) < 2:
            return 0.0

        # 计算相关系数
        correlation = np.corrcoef(load_harmonic, thickness_harmonic)[0, 1]
        return max(0.0, correlation)  # 保证非负

    def extract_steady_component(
        self,
        load_spectrum: np.ndarray,
        thickness_spectrum: np.ndarray,
        harmonic_indices_list: List[np.ndarray],
        correlation: float
    ) -> np.ndarray:
        """
        提取定常载荷噪声成分

        参数:
            load_spectrum: 原始载荷噪声频谱
            thickness_spectrum: 厚度噪声频谱
            harmonic_indices_list: 谐频索引列表
            correlation: 载荷与厚度噪声的相关系数

        返回:
            steady_load: 定常载荷噪声频谱
        """
        steady_load = np.zeros_like(load_spectrum)

        # 仅在谐频范围内提取定常成分
        for indices in harmonic_indices_list:
            # 定常载荷与厚度噪声成比例，比例系数由相关性决定
            # 这里假设谐频位置的载荷噪声中，与厚度噪声相关的部分为定常载荷
            thickness_amp = thickness_spectrum[indices]
            load_amp = load_spectrum[indices]

            # 计算比例系数，避免除零
            ratio = np.divide(
                load_amp,
                thickness_amp,
                out=np.zeros_like(load_amp),
                where=thickness_amp > 1e-10
            )

            # 使用加权比例，相关性越高，定常成分占比越大
            steady_ratio = np.clip(correlation, 0.2, 0.8)  # 限制在合理范围内
            steady_load[indices] = load_amp * steady_ratio

        return steady_load

    def separate_load_noise(
        self,
        load_spectrum: np.ndarray,
        thickness_spectrum: np.ndarray,
        fundamental_freq: float,
        max_harmonic_order: int = 30,
        bandwidth_ratio: float = 0.03
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        分离定常和非定常载荷噪声

        参数:
            load_spectrum: 载荷噪声频谱幅值
            thickness_spectrum: 厚度噪声频谱幅值（作为参考）
            fundamental_freq: 基频频率（叶片通过频率）
            max_harmonic_order: 最大谐频阶数
            bandwidth_ratio: 谐频带宽比

        返回:
            steady_load: 定常载荷噪声频谱
            unsteady_load: 非定常载荷噪声频谱
        """
        # 1. 识别谐频带
        harmonic_indices_list = self.identify_harmonic_bands(
            fundamental_freq,
            max_harmonic_order,
            bandwidth_ratio
        )

        if not harmonic_indices_list:
            # 没有识别到谐频，所有载荷都视为非定常
            return np.zeros_like(load_spectrum), load_spectrum.copy()

        # 2. 计算相关性
        correlation = self.calculate_correlation(
            load_spectrum,
            thickness_spectrum,
            harmonic_indices_list
        )

        # 3. 提取定常成分
        steady_load = self.extract_steady_component(
            load_spectrum,
            thickness_spectrum,
            harmonic_indices_list,
            correlation
        )

        # 4. 非定常成分 = 总载荷 - 定常成分
        unsteady_load = load_spectrum - steady_load

        # 确保非负
        unsteady_load = np.maximum(unsteady_load, 0)

        # 能量守恒验证
        original_energy = np.sum(load_spectrum ** 2)
        separated_energy = np.sum(steady_load ** 2) + np.sum(unsteady_load ** 2)
        energy_error = abs(separated_energy - original_energy) / original_energy if original_energy > 0 else 0

        if energy_error > 0.01:  # 允许1%的误差
            print(f"Warning: Energy conservation error {energy_error:.2%} in load separation")

        return steady_load, unsteady_load


class EnhancedBandContributionAnalyzer(BandContributionAnalyzer):
    """
    增强的频带贡献分析器，支持多噪声源的贡献分析
    """

    def __init__(self, frequencies: np.ndarray, reference_pressure: float = P_REF):
        """
        初始化增强的频带贡献分析器

        参数:
            frequencies: 频率数组
            reference_pressure: 参考声压，默认和utils保持一致（2e-5 Pa，空气声学标准）
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
        """
        分析各噪声源在频带内的能量贡献

        参数:
            spectra_dict: 噪声源频谱字典，键为源名称，值为幅值数组
            band_type: 频带类型，'octave'或'custom'
            fraction: 倍频程分数，1=1倍频程，3=1/3倍频程，12=1/12倍频程
            f_low: 最低分析频率
            f_high: 最高分析频率
            custom_bands: 自定义频带列表，每个元素包含'lower', 'center', 'upper'

        返回:
            band_results: 各频带的分析结果列表
        """
        # 创建频带
        if band_type == 'custom' and custom_bands is not None:
            bands = custom_bands
        else:
            bands = self.create_octave_bands(fraction=fraction, f_low=f_low, f_high=f_high)

        results = []

        for band in bands:
            # 找到频带内的频率索引
            indices = np.where(
                (self.freqs >= band['lower']) &
                (self.freqs <= band['upper'])
            )[0]

            if len(indices) == 0:
                continue

            band_result = {
                'center_freq': band['center'],
                'lower_bound': band['lower'],
                'upper_bound': band['upper'],
                'n_points': len(indices)
            }

            # 计算各噪声源在频带内的能量
            source_energies = {}
            total_energy = 0.0

            for source_name, spectrum in spectra_dict.items():
                if len(spectrum) != len(self.freqs):
                    raise ValueError(f"Spectrum length for {source_name} does not match frequency array")

                band_spectrum = spectrum[indices]
                energy = np.sum(band_spectrum ** 2)
                source_energies[source_name] = energy
                total_energy += energy

            # 计算各源的能量占比和SPL
            for source_name, energy in source_energies.items():
                ratio = energy / total_energy if total_energy > 0 else 0.0
                spl = 10 * np.log10(energy / (self.reference_pressure ** 2) + 1e-12) if energy > 0 else -np.inf

                band_result[f'{source_name}_energy'] = energy
                band_result[f'{source_name}_ratio'] = ratio
                band_result[f'{source_name}_spl'] = spl

            # 计算谐频占比（如果提供了谐频和宽频数据）
            if 'harmonic' in source_energies and 'broadband' in source_energies:
                harmonic_energy = source_energies['harmonic']
                band_total = harmonic_energy + source_energies['broadband']
                harmonic_ratio = harmonic_energy / band_total if band_total > 0 else 0.0
                band_result['harmonic_ratio_in_band'] = harmonic_ratio

            results.append(band_result)

        return results

    def calculate_global_statistics(
        self,
        spectra_dict: Dict[str, np.ndarray]
    ) -> Dict:
        """
        计算全频段的全局统计信息

        参数:
            spectra_dict: 噪声源频谱字典

        返回:
            global_stats: 全局统计信息字典
        """
        global_stats = {}

        # 总能量
        total_energy = 0.0
        source_energies = {}

        for source_name, spectrum in spectra_dict.items():
            energy = np.sum(spectrum ** 2)
            source_energies[source_name] = energy
            total_energy += energy

        # 各源总占比
        for source_name, energy in source_energies.items():
            ratio = energy / total_energy if total_energy > 0 else 0.0
            spl = 10 * np.log10(energy / (self.reference_pressure ** 2) + 1e-12) if energy > 0 else -np.inf

            global_stats[f'{source_name}_total_energy'] = energy
            global_stats[f'{source_name}_total_ratio'] = ratio
            global_stats[f'{source_name}_total_spl'] = spl

        # 分频段统计（低、中、高）
        freq_bands = [
            ('low', 0, 250),
            ('mid', 250, 2000),
            ('high', 2000, np.inf)
        ]

        for band_name, f_low, f_high in freq_bands:
            indices = np.where((self.freqs >= f_low) & (self.freqs < f_high))[0]

            if len(indices) == 0:
                continue

            band_total = 0.0
            band_source_energies = {}

            for source_name, spectrum in spectra_dict.items():
                band_energy = np.sum(spectrum[indices] ** 2)
                band_source_energies[source_name] = band_energy
                band_total += band_energy

            for source_name, energy in band_source_energies.items():
                ratio = energy / band_total if band_total > 0 else 0.0
                global_stats[f'{source_name}_{band_name}_energy'] = energy
                global_stats[f'{source_name}_{band_name}_ratio'] = ratio

        # 谐频相对于（谐频+宽频）的能量占比
        if 'harmonic' in source_energies and 'broadband' in source_energies:
            harmonic_total = source_energies['harmonic']
            broadband_total = source_energies['broadband']
            global_stats['harmonic_to_broadband_ratio'] = harmonic_total / (harmonic_total + broadband_total) if (harmonic_total + broadband_total) > 0 else 0.0

        return global_stats


class SourceContributionAnalyzer:
    """
    源项贡献分析主类：协调各模块完成完整的源项频域贡献量化分析
    """

    def __init__(
        self,
        frequencies: np.ndarray,
        reference_pressure: float = P_REF
    ):
        """
        初始化源项贡献分析器

        参数:
            frequencies: 频率数组
            reference_pressure: 参考声压，默认和utils保持一致（2e-5 Pa，空气声学标准）
        """
        self.freqs = frequencies
        self.ref_pressure = reference_pressure

        # 初始化子分析器
        self.freq_separator = FrequencySeparator(frequencies)
        self.load_separator = LoadNoiseSeparator(frequencies)
        self.band_analyzer = EnhancedBandContributionAnalyzer(frequencies, reference_pressure)
        self.peak_analyzer = PeakFrequencyAnalyzer(frequencies)

    def analyze(
        self,
        thickness_spectrum: np.ndarray,
        load_spectrum: np.ndarray,
        total_spectrum: Optional[np.ndarray] = None,
        fundamental_freq: Optional[float] = None,
        harmonic_bandwidth_ratio: float = 0.03,
        max_harmonic_order: int = 30,
        band_type: str = 'octave',
        band_fraction: int = 3,
        band_f_low: float = 10,
        band_f_high: float = 20000
    ) -> Dict:
        """
        执行完整的源项贡献分析

        参数:
            thickness_spectrum: 厚度噪声频谱幅值
            load_spectrum: 载荷噪声频谱幅值
            total_spectrum: 总噪声频谱幅值，如不提供则由厚度+载荷合成
            fundamental_freq: 基频频率，如不提供则自动识别
            harmonic_bandwidth_ratio: 谐频带宽比
            max_harmonic_order: 最大谐频阶数
            band_type: 频带分析类型，'octave'或'custom'
            band_fraction: 倍频程分数
            band_f_low: 最低分析频率
            band_f_high: 最高分析频率

        返回:
            results: 包含所有分析结果的字典
        """
        # 合成总频谱（如果未提供）
        if total_spectrum is None:
            total_spectrum = thickness_spectrum + load_spectrum

        # 1. 谐频识别
        if fundamental_freq is None:
            # 自动识别基频
            peak_result = self.peak_analyzer.analyze_spectrum(total_spectrum)
            fundamental_freq = peak_result['fundamental_freq']
            harmonic_freqs = peak_result['harmonic_freqs']
        else:
            # 使用提供的基频生成谐频序列
            harmonic_freqs = fundamental_freq * np.arange(1, max_harmonic_order + 1)
            # 过滤掉超过频率范围的谐频
            harmonic_freqs = harmonic_freqs[harmonic_freqs <= self.freqs.max()]

        # 2. 谐频与宽频分离（基于总频谱）
        harmonic_spectrum, broadband_spectrum = self.freq_separator.separate_by_harmonic_extraction(
            total_spectrum,
            harmonic_freqs,
            harmonic_bandwidth_ratio
        )

        # 3. 定常与非定常载荷分离
        steady_load, unsteady_load = self.load_separator.separate_load_noise(
            load_spectrum,
            thickness_spectrum,
            fundamental_freq,
            max_harmonic_order,
            harmonic_bandwidth_ratio
        )

        # 4. 收集所有频谱
        spectra_dict = {
            'thickness': thickness_spectrum,
            'steady_load': steady_load,
            'unsteady_load': unsteady_load,
            'total': total_spectrum,
            'harmonic': harmonic_spectrum,
            'broadband': broadband_spectrum
        }

        # 5. 频带贡献分析
        band_results = self.band_analyzer.analyze_source_contribution(
            spectra_dict,
            band_type=band_type,
            fraction=band_fraction,
            f_low=band_f_low,
            f_high=band_f_high
        )

        # 6. 全局统计
        global_stats = self.band_analyzer.calculate_global_statistics(spectra_dict)

        # 7. 谐频点详细分析
        harmonic_results = self._analyze_harmonic_points(
            spectra_dict,
            harmonic_freqs,
            harmonic_bandwidth_ratio
        )

        # 8. 详细频率点数据
        detail_data = self._create_detail_data(spectra_dict)

        # 组装结果
        results = {
            'frequencies': self.freqs,
            'spectra': spectra_dict,
            'fundamental_freq': fundamental_freq,
            'harmonic_freqs': harmonic_freqs,
            'band_results': band_results,
            'global_stats': global_stats,
            'harmonic_results': harmonic_results,
            'detail_data': detail_data
        }

        return results

    def _analyze_harmonic_points(
        self,
        spectra_dict: Dict[str, np.ndarray],
        harmonic_freqs: np.ndarray,
        bandwidth_ratio: float = 0.03
    ) -> List[Dict]:
        """
        分析各阶谐频点的噪声源贡献

        参数:
            spectra_dict: 噪声源频谱字典
            harmonic_freqs: 谐频频率数组
            bandwidth_ratio: 谐频带宽比

        返回:
            harmonic_results: 各谐频的分析结果
        """
        results = []

        for order, freq in enumerate(harmonic_freqs, 1):
            bandwidth = freq * bandwidth_ratio
            lower_bound = freq - bandwidth / 2
            upper_bound = freq + bandwidth / 2

            indices = np.where((self.freqs >= lower_bound) & (self.freqs <= upper_bound))[0]

            if len(indices) == 0:
                continue

            # 找到幅值最大的点作为代表
            max_idx = indices[np.argmax(spectra_dict['total'][indices])]
            actual_freq = self.freqs[max_idx]

            harmonic_result = {
                'harmonic_order': order,
                'nominal_freq': freq,
                'actual_freq': actual_freq
            }

            # 各源在该谐频点的幅值和SPL（和utils.rfft计算方式保持一致）
            for source_name, spectrum in spectra_dict.items():
                amp = spectrum[max_idx]
                spl = 20 * np.log10(amp / self.ref_pressure + 1e-12) if amp > 0 else -np.inf
                harmonic_result[f'{source_name}_amp'] = amp
                harmonic_result[f'{source_name}_spl'] = spl

            # 计算该谐频点的源贡献占比
            total_amp = (spectra_dict['thickness'][max_idx] ** 2 +
                        spectra_dict['steady_load'][max_idx] ** 2 +
                        spectra_dict['unsteady_load'][max_idx] ** 2) ** 0.5

            if total_amp > 0:
                harmonic_result['thickness_ratio'] = (spectra_dict['thickness'][max_idx] ** 2) / (total_amp ** 2)
                harmonic_result['steady_load_ratio'] = (spectra_dict['steady_load'][max_idx] ** 2) / (total_amp ** 2)
                harmonic_result['unsteady_load_ratio'] = (spectra_dict['unsteady_load'][max_idx] ** 2) / (total_amp ** 2)
            else:
                harmonic_result['thickness_ratio'] = 0.0
                harmonic_result['steady_load_ratio'] = 0.0
                harmonic_result['unsteady_load_ratio'] = 0.0

            results.append(harmonic_result)

        return results

    def _create_detail_data(self, spectra_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        创建详细频率点数据表

        参数:
            spectra_dict: 噪声源频谱字典

        返回:
            detail_df: 包含所有频率点详细数据的DataFrame
        """
        data = {
            'Frequency(Hz)': self.freqs
        }

        for source_name, spectrum in spectra_dict.items():
            data[f'{source_name}_amp(Pa)'] = spectrum
            # 和utils.rfft中保持一致的SPL计算方式
            data[f'{source_name}_SPL(dB)'] = 20 * np.log10(spectrum / self.ref_pressure + 1e-12)
            # 能量（Pa²·Hz）
            data[f'{source_name}_energy'] = spectrum ** 2

        df = pd.DataFrame(data)
        return df
