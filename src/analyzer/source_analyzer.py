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

    def separate_load_noise(
        self,
        load_complex: np.ndarray,
        thickness_complex: np.ndarray,
        fundamental_freq: float,
        max_harmonic_order: int = 30,
        bandwidth_ratio: float = 0.03,
        check_phase_consistency: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        基于相位约束法分离定常和非定常载荷噪声
        以厚度噪声相位为基准，将载荷噪声投影到基准相位轴，同相分量为定常载荷，正交分量为非定常载荷

        参数:
            load_complex: 载荷噪声复数频谱
            thickness_complex: 厚度噪声复数频谱（作为相位基准）
            fundamental_freq: 基频频率（叶片通过频率）
            max_harmonic_order: 最大谐频阶数
            bandwidth_ratio: 谐频带宽比
            check_phase_consistency: 是否检查相位一致性，返回相位统计信息

        返回:
            steady_load_amp: 定常载荷噪声幅值频谱
            unsteady_load_amp: 非定常载荷噪声幅值频谱
            phase_stats: 相位统计信息（仅当check_phase_consistency=True时返回有效数据）
        """
        # 1. 识别谐频带
        harmonic_indices_list = self.identify_harmonic_bands(
            fundamental_freq,
            max_harmonic_order,
            bandwidth_ratio
        )

        # 初始化定常和非定常复数频谱
        steady_load_complex = np.zeros_like(load_complex)
        unsteady_load_complex = np.zeros_like(load_complex)

        phase_stats = {
            'phase_diffs': [],
            'phase_diff_variances': []
        }

        if harmonic_indices_list:
            # 2. 逐个处理各阶谐频带
            for indices in harmonic_indices_list:
                # 提取谐频带内的复数频谱
                thick_complex_band = thickness_complex[indices]
                load_complex_band = load_complex[indices]

                # 计算厚度噪声在该谐频带的平均相位作为基准相位
                thick_phases = np.angle(thick_complex_band)
                # 相位解缠绕后取平均，避免边界跳变影响
                unwrapped_thick_phases = np.unwrap(thick_phases)
                ref_phase = np.mean(unwrapped_thick_phases)

                # 计算载荷噪声在该谐频带的相位
                load_phases = np.angle(load_complex_band)
                unwrapped_load_phases = np.unwrap(load_phases)

                # 计算相位差
                phase_diffs = unwrapped_load_phases - ref_phase
                # 将相位差限制在[-π, π]范围内
                phase_diffs = np.mod(phase_diffs + np.pi, 2 * np.pi) - np.pi

                if check_phase_consistency:
                    phase_stats['phase_diffs'].extend(phase_diffs.tolist())
                    phase_stats['phase_diff_variances'].append(np.var(phase_diffs))

                # 计算载荷幅值
                load_amps = np.abs(load_complex_band)

                # 同相分量（定常载荷）：幅值 * cos(相位差)，相位与基准相位一致
                steady_amps = load_amps * np.cos(phase_diffs)
                steady_complex_band = steady_amps * np.exp(1j * ref_phase)

                # 正交分量（非定常载荷）：幅值 * sin(相位差)，相位比基准相位超前π/2
                unsteady_amps = load_amps * np.sin(phase_diffs)
                unsteady_complex_band = unsteady_amps * np.exp(1j * (ref_phase + np.pi/2))

                # 赋值到结果数组
                steady_load_complex[indices] = steady_complex_band
                unsteady_load_complex[indices] = unsteady_complex_band

        # 3. 宽频区域（谐频带外）的载荷全部作为非定常载荷
        all_harmonic_indices = np.concatenate(harmonic_indices_list) if harmonic_indices_list else np.array([], dtype=int)
        broadband_mask = np.ones(len(self.freqs), dtype=bool)
        broadband_mask[all_harmonic_indices] = False
        unsteady_load_complex[broadband_mask] = load_complex[broadband_mask]

        # 4. 能量守恒验证
        original_energy = np.sum(np.abs(load_complex) ** 2)
        separated_energy = np.sum(np.abs(steady_load_complex) ** 2) + np.sum(np.abs(unsteady_load_complex) ** 2)
        energy_error = abs(separated_energy - original_energy) / original_energy if original_energy > 0 else 0

        if energy_error > 0.001:  # 本项目要求严格，允许0.1%的误差
            print(f"Warning: Energy conservation error {energy_error:.2%} in load separation")

        # 5. 提取幅值频谱返回（保持与原有接口兼容）
        steady_load_amp = np.abs(steady_load_complex)
        unsteady_load_amp = np.abs(unsteady_load_complex)

        # 6. 相位一致性统计
        if check_phase_consistency and phase_stats['phase_diffs']:
            phase_stats['mean_phase_diff'] = np.mean(phase_stats['phase_diffs'])
            phase_stats['overall_phase_diff_variance'] = np.var(phase_stats['phase_diffs'])
            phase_stats['max_phase_diff_variance'] = np.max(phase_stats['phase_diff_variances']) if phase_stats['phase_diff_variances'] else 0.0
            # 转换为角度更直观
            phase_stats['mean_phase_diff_deg'] = np.rad2deg(phase_stats['mean_phase_diff'])
            phase_stats['overall_phase_diff_variance_deg'] = np.rad2deg(phase_stats['overall_phase_diff_variance'])
            phase_stats['max_phase_diff_variance_deg'] = np.rad2deg(phase_stats['max_phase_diff_variance'])

        return steady_load_amp, unsteady_load_amp, phase_stats


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
        """
        执行完整的源项贡献分析（基于相位约束法）

        参数:
            thickness_complex: 厚度噪声复数频谱
            load_complex: 载荷噪声复数频谱
            total_complex: 总噪声复数频谱，如不提供则由厚度+载荷合成
            fundamental_freq: 基频频率，如不提供则自动识别
            harmonic_bandwidth_ratio: 谐频带宽比
            max_harmonic_order: 最大谐频阶数
            band_type: 频带分析类型，'octave'或'custom'
            band_fraction: 倍频程分数
            band_f_low: 最低分析频率
            band_f_high: 最高分析频率
            check_phase_consistency: 是否检查相位一致性，会在结果中添加相位统计信息

        返回:
            results: 包含所有分析结果的字典
        """
        # 提取幅值频谱用于后续分析（保持兼容）
        thickness_spectrum = np.abs(thickness_complex)
        load_spectrum = np.abs(load_complex)

        # 合成总频谱（如果未提供）
        if total_complex is None:
            total_complex = thickness_complex + load_complex
        total_spectrum = np.abs(total_complex)

        # 1. 谐频识别
        if fundamental_freq is None:
            # 自动识别基频（基于总幅值频谱，和原有逻辑保持一致）
            peak_result = self.peak_analyzer.analyze_spectrum(total_spectrum)
            fundamental_freq = peak_result['fundamental_freq']
            harmonic_freqs = peak_result['harmonic_freqs']
        else:
            # 使用提供的基频生成谐频序列
            harmonic_freqs = fundamental_freq * np.arange(1, max_harmonic_order + 1)
            # 过滤掉超过频率范围的谐频
            harmonic_freqs = harmonic_freqs[harmonic_freqs <= self.freqs.max()]

        # 2. 谐频与宽频分离（基于总幅值频谱，和原有逻辑保持一致）
        harmonic_spectrum, broadband_spectrum = self.freq_separator.separate_by_harmonic_extraction(
            total_spectrum,
            harmonic_freqs,
            harmonic_bandwidth_ratio
        )

        # 3. 定常与非定常载荷分离（相位约束法，使用复数频谱）
        steady_load, unsteady_load, phase_stats = self.load_separator.separate_load_noise(
            load_complex,
            thickness_complex,
            fundamental_freq,
            max_harmonic_order,
            harmonic_bandwidth_ratio,
            check_phase_consistency
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
            'detail_data': detail_data,
            'phase_stats': phase_stats if check_phase_consistency else None
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
