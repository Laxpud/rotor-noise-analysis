"""
谐频/宽频统计分析模块 (Harmonic/Broadband Statistical Analysis).

提供 `HarmonicBroadbandAnalyzer` 类，从频域幅值谱中检测谐频、分离
谐频/宽频分量，并生成多层次的统计信息：总体摘要、分频段统计、
逐谐频点明细、全频段逐点明细。

典型工作流
----------
>>> analyzer = HarmonicBroadbandAnalyzer(freqs)
>>> result = analyzer.analyze(
...     total_spectrum=amp_total, load_spectrum=amp_load,
...     bpf=None, max_order=30, bandwidth_ratio=0.03)
>>> print(result['summary'])
"""

import numpy as np
import pandas as pd
from spectral import PeakFrequencyAnalyzer, BandContributionAnalyzer
from .frequency import FrequencySeparator
from signal_utils import P_REF


class HarmonicBroadbandAnalyzer:
    """谐频/宽频统计分析器 -- 检测谐频、分离频谱、计算多层次统计.

    内部复用 ``PeakFrequencyAnalyzer`` 进行谐频检测、
    ``FrequencySeparator`` 进行谐频/宽频分离、
    ``BandContributionAnalyzer`` 进行 octave band 频段划分。

    Parameters
    ----------
    frequencies : np.ndarray
        频率轴 (Hz)，一维数组。
    """

    def __init__(self, frequencies: np.ndarray):
        self.freqs = np.asarray(frequencies)

    # ---- 谐频检测 ----

    def detect_harmonics(self, amplitude_spectrum: np.ndarray,
                         bpf: float = None, max_order: int = 30,
                         prominence: float = 0.005) -> dict:
        """检测谐频：自动识别 BPF 或使用用户指定的 BPF.

        Parameters
        ----------
        amplitude_spectrum : np.ndarray
            用于检测的幅值谱（通常用 Total 分量以获得最好的峰值信噪比）。
        bpf : float, optional
            叶片通过频率 (Hz)。为 None 时自动检测。
        max_order : int
            最大谐频阶数，默认 30。
        prominence : float
            自动检测时的峰值突出度阈值，默认 0.005。

        Returns
        -------
        dict
            fundamental_freq : float or None
            harmonic_freqs : np.ndarray
            harmonic_indices : list[int]
        """
        if bpf is not None:
            fundamental_freq = bpf
            max_freq = np.max(self.freqs)
            max_actual = min(max_order, int(max_freq / fundamental_freq))
            harmonic_indices = []
            for k in range(1, max_actual + 1):
                target = k * fundamental_freq
                idx = int(np.argmin(np.abs(self.freqs - target)))
                harmonic_indices.append(idx)
            harmonic_freqs = self.freqs[np.array(harmonic_indices)]
        else:
            analyzer = PeakFrequencyAnalyzer(self.freqs)
            result = analyzer.analyze_spectrum(amplitude_spectrum, prominence=prominence)
            fundamental_freq = result['fundamental_freq']
            harmonic_indices = list(result['harmonic_indices'])
            harmonic_freqs = np.asarray(result['harmonic_freqs'])

        return {
            'fundamental_freq': fundamental_freq,
            'harmonic_freqs': harmonic_freqs,
            'harmonic_indices': harmonic_indices,
        }

    # ---- 谐频/宽频分离 ----

    def separate(self, amplitude_spectrum: np.ndarray,
                 harmonic_freqs: np.ndarray,
                 bandwidth_ratio: float = 0.03) -> tuple:
        """将幅值谱分离为谐频分量和宽频分量.

        Parameters
        ----------
        amplitude_spectrum : np.ndarray
            待分离的幅值谱（如 Load 分量）。
        harmonic_freqs : np.ndarray
            谐频频率列表 (Hz)。
        bandwidth_ratio : float
            窄带提取的带宽比，默认 0.03。

        Returns
        -------
        (harmonic_amp, broadband_amp) : tuple[np.ndarray, np.ndarray]
        """
        separator = FrequencySeparator(self.freqs)
        return separator.separate_by_harmonic_extraction(
            amplitude_spectrum, harmonic_freqs, bandwidth_ratio
        )

    # ---- 总体统计摘要 ----

    def compute_summary(self, tag: str, amplitude_spectrum: np.ndarray,
                        harmonic_amp: np.ndarray, broadband_amp: np.ndarray,
                        fundamental_freq: float, num_harmonics: int) -> dict:
        """计算谐频/宽频的总体统计摘要（全局 + Low/Mid/High 分频段）.

        Parameters
        ----------
        tag : str
            列名前缀，FF 场景用 ``""``，FF+SR 场景用 ``"FF_"`` / ``"SR_"`` / ``"merged_"``.
        amplitude_spectrum : np.ndarray
            原始幅值谱（Load 分量）。
        harmonic_amp : np.ndarray
            分离后的谐频分量幅值谱。
        broadband_amp : np.ndarray
            分离后的宽频分量幅值谱。
        fundamental_freq : float
            基频 / BPF (Hz)。
        num_harmonics : int
            谐频数量。

        Returns
        -------
        dict
            单行 summary 数据，键为带 tag 前缀的列名。
        """
        total_energy = np.sum(amplitude_spectrum ** 2)
        harmonic_energy = np.sum(harmonic_amp ** 2)
        broadband_energy = np.sum(broadband_amp ** 2)

        harmonic_ratio = harmonic_energy / total_energy if total_energy > 0 else 0.0
        broadband_ratio = broadband_energy / total_energy if total_energy > 0 else 0.0
        harmonic_spl = 10.0 * np.log10(harmonic_energy / P_REF ** 2 + 1e-12)
        broadband_spl = 10.0 * np.log10(broadband_energy / P_REF ** 2 + 1e-12)

        result = {
            f'{tag}Fundamental Frequency(Hz)': fundamental_freq,
            f'{tag}Number of Harmonics': num_harmonics,
            f'{tag}Total Energy': total_energy,
            f'{tag}Harmonic Energy': harmonic_energy,
            f'{tag}Broadband Energy': broadband_energy,
            f'{tag}Harmonic Ratio': harmonic_ratio,
            f'{tag}Broadband Ratio': broadband_ratio,
            f'{tag}Harmonic SPL(dB)': harmonic_spl,
            f'{tag}Broadband SPL(dB)': broadband_spl,
        }

        # Low / Mid / High 分频段
        bands_def = {'Low': (0.0, 250.0), 'Mid': (250.0, 2000.0), 'High': (2000.0, np.inf)}
        for band_name, (f_low, f_high) in bands_def.items():
            idx = np.where((self.freqs >= f_low) & (self.freqs < f_high))[0]
            if len(idx) == 0:
                result[f'{tag}Harmonic Energy {band_name}'] = 0.0
                result[f'{tag}Harmonic Ratio {band_name}'] = 0.0
                result[f'{tag}Broadband Energy {band_name}'] = 0.0
                result[f'{tag}Broadband Ratio {band_name}'] = 0.0
                continue
            band_total = np.sum(amplitude_spectrum[idx] ** 2)
            band_h = np.sum(harmonic_amp[idx] ** 2)
            band_b = np.sum(broadband_amp[idx] ** 2)
            result[f'{tag}Harmonic Energy {band_name}'] = band_h
            result[f'{tag}Harmonic Ratio {band_name}'] = band_h / band_total if band_total > 0 else 0.0
            result[f'{tag}Broadband Energy {band_name}'] = band_b
            result[f'{tag}Broadband Ratio {band_name}'] = band_b / band_total if band_total > 0 else 0.0

        return result

    # ---- 分频段 (octave band) 统计 ----

    def compute_band_detail(self, amplitude_spectrum: np.ndarray,
                            harmonic_amp: np.ndarray, broadband_amp: np.ndarray,
                            f_low: float = 10.0, f_high: float = 20000.0,
                            fraction: int = 3) -> pd.DataFrame:
        """按 octave band 计算谐频/宽频的能量、占比和 SPL.

        Parameters
        ----------
        amplitude_spectrum : np.ndarray
            原始幅值谱。
        harmonic_amp : np.ndarray
            谐频分量幅值谱。
        broadband_amp : np.ndarray
            宽频分量幅值谱。
        f_low : float
            频带中心频率下限 (Hz)。
        f_high : float
            频带中心频率上限 (Hz)。
        fraction : int
            Octave band 分数 (1, 3, 12)。

        Returns
        -------
        pd.DataFrame
            每行一个 octave band。
        """
        band_analyzer = BandContributionAnalyzer(self.freqs)
        bands = band_analyzer.create_octave_bands(fraction=fraction, f_low=f_low, f_high=f_high)

        rows = []
        for band in bands:
            indices = np.where(
                (self.freqs >= band['lower']) & (self.freqs <= band['upper'])
            )[0]
            if len(indices) == 0:
                continue

            total_e = np.sum(amplitude_spectrum[indices] ** 2)
            harmonic_e = np.sum(harmonic_amp[indices] ** 2)
            broadband_e = np.sum(broadband_amp[indices] ** 2)

            total_spl = 10.0 * np.log10(total_e / P_REF ** 2 + 1e-12) if total_e > 0 else -np.inf
            h_ratio = harmonic_e / total_e if total_e > 0 else 0.0
            b_ratio = broadband_e / total_e if total_e > 0 else 0.0
            h_spl = 10.0 * np.log10(harmonic_e / P_REF ** 2 + 1e-12) if harmonic_e > 0 else -np.inf
            b_spl = 10.0 * np.log10(broadband_e / P_REF ** 2 + 1e-12) if broadband_e > 0 else -np.inf

            rows.append({
                'Center Frequency(Hz)': band['center'],
                'Lower Bound(Hz)': band['lower'],
                'Upper Bound(Hz)': band['upper'],
                'Total Energy': total_e,
                'Total SPL(dB)': total_spl,
                'Harmonic Energy': harmonic_e,
                'Harmonic Ratio': h_ratio,
                'Harmonic SPL(dB)': h_spl,
                'Broadband Energy': broadband_e,
                'Broadband Ratio': b_ratio,
                'Broadband SPL(dB)': b_spl,
            })

        return pd.DataFrame(rows)

    # ---- 逐谐频点明细 ----

    def compute_harmonic_detail(self, amplitude_spectrum: np.ndarray,
                                harmonic_indices: list,
                                fundamental_freq: float,
                                bandwidth_ratio: float = 0.03) -> pd.DataFrame:
        """计算每个谐频阶次的明细数据.

        Parameters
        ----------
        amplitude_spectrum : np.ndarray
            原始幅值谱。
        harmonic_indices : list[int]
            各谐频阶次对应的频率索引。
        fundamental_freq : float
            基频 (Hz)。
        bandwidth_ratio : float
            谐频提取带宽比，用于计算 Energy in Band。

        Returns
        -------
        pd.DataFrame
            每行一个谐频阶次。
        """
        total_harmonic_energy = 0.0
        band_energies = []
        rows = []

        # 第一遍：计算每个谐频带内的能量
        for k, idx in enumerate(harmonic_indices):
            order = k + 1
            nominal_freq = order * fundamental_freq
            bw = nominal_freq * bandwidth_ratio
            lower = nominal_freq - bw / 2.0
            upper = nominal_freq + bw / 2.0
            band_idx = np.where((self.freqs >= lower) & (self.freqs <= upper))[0]
            band_e = np.sum(amplitude_spectrum[band_idx] ** 2) if len(band_idx) > 0 else 0.0
            band_energies.append(band_e)
            total_harmonic_energy += band_e

        # 第二遍：构建输出行
        for k, idx in enumerate(harmonic_indices):
            order = k + 1
            nominal_freq = order * fundamental_freq
            actual_freq = self.freqs[idx]
            amp = amplitude_spectrum[idx]
            spl = 20.0 * np.log10(amp / P_REF + 1e-12) if amp > 0 else -np.inf
            band_e = band_energies[k]
            ratio = band_e / total_harmonic_energy if total_harmonic_energy > 0 else 0.0

            rows.append({
                'Harmonic Order': order,
                'Nominal Frequency(Hz)': nominal_freq,
                'Actual Frequency(Hz)': actual_freq,
                'Amplitude(Pa)': amp,
                'SPL(dB)': spl,
                'Energy in Band': band_e,
                'Ratio to Total Harmonic': ratio,
            })

        return pd.DataFrame(rows)

    # ---- 全频段逐点明细 ----

    def compute_full_detail(self, amplitude_spectrum: np.ndarray,
                            harmonic_amp: np.ndarray,
                            broadband_amp: np.ndarray) -> pd.DataFrame:
        """生成全频段逐点的谐频/宽频分离明细.

        Parameters
        ----------
        amplitude_spectrum : np.ndarray
            原始幅值谱。
        harmonic_amp : np.ndarray
            谐频分量幅值谱。
        broadband_amp : np.ndarray
            宽频分量幅值谱。

        Returns
        -------
        pd.DataFrame
            频率轴逐行数据。
        """
        eps = 1e-12
        rows = {
            'Frequency(Hz)': self.freqs,
            'Original Amplitude(Pa)': amplitude_spectrum,
            'Original SPL(dB)': 20.0 * np.log10(np.maximum(amplitude_spectrum, eps) / P_REF),
            'Harmonic Amplitude(Pa)': harmonic_amp,
            'Harmonic SPL(dB)': 20.0 * np.log10(np.maximum(harmonic_amp, eps) / P_REF),
            'Broadband Amplitude(Pa)': broadband_amp,
            'Broadband SPL(dB)': 20.0 * np.log10(np.maximum(broadband_amp, eps) / P_REF),
        }
        return pd.DataFrame(rows)

    # ---- 一站式分析 ----

    def analyze(self, total_spectrum: np.ndarray, load_spectrum: np.ndarray,
                bpf: float = None, max_order: int = 30,
                bandwidth_ratio: float = 0.03,
                band_fraction: int = 3, f_low: float = 10.0, f_high: float = 20000.0,
                tag: str = "") -> dict:
        """一站式分析：检测谐频 → 分离频谱 → 计算全部统计.

        Parameters
        ----------
        total_spectrum : np.ndarray
            Total 噪声幅值谱，用于谐频检测（更好的峰值信噪比）。
        load_spectrum : np.ndarray
            Load 噪声幅值谱，用于谐频/宽频分离和统计。
        bpf : float, optional
            叶片通过频率 (Hz)。为 None 则自动检测。
        max_order : int
            最大谐频阶数。
        bandwidth_ratio : float
            谐频提取带宽比。
        band_fraction : int
            Octave band 分数。
        f_low : float
            Octave band 中心频率下限 (Hz)。
        f_high : float
            Octave band 中心频率上限 (Hz)。
        tag : str
            列名前缀。

        Returns
        -------
        dict
            summary       — 总体统计 dict
            band_df       — 分频段统计 DataFrame
            harmonic_df   — 逐谐频点明细 DataFrame
            detail_df     — 全频段逐点明细 DataFrame
            fundamental_freq — 基频 (Hz)
        """
        harm_info = self.detect_harmonics(total_spectrum, bpf=bpf, max_order=max_order)
        fundamental_freq = harm_info['fundamental_freq']
        harmonic_freqs = harm_info['harmonic_freqs']
        harmonic_indices = harm_info['harmonic_indices']

        if fundamental_freq is None or len(harmonic_indices) == 0:
            return {
                'summary': {}, 'band_df': pd.DataFrame(),
                'harmonic_df': pd.DataFrame(), 'detail_df': pd.DataFrame(),
                'fundamental_freq': None,
            }

        harmonic_amp, broadband_amp = self.separate(
            load_spectrum, harmonic_freqs, bandwidth_ratio
        )

        summary = self.compute_summary(
            tag, load_spectrum, harmonic_amp, broadband_amp,
            fundamental_freq, len(harmonic_indices)
        )
        band_df = self.compute_band_detail(
            load_spectrum, harmonic_amp, broadband_amp, f_low, f_high, band_fraction
        )
        harmonic_df = self.compute_harmonic_detail(
            load_spectrum, harmonic_indices, fundamental_freq, bandwidth_ratio
        )
        detail_df = self.compute_full_detail(load_spectrum, harmonic_amp, broadband_amp)

        return {
            'summary': summary,
            'band_df': band_df,
            'harmonic_df': harmonic_df,
            'detail_df': detail_df,
            'fundamental_freq': fundamental_freq,
        }
