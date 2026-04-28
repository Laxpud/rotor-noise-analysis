"""
噪声分解子包 (Noise Decomposition Sub-package).

本子包提供旋翼气动噪声的多层次分解与源项贡献分析功能，
支持以下三类分析方法：

1. **谐频/宽频分离** (:class:`FrequencySeparator`)
   - 将总噪声频谱分解为离散谐频分量和连续宽频分量。
   - 基于已知谐频频率（BPF 及其倍频）的窄带提取法。

2. **相位约束载荷分离** (:class:`PhaseConstraintSeparator`)
   - 以厚度噪声相位为参考，将载荷噪声分解为定常和非定常分量。
   - 基于同相/正交投影的频域分解方法。
   - **注意**: 内含"同相假设"，建议使用 CyclicSpectrumAnalyzer 交叉验证。

3. **循环平稳谱分析** (:class:`CyclicSpectrumAnalyzer`)
   - 不依赖厚度噪声参考，通过 FAM 算法计算循环谱密度 (SCD)。
   - 从统计循环平稳特性定量评估定常/非定常载荷贡献。

4. **源项贡献分析** (:class:`SourceContributionAnalyzer`)
   - 集成以上方法的完整分析流水线。
   - 辅以频带贡献分析 (:class:`SourceBandAnalyzer`)。

典型工作流
----------
>>> from src.decomposition import SourceContributionAnalyzer
>>> analyzer = SourceContributionAnalyzer(frequencies)
>>> result = analyzer.analyze(thickness_fft, load_fft, fundamental_freq=bpf)
>>> print(result['global_stats']['steady_load_total_ratio'])
"""

from .frequency import FrequencySeparator
from .phase_constraint import PhaseConstraintSeparator
from .cyclic_spectrum import CyclicSpectrumAnalyzer
from .contribution import SourceBandAnalyzer, SourceContributionAnalyzer

__all__ = [
    'FrequencySeparator',
    'PhaseConstraintSeparator',
    'CyclicSpectrumAnalyzer',
    'SourceBandAnalyzer',
    'SourceContributionAnalyzer',
]
