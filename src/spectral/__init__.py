'''
声学数据频谱分析子包。

提供用于旋翼噪声频谱频域分析的类：
  - `PeakFrequencyAnalyzer`：峰值检测与谐波识别
  - `BandContributionAnalyzer`：倍频程频带能量分布分析

Usage
-----
    from spectral import PeakFrequencyAnalyzer, BandContributionAnalyzer
'''

from .peaks import PeakFrequencyAnalyzer
from .bands import BandContributionAnalyzer

# hatchling 构建系统使用的版本号
__version__ = "0.1.0"
