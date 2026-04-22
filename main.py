"""
旋翼气动噪声分析工具主入口
支持多种分析功能：
1. 峰值频率与谐频分析
2. 频带能量贡献分析
3. 源项频域贡献量化分析（新功能）
"""
import argparse
import os
import sys
from typing import List

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def run_peak_analysis(file_path: str, filename_prefix: List[str], has_reflection: bool = False):
    """运行峰值频率分析"""
    if has_reflection:
        from analysis_cases import PeakFrequenceAnalyze
    else:
        from analysis_cases01 import PeakFrequenceAnalyze

    print(f"Running peak frequency analysis on {file_path}...")
    PeakFrequenceAnalyze(file_path, filename_prefix)
    print("Peak analysis completed!")


def run_band_analysis(file_path: str, filename_prefix: List[str], group_prefixes: List[str] = None, has_reflection: bool = False):
    """运行频带贡献分析"""
    if has_reflection:
        from analysis_cases import BandCountributionAnalyze
    else:
        from analysis_cases01 import BandCountributionAnalyze

    print(f"Running band contribution analysis on {file_path}...")
    BandCountributionAnalyze(file_path, filename_prefix, group_prefixes)
    print("Band analysis completed!")


def run_source_analysis(
    file_path: str,
    filename_prefix: List[str],
    group_prefixes: List[str] = None,
    has_reflection: bool = False,
    fundamental_freq: float = None,
    harmonic_bandwidth_ratio: float = 0.03,
    max_harmonic_order: int = 30,
    band_type: str = 'octave',
    band_fraction: int = 3,
    band_f_low: float = 10,
    band_f_high: float = 20000
):
    """运行源项频域贡献量化分析"""
    if has_reflection:
        from analysis_source_cases import SourceContributionAnalyze
    else:
        from analysis_source_cases01 import SourceContributionAnalyze

    print(f"Running source contribution analysis on {file_path}...")
    if fundamental_freq is not None:
        print(f"Using specified fundamental frequency: {fundamental_freq:.2f} Hz")
    else:
        print("Fundamental frequency will be automatically identified")

    SourceContributionAnalyze(
        file_path=file_path,
        filename_prefix=filename_prefix,
        group_prefixes=group_prefixes,
        fundamental_freq=fundamental_freq,
        harmonic_bandwidth_ratio=harmonic_bandwidth_ratio,
        max_harmonic_order=max_harmonic_order,
        band_type=band_type,
        band_fraction=band_fraction,
        band_f_low=band_f_low,
        band_f_high=band_f_high
    )
    print("Source contribution analysis completed!")


def run_full_analysis(file_path: str, filename_prefix: List[str], group_prefixes: List[str] = None, has_reflection: bool = False, **kwargs):
    """运行完整分析流程：峰值分析 -> 频带分析 -> 源项分析"""
    print("Running full analysis pipeline...")
    run_peak_analysis(file_path, filename_prefix, has_reflection)
    run_band_analysis(file_path, filename_prefix, group_prefixes, has_reflection)
    run_source_analysis(file_path, filename_prefix, group_prefixes, has_reflection, **kwargs)
    print("Full analysis pipeline completed!")


def main():
    parser = argparse.ArgumentParser(description="旋翼气动噪声分析工具")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 通用参数组
    def add_common_arguments(subparser):
        subparser.add_argument('file_path', help='数据文件所在目录路径')
        subparser.add_argument('filename_prefix', nargs='+', help='要处理的文件名前缀（不包含_FreqDomain.csv后缀）')
        subparser.add_argument('--group-prefixes', nargs='+', help='分组保存汇总数据的前缀列表')
        subparser.add_argument('--has-reflection', action='store_true', help='数据是否包含表面反射分量（即是否是analysis_cases.py对应的格式）')

    # 峰值分析命令
    peak_parser = subparsers.add_parser('peak', help='峰值频率与谐频分析')
    add_common_arguments(peak_parser)

    # 频带分析命令
    band_parser = subparsers.add_parser('band', help='频带能量贡献分析')
    add_common_arguments(band_parser)

    # 源项分析命令
    source_parser = subparsers.add_parser('source', help='源项频域贡献量化分析')
    add_common_arguments(source_parser)
    source_parser.add_argument('--fundamental-freq', type=float, help='基频（叶片通过频率），如不提供则自动识别')
    source_parser.add_argument('--harmonic-bandwidth-ratio', type=float, default=0.03, help='谐频带宽比（相对于谐频频率），默认0.03')
    source_parser.add_argument('--max-harmonic-order', type=int, default=30, help='最大谐频阶数，默认30')
    source_parser.add_argument('--band-type', choices=['octave', 'custom'], default='octave', help='频带类型，默认octave（倍频程）')
    source_parser.add_argument('--band-fraction', type=int, default=3, help='倍频程分数，1=1倍频程，3=1/3倍频程，12=1/12倍频程，默认3')
    source_parser.add_argument('--band-f-low', type=float, default=10, help='最低分析频率，默认10Hz')
    source_parser.add_argument('--band-f-high', type=float, default=20000, help='最高分析频率，默认20000Hz')

    # 完整分析命令
    full_parser = subparsers.add_parser('full', help='运行完整分析流程（峰值+频带+源项）')
    add_common_arguments(full_parser)
    full_parser.add_argument('--fundamental-freq', type=float, help='基频（叶片通过频率），如不提供则自动识别')
    full_parser.add_argument('--harmonic-bandwidth-ratio', type=float, default=0.03, help='谐频带宽比（相对于谐频频率），默认0.03')
    full_parser.add_argument('--max-harmonic-order', type=int, default=30, help='最大谐频阶数，默认30')
    full_parser.add_argument('--band-type', choices=['octave', 'custom'], default='octave', help='频带类型，默认octave（倍频程）')
    full_parser.add_argument('--band-fraction', type=int, default=3, help='倍频程分数，1=1倍频程，3=1/3倍频程，12=1/12倍频程，默认3')
    full_parser.add_argument('--band-f-low', type=float, default=10, help='最低分析频率，默认10Hz')
    full_parser.add_argument('--band-f-high', type=float, default=20000, help='最高分析频率，默认20000Hz')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # 执行对应命令
    if args.command == 'peak':
        run_peak_analysis(args.file_path, args.filename_prefix, args.has_reflection)
    elif args.command == 'band':
        run_band_analysis(args.file_path, args.filename_prefix, args.group_prefixes, args.has_reflection)
    elif args.command == 'source':
        run_source_analysis(
            args.file_path,
            args.filename_prefix,
            args.group_prefixes,
            args.has_reflection,
            args.fundamental_freq,
            args.harmonic_bandwidth_ratio,
            args.max_harmonic_order,
            args.band_type,
            args.band_fraction,
            args.band_f_low,
            args.band_f_high
        )
    elif args.command == 'full':
        run_full_analysis(
            args.file_path,
            args.filename_prefix,
            args.group_prefixes,
            args.has_reflection,
            fundamental_freq=args.fundamental_freq,
            harmonic_bandwidth_ratio=args.harmonic_bandwidth_ratio,
            max_harmonic_order=args.max_harmonic_order,
            band_type=args.band_type,
            band_fraction=args.band_fraction,
            band_f_low=args.band_f_low,
            band_f_high=args.band_f_high
        )


if __name__ == "__main__":
    main()
