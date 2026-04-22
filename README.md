# 旋翼气动噪声分析系统

## 🎯 功能简介
本系统用于旋翼气动噪声的频域分析，包含以下功能：
1. **峰值频率与谐频分析**：识别基频和各阶谐频，提取谐频处的噪声级
2. **频带能量贡献分析**：1/3倍频程频带的能量分布统计，识别主导噪声频段
3. ✅ **源项频域贡献量化分析**（新增功能）：
   - 分离谐频噪声与宽频噪声
   - 分解厚度噪声、定常载荷噪声、非定常载荷噪声的能量占比
   - 分析各频带内的谐频能量占比和噪声源分布

## 📚 文档说明
完整使用说明请查看：[doc/源项频域贡献量化分析使用手册.md](doc/源项频域贡献量化分析使用手册.md)

## 🚀 快速使用
### 命令行方式（推荐）
```bash
# 查看帮助
python main.py --help

# 源项分析（仅自由场数据）
python main.py source Case01 Case01_Rotor_OBS0001 --group-prefixes Case01_Rotor

# 源项分析（含表面反射数据）
python main.py source Case03 Case03_Rotor_OBS0001 --has-reflection --group-prefixes Case03_Rotor --fundamental-freq 25.0

# 完整分析流程
python main.py full Case03 Case03_Rotor_OBS0001 --has-reflection --fundamental-freq 25.0
```

### 直接运行脚本
修改对应脚本中的路径配置，然后运行：
```bash
# 仅自由场源项分析
python src/analysis_source.py

# 含表面反射源项分析
python src/analysis_source01.py
```

## 📦 依赖环境
- Python 3.8+
- NumPy
- SciPy
- Pandas

安装依赖：
```bash
pip install numpy scipy pandas
```

## 📊 示例数据
- `Case01/`：仅自由场数据示例
- `Case03/`：含表面反射数据示例

每个示例数据包含`*_FreqDomain.csv`格式的频域数据，可直接用于测试。

---
*Original: Rotor noise analysis and visualization tools for personal data analysis.*
