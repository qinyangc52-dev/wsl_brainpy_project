# ECMM BrainPy：扩展临界性模块化脑网络

本项目将原 C/C++ **Extended Criticality Modular Model（ECMM）** 重构为
Python + BrainPy + JAX 实现。项目覆盖网络离线构建、GPU 动力学仿真、断点续跑、
在线监测、雪崩分析、EEGLAB 导出和多参数批量实验。

项目采用“离线网络 artifact + 运行时 simulation”的分层方式：网络结构、pattern
或 STDP 参数发生变化时重新构建 artifact；只改变 `sigma`、噪声、仿真时长等动力学
参数时，可复用已有 artifact 创建新的 run。

> 注意：BrainPy 版本使用同步固定步长推进，与原 C/C++ 程序的逐事件解析推进存在
> 数值语义差异。`src/ecmm/models/reference.py` 提供小规模事件参考实现，用于验证这种
> 近似误差。

## 主要功能

- 从原始 `tract1.c` 提取 66 区 tractography 数据；
- 复现原程序的随机数发生器和 pattern 构建过程；
- 按原 STDP 核函数分块构建稀疏 CSR 连接矩阵；
- 使用 BrainPy/JAX 在 CPU 或 GPU 上运行双指数 LIF 网络；
- 保存全部 spike、模块放电率、phase overlap、统计量和运行摘要；
- 使用 HDF5 流式写入与原子 checkpoint 支持暂停和断点续跑；
- 执行神经雪崩、幂律和 size-duration scaling 分析；
- 将区域放电率导出为 EEGLAB `.set/.fdt` 文件；
- 批量执行 5 个 `sigma` × 5 个 seed 的 300 秒参数扫描。

## 项目结构

下面的目录树以当前 Git 仓库为准。`artifacts/`、`runs/`、`figures/` 等本地生成
目录也列在树中，但默认不会提交到 Git。

```text
wsl_brainpy_project/
├── README.md                         # 项目总览、目录说明和使用入口
├── pyproject.toml                    # Python 包信息、依赖、CLI 和 pytest 配置
├── .gitignore                        # 忽略虚拟环境、artifact、run 和大型结果
│
├── configs/                          # YAML 实验配置
│   ├── prototype.yaml                # 小规模快速回归配置
│   ├── full.yaml                     # 66 区正式规模、20 秒配置
│   ├── full_300s.yaml                # 66 区正式规模、300 秒配置
│   └── generated/                    # 各网络 seed 的正式配置
│       ├── full_seed_1256874.yaml
│       ├── full_seed_1256875.yaml
│       ├── full_seed_1256876.yaml
│       ├── full_seed_1256877.yaml
│       └── full_seed_1256878.yaml
│
├── data/
│   └── tractography_66.npz           # 从 tract1.c 提取的脑区大小、距离和纤维权重
│
├── requirements/
│   ├── README.md                     # CUDA 13 锁文件的适用范围和更新规则
│   └── remote-gpu-cu13.lock          # Python 3.12/Linux/CUDA 13 验收依赖快照
│
├── docs/
│   ├── architecture.md               # 包分层、配置契约和旧 SEED 参数映射
│   ├── production_hardening.md       # artifact、checkpoint 和远端部署约束
│   ├── eeglab_export.md              # EEGLAB 导出格式与使用方法
│   └── 300s精细扫描.md                # 25 个正式长时程实验的设计和运行说明
│
├── src/ecmm/                         # ecmm Python 包
│   ├── __init__.py                   # 包级公共入口
│   ├── artifacts.py                  # artifact 保存、哈希、完整性验证和路径解析
│   ├── connectome.py                 # tractography 提取、读取和有效连接计算
│   ├── dynamics.py                   # BrainPy 稀疏突触、双指数 LIF 和完整网络
│   ├── eeglab.py                     # run.h5 到 EEGLAB SET/FDT 的转换
│   ├── legacy_fixture.py             # 读取原 C/C++ 网络连接 golden fixture
│   ├── legacy_rng.py                 # 与旧 random.c 对齐的随机数发生器
│   ├── patterns.py                   # pattern、phase、神经元和索引结构构建
│   ├── simulation.py                 # 兼容旧调用方式的仿真入口
│   ├── stdp.py                       # STDP 核函数和分块 CSR 权重生成
│   │
│   ├── config/
│   │   ├── __init__.py               # 配置模块公共导出
│   │   ├── schema.py                 # 配置 dataclass、默认值和合法性校验
│   │   ├── loaders.py                # YAML/旧 SEED 加载、覆盖和序列化
│   │   └── legacy.py                 # 旧 SEED 文本及 cue 解析器
│   │
│   ├── data/
│   │   └── __init__.py               # tractography 与旧 fixture I/O 统一入口
│   ├── offline/
│   │   └── __init__.py               # 离线网络构建 API 聚合入口
│   ├── models/
│   │   ├── __init__.py               # 动力学模型公共导出
│   │   ├── inputs.py                 # 噪声、cue 和 sigma 时间调度输入
│   │   └── reference.py              # 原事件推进语义的小规模参考实现
│   ├── runtime/
│   │   ├── __init__.py               # 运行时 API 聚合入口
│   │   ├── runner.py                 # 分块仿真、恢复、停止条件和运行摘要
│   │   ├── checkpoint.py             # 原子保存和加载网络/监测器状态
│   │   └── store.py                  # run.h5 的流式写入、截断和数据集管理
│   ├── monitors/
│   │   ├── __init__.py               # 监测器协议和公共导出
│   │   └── suite.py                  # rate、phase overlap 和活动统计监测器
│   ├── analysis/
│   │   ├── __init__.py               # 分析模块公共导出
│   │   ├── metrics.py                # 分箱放电率和 phase overlap 指标
│   │   ├── avalanche.py              # 雪崩提取、幂律拟合和基础绘图
│   │   ├── legacy.py                 # 导出兼容原程序的文本结果
│   │   └── pipeline.py               # legacy 导出与雪崩分析的统一流水线
│   └── cli/
│       ├── __init__.py               # CLI 包标记
│       └── main.py                   # ecmm config/simulate/resume/analyze 命令
│
├── scripts/                          # 独立工作流和运维脚本
│   ├── extract_tractography.py       # 从仓库上级 tract1.c 生成 tractography_66.npz
│   ├── build_legacy_fixture.sh       # 编译并运行原 C/C++ prototype，生成基准连接
│   ├── build_artifact.py             # 构建 pattern、STDP CSR 和 artifact manifest
│   ├── run_prototype.py              # 运行或恢复小规模 BrainPy prototype
│   ├── check_gpu.py                  # 检查 BrainPy/JAX 版本和 GPU 可见性
│   ├── check_backend_consistency.py  # 用固定输入比较 CPU/GPU 后端轨迹
│   ├── setup_wsl_gpu.sh              # 创建 WSL 虚拟环境并安装 CUDA 13 依赖
│   ├── run_fine_sigma_seed5_300s.py  # 顺序执行 5 sigma × 5 seed 的 300 秒扫描
│   ├── summarize_300s_scan_avalanche.py # 汇总 25 个 run 的雪崩及标度指标
│   ├── plot_sigma11_fig5.py          # 绘制单个最佳 run 的三面板 Fig.5 风格图
│   ├── plot_300s_all_runs.py         # 为全部 25 个 run 生成统一坐标范围图
│   ├── export_run_to_eeglab.py       # 导出单个 run 为 EEGLAB SET/FDT
│   ├── export_scan_to_eeglab.py      # 校验并批量导出 25 个正式 run
│   ├── extract_zstd_tar.py           # 安全解压远端 .tar.zst 快照
│   └── sync_remote_project.sh        # 下载、校验、备份并同步远端项目快照
│
├── tests/                            # pytest 自动化测试
│   ├── test_config_contract.py       # 配置契约、旧 SEED、覆盖和迁移路径测试
│   ├── test_connectome.py            # tractography 形状、对称性和公式测试
│   ├── test_legacy_rng.py            # 旧随机数序列及可复现性测试
│   ├── test_patterns.py              # pattern bank 不变量测试
│   ├── test_stdp.py                  # STDP 分支和 CSR 对角线测试
│   ├── test_golden_fixture.py        # Python artifact 与 C/C++ fixture 对照
│   ├── test_dynamics.py              # 稀疏方向、双指数状态和抑制规则测试
│   ├── test_event_reference.py       # 固定步长与事件参考模型对照
│   ├── test_inputs_and_state.py      # 噪声、cue、sigma 和状态恢复测试
│   ├── test_runner_resume.py         # pause/resume、checkpoint 和 artifact 校验
│   ├── test_analysis.py              # rate 与 phase overlap 基础指标测试
│   ├── test_monitors_analysis.py     # 监测、legacy 导出和雪崩端到端测试
│   ├── test_eeglab_export.py         # EEGLAB 数据布局和元数据测试
│   └── test_fine_sigma_seed5_300s.py # 25 任务扫描计划和命令测试
│
├── artifacts/                        # [生成/忽略] 可复用网络 artifact
├── runs/                             # [生成/忽略] 仿真结果、checkpoint 和 run.h5
├── figures/                          # [生成/忽略] 绘图结果
├── comparison/                       # [生成/忽略] 后端或扫描比较结果
├── eeglab_exports/                   # [生成/忽略] EEGLAB SET/FDT 文件
├── logs/                             # [生成/忽略] 批量任务日志
├── remote_results/                   # [生成/忽略] 远端下载结果
├── legacy_reference/                 # [生成/忽略] C/C++ prototype 基准输出
├── .venv/                            # [本地/忽略] NTFS 中的本地环境目录
└── .pytest_cache/                    # [生成/忽略] pytest 缓存
```

## 核心数据流

```text
tract1.c
   │  scripts/extract_tractography.py
   ▼
data/tractography_66.npz
   │  scripts/build_artifact.py
   ▼
artifacts/<artifact_name>/
   ├── patterns.npz
   ├── connectivity_csr.npz
   └── manifest.json
   │  ecmm simulate / SimulationRunner
   ▼
runs/<run_name>/
   ├── config.resolved.yaml
   ├── run_manifest.json
   ├── checkpoint.npz
   ├── run.h5
   └── summary.json
   │  ecmm analyze / EEGLAB export / plotting scripts
   ▼
analysis/、legacy/、figures/、comparison/、eeglab_exports/
```

### Artifact 内容

- `patterns.npz`：pattern 选择、phase、模块位置和神经元索引；
- `connectivity_csr.npz`：以 `post × pre` 方向保存的稀疏 STDP 权重；
- `manifest.json`：结构配置、seed、形状、非零元数量及 SHA-256 校验值。

### Run 内容

- `config.resolved.yaml`：应用命令行覆盖后的最终配置；
- `run_manifest.json`：配置哈希、artifact 身份、路径和运行状态；
- `checkpoint.npz`：网络状态、监测器状态和已完成 step；
- `run.h5`：全部 spike、模块 rate、统计量和 phase overlap；
- `summary.json`：完成状态、耗时、设备、spike 数和性能摘要。

## 环境要求

推荐环境为 Windows 11 + WSL2 Ubuntu、Python 3.12、NVIDIA GPU，以及
CUDA 13 对应的 JAX/BrainPy 依赖。虚拟环境建议放在 WSL 的 Linux 文件系统，
避免在 `/mnt/c` 上安装大量 Python 小文件造成 NTFS I/O 开销。

```bash
wsl -d Ubuntu
cd /mnt/c/SAO/Extended-Criticality--Modular-Model-main/wsl_brainpy_project
bash scripts/setup_wsl_gpu.sh
source ~/.venvs/ecmm-brainpy/bin/activate
```

手动安装方式：

```bash
python3.12 -m venv ~/.venvs/ecmm-brainpy
source ~/.venvs/ecmm-brainpy/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/remote-gpu-cu13.lock
python -m pip install -e . --no-deps
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python scripts/check_gpu.py
```

## 快速开始

### 1. 验证配置

```bash
ecmm config validate configs/prototype.yaml
ecmm config show configs/full.yaml --set runtime.sigma=6.87
ecmm config convert ../SEED -o /tmp/ecmm-full.yaml
```

### 2. 构建网络 artifact

```bash
python scripts/extract_tractography.py
sh scripts/build_legacy_fixture.sh
python scripts/build_artifact.py --config configs/prototype.yaml
pytest -q
```

### 3. 运行和恢复 prototype

```bash
ecmm simulate configs/prototype.yaml \
  --artifact artifacts/prototype_seed_1256878 \
  --output runs/prototype

ecmm resume runs/prototype
```

如果只迁移了单个 run，可显式覆盖失效的 artifact 路径：

```bash
ecmm resume runs/prototype --artifact artifacts/prototype_seed_1256878
```

### 4. 分析结果

```bash
ecmm analyze runs/prototype
ecmm analyze runs/prototype --no-figure
```

### 5. 导出 EEGLAB

```bash
python scripts/export_run_to_eeglab.py runs/prototype \
  --output-dir eeglab_exports/prototype \
  --filename-stem sub-01_task-ecmm \
  --target-sfreq 500
```

导出的信号是 **66 个模型脑区的群体放电率**，不是头皮 EEG 电压。

### 6. 执行 300 秒参数扫描

```bash
# 仅检查 25 个任务计划
python scripts/run_fine_sigma_seed5_300s.py --dry-run

# 正式运行，并允许恢复未完成任务
python scripts/run_fine_sigma_seed5_300s.py --resume-incomplete

# 汇总、绘图和 EEGLAB 批量导出
python scripts/summarize_300s_scan_avalanche.py
python scripts/plot_300s_all_runs.py
python scripts/export_scan_to_eeglab.py
```

## 测试

```bash
pytest -q
python scripts/check_gpu.py
```

CPU/GPU 后端比较：

```bash
JAX_PLATFORMS=cpu python scripts/check_backend_consistency.py \
  --output comparison/backend_cpu.npz

python scripts/check_backend_consistency.py \
  --output comparison/backend_gpu.npz
```

## 配置分层

| 配置段 | 作用 |
|---|---|
| `network` | 模块数、神经元数、pattern、拓扑、phase 和 tractography 参数 |
| `runtime` | `sigma/delta/alpha/rho`、时间步长、时长和 sigma 调度 |
| `seeds` | 网络、随机流、离线构建和动力学 seed |
| `artifact` | artifact 名称、权重类型和 STDP 分块大小 |
| `monitors` | rate、overlap、flush 和 playback 统计窗口 |
| `execution` | 进度、停止条件、运行时限和 checkpoint 周期 |
| `io` | run 名称、输出路径和 legacy spike 导出上限 |
| `cues` | 强制 pattern cue 的时间、频率和 spike 数 |

## 进一步文档

- [`docs/architecture.md`](docs/architecture.md)：软件分层和配置契约；
- [`docs/production_hardening.md`](docs/production_hardening.md)：正式运行的完整性约束；
- [`docs/eeglab_export.md`](docs/eeglab_export.md)：EEGLAB 导出说明；
- [`docs/300s精细扫描.md`](docs/300s精细扫描.md)：300 秒 × 25 任务实验设计；
- [`requirements/README.md`](requirements/README.md)：GPU 依赖锁定说明。

## 生成文件与版本控制

以下目录可能包含大型二进制结果，默认由 `.gitignore` 排除：

```text
artifacts/  runs/  remote_results/  eeglab_exports/
figures/    comparison/  logs/       legacy_reference/
```

提交代码前建议执行：

```bash
git status --short
pytest -q
```
