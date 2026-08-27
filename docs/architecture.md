# ECMM BrainPy 完整重构架构

## 项目边界

本项目的目标是用 Python、BrainPy 和 JAX 替代原 C++ 网络构建、动力学执行、在线监测、分析和批处理入口。离线 artifact 与运行时状态严格分离：网络结构参数改变时才重建 artifact，动力学参数改变时只创建新的 run。

## 分层结构

```text
ecmm.config     配置 schema、旧 SEED 读取、覆盖和校验
ecmm.data       tractography、artifact 底层数据与旧 fixture I/O
ecmm.offline    pattern、Legacy RNG、STDP 和 artifact 构建
ecmm.models     BrainPy 神经元、稀疏突触和后续输入模型
ecmm.runtime    仿真执行边界；后续加入 checkpoint/resume
ecmm.monitors   在线 rate、overlap、Fano、进度监测协议
ecmm.analysis   运行后统计与后续 avalanche 分析
ecmm.cli        统一命令行入口
```

根级旧模块目前保留为兼容实现，现有测试和外部脚本无需立即修改；新代码使用上述分层入口。

## 配置契约

`ProjectConfig` 由以下不可变 dataclass 构成：

- `NetworkConfig`：拓扑、模块、pattern、phase 和 tractography 参数。
- `RuntimeConfig`：动力学、噪声、积分、sigma 调度参数。
- `SeedConfig`：`seed/seed2/seed3/seed4` 的明确随机流语义。
- `ArtifactConfig`：artifact 名称、dtype 和离线 block 大小。
- `MonitorConfig`：rate/overlap 窗口及观察 pattern 数量。
- `ExecutionConfig`：进度、停止、playback、CPU/checkpoint 参数。
- `IOConfig`：运行名、目录、旧 file 位掩码和最大 spike 数。
- `CueConfig`：pattern、开始时间、spike 数与频率。

所有 YAML 未知字段都会报错；所有尺寸、范围、时间、拓扑和 cue 引用在运行前验证。`Z/K` 同时支持单个整数和逐模块数组。

## 旧参数映射

| 旧 SEED | 新配置 |
|---|---|
| `topo/S/Z/G/K/P/f/sort/swap/range` | `network.*` |
| `sigma/delta/alpha/rho/noise/bin/tmax` | `runtime.*` |
| `smin/smax` | `runtime.sigma_min/sigma_max` |
| `flush/flush2/tmin/twin/fmin/pout/pout2` | `monitors.*` |
| `rstop/nstop/tplay/progress_interval` | `execution.*` |
| `seed/seed2/seed3/seed4` | `seeds.network/stream/offline/dynamics` |
| `name/tmpdir/outdir/file/maxsp` | `io.*` |
| `begin cue description` | `cues[]` |

`seed3=0` 时离线构建使用 `seed`；`seed4=0` 时动力学使用 `seed`。这通过 `effective_offline/effective_dynamics` 属性明确表达。

## CLI

```bash
ecmm config validate configs/full.yaml
ecmm config show configs/full.yaml --set runtime.sigma=6.87
ecmm config convert ../SEED -o /tmp/ecmm-full.yaml
```

当前第 10 项只开放配置 CLI。`build/simulate/resume/analyze/scan` 将随对应运行模块完成后加入，避免暴露尚未满足契约的命令。
