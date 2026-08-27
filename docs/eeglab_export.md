# BrainPy 区域活动导出到 EEGLAB

正式长时程配置为 `configs/full_300s.yaml`：动力学步长保持 `0.1 ms`，
区域放电率以 `1 ms` 间隔写入 `run.h5`，连续仿真 `300 s`。原始
`run.h5` 始终保留；EEGLAB 文件是由它生成的派生数据。

## 正式仿真

```bash
source /root/rivermind-data/ecmm/.venvs/ecmm-brainpy/bin/activate
cd /root/rivermind-data/ecmm/wsl_brainpy_project
export XLA_PYTHON_CLIENT_PREALLOCATE=false

ecmm simulate configs/full_300s.yaml \
  --artifact artifacts/full_seed_1256874 \
  --output runs/full_300s_sigma6p90_seed1256874
```

每个正式条件和 dynamics seed 必须使用独立输出目录。不要把多个独立的
`20 s` 运行拼接成一条信号，也不要覆盖历史运行目录。

## 导出 500 Hz EEGLAB 文件

```bash
python scripts/export_run_to_eeglab.py \
  runs/full_300s_sigma6p90_seed1256874 \
  --output-dir eeglab_exports/condition-A \
  --filename-stem sub-01_ses-1_task-simulation \
  --subject 01 \
  --condition A \
  --session 1
```

导出内容包括：

- `*.set`：EEGLAB 元数据；
- `*.fdt`：66通道、500 Hz、little-endian float32 外部数据；
- `*.export.json`：来源、采样率、维度、信号定义和SHA-256校验值。

默认通道名为 `ROI01` 到 `ROI66`。可通过 `--channel-labels labels.txt`
提供66行自定义脑区名称。导出器使用 `scipy.signal.resample_poly` 从1000 Hz
抗混叠降采样到500 Hz，不会修改源 `run.h5`。

这些通道表示模型脑区的群体放电率，不是头皮电极电压；写成 `.set/.fdt`
只用于复用 EEGLAB 和 MATLAB 分析流程，不能改变信号的物理含义。
