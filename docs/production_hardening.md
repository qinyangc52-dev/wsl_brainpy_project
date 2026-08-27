# 正式部署加固说明

本轮加固不改变 BrainPy 动力学公式、随机流或离线网络生成算法，只约束配置、artifact、运行记录和远端环境之间的关系。

## 1. Artifact 结构强校验

`SimulationRunner` 和分析入口都会用当前配置重新计算 `structural_hash`，并与 artifact manifest 比较。哈希覆盖：

- 完整 `network` 配置；
- `seeds.network`；
- `artifact.dtype`。

结构哈希一致后，还会重新计算 `patterns.npz` 和 `connectivity_csr.npz` 的 SHA256。网络参数、网络 seed 或 dtype 改变后必须重建 artifact；动力学、监测和运行时参数改变不需要重建。

## 2. 可迁移的 resume 和 analyze

run manifest schema 2 同时保存：

- `artifact_dir`：创建或最近一次恢复时的绝对路径；
- `artifact_relative`：相对 run 目录的可迁移路径；
- `artifact_identity`：结构哈希和两个数据文件哈希。

整个项目树迁移后会优先使用相对路径。只复制 run 目录或重新组织 artifact 时，显式指定：

```bash
ecmm resume runs/full --artifact artifacts/full_seed_1256874
ecmm analyze runs/full --artifact artifacts/full_seed_1256874
```

显式 artifact 仍须通过结构、文件完整性和 run identity 校验。

## 3. 远端依赖锁定

`requirements/remote-gpu-cu13.lock` 来自已通过完整测试的 Python 3.12、Linux x86_64、CUDA 13 WSL 环境，包含 BrainPy、JAX/JAXLIB、CUDA plugin、NumPy、SciPy、HDF5、分析和测试依赖的精确版本。

安装顺序：

```bash
python -m pip install -r requirements/remote-gpu-cu13.lock
python -m pip install -e . --no-deps
python scripts/check_gpu.py
pytest -q
```

运行器和 GPU 检查脚本默认设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false`，避免 JAX 在共享 GPU 或显存已有占用时一次性预留大部分显存。若远端 5090 为独占设备且已验证预分配策略，可以在启动进程前显式覆盖该环境变量。

其他操作系统、CPU 架构或 CUDA 主版本应生成独立 lock，不能覆盖此文件后继续声称环境等价。

## 4. Checkpoint 与 spike 上限

`execution.checkpoint_interval_s` 现在表示两次周期 checkpoint 之间的墙钟秒数：

- `> 0`：达到该墙钟间隔后，在当前 chunk 边界保存；
- `0`：每个 chunk 保存；
- 无论间隔是多少，启动基线、正常完成、暂停、SIGINT、错误、非有限状态、rate stop 和 wall-time stop 都会强制保存。

HDF5 仍按 chunk 持续追加。若进程被 `SIGKILL` 或服务器断电，resume 会把 checkpoint 之后尚未提交的 HDF5 尾部截断，再从最后 checkpoint 重算。

`io.max_spikes` 只限制 `spikes3-*.dat` 的 legacy 文本导出行数，不限制 `run.h5`。每次导出会生成 `legacy/export_manifest.json`，记录 HDF5 spike 总数、实际导出数、上限和是否发生截断。
