# ECMM BrainPy 重构项目

本目录用于将原 C++ 模块化临界脑网络迁移到 Python + BrainPy/JAX。

完整分层和配置契约见 `docs/architecture.md`。正式规模配置为
`configs/full.yaml`；`configs/prototype.yaml` 仅用于快速回归测试。

当前阶段已实现离线网络构建、C++ golden fixture 和首个 BrainPy/JAX GPU 动力学原型：

1. 从 `tract1.c` 提取 66 区经验脑结构数据；
2. 复现原项目随机数发生器；
3. 构建 pattern、phase 和索引结构；
4. 分块生成 STDP CSR 权重；
5. 保存可长期复用的 network artifact；
6. 与原 C++ 小型网络逐项对照。
7. 使用 `DualExponentialLIF` 和事件 CSR 完成分段 GPU 仿真；
8. 保存 spike、module rate、phase overlap 和性能摘要。

## WSL 环境

```bash
cd /mnt/c/SAO/Extended-Criticality--Modular-Model-main/wsl_brainpy_project
source ~/.venvs/ecmm-brainpy/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python -m pip install -e .
```

虚拟环境放在 WSL 原生文件系统中，避免在 `/mnt/c` 安装大型 CUDA/JAX 包造成 NTFS I/O 阻塞。

## 构建 prototype artifact

```bash
python scripts/extract_tractography.py
sh scripts/build_legacy_fixture.sh
python scripts/build_artifact.py --config configs/prototype.yaml
pytest -q
```

生成物位于 `artifacts/prototype_seed_1256878/`。

## 配置验证与旧 SEED 转换

```bash
ecmm config validate configs/full.yaml
ecmm config show configs/full.yaml --set runtime.sigma=6.87
ecmm config convert ../SEED -o /tmp/ecmm-full.yaml
```

## 运行首个 GPU 实验

```bash
ecmm simulate configs/prototype.yaml \
  --artifact artifacts/prototype_seed_1256878 \
  --output runs/prototype

ecmm resume runs/prototype
ecmm analyze runs/prototype
```

生产运行使用 HDF5 流式保存和原子 checkpoint；实现与验收结果见
`docs/tasks11_13_completion_report.md`。GPU 版本使用同步固定步长，和原 C++ 的逐事件解析推进存在明确的数值语义差异。

## 远端 GPU 部署

远端 CUDA 13 环境使用验收版本锁文件，不再现场解析浮动依赖：

```bash
python3.12 -m venv ~/.venvs/ecmm-brainpy
source ~/.venvs/ecmm-brainpy/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/remote-gpu-cu13.lock
python -m pip install -e . --no-deps
python scripts/check_gpu.py
pytest -q
```

目录迁移后，优先使用 run manifest 中的相对 artifact 路径；只复制单个 run 时可显式覆盖：

```bash
ecmm resume runs/full --artifact artifacts/full_seed_1256874
ecmm analyze runs/full --artifact artifacts/full_seed_1256874
```

正式部署加固的完整契约见 `docs/production_hardening.md`。
