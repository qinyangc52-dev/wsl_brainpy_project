# 依赖锁定说明

`remote-gpu-cu13.lock` 是当前已验收环境的完整 `pip freeze` 快照，适用于
Python 3.12、Linux x86_64 和 CUDA 13。它不包含项目自身的 editable 路径。

远端安装：

```bash
python3.12 -m venv ~/.venvs/ecmm-brainpy
source ~/.venvs/ecmm-brainpy/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/remote-gpu-cu13.lock
python -m pip install -e . --no-deps
python scripts/check_gpu.py
pytest -q
```

只有在完整测试和 GPU 验收重新通过后，才应更新 lock 文件。若远端平台不是
Linux x86_64/CUDA 13，不能直接复用该文件，应为对应平台单独生成 lock。
