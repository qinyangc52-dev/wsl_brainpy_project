import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import brainpy as bp
import jax


devices = jax.devices()
print(f"BrainPy: {bp.__version__}")
print(f"JAX: {jax.__version__}")
print(f"Devices: {devices}")
if not any(device.platform == "gpu" for device in devices):
    raise SystemExit("No JAX GPU device detected")
