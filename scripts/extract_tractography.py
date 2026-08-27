from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))

from ecmm.data import save_tractography


output = save_tractography(ROOT / "tract1.c", PROJECT / "data" / "tractography_66.npz")
print(output)
