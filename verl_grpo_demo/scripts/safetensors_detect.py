from pathlib import Path
from safetensors import safe_open

MODEL_DIR = Path("/root/autodl-tmp/rl_playground/hf_cache/hub/"
                 "models--Qwen--Qwen2.5-1.5B/snapshots/"
                 "8faed761d45a263340a0528343f099c05c9a4323")

shards = sorted(MODEL_DIR.glob("*.safetensors"))
assert shards, f"no safetensors under {MODEL_DIR}"
print(f"[shards] {len(shards)} file(s)")

# with safe_open(shards[0], framework="pt", device="cpu") as f:
#     print(f"[keys] {len(f.keys())} key(s)")
#     print(f.metadata())

from collections import Counter
dtypes = Counter()
with safe_open(shards[0], framework="pt", device="cpu") as f:
    for k in f.keys():
        dtypes[f.get_slice(k).get_dtype()] += 1
print(dtypes)