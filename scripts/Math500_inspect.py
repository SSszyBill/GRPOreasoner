import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/autodl-tmp/rl_playground/hf_cache"))
from datasets import load_dataset

ds = load_dataset("HuggingFaceH4/MATH-500")
print("splits:", {k: len(v) for k, v in ds.items()})
d = ds["test"]
print("字段:", list(d.features.keys()), "| 条数:", len(d))
for i in range(2):
    print(f"\n--- 样本 {i} ---")
    for k, v in d[i].items():
        s = str(v)
        print(f"[{k}]: {s[:400]}{'...(截断)' if len(s) > 400 else ''}")