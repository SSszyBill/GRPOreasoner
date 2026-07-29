import os, glob, random
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/autodl-tmp/rl_playground/hf_cache"))
from datasets import load_dataset

def peek(name, ds, n=3):
    print(f"\n{'='*28} {name} {'='*28}")
    print("字段:", list(ds.features.keys()), "| 样本数:", len(ds))
    for idx, i in enumerate(random.sample(range(len(ds)), min(n, len(ds)))):
        print(f"\n--- {name} 样本 {idx} (row={i}) ---")
        for k, v in ds[i].items():
            s = str(v)
            print(f"[{k}] ({type(v).__name__}): {s[:600]}{' ...(截断)' if len(s)>600 else ''}")

# GSM8K：本地 train + test 都看
GSM_DIR = os.path.expanduser("~/autodl-tmp/rl_playground/data/gsm8k")
for split in ["train", "test"]:
    ds = load_dataset("parquet", data_files=os.path.join(GSM_DIR, f"{split}.parquet"), split="train")
    peek(f"GSM8K-{split}", ds, n=2)

# NuminaMath-CoT：不指定 split，整个 DatasetDict 全拉，先打印它到底有哪些 split
numina = load_dataset("AI-MO/NuminaMath-CoT")            # 返回 DatasetDict
print("\nNuminaMath 全部 split:", {k: len(v) for k, v in numina.items()})
for split_name, split_ds in numina.items():
    peek(f"NuminaMath-CoT-{split_name}", split_ds, n=3)