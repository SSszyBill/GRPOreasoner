#!/usr/bin/env python3
"""Merge verl LoRA .pt checkpoint into a standard HuggingFace model.

verl 保存的是 PEFT-wrapped state_dict（base_model.model.*.base_layer.weight
+ .lora_A.default.weight + .lora_B.default.weight），需要手动 merge 后才能
被 vLLM / eval harness 直接加载。
"""
import argparse, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def merge(ckpt_path, base_path, out_dir, alpha=32.0, rank=32, dtype="bfloat16"):
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[dtype]
    scale = alpha / rank
    print(f"[1/4] Loading state_dict from {ckpt_path} ...")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert isinstance(state, dict), f"expected dict, got {type(state)}"

    # Enumerate LoRA-targeted modules
    lora_modules = set()
    for k in state.keys():
        if ".lora_A.default.weight" in k:
            m = k.replace("base_model.model.", "", 1).replace(".lora_A.default.weight", "")
            lora_modules.add(m)
    target_types = sorted({m.split(".")[-1] for m in lora_modules})
    print(f"[2/4] Found {len(lora_modules)} LoRA modules")
    print(f"      Target module types: {target_types}")
    
    #   Build merged state_dict
    print(f"[3/4] Merging LoRA (alpha={alpha}, rank={rank}, scale={scale}) ...")
    new_state = {}
    for k, v in state.items():
        if ".lora_A." in k or ".lora_B." in k:
            continue
        assert k.startswith("base_model.model."), f"unexpected key: {k}"
        stripped = k[len("base_model.model."):]
        if ".base_layer." in stripped:
            module, suffix = stripped.split(".base_layer.")  # suffix = "weight" or "bias"
            target_key = f"{module}.{suffix}"
            if suffix == "weight" and module in lora_modules:
                A = state[f"base_model.model.{module}.lora_A.default.weight"].to(torch_dtype)
                B = state[f"base_model.model.{module}.lora_B.default.weight"].to(torch_dtype)
                W = v.to(torch_dtype) + scale * (B @ A)
                new_state[target_key] = W
            else:
                new_state[target_key] = v.to(torch_dtype) if v.dtype.is_floating_point else v
        else:
            new_state[stripped] = v.to(torch_dtype) if v.dtype.is_floating_point else v
    
    # Load into base HF architecture
    print(f"[4/4] Loading into {base_path} architecture and saving to {out_dir} ...")
    model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch_dtype)
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"      Missing keys: {len(missing)} | Unexpected: {len(unexpected)}")
    if missing[:3]:
        print(f"      (first missing): {missing[:3]}")
    if unexpected[:3]:
        print(f"      (first unexpected): {unexpected[:3]}")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_path).save_pretrained(out_dir)
    print(f"✅ Merged model saved to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="verl LoRA .pt checkpoint")
    p.add_argument("--base", required=True, help="Base HF model path")
    p.add_argument("--out", required=True, help="Output dir for merged model")
    p.add_argument("--alpha", type=float, default=32.0)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = p.parse_args()
    merge(args.ckpt, args.base, args.out, args.alpha, args.rank, args.dtype)