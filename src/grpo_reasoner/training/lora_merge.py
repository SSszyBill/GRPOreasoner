"""Merge a verl LoRA .pt checkpoint into a standard HuggingFace model.

verl saves PEFT-wrapped state_dict:
    base_model.model.<layer>.base_layer.weight
    base_model.model.<layer>.lora_A.default.weight
    base_model.model.<layer>.lora_B.default.weight

We fold the A/B updates into base_layer, strip the PEFT prefix, and save a
plain HF model directory that vLLM / eval harness can load directly.

CLI:
    python -m grpo_reasoner.training.lora_merge \\
        --ckpt path/to/actor.pt \\
        --base Qwen/Qwen2.5-1.5B-Instruct \\
        --out  path/to/merged/ \\
        --alpha 32 --rank 32 --dtype bfloat16
"""
from __future__ import annotations

import argparse
import os
from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTypeName = Literal["bfloat16", "float16", "float32"]
_DTYPE_MAP: dict[str, torch.dtype] = {
"bfloat16": torch.bfloat16,
"float16": torch.float16,
"float32": torch.float32,
}
def merge(
    ckpt_path: str,
    base_path: str,
    out_dir: str,
    alpha: float = 32.0,
    rank: int = 32,
    dtype: DTypeName = "bfloat16",
    logger=print,
    ) -> None:
    """Merge LoRA delta into base weights and save a standard HF dir."""
    torch_dtype = _DTYPE_MAP[dtype]
    scale = alpha / rank
    logger(f"[1/4] Loading state_dict from {ckpt_path} ...")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert isinstance(state, dict), f"expected dict, got {type(state)}"
    lora_modules: set[str] = set()
    for key in state.keys():
        if ".lora_A.default.weight" in key:
            module = (
                key.replace("base_model.model.", "", 1)
                .replace(".lora_A.default.weight", "")
            )
            lora_modules.add(module)
    target_types = sorted({m.split(".")[-1] for m in lora_modules})
    logger(f"[2/4] Found {len(lora_modules)} LoRA modules")
    logger(f"      Target module types: {target_types}")
    logger(f"[3/4] Merging LoRA (alpha={alpha}, rank={rank}, scale={scale}) ...")
    new_state: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if ".lora_A." in key or ".lora_B." in key:
            continue
        assert key.startswith("base_model.model."), f"unexpected key: {key}"
        stripped = key[len("base_model.model."):]
        if ".base_layer." in stripped:
            module, suffix = stripped.split(".base_layer.")
            target_key = f"{module}.{suffix}"
            if suffix == "weight" and module in lora_modules:
                lora_a = state[f"base_model.model.{module}.lora_A.default.weight"].to(torch_dtype)
                lora_b = state[f"base_model.model.{module}.lora_B.default.weight"].to(torch_dtype)
                merged = tensor.to(torch_dtype) + scale * (lora_b @ lora_a)
                new_state[target_key] = merged
            else:
                new_state[target_key] = (
                tensor.to(torch_dtype) if tensor.dtype.is_floating_point else tensor
                )
        else:
            new_state[stripped] = (
                tensor.to(torch_dtype) if tensor.dtype.is_floating_point else tensor
            )
    logger(f"[4/4] Loading into {base_path} architecture and saving to {out_dir} ...")
    model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch_dtype)
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    logger(f"      Missing keys: {len(missing)} | Unexpected: {len(unexpected)}")
    if missing[:3]:
        logger(f"      (first missing): {missing[:3]}")
    if unexpected[:3]:
        logger(f"      (first unexpected): {unexpected[:3]}")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_path).save_pretrained(out_dir)
    logger(f"✅ Merged model saved to {out_dir}")
def _build_argparser() -> argparse.ArgumentParser:
    parser_doc = __doc__
    parser_doc = parser_doc.strip().split("nn")[0]
    parser = argparse.ArgumentParser(description=parser_doc.strip().split("nn")[0])
    parser.add_argument("--ckpt", required=True, help="verl LoRA .pt checkpoint")
    parser.add_argument("--base", required=True, help="Base HF model path or repo id")
    parser.add_argument("--out", required=True, help="Output dir for merged model")
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--dtype", default="bfloat16", choices=list(_DTYPE_MAP))
    return parser
def main() -> None:
    args = _build_argparser().parse_args()
    merge(args.ckpt, args.base, args.out, args.alpha, args.rank, args.dtype)
if __name__ == "__main__":  
    main()