#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

MODEL_ID = "google/medgemma-1.5-4b-it"

SYSTEM_PROMPT_SHORT = 'Answer with ONE WORD: "normal" or "disease".'
TASK_PROMPT = (
    "This is a medical image.\n"
    "Question: Does this image show normal anatomy or signs of disease?\n\n"
)

def set_env_like_you():
    assert os.environ.get("HF_HOME"), "HF_HOME not set (use TMPDIR/hf_cache etc.)"
    hf_home = os.environ["HF_HOME"]
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(hf_home, "hub"))

def make_dummy_image(size=224) -> Image.Image:
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:, :, 0] = 128
    arr[::8, :, 1] = 255
    arr[:, ::8, 2] = 255
    return Image.fromarray(arr, mode="RGB")

def resolve_submodule(model: nn.Module, path: str) -> nn.Module:
    cur = model
    for p in path.split("."):
        if p.isdigit():
            cur = cur[int(p)]
        else:
            if not hasattr(cur, p):
                raise AttributeError(f"Missing submodule '{p}' in path '{path}'")
            cur = getattr(cur, p)
    if not isinstance(cur, nn.Module):
        raise TypeError(f"Resolved object is not nn.Module: {type(cur)}")
    return cur

@torch.no_grad()
def _param_snapshot(params):
    return [p.detach().clone() for p in params]

class SimpleAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x)

def build_inputs_medgemma(processor, image: Image.Image, text: str, device: str):
    # medgemma placeholder requires apply_chat_template
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": text}],
    }]
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    model_inputs = processor(text=[chat_text], images=[image], padding=True, return_tensors="pt")

    out = {}
    for k, v in dict(model_inputs).items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out

def check_trainable_via_hook_medgemma(
    model: nn.Module,
    inputs: dict,
    hook_path: str,
    device: str = "cuda",
    lr: float = 1e-3,
    steps: int = 1,
    num_classes: int = 2,
    pool: str = "mean",   # mean over tokens
):
    model = model.to(device)
    model.eval()

    captured = {"x": None}

    def hook_fn(module, inp, out):
        captured["x"] = out  # detach 금지

    target = resolve_submodule(model, hook_path)
    handle = target.register_forward_hook(hook_fn)

    adapter = None
    head = None
    opt = None

    ok_grad = False
    ok_update = False
    last_x = None

    for _ in range(steps):
        captured["x"] = None

        # forward (no_grad 금지)
        _ = model(**inputs, output_hidden_states=True, return_dict=True)

        if captured["x"] is None:
            handle.remove()
            raise RuntimeError(f"Hook did not fire. hook_path='{hook_path}'")

        x = captured["x"]
        if not torch.is_tensor(x):
            handle.remove()
            raise TypeError(f"Captured output is not Tensor: {type(x)}")

        last_x = x

        # bf16 -> fp32 for adapter stability
        x = x.float()

        # x: [B, T, D] expected (you saw [1,256,2560])
        if x.dim() == 3:
            if pool == "mean":
                x2 = x.mean(dim=1)          # [B, D]
            elif pool == "cls":
                x2 = x[:, 0, :]             # [B, D]
            else:
                raise ValueError("pool must be mean or cls")
        elif x.dim() == 2:
            x2 = x
        else:
            raise RuntimeError(f"Unexpected captured dim: {x.dim()} shape={list(x.shape)}")

        if adapter is None:
            adapter = SimpleAdapter(in_dim=x2.shape[-1], out_dim=512).to(device)
            head = nn.Linear(adapter.out_dim, num_classes).to(device)
            opt = torch.optim.AdamW(list(adapter.parameters()) + list(head.parameters()), lr=lr)

        adapter.train()
        head.train()

        z = adapter(x2)
        logits = head(z)

        y = torch.randint(0, num_classes, (logits.shape[0],), device=device)
        loss = F.cross_entropy(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()

        def has_grad(mod):
            for p in mod.parameters():
                if p.grad is not None and torch.isfinite(p.grad).all():
                    return True
            return False

        ok_grad = ok_grad or has_grad(adapter) or has_grad(head)

        before = _param_snapshot(list(adapter.parameters()) + list(head.parameters()))
        opt.step()
        after = _param_snapshot(list(adapter.parameters()) + list(head.parameters()))
        delta = sum((b - a).abs().sum().item() for b, a in zip(before, after))
        ok_update = ok_update or (delta > 0)

    handle.remove()

    return {
        "hook_path": hook_path,
        "captured_dtype": str(last_x.dtype).replace("torch.", "") if torch.is_tensor(last_x) else None,
        "captured_shape": list(last_x.shape) if torch.is_tensor(last_x) else None,
        "pooled_shape": list(x2.shape),
        "grad_ok": bool(ok_grad),
        "update_ok": bool(ok_update),
    }

def load_medgemma(device: str):
    import transformers
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    AutoImageTextToText = getattr(transformers, "AutoModelForImageTextToText", None)
    if AutoImageTextToText is not None:
        model = AutoImageTextToText.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=None, low_cpu_mem_usage=True
        ).to(device).eval()
        return model, processor

    AutoVision2Seq = getattr(transformers, "AutoModelForVision2Seq", None)
    if AutoVision2Seq is not None:
        model = AutoVision2Seq.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=None, low_cpu_mem_usage=True
        ).to(device).eval()
        return model, processor

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=None, low_cpu_mem_usage=True
    ).to(device).eval()
    return model, processor

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--hook_path", type=str, default="model.multi_modal_projector")
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--pool", type=str, default="mean", choices=["mean", "cls"])
    return p.parse_args()

def main():
    set_env_like_you()
    args = parse_args()

    model, processor = load_medgemma(args.device)

    img = make_dummy_image(224)
    text = SYSTEM_PROMPT_SHORT + "\n\n" + TASK_PROMPT
    inputs = build_inputs_medgemma(processor, img, text, args.device)

    rep = check_trainable_via_hook_medgemma(
        model=model,
        inputs=inputs,
        hook_path=args.hook_path,
        device=args.device,
        lr=args.lr,
        steps=args.steps,
        pool=args.pool,
    )

    print(json.dumps(rep, indent=2))

if __name__ == "__main__":
    import json
    main()
