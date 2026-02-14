#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

# ----------------------------
# model load (Qwen3)
# ----------------------------
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

def set_env_like_you():
    assert os.environ.get("HF_HOME"), "HF_HOME not set"
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

def build_inputs_qwen3(processor, image: Image.Image, text: str, device: str):
    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": image}, {"type": "text", "text": text}],
    }]
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    model_inputs = processor(text=[chat_text], images=[image], padding=True, return_tensors="pt")

    out = {}
    for k, v in dict(model_inputs).items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out

def check_trainable_via_hook(
    model: nn.Module,
    inputs: dict,
    hook_path: str,
    adapter: nn.Module,
    device: str = "cuda",
    lr: float = 1e-3,
    steps: int = 1,
    num_classes: int = 2,
):
    model = model.to(device)
    adapter = adapter.to(device)

    captured = {"x": None}

    def hook_fn(module, inp, out):
        captured["x"] = out  # detach 금지

    target = resolve_submodule(model, hook_path)
    handle = target.register_forward_hook(hook_fn)

    head = None
    opt = None

    inputs_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    model.eval()
    adapter.train()

    ok_grad = False
    ok_update = False
    last_tensor = None

    for _ in range(steps):
        captured["x"] = None

        _ = model(**inputs_dev, output_hidden_states=True, return_dict=True)

        if captured["x"] is None:
            handle.remove()
            raise RuntimeError(f"Hook did not fire. hook_path='{hook_path}'")

        x = captured["x"]

        # dict 방어
        if isinstance(x, dict):
            x = x.get("pooler_output", None) or x.get("last_hidden_state", None)
            if x is None:
                for vv in captured["x"].values():
                    if torch.is_tensor(vv):
                        x = vv
                        break

        if not torch.is_tensor(x):
            handle.remove()
            raise TypeError(f"Captured output is not a Tensor. Got: {type(x)}")

        last_tensor = x

        # ✅ 핵심 수정: dtype 맞추기
        x = x.float()          # Half -> Float (adapter/head는 기본 Float)

        z = adapter(x)

        if head is None:
            head = nn.Linear(z.shape[-1], num_classes).to(device)
            opt = torch.optim.AdamW(list(adapter.parameters()) + list(head.parameters()), lr=lr)

        logits = head(z)
        y = torch.randint(0, num_classes, (logits.shape[0],), device=device)

        loss = F.cross_entropy(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()

        def has_grad(mod):
            for p in mod.parameters():
                if p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all():
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
        "captured_dtype": str(last_tensor.dtype).replace("torch.", "") if torch.is_tensor(last_tensor) else None,
        "captured_shape": list(last_tensor.shape) if torch.is_tensor(last_tensor) else None,
        "grad_ok": bool(ok_grad),
        "update_ok": bool(ok_update),
    }

def main():
    set_env_like_you()

    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--hook_path", type=str, default="model.visual.merger.linear_fc2")
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    device = args.device
    dtype = torch.float16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=dtype, device_map=None, low_cpu_mem_usage=True
    ).to(device)

    img = make_dummy_image(224)
    text = 'Answer with ONE WORD: "normal" or "disease".\n\nThis is a medical image.'
    inputs = build_inputs_qwen3(processor, img, text, device)

    # Qwen3 merger.linear_fc2 output이 [64,4096] 이었으니 in_dim=4096
    adapter = SimpleAdapter(in_dim=4096, out_dim=512)

    rep = check_trainable_via_hook(
        model=model,
        inputs=inputs,
        hook_path=args.hook_path,
        adapter=adapter,
        device=device,
        lr=args.lr,
        steps=args.steps,
        num_classes=2,
    )

    print(json.dumps(rep, indent=2))

if __name__ == "__main__":
    import json
    main()
