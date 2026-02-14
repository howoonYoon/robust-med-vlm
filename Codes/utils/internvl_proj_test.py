#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
InternVL projector: clean-weak pair -> pooled projector embeddings -> InfoNCE -> backward -> update sanity test

CSV: /SAN/ioo/HORIZON/howoon/clean_weak_1_test.csv
- must contain exactly 1 clean image path + 1 weak image path per row (or at least the first row).

This script:
1) loads InternVL
2) finds clean/weak path columns robustly
3) runs 2 forwards (clean, weak), hooks model.multi_modal_projector output
4) pools to (D,)
5) computes InfoNCE on the pair (positive = clean<->weak)
6) backward + AdamW step on projector only
7) prints a JSON report (shape/dtype/grad/update/cosine/temperature/loss)

Run:
  python internvl_infonce_pair_sanity.py \
    --model_id OpenGVLab/InternVL3_5-8B-HF \
    --csv /SAN/ioo/HORIZON/howoon/clean_weak_1_test.csv \
    --row 0 \
    --out_json /SAN/ioo/HORIZON/howoon/internvl_pair_sanity.json \
    --freeze_except_projector
"""

import os
import gc
import json
import argparse
from typing import Any, Dict, Tuple, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from transformers import AutoModelForCausalLM, AutoProcessor

os.environ.setdefault("TRANSFORMERS_USE_FAST_TOKENIZER", "0")


# -------------------------
# Helpers
# -------------------------
def get_module_by_path(root: nn.Module, path: str) -> nn.Module:
    cur: nn.Module = root
    for part in path.split("."):
        if not hasattr(cur, part):
            raise AttributeError(f"Module path not found: '{path}' (missing '{part}')")
        cur = getattr(cur, part)
    return cur


def tensor_stats(x: torch.Tensor) -> Dict[str, Any]:
    xd = x.detach()
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype).replace("torch.", ""),
        "device": str(x.device),
        "min": float(xd.min().cpu()),
        "max": float(xd.max().cpu()),
        "nan": int(torch.isnan(xd).sum().cpu()),
        "inf": int(torch.isinf(xd).sum().cpu()),
        "requires_grad": bool(x.requires_grad),
    }


def clone_params(module: nn.Module) -> Dict[str, torch.Tensor]:
    out = {}
    for n, p in module.named_parameters():
        if p.requires_grad:
            out[n] = p.detach().clone().cpu()
    return out


def params_changed(before: Dict[str, torch.Tensor], module: nn.Module, atol=0.0) -> bool:
    for n, p in module.named_parameters():
        if not p.requires_grad:
            continue
        if n not in before:
            continue
        after = p.detach().cpu()
        if not torch.allclose(before[n], after, atol=atol, rtol=0.0):
            return True
    return False


def find_clean_weak_paths(row: pd.Series) -> Tuple[str, str]:
    """
    Robustly locate clean/weak path columns.
    Accepts common variants:
      clean:  clean, clean_path, path_clean, img_clean, clean_image, clean_img
      weak:   weak, weak_path, path_weak, artifact, artefact, artifact_path, weak_image, weak_img
    If not found, fallback: first two columns as (clean, weak).
    """
    cols = {c.lower(): c for c in row.index}

    clean_keys = [
        "clean", "clean_path", "path_clean", "img_clean", "clean_image", "clean_img", "cleanfile", "clean_file"
    ]
    weak_keys = [
        "weak", "weak_path", "path_weak", "artifact", "artefact", "artifact_path", "artefact_path",
        "weak_image", "weak_img", "weakfile", "weak_file"
    ]

    clean_col = next((cols[k] for k in clean_keys if k in cols), None)
    weak_col = next((cols[k] for k in weak_keys if k in cols), None)

    # Another common pattern: columns contain substrings
    if clean_col is None:
        for c in row.index:
            cl = str(c).lower()
            if "clean" in cl and "path" in cl:
                clean_col = c
                break
        if clean_col is None:
            for c in row.index:
                if "clean" in str(c).lower():
                    clean_col = c
                    break

    if weak_col is None:
        for c in row.index:
            cl = str(c).lower()
            if (("weak" in cl) or ("artifact" in cl) or ("artefact" in cl)) and ("path" in cl):
                weak_col = c
                break
        if weak_col is None:
            for c in row.index:
                cl = str(c).lower()
                if ("weak" in cl) or ("artifact" in cl) or ("artefact" in cl):
                    weak_col = c
                    break

    if clean_col is None or weak_col is None:
        # fallback: first two columns
        values = list(row.values)
        if len(values) < 2:
            raise ValueError("CSV row must contain at least 2 columns (clean path, weak path).")
        clean_path = str(values[0])
        weak_path = str(values[1])
        return clean_path, weak_path

    return str(row[clean_col]), str(row[weak_col])


def load_rgb(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def build_inputs(processor: AutoProcessor, image: Image.Image, device: str) -> Dict[str, torch.Tensor]:
    # keep prompt simple; projector still runs when an image is present
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": 'Answer with ONE WORD: "normal" or "disease".'}
        ]}
    ]
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = 'Answer with ONE WORD: "normal" or "disease".'

    inputs = processor(images=image, text=text, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}


def forward_capture_projector(
    model: nn.Module,
    hook_module: nn.Module,
    inputs: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Run forward once and capture projector output tensor via a forward hook.
    Returns:
      z_tokens: (B, T, D) tensor
      meta: {hook_fired, captured_stats}
    """
    captured = {"fired": 0, "tensor": None}

    def hook_fn(_m, _inp, out):
        captured["fired"] += 1
        if isinstance(out, torch.Tensor):
            captured["tensor"] = out
        elif isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
            captured["tensor"] = out[0]

    h = hook_module.register_forward_hook(hook_fn)
    out = model(**inputs)  # logits not used; we just want projector activation
    _ = out  # keep for clarity
    h.remove()

    if captured["tensor"] is None:
        raise RuntimeError("Projector hook did not capture a tensor output. Check hook_path.")

    meta = {
        "hook_fired": int(captured["fired"]),
        "captured": tensor_stats(captured["tensor"]),
    }
    return captured["tensor"], meta


def info_nce_pair(zc: torch.Tensor, zw: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    zc, zw: (D,) or (1,D)
    We build a 2x2 similarity matrix for [zc, zw] vs [zc, zw]
    Positive pairs are diagonal swapped: clean->weak and weak->clean
    Loss = (CE row0 targets=1) + (CE row1 targets=0) / 2
    """
    if zc.dim() == 1:
        zc = zc.unsqueeze(0)
    if zw.dim() == 1:
        zw = zw.unsqueeze(0)

    # normalize
    zc = F.normalize(zc, dim=-1)
    zw = F.normalize(zw, dim=-1)

    Z = torch.cat([zc, zw], dim=0)  # (2, D)
    logits = (Z @ Z.t()) / temperature  # (2,2)

    # remove self-sim? for 2-sample case, keep it but target other sample
    targets = torch.tensor([1, 0], device=logits.device, dtype=torch.long)
    loss = F.cross_entropy(logits, targets)
    return loss


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="OpenGVLab/InternVL3_5-8B-HF")
    ap.add_argument("--csv", type=str, default="/SAN/ioo/HORIZON/howoon/clean_weak_1_test.csv")
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--hook_path", type=str, default="model.multi_modal_projector")
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--freeze_except_projector", action="store_true")
    ap.add_argument("--out_json", type=str, default="")
    ap.add_argument("--path_from", type=str, default=r"C:\Users\hanna\Lectures\Research_Project\Codes\Dataset\vlm_prompt_dataset")
    ap.add_argument("--path_to", type=str, default="/SAN/ioo/HORIZON/howoon")

    args = ap.parse_args()

    torch.manual_seed(0)
    device = args.device

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # Load CSV
    df = pd.read_csv(args.csv)
    if len(df) == 0:
        raise ValueError("CSV is empty.")

    # ---- build (clean_path, weak_path) pairs from row-based CSV ----
    need = {"fileindex", "severity", "filepath"}
    colmap = {c.lower(): c for c in df.columns}
    if not need.issubset(set(colmap.keys())):
        raise ValueError(f"CSV must contain columns {need}, got: {list(df.columns)}")

    c_fileindex = colmap["fileindex"]
    c_severity  = colmap["severity"]
    c_filepath  = colmap["filepath"]

    pairs = []
    for fx, g in df.groupby(c_fileindex):
        sev = g[c_severity].astype(str).str.lower()
        g_clean = g[sev.eq("clean")]
        g_weak  = g[~sev.eq("clean")]

        if len(g_clean) == 0 or len(g_weak) == 0:
            continue

        clean_path = str(g_clean.iloc[0][c_filepath])
        weak_path  = str(g_weak.iloc[0][c_filepath])
        pairs.append((str(fx), clean_path, weak_path))

    if len(pairs) == 0:
        raise ValueError("No (clean, weak) pairs found. Check severity/fileindex/filepath columns.")

    if args.row < 0 or args.row >= len(pairs):
        raise IndexError(f"--row {args.row} out of range. Found {len(pairs)} pairs.")

    fileindex, clean_path, weak_path = pairs[args.row]

    def map_path(p: str, path_from: str, path_to: str) -> str:
        p = str(p).replace("\\", "/")
        pf = str(path_from).replace("\\", "/")
        if p.lower().startswith(pf.lower()):
            p = path_to.rstrip("/") + p[len(pf):]
        return p

    clean_path = map_path(clean_path, args.path_from, args.path_to)
    weak_path  = map_path(weak_path,  args.path_from, args.path_to)


    img_c = load_rgb(clean_path)
    img_w = load_rgb(weak_path)

    # Load model + processor
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=None,
    ).to(device)

    model.train()

    # Optionally freeze everything except projector
    if args.freeze_except_projector:
        for p in model.parameters():
            p.requires_grad_(False)

    hook_module = get_module_by_path(model, args.hook_path)
    for p in hook_module.parameters():
        p.requires_grad_(True)

    # Optimizer on projector only
    opt = torch.optim.AdamW([p for p in hook_module.parameters() if p.requires_grad], lr=args.lr)

    # Build inputs
    inputs_c = build_inputs(processor, img_c, device)
    inputs_w = build_inputs(processor, img_w, device)

    # Capture projector tokens for clean/weak
    zc_tokens, meta_c = forward_capture_projector(model, hook_module, inputs_c)
    zw_tokens, meta_w = forward_capture_projector(model, hook_module, inputs_w)

    # Pool to (D,)
    zc = zc_tokens.mean(dim=1).squeeze(0)  # (D,)
    zw = zw_tokens.mean(dim=1).squeeze(0)  # (D,)

    # InfoNCE
    loss = info_nce_pair(zc, zw, temperature=args.temperature)

    # Before-update snapshot
    before = clone_params(hook_module)

    opt.zero_grad(set_to_none=True)
    loss.backward()

    # grad stats
    grad_ok = False
    max_g = 0.0
    for p in hook_module.parameters():
        if p.grad is None:
            continue
        g = float(p.grad.detach().abs().max().cpu())
        max_g = max(max_g, g)
        if g > 0:
            grad_ok = True

    opt.step()
    update_ok = params_changed(before, hook_module)

    # cosine similarity (normalized)
    cos = float(F.cosine_similarity(F.normalize(zc, dim=0), F.normalize(zw, dim=0), dim=0).detach().cpu())

    report: Dict[str, Any] = {
        "model_id": args.model_id,
        "csv": args.csv,
        "row": int(args.row),
        "clean_path": clean_path,
        "weak_path": weak_path,
        "hook_path": args.hook_path,
        "clean_capture": meta_c,
        "weak_capture": meta_w,
        "pooled_shape_clean": list(zc.shape),
        "pooled_shape_weak": list(zw.shape),
        "temperature": float(args.temperature),
        "loss_infonce": float(loss.detach().cpu()),
        "cosine_clean_weak": cos,
        "grad_ok": bool(grad_ok),
        "max_abs_grad": float(max_g),
        "update_ok": bool(update_ok),
    }

    print(json.dumps(report, indent=2))

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[saved] {args.out_json}")

    # cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
