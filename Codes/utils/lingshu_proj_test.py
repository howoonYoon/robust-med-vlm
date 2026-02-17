#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lingshu (Qwen2.5-VL-like) projector/connector sanity train:
- find connector module automatically
- hook its output (tokens x hidden)
- pool -> (D,)
- InfoNCE clean<->weak
- backward + AdamW step on connector only
- report JSON: hook fired, captured stats, grad_ok, update_ok

Run:
  python lingshu_proj_sanity.py \
    --model_id <YOUR_LINGSHU_MODEL_ID> \
    --csv /SAN/ioo/HORIZON/howoon/clean_weak_1_test.csv \
    --row 0 \
    --out_json /SAN/ioo/HORIZON/howoon/lingshu_pair_sanity.json \
    --freeze_except_projector

Notes:
- This script assumes your CSV has fileindex/severity/filepath like you used before.
- It auto-detects connector/projector by running one forward and looking for a tensor output
  whose last dim == language hidden_size (e.g., 3584/4096/etc).
"""

import os
import gc
import json
import argparse
from typing import Any, Dict, Tuple, Optional, List

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor



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


def load_rgb(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def build_inputs(processor: AutoProcessor, image: Image.Image, device: str) -> Dict[str, torch.Tensor]:
    # minimal prompt; image must be included
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


def info_nce_pair(zc: torch.Tensor, zw: torch.Tensor, temperature: float) -> torch.Tensor:
    if zc.dim() == 1:
        zc = zc.unsqueeze(0)
    if zw.dim() == 1:
        zw = zw.unsqueeze(0)

    zc = F.normalize(zc, dim=-1)
    zw = F.normalize(zw, dim=-1)

    Z = torch.cat([zc, zw], dim=0)          # (2, D)
    logits = (Z @ Z.t()) / temperature      # (2, 2)
    targets = torch.tensor([1, 0], device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits, targets)


# -------------------------
# Auto-detect "projector/connector" for Qwen2.5-VL-like models
# -------------------------
def get_language_hidden_size(model: nn.Module) -> Optional[int]:
    # try a few common locations
    for attr_path in [
        "config.hidden_size",
        "model.config.hidden_size",
        "language_model.config.hidden_size",
        "model.model.config.hidden_size",
    ]:
        cur = model
        ok = True
        for part in attr_path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and isinstance(cur, int):
            return cur
    return None


def auto_find_connector(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    hidden_size: int,
    max_candidates: int = 20,
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Runs one forward with many lightweight hooks and tries to find a module that:
      - outputs a Tensor (or tuple/list containing Tensor)
      - last dimension == hidden_size (LLM hidden)
      - looks like a connector/projector based on name heuristics OR output shape
    Returns:
      (best_path or None, candidate_summaries)
    """

    # name heuristics to prioritize
    name_keywords = [
        "projector", "mm_projector", "multi_modal", "multimodal",
        "connector", "vision_proj", "visual_proj", "vision_projection",
        "resampler", "merger", "adapter", "bridge"
    ]

    records: List[Dict[str, Any]] = []
    hooks = []

    def make_hook(name: str):
        def _hook(_m, _inp, out):
            t = None
            if isinstance(out, torch.Tensor):
                t = out
            elif isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                t = out[0]
            elif isinstance(out, dict):
                # some modules output dicts; try common fields
                for k in ["last_hidden_state", "hidden_states", "pooler_output", "image_embeds", "embeds"]:
                    if k in out and isinstance(out[k], torch.Tensor):
                        t = out[k]
                        break
            if t is None:
                return

            # only keep reasonably small tensors (avoid huge intermediates)
            # but allow (B,T,D), (T,D), (B,D)
            if t.dim() >= 2 and t.shape[-1] == hidden_size:
                rec = {
                    "name": name,
                    "shape": list(t.shape),
                    "dtype": str(t.dtype).replace("torch.", ""),
                    "device": str(t.device),
                    "min": float(t.detach().min().cpu()),
                    "max": float(t.detach().max().cpu()),
                }
                records.append(rec)
        return _hook

    # register hooks on all modules, but skip extremely deep submodules to reduce overhead
    for name, mod in model.named_modules():
        if name == "":
            continue
        # skip norms/activations to reduce spam
        low = name.lower()
        if any(x in low for x in ["layernorm", "rmsnorm", ".norm", "rotary", "dropout", "act_fn"]):
            continue
        try:
            hooks.append(mod.register_forward_hook(make_hook(name)))
        except Exception:
            pass

    # run one forward
    with torch.no_grad():
        _ = model(**inputs)

    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    # prioritize candidates
    def score(rec: Dict[str, Any]) -> int:
        n = rec["name"].lower()
        s = 0
        if any(k in n for k in name_keywords):
            s += 100
        # connector often outputs (B, T, hidden) or (T, hidden)
        sh = rec["shape"]
        if len(sh) == 3:
            s += 20
        if len(sh) == 2:
            s += 10
        # prefer not-too-early vision blocks
        if "visual" in n or "vision" in n:
            s -= 10  # connector is usually after vision encoder
        if "lm_head" in n:
            s -= 50
        return s

    records_sorted = sorted(records, key=score, reverse=True)

    # deduplicate by name keep first
    seen = set()
    candidates = []
    for r in records_sorted:
        if r["name"] in seen:
            continue
        seen.add(r["name"])
        candidates.append(r)
        if len(candidates) >= max_candidates:
            break

    best = candidates[0]["name"] if len(candidates) > 0 else None
    return best, candidates


def forward_capture(
    model: nn.Module,
    hook_module: nn.Module,
    inputs: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    captured = {"fired": 0, "tensor": None}

    def hook_fn(_m, _inp, out):
        captured["fired"] += 1
        t = None
        if isinstance(out, torch.Tensor):
            t = out
        elif isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
            t = out[0]
        elif isinstance(out, dict):
            for k in ["last_hidden_state", "hidden_states", "pooler_output", "image_embeds", "embeds"]:
                if k in out and isinstance(out[k], torch.Tensor):
                    t = out[k]
                    break
        if t is not None:
            captured["tensor"] = t

    h = hook_module.register_forward_hook(hook_fn)
    out = model(**inputs)
    _ = out
    h.remove()

    if captured["tensor"] is None:
        raise RuntimeError("Hook did not capture a tensor output. Wrong hook module/path.")

    meta = {"hook_fired": int(captured["fired"]), "captured": tensor_stats(captured["tensor"])}
    return captured["tensor"], meta


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="lingshu-medical-mllm/Lingshu-7B")
    ap.add_argument("--csv", type=str, default="/SAN/ioo/HORIZON/howoon/clean_weak_1_test.csv")
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--hook_path", type=str, default="")  # optional manual override
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--freeze_except_projector", action="store_true")
    ap.add_argument("--out_json", type=str, default="")
    ap.add_argument("--path_from", type=str, default=r"C:\Users\hanna\Lectures\Research_Project\Codes\Dataset\vlm_prompt_dataset")
    ap.add_argument("--path_to", type=str, default="/SAN/ioo/HORIZON/howoon")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = args.device
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # Load CSV (same convention as your InternVL script)
    df = pd.read_csv(args.csv)
    if len(df) == 0:
        raise ValueError("CSV is empty.")

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

    model = AutoModel.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        dtype=dtype,          # torch_dtype 말고 dtype
        device_map=None,
    ).to(device)


    model.train()

    # Build inputs once (for auto-detect)
    inputs_probe = build_inputs(processor, img_c, device)

    hidden_size = get_language_hidden_size(model)
    if hidden_size is None:
        # fallback: try common config keys
        hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    if hidden_size is None:
        raise RuntimeError("Could not infer language hidden_size from model.config. Please set --hook_path manually.")

    # Freeze all (optional)
    if args.freeze_except_projector:
        for p in model.parameters():
            p.requires_grad_(False)

    # Choose hook path: user override or auto
    chosen_path = args.hook_path.strip()
    candidates = []
    if chosen_path == "":
        chosen_path, candidates = auto_find_connector(model, inputs_probe, hidden_size=hidden_size, max_candidates=30)
        if chosen_path is None:
            raise RuntimeError(
                "Failed to auto-detect connector/projector. "
                "Re-run with --hook_path <module.path>. "
                "Tip: print candidates by temporarily disabling this raise."
            )

    hook_module = get_module_by_path(model, chosen_path)

    # Enable grads only on hook_module
    for p in hook_module.parameters():
        p.requires_grad_(True)

    opt = torch.optim.AdamW([p for p in hook_module.parameters() if p.requires_grad], lr=args.lr)

    # Build inputs for clean/weak
    inputs_c = build_inputs(processor, img_c, device)
    inputs_w = build_inputs(processor, img_w, device)

    # Capture connector outputs
    zc_tokens, meta_c = forward_capture(model, hook_module, inputs_c)
    zw_tokens, meta_w = forward_capture(model, hook_module, inputs_w)

    # Pool to (D,)
    # Support shapes: (B,T,D), (T,D), (B,D)
    def pool_to_vec(z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 3:
            # (B,T,D)
            return z.mean(dim=1).squeeze(0)
        if z.dim() == 2:
            # (T,D)
            return z.mean(dim=0)
        if z.dim() == 1:
            return z
        # last resort: flatten then mean
        return z.view(-1, z.shape[-1]).mean(dim=0)

    zc = pool_to_vec(zc_tokens)
    zw = pool_to_vec(zw_tokens)

    loss = info_nce_pair(zc, zw, temperature=args.temperature)

    before = clone_params(hook_module)

    opt.zero_grad(set_to_none=True)
    loss.backward()

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

    cos = float(F.cosine_similarity(F.normalize(zc, dim=0), F.normalize(zw, dim=0), dim=0).detach().cpu())

    report: Dict[str, Any] = {
        "model_id": args.model_id,
        "csv": args.csv,
        "row": int(args.row),
        "fileindex": fileindex,
        "clean_path": clean_path,
        "weak_path": weak_path,
        "language_hidden_size": int(hidden_size),
        "hook_path": chosen_path,
        "auto_candidates_top": candidates[:10] if candidates else [],
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

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
