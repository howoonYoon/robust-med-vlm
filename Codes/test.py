#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vision-encoder probe / smoke-test script (NO recursion, NO base_model attribute probing).

What it does
- Loads each backend model + processor
- Loads one sample image from your TRAIN_CSV
- Lists likely vision-related submodules using model.named_modules() only
- Chooses top candidate module paths
- Runs a smoke test:
    processor(images)->pixel_values -> candidate_module(pixel_values)
  and prints output shape/dtype/device.

Run:
  python probe_vision_encoder_no_recursion.py

Notes
- InternVL processor error fix included (force slow tokenizer use_fast=False).
- This script is for debugging only; it does NOT train anything.
"""

import os
import gc
from typing import Any, List, Tuple

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn


# =========================
# Config
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

csv_path = "/SAN/ioo/HORIZON/howoon"
TRAIN_CSV = os.path.join(csv_path, "vlm_clean_train_2520.csv")
base_out_dir = "/SAN/ioo/HORIZON/howoon"

MODEL_ID_BY_BACKEND = {
    "qwen3":    "Qwen/Qwen3-VL-8B-Instruct",
    "medgemma": "google/medgemma-1.5-4b-it",
    "internvl": "OpenGVLab/InternVL3_5-8B",
    "lingshu":  "lingshu-medical-mllm/Lingshu-7B",
}

BACKENDS = ["qwen3", "medgemma", "internvl", "lingshu"]

MAX_SUSPECTS_PRINT = 80
MAX_CANDIDATES = 12
MAX_SMOKE_TEST = 5


# =========================
# Load one image from CSV
# =========================
def load_split_csv(path: str, base_out_dir: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "binarylab" in df.columns and "binarylabel" not in df.columns:
        df = df.rename(columns={"binarylab": "binarylabel"})

    df["severity_norm"] = df["severity"].astype(str).str.lower()
    df["dataset_norm"] = df["dataset"].astype(str).str.lower()

    # clean-only
    df = df[df["severity_norm"].isin(["clean"])].copy()

    # windows path -> linux path
    df["filepath"] = (
        df["filepath"]
        .astype(str)
        .str.replace(
            r"C:\Users\hanna\Lectures\Research_Project\Codes\Dataset\vlm_prompt_dataset",
            base_out_dir,
            regex=False,
        )
        .str.replace("\\", "/", regex=False)
    )

    if "fileindex" not in df.columns:
        raise ValueError(f"{path} 에 fileindex 컬럼 필요함")

    return df.reset_index(drop=True)


def pick_one_image(df: pd.DataFrame) -> str:
    for p in df["filepath"].tolist():
        if isinstance(p, str) and os.path.exists(p):
            return p
    raise FileNotFoundError("CSV 안에서 존재하는 filepath를 하나도 못 찾았어.")


# =========================
# Backend loader (internvl fix included)
# =========================
def load_backend(backend: str, model_id: str):
    import transformers
    from transformers import AutoProcessor

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    if backend == "medgemma" and device == "cuda":
        torch_dtype = torch.bfloat16

    common = dict(
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )

    if backend == "qwen3":
        from transformers import Qwen3VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_id)
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **common)
        model.eval()
        return model, processor

    if backend == "lingshu":
        from transformers import Qwen2_5_VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_id)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **common)
        model.eval()
        return model, processor

    if backend == "internvl":
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        # ✅ 핵심: start_image_token 등 커스텀 속성 때문에 slow tokenizer 강제
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_fast=False
        )

        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        if hasattr(processor, "tokenizer"):
            processor.tokenizer = tokenizer

        model = AutoModel.from_pretrained(model_id, trust_remote_code=True, **common)
        model.eval()
        return model, processor

    if backend == "medgemma":
        processor = AutoProcessor.from_pretrained(model_id)

        AutoImageTextToText = getattr(transformers, "AutoModelForImageTextToText", None)
        if AutoImageTextToText is not None:
            model = AutoImageTextToText.from_pretrained(model_id, **common)
            model.eval()
            return model, processor

        AutoVision2Seq = getattr(transformers, "AutoModelForVision2Seq", None)
        if AutoVision2Seq is not None:
            model = AutoVision2Seq.from_pretrained(model_id, **common)
            model.eval()
            return model, processor

        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_id, **common)
        model.eval()
        return model, processor

    raise ValueError(f"Unknown backend: {backend}")


# =========================
# Module path helpers
# =========================
def get_module_by_path(model: nn.Module, path: str) -> nn.Module:
    cur: Any = model
    for part in path.split("."):
        cur = getattr(cur, part)
    if not isinstance(cur, nn.Module):
        raise TypeError(f"Path {path} did not resolve to nn.Module (got {type(cur)})")
    return cur


def list_vision_suspects_named_modules(model: nn.Module) -> List[Tuple[str, str]]:
    """
    Safe: uses only named_modules() (no recursion into HF base_model properties).
    """
    suspects: List[Tuple[str, str]] = []
    for name, mod in model.named_modules():
        low = name.lower()
        if any(x in low for x in ["vision", "visual", "vit", "clip", "siglip", "image", "tower", "encoder"]):
            suspects.append((name, type(mod).__name__))

    print(f"Total suspects: {len(suspects)}")
    for i, (n, t) in enumerate(suspects[:MAX_SUSPECTS_PRINT]):
        print(f"{i:02d}  {n:70s}  {t}")
    if len(suspects) > MAX_SUSPECTS_PRINT:
        print(f"... (showing first {MAX_SUSPECTS_PRINT})")

    return suspects


def pick_candidate_modules_from_suspects(suspects: List[Tuple[str, str]], max_candidates: int = 12) -> List[str]:
    """
    Choose likely vision encoder module paths from suspects, with priority rules.
    """
    priority_keys = ["vision_model", "vision_tower", "vision_encoder", "visual_encoder", "image_encoder"]
    hard_exclude = ["embed_tokens", "lm_head", "language_model", "text_model", "token"]

    priority: List[str] = []
    rest: List[str] = []

    for name, _ in suspects:
        low = name.lower()
        if any(x in low for x in hard_exclude):
            continue
        if any(pk in low for pk in priority_keys):
            priority.append(name)
        else:
            rest.append(name)

    out: List[str] = []
    seen = set()
    for n in priority + rest:
        if n not in seen:
            seen.add(n)
            out.append(n)
        if len(out) >= max_candidates:
            break

    print("\nCandidates to smoke-test:")
    if not out:
        print("  (none)")
    for i, n in enumerate(out):
        print(f"  [{i}] {n}")
    return out


# =========================
# Smoke test
# =========================
@torch.inference_mode()
def smoke_test_module_forward(base_model: nn.Module, processor, image_pil: Image.Image, module_path: str):
    inputs = processor(images=[image_pil], padding=True, return_tensors="pt")
    pv = inputs.get("pixel_values", None)
    if pv is None:
        raise RuntimeError("processor(images=...) did not return pixel_values")

    mod = get_module_by_path(base_model, module_path)

    # move pixel_values to module's device
    try:
        dev = next(mod.parameters()).device
    except StopIteration:
        # module with no parameters; fallback to model device
        dev = next(base_model.parameters()).device
    pv = pv.to(dev)

    # forward
    try:
        out = mod(pixel_values=pv)
    except TypeError:
        out = mod(pv)

    # normalize output -> tensor
    if isinstance(out, torch.Tensor):
        feat = out
    elif hasattr(out, "last_hidden_state") and isinstance(out.last_hidden_state, torch.Tensor):
        feat = out.last_hidden_state
    elif isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
        feat = out[0]
    else:
        raise RuntimeError(f"Unexpected output type: {type(out)}")

    shape = tuple(feat.shape)
    print(f"✅ SMOKE OK  | module={module_path} | out_shape={shape} | dtype={feat.dtype} | device={feat.device}")

    # pooled hint
    if feat.dim() == 3:
        pooled = feat.mean(dim=1)
        print(f"   pooled(B,D) shape = {tuple(pooled.shape)}")
    elif feat.dim() == 2:
        print("   already pooled (B,D)")
    else:
        print(f"   WARNING: unexpected dim={feat.dim()}")

    # NaN/Inf check
    if torch.isnan(feat).any() or torch.isinf(feat).any():
        print("   ⚠️ WARNING: NaN/Inf detected in vision output")

    return feat


# =========================
# Main
# =========================
def main():
    df = load_split_csv(TRAIN_CSV, base_out_dir)
    img_path = pick_one_image(df)
    img = Image.open(img_path).convert("RGB")
    print("Using image:", img_path)

    for backend in BACKENDS:
        print("\n" + "=" * 70)
        print(f"[{backend}] loading...")
        model_id = MODEL_ID_BY_BACKEND[backend]

        try:
            base_model, processor = load_backend(backend, model_id)
        except Exception as e:
            print(f"❌ LOAD FAILED [{backend}] | err={repr(e)}")
            continue

        print(f"[{backend}] loaded model_id={model_id}")
        print(f"[{backend}] listing suspected vision submodules (named_modules only)...")
        suspects = list_vision_suspects_named_modules(base_model)

        candidates = pick_candidate_modules_from_suspects(suspects, max_candidates=MAX_CANDIDATES)

        if not candidates:
            print(f"❌ [{backend}] no candidate module paths found to smoke-test.")
        else:
            print(f"\n[{backend}] smoke testing candidates (up to {min(MAX_SMOKE_TEST, len(candidates))})...")
            tested = 0
            for path in candidates[:MAX_SMOKE_TEST]:
                try:
                    _ = smoke_test_module_forward(base_model, processor, img, path)
                    tested += 1
                except Exception as e:
                    print(f"❌ SMOKE FAIL | module={path} | err={repr(e)}")

            if tested == 0:
                print(f"⚠️ [{backend}] all smoke tests failed.")
                print("   -> This usually means the vision encoder isn't directly callable with pixel_values,")
                print("      so you need backend-specific extraction (full forward + image_embeds/vision_features).")

        # cleanup
        del base_model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
