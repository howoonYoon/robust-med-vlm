#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, gc, json, re, argparse
from typing import Optional, Any, Dict, List
import numpy as np
import pandas as pd
from PIL import Image

import torch

# -------------------------
# Args
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", type=str, required=True, choices=["qwen3", "internvl", "lingshu","medgemma"])
    p.add_argument("--demo_csv", type=str, required=True, help="few_shot_examples.csv")

    # IMPORTANT:
    # - query_clean_csv: standalone clean set (NOT paired with anything)
    # - query_weak_csv : pair-set CSV (contains clean+weak entries per fileindex)
    p.add_argument("--query_clean_csv", type=str, required=True, help="standalone clean set csv (e.g., vlm_clean.csv)")
    p.add_argument("--query_weak_csv", type=str, required=True, help="pair-set csv (e.g., clean_weak.csv: clean 1 + weak N per fileindex)")

    p.add_argument("--max_new_tokens", type=int, default=2)
    p.add_argument("--out_json", type=str, required=True)
    p.add_argument("--parse_mode", type=str, default="strict", choices=["strict", "relaxed"])
    return p.parse_args()

args = parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device, flush=True)
DBG_ONCE = {"medgemma": False, "internvl": False}


# HF cache
assert os.environ.get("HF_HOME"), "HF_HOME not set"
hf_home = os.environ["HF_HOME"]
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(hf_home, "hub"))

BASE_OUT_DIR = "/SAN/ioo/HORIZON/howoon"

PROMPT_BY_DATASET = {
    "mri":    "This is a brain MRI scan.\nQuestion: Does this image show normal anatomy or signs of disease?\n\n",
    "oct":    "This is a retinal OCT scan.\nQuestion: Does this image show normal anatomy or signs of disease?\n\n",
    "xray":   "This is a chest X-ray image.\nQuestion: Does this image show normal anatomy or signs of disease?\n\n",
    "fundus": "This is a retinal fundus photograph.\nQuestion: Does this image show normal anatomy or signs of disease?\n\n",
}
SYSTEM_PROMPT_SHORT = (
    "Answer with ONE WORD only: normal or disease.\n"
    "Rules:\n"
    "- Output must be exactly: normal OR disease\n"
    "- Do not add punctuation, explanations, or extra words.\n"
)

MODEL_ID_BY_BACKEND = {
    "qwen3":    "Qwen/Qwen3-VL-8B-Instruct",
    "internvl": "OpenGVLab/InternVL3_5-8B-HF",
    "lingshu":  "lingshu-medical-mllm/Lingshu-7B",
    "medgemma": "google/medgemma-1.5-4b-it",
}

# -------------------------
# CSV load
# -------------------------
def _stable_fileindex_from_path(p: str) -> str:
    b = os.path.basename(str(p)).replace("\\", "/")
    stem, _ = os.path.splitext(b)
    return stem

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "binarylab" in df.columns and "binarylabel" not in df.columns:
        df = df.rename(columns={"binarylab": "binarylabel"})

    # if severity missing, default clean (but NOTE: pair csv SHOULD have severity to separate clean/weak)
    if "severity" not in df.columns:
        df["severity"] = "clean"

    # path normalize
    if "filepath" in df.columns:
        df["filepath"] = (
            df["filepath"].astype(str)
            .str.replace(
                r"C:\Users\hanna\Lectures\Research_Project\Codes\Dataset\vlm_prompt_dataset",
                BASE_OUT_DIR,
                regex=False,
            )
            .str.replace("\\", "/", regex=False)
        )
    else:
        raise RuntimeError(f"'filepath' column missing in {path}")

    # fileindex
    if "fileindex" not in df.columns:
        df["fileindex"] = df["filepath"].map(_stable_fileindex_from_path)

    df["severity_norm"] = df["severity"].fillna("clean").astype(str).str.lower()
    if "dataset" not in df.columns:
        raise RuntimeError(f"'dataset' column missing in {path}")
    df["dataset_norm"] = df["dataset"].astype(str).str.lower()
    return df.reset_index(drop=True)

# -------------------------
# text -> label
# -------------------------
def _normalize_answer_text(s: str) -> str:
    s = (s or "").strip().lower()
    for ch in ["\"", "'", ".", ",", "!", "?", ":", ";", "\n", "\t", "(", ")", "[", "]", "{", "}"]:
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s

def _strip_code_fence(t: str) -> str:
    t = (t or "").strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if len(lines) >= 2 and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if len(lines) >= 1 and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t

def _text_to_label_strict(s: str) -> Optional[int]:
    """
    STRICT v2:
    - allow small prefixes like "Answer:", "Output:", etc.
    - then first meaningful token must be normal/disease
    """
    t = _strip_code_fence(s or "").strip()
    s_norm = _normalize_answer_text(t)
    toks = s_norm.split()
    if not toks:
        return None

    allowed_prefix = {"output", "answer", "label", "prediction", "pred", "result"}
    i = 0
    while i < len(toks) and toks[i] in allowed_prefix:
        i += 1

    first = toks[i] if i < len(toks) else ""
    if first == "disease":
        return 1
    if first == "normal":
        return 0
    return None


def _text_to_label_relaxed(s: str) -> Optional[int]:
    # currently same behavior; add heuristics if needed
    return _text_to_label_strict(s)

def text_to_label(s: str, mode: str) -> Optional[int]:
    return _text_to_label_strict(s) if mode == "strict" else _text_to_label_relaxed(s)

# -------------------------
# Backend loader
# -------------------------
def load_backend(backend: str, model_id: str):
    from transformers import AutoProcessor

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    if backend in ["internvl", "medgemma"] and device == "cuda":
        torch_dtype = torch.bfloat16  # 둘 다 bf16이 더 안전한 경우 많음

    cache_dir = os.environ.get("TRANSFORMERS_CACHE") or os.environ.get("HF_HOME") or None
    common = dict(torch_dtype=torch_dtype, device_map=None, low_cpu_mem_usage=True)

    if backend == "qwen3":
        from transformers import Qwen3VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, cache_dir=cache_dir, **common)
        return model.to(device).eval(), processor

    if backend == "lingshu":
        from transformers import Qwen2_5_VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, cache_dir=cache_dir, **common)
        return model.to(device).eval(), processor

    if backend == "internvl":
        from transformers import AutoModelForImageTextToText
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, cache_dir=cache_dir)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, trust_remote_code=True, cache_dir=cache_dir,
            torch_dtype=torch_dtype, low_cpu_mem_usage=True, device_map=None,
        )
        return model.to(device).eval(), processor

    if backend == "medgemma":
        # ✅ MedGemma는 환경에 따라 모델 클래스가 다를 수 있어서 try/fallback이 안전
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, cache_dir=cache_dir)

        model = None
        last_err = None

        # 1) 가장 흔한 멀티모달 text-generation 계열
        try:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(
                model_id, trust_remote_code=True, cache_dir=cache_dir,
                torch_dtype=torch_dtype, low_cpu_mem_usage=True, device_map=None,
            )
        except Exception as e:
            last_err = e

        # 2) Vision2Seq 계열로 들어오는 경우
        if model is None:
            try:
                from transformers import AutoModelForVision2Seq
                model = AutoModelForVision2Seq.from_pretrained(
                    model_id, trust_remote_code=True, cache_dir=cache_dir,
                    torch_dtype=torch_dtype, low_cpu_mem_usage=True, device_map=None,
                )
                last_err = None
            except Exception as e:
                last_err = e

        # 3) 마지막 fallback (그래도 안되면 에러를 명확히)
        if model is None:
            raise RuntimeError(f"Failed to load medgemma model. Last error: {repr(last_err)}")

        return model.to(device).eval(), processor

    raise ValueError(backend)

# -------------------------
# Few-shot demo builder (모달리티별 고정)
# -------------------------
def build_demos_by_modality(df_demo: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    demos_by_mod = {}
    for mod, g in df_demo.groupby("dataset_norm"):
        demos = []
        for _, row in g.iterrows():
            img = Image.open(row["filepath"]).convert("RGB")
            y = int(row["binarylabel"])
            ans = "disease" if y == 1 else "normal"
            prompt = SYSTEM_PROMPT_SHORT + "\n" + PROMPT_BY_DATASET.get(mod, PROMPT_BY_DATASET["mri"])
            demos.append({"image": img, "text": prompt, "answer": ans})
        demos_by_mod[str(mod)] = demos
    return demos_by_mod

def _build_qwen_style_messages_with_demos(demos, query_img, query_text):
    messages = []
    for ex in demos:
        messages.append({
            "role": "user",
            "content": [{"type": "image", "image": ex["image"]}, {"type": "text", "text": ex["text"]}],
        })
        messages.append({"role": "assistant", "content": [{"type": "text", "text": ex["answer"]}]})
    messages.append({
        "role": "user",
        "content": [{"type": "image", "image": query_img}, {"type": "text", "text": query_text}],
    })
    return messages

def _build_internvl_single_prompt_with_demos(processor, demos, query_text):
    image_tok = getattr(processor, "image_token", None) or "<image>"
    parts = []
    for ex in demos:
        parts.append(f"{image_tok}\n{ex['text']}\n{ex['answer']}\n")
    parts.append(f"{image_tok}\n{query_text}\n")
    return "\n".join(parts)

def _decode_generated_text(tok, inputs: Dict[str, Any], sequences: torch.Tensor, max_new_tokens: int) -> str:
    """
    Prefer: decode only generated part (sequences[:, input_len:])
    Fallback: decode tail max_new_tokens
    """
    input_ids = inputs.get("input_ids", None)

    # 1) generated-part decode
    if input_ids is not None and torch.is_tensor(input_ids) and sequences.ndim == 2:
        in_len = int(input_ids.shape[1])
        if sequences.shape[1] > in_len:
            gen = sequences[:, in_len:]
            t = tok.batch_decode(gen, skip_special_tokens=True)[0].strip()
            if t:
                # take last non-empty line
                parts = [p.strip() for p in re.split(r"[\n\r]+", t) if p.strip()]
                return parts[-1] if parts else t

    # 2) fallback: tail
    tail = sequences[:, -max_new_tokens:] if sequences.ndim == 2 else sequences
    t = tok.batch_decode(tail, skip_special_tokens=True)[0].strip()
    if not t:
        t = tok.batch_decode(sequences, skip_special_tokens=True)[0].strip()
    parts = [p.strip() for p in re.split(r"[\n\r]+", t) if p.strip()]
    return parts[-1] if parts else t

def _dbg_print_inputs_once(tag: str, inputs: Dict[str, Any], max_items: int = 50):
    # tag: "medgemma" or "internvl"
    if DBG_ONCE.get(tag, False):
        return
    DBG_ONCE[tag] = True

    print(f"\n[{tag} debug] processor outputs keys = {list(inputs.keys())}", flush=True)

    # 텐서들 shape/dtype/device
    for k in list(inputs.keys())[:max_items]:
        v = inputs[k]
        if torch.is_tensor(v):
            print(
                f"[{tag} debug] {k}: shape={tuple(v.shape)} dtype={v.dtype} device={v.device} "
                f"min={float(v.min()) if v.numel() else 'NA'} max={float(v.max()) if v.numel() else 'NA'}",
                flush=True
            )
        elif isinstance(v, (list, tuple)) and len(v) and torch.is_tensor(v[0]):
            print(f"[{tag} debug] {k}: list[{len(v)}] of tensors, first shape={tuple(v[0].shape)} dtype={v[0].dtype}", flush=True)
        else:
            # 너무 길면 잘라서
            s = str(v)
            if len(s) > 200:
                s = s[:200] + " ... (truncated)"
            print(f"[{tag} debug] {k}: {type(v).__name__} = {s}", flush=True)

    print(f"[{tag} debug] end\n", flush=True)


@torch.no_grad()
def infer_one(model, processor, backend: str, demos, img: Image.Image, text: str,
              max_new_tokens: int, parse_mode: str):

    model_id = MODEL_ID_BY_BACKEND[backend]
    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=(backend in ["internvl", "medgemma"]))

    # -------------------------
    # Qwen/Lingshu: 그대로
    # -------------------------
    if backend in ["qwen3", "lingshu"]:
        msgs = _build_qwen_style_messages_with_demos(demos, img, text)
        chat_text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        all_images = [ex["image"] for ex in demos] + [img]
        inputs = processor(text=[chat_text], images=all_images, padding=True, return_tensors="pt").to(device)

    # -------------------------
    # InternVL: 그대로 (inline <image>)
    # -------------------------
    elif backend == "internvl":
        prompt = _build_internvl_single_prompt_with_demos(processor, demos, text)
        all_images = [ex["image"] for ex in demos] + [img]
        inputs = processor(text=[prompt], images=all_images, padding=True, return_tensors="pt").to(device)

        _dbg_print_inputs_once(backend, inputs)

        if device == "cuda":
            for k in ["pixel_values", "images", "vision_x"]:
                if k in inputs and torch.is_tensor(inputs[k]):
                    inputs[k] = inputs[k].float()

    # -------------------------
    # ✅ MedGemma: apply_chat_template 로 이미지토큰 자동 삽입
    # -------------------------
    elif backend == "medgemma":
        messages = []
        for ex in demos:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": ex["text"]},
                ],
            })
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": ex["answer"]}],
            })

        messages.append({
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        })

        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        all_images = [ex["image"] for ex in demos] + [img]

        # ✅ 여기서 processor가 chat_text 안에 "이미지 토큰"을 자동으로 넣어줌
        inputs = processor(text=chat_text, images=all_images, padding=True, return_tensors="pt").to(device)

        _dbg_print_inputs_once("medgemma", inputs)

        if device == "cuda":
            for k in ["pixel_values", "images", "vision_x"]:
                if k in inputs and torch.is_tensor(inputs[k]):
                    inputs[k] = inputs[k].float()

    else:
        raise RuntimeError(backend)

    gen_out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        num_beams=1,
        return_dict_in_generate=True,
        pad_token_id=getattr(tok, "pad_token_id", None),
        eos_token_id=getattr(tok, "eos_token_id", None),
    )
    raw = _decode_generated_text(tok, inputs, gen_out.sequences, max_new_tokens=max_new_tokens)

    lab = text_to_label(raw, mode=parse_mode)
    return raw, (None if lab is None else int(lab))



# -------------------------
# Metrics
# -------------------------
def compute_binary_metrics(yt: np.ndarray, yp: np.ndarray):
    tp = int(((yp == 1) & (yt == 1)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    eps = 1e-12
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return dict(acc=float(acc), precision=float(precision), recall=float(recall), f1=float(f1),
                tp=tp, tn=tn, fp=fp, fn=fn, n=int(len(yt)))

def compute_metrics_from_records(records: List[Dict[str, Any]]):
    yt, yp = [], []
    n_nan = 0
    for r in records:
        y = r.get("y_true", None)
        p = r.get("y_pred", None)
        if p is None:
            n_nan += 1
            continue
        if y is None:
            continue
        yt.append(int(y))
        yp.append(int(p))
    out = {
        "n_total": int(len(records)),
        "n_pred_finite": int(len(yp)),
        "n_nan": int(n_nan),
        "metrics": compute_binary_metrics(np.array(yt), np.array(yp)) if len(yp) else {},
    }
    return out

# -------------------------
# Run query inference (single list)
# -------------------------
def run_query_df(model, processor, backend: str, demos_by_mod: dict, dfq: pd.DataFrame, split_name: str):
    outputs = []
    y_true_list, y_pred_list = [], []
    n_nan = 0

    for i, row in dfq.iterrows():
        mod = str(row["dataset_norm"]).lower()
        if mod not in demos_by_mod:
            raise RuntimeError(f"demo missing for modality={mod} in demo_csv")

        demos = demos_by_mod[mod]
        img = Image.open(row["filepath"]).convert("RGB")
        y_true = None if ("binarylabel" not in row or pd.isna(row["binarylabel"])) else int(row["binarylabel"])
        prompt = SYSTEM_PROMPT_SHORT + "\n" + PROMPT_BY_DATASET.get(mod, PROMPT_BY_DATASET["mri"])

        raw, y_pred = infer_one(
            model, processor, backend, demos, img, prompt,
            max_new_tokens=args.max_new_tokens, parse_mode=args.parse_mode
        )

        outputs.append({
            "split": split_name,
            "idx": int(i),
            "fileindex": str(row["fileindex"]),
            "path": str(row["filepath"]),
            "dataset": mod,
            "severity": str(row.get("severity_norm", "clean")),
            "y_true": None if y_true is None else int(y_true),
            "y_pred": y_pred,
            "raw_text": raw,
        })

        if y_pred is None:
            n_nan += 1
        else:
            if y_true is not None:
                y_true_list.append(int(y_true))
                y_pred_list.append(int(y_pred))

    metrics = {}
    if len(y_true_list) > 0:
        metrics = compute_binary_metrics(np.array(y_true_list), np.array(y_pred_list))

    return {
        "split": split_name,
        "n_total": int(len(dfq)),
        "n_pred_finite": int(len(y_pred_list)),
        "n_nan": int(n_nan),
        "metrics": metrics,
        "per_image": outputs,
    }

# -------------------------
# Pair evaluation INSIDE the pair-set (clean_weak.csv only)
# -------------------------
def eval_pairs_within_pairset(pair_res: dict):
    """
    pair_res: run_query_df 결과 from clean_weak.csv (contains both clean and weak rows)

    Expectations:
      - same fileindex groups contain 1 clean + N weak
      - severity field distinguishes clean vs weak (severity == 'clean' treated as clean-side, else weak-side)
    """
    # group by fileindex
    groups = {}
    for r in pair_res["per_image"]:
        fid = str(r["fileindex"])
        groups.setdefault(fid, []).append(r)

    per_pair = []
    used = 0
    skipped = 0
    flip_any = 0
    flip_majority = 0

    yt = []
    pc = []
    pm = []
    pw = []

    by_modality = {}

    def _ensure_mod(mod):
        if mod not in by_modality:
            by_modality[mod] = dict(
                n_pairs_total=0,
                n_pairs_used=0,
                n_pairs_skipped=0,
                flip_any=0,
                flip_majority=0,
                yt=[], pc=[], pm=[], pw=[],
            )
        return by_modality[mod]

    for fid, items in groups.items():
        # split clean vs weak
        clean_items = [x for x in items if str(x.get("severity", "clean")).lower() == "clean"]
        weak_items  = [x for x in items if str(x.get("severity", "clean")).lower() != "clean"]

        # choose clean (prefer the one with finite pred, else first)
        clean = None
        for c in clean_items:
            if c.get("y_pred", None) is not None:
                clean = c
                break
        if clean is None and len(clean_items) > 0:
            clean = clean_items[0]

        mod = str((clean or items[0]).get("dataset", "unknown")).lower()
        mref = _ensure_mod(mod)
        mref["n_pairs_total"] += 1

        # collect weak finite preds
        weak_preds = []
        for w in weak_items:
            if w.get("y_pred", None) is not None:
                weak_preds.append(int(w["y_pred"]))

        # require: clean exists, has y_true, clean pred finite, and at least one weak finite
        if clean is None or clean.get("y_true", None) is None or clean.get("y_pred", None) is None or len(weak_preds) == 0:
            skipped += 1
            mref["n_pairs_skipped"] += 1
            per_pair.append({
                "fileindex": fid,
                "dataset": mod,
                "skipped": True,
                "reason": "missing_clean_or_label_or_clean_nan_or_all_weak_nan_or_no_weak_rows",
                "clean_candidates": clean_items,
                "weak_list": weak_items,
            })
            continue

        y = int(clean["y_true"])
        pred_c = int(clean["y_pred"])

        # majority over weak preds
        pred_majority = 1 if (sum(weak_preds) >= (len(weak_preds) / 2)) else 0

        # worst-case wrt GT
        if y == 0:
            pred_worstgt = 1 if any(p == 1 for p in weak_preds) else int(pred_majority)
        else:
            pred_worstgt = 0 if any(p == 0 for p in weak_preds) else int(pred_majority)

        this_flip_any = any(pred_c != p for p in weak_preds)
        this_flip_majority = (pred_c != int(pred_majority))

        used += 1
        mref["n_pairs_used"] += 1

        if this_flip_any:
            flip_any += 1
            mref["flip_any"] += 1
        if this_flip_majority:
            flip_majority += 1
            mref["flip_majority"] += 1

        yt.append(y); pc.append(pred_c); pm.append(int(pred_majority)); pw.append(int(pred_worstgt))
        mref["yt"].append(y); mref["pc"].append(pred_c); mref["pm"].append(int(pred_majority)); mref["pw"].append(int(pred_worstgt))

        per_pair.append({
            "fileindex": fid,
            "dataset": mod,
            "skipped": False,
            "y_true": y,
            "pred_clean": pred_c,
            "weak_preds": weak_preds,
            "agg": {
                "pred_majority": int(pred_majority),
                "pred_worst_gt": int(pred_worstgt),
                "flip_any": bool(this_flip_any),
                "flip_majority": bool(this_flip_majority),
            },
            "clean_path": clean.get("path", None),
            "weak_paths": [w.get("path", None) for w in weak_items],
            "clean_raw": clean.get("raw_text", None),
            "weak_raws": [w.get("raw_text", None) for w in weak_items],
        })

    out = {
        "n_fileindex_groups": int(len(groups)),
        "n_pairs_used": int(used),
        "n_pairs_skipped": int(skipped),
        "flip_rate_any": float(flip_any / max(1, used)),
        "flip_rate_majority": float(flip_majority / max(1, used)),
        "clean_metrics": compute_binary_metrics(np.array(yt), np.array(pc)) if used else {},
        "weak_majority_metrics": compute_binary_metrics(np.array(yt), np.array(pm)) if used else {},
        "weak_worst_gt_metrics": compute_binary_metrics(np.array(yt), np.array(pw)) if used else {},
        "per_pair": per_pair,
    }

    by_mod_out = {}
    for mod, m in by_modality.items():
        u = int(m["n_pairs_used"])
        by_mod_out[mod] = {
            "n_pairs_total": int(m["n_pairs_total"]),
            "n_pairs_used": u,
            "n_pairs_skipped": int(m["n_pairs_skipped"]),
            "flip_rate_any": float(m["flip_any"] / max(1, u)) if u else 0.0,
            "flip_rate_majority": float(m["flip_majority"] / max(1, u)) if u else 0.0,
            "clean_metrics": compute_binary_metrics(np.array(m["yt"]), np.array(m["pc"])) if u else {},
            "weak_majority_metrics": compute_binary_metrics(np.array(m["yt"]), np.array(m["pm"])) if u else {},
            "weak_worst_gt_metrics": compute_binary_metrics(np.array(m["yt"]), np.array(m["pw"])) if u else {},
        }
    out["by_modality"] = by_mod_out
    return out

# -------------------------
# Main
# -------------------------
def main():
    backend = args.backend
    model_id = MODEL_ID_BY_BACKEND[backend]
    model, processor = load_backend(backend, model_id)

    df_demo = load_csv(args.demo_csv)
    demos_by_mod = build_demos_by_modality(df_demo)
    for mod, demos in demos_by_mod.items():
        print(f"[demo] {mod}: {len(demos)} images", flush=True)

    # 1) Standalone clean set (NO pairing)
    df_clean = load_csv(args.query_clean_csv)
    res_clean = run_query_df(model, processor, backend, demos_by_mod, df_clean, "standalone_clean")

    # 2) Pair-set csv (contains clean+weak within the same fileindex groups)
    df_pair = load_csv(args.query_weak_csv)
    res_pair_all = run_query_df(model, processor, backend, demos_by_mod, df_pair, "pair_set_all")

    # weak-only slice metrics (optional but useful)
    weak_records = [r for r in res_pair_all["per_image"] if str(r.get("severity", "clean")).lower() != "clean"]
    pair_weak_only = compute_metrics_from_records(weak_records)

    # ✅ Correct pair robustness: computed ONLY within pair-set (clean_weak.csv)
    paired_within = eval_pairs_within_pairset(res_pair_all)

    results = {
        "backend": backend,
        "demo_csv": args.demo_csv,
        "standalone_clean_csv": args.query_clean_csv,
        "pair_set_csv": args.query_weak_csv,
        "max_new_tokens": int(args.max_new_tokens),
        "parse_mode": args.parse_mode,
        "demo_summary": {k: len(v) for k, v in demos_by_mod.items()},

        # outputs
        "standalone_clean": res_clean,
        "pair_set_all": res_pair_all,
        "pair_set_weak_only": pair_weak_only,
        "paired_within_pair_set": paired_within,
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[saved] {args.out_json}", flush=True)

    del model, processor
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
