#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VLM-only evaluation (TEXT classifier) + per-image output logging to JSON

Fixes / improvements vs previous:
1) Robust decoding: do NOT slice by input length; instead decode last max_new_tokens or full and extract.
2) Stronger system prompt + optional strict parsing mode (default strict for fairness).
3) Pair indexing: treat anything != clean as artifact (not only "weak").
4) Safer out_json directory creation.
5) Explicit deterministic generation args.
6) Cleaner handling of NaN / unparsable outputs.

Run:
  python vlm_text_eval.py --backend qwen3 \
    --test_clean_csv /path/clean.csv \
    --test_pair_csv  /path/pairs.csv \
    --out_json /path/out.json
"""

import os
import gc
import json
import re
import argparse
from typing import Optional, Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Args
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", type=str, required=True,
                   choices=["all","qwen3", "medgemma", "internvl", "lingshu"])
    p.add_argument("--test_clean_csv", type=str, required=True,
                   help="CSV containing clean-only test set")
    p.add_argument("--test_pair_csv", type=str, required=True,
                   help="CSV containing clean+artifact test set (grouped by fileindex)")
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=2,
                   help="Generation length for single-token classification (use >=4 for stability).")
    p.add_argument("--out_json", type=str, default=None,
                   help="Write metrics + per-image outputs to this JSON path")
    p.add_argument("--parse_mode", type=str, default="strict",
                   choices=["strict", "relaxed"],
                   help="strict: first token must be normal/disease else NaN. relaxed: keyword heuristics.")
    p.add_argument("--artifact_policy", type=str, default="not_clean",
                   choices=["weak_only", "not_clean"],
                   help="How to select artifact samples in pairs: weak_only or everything != clean (default).")
    return p.parse_args()


args = parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device, flush=True)

# HF cache
assert os.environ.get("HF_HOME"), "HF_HOME not set"
hf_home = os.environ["HF_HOME"]
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(hf_home, "hub"))


# -------------------------
# Prompts
# -------------------------
PROMPT_BY_DATASET = {
    "mri": (
        "This is a brain MRI scan.\n"
        "Question: Does this image show normal anatomy or signs of disease?\n\n"
    ),
    "oct": (
        "This is a retinal OCT scan.\n"
        "Question: Does this image show normal anatomy or signs of disease?\n\n"
    ),
    "xray": (
        "This is a chest X-ray image.\n"
        "Question: Does this image show normal anatomy or signs of disease?\n\n"
    ),
    "fundus": (
        "This is a retinal fundus photograph.\n"
        "Question: Does this image show normal anatomy or signs of disease?\n\n"
    ),
}

# Stronger system instruction (helps stop punctuation / extra words)
SYSTEM_PROMPT_SHORT = (
    "You are a medical image classifier.\n"
    "Answer with ONE WORD only: \"normal\" or \"disease\".\n"
    "Do not add any explanation or punctuation."
)



# -------------------------
# Model IDs
# -------------------------
MODEL_ID_BY_BACKEND = {
    "qwen3":    "Qwen/Qwen3-VL-8B-Instruct",
    "medgemma": "google/medgemma-1.5-4b-it",
    "internvl": "OpenGVLab/InternVL3_5-8B-HF",
    "lingshu":  "lingshu-medical-mllm/Lingshu-7B",
}


# -------------------------
# Utils
# -------------------------
def to_device(x, target_device: torch.device):
    if torch.is_tensor(x):
        return x.to(target_device, non_blocking=True)

    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            # keep meta strings on CPU
            if k in ("paths", "path", "meta", "severities", "fileindices", "datasets"):
                out[k] = v
            else:
                out[k] = to_device(v, target_device)
        return out

    if isinstance(x, (list, tuple)):
        if len(x) > 0 and all(isinstance(v, str) for v in x):
            return x
        return type(x)(to_device(v, target_device) for v in x)

    return x


def compute_binary_metrics_from_preds(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_true = y_true.detach().cpu().to(torch.int64)
    y_pred = y_pred.detach().cpu().to(torch.int64)

    tp = int(((y_pred == 1) & (y_true == 1)).sum().item())
    tn = int(((y_pred == 0) & (y_true == 0)).sum().item())
    fp = int(((y_pred == 1) & (y_true == 0)).sum().item())
    fn = int(((y_pred == 0) & (y_true == 1)).sum().item())

    eps = 1e-12
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n": int(len(y_true)),
    }


def _get_tokenizer_from_processor(processor, model_id: Optional[str] = None, trust_remote_code: bool = False):
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        return processor.tokenizer
    if model_id is not None:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    raise RuntimeError("Tokenizer not found in processor and model_id not provided.")


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
    STRICT v3:
    - first meaningful token must be normal/disease
    - allow small prefixes like "Output:", "Answer:", etc.
    - if first token is not a label, allow verbose sentence only when it
      contains exactly one of {normal, disease} and not both.
    """
    t = _strip_code_fence(s or "").strip()
    s_norm = _normalize_answer_text(t)
    toks = s_norm.split()
    if not toks:
        return None

    allowed_prefix = {"output", "answer", "label", "prediction", "pred", "result"}
    i = 0
    while i < len(toks) and (toks[i] in allowed_prefix or toks[i] in {"-", ":"}):
        i += 1

    first = toks[i] if i < len(toks) else ""
    if first == "disease":
        return 1
    if first == "normal":
        return 0

    has_normal = ("normal" in toks)
    has_disease = ("disease" in toks)
    if has_normal ^ has_disease:
        return 0 if has_normal else 1
    return None




def _text_to_label_relaxed(s: str) -> Optional[int]:
    """
    RELAXED: keyword heuristics (less fair, but can be useful in appendix).
    """
    t = _strip_code_fence(s or "").strip()
    low = t.lower()

    # JSON whole
    try:
        obj = json.loads(t)
        label = obj.get("label", None)
        if isinstance(label, str):
            label = label.strip().lower()
        if label == "disease":
            return 1
        if label == "normal":
            return 0
        if label in ("distorted", "ungradable"):
            return None
    except Exception:
        pass

    # JSON inside braces
    m = re.search(r"\{.*?\}", t, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            label = obj.get("label", None)
            if isinstance(label, str):
                label = label.strip().lower()
            if label == "disease":
                return 1
            if label == "normal":
                return 0
            if label in ("distorted", "ungradable"):
                return None
        except Exception:
            pass

    # explicit "can't assess"
    if re.search(
        r"\b(distorted|ungradable|not gradable|non[- ]diagnostic|cannot be assessed|"
        r"too (noisy|blurred|blurry)|severely degraded)\b",
        low,
    ):
        return None

    # strong negatives -> normal
    neg_disease_pattern = re.compile(
        r"no (evidence of )?(significant )?(acute )?"
        r"(disease|abnormalit(?:y|ies)|lesion|pneumonia|pathology|finding[s]?)"
    )
    neg_env_pattern = re.compile(
        r"(normal (study|scan|examination|chest x[- ]ray|x[- ]ray|mri|oct|image))"
        r"|(\bunremarkable( study| scan| appearance)?\b)"
        r"|(\bwithin normal limits\b|\bwnl\b)"
    )
    hedge_pattern = re.compile(
        r"cannot (exclude|rule out)|suspicious for|concern(ing)? for|suggestive of"
    )

    if (neg_disease_pattern.search(low) or neg_env_pattern.search(low)) and not hedge_pattern.search(low):
        return 0

    # disease keywords
    disease_keywords = [
        "disease", "lesion", "abnormal", "opacity", "consolidation",
        "nodule", "mass", "effusion", "infiltrate", "infiltration",
        "edema", "oedema", "hemorrhage", "haemorrhage",
        "stroke", "infarct", "ischemia", "ischaemia",
        "plaque", "drusen", "fluid", "thickening",
        "collapse", "atelectasis", "pneumonia", "tumour", "tumor",
        "metastasis", "infection", "pathology"
    ]
    for kw in disease_keywords:
        if kw in low:
            if re.search(rf"no (evidence of )?(significant )?(acute )?{kw}", low):
                continue
            return 1

    if re.search(r"\bno (significant )?abnormalit(?:y|ies)\b", low) and not hedge_pattern.search(low):
        return 0

    if re.search(r"\bnormal\b", low) and not re.search(r"\bdisease\b|\babnormal\b|\blesion\b", low):
        return 0

    # fallback: strict first token
    return _text_to_label_strict(t)


def text_to_label(s: str, mode: str) -> Optional[int]:
    return _text_to_label_strict(s) if mode == "strict" else _text_to_label_relaxed(s)


# -------------------------
# Backend loader
# -------------------------
def load_backend(backend: str, model_id: str):
    import transformers
    from transformers import AutoProcessor

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    if backend in ["medgemma", "internvl"] and device == "cuda":
        torch_dtype = torch.bfloat16

    cache_dir = os.environ.get("TRANSFORMERS_CACHE") or os.environ.get("HF_HOME") or None

    common = dict(
        torch_dtype=torch_dtype,
        device_map=None,
        low_cpu_mem_usage=True,
    )

    if backend == "qwen3":
        from transformers import Qwen3VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, cache_dir=cache_dir, **common)
        model.to(device).eval()
        return model, processor

    if backend == "lingshu":
        from transformers import Qwen2_5_VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, cache_dir=cache_dir, **common)
        model.to(device).eval()
        return model, processor

    if backend == "internvl":
        from transformers import AutoProcessor, AutoModelForImageTextToText
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, cache_dir=cache_dir)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map=None,
        )
        model.to(device).eval()
        return model, processor

    if backend == "medgemma":
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)

        AutoImageTextToText = getattr(transformers, "AutoModelForImageTextToText", None)
        if AutoImageTextToText is not None:
            model = AutoImageTextToText.from_pretrained(model_id, cache_dir=cache_dir, **common)
            model.to(device).eval()
            return model, processor

        AutoVision2Seq = getattr(transformers, "AutoModelForVision2Seq", None)
        if AutoVision2Seq is not None:
            model = AutoVision2Seq.from_pretrained(model_id, cache_dir=cache_dir, **common)
            model.to(device).eval()
            return model, processor

        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, **common)
        model.to(device).eval()
        return model, processor

    raise ValueError(f"Unknown backend: {backend}")


# -------------------------
# CSV load
# -------------------------
BASE_OUT_DIR = "/SAN/ioo/HORIZON/howoon"


def load_split_csv(path: str, base_out_dir: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "binarylab" in df.columns and "binarylabel" not in df.columns:
        df = df.rename(columns={"binarylab": "binarylabel"})

    if "severity" not in df.columns:
        df["severity"] = "clean"

    df["severity_norm"] = df["severity"].fillna("clean").astype(str).str.lower()
    df["dataset_norm"]  = df["dataset"].astype(str).str.lower()

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
        raise ValueError(f"{path} needs fileindex column")

    return df.reset_index(drop=True)


# -------------------------
# Dataset + collate
# -------------------------
class SingleImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, prompt_by_dataset: Dict[str, str], system_prompt: Optional[str] = None):
        self.df = df.copy().reset_index(drop=True)
        self.prompt_by_dataset = prompt_by_dataset
        self.system_prompt = system_prompt

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["filepath"]
        img = Image.open(img_path).convert("RGB")

        modality = str(row["dataset_norm"]).lower()
        label = int(row["binarylabel"])
        sev = str(row["severity_norm"]).lower()

        task_prompt = self.prompt_by_dataset.get(
            modality,
            "This is a medical image.\nQuestion: Does this image show normal anatomy or signs of disease?\n\n",
        )
        full_text = (self.system_prompt + "\n" + task_prompt) if self.system_prompt else task_prompt
        # Add an explicit answer slot to reduce sentence-style generations.
        full_text = full_text.rstrip() + "\nFinal answer (one word):"

        return {
            "image": img,
            "input_text": full_text,
            "label": label,
            "path": img_path,
            "severity": sev,
            "fileindex": str(row["fileindex"]),
            "dataset": modality,
        }


def make_single_collate_fn(processor, backend: str):
    def collate(batch):
        images = [b["image"] for b in batch]
        texts  = [b["input_text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        paths  = [b["path"] for b in batch]
        sevs   = [b["severity"] for b in batch]
        fids   = [str(b["fileindex"]) for b in batch]
        dsets  = [str(b["dataset"]) for b in batch]

        if backend in ["lingshu", "qwen3"]:
            messages_list = []
            for img, txt in zip(images, texts):
                messages_list.append([{
                    "role": "user",
                    "content": [{"type": "image", "image": img}, {"type": "text", "text": txt}],
                }])
            chat_texts = [
                processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in messages_list
            ]
            model_inputs = processor(text=chat_texts, images=images, padding=True, return_tensors="pt")

        elif backend == "internvl":
            image_tok = getattr(processor, "image_token", None) or "<image>"
            inline_texts = [f"{image_tok}\n{txt}" for txt in texts]
            model_inputs = processor(text=inline_texts, images=images, padding=True, return_tensors="pt")

        elif backend == "medgemma":
            if not hasattr(processor, "apply_chat_template"):
                raise RuntimeError("medgemma needs apply_chat_template")
            messages_list = []
            for img, txt in zip(images, texts):
                messages_list.append([{
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": txt}],
                }])
            chat_texts = [
                processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in messages_list
            ]
            images_nested = [[img] for img in images]
            model_inputs = processor(text=chat_texts, images=images_nested, padding=True, return_tensors="pt")

        else:
            model_inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")

        out = dict(model_inputs)
        out["labels_cls"] = labels
        out["paths"] = paths
        out["severities"] = sevs
        out["fileindices"] = fids
        out["datasets"] = dsets
        return out
    return collate


def build_pair_index(df_pair: pd.DataFrame, artifact_policy: str) -> List[Tuple[str, pd.Series, List[pd.Series]]]:
    items = []
    for fid, g in df_pair.groupby("fileindex"):
        g = g.reset_index(drop=True)
        clean_rows = g[g["severity_norm"] == "clean"]
        if artifact_policy == "weak_only":
            art_rows = g[g["severity_norm"] == "weak"]
        else:
            art_rows = g[g["severity_norm"] != "clean"]

        if len(clean_rows) == 0 or len(art_rows) == 0:
            continue
        clean_row = clean_rows.iloc[0]
        art_list = [art_rows.iloc[i] for i in range(len(art_rows))]
        items.append((str(fid), clean_row, art_list))
    return items


# -------------------------
# Forward (generation + robust decoding)
# -------------------------
def _decode_generated_text(
    tok,
    input_ids: Optional[torch.Tensor],
    sequences: torch.Tensor,
    max_new_tokens: int,
) -> List[str]:
    """
    Robustly decode generated answer.
    1) Best: decode generated part only (sequences[:, input_len:])
    2) Fallback: decode tail max_new_tokens
    3) If still empty: full decode then take last non-empty line
    """
    # 1) best: decode generated part if possible
    if input_ids is not None and torch.is_tensor(input_ids) and sequences.ndim == 2:
        in_len = int(input_ids.shape[1])
        if sequences.shape[1] > in_len:
            gen = sequences[:, in_len:]
            texts = tok.batch_decode(gen, skip_special_tokens=True)
            return [t.strip() for t in texts]

    # 2) fallback: decode only tail tokens
    tail = sequences[:, -max_new_tokens:] if sequences.ndim == 2 else sequences
    texts = tok.batch_decode(tail, skip_special_tokens=True)
    texts = [t.strip() for t in texts]

    def _is_effectively_empty(s: str):
        return len(_normalize_answer_text(s)) == 0

    # 3) fallback: full decode if tail is empty-ish
    if any(_is_effectively_empty(t) for t in texts):
        full = tok.batch_decode(sequences, skip_special_tokens=True)
        full = [t.strip() for t in full]
        texts2 = []
        for f in full:
            parts = [p.strip() for p in re.split(r"[\n\r]+", f) if p.strip()]
            texts2.append(parts[-1] if parts else f.strip())
        texts = texts2

    return texts




@torch.no_grad()
def forward_vlm_only_get_pred_and_text(
    base_model,
    processor,
    backend: str,
    batch: Dict[str, Any],
    max_new_tokens: int = 2,
    parse_mode: str = "strict",
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Returns (y_true, y_pred, texts)
      - y_true: [B] int64 labels
      - y_pred: [B] float32 with {0,1} or NaN if unparsable
      - texts:  list[str] decoded generated text
    """
    model_id = MODEL_ID_BY_BACKEND[backend]
    tok = _get_tokenizer_from_processor(
        processor,
        model_id=model_id,
        trust_remote_code=(backend == "internvl"),
    )

    target_device = next(base_model.parameters()).device
    batch = to_device(batch, target_device)

    y_true = batch["labels_cls"].detach().cpu()

    d = dict(batch)
    for k in ["labels_cls", "paths", "severities", "fileindices", "datasets", "labels_token"]:
        d.pop(k, None)

    # autocast quirks
    if backend == "internvl" and device == "cuda":
        for k in ["pixel_values", "images", "image", "vision_x"]:
            if k in d and torch.is_tensor(d[k]):
                d[k] = d[k].to(dtype=torch.float32)
        autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=False)
    else:
        autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=(device == "cuda"))

    # deterministic generation
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        num_beams=1,
        return_dict_in_generate=True,
    )
    # (optional) pad/eos ids if available
    if hasattr(tok, "pad_token_id") and tok.pad_token_id is not None:
        gen_kwargs["pad_token_id"] = tok.pad_token_id
    if hasattr(tok, "eos_token_id") and tok.eos_token_id is not None:
        gen_kwargs["eos_token_id"] = tok.eos_token_id

    with autocast_ctx:
        gen_out = base_model.generate(**d, **gen_kwargs)

    seq = gen_out.sequences  # [B, T]
    input_ids = d.get("input_ids", None) if isinstance(d, dict) else None

    texts = _decode_generated_text(tok, input_ids, seq, max_new_tokens=max_new_tokens)

    preds = []
    texts_norm = []
    for t in texts:
        lab = text_to_label(t, mode=parse_mode)
        preds.append(float("nan") if lab is None else int(lab))
        if lab is None:
            texts_norm.append(t)
        else:
            texts_norm.append("disease" if int(lab) == 1 else "normal")

    y_pred = torch.tensor(preds, dtype=torch.float32)
    return y_true, y_pred, texts_norm


# -------------------------
# Eval: single
# -------------------------
def eval_single_loader_vlm_only_text(
    base_model,
    processor,
    backend: str,
    loader: DataLoader,
    max_new_tokens: int,
    split_name: str,
    parse_mode: str,
):
    """
    Returns:
      metrics_all, metrics_clean, metrics_art, n_ok, n_nan, per_image_records(list[dict]), by_modality(dict)
    """
    all_y, all_pred = [], []
    all_sev = []
    all_dset = []
    per_image = []

    for batch in loader:
        y, pred, texts = forward_vlm_only_get_pred_and_text(
            base_model, processor, backend, batch,
            max_new_tokens=max_new_tokens,
            parse_mode=parse_mode,
        )

        all_y.append(y)
        all_pred.append(pred)

        all_sev.extend(batch["severities"])
        all_dset.extend(batch.get("datasets", ["unknown"] * len(batch["paths"])))

        # per-image logging
        paths = batch["paths"]
        fids  = batch["fileindices"]
        dsets = batch.get("datasets", ["unknown"] * len(paths))
        y_np = y.numpy().astype(int)
        pred_np = pred.detach().cpu().numpy()

        for i in range(len(paths)):
            per_image.append({
                "split": split_name,
                "path": str(paths[i]),
                "fileindex": str(fids[i]),
                "severity": str(batch["severities"][i]),
                "dataset": str(dsets[i]),
                "y_true": int(y_np[i]),
                "y_pred": None if not np.isfinite(pred_np[i]) else int(pred_np[i]),
                "raw_text": texts[i],
            })

    y_true = torch.cat(all_y, dim=0)
    y_pred = torch.cat(all_pred, dim=0)

    finite = torch.isfinite(y_pred)
    y_true2 = y_true[finite]
    y_pred2 = y_pred[finite].to(torch.int64)

    m_all = compute_binary_metrics_from_preds(y_true2, y_pred2) if int(finite.sum()) > 0 else {}

    all_sev = np.array(all_sev, dtype=object)
    all_dset = np.array(all_dset, dtype=object)

    sev2 = all_sev[finite.detach().cpu().numpy()] if len(all_sev) else np.array([], dtype=object)
    dset2 = all_dset[finite.detach().cpu().numpy()] if len(all_dset) else np.array([], dtype=object)

    clean_mask = (sev2 == "clean")
    art_mask   = (sev2 != "clean")

    m_clean = compute_binary_metrics_from_preds(y_true2[clean_mask], y_pred2[clean_mask]) if clean_mask.sum() > 0 else {}
    m_art   = compute_binary_metrics_from_preds(y_true2[art_mask],   y_pred2[art_mask])   if art_mask.sum() > 0 else {}

    # -------------------------
    # NEW: by-modality metrics
    # -------------------------
    by_modality = {}
    if int(finite.sum()) > 0:
        modalities = sorted(set([str(x) for x in dset2.tolist()]))
        for mod in modalities:
            mod_mask = (dset2 == mod)

            m_mod_all = compute_binary_metrics_from_preds(y_true2[mod_mask], y_pred2[mod_mask]) if mod_mask.sum() > 0 else {}
            m_mod_clean = compute_binary_metrics_from_preds(
                y_true2[mod_mask & clean_mask],
                y_pred2[mod_mask & clean_mask],
            ) if (mod_mask & clean_mask).sum() > 0 else {}
            m_mod_art = compute_binary_metrics_from_preds(
                y_true2[mod_mask & art_mask],
                y_pred2[mod_mask & art_mask],
            ) if (mod_mask & art_mask).sum() > 0 else {}

            by_modality[mod] = {
                "all": m_mod_all,
                "clean": m_mod_clean,
                "artifact": m_mod_art,
                "n_finite": int(mod_mask.sum()),
                "n_nan": int(((all_dset == mod) & (~finite.detach().cpu().numpy())).sum()),
            }

    return m_all, m_clean, m_art, int(finite.sum().item()), int((~finite).sum().item()), per_image, by_modality

# -------------------------
# Eval: paired
# -------------------------
def eval_pairs_vlm_only(
    base_model,
    processor,
    backend: str,
    pair_items: List[Tuple[str, pd.Series, List[pd.Series]]],
    max_new_tokens: int,
    parse_mode: str,
):
    """
    Returns:
      stats dict + per_pair_records(list[dict]) + by_modality(dict)

    Aggregations on artifact samples:
      - majority: majority vote among finite artifact preds
      - worst_gt: "worst-case w.r.t. ground truth"
        If ANY finite artifact prediction can make the case wrong, pick that wrong label.
        Else fall back to majority.
    """

    def make_one_batch(img: Image.Image, text: str, label: int):
        if backend in ["lingshu", "qwen3"]:
            messages = [[{
                "role": "user",
                "content": [{"type": "image", "image": img}, {"type": "text", "text": text}],
            }]]
            chat_text = processor.apply_chat_template(messages[0], tokenize=False, add_generation_prompt=True)
            model_inputs = processor(text=[chat_text], images=[img], padding=True, return_tensors="pt")

        elif backend == "internvl":
            image_tok = getattr(processor, "image_token", None) or "<image>"
            inline_text = f"{image_tok}\n{text}"
            model_inputs = processor(text=[inline_text], images=[img], padding=True, return_tensors="pt")

        elif backend == "medgemma":
            messages = [[{
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": text}],
            }]]
            chat_text = processor.apply_chat_template(messages[0], tokenize=False, add_generation_prompt=True)
            model_inputs = processor(text=[chat_text], images=[[img]], padding=True, return_tensors="pt")

        else:
            model_inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt")

        out = dict(model_inputs)
        out["labels_cls"] = torch.tensor([label], dtype=torch.long)
        out["paths"] = ["<pair>"]
        out["severities"] = ["<pair>"]
        out["fileindices"] = [0]
        out["datasets"] = ["<pair>"]
        return out

    total_pairs = int(len(pair_items))
    n_pairs_used = 0
    n_pairs_skipped_nan = 0

    # flip stats
    flip_majority = 0
    flip_any = 0

    # metric lists (overall)
    y_list = []
    pred_clean_list = []
    pred_majority_list = []
    pred_worstgt_list = []

    # modality-wise accumulators
    mod_acc = {}  # mod -> dict of lists/counters

    def _ensure_mod(mod):
        if mod not in mod_acc:
            mod_acc[mod] = dict(
                total=0,
                used=0,
                skipped_nan=0,
                flip_majority=0,
                flip_any=0,
                y=[],
                pc=[],
                pm=[],
                pw=[],
            )
        return mod_acc[mod]

    per_pair = []

    for fid, clean_row, art_rows in pair_items:
        modality = str(clean_row["dataset_norm"]).lower()
        mref = _ensure_mod(modality)
        mref["total"] += 1

        task_prompt = PROMPT_BY_DATASET.get(modality, PROMPT_BY_DATASET["mri"])
        full_text = SYSTEM_PROMPT_SHORT + "\n" + task_prompt

        y = int(clean_row["binarylabel"])

        # ---- clean ----
        img_c = Image.open(clean_row["filepath"]).convert("RGB")
        batch_c = make_one_batch(img_c, full_text, y)
        _, pred_c_t, text_c = forward_vlm_only_get_pred_and_text(
            base_model, processor, backend, batch_c,
            max_new_tokens=max_new_tokens,
            parse_mode=parse_mode,
        )
        pred_c = float(pred_c_t.item())
        text_c = text_c[0]

        # ---- artifacts ----
        art_entries = []
        art_preds = []
        for wr in art_rows:
            img_w = Image.open(wr["filepath"]).convert("RGB")
            batch_w = make_one_batch(img_w, full_text, y)
            _, pred_w_t, text_w = forward_vlm_only_get_pred_and_text(
                base_model, processor, backend, batch_w,
                max_new_tokens=max_new_tokens,
                parse_mode=parse_mode,
            )
            pred_w = float(pred_w_t.item())
            art_preds.append(pred_w)
            art_entries.append({
                "path": str(wr["filepath"]),
                "severity": str(wr["severity_norm"]),
                "y_true": y,
                "y_pred": None if not np.isfinite(pred_w) else int(pred_w),
                "raw_text": text_w[0],
            })

        # require clean finite + at least one artifact finite
        art_preds_f = [p for p in art_preds if np.isfinite(p)]
        if (not np.isfinite(pred_c)) or (len(art_preds_f) == 0):
            n_pairs_skipped_nan += 1
            mref["skipped_nan"] += 1
            per_pair.append({
                "fileindex": str(fid),
                "dataset": modality,
                "skipped": True,
                "reason": "clean_nan_or_all_art_nan",
                "clean": {
                    "path": str(clean_row["filepath"]),
                    "y_true": y,
                    "y_pred": None if not np.isfinite(pred_c) else int(pred_c),
                    "raw_text": text_c,
                },
                "artifact": art_entries,
            })
            continue

        # ---- used pair ----
        n_pairs_used += 1
        mref["used"] += 1

        pred_c_i = int(pred_c)
        art_preds_i = [int(p) for p in art_preds_f]  # finite only

        # majority vote over artifact samples
        pred_majority = 1 if (sum(art_preds_i) >= (len(art_preds_i) / 2)) else 0

        # worst-case wrt ground truth:
        # if ANY artifact pred can make it wrong, choose that wrong label; else fallback to majority.
        if y == 0:
            # GT normal: wrong is disease(1)
            pred_worstgt = 1 if any(p == 1 for p in art_preds_i) else int(pred_majority)
        else:
            # GT disease: wrong is normal(0)
            pred_worstgt = 0 if any(p == 0 for p in art_preds_i) else int(pred_majority)

        # flips
        this_flip_majority = (pred_c_i != pred_majority)
        this_flip_any = any(pred_c_i != p for p in art_preds_i)

        if this_flip_majority:
            flip_majority += 1
            mref["flip_majority"] += 1
        if this_flip_any:
            flip_any += 1
            mref["flip_any"] += 1

        # store lists for metrics
        y_list.append(y)
        pred_clean_list.append(pred_c_i)
        pred_majority_list.append(int(pred_majority))
        pred_worstgt_list.append(int(pred_worstgt))

        mref["y"].append(y)
        mref["pc"].append(pred_c_i)
        mref["pm"].append(int(pred_majority))
        mref["pw"].append(int(pred_worstgt))

        per_pair.append({
            "fileindex": str(fid),
            "dataset": modality,
            "skipped": False,
            "clean": {
                "path": str(clean_row["filepath"]),
                "y_true": y,
                "y_pred": pred_c_i,
                "raw_text": text_c,
            },
            "artifact": art_entries,
            "agg": {
                "pred_majority": int(pred_majority),
                "pred_worst_gt": int(pred_worstgt),
                "flip_majority": bool(this_flip_majority),
                "flip_any": bool(this_flip_any),
            }
        })

    # ---- overall tensors ----
    y_t = torch.tensor(y_list, dtype=torch.long) if n_pairs_used else torch.tensor([], dtype=torch.long)
    pc  = torch.tensor(pred_clean_list, dtype=torch.long) if n_pairs_used else torch.tensor([], dtype=torch.long)
    pm  = torch.tensor(pred_majority_list, dtype=torch.long) if n_pairs_used else torch.tensor([], dtype=torch.long)
    pw  = torch.tensor(pred_worstgt_list, dtype=torch.long) if n_pairs_used else torch.tensor([], dtype=torch.long)

    pair_use_rate = float(n_pairs_used / max(1, total_pairs))

    out = {
        "n_pairs_total_indexed": int(total_pairs),
        "n_pairs_used": int(n_pairs_used),
        "n_pairs_skipped_nan": int(n_pairs_skipped_nan),
        "pair_use_rate": pair_use_rate,

        "clean_metrics": compute_binary_metrics_from_preds(y_t, pc) if n_pairs_used else {},
        "art_majority_metrics": compute_binary_metrics_from_preds(y_t, pm) if n_pairs_used else {},
        "art_worst_gt_metrics": compute_binary_metrics_from_preds(y_t, pw) if n_pairs_used else {},

        "flip_rate_majority": float(flip_majority / max(1, n_pairs_used)),
        "flip_rate_any": float(flip_any / max(1, n_pairs_used)),
    }

    # ---- modality-wise stats ----
    by_modality = {}
    for mod, m in mod_acc.items():
        used = int(m["used"])
        total = int(m["total"])
        skip = int(m["skipped_nan"])

        if used > 0:
            y_m = torch.tensor(m["y"], dtype=torch.long)
            pc_m = torch.tensor(m["pc"], dtype=torch.long)
            pm_m = torch.tensor(m["pm"], dtype=torch.long)
            pw_m = torch.tensor(m["pw"], dtype=torch.long)
            by_modality[mod] = {
                "n_pairs_total_indexed": total,
                "n_pairs_used": used,
                "n_pairs_skipped_nan": skip,
                "pair_use_rate": float(used / max(1, total)),

                "clean_metrics": compute_binary_metrics_from_preds(y_m, pc_m),
                "art_majority_metrics": compute_binary_metrics_from_preds(y_m, pm_m),
                "art_worst_gt_metrics": compute_binary_metrics_from_preds(y_m, pw_m),

                "flip_rate_majority": float(m["flip_majority"] / max(1, used)),
                "flip_rate_any": float(m["flip_any"] / max(1, used)),
            }
        else:
            by_modality[mod] = {
                "n_pairs_total_indexed": total,
                "n_pairs_used": 0,
                "n_pairs_skipped_nan": skip,
                "pair_use_rate": 0.0,
                "clean_metrics": {},
                "art_majority_metrics": {},
                "art_worst_gt_metrics": {},
                "flip_rate_majority": 0.0,
                "flip_rate_any": 0.0,
            }

    return out, per_pair, by_modality

# -------------------------
# Main
# -------------------------
def _safe_makedirs_for_file(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _add_backend_suffix(out_json: str, backend: str) -> str:
    if out_json is None:
        return None
    root, ext = os.path.splitext(out_json)
    ext = ext if ext else ".json"
    return f"{root}_{backend}{ext}"


def main():
    backends =  ["qwen3", "medgemma", "internvl", "lingshu"] if args.backend == "all" else [args.backend]
    for backend in backends:
        model_id = MODEL_ID_BY_BACKEND[backend]

        print("\n==============================", flush=True)
        print(f"== VLM-ONLY TEXT EVAL backend: {backend}", flush=True)
        print(f"== parse_mode: {args.parse_mode} | artifact_policy: {args.artifact_policy}", flush=True)
        print(f"== test_clean_csv: {args.test_clean_csv}", flush=True)
        print(f"== test_pair_csv : {args.test_pair_csv}", flush=True)
        print(f"== bs={args.bs} max_new_tokens={args.max_new_tokens}", flush=True)
        print("==============================", flush=True)

        base_model, processor = load_backend(backend, model_id)

        # ---- SINGLE IMAGE TESTS ----
        df_clean = load_split_csv(args.test_clean_csv, BASE_OUT_DIR)
        df_clean = df_clean[df_clean["severity_norm"] == "clean"].reset_index(drop=True)

        df_pair = load_split_csv(args.test_pair_csv, BASE_OUT_DIR)
        if args.artifact_policy == "weak_only":
            df_art = df_pair[df_pair["severity_norm"] == "weak"].reset_index(drop=True)
        else:
            df_art = df_pair[df_pair["severity_norm"] != "clean"].reset_index(drop=True)

        ds_clean = SingleImageDataset(df_clean, PROMPT_BY_DATASET, SYSTEM_PROMPT_SHORT)
        ds_art   = SingleImageDataset(df_art,   PROMPT_BY_DATASET, SYSTEM_PROMPT_SHORT)

        collate = make_single_collate_fn(processor, backend)
        dl_clean = DataLoader(ds_clean, batch_size=args.bs, shuffle=False, collate_fn=collate, pin_memory=(device == "cuda"))
        dl_art   = DataLoader(ds_art,   batch_size=args.bs, shuffle=False, collate_fn=collate, pin_memory=(device == "cuda"))

        m_clean_all, m_clean_only, _, n_ok_clean, n_nan_clean, per_clean, bymod_clean = eval_single_loader_vlm_only_text(
            base_model, processor, backend, dl_clean,
            max_new_tokens=args.max_new_tokens,
            split_name="single_clean",
            parse_mode=args.parse_mode,
        )

        m_art_all, _, m_art_only, n_ok_art, n_nan_art, per_art, bymod_art = eval_single_loader_vlm_only_text(
            base_model, processor, backend, dl_art,
            max_new_tokens=args.max_new_tokens,
            split_name="single_artifact",
            parse_mode=args.parse_mode,
        )


        # ---- PAIRED ROBUSTNESS ----
        pair_items = build_pair_index(df_pair, artifact_policy=args.artifact_policy)
        pair_stats, per_pairs, pair_bymod = eval_pairs_vlm_only(
            base_model=base_model,
            processor=processor,
            backend=backend,
            pair_items=pair_items,
            max_new_tokens=args.max_new_tokens,
            parse_mode=args.parse_mode,
        )


        results = {
            "backend": backend,
            "vlm_only": True,
            "parse_mode": args.parse_mode,
            "artifact_policy": args.artifact_policy,
            "max_new_tokens": int(args.max_new_tokens),
            "single": {
                "n_clean_total": int(len(df_clean)),
                "n_clean_used_finite": int(n_ok_clean),
                "n_clean_nan": int(n_nan_clean),

                "n_art_total": int(len(df_art)),
                "n_art_used_finite": int(n_ok_art),
                "n_art_nan": int(n_nan_art),

                "clean_metrics": m_clean_all,
                "art_metrics": m_art_all,
                "clean_only_metrics": m_clean_only,
                "art_only_metrics": m_art_only,

                "by_modality": {
                        "single_clean": bymod_clean,
                        "single_artifact": bymod_art,
                    },
                },
            "paired": pair_stats,
            "paired_by_modality": pair_bymod,

            "per_image_outputs": {
                "single_clean": per_clean,
                "single_artifact": per_art,
                "paired": per_pairs,
            }
        }

        print("\n====== VLM-only Single-image test ======")
        print(f"Clean total={results['single']['n_clean_total']} | used={results['single']['n_clean_used_finite']} | NaN={results['single']['n_clean_nan']}")
        print(f"Art   total={results['single']['n_art_total']}   | used={results['single']['n_art_used_finite']}   | NaN={results['single']['n_art_nan']}")
        print("\n====== Paired robustness ======")
        print(f"Pairs indexed={results['paired'].get('n_pairs_total_indexed', 0)} | used={results['paired']['n_pairs_used']} | skipped(NaN)={results['paired']['n_pairs_skipped_nan']} | use_rate={results['paired'].get('pair_use_rate', 0.0):.4f}")
        print(f"Flip majority={results['paired']['flip_rate_majority']:.4f} | Flip any={results['paired']['flip_rate_any']:.4f}")

        if args.out_json:
            out_path = _add_backend_suffix(args.out_json, backend) if args.backend == "all" else args.out_json
            _safe_makedirs_for_file(out_path)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n[saved] {out_path}")

        del base_model, processor
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()
