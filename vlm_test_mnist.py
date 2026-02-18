#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Corruption-aware VLM text eval for CSV format:
- severity: 0..4 (ALL are artifact images; severity=0 is still artifact)
- clean image per clean_index exists at clean_filepath
- artifact image path is filepath
- corruption column exists (or corrupt_detail)

Outputs:
- clean metrics (unique clean_index)
- artifact metrics pooled
- artifact metrics by severity
- artifact metrics by corruption (pooled + by severity)
- robustness scores:
    per corruption:
      rBE_no_baseline_mean_c = mean_s (BE_{s,c} - BE_clean)
      dF1_mean_c             = mean_s (F1_clean - F1_{s,c})
    overall:
      rBE_over_corruptions = mean_c rBE_no_baseline_mean_c
      dF1_over_corruptions = mean_c dF1_mean_c
- paired robustness overall + by corruption

Run:
  python vlm_text_eval_corrupt_csv.py --backend qwen3 \
    --test_csv /path/octmnist_corrupt.csv \
    --out_json /path/out.json \
    --corrupt_key corruption
"""

import os, gc, json, re, argparse
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
                   choices=["all", "qwen3", "medgemma", "internvl", "lingshu"])
    p.add_argument("--test_csv", type=str, required=True)
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--out_json", type=str, default=None)
    p.add_argument("--parse_mode", type=str, default="strict",
                   choices=["strict", "relaxed"])
    p.add_argument("--base_out_dir", type=str, default="/SAN/ioo/HORIZON/howoon",
                   help="Replace Windows dataset root with this path.")
    p.add_argument("--corrupt_key", type=str, default="corruption",
                   choices=["corruption", "corrupt_detail"],
                   help="Which column to use as corruption name.")
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
SYSTEM_PROMPT_SHORT = (
    "You are a binary medical image classifier.\n"
    "Output ONLY one token.\n"
    "Allowed outputs:\n"
    "- normal\n"
    "- disease\n"
    "Do NOT output sentences.\n"
    "Do NOT explain.\n"
    "If uncertain, still output one of them.\n"
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
# Metrics
# -------------------------
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

    # balanced accuracy + balanced error
    tpr = tp / (tp + fn + eps)
    tnr = tn / (tn + fp + eps)
    bacc = 0.5 * (tpr + tnr)
    be = 1.0 - bacc

    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_acc": float(bacc),
        "balanced_error": float(be),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n": int(len(y_true)),
    }

# -------------------------
# Text parsing
# -------------------------
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

def _normalize_answer_text(s: str) -> str:
    s = (s or "").strip().lower()
    for ch in ["\"", "'", ".", ",", "!", "?", ":", ";", "\n", "\t", "(", ")", "[", "]", "{", "}"]:
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s

def _text_to_label_strict(s: str) -> Optional[int]:
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
    return None

def _text_to_label_relaxed(s: str) -> Optional[int]:
    t = _strip_code_fence(s or "").strip()
    low = t.lower()
    if "disease" in low or "abnormal" in low or "lesion" in low:
        return 1
    if re.search(r"\bnormal\b", low) and not re.search(r"\bdisease\b|\babnormal\b|\blesion\b", low):
        return 0
    return _text_to_label_strict(t)

def text_to_label(s: str, mode: str) -> Optional[int]:
    return _text_to_label_strict(s) if mode == "strict" else _text_to_label_relaxed(s)

# -------------------------
# Utils
# -------------------------
def to_device(x, target_device: torch.device):
    if torch.is_tensor(x):
        return x.to(target_device, non_blocking=True)
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k in ("paths", "clean_indices", "severities", "datasets", "corruptions"):
                out[k] = v
            else:
                out[k] = to_device(v, target_device)
        return out
    if isinstance(x, (list, tuple)):
        if len(x) > 0 and all(isinstance(v, str) for v in x):
            return x
        return type(x)(to_device(v, target_device) for v in x)
    return x

def _get_tokenizer_from_processor(processor, model_id: Optional[str] = None, trust_remote_code: bool = False):
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        return processor.tokenizer
    if model_id is not None:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    raise RuntimeError("Tokenizer not found and model_id not provided.")

def _decode_generated_text(tok, input_ids, sequences: torch.Tensor, max_new_tokens: int) -> List[str]:
    if input_ids is not None and torch.is_tensor(input_ids) and sequences.ndim == 2:
        in_len = int(input_ids.shape[1])
        if sequences.shape[1] > in_len:
            gen = sequences[:, in_len:]
            texts = tok.batch_decode(gen, skip_special_tokens=True)
            return [t.strip() for t in texts]

    tail = sequences[:, -max_new_tokens:] if sequences.ndim == 2 else sequences
    texts = tok.batch_decode(tail, skip_special_tokens=True)
    texts = [t.strip() for t in texts]

    def _empty(s: str):
        return len(_normalize_answer_text(s)) == 0

    if any(_empty(t) for t in texts):
        full = tok.batch_decode(sequences, skip_special_tokens=True)
        full = [t.strip() for t in full]
        out = []
        for f in full:
            parts = [p.strip() for p in re.split(r"[\n\r]+", f) if p.strip()]
            out.append(parts[-1] if parts else f.strip())
        texts = out
    return texts

def modality_from_dataset_name(name: str) -> str:
    n = (name or "").lower()
    if "oct" in n:
        return "oct"
    if "xray" in n or "chest" in n or "cxr" in n:
        return "xray"
    if "pneumonia" in n or "pneumoniamnist" in n:
        return "xray"
    if "fundus" in n:
        return "fundus"
    if "mri" in n or "brain" in n:
        return "mri"
    return "mri"

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
    common = dict(torch_dtype=torch_dtype, device_map=None, low_cpu_mem_usage=True)

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
            model_id, trust_remote_code=True, cache_dir=cache_dir,
            torch_dtype=torch_dtype, low_cpu_mem_usage=True, device_map=None
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
# CSV loader for this format
# -------------------------
def load_corrupt_csv(path: str, base_out_dir: str, corrupt_key: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    need = ["dataset", "severity", "clean_index", "binarylabel", "filepath", "clean_filepath", corrupt_key]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"CSV must contain column: {c}")

    def _fix_path(s: str) -> str:
        s = str(s)
        s = s.replace(
            r"C:\Users\hanna\Lectures\Research_Project\Codes\Dataset\vlm_prompt_dataset",
            base_out_dir
        )
        s = s.replace("\\", "/")
        return s

    df["filepath"] = df["filepath"].apply(_fix_path)
    df["clean_filepath"] = df["clean_filepath"].apply(_fix_path)

    df["dataset_norm"] = df["dataset"].astype(str).str.lower().map(modality_from_dataset_name)
    df["severity_int"] = pd.to_numeric(df["severity"], errors="coerce").fillna(-1).astype(int)
    df["clean_index_str"] = df["clean_index"].astype(str)
    df["binarylabel"] = pd.to_numeric(df["binarylabel"], errors="coerce").fillna(0).astype(int)

    df["corruption_name"] = df[corrupt_key].astype(str).fillna("unknown").str.lower()
    return df.reset_index(drop=True)

def build_clean_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cid, gg in df.groupby("clean_index_str"):
        r0 = gg.iloc[0]
        rows.append({
            "clean_index_str": r0["clean_index_str"],
            "dataset_norm": r0["dataset_norm"],
            "binarylabel": int(r0["binarylabel"]),
            "path": r0["clean_filepath"],
        })
    return pd.DataFrame(rows)

def build_art_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["path"] = out["filepath"]
    return out[["clean_index_str", "dataset_norm", "binarylabel", "severity_int", "path", "corruption_name"]].reset_index(drop=True)

def build_pair_index(df_art: pd.DataFrame, df_clean: pd.DataFrame) -> List[Tuple[str, dict, List[dict]]]:
    clean_map = {r["clean_index_str"]: r for r in df_clean.to_dict("records")}
    items = []
    for cid, gg in df_art.groupby("clean_index_str"):
        if cid not in clean_map:
            continue
        clean_row = clean_map[cid]
        arts = []
        for _, r in gg.sort_values("severity_int").iterrows():
            arts.append({
                "path": r["path"],
                "severity": int(r["severity_int"]),
                "dataset_norm": r["dataset_norm"],
                "binarylabel": int(r["binarylabel"]),
                "corruption_name": str(r["corruption_name"]),
            })
        if len(arts) == 0:
            continue
        items.append((cid, clean_row, arts))
    return items

# -------------------------
# Dataset + collate
# -------------------------
class SimpleImageDataset(Dataset):
    def __init__(self, rows: List[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(r["path"]).convert("RGB")
        mod = str(r["dataset_norm"])
        task_prompt = PROMPT_BY_DATASET.get(mod, PROMPT_BY_DATASET["mri"])
        text = SYSTEM_PROMPT_SHORT + "\n" + task_prompt
        return {
            "image": img,
            "input_text": text,
            "label": int(r["binarylabel"]),
            "path": str(r["path"]),
            "clean_index": str(r.get("clean_index_str", "")),
            "severity": r.get("severity_int", None),
            "dataset": mod,
            "corruption": str(r.get("corruption_name", "unknown")),
        }

def make_collate(processor, backend: str):
    def collate(batch):
        images = [b["image"] for b in batch]
        texts  = [b["input_text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

        paths  = [b["path"] for b in batch]
        cids   = [b["clean_index"] for b in batch]
        sevs   = [b["severity"] for b in batch]
        dsets  = [b["dataset"] for b in batch]
        cors   = [b["corruption"] for b in batch]

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
        out["clean_indices"] = cids
        out["severities"] = sevs
        out["datasets"] = dsets
        out["corruptions"] = cors
        return out
    return collate

# -------------------------
# Forward
# -------------------------
@torch.no_grad()
def forward_get_pred_and_text(base_model, processor, backend: str, batch: Dict[str, Any],
                              max_new_tokens: int, parse_mode: str):
    model_id = MODEL_ID_BY_BACKEND[backend]
    tok = _get_tokenizer_from_processor(processor, model_id=model_id,
                                        trust_remote_code=(backend == "internvl"))

    target_device = next(base_model.parameters()).device
    batch = to_device(batch, target_device)

    y_true = batch["labels_cls"].detach().cpu()

    d = dict(batch)
    for k in ["labels_cls", "paths", "clean_indices", "severities", "datasets", "corruptions"]:
        d.pop(k, None)

    if backend == "internvl" and device == "cuda":
        for k in ["pixel_values", "images", "image", "vision_x"]:
            if k in d and torch.is_tensor(d[k]):
                d[k] = d[k].to(dtype=torch.float32)
        autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=False)
    else:
        autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=(device == "cuda"))

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        num_beams=1,
        return_dict_in_generate=True,
    )
    if getattr(tok, "pad_token_id", None) is not None:
        gen_kwargs["pad_token_id"] = tok.pad_token_id
    if getattr(tok, "eos_token_id", None) is not None:
        gen_kwargs["eos_token_id"] = tok.eos_token_id

    with autocast_ctx:
        gen_out = base_model.generate(**d, **gen_kwargs)

    seq = gen_out.sequences
    input_ids = d.get("input_ids", None)
    texts = _decode_generated_text(tok, input_ids, seq, max_new_tokens=max_new_tokens)

    preds = []
    for t in texts:
        lab = text_to_label(t, mode=parse_mode)
        preds.append(float("nan") if lab is None else int(lab))
    y_pred = torch.tensor(preds, dtype=torch.float32)

    return y_true, y_pred, texts

# -------------------------
# Eval (single loader) with metadata arrays returned
# -------------------------
def eval_loader(base_model, processor, backend: str, loader: DataLoader,
                split_name: str, max_new_tokens: int, parse_mode: str):
    all_y, all_p = [], []
    recs = []

    sevs = []
    cors = []
    cids = []
    dsets = []

    for batch in loader:
        y, p, texts = forward_get_pred_and_text(base_model, processor, backend, batch,
                                                max_new_tokens=max_new_tokens, parse_mode=parse_mode)
        all_y.append(y)
        all_p.append(p)

        paths = batch["paths"]
        for i in range(len(paths)):
            recs.append({
                "split": split_name,
                "path": str(paths[i]),
                "clean_index": str(batch["clean_indices"][i]),
                "severity": batch["severities"][i],
                "corruption": str(batch["corruptions"][i]),
                "dataset": str(batch["datasets"][i]),
                "y_true": int(y[i].item()),
                "y_pred": None if not np.isfinite(float(p[i].item())) else int(p[i].item()),
                "raw_text": texts[i],
            })

        sevs.extend(batch["severities"])
        cors.extend(batch["corruptions"])
        cids.extend(batch["clean_indices"])
        dsets.extend(batch["datasets"])

    y_true = torch.cat(all_y, dim=0) if all_y else torch.tensor([], dtype=torch.long)
    y_pred = torch.cat(all_p, dim=0) if all_p else torch.tensor([], dtype=torch.float32)

    finite = torch.isfinite(y_pred)
    y_true2 = y_true[finite]
    y_pred2 = y_pred[finite].to(torch.int64)

    metrics = compute_binary_metrics_from_preds(y_true2, y_pred2) if int(finite.sum()) > 0 else {}
    return (
        metrics,
        int(finite.sum().item()),
        int((~finite).sum().item()),
        recs,
        y_true, y_pred, finite,
        np.array(sevs, dtype=object),
        np.array(cors, dtype=object),
    )

# -------------------------
# Metrics slicing helpers
# -------------------------
def _metrics_for_mask(y_true, y_pred, finite, mask_np: np.ndarray) -> dict:
    fin_np = finite.detach().cpu().numpy()
    ok = fin_np & mask_np
    if ok.sum() == 0:
        return {}
    yt = torch.tensor(y_true.detach().cpu().numpy()[ok], dtype=torch.long)
    yp = torch.tensor(y_pred.detach().cpu().numpy()[ok], dtype=torch.long)
    return compute_binary_metrics_from_preds(yt, yp)

def compute_metrics_by_severity_and_corruption(y_true, y_pred, finite, sevs_np, cors_np):
    out = {}
    all_sev = sorted(set([int(s) for s in sevs_np.tolist() if s is not None]))
    all_cor = sorted(set([str(c) for c in cors_np.tolist()]))

    for c in all_cor:
        out[c] = {
            "pooled": {},
            "by_severity": {},
        }
        mask_c = (cors_np == c)
        out[c]["pooled"] = _metrics_for_mask(y_true, y_pred, finite, mask_c)

        for s in all_sev:
            mask_cs = (cors_np == c) & (sevs_np == s)
            out[c]["by_severity"][str(s)] = _metrics_for_mask(y_true, y_pred, finite, mask_cs)

    return out, all_cor, all_sev

def compute_robustness_per_corruption(clean_metrics: dict, metrics_by_corruption: dict, severities: List[int]):
    """
    For each corruption c:
      rBE_c = mean_s (BE_{s,c} - BE_clean)
      dF1_c = mean_s (F1_clean - F1_{s,c})
    """
    be_clean = clean_metrics.get("balanced_error", None)
    f1_clean = clean_metrics.get("f1", None)

    per_c = {}
    for c, dd in metrics_by_corruption.items():
        be_diffs = []
        f1_drops = []
        for s in severities:
            ms = dd.get("by_severity", {}).get(str(s), {}) or {}
            if be_clean is not None and "balanced_error" in ms:
                be_diffs.append(float(ms["balanced_error"]) - float(be_clean))
            if f1_clean is not None and "f1" in ms:
                f1_drops.append(float(f1_clean) - float(ms["f1"]))

        per_c[c] = {}
        if len(be_diffs) > 0:
            per_c[c]["rBE_no_baseline_mean"] = float(np.mean(be_diffs))
            per_c[c]["rBE_no_baseline_by_severity"] = {str(severities[i]): float(be_diffs[i]) for i in range(len(be_diffs))}
        if len(f1_drops) > 0:
            per_c[c]["dF1_mean"] = float(np.mean(f1_drops))
            per_c[c]["dF1_by_severity"] = {str(severities[i]): float(f1_drops[i]) for i in range(len(f1_drops))}

    # overall averages across corruptions
    rbe_vals = [v.get("rBE_no_baseline_mean") for v in per_c.values() if v.get("rBE_no_baseline_mean") is not None]
    df1_vals = [v.get("dF1_mean") for v in per_c.values() if v.get("dF1_mean") is not None]

    overall = {}
    if len(rbe_vals) > 0:
        overall["rBE_over_corruptions"] = float(np.mean(rbe_vals))
    if len(df1_vals) > 0:
        overall["dF1_over_corruptions"] = float(np.mean(df1_vals))

    return per_c, overall

# -------------------------
# Paired robustness by corruption
# -------------------------
def eval_pairs_overall_and_by_corruption(base_model, processor, backend: str, pair_items,
                                         max_new_tokens: int, parse_mode: str):
    """
    For each clean_index:
      clean pred vs artifact preds (sev 0..4)
    We compute overall + by corruption (based on artifact corruption_name).
    If a clean_index has mixed corruptions (rare), it contributes to each relevant corruption bucket separately.
    """
    # overall accum
    overall = dict(total=0, used=0, skipped=0, flip_maj=0, flip_any=0, y=[], pc=[], pm=[], pw=[], per_pair=[])

    # by corruption accum
    byc = {}  # c -> same structure

    def _ensure(c):
        if c not in byc:
            byc[c] = dict(total=0, used=0, skipped=0, flip_maj=0, flip_any=0, y=[], pc=[], pm=[], pw=[], per_pair=[])
        return byc[c]

    def make_one_inputs(img: Image.Image, text: str):
        if backend in ["lingshu", "qwen3"]:
            messages = [[{
                "role": "user",
                "content": [{"type": "image", "image": img}, {"type": "text", "text": text}],
            }]]
            chat = processor.apply_chat_template(messages[0], tokenize=False, add_generation_prompt=True)
            return processor(text=[chat], images=[img], padding=True, return_tensors="pt")
        if backend == "internvl":
            image_tok = getattr(processor, "image_token", None) or "<image>"
            inline = f"{image_tok}\n{text}"
            return processor(text=[inline], images=[img], padding=True, return_tensors="pt")
        if backend == "medgemma":
            messages = [[{
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": text}],
            }]]
            chat = processor.apply_chat_template(messages[0], tokenize=False, add_generation_prompt=True)
            return processor(text=[chat], images=[[img]], padding=True, return_tensors="pt")
        return processor(text=[text], images=[img], padding=True, return_tensors="pt")

    for cid, clean_row, art_rows in pair_items:
        mod = str(clean_row["dataset_norm"])
        task_prompt = PROMPT_BY_DATASET.get(mod, PROMPT_BY_DATASET["mri"])
        full_text = SYSTEM_PROMPT_SHORT + "\n" + task_prompt
        y = int(clean_row["binarylabel"])

        # overall counts
        overall["total"] += 1

        # clean pred (shared)
        img_c = Image.open(clean_row["path"]).convert("RGB")
        inp_c = make_one_inputs(img_c, full_text)
        batch_c = dict(inp_c)
        batch_c["labels_cls"] = torch.tensor([y], dtype=torch.long)
        batch_c["paths"] = [clean_row["path"]]
        batch_c["clean_indices"] = [cid]
        batch_c["severities"] = ["clean"]
        batch_c["datasets"] = [mod]
        batch_c["corruptions"] = ["clean"]

        _, pred_c_t, text_c = forward_get_pred_and_text(base_model, processor, backend, batch_c,
                                                        max_new_tokens=max_new_tokens, parse_mode=parse_mode)
        pred_c = float(pred_c_t.item())
        txt_c = text_c[0]

        # group artifacts by corruption
        arts_by_c = {}
        for ar in art_rows:
            c = str(ar.get("corruption_name", "unknown"))
            arts_by_c.setdefault(c, []).append(ar)

        # Each corruption bucket evaluated separately (same clean pred)
        for c, rows_c in arts_by_c.items():
            bref = _ensure(c)
            bref["total"] += 1  # number of clean_indices considered for this corruption

            art_entries = []
            art_preds = []

            for ar in rows_c:
                img_a = Image.open(ar["path"]).convert("RGB")
                inp_a = make_one_inputs(img_a, full_text)
                batch_a = dict(inp_a)
                batch_a["labels_cls"] = torch.tensor([y], dtype=torch.long)
                batch_a["paths"] = [ar["path"]]
                batch_a["clean_indices"] = [cid]
                batch_a["severities"] = [ar["severity"]]
                batch_a["datasets"] = [mod]
                batch_a["corruptions"] = [c]

                _, pred_a_t, text_a = forward_get_pred_and_text(base_model, processor, backend, batch_a,
                                                                max_new_tokens=max_new_tokens, parse_mode=parse_mode)
                pa = float(pred_a_t.item())
                art_preds.append(pa)
                art_entries.append({
                    "path": str(ar["path"]),
                    "severity": int(ar["severity"]),
                    "corruption": c,
                    "y_true": y,
                    "y_pred": None if not np.isfinite(pa) else int(pa),
                    "raw_text": text_a[0],
                })

            art_f = [p for p in art_preds if np.isfinite(p)]
            if (not np.isfinite(pred_c)) or len(art_f) == 0:
                bref["skipped"] += 1
                bref["per_pair"].append({
                    "clean_index": cid,
                    "corruption": c,
                    "skipped": True,
                    "reason": "clean_nan_or_all_art_nan",
                    "clean": {"path": str(clean_row["path"]), "y_true": y,
                              "y_pred": None if not np.isfinite(pred_c) else int(pred_c),
                              "raw_text": txt_c},
                    "artifact": art_entries,
                })
                continue

            pc = int(pred_c)
            arts_i = [int(p) for p in art_f]

            pred_majority = 1 if (sum(arts_i) >= (len(arts_i) / 2)) else 0
            if y == 0:
                pred_worstgt = 1 if any(p == 1 for p in arts_i) else int(pred_majority)
            else:
                pred_worstgt = 0 if any(p == 0 for p in arts_i) else int(pred_majority)

            this_flip_majority = (pc != pred_majority)
            this_flip_any = any(pc != p for p in arts_i)

            bref["used"] += 1
            bref["flip_maj"] += int(this_flip_majority)
            bref["flip_any"] += int(this_flip_any)
            bref["y"].append(y)
            bref["pc"].append(pc)
            bref["pm"].append(int(pred_majority))
            bref["pw"].append(int(pred_worstgt))

            bref["per_pair"].append({
                "clean_index": cid,
                "corruption": c,
                "skipped": False,
                "clean": {"path": str(clean_row["path"]), "y_true": y, "y_pred": pc, "raw_text": txt_c},
                "artifact": art_entries,
                "agg": {
                    "pred_majority": int(pred_majority),
                    "pred_worst_gt": int(pred_worstgt),
                    "flip_majority": bool(this_flip_majority),
                    "flip_any": bool(this_flip_any),
                }
            })

        # also keep overall per_pair raw once (optional): we’ll store from byc only to avoid duplication

    def _finalize(bucket):
        used = int(bucket["used"])
        total = int(bucket["total"])
        skipped = int(bucket["skipped"])
        if used > 0:
            y_t = torch.tensor(bucket["y"], dtype=torch.long)
            pc  = torch.tensor(bucket["pc"], dtype=torch.long)
            pm  = torch.tensor(bucket["pm"], dtype=torch.long)
            pw  = torch.tensor(bucket["pw"], dtype=torch.long)
            return {
                "n_pairs_total_indexed": total,
                "n_pairs_used": used,
                "n_pairs_skipped_nan": skipped,
                "pair_use_rate": float(used / max(1, total)),
                "clean_metrics": compute_binary_metrics_from_preds(y_t, pc),
                "art_majority_metrics": compute_binary_metrics_from_preds(y_t, pm),
                "art_worst_gt_metrics": compute_binary_metrics_from_preds(y_t, pw),
                "flip_rate_majority": float(bucket["flip_maj"] / max(1, used)),
                "flip_rate_any": float(bucket["flip_any"] / max(1, used)),
            }
        return {
            "n_pairs_total_indexed": total,
            "n_pairs_used": 0,
            "n_pairs_skipped_nan": skipped,
            "pair_use_rate": 0.0,
            "clean_metrics": {},
            "art_majority_metrics": {},
            "art_worst_gt_metrics": {},
            "flip_rate_majority": 0.0,
            "flip_rate_any": 0.0,
        }

    byc_stats = {c: _finalize(b) for c, b in byc.items()}
    byc_pairs = {c: b["per_pair"] for c, b in byc.items()}

    # build overall by averaging over corruptions? (not weighted) — optional
    # Here we compute a weighted overall by aggregating all used pairs across all corruptions.
    all_y, all_pc, all_pm, all_pw = [], [], [], []
    flip_maj, flip_any = 0, 0
    total_cnt, used_cnt, skipped_cnt = 0, 0, 0
    for c, b in byc.items():
        total_cnt += int(b["total"])
        used_cnt += int(b["used"])
        skipped_cnt += int(b["skipped"])
        flip_maj += int(b["flip_maj"])
        flip_any += int(b["flip_any"])
        all_y.extend(b["y"])
        all_pc.extend(b["pc"])
        all_pm.extend(b["pm"])
        all_pw.extend(b["pw"])

    if used_cnt > 0:
        y_t = torch.tensor(all_y, dtype=torch.long)
        pc  = torch.tensor(all_pc, dtype=torch.long)
        pm  = torch.tensor(all_pm, dtype=torch.long)
        pw  = torch.tensor(all_pw, dtype=torch.long)
        overall_stats = {
            "n_pairs_total_indexed": int(total_cnt),
            "n_pairs_used": int(used_cnt),
            "n_pairs_skipped_nan": int(skipped_cnt),
            "pair_use_rate": float(used_cnt / max(1, total_cnt)),
            "clean_metrics": compute_binary_metrics_from_preds(y_t, pc),
            "art_majority_metrics": compute_binary_metrics_from_preds(y_t, pm),
            "art_worst_gt_metrics": compute_binary_metrics_from_preds(y_t, pw),
            "flip_rate_majority": float(flip_maj / max(1, used_cnt)),
            "flip_rate_any": float(flip_any / max(1, used_cnt)),
        }
    else:
        overall_stats = {
            "n_pairs_total_indexed": int(total_cnt),
            "n_pairs_used": 0,
            "n_pairs_skipped_nan": int(skipped_cnt),
            "pair_use_rate": 0.0,
            "clean_metrics": {},
            "art_majority_metrics": {},
            "art_worst_gt_metrics": {},
            "flip_rate_majority": 0.0,
            "flip_rate_any": 0.0,
        }

    return overall_stats, byc_stats, byc_pairs

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
    backends = ["qwen3", "medgemma", "internvl", "lingshu"] if args.backend == "all" else [args.backend]

    df = load_corrupt_csv(args.test_csv, args.base_out_dir, args.corrupt_key)
    df_clean = build_clean_df(df)        # unique clean per clean_index
    df_art   = build_art_df(df)          # all artifacts (sev 0..4)
    pair_items = build_pair_index(df_art, df_clean)

    severities_present = sorted(set(df_art["severity_int"].tolist()))
    corruptions_present = sorted(set(df_art["corruption_name"].tolist()))
    print("Severities present:", severities_present, flush=True)
    print("Corruptions present:", corruptions_present[:10], ("..." if len(corruptions_present) > 10 else ""), flush=True)

    for backend in backends:
        model_id = MODEL_ID_BY_BACKEND[backend]
        print("\n==============================", flush=True)
        print(f"== backend: {backend} | parse_mode: {args.parse_mode}", flush=True)
        print(f"== corrupt_key: {args.corrupt_key}", flush=True)
        print("==============================", flush=True)

        base_model, processor = load_backend(backend, model_id)

        collate = make_collate(processor, backend)

        # ---- CLEAN ----
        clean_rows = df_clean.to_dict("records")
        ds_clean = SimpleImageDataset([{
            "path": r["path"],
            "dataset_norm": r["dataset_norm"],
            "binarylabel": r["binarylabel"],
            "clean_index_str": r["clean_index_str"],
            "severity_int": "clean",
            "corruption_name": "clean"
        } for r in clean_rows])
        dl_clean = DataLoader(ds_clean, batch_size=args.bs, shuffle=False,
                              collate_fn=collate, pin_memory=(device == "cuda"))

        m_clean, n_ok_clean, n_nan_clean, per_clean, *_ = eval_loader(
            base_model, processor, backend, dl_clean,
            split_name="clean", max_new_tokens=args.max_new_tokens, parse_mode=args.parse_mode
        )

        # ---- ARTIFACT (ALL) ----
        art_rows = df_art.to_dict("records")
        ds_art = SimpleImageDataset([{
            "path": r["path"],
            "dataset_norm": r["dataset_norm"],
            "binarylabel": r["binarylabel"],
            "clean_index_str": r["clean_index_str"],
            "severity_int": int(r["severity_int"]),
            "corruption_name": str(r["corruption_name"]),
        } for r in art_rows])
        dl_art = DataLoader(ds_art, batch_size=args.bs, shuffle=False,
                            collate_fn=collate, pin_memory=(device == "cuda"))

        m_art, n_ok_art, n_nan_art, per_art, y_true_art, y_pred_art, finite_art, sevs_np, cors_np = eval_loader(
            base_model, processor, backend, dl_art,
            split_name="artifact_all", max_new_tokens=args.max_new_tokens, parse_mode=args.parse_mode
        )

        # ---- CORRUPTION + SEVERITY metrics ----
        metrics_by_corruption, all_cor, all_sev = compute_metrics_by_severity_and_corruption(
            y_true_art, y_pred_art, finite_art, sevs_np, cors_np
        )

        # also severity-only (pooled over corruptions)
        metrics_by_severity_pooled = {}
        for s in all_sev:
            mask_s = (sevs_np == s)
            metrics_by_severity_pooled[str(s)] = _metrics_for_mask(y_true_art, y_pred_art, finite_art, mask_s)

        # ---- Robustness scores per corruption + overall ----
        per_corrupt_scores, overall_scores = compute_robustness_per_corruption(
            clean_metrics=m_clean,
            metrics_by_corruption=metrics_by_corruption,
            severities=all_sev
        )

        # ---- Paired robustness overall + by corruption ----
        paired_overall, paired_by_corruption, paired_pairs_by_corruption = eval_pairs_overall_and_by_corruption(
            base_model, processor, backend, pair_items,
            max_new_tokens=args.max_new_tokens, parse_mode=args.parse_mode
        )

        results = {
            "backend": backend,
            "parse_mode": args.parse_mode,
            "max_new_tokens": int(args.max_new_tokens),
            "corrupt_key": args.corrupt_key,

            "clean": {
                "n_clean_total": int(len(df_clean)),
                "n_clean_used_finite": int(n_ok_clean),
                "n_clean_nan": int(n_nan_clean),
                "metrics": m_clean,
            },

            "artifact": {
                "n_art_total": int(len(df_art)),
                "n_art_used_finite": int(n_ok_art),
                "n_art_nan": int(n_nan_art),
                "metrics_all_severities": m_art,
                "metrics_by_severity_pooled": metrics_by_severity_pooled,      # severity only
                "metrics_by_corruption": metrics_by_corruption,                # corruption -> pooled + by severity
            },

            "robustness_scores": {
                "per_corruption": per_corrupt_scores,
                "overall": overall_scores,
            },

            "paired": {
                "overall": paired_overall,
                "by_corruption": paired_by_corruption,
            },

            "per_image_outputs": {
                "clean": per_clean,
                "artifact": per_art,
                "paired_by_corruption": paired_pairs_by_corruption,  # (큰 파일 될 수 있음)
            }
        }

        print("\n====== Summary ======")
        print(f"clean f1={results['clean']['metrics'].get('f1', None)} | clean BE={results['clean']['metrics'].get('balanced_error', None)}")
        print(f"art   f1={results['artifact']['metrics_all_severities'].get('f1', None)} | art BE={results['artifact']['metrics_all_severities'].get('balanced_error', None)}")
        print(f"overall rBE_over_corruptions={results['robustness_scores']['overall'].get('rBE_over_corruptions', None)}")
        print(f"overall dF1_over_corruptions={results['robustness_scores']['overall'].get('dF1_over_corruptions', None)}")
        print(f"paired flip_majority={results['paired']['overall'].get('flip_rate_majority', None)}")

        if args.out_json:
            out_path = _add_backend_suffix(args.out_json, backend) if args.backend == "all" else args.out_json
            _safe_makedirs_for_file(out_path)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"[saved] {out_path}")

        del base_model, processor
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

if __name__ == "__main__":
    main()
