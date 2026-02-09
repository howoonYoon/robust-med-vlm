#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VLM-only evaluation (NO adapter/classifier, NO .pt loading)

- Single-image metrics:
  * Clean test (e.g., 540 images)
  * Artefact/weak test (e.g., 450 images from pair csv)
  * AUROC/Acc/Precision/Recall/F1 using a "disease score" extracted from generation logits

- Paired robustness (e.g., 90 fileindex groups):
  * clean score (1 image)
  * weak scores (5 images)
  * aggregated weak score by mean / worst(min)
  * AUROC on (clean 90), (weak mean 90), (weak worst 90)
  * flip-rate vs threshold
  * delta score stats (clean - weak)

Important:
- Score is computed from FIRST generated token probability for ["disease","normal"] (approx).
  If your model tends to generate other words first, scores may be noisy.
"""

import os
import gc
import json
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
                   choices=["qwen3", "medgemma", "internvl", "lingshu"])
    p.add_argument("--test_clean_csv", type=str, required=True,
                   help="CSV containing clean-only test set (e.g., 540 images)")
    p.add_argument("--test_pair_csv", type=str, required=True,
                   help="CSV containing clean+weak test set (90 fileindex groups, each clean 1 + weak 5)")
    p.add_argument("--bs", type=int, default=1)
    #p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--max_new_tokens", type=int, default=3)
    p.add_argument("--out_json", type=str, default=None)
    return p.parse_args()

args = parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# HF cache (optional but matches your setup)
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
SYSTEM_PROMPT_SHORT = 'Answer with ONE WORD: "normal" or "disease".'

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
            if k in ("paths", "path", "meta", "severities", "fileindices"):
                out[k] = v
            else:
                out[k] = to_device(v, target_device)
        return out

    if isinstance(x, (list, tuple)):
        if len(x) > 0 and all(isinstance(v, str) for v in x):
            return x
        return type(x)(to_device(v, target_device) for v in x)

    return x


# def compute_binary_metrics(y_true: torch.Tensor, y_score: torch.Tensor, thr: float = 0.5):
#     y_true = y_true.detach().cpu().to(torch.int64)
#     y_score = y_score.detach().cpu().to(torch.float32)

#     y_pred = (y_score >= thr).to(torch.int64)

#     tp = int(((y_pred == 1) & (y_true == 1)).sum().item())
#     tn = int(((y_pred == 0) & (y_true == 0)).sum().item())
#     fp = int(((y_pred == 1) & (y_true == 0)).sum().item())
#     fn = int(((y_pred == 0) & (y_true == 1)).sum().item())

#     eps = 1e-12
#     acc = (tp + tn) / (tp + tn + fp + fn + eps)
#     precision = tp / (tp + fp + eps)
#     recall = tp / (tp + fn + eps)
#     f1 = 2 * precision * recall / (precision + recall + eps)

#     # AUROC (Mann–Whitney U)
#     pos = (y_true == 1)
#     neg = (y_true == 0)
#     if int(pos.sum()) == 0 or int(neg.sum()) == 0:
#         auroc = float("nan")
#     else:
#         order = torch.argsort(y_score)  # ascending
#         ranks = torch.empty_like(order, dtype=torch.float32)
#         ranks[order] = torch.arange(1, len(y_score) + 1, dtype=torch.float32)
#         sum_ranks_pos = ranks[pos].sum()
#         n_pos = float(pos.sum().item())
#         n_neg = float(neg.sum().item())
#         u = sum_ranks_pos - n_pos * (n_pos + 1.0) / 2.0
#         auroc = float((u / (n_pos * n_neg)).item())

#     return {
#         "acc": float(acc),
#         "precision": float(precision),
#         "recall": float(recall),
#         "f1": float(f1),
#         "auroc": float(auroc),
#         "tp": tp, "tn": tn, "fp": fp, "fn": fn,
#         "n": int(len(y_true)),
#     }


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



def _get_tokenizer_from_processor(processor):
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        return processor.tokenizer
    raise RuntimeError("processor.tokenizer missing; cannot decode or map tokens.")


def _normalize_answer_text(s: str) -> str:
    s = (s or "").strip().lower()
    for ch in ["\"", "'", ".", ",", "!", "?", ":", ";", "\n", "\t"]:
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s


def _first_token_id(tok, candidates: List[str]) -> Optional[int]:
    for w in candidates:
        ids = tok.encode(w, add_special_tokens=False)
        if len(ids) >= 1:
            return ids[0]
    return None

def _text_to_label(s: str) -> Optional[int]:
    s = _normalize_answer_text(s)
    # 첫 단어만 보고 싶으면:
    first = s.split()[0] if len(s.split()) > 0 else ""

    if first == "disease":
        return 1
    if first == "normal":
        return 0

    # 가끔 "disease." "normal." 같은 변형이 있을 수 있으니 여유있게:
    if "disease" in s and "normal" not in s:
        return 1
    if "normal" in s and "disease" not in s:
        return 0

    return None  # 못 알아먹는 답



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
        model.to(device); model.eval()
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

    df["severity_is_clean"] = df["severity"].isna()
    df["severity_is_weak"]  = df["severity"].fillna("").astype(str).str.lower().eq("weak")
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
        full_text = (self.system_prompt + "\n\n" + task_prompt) if self.system_prompt else task_prompt

        return {
            "image": img,
            "input_text": full_text,
            "label": label,
            "path": img_path,
            "severity": sev,
            "fileindex": str(row["fileindex"]),
        }


def make_single_collate_fn(processor, backend: str):
    def collate(batch):
        images = [b["image"] for b in batch]
        texts  = [b["input_text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        paths  = [b["path"] for b in batch]
        sevs   = [b["severity"] for b in batch]
        fids   = [str(b["fileindex"]) for b in batch]

        if backend in ["lingshu", "qwen3"]:
            messages_list = []
            for img, txt in zip(images, texts):
                messages_list.append([{
                    "role": "user",
                    "content": [{"type": "image", "image": img}, {"type": "text", "text": txt}],
                }])
            chat_texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                         for m in messages_list]
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
            chat_texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                         for m in messages_list]
            images_nested = [[img] for img in images]
            model_inputs = processor(text=chat_texts, images=images_nested, padding=True, return_tensors="pt")

        else:
            model_inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")

        out = dict(model_inputs)
        out["labels_cls"] = labels
        out["paths"] = paths
        out["severities"] = sevs
        out["fileindices"] = fids
        return out
    return collate


def build_pair_index(df_pair: pd.DataFrame) -> List[Tuple[int, pd.Series, List[pd.Series]]]:
    """
    Returns list of tuples:
      (fileindex, clean_row, [weak_rows...])
    Expects exactly 1 clean and >=1 weak per fileindex.
    """
    items = []
    for fid, g in df_pair.groupby("fileindex"):
        g = g.reset_index(drop=True)
        clean_rows = g[g["severity_norm"] == "clean"]
        weak_rows  = g[g["severity_norm"] == "weak"]
        if len(clean_rows) == 0 or len(weak_rows) == 0:
            continue
        clean_row = clean_rows.iloc[0]
        weak_list = [weak_rows.iloc[i] for i in range(len(weak_rows))]
        items.append((str(fid), clean_row, weak_list))
    return items


# -------------------------
# VLM-only forward
# -------------------------
# @torch.no_grad()
# def forward_vlm_only_get_prob(base_model, processor, backend: str, batch: Dict[str, Any],
#                               max_new_tokens: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Returns (y_true, p_disease) where p_disease in [0,1] (approx from first generated token).
#     """
#     tok = _get_tokenizer_from_processor(processor)

#     target_device = next(base_model.parameters()).device
#     batch = to_device(batch, target_device)

#     y_true = batch["labels_cls"].detach().cpu()

#     d = dict(batch)
#     for k in ["labels_cls", "paths", "severities", "fileindices", "labels_token"]:
#         d.pop(k, None)

#     # internvl 안정화 유지
#     if backend == "internvl" and device == "cuda":
#         for k in ["pixel_values", "images", "image", "vision_x"]:
#             if k in d and torch.is_tensor(d[k]):
#                 d[k] = d[k].to(dtype=torch.float32)
#         autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=False)
#     else:
#         autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=(device == "cuda"))

#     with autocast_ctx:
#         gen_out = base_model.generate(
#             **d,
#             max_new_tokens=max_new_tokens,
#             do_sample=False,
#             return_dict_in_generate=True,
#             output_scores=True,
#         )

#     # first-step score extraction (approx)
#     scores0 = gen_out.scores[0].detach().float()  # [B, V]
#     p_vocab = torch.softmax(scores0, dim=-1)

#     id_d = _first_token_id(tok, [" disease", "disease", " Disease", "D"])
#     id_n = _first_token_id(tok, [" normal", "normal", " Normal", "N"])

#     if id_d is None or id_n is None:
#         p_disease = torch.full((scores0.shape[0],), float("nan"))
#     else:
#         pd = p_vocab[:, id_d]
#         pn = p_vocab[:, id_n]
#         p_disease = (pd / (pd + pn + 1e-12)).detach().cpu()

#     return y_true, p_disease

@torch.no_grad()
def forward_vlm_only_get_pred(base_model, processor, backend: str, batch: Dict[str, Any],
                              max_new_tokens: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (y_true, y_pred) where y_pred is 0/1 from generated text ("normal"/"disease").
    Unparseable -> NaN (will be filtered out).
    """
    tok = _get_tokenizer_from_processor(processor)

    target_device = next(base_model.parameters()).device
    batch = to_device(batch, target_device)

    y_true = batch["labels_cls"].detach().cpu()

    d = dict(batch)
    for k in ["labels_cls", "paths", "severities", "fileindices", "labels_token"]:
        d.pop(k, None)

    if backend == "internvl" and device == "cuda":
        for k in ["pixel_values", "images", "image", "vision_x"]:
            if k in d and torch.is_tensor(d[k]):
                d[k] = d[k].to(dtype=torch.float32)
        autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=False)
    else:
        autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=(device == "cuda"))

    with autocast_ctx:
        gen_out = base_model.generate(
            **d,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
        )

    # 새로 생성된 토큰만 decode
    if "input_ids" in d and torch.is_tensor(d["input_ids"]):
        in_len = int(d["input_ids"].shape[1])
    else:
        # 백엔드에 따라 input_ids가 없을 수도 있음 -> 전체 decode 후 prompt 제거가 어려워짐
        # 웬만하면 input_ids가 있음. 없으면 그냥 전체 decode를 쓴다.
        in_len = 0

    seq = gen_out.sequences  # [B, T_total]
    new_tokens = seq[:, in_len:] if in_len > 0 else seq
    texts = tok.batch_decode(new_tokens, skip_special_tokens=True)

    preds = []
    for t in texts:
        lab = _text_to_label(t)
        preds.append(float("nan") if lab is None else int(lab))

    y_pred = torch.tensor(preds, dtype=torch.float32)  # NaN 포함 가능
    return y_true, y_pred


# def eval_single_loader_vlm_only(base_model, processor, backend: str, loader: DataLoader, thr: float, max_new_tokens: int):
#     all_y, all_p = [], []
#     all_sev, all_fid = [], []

#     for batch in loader:
#         y, p = forward_vlm_only_get_prob(base_model, processor, backend, batch, max_new_tokens=max_new_tokens)
#         all_y.append(y)
#         all_p.append(p)
#         all_sev.extend(batch["severities"])
#         all_fid.extend(batch["fileindices"])

#     y_true = torch.cat(all_y, dim=0)
#     y_prob = torch.cat(all_p, dim=0)

#     finite = torch.isfinite(y_prob)
#     y_true2 = y_true[finite]
#     y_prob2 = y_prob[finite]

#     m_all = compute_binary_metrics(y_true2, y_prob2, thr=thr)

#     all_sev = np.array(all_sev)
#     sev2 = all_sev[finite.numpy()]
#     clean_mask = sev2 == "clean"
#     weak_mask  = sev2 == "weak"

#     m_clean = compute_binary_metrics(y_true2[clean_mask], y_prob2[clean_mask], thr=thr) if clean_mask.sum() > 0 else {}
#     m_weak  = compute_binary_metrics(y_true2[weak_mask],  y_prob2[weak_mask],  thr=thr) if weak_mask.sum() > 0 else {}

#     return m_all, m_clean, m_weak, y_true2, y_prob2, int(finite.sum().item()), int((~finite).sum().item())


def eval_single_loader_vlm_only_text(base_model, processor, backend: str, loader: DataLoader, max_new_tokens: int):
    all_y, all_pred = [], []
    all_sev = []

    for batch in loader:
        y, pred = forward_vlm_only_get_pred(base_model, processor, backend, batch, max_new_tokens=max_new_tokens)
        all_y.append(y)
        all_pred.append(pred)
        all_sev.extend(batch["severities"])

    y_true = torch.cat(all_y, dim=0)
    y_pred = torch.cat(all_pred, dim=0)

    finite = torch.isfinite(y_pred)
    y_true2 = y_true[finite]
    y_pred2 = y_pred[finite].to(torch.int64)

    m_all = compute_binary_metrics_from_preds(y_true2, y_pred2)

    all_sev = np.array(all_sev)
    sev2 = all_sev[finite.numpy()]
    clean_mask = sev2 == "clean"
    weak_mask  = sev2 == "weak"

    m_clean = compute_binary_metrics_from_preds(y_true2[clean_mask], y_pred2[clean_mask]) if clean_mask.sum() > 0 else {}
    m_weak  = compute_binary_metrics_from_preds(y_true2[weak_mask],  y_pred2[weak_mask])  if weak_mask.sum() > 0 else {}

    return m_all, m_clean, m_weak, int(finite.sum().item()), int((~finite).sum().item())


def eval_pairs_vlm_only(base_model, processor, backend: str,
                        pair_items: List[Tuple[int, pd.Series, List[pd.Series]]],
                        max_new_tokens: int):
    """
    Evaluate paired groups:
      - clean score (one image)
      - weak score list (k images)
    Compute AUROC on clean(90) and aggregated weak(90) by mean/worst.
    Compute flip rate vs threshold.
    """

    def make_one_batch(img: Image.Image, text: str, label: int):
        if backend in ["lingshu", "qwen3"]:
            messages = [[{
                "role": "user",
                "content": [{"type": "image", "image": img}, {"type": "text", "text": text}],
            }]]
            chat_text = processor.apply_chat_template(messages[0], tokenize=False, add_generation_prompt=False)
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
            chat_text = processor.apply_chat_template(messages[0], tokenize=False, add_generation_prompt=False)
            model_inputs = processor(text=[chat_text], images=[[img]], padding=True, return_tensors="pt")

        else:
            model_inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt")

        out = dict(model_inputs)
        out["labels_cls"] = torch.tensor([label], dtype=torch.long)
        out["paths"] = ["<pair>"]
        out["severities"] = ["<pair>"]
        out["fileindices"] = [0]
        return out
    
    n_pairs = 0
    n_skipped = 0
    flip_mean = 0
    flip_worst = 0

    # clean GT 기반 성능(weak-mean 예측을 GT랑 비교)
    y_list = []
    pred_clean_list = []
    pred_mean_list = []
    pred_worst_list = []

    for fid, clean_row, weak_rows in pair_items:
        modality = str(clean_row["dataset_norm"]).lower()
        task_prompt = PROMPT_BY_DATASET.get(modality, PROMPT_BY_DATASET["mri"])
        full_text = SYSTEM_PROMPT_SHORT + "\n\n" + task_prompt

        y = int(clean_row["binarylabel"])

        # clean pred
        img_c = Image.open(clean_row["filepath"]).convert("RGB")
        batch_c = make_one_batch(img_c, full_text, y)
        _, pred_c = forward_vlm_only_get_pred(base_model, processor, backend, batch_c, max_new_tokens=max_new_tokens)
        pred_c = float(pred_c.item())

        weak_preds = []
        for wr in weak_rows:
            img_w = Image.open(wr["filepath"]).convert("RGB")
            batch_w = make_one_batch(img_w, full_text, y)  # label은 GT(=clean의 label)로 통일해도 됨
            _, pred_w = forward_vlm_only_get_pred(base_model, processor, backend, batch_w, max_new_tokens=max_new_tokens)
            weak_preds.append(float(pred_w.item()))

        # NaN 있으면 skip
        if (not np.isfinite(pred_c)) or (not np.all(np.isfinite(weak_preds))):
            n_skipped += 1
            continue
        n_pairs += 1  # ✅ 이거 추가

        pred_c_i = int(pred_c)
        weak_preds_i = [int(p) for p in weak_preds]
        # mean = majority vote
        pred_mean = 1 if (sum(weak_preds_i) >= (len(weak_preds_i) / 2)) else 0

        # worst = any weak says disease
        pred_worst = 1 if any(p == 1 for p in weak_preds_i) else 0
        
        # flip rates
        if pred_c_i != pred_mean:
            flip_mean += 1
        if pred_c_i != pred_worst:
            flip_worst += 1


        y_list.append(y)
        pred_clean_list.append(pred_c_i)
        pred_mean_list.append(pred_mean)
        pred_worst_list.append(pred_worst)

    # paired 성능: GT(y) vs 예측(pred_*)
    y_t = torch.tensor(y_list, dtype=torch.long)
    pc  = torch.tensor(pred_clean_list, dtype=torch.long)
    pm  = torch.tensor(pred_mean_list, dtype=torch.long)
    pw  = torch.tensor(pred_worst_list, dtype=torch.long)

    out = {
        "n_pairs_used": int(n_pairs),
        "n_pairs_skipped_nan": int(n_skipped),

        "clean_metrics": compute_binary_metrics_from_preds(y_t, pc) if n_pairs else {},
        "art_mean_metrics": compute_binary_metrics_from_preds(y_t, pm) if n_pairs else {},
        "art_worst_metrics": compute_binary_metrics_from_preds(y_t, pw) if n_pairs else {},

        "flip_rate_mean": float(flip_mean / max(1, n_pairs)),
        "flip_rate_worst": float(flip_worst / max(1, n_pairs)),
    }
    return out

    # y_clean_list, p_clean_list = [], []
    # y_mean_list,  p_mean_list  = [], []
    # y_worst_list, p_worst_list = [], []

    # flip_mean = 0
    # flip_worst = 0
    # n_pairs = 0
    # n_skipped = 0

    # for fid, clean_row, weak_rows in pair_items:
    #     # prompt (use clean row modality)
    #     modality = str(clean_row["dataset_norm"]).lower()
    #     task_prompt = PROMPT_BY_DATASET.get(modality, PROMPT_BY_DATASET["mri"])
    #     full_text = SYSTEM_PROMPT_SHORT + "\n\n" + task_prompt

    #     img_c = Image.open(clean_row["filepath"]).convert("RGB")
    #     y_c = int(clean_row["binarylabel"])

    #     batch_c = make_one_batch(img_c, full_text, y_c)
    #     y_c_t, p_c_t = forward_vlm_only_get_prob(base_model, processor, backend, batch_c, max_new_tokens=max_new_tokens)
    #     p_clean = float(p_c_t.item())

    #     p_ws = []
    #     for wr in weak_rows:
    #         img_w = Image.open(wr["filepath"]).convert("RGB")
    #         y_w = int(wr["binarylabel"])
    #         batch_w = make_one_batch(img_w, full_text, y_w)
    #         _, p_w_t = forward_vlm_only_get_prob(base_model, processor, backend, batch_w, max_new_tokens=max_new_tokens)
    #         p_ws.append(float(p_w_t.item()))

    #     # if any NaN -> skip pair (keeps metrics clean)
    #     if (not np.isfinite(p_clean)) or (not np.all(np.isfinite(p_ws))):
    #         n_skipped += 1
    #         continue

    #     n_pairs += 1
    #     p_mean = float(np.mean(p_ws))
    #     p_worst = float(np.min(p_ws))

    #     y_clean_list.append(y_c); p_clean_list.append(p_clean)
    #     y_mean_list.append(y_c);  p_mean_list.append(p_mean)
    #     y_worst_list.append(y_c); p_worst_list.append(p_worst)

    #     pred_c    = 1 if p_clean >= thr else 0
    #     pred_mean = 1 if p_mean  >= thr else 0
    #     pred_wrst = 1 if p_worst >= thr else 0
    #     if pred_c != pred_mean:
    #         flip_mean += 1
    #     if pred_c != pred_wrst:
    #         flip_worst += 1

    # y_clean = torch.tensor(y_clean_list, dtype=torch.long)
    # p_clean = torch.tensor(p_clean_list, dtype=torch.float32)
    # y_mean  = torch.tensor(y_mean_list,  dtype=torch.long)
    # p_mean  = torch.tensor(p_mean_list,  dtype=torch.float32)
    # y_worst = torch.tensor(y_worst_list, dtype=torch.long)
    # p_worst = torch.tensor(p_worst_list, dtype=torch.float32)

    # m_clean = compute_binary_metrics(y_clean, p_clean, thr=thr) if len(y_clean) else {"auroc": float("nan"), "n": 0}
    # m_mean  = compute_binary_metrics(y_mean,  p_mean,  thr=thr) if len(y_mean)  else {"auroc": float("nan"), "n": 0}
    # m_worst = compute_binary_metrics(y_worst, p_worst, thr=thr) if len(y_worst) else {"auroc": float("nan"), "n": 0}

    # p_clean_np = np.array(p_clean_list, dtype=np.float32)
    # p_mean_np  = np.array(p_mean_list,  dtype=np.float32)
    # p_worst_np = np.array(p_worst_list, dtype=np.float32)
    # y_np       = np.array(y_clean_list, dtype=np.int64)

    # dp_mean  = p_clean_np - p_mean_np
    # dp_worst = p_clean_np - p_worst_np

    # pos = (y_np == 1)

    # out = {
    #     "n_pairs_used": int(n_pairs),
    #     "n_pairs_skipped_nan": int(n_skipped),
    #     "auroc_pair_clean": float(m_clean.get("auroc", float("nan"))),
    #     "auroc_pair_art_mean": float(m_mean.get("auroc", float("nan"))),
    #     "drop_mean": float(m_mean.get("auroc", float("nan")) - m_clean.get("auroc", float("nan")))
    #                  if (np.isfinite(m_mean.get("auroc", np.nan)) and np.isfinite(m_clean.get("auroc", np.nan))) else float("nan"),
    #     "auroc_pair_art_worst": float(m_worst.get("auroc", float("nan"))),
    #     "drop_worst": float(m_worst.get("auroc", float("nan")) - m_clean.get("auroc", float("nan")))
    #                   if (np.isfinite(m_worst.get("auroc", np.nan)) and np.isfinite(m_clean.get("auroc", np.nan))) else float("nan"),
    #     "flip_rate_mean": float(flip_mean / max(1, n_pairs)),
    #     "flip_rate_worst": float(flip_worst / max(1, n_pairs)),
    #     "dp_mean_avg": float(np.mean(dp_mean)) if len(dp_mean) else float("nan"),
    #     "dp_worst_avg": float(np.mean(dp_worst)) if len(dp_worst) else float("nan"),
    #     "dp_mean_pos_avg": float(np.mean(dp_mean[pos])) if pos.any() else float("nan"),
    #     "dp_worst_pos_avg": float(np.mean(dp_worst[pos])) if pos.any() else float("nan"),
    # }
    # return out


# -------------------------
# Main
# -------------------------
def main():
    backend = args.backend
    model_id = MODEL_ID_BY_BACKEND[backend]

    print("\n==============================")
    print(f"== VLM-ONLY EVAL backend: {backend}")
    print(f"== test_clean_csv: {args.test_clean_csv}")
    print(f"== test_pair_csv : {args.test_pair_csv}")
    #print(f"== thr={args.thr} bs={args.bs} max_new_tokens={args.max_new_tokens}")
    print("==============================")

    base_model, processor = load_backend(backend, model_id)

    # ---- SINGLE IMAGE TESTS ----
    df_clean = load_split_csv(args.test_clean_csv, BASE_OUT_DIR)
    df_clean = df_clean[df_clean["severity_norm"] == "clean"].reset_index(drop=True)

    df_pair = load_split_csv(args.test_pair_csv, BASE_OUT_DIR)
    df_pair_weak  = df_pair[df_pair["severity_norm"] == "weak"].reset_index(drop=True)

    ds_clean = SingleImageDataset(df_clean, PROMPT_BY_DATASET, SYSTEM_PROMPT_SHORT)
    ds_weak  = SingleImageDataset(df_pair_weak, PROMPT_BY_DATASET, SYSTEM_PROMPT_SHORT)

    collate = make_single_collate_fn(processor, backend)
    dl_clean = DataLoader(ds_clean, batch_size=args.bs, shuffle=False, collate_fn=collate, pin_memory=(device == "cuda"))
    dl_weak  = DataLoader(ds_weak,  batch_size=args.bs, shuffle=False, collate_fn=collate, pin_memory=(device == "cuda"))

    # m_clean_all, _, _, _, _, n_ok_clean, n_nan_clean = eval_single_loader_vlm_only(
    #     base_model, processor, backend, dl_clean, thr=args.thr, max_new_tokens=args.max_new_tokens
    # )
    # m_weak_all,  _, _, _, _, n_ok_weak,  n_nan_weak  = eval_single_loader_vlm_only(
    #     base_model, processor, backend, dl_weak,  thr=args.thr, max_new_tokens=args.max_new_tokens
    # )

    m_clean_all, m_clean, _, n_ok_clean, n_nan_clean = eval_single_loader_vlm_only_text(
    base_model, processor, backend, dl_clean, max_new_tokens=args.max_new_tokens
    )
    m_weak_all,  _, m_weak, n_ok_weak,  n_nan_weak  = eval_single_loader_vlm_only_text(
        base_model, processor, backend, dl_weak,  max_new_tokens=args.max_new_tokens
    )


    # auroc_clean = m_clean_all["auroc"]
    # auroc_art   = m_weak_all["auroc"]
    # auroc_macro = (auroc_clean + auroc_art) / 2.0 if (np.isfinite(auroc_clean) and np.isfinite(auroc_art)) else float("nan")

    # ---- PAIRED ROBUSTNESS ----
    pair_items = build_pair_index(df_pair)
    pair_stats = eval_pairs_vlm_only(
        base_model=base_model,
        processor=processor,
        backend=backend,
        pair_items=pair_items,
        max_new_tokens=args.max_new_tokens,
    )

    results = {
        "backend": backend,
        "vlm_only": True,
        #"thr": float(args.thr),
        "max_new_tokens": int(args.max_new_tokens),
        "single": {
            "n_clean_total": int(len(df_clean)),
            "n_clean_used_finite": int(n_ok_clean),
            "n_clean_nan": int(n_nan_clean),
            "n_art_total": int(len(df_pair_weak)),
            "n_art_used_finite": int(n_ok_weak),
            "n_art_nan": int(n_nan_weak),
            # "auroc_clean": float(auroc_clean),
            # "auroc_art": float(auroc_art),
            # "auroc_macro": float(auroc_macro),
            "clean_metrics": m_clean_all,
            "art_metrics": m_weak_all,
        },
        "paired": pair_stats,
    }

    print("\n====== VLM-only Single-image test ======")
    print(f"Clean total={results['single']['n_clean_total']} | used={results['single']['n_clean_used_finite']} | NaN={results['single']['n_clean_nan']}")
    print(f"Art   total={results['single']['n_art_total']}   | used={results['single']['n_art_used_finite']}   | NaN={results['single']['n_art_nan']}")
    # print(f"Clean: AUROC={results['single']['auroc_clean']:.4f}  Acc={results['single']['clean_metrics']['acc']:.4f}")
    # print(f"Art  : AUROC={results['single']['auroc_art']:.4f}    Acc={results['single']['art_metrics']['acc']:.4f}")
    # print(f"Macro avg AUROC: {results['single']['auroc_macro']:.4f}")

    print("\n====== Paired robustness ======")
    print(f"Pairs used={results['paired']['n_pairs_used']} | skipped(NaN)={results['paired']['n_pairs_skipped_nan']}")
    # print(f"AUROC_pair_clean     : {results['paired']['auroc_pair_clean']:.4f}")
    # print(f"AUROC_pair_art_mean  : {results['paired']['auroc_pair_art_mean']:.4f} | Drop_mean={results['paired']['drop_mean']:.4f} | Flip_mean={results['paired']['flip_rate_mean']:.4f}")
    # print(f"AUROC_pair_art_worst : {results['paired']['auroc_pair_art_worst']:.4f} | Drop_worst={results['paired']['drop_worst']:.4f} | Flip_worst={results['paired']['flip_rate_worst']:.4f}")
    #print(f"dp_mean_avg={results['paired']['dp_mean_avg']:.4f} | dp_worst_avg={results['paired']['dp_worst_avg']:.4f}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[saved] {args.out_json}")

    del base_model, processor
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()
