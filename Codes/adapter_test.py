#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
import json
import argparse
from typing import Optional, Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Args
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", type=str, required=True,
                   choices=["qwen3", "medgemma", "internvl", "lingshu"])
    p.add_argument("--layer", type=str, default="last", choices=["last", "hs_-4"])

    p.add_argument("--ckpt", type=str, required=True, help="path to best.pt")
    p.add_argument("--test_clean_csv", type=str, required=True,
                   help="CSV containing clean-only test set (e.g., 540 images)")
    p.add_argument("--test_pair_csv", type=str, required=True,
                   help="CSV containing clean+weak test set (90 fileindex groups, each clean 1 + weak 5)")

    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--out_json", type=str, default=None)

    # inference policy
    p.add_argument("--adapter_on_clean", action="store_true",
                   help="If set, apply adapter to clean images too (recommended). Otherwise clean bypasses adapter.")
    p.add_argument("--thr", type=float, default=0.5)
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
# Model IDs (same as yours)
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
            if k in ("paths", "path", "meta", "severities"):
                out[k] = v
            else:
                out[k] = to_device(v, target_device)
        return out

    if isinstance(x, (list, tuple)):
        if len(x) > 0 and all(isinstance(v, str) for v in x):
            return x
        return type(x)(to_device(v, target_device) for v in x)

    return x


def compute_binary_metrics(y_true: torch.Tensor, y_score: torch.Tensor, thr: float = 0.5):
    y_true = y_true.detach().cpu().to(torch.int64)
    y_score = y_score.detach().cpu().to(torch.float32)

    y_pred = (y_score >= thr).to(torch.int64)

    tp = int(((y_pred == 1) & (y_true == 1)).sum().item())
    tn = int(((y_pred == 0) & (y_true == 0)).sum().item())
    fp = int(((y_pred == 1) & (y_true == 0)).sum().item())
    fn = int(((y_pred == 0) & (y_true == 1)).sum().item())

    eps = 1e-12
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    # AUROC (Mann–Whitney U)
    pos = (y_true == 1)
    neg = (y_true == 0)
    if int(pos.sum()) == 0 or int(neg.sum()) == 0:
        auroc = float("nan")
    else:
        order = torch.argsort(y_score)  # ascending
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(1, len(y_score) + 1, dtype=torch.float32)
        sum_ranks_pos = ranks[pos].sum()
        n_pos = float(pos.sum().item())
        n_neg = float(neg.sum().item())
        u = sum_ranks_pos - n_pos * (n_pos + 1.0) / 2.0
        auroc = float((u / (n_pos * n_neg)).item())

    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(auroc),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# -------------------------
# Adapter + Wrapper (same logic as yours)
# -------------------------
class VisualAdapter(nn.Module):
    def __init__(self, hidden_dim: int, bottleneck: int = 256):
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck)
        self.act  = nn.ReLU()
        self.up   = nn.Linear(bottleneck, hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.up(self.act(self.down(h)))


class VLMAdapterWrapper(nn.Module):
    def __init__(self, base_model, backend: str, layer_choice: str = "last"):
        super().__init__()
        self.base_model = base_model
        self.backend = backend
        self.layer_choice = layer_choice

        for p in self.base_model.parameters():
            p.requires_grad = False

        emb = self._get_text_embeddings(self.base_model)
        hidden_dim = emb.weight.shape[1]
        embed_device = emb.weight.device

        self.hidden_dim = hidden_dim
        self.adapter = VisualAdapter(hidden_dim).to(device=embed_device, dtype=torch.float32)
        self.classifier = nn.Linear(hidden_dim, 2).to(device=embed_device, dtype=torch.float32)

        self.base_model.eval()

    def _get_text_embeddings(self, m: nn.Module) -> nn.Module:
        if hasattr(m, "get_input_embeddings") and callable(m.get_input_embeddings):
            emb = m.get_input_embeddings()
            if emb is not None:
                return emb

        for attr in ["language_model", "lm", "model", "text_model"]:
            if hasattr(m, attr):
                sub = getattr(m, attr)
                if hasattr(sub, "get_input_embeddings") and callable(sub.get_input_embeddings):
                    emb = sub.get_input_embeddings()
                    if emb is not None:
                        return emb

        candidates = [
            ("language_model", "model", "embed_tokens"),
            ("model", "embed_tokens"),
            ("text_model", "embed_tokens"),
            ("language_model", "embed_tokens"),
        ]
        for path in candidates:
            cur: Any = m
            ok = True
            for p in path:
                if hasattr(cur, p):
                    cur = getattr(cur, p)
                else:
                    ok = False
                    break
            if ok and isinstance(cur, nn.Module) and hasattr(cur, "weight"):
                return cur

        raise ValueError("Cannot locate text input embeddings to infer hidden_dim.")

    def extract_features(self, outputs, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        H = self.hidden_dim

        def get(obj, name: str):
            if hasattr(obj, name) and getattr(obj, name) is not None:
                return getattr(obj, name)
            if isinstance(obj, dict) and name in obj and obj[name] is not None:
                return obj[name]
            return None

        hs = get(outputs, "hidden_states")
        t  = get(outputs, "last_hidden_state")

        if self.layer_choice == "hs_-4":
            if isinstance(hs, (list, tuple)) and len(hs) >= 5 and torch.is_tensor(hs[-4]) and hs[-4].dim() == 3 and hs[-4].shape[-1] == H:
                x = hs[-4]
            elif torch.is_tensor(t) and t.dim() == 3 and t.shape[-1] == H:
                x = t
            else:
                raise RuntimeError(f"{self.backend}: cannot use hs_-4")
        else:
            if torch.is_tensor(t) and t.dim() == 3 and t.shape[-1] == H:
                x = t
            elif isinstance(hs, (list, tuple)) and len(hs) > 0 and torch.is_tensor(hs[-1]) and hs[-1].dim() == 3 and hs[-1].shape[-1] == H:
                x = hs[-1]
            else:
                raise RuntimeError(f"{self.backend}: cannot find usable hidden state")

        # masked mean pooling
        if attention_mask is None or attention_mask.dim() != 2:
            return x.mean(dim=1)

        T = min(x.shape[1], attention_mask.shape[1])
        x = x[:, :T, :]
        m = attention_mask[:, :T].to(dtype=x.dtype, device=x.device).unsqueeze(-1)
        x_sum = (x * m).sum(dim=1)
        denom = m.sum(dim=1).clamp_min(1.0)
        return x_sum / denom


# -------------------------
# Backend loader (matches your stable setting)
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
        model.to(device)
        model.eval()
        return model, processor

    if backend == "lingshu":
        from transformers import Qwen2_5_VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, cache_dir=cache_dir, **common)
        model.to(device)
        model.eval()
        return model, processor

    if backend == "internvl":
        from transformers import AutoModel, AutoProcessor as AP2
        processor = AP2.from_pretrained(model_id, trust_remote_code=True, cache_dir=cache_dir)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True, cache_dir=cache_dir, **common)
        model.to(device)
        model.eval()
        return model, processor

    if backend == "medgemma":
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        AutoImageTextToText = getattr(transformers, "AutoModelForImageTextToText", None)
        if AutoImageTextToText is not None:
            model = AutoImageTextToText.from_pretrained(model_id, cache_dir=cache_dir, **common)
            model.to(device); model.eval()
            return model, processor

        AutoVision2Seq = getattr(transformers, "AutoModelForVision2Seq", None)
        if AutoVision2Seq is not None:
            model = AutoVision2Seq.from_pretrained(model_id, cache_dir=cache_dir, **common)
            model.to(device); model.eval()
            return model, processor

        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, **common)
        model.to(device); model.eval()
        return model, processor

    raise ValueError(f"Unknown backend: {backend}")


# -------------------------
# CSV load (same path fix as yours)
# -------------------------
BASE_OUT_DIR = "/SAN/ioo/HORIZON/howoon"

def load_split_csv(path: str, base_out_dir: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "binarylab" in df.columns and "binarylabel" not in df.columns:
        df = df.rename(columns={"binarylab": "binarylabel"})

    df["severity_is_clean"] = df["severity"].isna()
    df["severity_is_weak"]  = df["severity"].fillna("").astype(str).str.lower().eq("weak")
    # normalize severity; treat null as clean
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
# Datasets for TEST
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

        return {"image": img, "input_text": full_text, "label": label, "path": img_path, "severity": sev, "fileindex": row["fileindex"]}


def make_single_collate_fn(processor, backend: str):
    def collate(batch):
        images = [b["image"] for b in batch]
        texts  = [b["input_text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        paths  = [b["path"] for b in batch]
        sevs   = [b["severity"] for b in batch]
        fids   = [int(b["fileindex"]) for b in batch]

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
    Expects exactly 1 clean and >=1 weak per fileindex in test_pair_csv.
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
        items.append((int(fid), clean_row, weak_list))
    return items


# -------------------------
# Forward helpers
# -------------------------
@torch.no_grad()
def forward_batch_get_prob(vlm: VLMAdapterWrapper, batch: Dict[str, Any], apply_adapter: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (y_true, prob_disease)
    """
    target_device = next(vlm.adapter.parameters()).device
    batch = to_device(batch, target_device)

    y = batch["labels_cls"]
    d = dict(batch)
    d.pop("labels_cls", None)
    d.pop("paths", None)
    d.pop("severities", None)
    d.pop("fileindices", None)
    d.pop("labels_token", None)

    # internvl 안정화 (너 로직 유지)
    if vlm.backend == "internvl" and device == "cuda":
        for k in ["pixel_values", "images", "image", "vision_x"]:
            if k in d and torch.is_tensor(d[k]):
                d[k] = d[k].to(dtype=torch.float32)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            outputs = vlm.base_model(**d, output_hidden_states=True, return_dict=True)
    else:
        outputs = vlm.base_model(**d, output_hidden_states=True, return_dict=True)

    attn = d.get("attention_mask", None)
    h_base = vlm.extract_features(outputs, attention_mask=attn).float()
    h_base = F.layer_norm(h_base, (h_base.shape[-1],))

    if apply_adapter:
        h = vlm.adapter(h_base)
    else:
        h = h_base

    logits = vlm.classifier(h)
    prob = torch.softmax(logits.float(), dim=1)[:, 1]
    return y.detach().cpu(), prob.detach().cpu()


def eval_single_loader(vlm: VLMAdapterWrapper, loader: DataLoader, thr: float, adapter_on: bool):
    all_y, all_p, all_sev, all_fid = [], [], [], []
    for batch in loader:
        y, p = forward_batch_get_prob(vlm, batch, apply_adapter=adapter_on)
        all_y.append(y)
        all_p.append(p)
        all_sev.extend(batch["severities"])
        all_fid.extend(batch["fileindices"])

    y_true = torch.cat(all_y, dim=0)
    y_prob = torch.cat(all_p, dim=0)
    m_all = compute_binary_metrics(y_true, y_prob, thr=thr)

    all_sev = np.array(all_sev)
    clean_mask = all_sev == "clean"
    weak_mask  = all_sev == "weak"

    m_clean = compute_binary_metrics(y_true[clean_mask], y_prob[clean_mask], thr=thr) if clean_mask.sum() > 0 else {}
    m_weak  = compute_binary_metrics(y_true[weak_mask],  y_prob[weak_mask],  thr=thr) if weak_mask.sum() > 0 else {}

    return m_all, m_clean, m_weak, y_true, y_prob


def eval_pairs(vlm: VLMAdapterWrapper,
              processor,
              backend: str,
              pair_items: List[Tuple[int, pd.Series, List[pd.Series]]],
              thr: float,
              adapter_on_clean: bool):
    """
    Evaluate 90 paired groups:
      - clean prob (one image)
      - weak prob list (5 images)
    Compute AUROC on clean(90) and aggregated weak(90) by mean/worst.
    Compute flip rate vs threshold.
    """
    # We will run inference image-by-image for simplicity/robustness.
    y_clean_list, p_clean_list = [], []
    y_w_mean_list, p_w_mean_list = [], []
    y_w_worst_list, p_w_worst_list = [], []

    flip_mean = 0
    flip_worst = 0
    n_pairs = 0

    # Helper: make a single-item batch using your processor path
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

    for fid, clean_row, weak_rows in pair_items:
        n_pairs += 1

        # prompt
        modality = str(clean_row["dataset_norm"]).lower()
        task_prompt = PROMPT_BY_DATASET.get(modality, PROMPT_BY_DATASET["mri"])
        full_text = (SYSTEM_PROMPT_SHORT + "\n\n" + task_prompt)

        # clean
        img_c = Image.open(clean_row["filepath"]).convert("RGB")
        y_c = int(clean_row["binarylabel"])

        batch_c = make_one_batch(img_c, full_text, y_c)
        y_c_t, p_c_t = forward_batch_get_prob(vlm, batch_c, apply_adapter=adapter_on_clean)
        p_clean = float(p_c_t.item())

        # weak list
        p_ws = []
        for wr in weak_rows:
            img_w = Image.open(wr["filepath"]).convert("RGB")
            y_w = int(wr["binarylabel"])  # should match y_c
            batch_w = make_one_batch(img_w, full_text, y_w)
            _, p_w_t = forward_batch_get_prob(vlm, batch_w, apply_adapter=True)  # adapter ON for artefact
            p_ws.append(float(p_w_t.item()))

        p_mean = float(np.mean(p_ws))
        p_worst = float(np.min(p_ws))

        # store
        y_clean_list.append(y_c)
        p_clean_list.append(p_clean)

        y_w_mean_list.append(y_c)
        p_w_mean_list.append(p_mean)

        y_w_worst_list.append(y_c)
        p_w_worst_list.append(p_worst)

        # flip rates
        pred_c = 1 if p_clean >= thr else 0
        pred_mean = 1 if p_mean >= thr else 0
        pred_worst = 1 if p_worst >= thr else 0
        if pred_c != pred_mean:
            flip_mean += 1
        if pred_c != pred_worst:
            flip_worst += 1

    y_clean = torch.tensor(y_clean_list, dtype=torch.long)
    p_clean = torch.tensor(p_clean_list, dtype=torch.float32)

    y_mean = torch.tensor(y_w_mean_list, dtype=torch.long)
    p_mean = torch.tensor(p_w_mean_list, dtype=torch.float32)

    y_worst = torch.tensor(y_w_worst_list, dtype=torch.long)
    p_worst = torch.tensor(p_w_worst_list, dtype=torch.float32)

    m_clean = compute_binary_metrics(y_clean, p_clean, thr=thr)
    m_mean  = compute_binary_metrics(y_mean,  p_mean,  thr=thr)
    m_worst = compute_binary_metrics(y_worst, p_worst, thr=thr)

    p_clean_np = np.array(p_clean_list, dtype=np.float32)
    p_mean_np  = np.array(p_w_mean_list, dtype=np.float32)
    p_worst_np = np.array(p_w_worst_list, dtype=np.float32)
    y_np       = np.array(y_clean_list, dtype=np.int64)

    dp_mean  = p_clean_np - p_mean_np
    dp_worst = p_clean_np - p_worst_np

    # overall
    dp_mean_avg  = float(np.mean(dp_mean)) if len(dp_mean) > 0 else float("nan")
    dp_worst_avg = float(np.mean(dp_worst)) if len(dp_worst) > 0 else float("nan")

    # GT positive subset (y==1)
    pos = (y_np == 1)
    dp_mean_pos_avg  = float(np.mean(dp_mean[pos])) if pos.any() else float("nan")
    dp_worst_pos_avg = float(np.mean(dp_worst[pos])) if pos.any() else float("nan")


    out = {
        "n_pairs": int(n_pairs),
        "auroc_pair_clean": m_clean["auroc"],
        "auroc_pair_art_mean": m_mean["auroc"],
        "drop_mean": (m_mean["auroc"] - m_clean["auroc"]) if (np.isfinite(m_mean["auroc"]) and np.isfinite(m_clean["auroc"])) else float("nan"),
        "auroc_pair_art_worst": m_worst["auroc"],
        "drop_worst": (m_worst["auroc"] - m_clean["auroc"]) if (np.isfinite(m_worst["auroc"]) and np.isfinite(m_clean["auroc"])) else float("nan"),
        "flip_rate_mean": float(flip_mean / max(1, n_pairs)),
        "flip_rate_worst": float(flip_worst / max(1, n_pairs)),
    }
    out.update({
        "dp_mean_avg": dp_mean_avg,
        "dp_worst_avg": dp_worst_avg,
        "dp_mean_pos_avg": dp_mean_pos_avg,
        "dp_worst_pos_avg": dp_worst_pos_avg,
    })

    return out


# -------------------------
# Main
# -------------------------
def main():
    backend = args.backend
    model_id = MODEL_ID_BY_BACKEND[backend]

    print("\n==============================")
    print(f"== EVAL backend: {backend} | layer={args.layer}")
    print(f"== ckpt: {args.ckpt}")
    print("==============================")

    base_model, processor = load_backend(backend, model_id)
    vlm = VLMAdapterWrapper(base_model, backend=backend, layer_choice=args.layer)

    # load ckpt
    ckpt = torch.load(args.ckpt, map_location="cpu")
    vlm.adapter.load_state_dict(ckpt["adapter"], strict=True)
    vlm.classifier.load_state_dict(ckpt["classifier"], strict=True)
    vlm.eval()

    # ---- SINGLE IMAGE TESTS ----
    df_clean = load_split_csv(args.test_clean_csv, BASE_OUT_DIR)
    # enforce clean-only
    df_clean = df_clean[df_clean["severity_norm"] == "clean"].reset_index(drop=True)

    # pair csv will contain both clean and weak; we will split it:
    df_pair = load_split_csv(args.test_pair_csv, BASE_OUT_DIR)
    df_pair_clean = df_pair[df_pair["severity_norm"] == "clean"].reset_index(drop=True)
    df_pair_weak  = df_pair[df_pair["severity_norm"] == "weak"].reset_index(drop=True)

    # loaders
    ds_clean = SingleImageDataset(df_clean, PROMPT_BY_DATASET, SYSTEM_PROMPT_SHORT)
    ds_weak  = SingleImageDataset(df_pair_weak, PROMPT_BY_DATASET, SYSTEM_PROMPT_SHORT)  # 450 artefact

    collate = make_single_collate_fn(processor, backend)
    dl_clean = DataLoader(ds_clean, batch_size=args.bs, shuffle=False, collate_fn=collate, pin_memory=(device == "cuda"))
    dl_weak  = DataLoader(ds_weak,  batch_size=args.bs, shuffle=False, collate_fn=collate, pin_memory=(device == "cuda"))

    # policy: adapter on artefact always; clean depends on flag
    m_clean_all, _, _, _, _ = eval_single_loader(vlm, dl_clean, thr=args.thr, adapter_on=args.adapter_on_clean)
    m_weak_all,  _, _, _, _ = eval_single_loader(vlm, dl_weak,  thr=args.thr, adapter_on=True)

    auroc_clean = m_clean_all["auroc"]
    auroc_art   = m_weak_all["auroc"]
    auroc_macro = (auroc_clean + auroc_art) / 2.0 if (np.isfinite(auroc_clean) and np.isfinite(auroc_art)) else float("nan")

    # ---- PAIRED ROBUSTNESS ----
    pair_items = build_pair_index(df_pair)
    pair_stats = eval_pairs(
        vlm=vlm,
        processor=processor,
        backend=backend,
        pair_items=pair_items,
        thr=args.thr,
        adapter_on_clean=args.adapter_on_clean,
    )

    results = {
        "backend": backend,
        "layer": args.layer,
        "ckpt": args.ckpt,
        "adapter_on_clean": bool(args.adapter_on_clean),
        "single": {
            "n_clean": int(len(df_clean)),
            "n_art": int(len(df_pair_weak)),
            "auroc_clean": float(auroc_clean),
            "auroc_art": float(auroc_art),
            "auroc_macro": float(auroc_macro),
            "clean_metrics": m_clean_all,
            "art_metrics": m_weak_all,
        },
        "paired": pair_stats,
    }

    print("\n====== Single-image test ======")
    print(f"Clean (n={results['single']['n_clean']}): AUROC={results['single']['auroc_clean']:.4f}  Acc={results['single']['clean_metrics']['acc']:.4f}")
    print(f"Art   (n={results['single']['n_art']}): AUROC={results['single']['auroc_art']:.4f}    Acc={results['single']['art_metrics']['acc']:.4f}")
    print(f"Macro avg AUROC: {results['single']['auroc_macro']:.4f}")

    print("\n====== Paired robustness (90 pairs) ======")
    print(f"Pairs n={results['paired']['n_pairs']}")
    print(f"AUROC_pair_clean     : {results['paired']['auroc_pair_clean']:.4f}")
    print(f"AUROC_pair_art_mean  : {results['paired']['auroc_pair_art_mean']:.4f} | Drop_mean={results['paired']['drop_mean']:.4f} | Flip_mean={results['paired']['flip_rate_mean']:.4f}")
    print(f"AUROC_pair_art_worst : {results['paired']['auroc_pair_art_worst']:.4f} | Drop_worst={results['paired']['drop_worst']:.4f} | Flip_worst={results['paired']['flip_rate_worst']:.4f}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n[saved] {args.out_json}")

    # cleanup
    del vlm, base_model, processor
    gc.collect()
    torch.cuda.empty_cache()
    if device == "cuda":
        torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()
