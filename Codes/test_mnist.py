#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VLM evaluation on NEW corruption CSV format + Prompt baseline (no training)
- EffNet excluded (as requested)
- Baseline AUROC not computed (set to NaN)
- All scores computed overall + by modality:
  - single clean/art metrics (acc/precision/recall/f1 + balanced_error)
  - corruption_analysis (overall + by modality)
  - CE/mCE vs prompt baseline (overall + by modality)

CSV required columns:
  dataset, severity(0..4), clean_index, binarylabel(or label), filepath, clean_filepath, corruption(or corrupt_detail)

Your rules:
- severity 0..4 are all artifacts (severity=0 still artifact)
- clean image referenced by clean_filepath per clean_index
"""

import os
import gc
import json
import argparse
import re
from typing import Optional, Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score


# =============================================================================
# Args
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--test_csv", type=str, required=True,
                   help="(new format) one CSV with severity 0..4 artifacts + clean_filepath per clean_index")
    p.add_argument("--corrupt_key", type=str, default="corruption",
                   choices=["corruption", "corrupt_detail"],
                   help="Which column to use as corruption name")

    p.add_argument("--out_json", type=str, required=True, help="output json base path")
    p.add_argument("--thr", type=float, default=0.5, help="threshold for Acc/F1 etc.")

    # VLM settings
    p.add_argument("--backend", type=str, required=True, choices=["all", "qwen3", "medgemma", "internvl", "lingshu"])
    p.add_argument("--layer", type=str, default="last", choices=["last", "hs_-4"])
    p.add_argument("--vlm_ckpt", type=str, default=None, help="path to VLM best.pt (adapter + classifier)")

    p.add_argument("--vlm_ckpt_qwen3", type=str, default=None)
    p.add_argument("--vlm_ckpt_medgemma", type=str, default=None)
    p.add_argument("--vlm_ckpt_internvl", type=str, default=None)
    p.add_argument("--vlm_ckpt_lingshu", type=str, default=None)

    p.add_argument("--adapter_on_clean", action="store_true",
                   help="apply adapter to clean images too (recommended for fair evaluation)")

    # baseline generation
    p.add_argument("--baseline_max_new_tokens", type=int, default=24,
                   help="max_new_tokens for prompt baseline generation")

    return p.parse_args()


args = parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device, flush=True)

# HF cache (VLM)
assert os.environ.get("HF_HOME"), "HF_HOME not set"
hf_home = os.environ["HF_HOME"]
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(hf_home, "hub"))


# =============================================================================
# Prompts (VLM)
# =============================================================================
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
SYSTEM_PROMPT_JSON = (
    'Return ONLY valid JSON with one key:\n'
    '{"label":"normal"} or {"label":"disease"}.\n'
    "No explanation, no markdown, no extra keys."
)

MODEL_ID_BY_BACKEND = {
    "qwen3":    "Qwen/Qwen3-VL-8B-Instruct",
    "medgemma": "google/medgemma-1.5-4b-it",
    "internvl": "OpenGVLab/InternVL3_5-8B-HF",
    "lingshu":  "lingshu-medical-mllm/Lingshu-7B",
}

BASE_OUT_DIR = "/SAN/ioo/HORIZON/howoon"


# =============================================================================
# Metrics
# =============================================================================
def compute_binary_metrics_from_preds_full(y_true: torch.Tensor, y_pred: torch.Tensor):
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

    tpr = recall
    tnr = tn / (tn + fp + eps)
    bacc = 0.5 * (tpr + tnr)
    be = 1.0 - bacc

    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "balanced_acc": float(bacc),
        "balanced_error": float(be),
        "auroc": float("nan"),  # baseline AUROC not used
    }


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

    # AUROC
    y_np = y_true.numpy()
    s_np = y_score.numpy()
    if (y_np == 1).sum() == 0 or (y_np == 0).sum() == 0:
        auroc = float("nan")
    else:
        auroc = float(roc_auc_score(y_np, s_np))

    # balanced acc / balanced error
    tpr = recall
    tnr = tn / (tn + fp + eps)
    bacc = 0.5 * (tpr + tnr)
    be = 1.0 - bacc

    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(auroc),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "balanced_acc": float(bacc),
        "balanced_error": float(be),
    }


def pick_binary_only_metrics(m: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(m, dict):
        return {}
    keys = ["acc", "precision", "recall", "f1", "tp", "tn", "fp", "fn"]
    return {k: m[k] for k in keys if k in m}


def finalize_single_block(n_clean, n_art, m_clean_all, m_art_all):
    auroc_clean = float(m_clean_all.get("auroc", float("nan"))) if isinstance(m_clean_all, dict) else float("nan")
    auroc_art   = float(m_art_all.get("auroc", float("nan"))) if isinstance(m_art_all, dict) else float("nan")
    auroc_macro = float((auroc_clean + auroc_art) / 2.0) if (np.isfinite(auroc_clean) and np.isfinite(auroc_art)) else float("nan")
    return {
        "n_clean": int(n_clean),
        "n_art": int(n_art),
        "auroc_clean": auroc_clean,
        "auroc_art": auroc_art,
        "auroc_macro": auroc_macro,
        "clean_metrics": m_clean_all,
        "art_metrics": m_art_all,
    }


def finalize_single_by_modality(y_true: torch.Tensor,
                                y_prob: torch.Tensor,
                                severities: List[str],
                                modalities: List[str],
                                thr: float):
    severities = np.array([str(s).lower() for s in severities], dtype=object)
    modalities = np.array([str(m).lower() for m in modalities], dtype=object)

    out = {}
    for mod in sorted(set(modalities.tolist())):
        m_mask = (modalities == mod)
        clean_mask = m_mask & (severities == "clean")
        art_mask   = m_mask & (severities != "clean")

        m_clean = compute_binary_metrics(y_true[torch.tensor(clean_mask)], y_prob[torch.tensor(clean_mask)], thr=thr) if clean_mask.sum() > 0 else {}
        m_art   = compute_binary_metrics(y_true[torch.tensor(art_mask)],   y_prob[torch.tensor(art_mask)],   thr=thr) if art_mask.sum() > 0 else {}

        auroc_clean = float(m_clean.get("auroc", float("nan"))) if isinstance(m_clean, dict) else float("nan")
        auroc_art   = float(m_art.get("auroc", float("nan"))) if isinstance(m_art, dict) else float("nan")
        auroc_macro = float((auroc_clean + auroc_art) / 2.0) if (np.isfinite(auroc_clean) and np.isfinite(auroc_art)) else float("nan")

        out[mod] = {
            "n_clean": int(clean_mask.sum()),
            "n_art": int(art_mask.sum()),
            "auroc_clean": auroc_clean,
            "auroc_art": auroc_art,
            "auroc_macro": auroc_macro,
            "clean_metrics": m_clean,
            "art_metrics": m_art,
        }
    return out


def finalize_single_by_modality_from_preds(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    severities: List[str],
    modalities: List[str],
):
    severities = np.array([str(s).lower() for s in severities], dtype=object)
    modalities = np.array([str(m).lower() for m in modalities], dtype=object)

    out = {}
    for mod in sorted(set(modalities.tolist())):
        m_mask = (modalities == mod)
        clean_mask = m_mask & (severities == "clean")
        art_mask   = m_mask & (severities != "clean")

        if clean_mask.sum() > 0:
            yt = y_true[torch.tensor(clean_mask, dtype=torch.bool)]
            yp = y_pred[torch.tensor(clean_mask, dtype=torch.bool)]
            m_clean = compute_binary_metrics_from_preds_full(yt, yp)
        else:
            m_clean = {}

        if art_mask.sum() > 0:
            yt = y_true[torch.tensor(art_mask, dtype=torch.bool)]
            yp = y_pred[torch.tensor(art_mask, dtype=torch.bool)]
            m_art = compute_binary_metrics_from_preds_full(yt, yp)
        else:
            m_art = {}

        out[mod] = {
            "n_clean": int(clean_mask.sum()),
            "n_art": int(art_mask.sum()),
            "clean_metrics": m_clean,
            "art_metrics": m_art,
        }
    return out


# =============================================================================
# Corruption analysis (trained/prob)
# =============================================================================
def _slice_metrics_prob(y_true: torch.Tensor, y_prob: torch.Tensor, mask: np.ndarray, thr: float) -> Dict[str, Any]:
    if mask.sum() == 0:
        return {}
    yt = y_true[torch.tensor(mask, dtype=torch.bool)]
    yp = y_prob[torch.tensor(mask, dtype=torch.bool)]
    return compute_binary_metrics(yt, yp, thr=thr)


def compute_corruption_severity_metrics_and_robustness(
    y_true: torch.Tensor,
    y_prob: torch.Tensor,
    severities: List[str],         # "clean" or "0".."4"
    corruptions: List[str],        # corruption names
    thr: float,
):
    sev = np.array([str(s).lower() for s in severities], dtype=object)
    cor = np.array([str(c).lower() for c in corruptions], dtype=object)

    clean_mask = (sev == "clean")
    m_clean = _slice_metrics_prob(y_true, y_prob, clean_mask, thr=thr)
    be_clean = m_clean.get("balanced_error", None)
    f1_clean = m_clean.get("f1", None)

    metrics_by_severity_pooled = {}
    sev_levels = sorted(set([s for s in sev.tolist() if s != "clean"]))
    for s in sev_levels:
        metrics_by_severity_pooled[str(s)] = _slice_metrics_prob(y_true, y_prob, (sev == s), thr=thr)

    metrics_by_corruption = {}
    for c in sorted(set(cor.tolist())):
        m_c_pooled = _slice_metrics_prob(y_true, y_prob, (cor == c) & (sev != "clean"), thr=thr)
        by_s = {}
        for s in sev_levels:
            by_s[str(s)] = _slice_metrics_prob(y_true, y_prob, (cor == c) & (sev == s), thr=thr)
        metrics_by_corruption[c] = {"pooled": m_c_pooled, "by_severity": by_s}

    per_c_scores = {}
    rbe_vals, df1_vals = [], []
    for c, dd in metrics_by_corruption.items():
        be_diffs, f1_drops = [], []
        for s in sev_levels:
            ms = dd["by_severity"].get(str(s), {}) or {}
            if be_clean is not None and "balanced_error" in ms:
                be_diffs.append(float(ms["balanced_error"]) - float(be_clean))
            if f1_clean is not None and "f1" in ms:
                f1_drops.append(float(f1_clean) - float(ms["f1"]))

        sc = {}
        if len(be_diffs) > 0:
            sc["rBE_no_baseline_mean"] = float(np.mean(be_diffs))
            rbe_vals.append(sc["rBE_no_baseline_mean"])
        if len(f1_drops) > 0:
            sc["dF1_mean"] = float(np.mean(f1_drops))
            df1_vals.append(sc["dF1_mean"])
        per_c_scores[c] = sc

    overall = {}
    if len(rbe_vals) > 0:
        overall["rBE_over_corruptions"] = float(np.mean(rbe_vals))
    if len(df1_vals) > 0:
        overall["dF1_over_corruptions"] = float(np.mean(df1_vals))

    return {
        "clean_metrics": m_clean,
        "metrics_by_severity_pooled": metrics_by_severity_pooled,
        "metrics_by_corruption": metrics_by_corruption,
        "robustness_scores": {
            "per_corruption": per_c_scores,
            "overall": overall,
        }
    }


# =============================================================================
# Corruption analysis (baseline/preds)
# =============================================================================
def compute_corruption_severity_metrics_from_preds(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    severities: List[str],
    corruptions: List[str],
):
    sev = np.array([str(s).lower() for s in severities], dtype=object)
    cor = np.array([str(c).lower() for c in corruptions], dtype=object)

    def _slice(mask: np.ndarray) -> Dict[str, Any]:
        if mask.sum() == 0:
            return {}
        yt = y_true[torch.tensor(mask, dtype=torch.bool)]
        yp = y_pred[torch.tensor(mask, dtype=torch.bool)]
        return compute_binary_metrics_from_preds_full(yt, yp)

    clean_mask = (sev == "clean")
    m_clean = _slice(clean_mask)
    be_clean = m_clean.get("balanced_error", None)
    f1_clean = m_clean.get("f1", None)

    metrics_by_severity_pooled = {}
    sev_levels = sorted(set([s for s in sev.tolist() if s != "clean"]))
    for s in sev_levels:
        metrics_by_severity_pooled[str(s)] = _slice(sev == s)

    metrics_by_corruption = {}
    for c in sorted(set(cor.tolist())):
        pooled = _slice((cor == c) & (sev != "clean"))
        by_s = {}
        for s in sev_levels:
            by_s[str(s)] = _slice((cor == c) & (sev == s))
        metrics_by_corruption[c] = {"pooled": pooled, "by_severity": by_s}

    per_c_scores = {}
    rbe_vals, df1_vals = [], []
    for c, dd in metrics_by_corruption.items():
        be_diffs, f1_drops = [], []
        for s in sev_levels:
            ms = dd["by_severity"].get(str(s), {}) or {}
            if be_clean is not None and "balanced_error" in ms:
                be_diffs.append(float(ms["balanced_error"]) - float(be_clean))
            if f1_clean is not None and "f1" in ms:
                f1_drops.append(float(f1_clean) - float(ms["f1"]))

        sc = {}
        if len(be_diffs) > 0:
            sc["rBE_no_baseline_mean"] = float(np.mean(be_diffs))
            rbe_vals.append(sc["rBE_no_baseline_mean"])
        if len(f1_drops) > 0:
            sc["dF1_mean"] = float(np.mean(f1_drops))
            df1_vals.append(sc["dF1_mean"])
        per_c_scores[c] = sc

    overall = {}
    if len(rbe_vals) > 0:
        overall["rBE_over_corruptions"] = float(np.mean(rbe_vals))
    if len(df1_vals) > 0:
        overall["dF1_over_corruptions"] = float(np.mean(df1_vals))

    return {
        "clean_metrics": m_clean,
        "metrics_by_severity_pooled": metrics_by_severity_pooled,
        "metrics_by_corruption": metrics_by_corruption,
        "robustness_scores": {
            "per_corruption": per_c_scores,
            "overall": overall,
        }
    }


# =============================================================================
# Modality-wise corruption analysis helpers
# =============================================================================
def compute_corruption_analysis_by_modality_prob(
    y_true: torch.Tensor,
    y_prob: torch.Tensor,
    severities: List[str],
    corruptions: List[str],
    modalities: List[str],
    thr: float,
):
    mods = np.array([str(m).lower() for m in modalities], dtype=object)
    out = {}
    for mod in sorted(set(mods.tolist())):
        msk = (mods == mod)
        if msk.sum() == 0:
            continue
        idx = torch.tensor(msk, dtype=torch.bool)
        out[mod] = compute_corruption_severity_metrics_and_robustness(
            y_true=y_true[idx],
            y_prob=y_prob[idx],
            severities=[severities[i] for i in np.where(msk)[0].tolist()],
            corruptions=[corruptions[i] for i in np.where(msk)[0].tolist()],
            thr=thr,
        )
    return out


def compute_corruption_analysis_by_modality_preds(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    severities: List[str],
    corruptions: List[str],
    modalities: List[str],
):
    mods = np.array([str(m).lower() for m in modalities], dtype=object)
    out = {}
    for mod in sorted(set(mods.tolist())):
        msk = (mods == mod)
        if msk.sum() == 0:
            continue
        idx = torch.tensor(msk, dtype=torch.bool)
        out[mod] = compute_corruption_severity_metrics_from_preds(
            y_true=y_true[idx],
            y_pred=y_pred[idx],
            severities=[severities[i] for i in np.where(msk)[0].tolist()],
            corruptions=[corruptions[i] for i in np.where(msk)[0].tolist()],
        )
    return out


# =============================================================================
# CE/mCE vs baseline
# =============================================================================
def compute_ce_mce_vs_prompt_baseline(trained_rob: Dict[str, Any], base_rob: Dict[str, Any], sev_levels=("0","1","2","3","4")):
    ce = {}
    trained_mb = (trained_rob or {}).get("metrics_by_corruption", {}) or {}
    base_mb    = (base_rob or {}).get("metrics_by_corruption", {}) or {}

    for c in trained_mb.keys():
        num = 0.0
        den = 0.0
        for s in sev_levels:
            acc_t = ((trained_mb.get(c, {}).get("by_severity", {}) or {}).get(str(s), {}) or {}).get("acc", None)
            acc_b = ((base_mb.get(c, {}).get("by_severity", {}) or {}).get(str(s), {}) or {}).get("acc", None)
            if acc_t is None or acc_b is None:
                continue
            num += (1.0 - float(acc_t))
            den += (1.0 - float(acc_b))
        ce[c] = float("nan") if den <= 1e-12 else float(num / den)

    vals = [v for v in ce.values() if np.isfinite(v)]
    mce = float(np.mean(vals)) if len(vals) else float("nan")
    return {"ce_by_corruption": ce, "mce": mce}


def compute_ce_mce_by_modality(trained_by_mod: Dict[str, Any], base_by_mod: Dict[str, Any]):
    out = {}
    for mod, rob_t in (trained_by_mod or {}).items():
        rob_b = (base_by_mod or {}).get(mod, None)
        if rob_b is None:
            out[mod] = {"ce_by_corruption": {}, "mce": float("nan")}
            continue

        sev_levels = sorted([
            k for k in (rob_t.get("metrics_by_severity_pooled", {}) or {}).keys()
            if str(k).lower() != "clean"
        ])
        if len(sev_levels) == 0:
            sev_levels = ["0","1","2","3","4"]

        out[mod] = compute_ce_mce_vs_prompt_baseline(rob_t, rob_b, sev_levels=tuple(sev_levels))
    return out


# =============================================================================
# CSV loading
# =============================================================================
def modality_from_dataset_name(name: str) -> str:
    n = (name or "").lower()
    if "oct" in n:
        return "oct"
    if "pneumonia" in n or "pneumoniamnist" in n or "chest" in n or "xray" in n or "cxr" in n:
        return "xray"
    if "fundus" in n:
        return "fundus"
    if "mri" in n or "brain" in n:
        return "mri"
    return "mri"


def load_corrupt_csv_for_eval(path: str, base_out_dir: str, corrupt_key: str = "corruption") -> pd.DataFrame:
    df = pd.read_csv(path)

    need = ["dataset", "severity", "clean_index", "filepath", "clean_filepath", corrupt_key]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"{path} missing column: {c}")

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

    df["dataset_norm"] = df["dataset"].astype(str).map(modality_from_dataset_name)
    df["severity_int"] = pd.to_numeric(df["severity"], errors="coerce").fillna(-1).astype(int)

    # binarylabel preferred; fallback label
    if "binarylabel" in df.columns:
        df["binarylabel"] = pd.to_numeric(df["binarylabel"], errors="coerce").fillna(0).astype(int)
    elif "label" in df.columns:
        df["binarylabel"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    else:
        raise ValueError(f"{path} missing column: binarylabel (or label)")

    df["clean_index_str"] = df["clean_index"].astype(str)
    df["corruption_name"] = df[corrupt_key].astype(str).fillna("unknown").str.lower()
    return df.reset_index(drop=True)


def make_df_clean_from_clean_filepath(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cid, gg in df.groupby("clean_index_str"):
        r0 = gg.iloc[0]
        rows.append({
            "filepath": r0["clean_filepath"],
            "binarylabel": int(r0["binarylabel"]),
            "dataset_norm": r0["dataset_norm"],
            "severity_norm": "clean",
            "fileindex": str(cid),
            "corruption_name": "clean",
        })
    return pd.DataFrame(rows).reset_index(drop=True)


def make_df_artifacts_from_filepath(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["severity_norm"] = out["severity_int"].astype(int).astype(str)   # "0".."4"
    out["fileindex"] = out["clean_index_str"].astype(str)
    return out[[
        "filepath", "binarylabel", "dataset_norm", "severity_norm",
        "fileindex", "severity_int", "corruption_name"
    ]].reset_index(drop=True)


# =============================================================================
# VLM model wrapper (adapter+classifier)
# =============================================================================
class VisualAdapter(nn.Module):
    def __init__(self, hidden_dim: int, bottleneck: int = 256):
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck)
        self.act  = nn.ReLU()
        self.up   = nn.Linear(bottleneck, hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.up(self.act(self.down(h)))


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query_projection = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = x.to(dtype=self.query_projection.weight.dtype, device=self.query_projection.weight.device)
        attn_logits = self.query_projection(x)
        if mask is not None:
            m = mask.unsqueeze(-1).to(dtype=x.dtype)
            attn_logits = attn_logits.masked_fill(m == 0, -1e9)
        attn_weights = F.softmax(attn_logits, dim=1)
        pooled = (x * attn_weights).sum(dim=1)
        return pooled, attn_weights


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
        self.pooler = AttentionPooling(hidden_dim).to(device=embed_device, dtype=torch.float32)
        self.adapter = VisualAdapter(hidden_dim).to(device=embed_device, dtype=torch.float32)
        self.classifier = nn.Linear(hidden_dim, 2).to(device=embed_device, dtype=torch.float32)
        self.use_attention_pooling = False

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

        if self.use_attention_pooling:
            pooled, _ = self.pooler(x, attention_mask)
            return pooled

        if attention_mask is None or attention_mask.dim() != 2:
            return x.mean(dim=1)

        Tm = min(x.shape[1], attention_mask.shape[1])
        x = x[:, :Tm, :]
        m = attention_mask[:, :Tm].to(dtype=x.dtype, device=x.device).unsqueeze(-1)
        x_sum = (x * m).sum(dim=1)
        denom = m.sum(dim=1).clamp_min(1.0)
        return x_sum / denom


def to_device(x, target_device: torch.device):
    if torch.is_tensor(x):
        return x.to(target_device, non_blocking=True)

    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k in ("paths", "path", "meta", "severities", "modalities", "fileindices", "corruptions"):
                out[k] = v
            else:
                out[k] = to_device(v, target_device)
        return out

    if isinstance(x, (list, tuple)):
        if len(x) > 0 and all(isinstance(v, str) for v in x):
            return x
        return type(x)(to_device(v, target_device) for v in x)

    return x


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
        model.to(device); model.eval()
        return model, processor

    if backend == "lingshu":
        from transformers import Qwen2_5_VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, cache_dir=cache_dir, **common)
        model.to(device); model.eval()
        return model, processor

    if backend == "internvl":
        from transformers import AutoModel, AutoProcessor as AP2
        processor = AP2.from_pretrained(model_id, trust_remote_code=True, cache_dir=cache_dir)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True, cache_dir=cache_dir, **common)
        model.to(device); model.eval()
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


def load_backend_for_prompt(backend: str, model_id: str):
    """
    Prompt baseline loader.
    For InternVL, use a generation-capable class because AutoModel(InternVLModel)
    may not expose .generate().
    """
    if backend != "internvl":
        return load_backend(backend, model_id)

    from transformers import AutoProcessor, AutoModelForImageTextToText

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    if device == "cuda":
        torch_dtype = torch.bfloat16

    cache_dir = os.environ.get("TRANSFORMERS_CACHE") or os.environ.get("HF_HOME") or None
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=True,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, cache_dir=cache_dir)
    model.to(device)
    model.eval()
    return model, processor


# =============================================================================
# Prompt-baseline generate + parse
# =============================================================================
def _parse_normal_disease(text: str) -> Optional[int]:
    t = (text or "").strip().lower()
    if not t:
        return None

    # 1) JSON-first parsing: {"label":"normal|disease"}
    m_json = re.search(
        r'\{\s*"label"\s*:\s*"(normal|disease)"\s*\}',
        t,
        flags=re.IGNORECASE,
    )
    if m_json:
        lbl = m_json.group(1).lower()
        return 0 if lbl == "normal" else 1

    # 1.5) Accept partial JSON when generation was cut:
    # e.g. {"label":"normal
    m_json_partial = re.search(
        r'"label"\s*:\s*"(normal|disease)\b',
        t,
        flags=re.IGNORECASE,
    )
    if m_json_partial:
        lbl = m_json_partial.group(1).lower()
        return 0 if lbl == "normal" else 1

    # Prefer the model's final answer span when prompt text is echoed.
    ans = t
    for marker in ["answer:", "assistant:"]:
        if marker in ans:
            ans = ans.split(marker)[-1].strip()

    # Evaluate only answer block before explanation tail.
    lines = [ln.strip() for ln in ans.splitlines() if ln.strip()]
    answer_lines = []
    for ln in lines:
        if ln.startswith("explanation:"):
            break
        answer_lines.append(ln)
    if not answer_lines:
        answer_lines = lines if lines else [ans]

    block = " ".join(answer_lines).strip()
    if not block:
        return None

    # If both labels appear in answer block, treat as ambiguous.
    has_normal_block = re.search(r"\bnormal\b", block) is not None
    has_disease_block = re.search(r"\bdisease\b", block) is not None
    if has_normal_block and has_disease_block:
        return None

    # First valid answer line wins.
    for ln in answer_lines:
        m_exact = re.fullmatch(r'["\']?\s*(normal|disease)\s*["\']?[.!?]?\s*', ln)
        if m_exact:
            return 0 if m_exact.group(1) == "normal" else 1
        if re.search(r"\bnormal\b", ln) is not None:
            return 0
        if re.search(r"\bdisease\b", ln) is not None:
            return 1

    # Fallback on answer block
    if has_normal_block:
        return 0
    if has_disease_block:
        return 1
    return None


@torch.no_grad()
def forward_vlm_prompt_pred_one(
    base_model,
    processor,
    backend: str,
    img: Image.Image,
    text: str,
    device: str,
    max_new_tokens: int = 8,
) -> Tuple[int, str]:
    if not hasattr(base_model, "generate"):
        raise RuntimeError(
            f"{backend} model class does not expose generate(); "
            "use a generation-capable loader for prompt baseline."
        )

    if backend in ["lingshu", "qwen3"]:
        messages = [[{
            "role": "user",
            "content": [{"type": "image", "image": img}, {"type": "text", "text": text}],
        }]]
        chat_text = processor.apply_chat_template(messages[0], tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[chat_text], images=[img], padding=True, return_tensors="pt")

    elif backend == "internvl":
        image_tok = getattr(processor, "image_token", None) or "<image>"
        inline_text = f"{image_tok}\n{text}"
        inputs = processor(text=[inline_text], images=[img], padding=True, return_tensors="pt")

    elif backend == "medgemma":
        messages = [[{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": text}],
        }]]
        chat_text = processor.apply_chat_template(messages[0], tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[chat_text], images=[img], padding=True, return_tensors="pt")

    else:
        inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt")

    for k, v in list(inputs.items()):
        if torch.is_tensor(v):
            inputs[k] = v.to(device)

    # MedGemma may start with markdown/code-fence + JSON, so very short generation
    # can truncate at '{"label":"'. Keep a backend-specific minimum budget.
    gen_max_new_tokens = int(max_new_tokens)
    if backend == "medgemma":
        gen_max_new_tokens = max(gen_max_new_tokens, 24)

    gen = base_model.generate(**inputs, max_new_tokens=gen_max_new_tokens, do_sample=False)
    decoded = processor.batch_decode(gen, skip_special_tokens=True)[0]

    # Parse only newly generated continuation when possible (avoid prompt-echo leakage).
    decoded_new = ""
    in_ids = inputs.get("input_ids", None)
    if torch.is_tensor(in_ids) and torch.is_tensor(gen) and gen.dim() == 2 and in_ids.dim() == 2:
        prompt_len = int(in_ids.shape[1])
        if gen.shape[1] > prompt_len:
            new_tok = gen[:, prompt_len:]
            decoded_new = processor.batch_decode(new_tok, skip_special_tokens=True)[0]

    parse_text = decoded_new if (decoded_new or "").strip() else decoded
    pred = _parse_normal_disease(parse_text)
    if pred is None:
        pred = 0
    return int(pred), decoded


# =============================================================================
# Dataset + collate
# =============================================================================
class SingleImageDatasetVLM(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy().reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["filepath"]
        img = Image.open(img_path).convert("RGB")

        modality = str(row["dataset_norm"]).lower()
        label = int(row["binarylabel"])
        sev = str(row["severity_norm"]).lower()

        task_prompt = PROMPT_BY_DATASET.get(
            modality,
            "This is a medical image.\nQuestion: Does this image show normal anatomy or signs of disease?\n\n",
        )
        full_text = SYSTEM_PROMPT_SHORT + "\n\n" + task_prompt

        return {
            "image": img,
            "input_text": full_text,
            "label": label,
            "path": img_path,
            "severity": sev,
            "fileindex": str(row["fileindex"]),
            "modality": modality,
            "corruption": str(row.get("corruption_name", "unknown")),
        }


def make_single_collate_fn_vlm(processor, backend: str):
    def collate(batch):
        images = [b["image"] for b in batch]
        texts  = [b["input_text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        paths  = [b["path"] for b in batch]
        sevs   = [b["severity"] for b in batch]
        fids   = [str(b["fileindex"]) for b in batch]
        mods   = [b["modality"] for b in batch]
        cors   = [str(b.get("corruption", "unknown")) for b in batch]

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
            messages_list = []
            for img, txt in zip(images, texts):
                messages_list.append([{
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": txt}],
                }])
            chat_texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                          for m in messages_list]
            model_inputs = processor(text=chat_texts, images=images, padding=True, return_tensors="pt")
        else:
            model_inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")

        out = dict(model_inputs)
        out["labels_cls"] = labels
        out["paths"] = paths
        out["severities"] = sevs
        out["fileindices"] = fids
        out["modalities"] = mods
        out["corruptions"] = cors
        return out
    return collate


@torch.no_grad()
def forward_vlm_prob(vlm: VLMAdapterWrapper, batch: Dict[str, Any], apply_adapter: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    target_device = next(vlm.adapter.parameters()).device
    batch = to_device(batch, target_device)

    y = batch["labels_cls"]
    d = dict(batch)
    for k in ["labels_cls", "paths", "severities", "fileindices", "modalities", "corruptions"]:
        d.pop(k, None)

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

    h = vlm.adapter(h_base) if apply_adapter else h_base
    logits = vlm.classifier(h)
    prob = torch.softmax(logits.float(), dim=1)[:, 1]
    return y.detach().cpu(), prob.detach().cpu()


def collect_single_vlm(vlm: VLMAdapterWrapper, loader: DataLoader, thr: float, adapter_on: bool):
    all_y, all_p = [], []
    all_sev, all_mod, all_cor = [], [], []
    for batch in loader:
        y, p = forward_vlm_prob(vlm, batch, apply_adapter=adapter_on)
        all_y.append(y); all_p.append(p)
        all_sev.extend(batch["severities"])
        all_mod.extend(batch["modalities"])
        all_cor.extend(batch.get("corruptions", ["unknown"] * len(batch["paths"])))

    y_true = torch.cat(all_y, dim=0)
    y_prob = torch.cat(all_p, dim=0)
    m_all = compute_binary_metrics(y_true, y_prob, thr=thr)
    return m_all, y_true, y_prob, all_sev, all_mod, all_cor


# =============================================================================
# Output schema
# =============================================================================
def make_empty_model_result():
    return {
        "model_type": "vlm",
        "id": None,
        "ckpt": None,
        "config": {
            "backend": None,
            "layer": None,
            "adapter_on_clean": None,
        },
        "single": {},
        "baseline": {
            "single": None,
            "single_by_modality": None,
            "corruption_analysis": None,
            "corruption_analysis_by_modality": None,
            "ce_mce_vs_baseline": None,
            "ce_mce_vs_baseline_by_modality": None,
        },
    }


# =============================================================================
# Main
# =============================================================================
def main():
    # backend/ckpt checks
    if args.backend == "all":
        missing = []
        if not args.vlm_ckpt_qwen3:    missing.append("--vlm_ckpt_qwen3")
        if not args.vlm_ckpt_medgemma: missing.append("--vlm_ckpt_medgemma")
        if not args.vlm_ckpt_internvl: missing.append("--vlm_ckpt_internvl")
        if not args.vlm_ckpt_lingshu:  missing.append("--vlm_ckpt_lingshu")
        if missing:
            raise ValueError("backend=all requires per-backend ckpts: " + ", ".join(missing))
    else:
        if args.vlm_ckpt is None:
            raise ValueError("--vlm_ckpt is required when backend != all")

    # load CSV -> clean/art
    df_raw = load_corrupt_csv_for_eval(args.test_csv, BASE_OUT_DIR, corrupt_key=args.corrupt_key)
    df_clean = make_df_clean_from_clean_filepath(df_raw)
    df_art   = make_df_artifacts_from_filepath(df_raw)

    results = {
        "schema_version": "vlm_baseline_modality_v1",
        "thr": float(args.thr),
        "data": {
            "test_csv": args.test_csv,
            "corrupt_key": args.corrupt_key,
            "n_clean": int(len(df_clean)),
            "n_art": int(len(df_art)),
        },
        "models": {}
    }

    backends = ["qwen3", "medgemma", "internvl", "lingshu"] if args.backend == "all" else [args.backend]
    ckpt_map_all = {
        "qwen3":    args.vlm_ckpt_qwen3,
        "medgemma": args.vlm_ckpt_medgemma,
        "internvl": args.vlm_ckpt_internvl,
        "lingshu":  args.vlm_ckpt_lingshu,
    }

    for backend in backends:
        model_id = MODEL_ID_BY_BACKEND[backend]
        vlm_ckpt = ckpt_map_all[backend] if args.backend == "all" else args.vlm_ckpt

        print("\n==============================", flush=True)
        print(f"== EVAL VLM backend={backend} layer={args.layer}", flush=True)
        print(f"== ckpt={vlm_ckpt}", flush=True)
        print("==============================", flush=True)

        base_model, processor = load_backend(backend, model_id)
        vlm = VLMAdapterWrapper(base_model, backend=backend, layer_choice=args.layer)

        ckpt = torch.load(vlm_ckpt, map_location="cpu")
        vlm.adapter.load_state_dict(ckpt["adapter"], strict=True)
        vlm.classifier.load_state_dict(ckpt["classifier"], strict=True)
        if "pooler" in ckpt:
            vlm.pooler.load_state_dict(ckpt["pooler"], strict=True)
            vlm.use_attention_pooling = True
            print("== detected pooler in ckpt: attention pooling enabled", flush=True)
        vlm.eval()

        ds_clean = SingleImageDatasetVLM(df_clean)
        ds_art   = SingleImageDatasetVLM(df_art)
        collate = make_single_collate_fn_vlm(processor, backend)

        dl_clean = DataLoader(ds_clean, batch_size=1, shuffle=False, collate_fn=collate, pin_memory=(device == "cuda"))
        dl_art   = DataLoader(ds_art,   batch_size=1, shuffle=False, collate_fn=collate, pin_memory=(device == "cuda"))

        # ---- trained (adapter+classifier) ----
        m_clean_all, y_c, p_c, sev_c, mod_c, cor_c = collect_single_vlm(
            vlm, dl_clean, thr=args.thr, adapter_on=bool(args.adapter_on_clean)
        )
        m_art_all, y_w, p_w, sev_w, mod_w, cor_w = collect_single_vlm(
            vlm, dl_art, thr=args.thr, adapter_on=True
        )

        y_all   = torch.cat([y_c, y_w], dim=0)
        p_all   = torch.cat([p_c, p_w], dim=0)
        sev_all = sev_c + sev_w
        mod_all = mod_c + mod_w
        cor_all = cor_c + cor_w

        trained_single = finalize_single_block(len(df_clean), len(df_art), m_clean_all, m_art_all)
        trained_by_mod = finalize_single_by_modality(y_all, p_all, sev_all, mod_all, thr=args.thr)

        trained_rob = compute_corruption_severity_metrics_and_robustness(
            y_true=y_all, y_prob=p_all,
            severities=sev_all, corruptions=cor_all,
            thr=args.thr,
        )
        trained_rob_by_mod = compute_corruption_analysis_by_modality_prob(
            y_true=y_all, y_prob=p_all,
            severities=sev_all, corruptions=cor_all, modalities=mod_all,
            thr=args.thr,
        )

        # ---- prompt baseline (no training) ----
        # InternVL can require a different generation-capable class for .generate().
        if backend == "internvl":
            base_model_prompt, processor_prompt = load_backend_for_prompt(backend, model_id)
        else:
            base_model_prompt, processor_prompt = base_model, processor

        yb_true, yb_pred = [], []
        sev_b, mod_b, cor_b = [], [], []

        def eval_df_prompt(df_part: pd.DataFrame):
            nonlocal yb_true, yb_pred, sev_b, mod_b, cor_b
            for i in range(len(df_part)):
                row = df_part.iloc[i]
                img = Image.open(row["filepath"]).convert("RGB")
                modality = str(row["dataset_norm"]).lower()
                y = int(row["binarylabel"])
                sev = str(row["severity_norm"]).lower()
                cor = str(row.get("corruption_name", "unknown")).lower()

                task_prompt = PROMPT_BY_DATASET.get(
                    modality,
                    "This is a medical image.\nQuestion: Does this image show normal anatomy or signs of disease?\n\n",
                )
                full_text = SYSTEM_PROMPT_JSON + "\n\n" + task_prompt

                pred, _ = forward_vlm_prompt_pred_one(
                    base_model=base_model_prompt,
                    processor=processor_prompt,
                    backend=backend,
                    img=img,
                    text=full_text,
                    device=device,
                    max_new_tokens=int(args.baseline_max_new_tokens),
                )

                yb_true.append(y)
                yb_pred.append(pred)
                sev_b.append(sev)
                mod_b.append(modality)
                cor_b.append(cor)

        eval_df_prompt(df_clean)
        eval_df_prompt(df_art)

        yb_true_t = torch.tensor(yb_true, dtype=torch.long)
        yb_pred_t = torch.tensor(yb_pred, dtype=torch.long)

        clean_mask = np.array([s == "clean" for s in sev_b], dtype=bool)
        art_mask   = ~clean_mask

        m_base_clean = compute_binary_metrics_from_preds_full(
            yb_true_t[torch.tensor(clean_mask)], yb_pred_t[torch.tensor(clean_mask)]
        ) if clean_mask.sum() else {}

        m_base_art = compute_binary_metrics_from_preds_full(
            yb_true_t[torch.tensor(art_mask)], yb_pred_t[torch.tensor(art_mask)]
        ) if art_mask.sum() else {}

        baseline_single = finalize_single_block(len(df_clean), len(df_art), m_base_clean, m_base_art)
        baseline_by_mod = finalize_single_by_modality_from_preds(yb_true_t, yb_pred_t, sev_b, mod_b)

        base_rob = compute_corruption_severity_metrics_from_preds(
            y_true=yb_true_t, y_pred=yb_pred_t,
            severities=sev_b, corruptions=cor_b,
        )
        base_rob_by_mod = compute_corruption_analysis_by_modality_preds(
            y_true=yb_true_t, y_pred=yb_pred_t,
            severities=sev_b, corruptions=cor_b, modalities=mod_b,
        )

        sev_levels_overall = sorted(list(set([s for s in sev_b if s != "clean"])))
        if len(sev_levels_overall) == 0:
            sev_levels_overall = ["0", "1", "2", "3", "4"]

        ce_mce_overall = compute_ce_mce_vs_prompt_baseline(trained_rob, base_rob, sev_levels=tuple(sev_levels_overall))
        ce_mce_by_mod  = compute_ce_mce_by_modality(trained_rob_by_mod, base_rob_by_mod)

        # ---- assemble result ----
        mr = make_empty_model_result()
        mr["id"] = f"{backend}:{args.layer}"
        mr["ckpt"] = vlm_ckpt
        mr["config"].update({
            "backend": backend,
            "layer": args.layer,
            "adapter_on_clean": bool(args.adapter_on_clean),
        })

        mr["single"] = trained_single
        mr["single"]["by_modality"] = trained_by_mod
        mr["single"]["corruption_analysis"] = trained_rob
        mr["single"]["corruption_analysis_by_modality"] = trained_rob_by_mod

        mr["baseline"]["single"] = baseline_single
        mr["baseline"]["single_by_modality"] = baseline_by_mod
        mr["baseline"]["corruption_analysis"] = base_rob
        mr["baseline"]["corruption_analysis_by_modality"] = base_rob_by_mod
        mr["baseline"]["ce_mce_vs_baseline"] = ce_mce_overall
        mr["baseline"]["ce_mce_vs_baseline_by_modality"] = ce_mce_by_mod

        model_key = f"vlm:{backend}:{args.layer}"
        results["models"][model_key] = mr

        # cleanup
        if backend == "internvl":
            del base_model_prompt, processor_prompt
        del vlm, base_model, processor
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # ---- save per-model json (same style) ----
    out_dir = os.path.dirname(os.path.abspath(args.out_json))
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.out_json))[0]

    for model_key, model_result in results["models"].items():
        safe_key = model_key.replace(":", "_").replace("/", "_")
        out_path = os.path.join(out_dir, f"{base_name}_{safe_key}.json")

        out_obj = {
            "schema_version": results["schema_version"],
            "thr": results["thr"],
            "data": results["data"],
            "models": {model_key: model_result}
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)

        print(f"[saved] {out_path}", flush=True)


if __name__ == "__main__":
    main()
