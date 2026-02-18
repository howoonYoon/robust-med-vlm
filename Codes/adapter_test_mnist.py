#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Eval script (VLM adapter+classifier and/or EfficientNet) for NEW corruption CSV format.

CSV format (minimum columns):
  dataset, corruption(or corrupt_detail), severity(0..4), clean_index, label, filepath, clean_filepath

Rules:
- severity 0..4 are ALL artifact images (severity=0 is still artifact)
- clean image is referenced by clean_filepath for each clean_index
- binary label is in column "label" (0/1)
- dataset may mix multiple datasets (e.g., octmnist, pneumoniamnist)
- corruption types may differ by dataset; we group by the corruption string directly.

Outputs per-model JSON (one file per model_key like your previous style) including:
- single clean / artifact metrics
- by modality (clean vs artifact)
- corruption analysis:
    - metrics_by_severity_pooled (over all corruptions)
    - metrics_by_corruption: pooled + by severity
    - robustness_scores:
        per corruption: rBE_no_baseline_mean, dF1_mean
        overall: rBE_over_corruptions, dF1_over_corruptions
- paired metrics:
    - clean vs artifacts (mean/worst prob + label-based majority/worst_gt)
    - NOTE: prob-based worst is GT-aware (normal: max prob is worst, disease: min prob is worst)

Run examples:
  # VLM only (single backend)
  python eval_newcsv.py --test_csv /path/new.csv --out_json /path/out.json \
    --run_vlm --backend qwen3 --layer last --vlm_ckpt /path/best.pt --adapter_on_clean

  # EffNet only
  python eval_newcsv.py --test_csv /path/new.csv --out_json /path/out.json \
    --run_effnet --eff_ckpt /path/eff_best.pt --eff_img_size 224

  # VLM all backends
  python eval_newcsv.py --test_csv /path/new.csv --out_json /path/out.json \
    --run_vlm --backend all --layer last \
    --vlm_ckpt_qwen3 /p/qwen.pt --vlm_ckpt_medgemma /p/med.pt \
    --vlm_ckpt_internvl /p/intern.pt --vlm_ckpt_lingshu /p/ling.pt
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# EffNet
import torchvision.transforms as T
import torchvision.models as tvm

from sklearn.metrics import roc_auc_score


# -------------------------
# Args
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()

    # data (new format)
    p.add_argument("--test_csv", type=str, required=True,
                   help="(new format) one CSV with severity 0..4 artifacts + clean_filepath per clean_index")
    p.add_argument("--corrupt_key", type=str, default="corruption",
                   choices=["corruption", "corrupt_detail"],
                   help="Which column to use as corruption name")

    # output
    p.add_argument("--out_json", type=str, required=True, help="output json base path")
    p.add_argument("--thr", type=float, default=0.5, help="threshold for Acc/F1 + flip rate")

    # run switches
    p.add_argument("--run_vlm", action="store_true", help="evaluate VLM adapter+classifier")
    p.add_argument("--run_effnet", action="store_true", help="evaluate EfficientNet classifier")

    # VLM settings
    p.add_argument("--backend", type=str, default=None, choices=["all", "qwen3", "medgemma", "internvl", "lingshu"])
    p.add_argument("--layer", type=str, default="last", choices=["last", "hs_-4"])
    p.add_argument("--vlm_ckpt", type=str, default=None, help="path to VLM best.pt (adapter + classifier)")

    # all-backend ckpts
    p.add_argument("--vlm_ckpt_qwen3", type=str, default=None)
    p.add_argument("--vlm_ckpt_medgemma", type=str, default=None)
    p.add_argument("--vlm_ckpt_internvl", type=str, default=None)
    p.add_argument("--vlm_ckpt_lingshu", type=str, default=None)

    p.add_argument("--adapter_on_clean", action="store_true",
                   help="apply adapter to clean images too (recommended for fair evaluation)")

    # EffNet settings
    p.add_argument("--eff_ckpt", type=str, default=None, help="path to EffNet best.pt")
    p.add_argument("--eff_variant", type=str, default="effv2_l",
                   choices=["effv2_s", "effv2_m", "effv2_l",
                            "eff_b0", "eff_b1", "eff_b2", "eff_b3", "eff_b4", "eff_b5", "eff_b6", "eff_b7"])
    p.add_argument("--eff_img_size", type=int, default=224, help="EffNet eval resize")
    return p.parse_args()


args = parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device, flush=True)

# HF cache (VLM)
if args.run_vlm:
    assert os.environ.get("HF_HOME"), "HF_HOME not set"
    hf_home = os.environ["HF_HOME"]
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(hf_home, "hub"))


# -------------------------
# Prompts (VLM)
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

MODEL_ID_BY_BACKEND = {
    "qwen3":    "Qwen/Qwen3-VL-8B-Instruct",
    "medgemma": "google/medgemma-1.5-4b-it",
    "internvl": "OpenGVLab/InternVL3_5-8B-HF",
    "lingshu":  "lingshu-medical-mllm/Lingshu-7B",
}

BASE_OUT_DIR = "/SAN/ioo/HORIZON/howoon"


# -------------------------
# Metrics + analysis utils
# -------------------------
def worst_prob_gt(p_ws: List[float], y: int) -> float:
    # p = P(disease)
    if y == 0:
        return float(np.max(p_ws))
    return float(np.min(p_ws))


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

        m_clean = compute_binary_metrics(y_true[clean_mask], y_prob[clean_mask], thr=thr) if clean_mask.sum() > 0 else {}
        m_art   = compute_binary_metrics(y_true[art_mask],   y_prob[art_mask],   thr=thr) if art_mask.sum() > 0 else {}

        auroc_clean = float(m_clean.get("auroc", float("nan")))
        auroc_art   = float(m_art.get("auroc", float("nan")))
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


def finalize_paired_by_modality(y_list, p_clean_list, p_mean_list, p_worst_list,
                                modalities, thr: float,
                                worst_metrics_key: str = "art_worst_metrics",
                                flip_worst_key: str = "flip_rate_worst"):
    y = np.array(y_list, dtype=np.int64)
    pc = np.array(p_clean_list, dtype=np.float32)
    pm = np.array(p_mean_list, dtype=np.float32)
    pw = np.array(p_worst_list, dtype=np.float32)
    mods = np.array([str(m).lower() for m in modalities], dtype=object)

    out = {}
    for mod in sorted(set(mods.tolist())):
        idx = (mods == mod)
        if idx.sum() == 0:
            continue

        y_t = torch.tensor(y[idx], dtype=torch.long)
        pc_t = torch.tensor(pc[idx], dtype=torch.float32)
        pm_t = torch.tensor(pm[idx], dtype=torch.float32)
        pw_t = torch.tensor(pw[idx], dtype=torch.float32)

        m_clean = compute_binary_metrics(y_t, pc_t, thr=thr)
        m_mean  = compute_binary_metrics(y_t, pm_t, thr=thr)
        m_worst = compute_binary_metrics(y_t, pw_t, thr=thr)

        pred_c = (pc[idx] >= thr).astype(np.int64)
        pred_m = (pm[idx] >= thr).astype(np.int64)
        pred_w = (pw[idx] >= thr).astype(np.int64)
        flip_mean = float((pred_c != pred_m).mean()) if idx.sum() else float("nan")
        flip_worst = float((pred_c != pred_w).mean()) if idx.sum() else float("nan")

        out[mod] = {
            "n_pairs": int(idx.sum()),
            "clean_metrics": pick_binary_only_metrics(m_clean),
            "art_mean_metrics": pick_binary_only_metrics(m_mean),
            worst_metrics_key: pick_binary_only_metrics(m_worst),
            "flip_rate_mean": flip_mean,
            flip_worst_key: flip_worst,
        }
    return out


def finalize_paired_label_by_modality(
    y_list: List[int],
    pred_clean_list: List[int],
    pred_majority_list: List[int],
    pred_worstgt_list: List[int],
    flip_majority_list: List[bool],
    flip_any_list: List[bool],
    pair_mods: List[str],
):
    y = np.array(y_list, dtype=np.int64)
    pc = np.array(pred_clean_list, dtype=np.int64)
    pm = np.array(pred_majority_list, dtype=np.int64)
    pw = np.array(pred_worstgt_list, dtype=np.int64)
    fm = np.array(flip_majority_list, dtype=np.bool_)
    fa = np.array(flip_any_list, dtype=np.bool_)
    mods = np.array([str(m).lower() for m in pair_mods], dtype=object)

    out = {}
    for mod in sorted(set(mods.tolist())):
        idx = (mods == mod)
        if idx.sum() == 0:
            continue

        y_t = torch.tensor(y[idx], dtype=torch.long)
        pc_t = torch.tensor(pc[idx], dtype=torch.long)
        pm_t = torch.tensor(pm[idx], dtype=torch.long)
        pw_t = torch.tensor(pw[idx], dtype=torch.long)

        out[mod] = {
            "n_pairs": int(idx.sum()),
            "clean_metrics": compute_binary_metrics_from_preds(y_t, pc_t),
            "art_majority_metrics": compute_binary_metrics_from_preds(y_t, pm_t),
            "art_worst_gt_metrics": compute_binary_metrics_from_preds(y_t, pw_t),
            "flip_rate_majority": float(fm[idx].mean()) if idx.sum() else float("nan"),
            "flip_rate_any": float(fa[idx].mean()) if idx.sum() else float("nan"),
        }
    return out


def _slice_metrics(y_true: torch.Tensor, y_prob: torch.Tensor, mask: np.ndarray, thr: float) -> Dict[str, Any]:
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

    # clean baseline
    clean_mask = (sev == "clean")
    m_clean = _slice_metrics(y_true, y_prob, clean_mask, thr=thr)
    be_clean = m_clean.get("balanced_error", None)
    f1_clean = m_clean.get("f1", None)

    # severity pooled (over corruptions)
    metrics_by_severity_pooled = {}
    sev_levels = sorted(set([s for s in sev.tolist() if s != "clean"]))
    for s in sev_levels:
        metrics_by_severity_pooled[str(s)] = _slice_metrics(y_true, y_prob, (sev == s), thr=thr)

    # corruption -> pooled + by severity
    metrics_by_corruption = {}
    for c in sorted(set(cor.tolist())):
        # artifacts only
        m_c_pooled = _slice_metrics(y_true, y_prob, (cor == c) & (sev != "clean"), thr=thr)
        by_s = {}
        for s in sev_levels:
            by_s[str(s)] = _slice_metrics(y_true, y_prob, (cor == c) & (sev == s), thr=thr)
        metrics_by_corruption[c] = {"pooled": m_c_pooled, "by_severity": by_s}

    # robustness scores
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
            sc["rBE_no_baseline_by_severity"] = {str(sev_levels[i]): float(be_diffs[i]) for i in range(len(be_diffs))}
            rbe_vals.append(sc["rBE_no_baseline_mean"])
        if len(f1_drops) > 0:
            sc["dF1_mean"] = float(np.mean(f1_drops))
            sc["dF1_by_severity"] = {str(sev_levels[i]): float(f1_drops[i]) for i in range(len(f1_drops))}
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


# -------------------------
# New CSV loader + grouping
# -------------------------
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

    need = ["dataset", "severity", "clean_index", "binarylabel", "filepath", "clean_filepath", corrupt_key]
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

    # binarylabel 우선 사용 (없으면 label로 fallback)
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
            "fileindex": str(cid),              # reuse expected field
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


def build_pair_index_from_cleanindex(df_clean: pd.DataFrame, df_art: pd.DataFrame):
    clean_map = {str(r["fileindex"]): r for _, r in df_clean.iterrows()}
    items = []
    for fid, g in df_art.groupby("fileindex"):
        fid = str(fid)
        if fid not in clean_map:
            continue
        clean_row = clean_map[fid]  # Series
        art_list = [g.iloc[i] for i in range(len(g))]
        items.append((fid, clean_row, art_list))
    return items


# -------------------------
# JSON schema (fixed)
# -------------------------
def make_empty_model_result():
    return {
        "model_type": None,
        "id": None,
        "ckpt": None,
        "config": {
            "backend": None,
            "layer": None,
            "adapter_on_clean": None,
            "eff_variant": None,
            "eff_img_size": None,
            "hidden": None,
            "freeze_backbone": None,
        },
        "single": {
            "n_clean": None,
            "n_art": None,
            "auroc_clean": None,
            "auroc_art": None,
            "auroc_macro": None,
            "clean_metrics": None,
            "art_metrics": None,
        },
        "paired": {
            "n_pairs": None,
            "clean_metrics": None,
            "art_mean_metrics": None,
            "art_worst_metrics": None,
            "flip_rate_mean": None,
            "flip_rate_worst": None,
            "art_worst_shift_metrics": None,
            "flip_rate_worst_shift": None,
            "label_based": {
                "clean_metrics": None,
                "art_majority_metrics": None,
                "art_worst_gt_metrics": None,
                "flip_rate_majority": None,
                "flip_rate_any": None,
            },
        },
    }


def finalize_single_block(n_clean, n_art, m_clean_all, m_art_all):
    auroc_clean = float(m_clean_all.get("auroc", float("nan")))
    auroc_art   = float(m_art_all.get("auroc", float("nan")))
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


def finalize_paired_block(pair_stats):
    return {
        "n_pairs": int(pair_stats.get("n_pairs", 0)),
        "clean_metrics": dict(pair_stats.get("clean_metrics", {})),
        "art_mean_metrics": dict(pair_stats.get("art_mean_metrics", {})),
        "art_worst_metrics": dict(pair_stats.get("art_worst_metrics", {})),
        "flip_rate_mean": float(pair_stats.get("flip_rate_mean", float("nan"))),
        "flip_rate_worst": float(pair_stats.get("flip_rate_worst", float("nan"))),
        "label_based": dict(pair_stats.get("label_based", {})),
        "art_worst_shift_metrics": dict(pair_stats.get("art_worst_shift_metrics", {})),
        "flip_rate_worst_shift": float(pair_stats.get("flip_rate_worst_shift", float("nan"))),
    }


# =============================================================================
# VLM PART
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

    common = dict(
        torch_dtype=torch_dtype,
        device_map=None,
        low_cpu_mem_usage=True,
    )

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
    for k in ["labels_cls", "paths", "severities", "fileindices", "labels_token", "modalities", "corruptions"]:
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


def eval_pairs_vlm(vlm: VLMAdapterWrapper, processor, backend: str,
                   pair_items: List[Tuple[str, pd.Series, List[pd.Series]]],
                   thr: float, adapter_on_clean: bool):
    y_clean_list, p_clean_list = [], []
    p_mean_list = []
    flip_mean = 0
    n_pairs = 0
    pair_mods = []

    p_worstgt_list = []
    p_worstshift_list = []
    flip_worstgt = 0
    flip_worstshift = 0

    label_flip_majority = 0
    label_flip_any = 0
    pred_clean_lbl_list, pred_majority_lbl_list, pred_worstgt_lbl_list = [], [], []
    flip_majority_lbl_list, flip_any_lbl_list = [], []

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
            model_inputs = processor(text=[chat_text], images=[img], padding=True, return_tensors="pt")
        else:
            model_inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt")

        out = dict(model_inputs)
        out["labels_cls"] = torch.tensor([label], dtype=torch.long)
        out["paths"] = ["<pair>"]
        out["severities"] = ["<pair>"]
        out["fileindices"] = ["<pair>"]
        out["modalities"] = ["<pair>"]
        out["corruptions"] = ["<pair>"]
        return out

    for fid, clean_row, art_rows in pair_items:
        n_pairs += 1

        modality = str(clean_row["dataset_norm"]).lower()
        task_prompt = PROMPT_BY_DATASET.get(modality, PROMPT_BY_DATASET["mri"])
        full_text = SYSTEM_PROMPT_SHORT + "\n\n" + task_prompt

        img_c = Image.open(clean_row["filepath"]).convert("RGB")
        y_c = int(clean_row["binarylabel"])

        batch_c = make_one_batch(img_c, full_text, y_c)
        _, p_c_t = forward_vlm_prob(vlm, batch_c, apply_adapter=adapter_on_clean)
        p_clean = float(p_c_t.item())

        p_ws = []
        for wr in art_rows:
            img_w = Image.open(wr["filepath"]).convert("RGB")
            batch_w = make_one_batch(img_w, full_text, y_c)
            _, p_w_t = forward_vlm_prob(vlm, batch_w, apply_adapter=True)
            p_ws.append(float(p_w_t.item()))

        p_mean = float(np.mean(p_ws))
        p_worst_gt = worst_prob_gt(p_ws, y_c)
        p_worst_shift = float(p_ws[int(np.argmax(np.abs(np.array(p_ws) - p_clean)))])

        y_clean_list.append(y_c)
        p_clean_list.append(p_clean)
        p_mean_list.append(p_mean)
        p_worstgt_list.append(p_worst_gt)
        p_worstshift_list.append(p_worst_shift)
        pair_mods.append(modality)

        pred_c = 1 if p_clean >= thr else 0
        pred_mean = 1 if p_mean >= thr else 0
        pred_prob_worstgt = 1 if p_worst_gt >= thr else 0
        pred_prob_worstshift = 1 if p_worst_shift >= thr else 0
        if pred_c != pred_mean:
            flip_mean += 1
        if pred_c != pred_prob_worstgt:
            flip_worstgt += 1
        if pred_c != pred_prob_worstshift:
            flip_worstshift += 1

        weak_preds = [1 if p >= thr else 0 for p in p_ws]
        pred_majority = 1 if (sum(weak_preds) >= (len(weak_preds) / 2)) else 0
        if y_c == 0:
            pred_worstgt = 1 if any(p == 1 for p in weak_preds) else pred_majority
        else:
            pred_worstgt = 0 if any(p == 0 for p in weak_preds) else pred_majority

        pred_clean_lbl_list.append(pred_c)
        pred_majority_lbl_list.append(pred_majority)
        pred_worstgt_lbl_list.append(pred_worstgt)

        if pred_c != pred_majority:
            label_flip_majority += 1
            flip_majority_lbl_list.append(True)
        else:
            flip_majority_lbl_list.append(False)

        if any(pred_c != p for p in weak_preds):
            label_flip_any += 1
            flip_any_lbl_list.append(True)
        else:
            flip_any_lbl_list.append(False)

    y = torch.tensor(y_clean_list, dtype=torch.long)
    p_clean_t = torch.tensor(p_clean_list, dtype=torch.float32)
    p_mean_t  = torch.tensor(p_mean_list, dtype=torch.float32)
    p_worstgt_t    = torch.tensor(p_worstgt_list, dtype=torch.float32)
    p_worstshift_t = torch.tensor(p_worstshift_list, dtype=torch.float32)

    m_clean = compute_binary_metrics(y, p_clean_t, thr=thr)
    m_mean  = compute_binary_metrics(y, p_mean_t,  thr=thr)
    m_worstgt    = compute_binary_metrics(y, p_worstgt_t,    thr=thr)
    m_worstshift = compute_binary_metrics(y, p_worstshift_t, thr=thr)

    out = {
        "n_pairs": int(n_pairs),
        "clean_metrics": pick_binary_only_metrics(m_clean),
        "art_mean_metrics": pick_binary_only_metrics(m_mean),
        "art_worst_metrics": pick_binary_only_metrics(m_worstgt),
        "flip_rate_mean": float(flip_mean / max(1, n_pairs)),
        "flip_rate_worst": float(flip_worstgt / max(1, n_pairs)),
        "art_worst_shift_metrics": pick_binary_only_metrics(m_worstshift),
        "flip_rate_worst_shift": float(flip_worstshift / max(1, n_pairs)),
        "label_based": {
            "clean_metrics": compute_binary_metrics_from_preds(
                torch.tensor(y_clean_list, dtype=torch.long),
                torch.tensor(pred_clean_lbl_list, dtype=torch.long),
            ) if n_pairs else {},
            "art_majority_metrics": compute_binary_metrics_from_preds(
                torch.tensor(y_clean_list, dtype=torch.long),
                torch.tensor(pred_majority_lbl_list, dtype=torch.long),
            ) if n_pairs else {},
            "art_worst_gt_metrics": compute_binary_metrics_from_preds(
                torch.tensor(y_clean_list, dtype=torch.long),
                torch.tensor(pred_worstgt_lbl_list, dtype=torch.long),
            ) if n_pairs else {},
            "flip_rate_majority": float(label_flip_majority / max(1, n_pairs)),
            "flip_rate_any": float(label_flip_any / max(1, n_pairs)),
        },
    }
    out["by_modality"] = finalize_paired_by_modality(
        y_clean_list, p_clean_list, p_mean_list, p_worstgt_list, pair_mods, thr=thr
    )
    out["by_modality_worst_shift"] = finalize_paired_by_modality(
        y_clean_list, p_clean_list, p_mean_list, p_worstshift_list, pair_mods, thr=thr,
        worst_metrics_key="art_worst_shift_metrics",
        flip_worst_key="flip_rate_worst_shift",
    )
    out["by_modality_label_based"] = finalize_paired_label_by_modality(
        y_clean_list,
        pred_clean_lbl_list,
        pred_majority_lbl_list,
        pred_worstgt_lbl_list,
        flip_majority_lbl_list,
        flip_any_lbl_list,
        pair_mods,
    )
    return out


# =============================================================================
# EFFNET PART
# =============================================================================
class EfficientNetBinary(nn.Module):
    def __init__(self, variant: str, hidden: int = 0, freeze_backbone: bool = False):
        super().__init__()

        if variant == "effv2_s":
            self.backbone = tvm.efficientnet_v2_s(weights=tvm.EfficientNet_V2_S_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
        elif variant == "effv2_m":
            self.backbone = tvm.efficientnet_v2_m(weights=tvm.EfficientNet_V2_M_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
        elif variant == "effv2_l":
            self.backbone = tvm.efficientnet_v2_l(weights=tvm.EfficientNet_V2_L_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
        elif variant == "eff_b0":
            self.backbone = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
        elif variant == "eff_b1":
            self.backbone = tvm.efficientnet_b1(weights=tvm.EfficientNet_B1_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
        elif variant == "eff_b2":
            self.backbone = tvm.efficientnet_b2(weights=tvm.EfficientNet_B2_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
        elif variant == "eff_b3":
            self.backbone = tvm.efficientnet_b3(weights=tvm.EfficientNet_B3_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
        elif variant == "eff_b4":
            self.backbone = tvm.efficientnet_b4(weights=tvm.EfficientNet_B4_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
        elif variant == "eff_b5":
            self.backbone = tvm.efficientnet_b5(weights=tvm.EfficientNet_B5_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
        elif variant == "eff_b6":
            self.backbone = tvm.efficientnet_b6(weights=tvm.EfficientNet_B6_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
        elif variant == "eff_b7":
            self.backbone = tvm.efficientnet_b7(weights=tvm.EfficientNet_B7_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
        else:
            raise ValueError(f"Unknown variant: {variant}")

        self.backbone.classifier = nn.Identity()

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if hidden and hidden > 0:
            self.head = nn.Sequential(
                nn.Linear(feat_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 2),
            )
        else:
            self.head = nn.Linear(feat_dim, 2)

        self.feat_dim = feat_dim
        self.hidden = int(hidden)
        self.freeze_backbone = bool(freeze_backbone)

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.head(feat)
        return logits


def make_eff_eval_tf(img_size: int):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


class SingleImageDatasetEff(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.copy().reset_index(drop=True)
        self.tf = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["filepath"]
        img = Image.open(path).convert("RGB")
        x = self.tf(img)
        y = int(row["binarylabel"])
        sev = str(row["severity_norm"]).lower()
        modality = str(row["dataset_norm"]).lower()
        return {
            "x": x,
            "y": torch.tensor(y, dtype=torch.long),
            "severity": sev,
            "fileindex": str(row["fileindex"]),
            "path": path,
            "modality": modality,
            "corruption": str(row.get("corruption_name", "unknown")),
        }


def collect_single_eff(model: nn.Module, loader: DataLoader, thr: float):
    model.eval()
    all_y, all_p = [], []
    all_sev, all_mod, all_cor = [], [], []

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        logits = model(x)
        prob = torch.softmax(logits.float(), dim=1)[:, 1]

        all_y.append(y.detach().cpu())
        all_p.append(prob.detach().cpu())
        all_sev.extend(batch["severity"])
        all_mod.extend(batch["modality"])
        all_cor.extend(batch.get("corruption", ["unknown"] * len(batch["path"])))

    y_true = torch.cat(all_y, dim=0)
    y_prob = torch.cat(all_p, dim=0)
    m_all = compute_binary_metrics(y_true, y_prob, thr=thr)
    return m_all, y_true, y_prob, all_sev, all_mod, all_cor


@torch.no_grad()
def eval_pairs_eff(model: nn.Module, pair_items, eff_tf, thr: float):
    model.eval()
    y_clean_list, p_clean_list = [], []
    p_mean_list = []
    flip_mean = 0
    n_pairs = 0
    pair_mods = []

    p_worstgt_list = []
    p_worstshift_list = []
    flip_worstgt = 0
    flip_worstshift = 0

    label_flip_majority = 0
    label_flip_any = 0
    pred_clean_lbl_list, pred_majority_lbl_list, pred_worstgt_lbl_list = [], [], []
    flip_majority_lbl_list, flip_any_lbl_list = [], []

    for fid, clean_row, art_rows in pair_items:
        n_pairs += 1

        img_c = eff_tf(Image.open(clean_row["filepath"]).convert("RGB")).unsqueeze(0).to(device)
        y_c = int(clean_row["binarylabel"])

        logits_c = model(img_c)
        p_clean = float(torch.softmax(logits_c.float(), dim=1)[:, 1].item())

        p_ws = []
        for wr in art_rows:
            img_w = eff_tf(Image.open(wr["filepath"]).convert("RGB")).unsqueeze(0).to(device)
            logits_w = model(img_w)
            p_ws.append(float(torch.softmax(logits_w.float(), dim=1)[:, 1].item()))

        p_mean = float(np.mean(p_ws))
        p_worst_gt = worst_prob_gt(p_ws, y_c)
        p_worst_shift = float(p_ws[int(np.argmax(np.abs(np.array(p_ws) - p_clean)))])

        y_clean_list.append(y_c)
        p_clean_list.append(p_clean)
        p_mean_list.append(p_mean)
        p_worstgt_list.append(p_worst_gt)
        p_worstshift_list.append(p_worst_shift)
        modality = str(clean_row["dataset_norm"]).lower()
        pair_mods.append(modality)

        pred_c = 1 if p_clean >= thr else 0
        pred_mean = 1 if p_mean >= thr else 0
        pred_prob_worstgt = 1 if p_worst_gt >= thr else 0
        pred_prob_worstshift = 1 if p_worst_shift >= thr else 0
        if pred_c != pred_mean:
            flip_mean += 1
        if pred_c != pred_prob_worstgt:
            flip_worstgt += 1
        if pred_c != pred_prob_worstshift:
            flip_worstshift += 1

        weak_preds = [1 if p >= thr else 0 for p in p_ws]
        pred_majority = 1 if (sum(weak_preds) >= (len(weak_preds) / 2)) else 0
        if y_c == 0:
            pred_worstgt = 1 if any(p == 1 for p in weak_preds) else pred_majority
        else:
            pred_worstgt = 0 if any(p == 0 for p in weak_preds) else pred_majority

        pred_clean_lbl_list.append(pred_c)
        pred_majority_lbl_list.append(pred_majority)
        pred_worstgt_lbl_list.append(pred_worstgt)

        if pred_c != pred_majority:
            label_flip_majority += 1
            flip_majority_lbl_list.append(True)
        else:
            flip_majority_lbl_list.append(False)

        if any(pred_c != p for p in weak_preds):
            label_flip_any += 1
            flip_any_lbl_list.append(True)
        else:
            flip_any_lbl_list.append(False)

    y = torch.tensor(y_clean_list, dtype=torch.long)
    p_clean_t = torch.tensor(p_clean_list, dtype=torch.float32)
    p_mean_t  = torch.tensor(p_mean_list, dtype=torch.float32)
    p_worstgt_t    = torch.tensor(p_worstgt_list, dtype=torch.float32)
    p_worstshift_t = torch.tensor(p_worstshift_list, dtype=torch.float32)

    m_clean = compute_binary_metrics(y, p_clean_t, thr=thr)
    m_mean  = compute_binary_metrics(y, p_mean_t,  thr=thr)
    m_worstgt    = compute_binary_metrics(y, p_worstgt_t,    thr=thr)
    m_worstshift = compute_binary_metrics(y, p_worstshift_t, thr=thr)

    out = {
        "n_pairs": int(n_pairs),
        "clean_metrics": pick_binary_only_metrics(m_clean),
        "art_mean_metrics": pick_binary_only_metrics(m_mean),
        "art_worst_metrics": pick_binary_only_metrics(m_worstgt),
        "flip_rate_mean": float(flip_mean / max(1, n_pairs)),
        "flip_rate_worst": float(flip_worstgt / max(1, n_pairs)),
        "art_worst_shift_metrics": pick_binary_only_metrics(m_worstshift),
        "flip_rate_worst_shift": float(flip_worstshift / max(1, n_pairs)),
        "label_based": {
            "clean_metrics": compute_binary_metrics_from_preds(
                torch.tensor(y_clean_list, dtype=torch.long),
                torch.tensor(pred_clean_lbl_list, dtype=torch.long),
            ) if n_pairs else {},
            "art_majority_metrics": compute_binary_metrics_from_preds(
                torch.tensor(y_clean_list, dtype=torch.long),
                torch.tensor(pred_majority_lbl_list, dtype=torch.long),
            ) if n_pairs else {},
            "art_worst_gt_metrics": compute_binary_metrics_from_preds(
                torch.tensor(y_clean_list, dtype=torch.long),
                torch.tensor(pred_worstgt_lbl_list, dtype=torch.long),
            ) if n_pairs else {},
            "flip_rate_majority": float(label_flip_majority / max(1, n_pairs)),
            "flip_rate_any": float(label_flip_any / max(1, n_pairs)),
        },
    }
    out["by_modality"] = finalize_paired_by_modality(
        y_clean_list, p_clean_list, p_mean_list, p_worstgt_list, pair_mods, thr=thr
    )
    out["by_modality_worst_shift"] = finalize_paired_by_modality(
        y_clean_list, p_clean_list, p_mean_list, p_worstshift_list, pair_mods, thr=thr,
        worst_metrics_key="art_worst_shift_metrics",
        flip_worst_key="flip_rate_worst_shift",
    )
    out["by_modality_label_based"] = finalize_paired_label_by_modality(
        y_clean_list,
        pred_clean_lbl_list,
        pred_majority_lbl_list,
        pred_worstgt_lbl_list,
        flip_majority_lbl_list,
        flip_any_lbl_list,
        pair_mods,
    )
    return out


# =============================================================================
# Main
# =============================================================================
def main():
    if not args.run_vlm and not args.run_effnet:
        raise ValueError("You must set at least one of --run_vlm / --run_effnet")

    if args.run_vlm:
        if args.backend is None:
            raise ValueError("--run_vlm requires --backend")
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

    if args.run_effnet and args.eff_ckpt is None:
        raise ValueError("--run_effnet requires --eff_ckpt")

    # ---- load new CSV and build clean/art/pairs ----
    df_raw = load_corrupt_csv_for_eval(args.test_csv, BASE_OUT_DIR, corrupt_key=args.corrupt_key)
    df_clean = make_df_clean_from_clean_filepath(df_raw)
    df_art = make_df_artifacts_from_filepath(df_raw)
    pair_items = build_pair_index_from_cleanindex(df_clean, df_art)

    # fixed schema output
    results = {
        "schema_version": "1.0",
        "thr": float(args.thr),
        "data": {
            "test_csv": args.test_csv,
            "corrupt_key": args.corrupt_key,
            "n_clean": int(len(df_clean)),
            "n_art": int(len(df_art)),
            "n_pairs": int(len(pair_items)),
        },
        "models": {}
    }

    # -------------------------
    # VLM evaluation
    # -------------------------
    if args.run_vlm:
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

            m_clean_all, y_c, p_c, sev_c, mod_c, cor_c = collect_single_vlm(
                vlm, dl_clean, thr=args.thr, adapter_on=bool(args.adapter_on_clean)
            )
            m_art_all,   y_w, p_w, sev_w, mod_w, cor_w = collect_single_vlm(
                vlm, dl_art, thr=args.thr, adapter_on=True
            )

            y_all   = torch.cat([y_c, y_w], dim=0)
            p_all   = torch.cat([p_c, p_w], dim=0)
            sev_all = sev_c + sev_w
            mod_all = mod_c + mod_w
            cor_all = cor_c + cor_w

            single_by_mod = finalize_single_by_modality(y_all, p_all, sev_all, mod_all, thr=args.thr)

            rob = compute_corruption_severity_metrics_and_robustness(
                y_true=y_all,
                y_prob=p_all,
                severities=sev_all,
                corruptions=cor_all,
                thr=args.thr,
            )

            pair_stats_vlm = eval_pairs_vlm(
                vlm=vlm,
                processor=processor,
                backend=backend,
                pair_items=pair_items,
                thr=args.thr,
                adapter_on_clean=bool(args.adapter_on_clean),
            )

            mr = make_empty_model_result()
            mr["model_type"] = "vlm"
            mr["id"] = f"{backend}:{args.layer}"
            mr["ckpt"] = vlm_ckpt
            mr["config"].update({
                "backend": backend,
                "layer": args.layer,
                "adapter_on_clean": bool(args.adapter_on_clean),
            })
            mr["single"] = finalize_single_block(len(df_clean), len(df_art), m_clean_all, m_art_all)
            mr["single"]["by_modality"] = single_by_mod
            mr["single"]["corruption_analysis"] = rob

            mr["paired"] = finalize_paired_block(pair_stats_vlm)
            mr["paired"]["by_modality"] = pair_stats_vlm.get("by_modality", {})
            mr["paired"]["by_modality_label_based"] = pair_stats_vlm.get("by_modality_label_based", {})
            mr["paired"]["by_modality_worst_shift"] = pair_stats_vlm.get("by_modality_worst_shift", {})

            results["models"][f"vlm:{backend}:{args.layer}"] = mr

            del vlm, base_model, processor
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    # -------------------------
    # EfficientNet evaluation
    # -------------------------
    if args.run_effnet:
        print("\n==============================", flush=True)
        print(f"== EVAL EffNet img_size={args.eff_img_size}", flush=True)
        print(f"== ckpt={args.eff_ckpt}", flush=True)
        print("==============================", flush=True)

        ckpt = torch.load(args.eff_ckpt, map_location="cpu")
        variant = ckpt.get("variant", args.eff_variant)
        hidden = int(ckpt.get("hidden", 0))
        freeze_backbone = bool(ckpt.get("freeze_backbone", False))

        model = EfficientNetBinary(variant=variant, hidden=hidden, freeze_backbone=freeze_backbone).to(device)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()

        eff_tf = make_eff_eval_tf(args.eff_img_size)
        ds_clean = SingleImageDatasetEff(df_clean, transform=eff_tf)
        ds_art   = SingleImageDatasetEff(df_art,   transform=eff_tf)

        dl_clean = DataLoader(ds_clean, batch_size=32, shuffle=False, num_workers=4, pin_memory=(device == "cuda"))
        dl_art   = DataLoader(ds_art,   batch_size=32, shuffle=False, num_workers=4, pin_memory=(device == "cuda"))

        m_clean_all, y_c, p_c, sev_c, mod_c, cor_c = collect_single_eff(model, dl_clean, thr=args.thr)
        m_art_all,   y_w, p_w, sev_w, mod_w, cor_w = collect_single_eff(model, dl_art,   thr=args.thr)

        y_all   = torch.cat([y_c, y_w], dim=0)
        p_all   = torch.cat([p_c, p_w], dim=0)
        sev_all = sev_c + sev_w
        mod_all = mod_c + mod_w
        cor_all = cor_c + cor_w

        single_by_mod = finalize_single_by_modality(y_all, p_all, sev_all, mod_all, thr=args.thr)
        rob = compute_corruption_severity_metrics_and_robustness(
            y_true=y_all,
            y_prob=p_all,
            severities=sev_all,
            corruptions=cor_all,
            thr=args.thr,
        )

        pair_stats_eff = eval_pairs_eff(model, pair_items, eff_tf, thr=args.thr)

        mr = make_empty_model_result()
        mr["model_type"] = "effnet"
        mr["id"] = f"{variant}:{int(args.eff_img_size)}"
        mr["ckpt"] = args.eff_ckpt
        mr["config"].update({
            "eff_variant": variant,
            "eff_img_size": int(args.eff_img_size),
            "hidden": hidden,
            "freeze_backbone": freeze_backbone,
        })
        mr["single"] = finalize_single_block(len(df_clean), len(df_art), m_clean_all, m_art_all)
        mr["single"]["by_modality"] = single_by_mod
        mr["single"]["corruption_analysis"] = rob

        mr["paired"] = finalize_paired_block(pair_stats_eff)
        mr["paired"]["by_modality"] = pair_stats_eff.get("by_modality", {})
        mr["paired"]["by_modality_label_based"] = pair_stats_eff.get("by_modality_label_based", {})
        mr["paired"]["by_modality_worst_shift"] = pair_stats_eff.get("by_modality_worst_shift", {})

        results["models"][f"effnet:{variant}:{int(args.eff_img_size)}"] = mr

        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # -------------------------
    # Save JSON (per-model)
    # -------------------------
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
            "models": {
                model_key: model_result
            }
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)

        print(f"[saved] {out_path}", flush=True)


if __name__ == "__main__":
    main()
