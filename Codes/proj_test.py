

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

# EffNet
import torchvision.transforms as T
import torchvision.models as tvm

from sklearn.metrics import roc_auc_score


# -------------------------
# Args
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--test_pair_csv", type=str, default=None,
                   help="CSV containing clean+weak test set")
    p.add_argument("--test_mnist_csv", type=str, default=None,
                   help="MNIST corruption CSV (dataset,severity,clean_index,filepath,clean_filepath,...)")
    p.add_argument("--mnist_corrupt_key", type=str, default="corruption",
                   choices=["corruption", "corrupt_detail"],
                   help="column name for corruption type in MNIST CSV")

    p.add_argument("--out_json", type=str, required=True, help="output json path")
    p.add_argument("--thr", type=float, default=0.5, help="threshold for Acc/F1 + flip rate")

    # VLM ?ㅼ젙
    p.add_argument("--backend", type=str, required=True, choices=["all","qwen3", "medgemma", "internvl", "lingshu"])
    p.add_argument("--layer", type=str, default=None, choices=["last", "hs_-4"])
    p.add_argument("--pool_mask", type=str, default=None, choices=["none", "attn_mask"])
    p.add_argument("--vlm_ckpt", type=str, default=None, help="path to VLM best.pt")
    # ??all-backend ckpts (explicit per model)
    p.add_argument("--vlm_ckpt_qwen3", type=str, default=None, help="(backend=all) ckpt for qwen3")
    p.add_argument("--vlm_ckpt_medgemma", type=str, default=None, help="(backend=all) ckpt for medgemma")
    p.add_argument("--vlm_ckpt_internvl", type=str, default=None, help="(backend=all) ckpt for internvl")
    p.add_argument("--vlm_ckpt_lingshu", type=str, default=None, help="(backend=all) ckpt for lingshu")
    p.add_argument("--projector_path", type=str, default=None, help="manual projector path override")
    p.add_argument("--batch_size", type=int, default=1)
    return p.parse_args()


args = parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# HF cache
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

PROJECTOR_CANDIDATE_PATHS = {
    "qwen3": [
        "model.visual.merger",
        "visual.merger",
        "model.visual.merger.linear_fc2",
    ],
    "lingshu": [
        "model.visual.merger",
        "visual.merger",
    ],
    "medgemma": [
        "model.multi_modal_projector",
        "multi_modal_projector",
    ],
    "internvl": [
        "multi_modal_projector",
        "model.multi_modal_projector",
    ],
}

BASE_OUT_DIR = "/SAN/ioo/HORIZON/howoon"


# -------------------------
# Common utils
# -------------------------
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
        art_mask = m_mask & (severities != "clean")

        m_clean = compute_binary_metrics(y_true[clean_mask], y_prob[clean_mask], thr=thr) if clean_mask.sum() > 0 else {}
        m_weak  = compute_binary_metrics(y_true[art_mask],  y_prob[art_mask],  thr=thr) if art_mask.sum() > 0 else {}

        auroc_clean = float(m_clean.get("auroc", float("nan")))
        auroc_art   = float(m_weak.get("auroc", float("nan")))
        auroc_macro = float((auroc_clean + auroc_art) / 2.0) if (np.isfinite(auroc_clean) and np.isfinite(auroc_art)) else float("nan")

        out[mod] = {
            "n_clean": int(clean_mask.sum()),
            "n_art": int(art_mask.sum()),
            "auroc_clean": auroc_clean,
            "auroc_art": auroc_art,
            "auroc_macro": auroc_macro,
            "clean_metrics": m_clean,
            "art_metrics": m_weak,
        }
    return out


def worst_prob_gt(p_ws: List[float], y: int) -> float:
    if y == 0:
        return float(np.max(p_ws))
    return float(np.min(p_ws))


def finalize_paired_by_modality(
    y_list: List[int],
    p_clean_list: List[float],
    p_mean_list: List[float],
    p_worst_list: List[float],
    modalities: List[str],
    thr: float,
    worst_metrics_key: str = "art_worst_metrics",
    flip_worst_key: str = "flip_rate_worst",
):
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
        m_mean = compute_binary_metrics(y_t, pm_t, thr=thr)
        m_worst = compute_binary_metrics(y_t, pw_t, thr=thr)

        pred_c = (pc[idx] >= thr).astype(np.int64)
        pred_m = (pm[idx] >= thr).astype(np.int64)
        pred_w = (pw[idx] >= thr).astype(np.int64)

        out[mod] = {
            "n_pairs": int(idx.sum()),
            "clean_metrics": pick_binary_only_metrics(m_clean),
            "art_mean_metrics": pick_binary_only_metrics(m_mean),
            worst_metrics_key: pick_binary_only_metrics(m_worst),
            "flip_rate_mean": float((pred_c != pred_m).mean()) if idx.sum() else float("nan"),
            flip_worst_key: float((pred_c != pred_w).mean()) if idx.sum() else float("nan"),
        }
    return out

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


def compute_binary_metrics_from_optional_preds(y_true_list: List[int], y_pred_list: List[Optional[int]]) -> Dict[str, Any]:
    n_total = int(len(y_true_list))
    valid_idx = [i for i, p in enumerate(y_pred_list) if p in (0, 1)]
    n_valid = int(len(valid_idx))
    n_null = int(n_total - n_valid)
    if n_valid == 0:
        return {
            "acc": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "tp": 0, "tn": 0, "fp": 0, "fn": 0,
            "n_total": n_total,
            "n_valid": n_valid,
            "n_null": n_null,
        }
    yt = torch.tensor([int(y_true_list[i]) for i in valid_idx], dtype=torch.long)
    yp = torch.tensor([int(y_pred_list[i]) for i in valid_idx], dtype=torch.long)
    m = compute_binary_metrics_from_preds(yt, yp)
    m["n_total"] = n_total
    m["n_valid"] = n_valid
    m["n_null"] = n_null
    return m


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


def finalize_single_by_modality_from_preds(
    y_true: torch.Tensor,
    y_pred: List[Optional[int]],
    severities: List[str],
    modalities: List[str],
):
    severities = np.array([str(s).lower() for s in severities], dtype=object)
    modalities = np.array([str(m).lower() for m in modalities], dtype=object)
    y_true_list = y_true.detach().cpu().tolist()

    out = {}
    for mod in sorted(set(modalities.tolist())):
        m_mask = (modalities == mod)
        clean_idx = [i for i, ok in enumerate((m_mask & (severities == "clean")).tolist()) if ok]
        weak_idx = [i for i, ok in enumerate((m_mask & (severities != "clean")).tolist()) if ok]

        m_clean = compute_binary_metrics_from_optional_preds(
            [y_true_list[i] for i in clean_idx],
            [y_pred[i] for i in clean_idx],
        ) if len(clean_idx) > 0 else {}
        m_weak = compute_binary_metrics_from_optional_preds(
            [y_true_list[i] for i in weak_idx],
            [y_pred[i] for i in weak_idx],
        ) if len(weak_idx) > 0 else {}

        out[mod] = {
            "n_clean": int(len(clean_idx)),
            "n_art": int(len(weak_idx)),
            "auroc_clean": float("nan"),
            "auroc_art": float("nan"),
            "auroc_macro": float("nan"),
            "clean_metrics": m_clean,
            "art_metrics": m_weak,
        }
    return out

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

    # ??AUROC (tie ?ы븿 ?뺥솗 泥섎━)
    y_np = y_true.numpy()
    s_np = y_score.numpy()
    if (y_np == 1).sum() == 0 or (y_np == 0).sum() == 0:
        auroc = float("nan")
    else:
        auroc = float(roc_auc_score(y_np, s_np))

    tnr = tn / (tn + fp + eps)
    bacc = 0.5 * (recall + tnr)
    be = 1.0 - bacc

    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(auroc),
        "balanced_acc": float(bacc),
        "balanced_error": float(be),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def pick_binary_only_metrics(m: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(m, dict):
        return {}
    keys = ["acc", "precision", "recall", "f1", "tp", "tn", "fp", "fn"]
    return {k: m[k] for k in keys if k in m}


def _slice_metrics(y_true: torch.Tensor, y_prob: torch.Tensor, mask: np.ndarray, thr: float) -> Dict[str, Any]:
    if mask.sum() == 0:
        return {}
    yt = y_true[torch.tensor(mask, dtype=torch.bool)]
    yp = y_prob[torch.tensor(mask, dtype=torch.bool)]
    return compute_binary_metrics(yt, yp, thr=thr)


def compute_corruption_severity_metrics_and_robustness(
    y_true: torch.Tensor,
    y_prob: torch.Tensor,
    severities: List[str],
    corruptions: List[str],
    thr: float,
):
    sev = np.array([str(s).lower() for s in severities], dtype=object)
    cor = np.array([str(c).lower() for c in corruptions], dtype=object)

    clean_mask = (sev == "clean")
    m_clean = _slice_metrics(y_true, y_prob, clean_mask, thr=thr)
    be_clean = m_clean.get("balanced_error", None)
    f1_clean = m_clean.get("f1", None)

    metrics_by_severity_pooled = {}
    sev_levels = sorted(set([s for s in sev.tolist() if s != "clean"]))
    for s in sev_levels:
        metrics_by_severity_pooled[str(s)] = _slice_metrics(y_true, y_prob, (sev == s), thr=thr)

    metrics_by_corruption = {}
    for c in sorted(set(cor.tolist())):
        m_c_pooled = _slice_metrics(y_true, y_prob, (cor == c) & (sev != "clean"), thr=thr)
        by_s = {}
        for s in sev_levels:
            by_s[str(s)] = _slice_metrics(y_true, y_prob, (cor == c) & (sev == s), thr=thr)
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
        },
    }


def build_per_image_prob_records(
    y_true: torch.Tensor,
    y_prob: torch.Tensor,
    thr: float,
    severities: List[str],
    modalities: List[str],
    paths: List[str],
    fileindices: List[str],
    corruptions: Optional[List[str]] = None,
):
    out = []
    yt = y_true.detach().cpu().tolist()
    yp = y_prob.detach().cpu().tolist()
    for i in range(len(yt)):
        p = float(yp[i])
        out.append({
            "path": str(paths[i]) if i < len(paths) else None,
            "fileindex": str(fileindices[i]) if i < len(fileindices) else None,
            "severity": str(severities[i]) if i < len(severities) else None,
            "modality": str(modalities[i]) if i < len(modalities) else None,
            "y_true": int(yt[i]),
            "score_disease": p,
            "pred_label": int(1 if p >= thr else 0),
            "corruption": (str(corruptions[i]) if (corruptions is not None and i < len(corruptions)) else None),
        })
    return out


def build_per_image_pred_records(
    y_true: torch.Tensor,
    y_pred: List[Optional[int]],
    severities: List[str],
    modalities: List[str],
    paths: List[str],
    fileindices: List[str],
    raws: Optional[List[str]] = None,
):
    out = []
    yt = y_true.detach().cpu().tolist()
    for i in range(len(yt)):
        p = y_pred[i] if i < len(y_pred) else None
        rec = {
            "path": str(paths[i]) if i < len(paths) else None,
            "fileindex": str(fileindices[i]) if i < len(fileindices) else None,
            "severity": str(severities[i]) if i < len(severities) else None,
            "modality": str(modalities[i]) if i < len(modalities) else None,
            "y_true": int(yt[i]),
            "pred_label": (int(p) if p in (0, 1) else None),
        }
        if raws is not None:
            rec["raw_text"] = str(raws[i]) if i < len(raws) else None
        out.append(rec)
    return out



def load_split_csv(path: str, base_out_dir: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "binarylab" in df.columns and "binarylabel" not in df.columns:
        df = df.rename(columns={"binarylab": "binarylabel"})

    # ??severity = null OR "clean" => clean?쇰줈 ?듭씪
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
            base_out_dir,
        )
        s = s.replace("\\", "/")
        return s

    df["filepath"] = df["filepath"].apply(_fix_path)
    df["clean_filepath"] = df["clean_filepath"].apply(_fix_path)
    df["dataset_norm"] = df["dataset"].astype(str).map(modality_from_dataset_name)
    df["severity_int"] = pd.to_numeric(df["severity"], errors="coerce").fillna(-1).astype(int)

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
    out["severity_norm"] = out["severity_int"].astype(int).astype(str)
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
        clean_row = clean_map[fid]
        art_list = [g.iloc[i] for i in range(len(g))]
        items.append((fid, clean_row, art_list))
    return items


def build_pair_index(df_pair: pd.DataFrame):
    """
    Returns list of tuples:
      (fileindex, clean_row, [weak_rows...])
    fileindex??string?????덉쓬 (?? '0_lung diseases') -> int濡?罹먯뒪??湲덉?
    """
    items = []
    for fid, g in df_pair.groupby("fileindex"):
        g = g.reset_index(drop=True)
        clean_rows = g[g["severity_norm"] == "clean"]
        weak_rows  = g[g["severity_norm"] != "clean"]
        if len(clean_rows) == 0 or len(weak_rows) == 0:
            continue
        clean_row = clean_rows.iloc[0]
        weak_list = [weak_rows.iloc[i] for i in range(len(weak_rows))]
        items.append((str(fid), clean_row, weak_list))  # ???ш린
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


def finalize_single_block(n_clean, n_art, m_clean_all, m_weak_all):
    auroc_clean = float(m_clean_all.get("auroc", float("nan")))
    auroc_art   = float(m_weak_all.get("auroc", float("nan")))
    auroc_macro = float((auroc_clean + auroc_art) / 2.0) if (np.isfinite(auroc_clean) and np.isfinite(auroc_art)) else float("nan")
    return {
        "n_clean": int(n_clean),
        "n_art": int(n_art),
        "auroc_clean": auroc_clean,
        "auroc_art": auroc_art,
        "auroc_macro": auroc_macro,
        "clean_metrics": m_clean_all,
        "art_metrics": m_weak_all,
    }


def finalize_paired_block(pair_stats):
    return {
        "n_pairs": int(pair_stats.get("n_pairs", 0)),
        "clean_metrics": dict(pair_stats.get("clean_metrics", {})),
        "art_mean_metrics": dict(pair_stats.get("art_mean_metrics", {})),
        "art_worst_metrics": dict(pair_stats.get("art_worst_metrics", {})),
        "flip_rate_mean": float(pair_stats.get("flip_rate_mean", float("nan"))),
        "flip_rate_worst": float(pair_stats.get("flip_rate_worst", float("nan"))),
        "art_worst_shift_metrics": dict(pair_stats.get("art_worst_shift_metrics", {})),
        "flip_rate_worst_shift": float(pair_stats.get("flip_rate_worst_shift", float("nan"))),
        "label_based": dict(pair_stats.get("label_based", {})),
    }


def resolve_submodule(root: nn.Module, path: str) -> nn.Module:
    cur: Any = root
    for p in path.split("."):
        if p.isdigit():
            cur = cur[int(p)]
        else:
            if not hasattr(cur, p):
                raise AttributeError(f"Missing submodule '{p}' in path '{path}'")
            cur = getattr(cur, p)
    if not isinstance(cur, nn.Module):
        raise TypeError(f"Resolved object is not nn.Module: {type(cur)}")
    return cur


def _module_trainable_param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def find_projector_module(base_model: nn.Module, backend: str, manual_path: Optional[str] = None) -> Tuple[nn.Module, str]:
    candidate_paths: List[str] = []
    if manual_path:
        candidate_paths.append(manual_path)
    candidate_paths.extend(PROJECTOR_CANDIDATE_PATHS.get(backend, []))

    for path in candidate_paths:
        try:
            mod = resolve_submodule(base_model, path)
            if _module_trainable_param_count(mod) > 0:
                return mod, path
        except Exception:
            continue

    scored = []
    for name, mod in base_model.named_modules():
        if not name:
            continue
        lname = name.lower()
        if any(k in lname for k in ["projector", "connector", "merger", "multi_modal"]):
            n_params = _module_trainable_param_count(mod)
            if n_params <= 0:
                continue
            score = 0
            if "projector" in lname:
                score += 4
            if "connector" in lname:
                score += 3
            if "merger" in lname:
                score += 2
            if "multi_modal" in lname:
                score += 1
            scored.append((score, n_params, name, mod))

    if not scored:
        raise ValueError(
            f"Projector module not found for backend={backend}. "
            f"Try --projector_path with one of known paths: {PROJECTOR_CANDIDATE_PATHS.get(backend, [])}"
        )

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, _, best_name, best_mod = scored[0]
    return best_mod, best_name





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
    def __init__(self, base_model, backend: str, layer_choice: str = "last", pool_mask: str = "none"):
        super().__init__()
        self.base_model = base_model
        self.backend = backend
        self.layer_choice = layer_choice
        self.pool_mask = str(pool_mask)

        for p in self.base_model.parameters():
            p.requires_grad = False

        emb = self._get_text_embeddings(self.base_model)
        hidden_dim = emb.weight.shape[1]
        embed_device = emb.weight.device

        self.hidden_dim = hidden_dim
        self.pooler = AttentionPooling(hidden_dim).to(device=embed_device, dtype=torch.float32)
        self.adapter = VisualAdapter(hidden_dim).to(device=embed_device, dtype=torch.float32)
        self.classifier = nn.Linear(hidden_dim, 2).to(device=embed_device, dtype=torch.float32)
        self.use_attention_pooling = True

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
            attn_for_pool = None
            if self.pool_mask == "attn_mask" and attention_mask is not None and attention_mask.dim() == 2:
                t = min(x.shape[1], attention_mask.shape[1])
                x = x[:, :t, :]
                attn_for_pool = attention_mask[:, :t]
            pooled, _ = self.pooler(x, attn_for_pool)
            return pooled

        if attention_mask is None or attention_mask.dim() != 2:
            return x.mean(dim=1)

        Tm = min(x.shape[1], attention_mask.shape[1])
        x = x[:, :Tm, :]
        m = attention_mask[:, :Tm].to(dtype=x.dtype, device=x.device).unsqueeze(-1)
        x_sum = (x * m).sum(dim=1)
        denom = m.sum(dim=1).clamp_min(1.0)
        return x_sum / denom


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
        img_seq_len = getattr(getattr(model, "config", None), "image_seq_length", None)
        if img_seq_len is not None:
            for attr in ["num_image_tokens", "num_image_token", "image_seq_len", "image_seq_length"]:
                if hasattr(processor, attr):
                    try:
                        setattr(processor, attr, int(img_seq_len))
                    except Exception:
                        pass
            tok = getattr(processor, "tokenizer", None)
            if tok is not None:
                for attr in ["num_image_tokens", "num_image_token", "image_seq_len", "image_seq_length"]:
                    if hasattr(tok, attr):
                        try:
                            setattr(tok, attr, int(img_seq_len))
                        except Exception:
                            pass
            print(f"[internvl] processor image tokens set to image_seq_length={int(img_seq_len)}")
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
    For InternVL, use a generation-capable class (AutoModelForImageTextToText),
    because AutoModel(InternVLModel) may not expose .generate().
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

    # First valid answer line wins (e.g., "Answer: normal" then "Explanation: ...").
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
) -> Tuple[Optional[int], str]:
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
    return pred, decoded


def eval_single_prompt_vlm(
    base_model,
    processor,
    backend: str,
    df_clean: pd.DataFrame,
    df_weak: pd.DataFrame,
    max_new_tokens: int,
):
    y_true, y_pred = [], []
    severities, modalities = [], []
    paths, fileindices, raws = [], [], []

    def _run_df(df_part: pd.DataFrame):
        nonlocal y_true, y_pred, severities, modalities, paths, fileindices, raws
        for i in range(len(df_part)):
            row = df_part.iloc[i]
            img = Image.open(row["filepath"]).convert("RGB")
            modality = str(row["dataset_norm"]).lower()
            y = int(row["binarylabel"])
            sev = str(row["severity_norm"]).lower()
            path = str(row["filepath"])
            fid = str(row["fileindex"])

            task_prompt = PROMPT_BY_DATASET.get(
                modality,
                "This is a medical image.\nQuestion: Does this image show normal anatomy or signs of disease?\n\n",
            )
            full_text = SYSTEM_PROMPT_JSON + "\n\n" + task_prompt

            pred, raw = forward_vlm_prompt_pred_one(
                base_model=base_model,
                processor=processor,
                backend=backend,
                img=img,
                text=full_text,
                device=device,
                max_new_tokens=int(max_new_tokens),
            )

            y_true.append(y)
            y_pred.append(pred)
            severities.append(sev)
            modalities.append(modality)
            paths.append(path)
            fileindices.append(fid)
            raws.append(raw)

    _run_df(df_clean)
    _run_df(df_weak)

    y_true_t = torch.tensor(y_true, dtype=torch.long)
    sev_arr = np.array(severities, dtype=object)

    clean_mask = sev_arr == "clean"
    weak_mask = sev_arr == "weak"

    clean_idx = [i for i, ok in enumerate(clean_mask.tolist()) if ok]
    weak_idx = [i for i, ok in enumerate(weak_mask.tolist()) if ok]

    m_clean = compute_binary_metrics_from_optional_preds(
        [y_true[i] for i in clean_idx], [y_pred[i] for i in clean_idx]
    ) if len(clean_idx) else {}
    m_weak = compute_binary_metrics_from_optional_preds(
        [y_true[i] for i in weak_idx], [y_pred[i] for i in weak_idx]
    ) if len(weak_idx) else {}

    return m_clean, m_weak, y_true_t, y_pred, severities, modalities, paths, fileindices, raws


def eval_pairs_prompt_vlm(
    base_model,
    processor,
    backend: str,
    pair_items: List[Tuple[str, pd.Series, List[pd.Series]]],
    thr: float,
    max_new_tokens: int,
):
    y_clean_list = []
    n_pairs = 0
    pair_mods = []

    label_flip_majority = 0
    label_flip_any = 0
    pred_clean_lbl_list, pred_majority_lbl_list, pred_worstgt_lbl_list = [], [], []
    flip_majority_lbl_list, flip_any_lbl_list = [], []
    n_pairs_null_skipped = 0

    for fid, clean_row, weak_rows in pair_items:
        n_pairs += 1

        modality = str(clean_row["dataset_norm"]).lower()
        task_prompt = PROMPT_BY_DATASET.get(modality, PROMPT_BY_DATASET["mri"])
        full_text = SYSTEM_PROMPT_JSON + "\n\n" + task_prompt

        img_c = Image.open(clean_row["filepath"]).convert("RGB")
        y_c = int(clean_row["binarylabel"])
        pred_c, _ = forward_vlm_prompt_pred_one(
            base_model=base_model,
            processor=processor,
            backend=backend,
            img=img_c,
            text=full_text,
            device=device,
            max_new_tokens=int(max_new_tokens),
        )

        weak_preds: List[int] = []
        weak_has_null = False
        for wr in weak_rows:
            img_w = Image.open(wr["filepath"]).convert("RGB")
            pred_w, _ = forward_vlm_prompt_pred_one(
                base_model=base_model,
                processor=processor,
                backend=backend,
                img=img_w,
                text=full_text,
                device=device,
                max_new_tokens=int(max_new_tokens),
            )
            if pred_w not in (0, 1):
                weak_has_null = True
            else:
                weak_preds.append(int(pred_w))

        if pred_c not in (0, 1) or weak_has_null or len(weak_preds) != len(weak_rows):
            n_pairs_null_skipped += 1
            continue

        pred_majority = 1 if (sum(weak_preds) >= (len(weak_preds) / 2)) else 0
        if y_c == 0:
            pred_lbl_worstgt = 1 if any(p == 1 for p in weak_preds) else pred_majority
        else:
            pred_lbl_worstgt = 0 if any(p == 0 for p in weak_preds) else pred_majority

        y_clean_list.append(y_c)
        pair_mods.append(modality)
        pred_clean_lbl_list.append(int(pred_c))
        pred_majority_lbl_list.append(int(pred_majority))
        pred_worstgt_lbl_list.append(int(pred_lbl_worstgt))

        if int(pred_c) != pred_majority:
            label_flip_majority += 1
            flip_majority_lbl_list.append(True)
        else:
            flip_majority_lbl_list.append(False)

        if any(int(pred_c) != p for p in weak_preds):
            label_flip_any += 1
            flip_any_lbl_list.append(True)
        else:
            flip_any_lbl_list.append(False)

    out = {
        "n_pairs": int(n_pairs),
        "n_pairs_null_skipped": int(n_pairs_null_skipped),
        "n_pairs_eval": int(max(0, n_pairs - n_pairs_null_skipped)),
        "label_based": {
            "clean_metrics": compute_binary_metrics_from_preds(
                torch.tensor(y_clean_list, dtype=torch.long),
                torch.tensor(pred_clean_lbl_list, dtype=torch.long),
            ) if len(y_clean_list) else {},
            "art_majority_metrics": compute_binary_metrics_from_preds(
                torch.tensor(y_clean_list, dtype=torch.long),
                torch.tensor(pred_majority_lbl_list, dtype=torch.long),
            ) if len(y_clean_list) else {},
            "art_worst_gt_metrics": compute_binary_metrics_from_preds(
                torch.tensor(y_clean_list, dtype=torch.long),
                torch.tensor(pred_worstgt_lbl_list, dtype=torch.long),
            ) if len(y_clean_list) else {},
            "flip_rate_majority": float(label_flip_majority / max(1, len(y_clean_list))),
            "flip_rate_any": float(label_flip_any / max(1, len(y_clean_list))),
            "n_null_pred_pairs": int(n_pairs_null_skipped),
        },
    }
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
            "modality": modality,  # ??異붽?
            "corruption": str(row["corruption_name"]) if "corruption_name" in row else "none",
        }



def make_single_collate_fn_vlm(processor, backend: str):
    def collate(batch):
        images = [b["image"] for b in batch]
        texts  = [b["input_text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        paths  = [b["path"] for b in batch]
        sevs   = [b["severity"] for b in batch]
        fids = [str(b["fileindex"]) for b in batch]
        mods = [b["modality"] for b in batch]
        corrs = [b["corruption"] for b in batch]


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
        out["modalities"] = mods  # ??異붽?
        out["corruptions"] = corrs
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


def eval_single_loader_vlm(vlm: VLMAdapterWrapper, loader: DataLoader, thr: float, adapter_on: bool):
    all_y, all_p, all_sev = [], [], []
    for batch in loader:
        y, p = forward_vlm_prob(vlm, batch, apply_adapter=adapter_on)
        all_y.append(y); all_p.append(p)
        all_sev.extend(batch["severities"])

    y_true = torch.cat(all_y, dim=0)
    y_prob = torch.cat(all_p, dim=0)

    m_all = compute_binary_metrics(y_true, y_prob, thr=thr)

    all_sev = np.array(all_sev)
    clean_mask = all_sev == "clean"
    weak_mask  = all_sev == "weak"

    m_clean = compute_binary_metrics(y_true[clean_mask], y_prob[clean_mask], thr=thr) if clean_mask.sum() > 0 else {}
    m_weak  = compute_binary_metrics(y_true[weak_mask],  y_prob[weak_mask],  thr=thr) if weak_mask.sum() > 0 else {}
    return m_all, m_clean, m_weak


def eval_pairs_vlm(vlm: VLMAdapterWrapper, processor, backend: str,
                   pair_items: List[Tuple[str, pd.Series, List[pd.Series]]],
                   thr: float, adapter_on_clean: bool, adapter_on_weak: bool = True):
    y_clean_list, p_clean_list = [], []
    p_mean_list = []
    p_worstgt_list = []
    p_worstshift_list = []
    flip_mean = 0
    flip_worstgt = 0
    flip_worstshift = 0
    n_pairs = 0
    pair_mods = []

    # label-based
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
        return out

    for fid, clean_row, weak_rows in pair_items:
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
        for wr in weak_rows:
            img_w = Image.open(wr["filepath"]).convert("RGB")
            batch_w = make_one_batch(img_w, full_text, y_c)
            _, p_w_t = forward_vlm_prob(vlm, batch_w, apply_adapter=adapter_on_weak)
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
            pred_lbl_worstgt = 1 if any(p == 1 for p in weak_preds) else pred_majority
        else:
            pred_lbl_worstgt = 0 if any(p == 0 for p in weak_preds) else pred_majority

        pred_clean_lbl_list.append(pred_c)
        pred_majority_lbl_list.append(pred_majority)
        pred_worstgt_lbl_list.append(pred_lbl_worstgt)

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
    p_mean_t = torch.tensor(p_mean_list, dtype=torch.float32)
    p_worstgt_t = torch.tensor(p_worstgt_list, dtype=torch.float32)
    p_worstshift_t = torch.tensor(p_worstshift_list, dtype=torch.float32)

    m_clean = compute_binary_metrics(y, p_clean_t, thr=thr)
    m_mean = compute_binary_metrics(y, p_mean_t, thr=thr)
    m_worstgt = compute_binary_metrics(y, p_worstgt_t, thr=thr)
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
        y_clean_list,
        p_clean_list,
        p_mean_list,
        p_worstshift_list,
        pair_mods,
        thr=thr,
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
            "modality": modality,  # ??異붽?
        }

@torch.no_grad()
def eval_single_loader_eff(model: nn.Module, loader: DataLoader, thr: float):
    model.eval()
    all_y, all_p, all_sev = [], [], []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        logits = model(x)
        prob = torch.softmax(logits.float(), dim=1)[:, 1]
        all_y.append(y.detach().cpu())
        all_p.append(prob.detach().cpu())
        all_sev.extend(batch["severity"])

    y_true = torch.cat(all_y, dim=0)
    y_prob = torch.cat(all_p, dim=0)

    m_all = compute_binary_metrics(y_true, y_prob, thr=thr)

    all_sev = np.array(all_sev)
    clean_mask = all_sev == "clean"
    weak_mask  = all_sev == "weak"

    m_clean = compute_binary_metrics(y_true[clean_mask], y_prob[clean_mask], thr=thr) if clean_mask.sum() > 0 else {}
    m_weak  = compute_binary_metrics(y_true[weak_mask],  y_prob[weak_mask],  thr=thr) if weak_mask.sum() > 0 else {}
    return m_all, m_clean, m_weak


@torch.no_grad()
def eval_pairs_eff(model: nn.Module, pair_items, eff_tf, img_size: int, thr: float):
    model.eval()
    y_clean_list, p_clean_list = [], []
    p_mean_list = []
    p_worstgt_list = []
    p_worstshift_list = []
    flip_mean = 0
    flip_worstgt = 0
    flip_worstshift = 0
    n_pairs = 0
    pair_mods = []

    # label-based
    label_flip_majority = 0
    label_flip_any = 0
    pred_clean_lbl_list, pred_majority_lbl_list, pred_worstgt_lbl_list = [], [], []
    flip_majority_lbl_list, flip_any_lbl_list = [], []

    for fid, clean_row, weak_rows in pair_items:
        n_pairs += 1

        img_c = eff_tf(Image.open(clean_row["filepath"]).convert("RGB")).unsqueeze(0).to(device)
        y_c = int(clean_row["binarylabel"])

        logits_c = model(img_c)
        p_clean = float(torch.softmax(logits_c.float(), dim=1)[:, 1].item())

        p_ws = []
        for wr in weak_rows:
            img_w = eff_tf(Image.open(wr["filepath"]).convert("RGB")).unsqueeze(0).to(device)
            logits_w = model(img_w)
            p_ws.append(float(torch.softmax(logits_w.float(), dim=1)[:, 1].item()))

        p_mean = float(np.mean(p_ws))
        p_worst_gt = worst_prob_gt(p_ws, y_c)
        p_worst_shift = float(p_ws[int(np.argmax(np.abs(np.array(p_ws) - p_clean)))])

        modality = str(clean_row["dataset_norm"]).lower()
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
            pred_lbl_worstgt = 1 if any(p == 1 for p in weak_preds) else pred_majority
        else:
            pred_lbl_worstgt = 0 if any(p == 0 for p in weak_preds) else pred_majority

        pred_clean_lbl_list.append(pred_c)
        pred_majority_lbl_list.append(pred_majority)
        pred_worstgt_lbl_list.append(pred_lbl_worstgt)

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
    p_mean_t = torch.tensor(p_mean_list, dtype=torch.float32)
    p_worstgt_t = torch.tensor(p_worstgt_list, dtype=torch.float32)
    p_worstshift_t = torch.tensor(p_worstshift_list, dtype=torch.float32)

    m_clean = compute_binary_metrics(y, p_clean_t, thr=thr)
    m_mean = compute_binary_metrics(y, p_mean_t, thr=thr)
    m_worstgt = compute_binary_metrics(y, p_worstgt_t, thr=thr)
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
        y_clean_list,
        p_clean_list,
        p_mean_list,
        p_worstshift_list,
        pair_mods,
        thr=thr,
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
def collect_single_vlm(vlm: VLMAdapterWrapper, loader: DataLoader, thr: float, adapter_on: bool):
    all_y, all_p = [], []
    all_sev, all_mod = [], []
    all_paths, all_fids = [], []
    all_corr = []
    for batch in loader:
        y, p = forward_vlm_prob(vlm, batch, apply_adapter=adapter_on)
        all_y.append(y); all_p.append(p)
        all_sev.extend(batch["severities"])
        all_mod.extend(batch["modalities"])
        all_paths.extend(batch["paths"])
        all_fids.extend(batch["fileindices"])
        all_corr.extend(batch.get("corruptions", ["none"] * len(batch["paths"])))

    y_true = torch.cat(all_y, dim=0)
    y_prob = torch.cat(all_p, dim=0)
    m_all = compute_binary_metrics(y_true, y_prob, thr=thr)
    return m_all, y_true, y_prob, all_sev, all_mod, all_paths, all_fids, all_corr

def collect_single_eff(model: nn.Module, loader: DataLoader, thr: float):
    model.eval()
    all_y, all_p = [], []
    all_sev, all_mod = [], []
    all_paths, all_fids = [], []

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        logits = model(x)
        prob = torch.softmax(logits.float(), dim=1)[:, 1]

        all_y.append(y.detach().cpu())
        all_p.append(prob.detach().cpu())
        all_sev.extend(batch["severity"])
        all_mod.extend(batch["modality"])
        all_paths.extend(batch["path"])
        all_fids.extend(batch["fileindex"])

    y_true = torch.cat(all_y, dim=0)
    y_prob = torch.cat(all_p, dim=0)
    m_all = compute_binary_metrics(y_true, y_prob, thr=thr)
    return m_all, y_true, y_prob, all_sev, all_mod, all_paths, all_fids


def maybe_load_projector_from_ckpt(
    base_model: nn.Module,
    backend: str,
    ckpt: Dict[str, Any],
    manual_projector_path: Optional[str] = None,
) -> Optional[str]:
    projector_state = ckpt.get("projector", None)
    if not isinstance(projector_state, dict):
        return None

    projector_path = manual_projector_path or ckpt.get("projector_path", None)
    proj_mod, resolved = find_projector_module(base_model, backend=backend, manual_path=projector_path)
    proj_mod.load_state_dict(projector_state, strict=True)
    return resolved


# =============================================================================
# Main
# =============================================================================
def main():
    if (args.test_pair_csv is None and args.test_mnist_csv is None) or (args.test_pair_csv is not None and args.test_mnist_csv is not None):
        raise ValueError("Set exactly one of --test_pair_csv or --test_mnist_csv")

    if args.backend == "all":
        missing = []
        if not args.vlm_ckpt_qwen3:
            missing.append("--vlm_ckpt_qwen3")
        if not args.vlm_ckpt_medgemma:
            missing.append("--vlm_ckpt_medgemma")
        if not args.vlm_ckpt_internvl:
            missing.append("--vlm_ckpt_internvl")
        if not args.vlm_ckpt_lingshu:
            missing.append("--vlm_ckpt_lingshu")
        if missing:
            raise ValueError("backend=all requires per-backend ckpts: " + ", ".join(missing))
    else:
        if args.vlm_ckpt is None:
            raise ValueError("--vlm_ckpt is required when backend != all")

    if args.test_mnist_csv is not None:
        eval_mode = "mnist_corrupt_csv"
        df_raw = load_corrupt_csv_for_eval(args.test_mnist_csv, BASE_OUT_DIR, corrupt_key=args.mnist_corrupt_key)
        df_clean = make_df_clean_from_clean_filepath(df_raw)
        df_pair_weak = make_df_artifacts_from_filepath(df_raw)
        pair_items = build_pair_index_from_cleanindex(df_clean, df_pair_weak)
    else:
        eval_mode = "pair_csv"
        df_pair = load_split_csv(args.test_pair_csv, BASE_OUT_DIR)
        df_clean = df_pair[df_pair["severity_norm"] == "clean"].reset_index(drop=True)
        df_pair_weak = df_pair[df_pair["severity_norm"] != "clean"].reset_index(drop=True)
        pair_items = build_pair_index(df_pair)

    # 출력 JSON 스키마
    results = {
        "schema_version": "1.0",
        "thr": float(args.thr),
        "data": {
            "eval_mode": eval_mode,
            "test_pair_csv": args.test_pair_csv,
            "test_mnist_csv": args.test_mnist_csv,
            "mnist_corrupt_key": args.mnist_corrupt_key if args.test_mnist_csv is not None else None,
            "n_clean": int(len(df_clean)),
            "n_art": int(len(df_pair_weak)),
            "n_pairs": int(len(pair_items)),
        },
        "models": {}
    }

    backends = ["qwen3", "medgemma", "internvl", "lingshu"] if args.backend == "all" else [args.backend]
    ckpt_map_all = {
        "qwen3": args.vlm_ckpt_qwen3,
        "medgemma": args.vlm_ckpt_medgemma,
        "internvl": args.vlm_ckpt_internvl,
        "lingshu": args.vlm_ckpt_lingshu,
    }

    for backend in backends:
        model_id = MODEL_ID_BY_BACKEND[backend]
        vlm_ckpt = ckpt_map_all[backend] if args.backend == "all" else args.vlm_ckpt
        ckpt = torch.load(vlm_ckpt, map_location="cpu")

        layer_choice = args.layer or str(ckpt.get("layer_choice", "last"))
        pool_mask = args.pool_mask or str(ckpt.get("pool_mask", "none"))
        has_adapter = isinstance(ckpt.get("adapter", None), dict)

        print("\n==============================")
        print(f"== EVAL VLM backend={backend} layer={layer_choice}")
        print(f"== ckpt={vlm_ckpt}")
        print("==============================")

        base_model, processor = load_backend(backend, model_id)
        vlm = VLMAdapterWrapper(base_model, backend=backend, layer_choice=layer_choice, pool_mask=pool_mask)

        if has_adapter:
            vlm.adapter.load_state_dict(ckpt["adapter"], strict=True)
        vlm.classifier.load_state_dict(ckpt["classifier"], strict=True)
        if "pooler" in ckpt and isinstance(ckpt["pooler"], dict):
            vlm.pooler.load_state_dict(ckpt["pooler"], strict=True)

        projector_loaded = None
        try:
            projector_loaded = maybe_load_projector_from_ckpt(
                base_model=base_model,
                backend=backend,
                ckpt=ckpt,
                manual_projector_path=args.projector_path,
            )
            if projector_loaded is not None:
                print(f"== projector loaded: {projector_loaded}")
        except Exception as e:
            print(f"[WARN] projector load skipped: {e}")

        vlm.eval()

        ds_clean = SingleImageDatasetVLM(df_clean)
        ds_weak = SingleImageDatasetVLM(df_pair_weak)

        collate = make_single_collate_fn_vlm(processor, backend)
        dl_clean = DataLoader(ds_clean, batch_size=args.batch_size, shuffle=False, collate_fn=collate, pin_memory=(device == "cuda"))
        dl_weak = DataLoader(ds_weak, batch_size=args.batch_size, shuffle=False, collate_fn=collate, pin_memory=(device == "cuda"))

        m_clean_all, y_c, p_c, sev_c, mod_c, path_c, fid_c, cor_c = collect_single_vlm(
            vlm, dl_clean, thr=args.thr, adapter_on=has_adapter
        )
        m_weak_all, y_w, p_w, sev_w, mod_w, path_w, fid_w, cor_w = collect_single_vlm(
            vlm, dl_weak, thr=args.thr, adapter_on=has_adapter
        )

        y_all = torch.cat([y_c, y_w], dim=0)
        p_all = torch.cat([p_c, p_w], dim=0)
        sev_all = sev_c + sev_w
        mod_all = mod_c + mod_w
        path_all = path_c + path_w
        fid_all = fid_c + fid_w
        cor_all = cor_c + cor_w

        single_by_mod = finalize_single_by_modality(y_all, p_all, sev_all, mod_all, thr=args.thr)
        pair_stats_vlm = eval_pairs_vlm(
            vlm=vlm,
            processor=processor,
            backend=backend,
            pair_items=pair_items,
            thr=args.thr,
            adapter_on_clean=has_adapter,
            adapter_on_weak=has_adapter,
        )

        mr = make_empty_model_result()
        mr["model_type"] = "vlm_adapter" if has_adapter else "vlm_proj"
        mr["id"] = f"{backend}:{layer_choice}"
        mr["ckpt"] = vlm_ckpt
        mr["config"].update({
            "backend": backend,
            "layer": layer_choice,
            "adapter_on_clean": bool(has_adapter),
        })
        mr["config"]["pool_mask"] = pool_mask
        mr["config"]["projector_path_loaded"] = projector_loaded

        mr["single"] = finalize_single_block(len(df_clean), len(df_pair_weak), m_clean_all, m_weak_all)
        mr["single"]["by_modality"] = single_by_mod
        if args.test_mnist_csv is not None:
            mr["single"]["corruption_analysis"] = compute_corruption_severity_metrics_and_robustness(
                y_true=y_all,
                y_prob=p_all,
                severities=sev_all,
                corruptions=cor_all,
                thr=args.thr,
            )
        mr["single"]["per_image"] = build_per_image_prob_records(
            y_true=y_all,
            y_prob=p_all,
            thr=args.thr,
            severities=sev_all,
            modalities=mod_all,
            paths=path_all,
            fileindices=fid_all,
            corruptions=cor_all,
        )

        mr["paired"] = finalize_paired_block(pair_stats_vlm)
        mr["paired"]["by_modality"] = pair_stats_vlm.get("by_modality", {})
        mr["paired"]["by_modality_label_based"] = pair_stats_vlm.get("by_modality_label_based", {})
        mr["paired"]["by_modality_worst_shift"] = pair_stats_vlm.get("by_modality_worst_shift", {})

        results["models"][f"{mr['model_type']}:{backend}:{layer_choice}"] = mr

        del vlm, base_model, processor
        gc.collect()
        torch.cuda.empty_cache()

    out_dir = os.path.dirname(os.path.abspath(args.out_json))
    os.makedirs(out_dir, exist_ok=True)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[saved] {args.out_json}")



if __name__ == "__main__":
    main()

