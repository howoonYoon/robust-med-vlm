#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
#os.environ.setdefault("HF_HUB_DISABLE_MMAP", "1")  # safetensors mmap ?? (ENOMEM ??)
assert os.environ.get("HF_HOME"), "HF_HOME not set (use TMPDIR/hf_cache etc.)"
hf_home = os.environ["HF_HOME"]
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(hf_home, "hub"))

import random
import gc
import json
from typing import Optional, Any, Dict, List, Tuple
from datetime import datetime
import argparse

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import roc_auc_score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", type=str, required=True,
                   choices=["all","qwen3", "medgemma", "internvl", "lingshu"])
    p.add_argument("--layer", type=str, default="last", choices=["last", "hs_-4"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--bs", type=int, default=1)
    # ✅ 추가: CSV args (지금 고정값을 default로)
    p.add_argument("--train_csv", type=str, default="/SAN/ioo/HORIZON/howoon/vlm_clean_weak_train_2520.csv")
    p.add_argument("--val_csv",   type=str, default="/SAN/ioo/HORIZON/howoon/vlm_clean_weak_val_540.csv")
    p.add_argument("--train_projector", action="store_true",
                   help="unfreeze and train backend projector module with adapter/classifier")
    p.add_argument("--projector_path", type=str, default=None,
                   help="manual projector module path override (e.g., model.visual.merger)")
    p.add_argument("--projector_lr", type=float, default=None,
                   help="optional LR for projector params (default: same as --lr)")

    return p.parse_args()

args = parse_args()

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================
# 0. ?? ??
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

TEST_ONE_ROW = False   # True? 1?? forward?
csv_path = "/SAN/ioo/HORIZON/howoon"

# TRAIN_CSV = os.path.join(csv_path, "vlm_clean_weak_train_2520.csv")
# VAL_CSV   = os.path.join(csv_path, "vlm_clean_weak_val_540.csv")
TRAIN_CSV = args.train_csv
VAL_CSV   = args.val_csv


MODEL_ID_BY_BACKEND = {
    "qwen3":    "Qwen/Qwen3-VL-8B-Instruct",
    "medgemma": "google/medgemma-1.5-4b-it",
    "internvl": "OpenGVLab/InternVL3_5-8B-HF",
    "lingshu":  "lingshu-medical-mllm/Lingshu-7B",
}

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

BACKENDS = ["qwen3", "medgemma", "internvl", "lingshu"] if args.backend == "all" else [args.backend]

# clean/weak loss weights
w_c = 1.0
w_w = 1.3
lambda_w = 0.2

LR = args.lr
PROJECTOR_LR = args.projector_lr if args.projector_lr is not None else args.lr
EPOCHS = args.epochs
POOL_LAYER_CHOICES = [args.layer]

BATCH_SIZE_DEFAULT = 1
BATCH_SIZE_BY_BACKEND = {
    "internvl": 1,
    "lingshu":  1,
    "qwen3":    1,
    "medgemma": 1,
}

SKIP_BAD_SAMPLES = False

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if device == "cuda":
    torch.cuda.manual_seed_all(SEED)

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

# ============================================================
# 1. Adapter + Classifier
# ============================================================
class VisualAdapter(nn.Module):
    def __init__(self, hidden_dim: int, bottleneck: int = 256):
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck)
        self.act  = nn.ReLU()
        self.up   = nn.Linear(bottleneck, hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        delta = self.up(self.act(self.down(h)))
        return h + delta

# 1. AttentionPooling 수정 (가중치 리턴 추가)
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
        
        attn_weights = F.softmax(attn_logits, dim=1) # (B, T, 1)
        pooled = (x * attn_weights).sum(dim=1)
        return pooled, attn_weights # 가중치도 함께 보냅니다.


class VLMAdapterWrapper(nn.Module):
    """
    base_model: frozen
    adapter + classifier: trainable (fp32)
    """
    def __init__(
        self,
        base_model,
        backend: str,
        layer_choice: str = "last",
        train_projector: bool = False,
        projector_path: Optional[str] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.backend = backend
        self.layer_choice = layer_choice
        self.train_projector = bool(train_projector)
        self.projector_module: Optional[nn.Module] = None
        self.projector_path: Optional[str] = None

        for p in self.base_model.parameters():
            p.requires_grad = False

        if self.train_projector:
            proj_mod, proj_path = find_projector_module(self.base_model, backend=self.backend, manual_path=projector_path)
            for p in proj_mod.parameters():
                p.requires_grad = True
            self.projector_module = proj_mod
            self.projector_path = proj_path
            print(f"[{self.backend}] projector enabled: {self.projector_path} "
                  f"(params={_module_trainable_param_count(proj_mod):,})")
        elif projector_path is not None:
            print(f"[{self.backend}] projector_path provided but --train_projector is off. Ignoring.")

        emb = self._get_text_embeddings(self.base_model)
        hidden_dim = emb.weight.shape[1]
        embed_device = emb.weight.device

        self.hidden_dim = hidden_dim

        # 신규: Pooling 레이어 추가
        self.pooler = AttentionPooling(hidden_dim).to(device=embed_device, dtype=torch.float32)

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

    def extract_features(self, outputs, attention_mask: Optional[torch.Tensor] = None, return_weights: bool = False) -> torch.Tensor:
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
                x = hs[-4]  # (B,T,H)
            elif torch.is_tensor(t) and t.dim() == 3 and t.shape[-1] == H:
                x = t
            else:
                raise ValueError("No usable hidden representation found.")
        else:  # "last"
            if torch.is_tensor(t) and t.dim() == 3 and t.shape[-1] == H:
                x = t
            elif isinstance(hs, (list, tuple)) and len(hs) > 0 and torch.is_tensor(hs[-1]) and hs[-1].dim() == 3 and hs[-1].shape[-1] == H:
                x = hs[-1]
            else:
                raise ValueError("No usable hidden representation found.")

        # if attention_mask is None:
        #     return x.mean(dim=1)

        # if attention_mask.dim() != 2:
        #     return x.mean(dim=1)

        # T = min(x.shape[1], attention_mask.shape[1])
        # x = x[:, :T, :]
        # m = attention_mask[:, :T].to(dtype=x.dtype, device=x.device).unsqueeze(-1)

        # x_sum = (x * m).sum(dim=1)
        # denom = m.sum(dim=1).clamp_min(1.0)
        # return x_sum / denom

        # [수정 포인트] 평균 대신 Attention Pooling 사용
        # self.pooler는 (pooled_features, attn_weights)를 반환하도록 설계되었습니다.
        pooled_h, attn_weights = self.pooler(x, attention_mask)

        if return_weights:
            return pooled_h, attn_weights

        return pooled_h


# ============================================================
# 2. Backend ??
# ============================================================
def load_backend(backend: str, model_id: str):
    import transformers
    from transformers import AutoProcessor

    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    if backend in ["medgemma", "internvl"] and device == "cuda":
        torch_dtype = torch.bfloat16

    common = dict(
        torch_dtype=torch_dtype,
        device_map=None,
        low_cpu_mem_usage=True,
    )

    if backend == "qwen3":
        from transformers import Qwen3VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_id)
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **common)
        model.to(device)
        model.eval()
        return model, processor

    if backend == "lingshu":
        from transformers import Qwen2_5_VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_id)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **common)
        model.to(device)
        model.eval()
        return model, processor

    if backend == "internvl":
        from transformers import AutoModel, AutoProcessor as AP2
        processor = AP2.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True, **common)
        model.to(device)
        model.eval()
        return model, processor

    if backend == "medgemma":
        processor = AutoProcessor.from_pretrained(model_id)

        AutoImageTextToText = getattr(transformers, "AutoModelForImageTextToText", None)
        
        if AutoImageTextToText is not None:
            model = AutoImageTextToText.from_pretrained(model_id, **common)
            model.to(device)
            model.eval()
            return model, processor

        AutoVision2Seq = getattr(transformers, "AutoModelForVision2Seq", None)
        if AutoVision2Seq is not None:
            model = AutoVision2Seq.from_pretrained(model_id, **common)
            model.to(device)
            model.eval()
            return model, processor

        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_id, **common)
        model.to(device)
        model.eval()
        return model, processor

    raise ValueError(f"Unknown backend: {backend}")


# ============================================================
# 3. CSV ??
# ============================================================
base_out_dir = "/SAN/ioo/HORIZON/howoon"

def load_split_csv(path: str, base_out_dir: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "binarylab" in df.columns and "binarylabel" not in df.columns:
        df = df.rename(columns={"binarylab": "binarylabel"})

    df["severity_norm"] = df["severity"].astype(str).str.lower()
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
        raise ValueError(f"{path} 에 fileindex 컬럼 필요함")

    return df.reset_index(drop=True)

train_df = load_split_csv(TRAIN_CSV, base_out_dir)
val_df   = load_split_csv(VAL_CSV, base_out_dir)
print("train_df shape:", train_df.shape, "val_df shape:", val_df.shape)


# ============================================================
# 4. Dataset / Collate (clean-weak pair pipeline ??)
# ============================================================
class CleanWeakPairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, prompt_by_dataset: Dict[str, str], system_prompt: Optional[str] = None):
        self.df = df.copy().reset_index(drop=True)
        self.prompt_by_dataset = prompt_by_dataset
        self.system_prompt = system_prompt

        sev = self.df["severity"]
        self.df["severity_norm"] = sev.fillna("clean").astype(str).str.lower()

        self.df["dataset_norm"]  = self.df["dataset"].astype(str).str.lower()

        self.pairs = []
        for fid, g in self.df.groupby("fileindex"):
            g = g.reset_index(drop=True)
            clean_rows = g[g["severity_norm"] == "clean"]
            weak_rows  = g[g["severity_norm"] == "weak"]
            if len(clean_rows) == 0 or len(weak_rows) == 0:
                continue
            clean_row = clean_rows.iloc[0]
            for _, weak_row in weak_rows.iterrows():
                self.pairs.append((clean_row, weak_row))

        print(f"Dataset - found {len(self.pairs)} clean/weak pairs.")

    def _make_sample(self, row):
        img_path = row["filepath"]
        img = Image.open(img_path).convert("RGB")
        arr = np.asarray(img).astype(np.float32)
        if not np.isfinite(arr).all():
            print("BAD IMAGE ARRAY:", img_path)

        modality = row["dataset_norm"]
        label = int(row["binarylabel"])

        task_prompt = self.prompt_by_dataset.get(
            modality,
            "This is a medical image.\nQuestion: Does this image show normal anatomy or signs of disease?\n\n",
        )

        full_text = (self.system_prompt + "\n\n" + task_prompt) if self.system_prompt else task_prompt
        return {"image": img, "input_text": full_text, "label": label, "path": img_path}

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        clean_row, weak_row = self.pairs[idx]
        return {"clean": self._make_sample(clean_row), "weak": self._make_sample(weak_row)}


def make_clean_weak_collate_fn(processor, backend: str):
    def collate(batch):
        def build_inputs(which):
            images, texts, labels_int, paths = [], [], [], []
            for item in batch:
                s = item[which]
                images.append(s["image"])
                texts.append(s["input_text"])
                labels_int.append(s["label"])
                paths.append(s.get("path", "<unknown>"))

            if backend in ["lingshu", "qwen3"]:
                messages_list = []
                for img, txt in zip(images, texts):
                    messages_list.append([{
                        "role": "user",
                        "content": [{"type": "image", "image": img}, {"type": "text", "text": txt}],
                    }])
                chat_texts = [
                    processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                    for m in messages_list
                ]
                model_inputs = processor(text=chat_texts, images=images, padding=True, return_tensors="pt")

            elif backend == "internvl":
                image_tok = getattr(processor, "image_token", None) or "<image>"
                inline_texts = [f"{image_tok}\n{txt}" for txt in texts]
                model_inputs = processor(text=inline_texts, images=images, padding=True, return_tensors="pt")

            elif backend == "medgemma":
                if not hasattr(processor, "apply_chat_template"):
                    raise RuntimeError("medgemma needs processor.apply_chat_template for image placeholder.")
                messages_list = []
                for img, txt in zip(images, texts):
                    messages_list.append([{
                        "role": "user",
                        "content": [{"type": "image"}, {"type": "text", "text": txt}],
                    }])
                chat_texts = [
                    processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                    for m in messages_list
                ]
                model_inputs = processor(text=chat_texts, images=images, padding=True, return_tensors="pt")

            else:
                model_inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")

            out = dict(model_inputs)
            out["labels_cls"] = torch.tensor(labels_int, dtype=torch.long)
            out["paths"] = paths
            return out

        return {"clean": build_inputs("clean"), "weak": build_inputs("weak")}

    return collate


# ============================================================
# 5. helper + epoch
# ============================================================

def to_device(x, target_device: torch.device):
    if torch.is_tensor(x):
        return x.to(target_device, non_blocking=True)

    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k in ("paths", "path", "meta"):
                out[k] = v
            else:
                out[k] = to_device(v, target_device)
        return out

    if isinstance(x, (list, tuple)):
        if len(x) > 0 and all(isinstance(v, str) for v in x):
            return x
        return type(x)(to_device(v, target_device) for v in x)

    return x


def get_vision_tensor(d: Dict[str, Any]):
    for k in ["pixel_values", "images", "image", "vision_x"]:
        v = d.get(k, None)
        if torch.is_tensor(v):
            return k, v
    return None, None


def finfo_str(x: torch.Tensor) -> str:
    x32 = x.detach().float().reshape(-1)

    nan = torch.isnan(x32)
    inf = torch.isinf(x32)
    finite = torch.isfinite(x32)

    n_nan = int(nan.sum().item())
    n_inf = int(inf.sum().item())

    if int(finite.sum().item()) > 0:
        xf = x32[finite]
        vmin = float(xf.min().item())
        vmax = float(xf.max().item())
        vmean = float(xf.mean().item())
    else:
        vmin, vmax, vmean = float("nan"), float("nan"), float("nan")

    return (
        f"shape={tuple(x.shape)} dtype={x.dtype} device={x.device} "
        f"min={vmin:.6g} max={vmax:.6g} mean={vmean:.6g} "
        f"nan={n_nan} inf={n_inf}"
    )

def compute_binary_metrics(y_true: torch.Tensor, y_score: torch.Tensor, thr: float = 0.5):
    """
    y_true: (N,) {0,1}
    y_score: (N,) probability for class 1 (disease)
    Returns: dict of python floats + ints
    """
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
    # sklearn AUROC requires both classes present
    y_true_np = y_true.numpy()
    y_score_np = y_score.numpy()

    if np.unique(y_true_np).size < 2:
        auroc = float("nan")
    else:
        auroc = float(roc_auc_score(y_true_np, y_score_np))

    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(auroc),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def run_epoch(vlm_adapt: VLMAdapterWrapper, loader, epoch: int, optimizer=None):
    train = optimizer is not None
    vlm_adapt.train() if train else vlm_adapt.eval()
    target_device = next(vlm_adapt.adapter.parameters()).device

    total_loss = total_ce = total_cons = 0.0
    correct = total = 0
    n_steps = 0
    all_y, all_score, all_sev = [], [], []

    def get_lambda_kl(epoch, max_epochs):
        # 0에서 시작해서 학습 절반 지점까지 1.0으로 선형 증가
        return min(1.0, epoch / (max_epochs / 2))


    def forward_base(inputs, return_attn = False):
        d = dict(inputs)
        d.pop("labels_cls", None)
        d.pop("labels_token", None)
        paths = d.pop("paths", None) # 시각화를 위해 path 보관

        k_vis, v = get_vision_tensor(d)
        if v is not None and (torch.isnan(v).any() or torch.isinf(v).any()):
            raise RuntimeError(f"NaN/Inf in vision input ({k_vis}) | {finfo_str(v)}")

        enable_base_grad = bool(train and vlm_adapt.train_projector and (vlm_adapt.projector_module is not None))
        grad_ctx = torch.enable_grad() if enable_base_grad else torch.no_grad()

        with grad_ctx:
            forward_kwargs = dict(d)

            if vlm_adapt.backend == "internvl" and device == "cuda":
                for k in ["pixel_values", "images", "image", "vision_x"]:
                    if k in forward_kwargs and torch.is_tensor(forward_kwargs[k]):
                        forward_kwargs[k] = forward_kwargs[k].to(dtype=torch.float32)

                with torch.amp.autocast(device_type="cuda", enabled=False):
                    try:
                        outputs = vlm_adapt.base_model(**forward_kwargs, output_hidden_states=True, return_dict=True)
                    except TypeError:
                        try:
                            outputs = vlm_adapt.base_model(**forward_kwargs, output_hidden_states=True)
                        except TypeError:
                            outputs = vlm_adapt.base_model(**forward_kwargs)
            else:
                try:
                    outputs = vlm_adapt.base_model(**forward_kwargs, output_hidden_states=True, return_dict=True)
                except TypeError:
                    try:
                        outputs = vlm_adapt.base_model(**forward_kwargs, output_hidden_states=True)
                    except TypeError:
                        outputs = vlm_adapt.base_model(**forward_kwargs)

        attn_mask = d.get("attention_mask", None)

        # [수정] extract_features에서 가중치 함께 추출
        if return_attn:
            h_base, weights = vlm_adapt.extract_features(outputs, attention_mask=attn_mask, return_weights=True)
            h_base = F.layer_norm(h_base.float(), (h_base.shape[-1],))
            return h_base, weights, paths
        else:
            h_base = vlm_adapt.extract_features(outputs, attention_mask=attn_mask).float()
            h_base = F.layer_norm(h_base, (h_base.shape[-1],))
            return h_base

    with torch.set_grad_enabled(train):
        for batch in loader:
            clean = to_device(batch["clean"], target_device)
            weak  = to_device(batch["weak"], target_device)

            y_c = clean["labels_cls"]
            y_w = weak["labels_cls"]


            # [수정] Clean과 Weak 모두 특징과 어텐션 가중치 추출
            h_clean, weights_c, paths_c = forward_base(clean, return_attn=True)
            h_weak_base, weights_w, paths_w = forward_base(weak, return_attn=True)

            if h_clean is None or h_weak_base is None:
                continue

            h_weak = vlm_adapt.adapter(h_weak_base)
            logits_c = vlm_adapt.classifier(h_clean)
            logits_w = vlm_adapt.classifier(h_weak)


            with torch.no_grad():
                preds_c = torch.argmax(logits_c, dim=1)
                preds_w = torch.argmax(logits_w, dim=1)
                correct += (preds_c == y_c).sum().item()
                correct += (preds_w == y_w).sum().item()
                total   += y_c.numel() + y_w.numel()

                prob_c = torch.softmax(logits_c.float(), dim=1)[:, 1]
                prob_w = torch.softmax(logits_w.float(), dim=1)[:, 1]
                all_y.append(y_c.detach().cpu())
                all_y.append(y_w.detach().cpu())
                all_score.append(prob_c.detach().cpu())
                all_score.append(prob_w.detach().cpu())
                all_sev.extend(["clean"] * y_c.numel())
                all_sev.extend(["weak"] * y_w.numel())

            # --- Loss 계산부 ---
            # 2. KL Divergence 계산 (Clean을 기준으로 Weak가 따라오게 함)
            # T(Temperature)는 분포를 부드럽게 만들어 학습을 돕습니다 (보통 1.0~2.0)
            T = 2.0
            # 학습 루프 내부
            current_lambda_kl = get_lambda_kl(epoch, EPOCHS)

            # 1. classifiaction loss
            L_c = F.cross_entropy(logits_c, y_c)
            L_w = F.cross_entropy(logits_w, y_w)
            L_ce = w_c * L_c + w_w * L_w

            # 2. feature consistency
            L_cons = F.mse_loss(h_weak, h_clean.detach())

            # 3. logit consistency (KL)
            # clean은 가이드 역할만 하고 업데이트는 weak 만
            p_clean = F.softmax(logits_c.detach() / T, dim=1) 
            log_p_weak = F.log_softmax(logits_w / T, dim=1)
            L_kl = F.kl_div(log_p_weak, p_clean, reduction='batchmean') * (T**2)

            # 4. Attention Consistency (Novelty 추가)
            # L_attn = F.mse_loss(weights_w, weights_c.detach())
            
            # total loss
            loss = L_ce + (lambda_w * L_cons) + (current_lambda_kl * L_kl) # + (0.1 * L_attn)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            # [추가] 시각화 (Validation 첫 배치에서 수행)
            # if not train and n_steps == 0:
            #     # 첫 번째 샘플(index 0)의 경로와 가중치 전달
            #     # 저장 파일명에 epoch을 넣어 구분하면 좋습니다.
            #     save_name = f"./results/viz_epoch_{epoch}_step_{n_steps}.png"
                
            #     # 가중치는 (B, T, 1) 형태이므로 첫 번째 샘플인 [0]을 슬라이싱
            #     visualize_vlm_attention(
            #         image_path=paths_c[0], 
            #         attn_weights=weights_c[0], 
            #         output_path=save_name
            #     )
            #     print(f"[*] Attention Heatmap saved: {save_name}")


            total_loss += float(loss.item())
            total_ce   += float(L_ce.item())
            total_cons += float(L_cons.item())
            n_steps += 1

    if n_steps == 0:
        return {
            "loss": float("nan"),
            "ce": float("nan"),
            "cons": float("nan"),
            "acc": float("nan"),
            "overall": {},
            "clean": {},
            "weak": {},
        }

    acc = correct / total if total > 0 else 0.0
    y_true = torch.cat(all_y, dim=0)
    y_score = torch.cat(all_score, dim=0)
    metrics_all = compute_binary_metrics(y_true, y_score)

    all_sev_np = np.array(all_sev)
    clean_mask = torch.tensor(all_sev_np == "clean", dtype=torch.bool)
    weak_mask  = torch.tensor(all_sev_np == "weak",  dtype=torch.bool)



    metrics_clean = (
        compute_binary_metrics(y_true[clean_mask], y_score[clean_mask])
        if clean_mask.sum() > 0 else {}
    )
    metrics_weak = (
        compute_binary_metrics(y_true[weak_mask], y_score[weak_mask])
        if weak_mask.sum() > 0 else {}
    )

    return {
        "loss": total_loss / n_steps,
        "ce": total_ce / n_steps,
        "cons": total_cons / n_steps,
        "acc": acc,
        "overall": metrics_all,
        "clean": metrics_clean,
        "weak": metrics_weak,
    }


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 6. BACKEND? ??
# ============================================================
SAVE_DIR = "./soft_prompt_ckpt"
os.makedirs(SAVE_DIR, exist_ok=True)

RESULTS_DIR = os.path.join(SAVE_DIR, "metrics_json")
os.makedirs(RESULTS_DIR, exist_ok=True)
for BACKEND in BACKENDS:
    print("\n==============================")
    print(f"== Training backend: {BACKEND}")
    print("==============================")

    model_id = MODEL_ID_BY_BACKEND[BACKEND]
    base_model, processor = load_backend(BACKEND, model_id)
    print("base_model device:", next(base_model.parameters()).device, flush=True)

    for layer_choice in POOL_LAYER_CHOICES:
        set_seed(SEED)
        print(f"\n=== BACKEND={BACKEND} | LAYER={layer_choice} ===")

        vlm_adapt = VLMAdapterWrapper(
            base_model,
            backend=BACKEND,
            layer_choice=layer_choice,
            train_projector=bool(args.train_projector),
            projector_path=args.projector_path,
        )

        collate_fn = make_clean_weak_collate_fn(processor, BACKEND)
        bs = BATCH_SIZE_BY_BACKEND.get(BACKEND, BATCH_SIZE_DEFAULT)

        if TEST_ONE_ROW:
            one = train_df.iloc[[0]].copy()
            test_ds = CleanWeakPairDataset(one, PROMPT_BY_DATASET, SYSTEM_PROMPT_SHORT)
            test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

            batch = next(iter(test_loader))

            target_device = next(vlm_adapt.adapter.parameters()).device
            clean = to_device(batch["clean"], target_device)
            weak  = to_device(batch["weak"], target_device)

            y_c = clean.pop("labels_cls")
            y_w = weak.pop("labels_cls")
            clean.pop("labels_token", None)
            weak.pop("labels_token", None)

            with torch.no_grad():
                def forward_once(inputs):
                    forward_kwargs = dict(inputs)

                    if BACKEND == "internvl" and device == "cuda":
                        for k in ["pixel_values", "images", "image", "vision_x"]:
                            if k in forward_kwargs and torch.is_tensor(forward_kwargs[k]):
                                forward_kwargs[k] = forward_kwargs[k].to(dtype=torch.float32)

                        with torch.amp.autocast(device_type="cuda", enabled=False):
                            try:
                                out = vlm_adapt.base_model(**forward_kwargs, output_hidden_states=True, return_dict=True)
                            except TypeError:
                                try:
                                    out = vlm_adapt.base_model(**forward_kwargs, output_hidden_states=True)
                                except TypeError:
                                    out = vlm_adapt.base_model(**forward_kwargs)
                    else:
                        try:
                            out = vlm_adapt.base_model(**forward_kwargs, output_hidden_states=True, return_dict=True)
                        except TypeError:
                            try:
                                out = vlm_adapt.base_model(**forward_kwargs, output_hidden_states=True)
                            except TypeError:
                                out = vlm_adapt.base_model(**forward_kwargs)
                    return out

                out_clean = forward_once(clean)
                out_weak = forward_once(weak)

                h_clean = vlm_adapt.extract_features(out_clean, attention_mask=clean.get("attention_mask", None)).float()
                h_weak_base = vlm_adapt.extract_features(out_weak, attention_mask=weak.get("attention_mask", None)).float()

                logits_c = vlm_adapt.classifier(h_clean)
                logits_w = vlm_adapt.classifier(vlm_adapt.adapter(h_weak_base))

            print(f"[{BACKEND}] OK | h_clean {tuple(h_clean.shape)} | h_weak {tuple(h_weak_base.shape)} | y_c {tuple(y_c.shape)} | y_w {tuple(y_w.shape)}")

            del vlm_adapt, test_ds, test_loader
            gc.collect()
            torch.cuda.empty_cache()
            continue

        # g = torch.Generator()
        # g.manual_seed(SEED)
        train_ds = CleanWeakPairDataset(train_df, PROMPT_BY_DATASET, SYSTEM_PROMPT_SHORT)
        val_ds   = CleanWeakPairDataset(val_df,   PROMPT_BY_DATASET, SYSTEM_PROMPT_SHORT)

        print(f"[{BACKEND}] Using batch_size = {bs}")
        print(f"[{BACKEND}] #train_pairs={len(train_ds)} #val_pairs={len(val_ds)}")

        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  collate_fn=collate_fn, pin_memory=(device=="cuda"))
        val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, collate_fn=collate_fn, pin_memory=(device=="cuda"))

        optim_param_groups = [
            {"params": list(vlm_adapt.pooler.parameters()), "lr": LR},
            {"params": list(vlm_adapt.adapter.parameters()), "lr": LR},
            {"params": list(vlm_adapt.classifier.parameters()), "lr": LR},
        ]
        if args.train_projector and vlm_adapt.projector_module is not None:
            proj_params = [p for p in vlm_adapt.projector_module.parameters() if p.requires_grad]
            if len(proj_params) > 0:
                optim_param_groups.append({"params": proj_params, "lr": PROJECTOR_LR})

        optimizer = AdamW(optim_param_groups, lr=LR)

        exp_tag = "weak_att_proj" if args.train_projector else "weak_att"
        BEST_CKPT = os.path.join(SAVE_DIR, f"{BACKEND}_{layer_choice}_{exp_tag}_best.pt")
        LAST_CKPT = os.path.join(SAVE_DIR, f"{BACKEND}_{layer_choice}_{exp_tag}_last.pt")

        metrics_path = os.path.join(RESULTS_DIR, f"{RUN_ID}_{BACKEND}_{exp_tag}_metrics.json")

        start_epoch = 0
        best_val_score = -1.0
        epoch_logs = []

        if os.path.exists(LAST_CKPT):
            ckpt = torch.load(LAST_CKPT, map_location="cpu")

            vlm_adapt.adapter.load_state_dict(ckpt["adapter"], strict=True)
            vlm_adapt.classifier.load_state_dict(ckpt["classifier"], strict=True)
            vlm_adapt.pooler.load_state_dict(ckpt["pooler"], strict=True)
            projector_state = ckpt.get("projector", None)
            if args.train_projector and vlm_adapt.projector_module is not None and isinstance(projector_state, dict):
                vlm_adapt.projector_module.load_state_dict(projector_state, strict=True)

            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])

            best_val_score = float(ckpt.get("best_val_score", -1.0))
            start_epoch = int(ckpt.get("epoch", 0))
            epoch_logs = ckpt.get("epoch_logs", [])
            RUN_ID = ckpt.get("run_id", RUN_ID)

            metrics_path = os.path.join(RESULTS_DIR, f"{RUN_ID}_{BACKEND}_{exp_tag}_metrics.json")

            print(f"[resume] Loaded {LAST_CKPT} | start_epoch={start_epoch} | best_val_score={best_val_score:.4f}")
        else:
            print("[resume] No last checkpoint found. Start fresh.")

        for epoch in range(start_epoch, EPOCHS):
            tr_m = run_epoch(vlm_adapt, train_loader, epoch=epoch, optimizer=optimizer)
            val_m = run_epoch(vlm_adapt, val_loader, epoch=epoch, optimizer=None)

            tr_acc = tr_m.get("acc", float("nan"))
            va_acc = val_m.get("acc", float("nan"))
            tr_auroc = tr_m.get("overall", {}).get("auroc", float("nan"))
            va_auroc = val_m.get("overall", {}).get("auroc", float("nan"))
            va_clean = val_m.get("clean", {}).get("auroc", float("nan"))
            va_weak  = val_m.get("weak", {}).get("auroc", float("nan"))

            print(
                f"[{BACKEND}] Epoch {epoch+1}/{EPOCHS}\n"
                f"Train: loss={tr_m['loss']:.4f} CE={tr_m['ce']:.4f} Cons={tr_m['cons']:.4f} "
                f"Acc={tr_acc*100:.2f}% AUROC={tr_auroc:.3f}\n"
                f"Val  : loss={val_m['loss']:.4f} CE={val_m['ce']:.4f} Cons={val_m['cons']:.4f} "
                f"Acc={va_acc*100:.2f}% AUROC={va_auroc:.3f} | Clean={va_clean:.3f} | Weak={va_weak:.3f}"
            )

            epoch_logs.append({
                "epoch": epoch + 1,
                "train": tr_m,
                "val": val_m,
            })

            score = va_weak

            if not np.isnan(score) and score > best_val_score:
                best_val_score = score
                torch.save(
                    {
                        "adapter": vlm_adapt.adapter.state_dict(),
                        "classifier": vlm_adapt.classifier.state_dict(),
                        "pooler": vlm_adapt.pooler.state_dict(),
                        "hidden_dim": vlm_adapt.hidden_dim,
                        "backend": BACKEND,
                        "model_id": model_id,
                        "train_projector": bool(args.train_projector),
                        "projector_path": vlm_adapt.projector_path,
                        "best_val_score": best_val_score,
                        "best_val_auroc_overall": va_auroc,
                        "epoch": epoch + 1,
                        "projector": (
                            vlm_adapt.projector_module.state_dict()
                            if (args.train_projector and vlm_adapt.projector_module is not None)
                            else None
                        ),
                    },
                    BEST_CKPT,
                )
                print(f"BEST saved: {BEST_CKPT} (best_val_auroc_weak={best_val_score:.4f})")


            payload = {
                "backend": BACKEND,
                "layer_choice": layer_choice,
                "model_id": model_id,
                "seed": SEED,
                "epochs": EPOCHS,
                "train_projector": bool(args.train_projector),
                "projector_path": vlm_adapt.projector_path,
                "projector_lr": float(PROJECTOR_LR),
                "logs": epoch_logs,
                "best_val_auroc": best_val_score,
                "best_val_auroc_overall": va_auroc,
            }
            tmp_path = metrics_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, metrics_path)

            torch.save(
                {
                    "run_id": RUN_ID,
                    "epoch": epoch + 1,
                    "adapter": vlm_adapt.adapter.state_dict(),
                    "classifier": vlm_adapt.classifier.state_dict(),
                    "pooler": vlm_adapt.pooler.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_score": best_val_score,
                    "backend": BACKEND,
                    "model_id": model_id,
                    "layer_choice": layer_choice,
                    "train_projector": bool(args.train_projector),
                    "projector_path": vlm_adapt.projector_path,
                    "projector": (
                        vlm_adapt.projector_module.state_dict()
                        if (args.train_projector and vlm_adapt.projector_module is not None)
                        else None
                    ),
                    "epoch_logs": epoch_logs,
                },
                LAST_CKPT,
            )
            print(f"[metrics] wrote {metrics_path} | epochs_logged={len(epoch_logs)}")

        del vlm_adapt, optimizer, train_ds, val_ds, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

    del base_model, processor
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
