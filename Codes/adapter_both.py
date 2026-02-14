#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
#os.environ.setdefault("HF_HUB_DISABLE_MMAP", "1")  # safetensors mmap 끄기 (ENOMEM 해결)
assert os.environ.get("HF_HOME"), "HF_HOME not set (use TMPDIR/hf_cache etc.)"
hf_home = os.environ["HF_HOME"]
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(hf_home, "hub"))


import random
import gc
import json
from typing import Optional, Any, Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import json
from datetime import datetime

import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", type=str, required=True,
                   choices=["qwen3", "medgemma", "internvl", "lingshu"])
    p.add_argument("--layer", type=str, default="last", choices=["last", "hs_-4"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--bs", type=int, default=1)
    return p.parse_args()

args = parse_args()



RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================
# 0. 공통 설정
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

TEST_ONE_ROW = False   # True면 1샘플 forward만
csv_path = "/SAN/ioo/HORIZON/howoon"

TRAIN_CSV = os.path.join(csv_path, "vlm_clean_weak_train_2520.csv")
VAL_CSV   = os.path.join(csv_path, "vlm_clean_weak_val_540.csv")

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

BACKENDS = ["qwen3", "medgemma", "internvl", "lingshu"]

LR = args.lr
EPOCHS = args.epochs
POOL_LAYER_CHOICES = [args.layer]   # ["last"] 대신

BATCH_SIZE_DEFAULT = 1
BATCH_SIZE_BY_BACKEND = {
    "internvl": 1,
    "lingshu":  1,
    "qwen3":    1,
    "medgemma": 1,
}



SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if device == "cuda":
    torch.cuda.manual_seed_all(SEED)

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


class VLMAdapterWrapper(nn.Module):
    """
    base_model: frozen
    adapter + classifier: trainable (fp32)
    """
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

        # ✅ 학습 파트는 fp32 고정 (NaN 방지)
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
    def extract_features(
        self,
        outputs,
        attention_mask: Optional[torch.Tensor] = None,
        debug: bool = False,
    ) -> torch.Tensor:
        H = self.hidden_dim

        def get(obj, name: str):
            if hasattr(obj, name) and getattr(obj, name) is not None:
                return getattr(obj, name)
            if isinstance(obj, dict) and name in obj and obj[name] is not None:
                return obj[name]
            return None

        # 1) pick sequence hidden states: prefer middle layer (-4), else last_hidden_state
        hs = get(outputs, "hidden_states")
        t  = get(outputs, "last_hidden_state")

        if self.layer_choice == "hs_-4":
            if not (isinstance(hs, (list, tuple)) and len(hs) >= 5):
                raise RuntimeError(f"{self.backend}: hidden_states[-4] 불가 (len={0 if hs is None else len(hs)})")
            if not (torch.is_tensor(hs[-4]) and hs[-4].dim() == 3 and hs[-4].shape[-1] == H):
                raise RuntimeError(f"{self.backend}: hidden_states[-4] shape 이상 -> {None if not torch.is_tensor(hs[-4]) else tuple(hs[-4].shape)}")
            x = hs[-4]
        else:
            if torch.is_tensor(t) and t.dim() == 3 and t.shape[-1] == H:
                x = t
            elif isinstance(hs, (list, tuple)) and len(hs) > 0 and torch.is_tensor(hs[-1]) and hs[-1].dim() == 3 and hs[-1].shape[-1] == H:
                x = hs[-1]
            else:
                raise ValueError("No usable hidden representation found.")

        # 2) masked mean pooling if attention_mask exists, else fallback mean pooling
        if attention_mask is None:
            return x.mean(dim=1)  # (B,H)

        if attention_mask.dim() != 2:
            # 이상한 마스크면 그냥 mean pooling으로 폴백
            return x.mean(dim=1)

        # length mismatch 안전 처리
        T = min(x.shape[1], attention_mask.shape[1])
        x = x[:, :T, :]
        m = attention_mask[:, :T].to(dtype=x.dtype, device=x.device).unsqueeze(-1)  # (B,T,1)

        x_sum = (x * m).sum(dim=1)                 # (B,H)
        denom = m.sum(dim=1).clamp_min(1.0)        # (B,1)
        return x_sum / denom                       # (B,H)


# ============================================================
# 2. Backend 로더
# ============================================================
def debug_batch(batch, backend):
    print("\n================ DEBUG BATCH ================")
    print("backend:", backend)

    for k, v in batch.items():
        if torch.is_tensor(v):
            print(f"{k:20s} -> shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")
        elif isinstance(v, list):
            print(f"{k:20s} -> list(len={len(v)}) type={type(v[0])}")
        else:
            print(f"{k:20s} -> {type(v)}")

    print("=============================================\n")

def load_backend(backend: str, model_id: str):
    import transformers
    from transformers import AutoProcessor

    # 기본 dtype
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # ✅ A100이면 internvl도 bf16이 안정적
    if backend in ["medgemma", "internvl"] and device == "cuda":
        torch_dtype = torch.bfloat16

    common = dict(
        torch_dtype=torch_dtype,
        device_map=None,            # ✅ auto 금지: 멀티GPU/워밍업 이슈 원인
        low_cpu_mem_usage=True,
    )
    cache_dir = os.environ.get("TRANSFORMERS_CACHE") or os.environ.get("HF_HOME") or None


    if backend == "qwen3":
        from transformers import Qwen3VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id,cache_dir=cache_dir, **common)
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

    # ✅ internvl은 trust_remote_code 필수
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
            model.to(device)
            model.eval()
            print('[medgemma] Model: AutoImageTextToText')
            return model, processor

        AutoVision2Seq = getattr(transformers, "AutoModelForVision2Seq", None)
        if AutoVision2Seq is not None:
            model = AutoVision2Seq.from_pretrained(model_id, cache_dir=cache_dir, **common)
            model.to(device)
            model.eval()
            print('[medgemma] Model: AutoVision2Seq')
            return model, processor

        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, **common)
        model.to(device)
        model.eval()
        print('[medgemma] Model: AutoModelForCausalLM')
        return model, processor


    raise ValueError(f"Unknown backend: {backend}")


# ============================================================
# 3. CSV 로드
# ============================================================
base_out_dir = "/SAN/ioo/HORIZON/howoon"

def load_split_csv(path: str, base_out_dir: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "binarylab" in df.columns and "binarylabel" not in df.columns:
        df = df.rename(columns={"binarylab": "binarylabel"})

    df["severity_norm"] = df["severity"].astype(str).str.lower()
    df["dataset_norm"]  = df["dataset"].astype(str).str.lower()

    #df = df[df["severity_norm"].isin(["clean"])].copy()

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


class CleanOnlyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, prompt_by_dataset: Dict[str, str], system_prompt: Optional[str] = None):
        self.df = df.copy().reset_index(drop=True)
        self.prompt_by_dataset = prompt_by_dataset
        self.system_prompt = system_prompt
        self.df["dataset_norm"] = self.df["dataset"].astype(str).str.lower()
        print(f"Dataset - found {len(self.df)} samples.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
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

        if self.system_prompt is not None:
            full_text = self.system_prompt + "\n\n" + task_prompt
        else:
            full_text = task_prompt
        is_clean = pd.isna(row["severity"])
        sev = "clean" if is_clean else str(row["severity"]).lower()
        return {"image": img, "input_text": full_text, "label": label, "path": img_path, "severity": sev}

def make_clean_only_collate_fn(processor, backend: str):
    def collate(batch):
        images = [b["image"] for b in batch]
        texts  = [b["input_text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        paths  = [b.get("path", "<unknown>") for b in batch]
        sevs   = [b.get("severity", "unknown") for b in batch]

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

            # ✅ 중요: Gemma3는 샘플별 이미지 list로 감싸야 batch 정합이 맞음
            images_nested = [[img] for img in images]

            model_inputs = processor(text=chat_texts, images=images_nested, padding=True, return_tensors="pt")


        else:
            model_inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")

        out = dict(model_inputs)
        out["labels_cls"] = labels
        out["paths"] = paths
        out["severities"] = sevs
        # collate 마지막에 임시로
        if "attention_mask" not in out:
            print("[warn] attention_mask missing. keys=", list(out.keys()))

        return out

    return collate


# ============================================================
# 4. helper + epoch
# ============================================================



def to_device(x, target_device: torch.device):
    """
    Move ONLY tensors to target_device.
    Keep python objects (e.g., paths: list[str]) untouched.
    """
    if torch.is_tensor(x):
        return x.to(target_device, non_blocking=True)

    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            # never touch metadata fields
            if k in ("paths", "path", "meta", "severities"):
                out[k] = v
            else:
                out[k] = to_device(v, target_device)
        return out

    if isinstance(x, (list, tuple)):
        # if it looks like list[str], keep as-is
        if len(x) > 0 and all(isinstance(v, str) for v in x):
            return x
        return type(x)(to_device(v, target_device) for v in x)

    return x



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
    pos = (y_true == 1)
    neg = (y_true == 0)
    if int(pos.sum()) == 0 or int(neg.sum()) == 0:
        auroc = float("nan")
        auprc = float("nan")
    else:
        order = torch.argsort(y_score)  # ascending
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(1, len(y_score) + 1, dtype=torch.float32)
        sum_ranks_pos = ranks[pos].sum()
        n_pos = float(pos.sum().item())
        n_neg = float(neg.sum().item())
        u = sum_ranks_pos - n_pos * (n_pos + 1.0) / 2.0
        auroc = float((u / (n_pos * n_neg)).item())

        # AUPRC (Average Precision approximation)
        desc = torch.argsort(y_score, descending=True)
        y_sorted = y_true[desc]
        cum_pos = torch.cumsum((y_sorted == 1).to(torch.float32), dim=0)
        k = torch.arange(1, len(y_sorted) + 1, dtype=torch.float32)
        precision_at_k = cum_pos / k
        pos_mask = (y_sorted == 1)
        auprc = float(precision_at_k[pos_mask].mean().item()) if int(pos_mask.sum()) > 0 else float("nan")

    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }

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
def run_epoch_clean_only(vlm_adapt, loader, optimizer=None):
    """
    - base_model은 항상 frozen + no_grad 로만 forward (메모리/속도)
    - adapter/classifier는 train일 때 grad ON (진짜 학습됨)
    - base feature NaN/Inf 나오면: 바로 에러 (네가 원한 동작)
    """
    train = optimizer is not None
    vlm_adapt.train() if train else vlm_adapt.eval()

    target_device = next(vlm_adapt.adapter.parameters()).device

    total_loss, n_steps = 0.0, 0
    all_y, all_score, all_sev = [], [], []

    for batch in loader:
        batch = to_device(batch, target_device)
        #debug_batch(batch, BACKEND)
        #print("---- MODEL OUTPUT ----")


        paths = batch.get("paths", ["<unknown>"])
        severities = batch.get("severities", ["unknown"])
        y = batch["labels_cls"]

        # model input dict 만들기 (메타 제거)
        d = dict(batch)
        d.pop("labels_cls", None)
        d.pop("labels_token", None)
        d.pop("paths", None)
        d.pop("severities", None)

        # vision input NaN/Inf 체크 (있으면 즉시 종료)
        k_vis, v = get_vision_tensor(d)
        if v is not None and (torch.isnan(v).any() or torch.isinf(v).any()):
            raise RuntimeError(
                f"NaN/Inf in vision input ({k_vis}) | path={paths[0]} | {finfo_str(v)}"
            )

        # -----------------------------
        # 1) base_model forward: no_grad
        # -----------------------------
        forward_kwargs = dict(d)

        # internvl만 vision tensor fp32 강제 + autocast off (안정성)
        if vlm_adapt.backend == "internvl" and device == "cuda":
            for k in ["pixel_values", "images", "image", "vision_x"]:
                if k in forward_kwargs and torch.is_tensor(forward_kwargs[k]):
                    forward_kwargs[k] = forward_kwargs[k].to(dtype=torch.float32)
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    outputs = vlm_adapt.base_model(
                        **forward_kwargs,
                        output_hidden_states=True,
                        return_dict=True,
                    )
        else:
            with torch.no_grad():
                outputs = vlm_adapt.base_model(
                    **forward_kwargs,
                    output_hidden_states=True,
                    return_dict=True,
                )

        # -----------------------------
        # 2) feature pooling (no_grad로 만들어진 h_base를
        #    adapter/classifier에 넣을 때는 grad가 살아있어야 함)
        #    => h_base는 "상수 입력"이지만 adapter/classifier 파라미터는 학습됨
        # -----------------------------
        attn = d.get("attention_mask", None)
        h_base = vlm_adapt.extract_features(outputs, attention_mask=attn).float()
        h_base = F.layer_norm(h_base, (h_base.shape[-1],))

        if not torch.isfinite(h_base).all():
            raise RuntimeError(
                f"NaN/Inf in base features | backend={vlm_adapt.backend} | path={paths[0]} | {finfo_str(h_base)}"
            )

        # -----------------------------
        # 3) trainable head forward/backward
        # -----------------------------
        h = vlm_adapt.adapter(h_base)          # grad ON (train일 때)
        logits = vlm_adapt.classifier(h)
        loss = F.cross_entropy(logits, y)


        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            prob = torch.softmax(logits.float(), dim=1)[:, 1]
            all_y.append(y.detach().cpu())
            all_score.append(prob.detach().cpu())
            all_sev.extend(severities)

        total_loss += float(loss.item())
        n_steps += 1

    if n_steps == 0:
        raise RuntimeError("No steps were run (empty loader?)")

    y_true = torch.cat(all_y, dim=0)
    y_score = torch.cat(all_score, dim=0)

    metrics_all = compute_binary_metrics(y_true, y_score)

    all_sev = np.array(all_sev)
    clean_mask = all_sev == "clean"
    weak_mask = all_sev == "weak"

    metrics_clean = (
        compute_binary_metrics(y_true[clean_mask], y_score[clean_mask])
        if clean_mask.sum() > 0 else {}
    )
    metrics_weak = (
        compute_binary_metrics(y_true[weak_mask], y_score[weak_mask])
        if weak_mask.sum() > 0 else {}
    )

    return total_loss / n_steps, {
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
# 5. BACKEND별 실행
# ============================================================
SAVE_DIR = "./soft_prompt_ckpt"
os.makedirs(SAVE_DIR, exist_ok=True)

RESULTS_DIR = os.path.join(SAVE_DIR, "metrics_json")
os.makedirs(RESULTS_DIR, exist_ok=True)

BACKEND = args.backend

print("\n==============================")
print(f"== Training backend: {BACKEND}")
print("==============================")

model_id = MODEL_ID_BY_BACKEND[BACKEND]
base_model, processor = load_backend(BACKEND, model_id)
print("base_model device:", next(base_model.parameters()).device, flush=True)


for layer_choice in POOL_LAYER_CHOICES:
    set_seed(SEED)
    print(f"\n=== BACKEND={BACKEND} | LAYER={layer_choice} ===")

    # ??device_map ????????wrapper??.to(device) ??? ????
    vlm_adapt = VLMAdapterWrapper(base_model, backend=BACKEND, layer_choice=layer_choice)

    collate_fn = make_clean_only_collate_fn(processor, BACKEND)
    bs = BATCH_SIZE_BY_BACKEND.get(BACKEND, BATCH_SIZE_DEFAULT)

    if TEST_ONE_ROW:
        one = train_df.sample(3)
        test_ds = CleanOnlyDataset(one, PROMPT_BY_DATASET, SYSTEM_PROMPT_SHORT)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

        target_device = next(vlm_adapt.adapter.parameters()).device

        for batch in test_loader:
            batch = to_device(batch, target_device)

            d = dict(batch)
            y = d.pop("labels_cls")
            paths = d.pop("paths", ["<unknown>"])
            d.pop("labels_token", None)

            with torch.no_grad():
                forward_kwargs = dict(d)

                # (너 기존 로직 유지)
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

            # ===== forward 직후 공통 디버그 출력 =====
            am = d.get("attention_mask", None)
            print(f"[{BACKEND}] attention_mask:", None if am is None else tuple(am.shape))

            lhs = getattr(out, "last_hidden_state", None)
            print(f"[{BACKEND}] out.last_hidden_state:", None if lhs is None else tuple(lhs.shape))

            hs = getattr(out, "hidden_states", None)
            if hs is None:
                print(f"[{BACKEND}] out.hidden_states: None")
            else:
                try:
                    print(f"[{BACKEND}] out.hidden_states[-1]:", tuple(hs[-1].shape))
                except Exception as e:
                    print(f"[{BACKEND}] out.hidden_states[-1]: ERROR {e}")
            # ======================================


            attn = d.get("attention_mask", None)
            hs = out.hidden_states if hasattr(out, "hidden_states") else None
            t  = out.last_hidden_state if hasattr(out, "last_hidden_state") else None

            h0 = vlm_adapt.extract_features(out, attention_mask=attn, debug=True).float()
            if torch.isnan(h0).any() or torch.isinf(h0).any():
                print("NaN/Inf in TEST_ONE_ROW | path:", paths[0])
                raise RuntimeError("NaN/Inf in base features (TEST_ONE_ROW)")

            logits = vlm_adapt.classifier(vlm_adapt.adapter(h0))

            print("----------------------\n")

            print(f"[{BACKEND}] OK | path={paths[0]} | h0 {tuple(h0.shape)} | logits {tuple(logits.shape)} | y {tuple(y.shape)}")

        del vlm_adapt, test_ds, test_loader
        gc.collect()
        torch.cuda.empty_cache()
        continue


    # g = torch.Generator()
    # g.manual_seed(SEED)
    train_ds = CleanOnlyDataset(train_df, PROMPT_BY_DATASET, SYSTEM_PROMPT_SHORT)
    val_ds   = CleanOnlyDataset(val_df,   PROMPT_BY_DATASET, SYSTEM_PROMPT_SHORT)

    print(f"[{BACKEND}] Using batch_size = {bs}")
    print(f"[{BACKEND}] #train={len(train_ds)} #val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  collate_fn=collate_fn, pin_memory=(device=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, collate_fn=collate_fn, pin_memory=(device=="cuda"))

    optimizer = AdamW(
        list(vlm_adapt.adapter.parameters()) + list(vlm_adapt.classifier.parameters()),
        lr=LR
    )

    BEST_CKPT = os.path.join(SAVE_DIR, f"{BACKEND}_{layer_choice}_best.pt")
    LAST_CKPT = os.path.join(SAVE_DIR, f"{BACKEND}_{layer_choice}_last.pt")

    # metrics 파일은 run_id 기준으로 저장 (resume 시 run_id를 last에서 복구)
    metrics_path = os.path.join(RESULTS_DIR, f"{RUN_ID}_{BACKEND}_{layer_choice}_metrics.json")

    # ---- resume ----
    start_epoch = 0
    best_val_score = -1.0
    epoch_logs = []

    if os.path.exists(LAST_CKPT):
        ckpt = torch.load(LAST_CKPT, map_location="cpu")

        vlm_adapt.adapter.load_state_dict(ckpt["adapter"], strict=True)
        vlm_adapt.classifier.load_state_dict(ckpt["classifier"], strict=True)

        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])

        best_val_score = float(ckpt.get("best_val_score", -1.0))
        start_epoch = int(ckpt.get("epoch", 0))  # 다음 epoch 시작점 (이미 epoch+1로 저장했으니 그대로 쓰면 됨)
        epoch_logs = ckpt.get("epoch_logs", [])
        RUN_ID = ckpt.get("run_id", RUN_ID)

        metrics_path = os.path.join(RESULTS_DIR, f"{RUN_ID}_{BACKEND}_{layer_choice}_metrics.json")

        print(f"[resume] Loaded {LAST_CKPT} | start_epoch={start_epoch} | best_val_score={best_val_score:.4f}")
    else:
        print("[resume] No last checkpoint found. Start fresh.")



    for epoch in range(start_epoch, EPOCHS):

        tr_loss, tr_m  = run_epoch_clean_only(vlm_adapt, train_loader, optimizer)
        val_loss, val_m = run_epoch_clean_only(vlm_adapt, val_loader, optimizer=None)

        tr_auroc = tr_m.get("overall", {}).get("auroc", float("nan"))
        va_auroc = val_m.get("overall", {}).get("auroc", float("nan"))
        va_clean = val_m.get("clean", {}).get("auroc", float("nan"))
        va_weak  = val_m.get("weak", {}).get("auroc", float("nan"))

        print(
            f"[{BACKEND}] Epoch {epoch+1}/{EPOCHS}\n"
            f"Train AUROC: {tr_auroc:.3f}\n"
            f"Val   AUROC: {va_auroc:.3f} | Clean: {va_clean:.3f} | Weak: {va_weak:.3f}"
        )

        epoch_logs.append({
            "epoch": epoch + 1,
            "train_loss": float(tr_loss),
            "val_loss": float(val_loss),
            "train": tr_m,
            "val": val_m,
        })

        score = val_m["overall"].get("auroc", float("nan"))

        if not np.isnan(score) and score > best_val_score:
            best_val_score = score
            torch.save(
                {
                    "adapter": vlm_adapt.adapter.state_dict(),
                    "classifier": vlm_adapt.classifier.state_dict(),
                    "hidden_dim": vlm_adapt.hidden_dim,
                    "backend": BACKEND,
                    "model_id": model_id,
                    "best_val_auroc_overall": best_val_score,
                    "epoch": epoch + 1,
                },
                BEST_CKPT,
            )
            print(f"BEST saved: {BEST_CKPT} (best_val_overall_auroc={best_val_score:.4f})")

        payload = {
            "backend": BACKEND,
            "layer_choice": layer_choice,
            "model_id": model_id,
            "seed": SEED,
            "epochs": EPOCHS,
            "logs": epoch_logs,
            "best_val_auroc_overall": best_val_score,
        }
        tmp_path = metrics_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, metrics_path)  # atomic write


        # 항상 last 저장 (resume용)
        torch.save(
            {
                "run_id": RUN_ID,
                "epoch": epoch + 1,
                "adapter": vlm_adapt.adapter.state_dict(),
                "classifier": vlm_adapt.classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_score": best_val_score,
                "backend": BACKEND,
                "model_id": model_id,
                "layer_choice": layer_choice,
                "epoch_logs": epoch_logs,
            },
            LAST_CKPT,
        )
        print(f"[metrics] wrote {metrics_path} | epochs_logged={len(epoch_logs)}")




    # if not TEST_ONE_ROW:
    #     metrics_path = os.path.join(RESULTS_DIR, f"{RUN_ID}_{BACKEND}_{layer_choice}_metrics.json")
    #     payload = {
    #         "backend": BACKEND,
    #         "layer_choice": layer_choice,
    #         "model_id": model_id,
    #         "seed": SEED,
    #         "epochs": EPOCHS,
    #         "logs": epoch_logs,
    #         "best_val_auroc_overall": best_val_score,
    #     }
    #     with open(metrics_path, "w", encoding="utf-8") as f:
    #         json.dump(payload, f, ensure_ascii=False, indent=2)

    del vlm_adapt, optimizer, train_ds, val_ds, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()
del base_model, processor
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()   # ⭐ 이거 추가
