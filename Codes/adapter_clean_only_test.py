#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import gc
from typing import Optional, Any, Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# ============================================================
# 0. 공통 설정
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

TEST_ONE_ROW = True   # True면 1샘플 forward만
csv_path = "/SAN/ioo/HORIZON/howoon"

TRAIN_CSV = os.path.join(csv_path, "vlm_clean_train_2520.csv")
VAL_CSV   = os.path.join(csv_path, "vlm_clean_val_540.csv")

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

LR = 5e-5
EPOCHS = 15
BATCH_SIZE_DEFAULT = 1
BATCH_SIZE_BY_BACKEND = {
    "internvl": 1,
    "lingshu":  1,
    "qwen3":    1,
    "medgemma": 1,
}

# ✅ NaN/Inf 샘플을 건너뛸지 (급한 디버그용)
SKIP_BAD_SAMPLES = False

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
    def __init__(self, base_model, backend: str):
        super().__init__()
        self.base_model = base_model
        self.backend = backend

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

    def extract_features(self, outputs, attention_mask: Optional[torch.Tensor] = None, debug: bool = False) -> torch.Tensor:
        """
        Unified fused feature extractor (TEXT+IMAGE fused pooling).

        Goal:
        - Always return (B, H) by pooling a (B, T, H) fused hidden-state tensor.
        - If attention_mask is shorter than sequence length T (common in VLMs where mask covers only text tokens),
        we EXTEND the mask with ones so that the extra tokens (likely image/vision tokens) are INCLUDED in pooling.
        """
        H = self.hidden_dim

        def is_seq_tensor(t: Any) -> bool:
            return torch.is_tensor(t) and t.dim() == 3 and t.shape[-1] == H

        def get(obj, name: str):
            if hasattr(obj, name) and getattr(obj, name) is not None:
                return getattr(obj, name)
            if isinstance(obj, dict) and name in obj and obj[name] is not None:
                return obj[name]
            return None

        def pool_fused(seq: torch.Tensor) -> torch.Tensor:
            # seq: (B, T, H)
            B, T, _ = seq.shape

            # If no mask, simple mean over T (includes text+image tokens, includes padding though)
            if attention_mask is None or (not torch.is_tensor(attention_mask)) or attention_mask.dim() != 2:
                return seq.mean(dim=1)

            attn = attention_mask.to(dtype=seq.dtype, device=seq.device)
            t_mask = attn.shape[1]

            if debug:
                print(f"[debug] backend={self.backend} seq_T={T} mask_T={t_mask}")



            if t_mask < T:
                # Extend with ones so extra tokens (likely image tokens) are INCLUDED
                pad = torch.ones((B, T - t_mask), dtype=seq.dtype, device=seq.device)
                attn = torch.cat([attn, pad], dim=1)
            elif t_mask > T:
                # Truncate if somehow longer
                attn = attn[:, :T]

            m = attn.unsqueeze(-1)  # (B, T, 1)
            denom = m.sum(dim=1).clamp_min(1.0)
            return (seq * m).sum(dim=1) / denom

        # 1) Standard: last_hidden_state (B,T,H)
        t = get(outputs, "last_hidden_state")
        if is_seq_tensor(t):
            return pool_fused(t)

        # 2) hidden_states[-1] (B,T,H)
        hs = get(outputs, "hidden_states")
        if isinstance(hs, (list, tuple)) and len(hs) > 0 and is_seq_tensor(hs[-1]):
            return pool_fused(hs[-1])

        # 3) encoder_last_hidden_state (B,T,H)
        enc = get(outputs, "encoder_last_hidden_state")
        if is_seq_tensor(enc):
            return pool_fused(enc)

        # 4) tuple/list outputs: scan for (B,T,H)
        if isinstance(outputs, (list, tuple)):
            for x in outputs:
                if is_seq_tensor(x):
                    return pool_fused(x)

        # 5) nested outputs common in some wrappers
        for key in ["language_model_output", "lm_output", "text_outputs", "outputs", "model_output"]:
            nested = get(outputs, key)
            if nested is not None:
                try:
                    return self.extract_features(nested, attention_mask=attention_mask, debug=debug)
                except Exception:
                    pass

        # 6) dict scan
        if isinstance(outputs, dict):
            for _, v in outputs.items():
                if is_seq_tensor(v):
                    return pool_fused(v)
        

        raise ValueError("No usable (B, T, H) fused hidden states found in outputs.")


# ============================================================
# 2. Backend 로더
# ============================================================
def load_backend(backend: str, model_id: str):
    import transformers
    from transformers import AutoProcessor

    # 기본 dtype
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # ✅ medgemma는 bf16이 fp16보다 안정적 (특히 A100)
    if backend == "medgemma" and device == "cuda":
        torch_dtype = torch.bfloat16

    common = dict(
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )

    if backend == "qwen3":
        from transformers import Qwen3VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_id)
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **common)
        model.eval()
        return model, processor

    if backend == "lingshu":
        from transformers import Qwen2_5_VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_id)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **common)
        model.eval()
        return model, processor

    # ✅ internvl은 trust_remote_code 필수
    if backend == "internvl":
        from transformers import AutoModel, AutoProcessor as AP2
        processor = AP2.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True, **common)
        model.eval()
        return model, processor

    if backend == "medgemma":
        processor = AutoProcessor.from_pretrained(model_id)

        AutoImageTextToText = getattr(transformers, "AutoModelForImageTextToText", None)
        if AutoImageTextToText is not None:
            model = AutoImageTextToText.from_pretrained(model_id, **common)
            model.eval()
            return model, processor

        AutoVision2Seq = getattr(transformers, "AutoModelForVision2Seq", None)
        if AutoVision2Seq is not None:
            model = AutoVision2Seq.from_pretrained(model_id, **common)
            model.eval()
            return model, processor

        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_id, **common)
        model.eval()
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

    df = df[df["severity_norm"].isin(["clean"])].copy()

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
        print(f"Dataset - found {len(self.df)} clean samples.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = row["filepath"]
        img = Image.open(img_path).convert("RGB")

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

        # ✅ 디버깅용 path 포함
        return {"image": img, "input_text": full_text, "label": label, "path": img_path}


def make_clean_only_collate_fn(processor, backend: str):
    def collate(batch):
        images = [b["image"] for b in batch]
        texts  = [b["input_text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        paths  = [b.get("path", "<unknown>") for b in batch]  # ✅

        # qwen3/lingshu: chat template
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

        # internvl: 보통 inline <image> 토큰 필요 or processor 자체가 처리
        elif backend == "internvl":
            image_tok = getattr(processor, "image_token", None) or "<image>"
            inline_texts = [f"{image_tok}\n{txt}" for txt in texts]
            model_inputs = processor(text=inline_texts, images=images, padding=True, return_tensors="pt")

        # medgemma: apply_chat_template로 image placeholder 강제
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
        out["labels_cls"] = labels
        out["paths"] = paths  # ✅ 텐서가 아니니 to_device에서 그대로 유지됨
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
            if k in ("paths", "path", "meta"):
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
    train = optimizer is not None
    vlm_adapt.train() if train else vlm_adapt.eval()
    target_device = next(vlm_adapt.adapter.parameters()).device

    total_loss, n_steps = 0.0, 0
    all_y, all_score = [], []

    with torch.set_grad_enabled(train):
        for batch in loader:
            # paths는 디버깅에 꼭 필요하니 먼저 빼자
            batch = to_device(batch, target_device)
            paths = batch.get("paths", ["<unknown>"])  # ✅ to_device 후

            y = batch["labels_cls"]

            d = dict(batch)
            d.pop("labels_cls", None)
            d.pop("labels_token", None)
            d.pop("paths", None)

            # ✅ (추적1) vision 입력에서 NaN/Inf 확인 (키가 pixel_values가 아닐 수도 있음)
            k_vis, v = get_vision_tensor(d)
            if v is not None and (torch.isnan(v).any() or torch.isinf(v).any()):
                raise RuntimeError(f"NaN/Inf in vision input ({k_vis}) | path={paths[0]} | {finfo_str(v)}")

      
            with torch.inference_mode():
                forward_kwargs = dict(d)

                if vlm_adapt.backend == "internvl" and device == "cuda":
                    # vision 입력 fp32 강제
                    for k in ["pixel_values", "images", "image", "vision_x"]:
                        if k in forward_kwargs and torch.is_tensor(forward_kwargs[k]):
                            forward_kwargs[k] = forward_kwargs[k].to(dtype=torch.float32)
                    # autocast 끄기
                    with torch.cuda.amp.autocast(enabled=False):
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


            attn = d.get("attention_mask", None)
            h_base = vlm_adapt.extract_features(outputs, attention_mask=attn).float()

            if torch.isnan(h_base).any() or torch.isinf(h_base).any():
                msg = f"NaN/Inf in base features | backend={vlm_adapt.backend} | path={paths[0]} | {finfo_str(h_base)}"
                if SKIP_BAD_SAMPLES:
                    print("[skip]", msg)
                    continue
                raise RuntimeError(msg)

            h = vlm_adapt.adapter(h_base)
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

            total_loss += float(loss.item())
            n_steps += 1

    if n_steps == 0:
        return 0.0, {"acc": float("nan"), "f1": float("nan"), "auroc": float("nan"), "auprc": float("nan")}

    y_true = torch.cat(all_y, dim=0)
    y_score = torch.cat(all_score, dim=0)
    metrics = compute_binary_metrics(y_true, y_score, thr=0.5)
    return total_loss / n_steps, metrics




# ============================================================
# 5. BACKEND별 실행
# ============================================================
SAVE_DIR = "./soft_prompt_ckpt"
os.makedirs(SAVE_DIR, exist_ok=True)

for BACKEND in BACKENDS:
    print("\n==============================")
    print(f"🚀 Training backend (CLEAN-ONLY): {BACKEND}")
    print("==============================")

    model_id = MODEL_ID_BY_BACKEND[BACKEND]
    base_model, processor = load_backend(BACKEND, model_id)

    # ✅ device_map 모델이므로 wrapper를 .to(device) 하지 말 것
    vlm_adapt = VLMAdapterWrapper(base_model, backend=BACKEND)

    collate_fn = make_clean_only_collate_fn(processor, BACKEND)
    bs = BATCH_SIZE_BY_BACKEND.get(BACKEND, BATCH_SIZE_DEFAULT)

    if TEST_ONE_ROW:
        one = train_df.iloc[[0]].copy()
        test_ds = CleanOnlyDataset(one, PROMPT_BY_DATASET, SYSTEM_PROMPT_SHORT)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

        batch = next(iter(test_loader))

        target_device = next(vlm_adapt.adapter.parameters()).device
        batch = to_device(batch, target_device)

        d = dict(batch)
        y = d.pop("labels_cls")
        paths = d.pop("paths", ["<unknown>"])
        d.pop("labels_token", None)

        with torch.inference_mode():
            forward_kwargs = dict(d)

            if BACKEND == "internvl" and device == "cuda":
                for k in ["pixel_values", "images", "image", "vision_x"]:
                    if k in forward_kwargs and torch.is_tensor(forward_kwargs[k]):
                        forward_kwargs[k] = forward_kwargs[k].to(dtype=torch.float32)

                with torch.cuda.amp.autocast(enabled=False):
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


            attn = d.get("attention_mask", None)
            h0 = vlm_adapt.extract_features(out, attention_mask=attn,debug=True).float()
            if torch.isnan(h0).any() or torch.isinf(h0).any():
                print("NaN/Inf in TEST_ONE_ROW | path:", paths[0])
                raise RuntimeError("NaN/Inf in base features (TEST_ONE_ROW)")

            h = vlm_adapt.adapter(h0)
            logits = vlm_adapt.classifier(h)

        print(f"[{BACKEND}] OK | path={paths[0]} | h0 {tuple(h0.shape)} | logits {tuple(logits.shape)} | y {tuple(y.shape)}")

        del vlm_adapt, base_model, processor, test_ds, test_loader
        gc.collect()
        torch.cuda.empty_cache()
        continue

    train_ds = CleanOnlyDataset(train_df, PROMPT_BY_DATASET, SYSTEM_PROMPT_SHORT)
    val_ds   = CleanOnlyDataset(val_df,   PROMPT_BY_DATASET, SYSTEM_PROMPT_SHORT)

    print(f"[{BACKEND}] Using batch_size = {bs}")
    print(f"[{BACKEND}] #train={len(train_ds)} #val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, collate_fn=collate_fn)

    optimizer = AdamW(list(vlm_adapt.adapter.parameters()) + list(vlm_adapt.classifier.parameters()), lr=LR)

    best_val_score = -1.0
    BEST_CKPT = os.path.join(SAVE_DIR, f"{BACKEND}_cleanonly_adapter_cls_best.pt")

    for epoch in range(EPOCHS):
        tr_loss, tr_m  = run_epoch_clean_only(vlm_adapt, train_loader, optimizer)
        val_loss, val_m = run_epoch_clean_only(vlm_adapt, val_loader, optimizer=None)

        print(
        f"[{BACKEND}] Epoch {epoch+1}/{EPOCHS} | "
        f"train loss={tr_loss:.4f} acc={tr_m['acc']*100:.2f}% f1={tr_m['f1']:.3f} auroc={tr_m['auroc']:.3f} auprc={tr_m['auprc']:.3f} | "
        f"val loss={val_loss:.4f} acc={val_m['acc']*100:.2f}% f1={val_m['f1']:.3f} auroc={val_m['auroc']:.3f} auprc={val_m['auprc']:.3f}"
        )
        score = val_m["auroc"]  # 추천
        if score > best_val_score:
            best_val_score = score
            torch.save(
                {
                    "adapter": vlm_adapt.adapter.state_dict(),
                    "classifier": vlm_adapt.classifier.state_dict(),
                    "hidden_dim": vlm_adapt.hidden_dim,
                    "backend": BACKEND,
                    "model_id": model_id,
                    "best_val_score (auroc)": best_val_score,
                    "epoch": epoch + 1,
                },
                BEST_CKPT,
            )
            print(f"💾 BEST saved: {BEST_CKPT} (best_val_auroc={best_val_score:.4f})")

    del vlm_adapt, base_model, processor, optimizer, train_ds, val_ds, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()
