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

TEST_ONE_ROW = False   # True면 1샘플 forward만
csv_path = "/SAN/ioo/HORIZON/howoon"

TRAIN_CSV = os.path.join(csv_path, "vlm_clean_train_2520.csv")
VAL_CSV   = os.path.join(csv_path, "vlm_clean_val_540.csv")

MODEL_ID_BY_BACKEND = {
    "qwen3":    "Qwen/Qwen3-VL-8B-Instruct",
    "medgemma": "google/medgemma-1.5-4b-it",
    "internvl": "OpenGVLab/InternVL3_5-8B",
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

    def extract_features(self, outputs, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        H = self.hidden_dim

        def masked_mean(hs: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
            m = attn_mask.unsqueeze(-1).to(hs.dtype)
            denom = m.sum(dim=1).clamp_min(1.0)
            return (hs * m).sum(dim=1) / denom

        def reduce_tensor(t: torch.Tensor) -> Optional[torch.Tensor]:
            if not isinstance(t, torch.Tensor):
                return None
            if t.dim() == 3:
                if t.shape[-1] == H:
                    if attention_mask is not None and attention_mask.dim() == 2:
                        T = min(attention_mask.shape[1], t.shape[1])
                        return masked_mean(t[:, :T, :], attention_mask[:, :T])
                    return t.mean(dim=1)
                if t.shape[1] == H:
                    return t.mean(dim=2)
            if t.dim() == 2:
                if t.shape[-1] == H:
                    return t
                if t.shape[0] == H:
                    return t.transpose(0, 1)
            return None

        def get(obj, name: str):
            if hasattr(obj, name) and getattr(obj, name) is not None:
                return getattr(obj, name)
            if isinstance(obj, dict) and name in obj and obj[name] is not None:
                return obj[name]
            return None

        hss = get(outputs, "hidden_states")
        if isinstance(hss, (list, tuple)) and len(hss) > 0 and isinstance(hss[-1], torch.Tensor):
            feat = reduce_tensor(hss[-1])
            if feat is not None:
                return feat

        hs = get(outputs, "last_hidden_state")
        if isinstance(hs, torch.Tensor):
            feat = reduce_tensor(hs)
            if feat is not None:
                return feat

        enc = get(outputs, "encoder_last_hidden_state")
        if isinstance(enc, torch.Tensor):
            feat = reduce_tensor(enc)
            if feat is not None:
                return feat

        for key in ["language_model_output", "lm_output", "text_outputs", "vision_outputs"]:
            nested = get(outputs, key)
            if nested is not None:
                try:
                    return self.extract_features(nested, attention_mask=attention_mask)
                except Exception:
                    pass

        if isinstance(outputs, (list, tuple)):
            for x in outputs:
                feat = reduce_tensor(x) if isinstance(x, torch.Tensor) else None
                if feat is not None:
                    return feat

        raise ValueError("No usable hidden representations in outputs.")


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
    # ✅ device_map 모델 안전: 입력을 "adapter가 올라간 디바이스"로 맞춘다
    if isinstance(x, torch.Tensor):
        return x.to(target_device, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_device(v, target_device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v, target_device) for v in x)
    return x


def run_epoch_clean_only(vlm_adapt: VLMAdapterWrapper, loader, optimizer=None):
    train = optimizer is not None
    vlm_adapt.train() if train else vlm_adapt.eval()

    # ✅ adapter 파라미터 디바이스로 입력을 맞춤
    target_device = next(vlm_adapt.adapter.parameters()).device

    total_loss = 0.0
    correct = total = 0
    n_steps = 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = to_device(batch, target_device)
            y = batch["labels_cls"]
            paths = batch.get("paths", ["<unknown>"])

            d = dict(batch)
            d.pop("labels_cls", None)
            d.pop("labels_token", None)
            d.pop("paths", None)  # ✅ base_model에 넘기면 안 됨

            with torch.no_grad():
                try:
                    outputs = vlm_adapt.base_model(**d, output_hidden_states=True)
                except TypeError:
                    outputs = vlm_adapt.base_model(**d)

            attn = d.get("attention_mask", None)
            h_base = vlm_adapt.extract_features(outputs, attention_mask=attn)

            # ✅ fp32로 변환해서 안정 학습
            h_base = h_base.float()

            if torch.isnan(h_base).any() or torch.isinf(h_base).any():
                # 디버깅 정보 출력
                print("NaN/Inf in base features | path:", paths[0])
                pv = d.get("pixel_values", None)
                if isinstance(pv, torch.Tensor):
                    try:
                        print("pixel_values:", pv.dtype, pv.device, float(pv.min()), float(pv.max()))
                    except Exception:
                        pass

                if SKIP_BAD_SAMPLES:
                    continue
                raise RuntimeError("NaN/Inf in base features")

            h = vlm_adapt.adapter(h_base)
            logits = vlm_adapt.classifier(h)

            loss = F.cross_entropy(logits, y)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.numel()

            total_loss += float(loss.item())
            n_steps += 1

    if n_steps == 0:
        return 0.0, 0.0
    acc = correct / total if total > 0 else 0.0
    return total_loss / n_steps, acc


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

        with torch.no_grad():
            try:
                out = vlm_adapt.base_model(**d, output_hidden_states=True)
            except TypeError:
                out = vlm_adapt.base_model(**d)

            attn = d.get("attention_mask", None)
            h0 = vlm_adapt.extract_features(out, attention_mask=attn).float()
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

    best_val_acc = -1.0
    BEST_CKPT = os.path.join(SAVE_DIR, f"{BACKEND}_cleanonly_adapter_cls_best.pt")

    for epoch in range(EPOCHS):
        tr_loss, tr_acc = run_epoch_clean_only(vlm_adapt, train_loader, optimizer)
        val_loss, val_acc = run_epoch_clean_only(vlm_adapt, val_loader, optimizer=None)

        print(
            f"[{BACKEND}] Epoch {epoch+1}/{EPOCHS} | "
            f"train loss={tr_loss:.4f} acc={tr_acc*100:.2f}% | "
            f"val loss={val_loss:.4f} acc={val_acc*100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "adapter": vlm_adapt.adapter.state_dict(),
                    "classifier": vlm_adapt.classifier.state_dict(),
                    "hidden_dim": vlm_adapt.hidden_dim,
                    "backend": BACKEND,
                    "model_id": model_id,
                    "best_val_acc": best_val_acc,
                    "epoch": epoch + 1,
                },
                BEST_CKPT,
            )
            print(f"💾 BEST saved: {BEST_CKPT} (val_acc={best_val_acc*100:.2f}%)")

    del vlm_adapt, base_model, processor, optimizer, train_ds, val_ds, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()
