import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from typing import Dict, Any, Optional

# ============================================================
# 0. 공통 설정
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

TRAIN_CSV = "prompt_tuning_train_900.csv"
VAL_CSV   = "prompt_tuning_val_180.csv"

MODEL_ID_BY_BACKEND = {
    "qwen3":    "Qwen/Qwen3-VL-2B-Instruct",
    "medgemma": "google/medgemma-4b-it",
    "internvl": "OpenGVLab/InternVL3_5-2B-HF",
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

# 지금은 lingshu, internvl만 학습
BACKENDS = ["lingshu", "internvl"]

LR = 2e-4
EPOCHS = 15
BATCH_SIZE_DEFAULT = 15
BATCH_SIZE_BY_BACKEND = {
    "internvl": 1,
    "lingshu": 2,
}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if device == "cuda":
    torch.cuda.manual_seed_all(SEED)

# AMP 설정: bfloat16 권장(대부분의 A100/H100 등에서 안정적)
AMP_ENABLED = (device == "cuda")
AMP_DTYPE = torch.bfloat16  # 필요하면 torch.float16로 바꿔도 됨

# ============================================================
# 1. Adapter + Classifier 정의
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
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

        # backbone freeze
        for p in self.base_model.parameters():
            p.requires_grad = False

        # hidden_dim은 input embedding dim으로 추정
        emb = self.base_model.get_input_embeddings()
        self.hidden_dim = emb.weight.shape[1]

        # adapter/classifier는 여기서 생성만 하고
        # 실제 device/dtype 고정은 "첫 배치에서 h 보고" 1회만 수행
        self.adapter = VisualAdapter(self.hidden_dim)
        self.classifier = nn.Linear(self.hidden_dim, 2)

    @staticmethod
    def _masked_mean(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        last_hidden: (B, T, H)
        attention_mask: (B, T) where 1=valid, 0=pad
        """
        m = attention_mask.unsqueeze(-1).to(dtype=last_hidden.dtype)  # (B,T,1)
        denom = m.sum(dim=1).clamp_min(1.0)                           # (B,1)
        return (last_hidden * m).sum(dim=1) / denom                   # (B,H)

    def extract_features(self, outputs: Any, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        가능한 hidden representation을 찾아 (B,H)로 축약.
        (B,T,H)인 경우 attention_mask가 있으면 masked mean 사용.
        """
        def get(obj, name):
            if hasattr(obj, name) and getattr(obj, name) is not None:
                return getattr(obj, name)
            if isinstance(obj, dict) and name in obj and obj[name] is not None:
                return obj[name]
            return None

        H = self.hidden_dim

        def reduce_tensor(t: torch.Tensor) -> Optional[torch.Tensor]:
            if t.dim() == 3:
                # (B,T,H)
                if t.shape[-1] == H:
                    if (
                        attention_mask is not None
                        and attention_mask.dim() == 2
                        and attention_mask.shape[0] == t.shape[0]
                        and attention_mask.shape[1] == t.shape[1]
                    ):
                        return self._masked_mean(t, attention_mask)
                    return t.mean(dim=1)
                # (B,H,T)
                if t.shape[1] == H:
                    return t.mean(dim=2)
            elif t.dim() == 2:
                # (B,H)
                if t.shape[-1] == H:
                    return t
                # (H,B)
                if t.shape[0] == H:
                    return t.transpose(0, 1)
            return None

        # 1) hidden_states 우선 (마지막 레이어)
        hss = get(outputs, "hidden_states")
        if isinstance(hss, (list, tuple)) and len(hss) > 0:
            hs_last = hss[-1]
            if isinstance(hs_last, torch.Tensor):
                feat = reduce_tensor(hs_last)
                if feat is not None:
                    return feat

        # 2) last_hidden_state
        hs = get(outputs, "last_hidden_state")
        if isinstance(hs, torch.Tensor):
            feat = reduce_tensor(hs)
            if feat is not None:
                return feat

        # 3) encoder_last_hidden_state
        enc = get(outputs, "encoder_last_hidden_state")
        if isinstance(enc, torch.Tensor):
            feat = reduce_tensor(enc)
            if feat is not None:
                return feat

        # 4) nested outputs (InternVL/Qwen 계열)
        for key in ["language_model_output", "lm_output", "text_outputs", "vision_outputs"]:
            nested = get(outputs, key)
            if nested is not None:
                try:
                    return self.extract_features(nested, attention_mask=attention_mask)
                except Exception:
                    pass

        # 5) tuple/list fallback
        if isinstance(outputs, (list, tuple)):
            for x in outputs:
                if isinstance(x, torch.Tensor):
                    feat = reduce_tensor(x)
                    if feat is not None:
                        return feat

        # 6) logits fallback (최후 수단)
        logits = get(outputs, "logits")
        if isinstance(logits, torch.Tensor):
            feat = reduce_tensor(logits)
            if feat is not None:
                return feat

        raise ValueError("No usable hidden representations in outputs.")


# ============================================================
# 2. Backend 로더
# ============================================================
def load_backend(backend: str, model_id: str):
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    if backend == "qwen3":
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else {"": "cpu"},
        )
        processor = AutoProcessor.from_pretrained(model_id)
        model.eval()
        return model, processor

    if backend in ["medgemma", "internvl"]:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_id)
        try:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else {"": "cpu"},
            )
        except Exception:
            from transformers import AutoModelForVision2Seq
            model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else {"": "cpu"},
            )
        model.eval()
        return model, processor

    if backend == "lingshu":
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else {"": "cpu"},
        )
        processor = AutoProcessor.from_pretrained(model_id)
        model.eval()
        return model, processor

    raise ValueError(f"Unknown backend: {backend}")


# ============================================================
# 3. train/val CSV 로드 (clean만)
# ============================================================
base_out_dir = os.path.expanduser("~/Scratch/vlm_prompt_dataset")

def load_split_csv_clean_only(path: str, base_out_dir: str) -> pd.DataFrame:
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

    return df

train_df = load_split_csv_clean_only(TRAIN_CSV, base_out_dir)
val_df   = load_split_csv_clean_only(VAL_CSV, base_out_dir)


class CleanOnlyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, processor, prompt_by_dataset, system_prompt: Optional[str] = None):
        self.df = df.copy()
        self.processor = processor
        self.prompt_by_dataset = prompt_by_dataset
        self.system_prompt = system_prompt
        self.df["dataset_norm"] = self.df["dataset"].astype(str).str.lower()
        print(f"Dataset(clean-only) - n={len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
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

        return {"image": img, "input_text": full_text, "label": label}


def make_clean_collate_fn(processor, backend: str):
    def collate(batch):
        images, texts, labels = [], [], []
        for s in batch:
            images.append(s["image"])
            texts.append(s["input_text"])
            labels.append(int(s["label"]))

        if backend == "lingshu":
            messages_list = []
            for img, txt in zip(images, texts):
                messages_list.append(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": txt},
                            ],
                        }
                    ]
                )

            chat_texts = [
                processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                for messages in messages_list
            ]

            model_inputs = processor(
                text=chat_texts,
                images=images,
                padding=True,
                return_tensors="pt",
            )

        elif backend == "internvl":
            image_token = getattr(processor, "image_token", "<image>")
            inline_texts = [f"{image_token}\n{txt}" for txt in texts]
            model_inputs = processor(
                text=inline_texts,
                images=images,
                return_tensors="pt",
                padding=True,
            )

        else:
            model_inputs = processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
            )

        out = dict(model_inputs)
        out["labels_cls"] = torch.tensor(labels, dtype=torch.long)
        return out

    return collate


# ============================================================
# 4. helper: device 이동 + safe forward filtering
# ============================================================
def to_device(batch: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out

def filter_model_inputs(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    forward에 넘길 키를 보수적으로 필터링.
    (backend/processor마다 반환 키가 달라서, 에러 방지 목적)
    """
    allowed = {
        "input_ids", "attention_mask", "token_type_ids",
        "pixel_values", "image_grid_thw",
        "position_ids",
        "inputs_embeds",
    }
    return {k: v for k, v in d.items() if k in allowed}

def forward_features(vlm_adapt: VLMAdapterWrapper, batch: Dict[str, Any]) -> torch.Tensor:
    """
    base_model forward + feature 추출 (masked mean 포함)
    """
    d = dict(batch)
    y = d.pop("labels_cls")
    attn_mask = d.get("attention_mask", None)
    d = filter_model_inputs(d)

    with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=AMP_ENABLED):
        outputs = vlm_adapt.base_model(**d, output_hidden_states=True, use_cache=False)
        h = vlm_adapt.extract_features(outputs, attention_mask=attn_mask)
    return h, y

def init_heads_once(vlm_adapt: VLMAdapterWrapper, train_loader: DataLoader) -> None:
    """
    device_map=auto 환경에서 adapter/classifier를 올바른 device/dtype로 1회 고정하기 위해,
    첫 배치 1회 forward로 h의 device/dtype을 확인한다.
    """
    first_batch = next(iter(train_loader))
    first_batch = to_device(first_batch)

    with torch.no_grad():
        h, _ = forward_features(vlm_adapt, first_batch)

    vlm_adapt.adapter.to(device=h.device, dtype=h.dtype)
    vlm_adapt.classifier.to(device=h.device, dtype=h.dtype)

def run_epoch_clean_only(vlm_adapt: VLMAdapterWrapper, loader: DataLoader, optimizer: Optional[AdamW] = None):
    train = optimizer is not None
    vlm_adapt.train() if train else vlm_adapt.eval()

    # bf16이면 scaler 없어도 되지만(GradScaler는 fp16 중심),
    # fp16로 바꿀 수 있으니 enabled 조건만 둬서 안전하게 유지
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and AMP_ENABLED and AMP_DTYPE == torch.float16))

    total_loss = 0.0
    correct = 0
    total = 0
    n_steps = 0

    for batch in loader:
        batch = to_device(batch)

        h, y = forward_features(vlm_adapt, batch)

        # logits 계산 (heads는 이미 올바른 device/dtype으로 고정되어 있어야 함)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=AMP_ENABLED):
            logits = vlm_adapt.classifier(vlm_adapt.adapter(h))
            loss = F.cross_entropy(logits, y.to(device=logits.device))

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item())
        n_steps += 1

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == y.to(device=logits.device)).sum().item())
            total += int(y.numel())

    avg_loss = total_loss / max(n_steps, 1)
    acc = float(correct) / float(total) if total > 0 else 0.0
    return float(avg_loss), float(acc)


# ============================================================
# 5. BACKEND별 학습 (clean-only)
# ============================================================
SAVE_DIR = "./clean_only_adapter_ckpt"
os.makedirs(SAVE_DIR, exist_ok=True)

for BACKEND in BACKENDS:
    print("\n==============================")
    print(f"🚀 Training backend (clean-only adapter): {BACKEND}")
    print("==============================")

    model_id = MODEL_ID_BY_BACKEND[BACKEND]

    # 1) base_model, processor 로드
    base_model, processor = load_backend(BACKEND, model_id)

    # 2) wrapper 생성
    vlm_adapt = VLMAdapterWrapper(base_model)

    # 3) Dataset / DataLoader
    train_ds = CleanOnlyDataset(
        df=train_df,
        processor=processor,
        prompt_by_dataset=PROMPT_BY_DATASET,
        system_prompt=SYSTEM_PROMPT_SHORT,
    )
    val_ds = CleanOnlyDataset(
        df=val_df,
        processor=processor,
        prompt_by_dataset=PROMPT_BY_DATASET,
        system_prompt=SYSTEM_PROMPT_SHORT,
    )

    collate_fn = make_clean_collate_fn(processor, BACKEND)

    bs = BATCH_SIZE_BY_BACKEND.get(BACKEND, BATCH_SIZE_DEFAULT)
    print(f"[{BACKEND}] Using batch_size = {bs}")
    print(f"[{BACKEND}] #train = {len(train_ds)}, #val = {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # ✅ 4) heads를 "첫 배치로" 올바른 device/dtype에 1회 고정
    init_heads_once(vlm_adapt, train_loader)

    # ✅ 5) optimizer는 heads 이동 후 생성 (AdamW 파라미터 참조 고정 문제 방지)
    optimizer = AdamW(
        list(vlm_adapt.adapter.parameters()) + list(vlm_adapt.classifier.parameters()),
        lr=LR,
    )

    best_val_acc = -1.0
    BEST_CKPT = os.path.join(SAVE_DIR, f"{BACKEND}_clean_adapter_cls_best.pt")

    for epoch in range(EPOCHS):
        tr_loss, tr_acc = run_epoch_clean_only(vlm_adapt, train_loader, optimizer)
        val_loss, val_acc = run_epoch_clean_only(vlm_adapt, val_loader, optimizer=None)

        tr_acc = float(tr_acc)
        val_acc = float(val_acc)

        print(
            f"[{BACKEND}][CLEAN] Epoch {epoch+1}/{EPOCHS} | "
            f"train: loss={tr_loss:.4f}, Acc={tr_acc*100:.2f}% | "
            f"val: loss={val_loss:.4f}, Acc={val_acc*100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
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
            print(f"💾 New BEST checkpoint saved! val_acc={best_val_acc*100:.2f}% → {BEST_CKPT}")

    # 정리
    del vlm_adapt, base_model, processor, optimizer, train_ds, val_ds, train_loader, val_loader
    if device == "cuda":
        torch.cuda.empty_cache()
