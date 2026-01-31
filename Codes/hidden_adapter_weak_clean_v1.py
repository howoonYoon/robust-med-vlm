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

# ============================================================
# 0. 공통 설정
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ✅ train/val CSV
TRAIN_CSV = "/home/howoyoon/data/vlm_clean_weak_train_2520.csv"
VAL_CSV   = "/home/howoyoon/data/vlm_clean_weak_val_540.csv"

# ✅ Windows → Linux path prefix 변환 규칙 (pg_last 기준)
WIN_PREFIX   = r"C:\Users\hanna\Lectures\Research_Project\Codes\Dataset\vlm_prompt_dataset"
LINUX_PREFIX = "/home/howoyoon/data"

MODEL_ID_BY_BACKEND = {
    "qwenvl":   "Qwen/Qwen3-VL-8B-Instruct",
    "medgemma": "google/medgemma-27b-it",
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

# ✅ 전부 학습
BACKENDS = ["qwenvl", "medgemma", "lingshu", "internvl"]

# CE 가중치 (weak에 살짝 더 큰 weight)
w_c = 1.0   # clean
w_w = 1.3   # weak

# consistency 가중치
lambda_w = 0.2  # clean vs weak

LR = 2e-4
EPOCHS = 15
BATCH_SIZE_DEFAULT = 8
BATCH_SIZE_BY_BACKEND = {
    "internvl": 1,
    "lingshu":  1,
    "qwenvl":   1,
    "medgemma": 1,
}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if device == "cuda":
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# 1. Adapter + Classifier 정의
# ============================================================
class VisualAdapter(nn.Module):
    def __init__(self, hidden_dim: int, bottleneck: int = 256):
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck)
        self.act  = nn.ReLU()
        self.up   = nn.Linear(bottleneck, hidden_dim)

    def forward(self, h):
        delta = self.up(self.act(self.down(h)))
        return h + delta  # residual


class VLMAdapterWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

        # backbone freeze
        for p in self.base_model.parameters():
            p.requires_grad = False

        input_embeddings = self.base_model.get_input_embeddings()
        hidden_dim   = input_embeddings.weight.shape[1]
        embed_dtype  = input_embeddings.weight.dtype
        embed_device = input_embeddings.weight.device

        self.hidden_dim = hidden_dim

        self.adapter = VisualAdapter(hidden_dim).to(device=embed_device, dtype=embed_dtype)
        self.classifier = nn.Linear(hidden_dim, 2).to(device=embed_device, dtype=embed_dtype)

    def extract_features(self, outputs):
        def get(obj, name):
            if hasattr(obj, name) and getattr(obj, name) is not None:
                return getattr(obj, name)
            if isinstance(obj, dict) and name in obj and obj[name] is not None:
                return obj[name]
            return None

        H = self.hidden_dim

        def reduce_tensor(t: torch.Tensor):
            if t.dim() == 3:
                # (B, T, H) or (B, H, T)
                if t.shape[-1] == H:
                    return t.mean(dim=1)
                if t.shape[1] == H:
                    return t.mean(dim=2)
            elif t.dim() == 2:
                if t.shape[-1] == H:
                    return t
                if t.shape[0] == H:
                    return t.transpose(0, 1)
            return None

        # 1) hidden_states
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

        # 4) nested outputs
        for key in ["language_model_output", "lm_output", "text_outputs", "vision_outputs"]:
            nested = get(outputs, key)
            if nested is not None:
                try:
                    return self.extract_features(nested)
                except Exception:
                    pass

        # 5) tuple/list fallback
        if isinstance(outputs, (list, tuple)):
            for x in outputs:
                if isinstance(x, torch.Tensor):
                    feat = reduce_tensor(x)
                    if feat is not None:
                        return feat

        raise ValueError("No usable hidden representations in outputs.")

# ============================================================
# 2. Backend 로더
# ============================================================
def load_backend(backend, model_id):
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    if backend in ["qwenvl", "qwen3"]:
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
# 3. train/val CSV 로드 (clean/weak만) + 경로 변환
# ============================================================
def load_split_csv(path: str):
    df = pd.read_csv(path)

    # binarylab → binarylabel
    if "binarylab" in df.columns and "binarylabel" not in df.columns:
        df = df.rename(columns={"binarylab": "binarylabel"})

    for col in ["fileindex", "filepath", "severity", "dataset", "binarylabel"]:
        if col not in df.columns:
            raise ValueError(f"{path} 에 '{col}' 컬럼 필요함")

    df["severity_norm"] = df["severity"].astype(str).str.lower()
    df["dataset_norm"]  = df["dataset"].astype(str).str.lower()

    # strong 제외
    df = df[df["severity_norm"].isin(["clean", "weak"])].copy()

    # ✅ Windows path → Linux path 변환
    df["filepath"] = (
        df["filepath"]
        .astype(str)
        .str.replace(WIN_PREFIX, LINUX_PREFIX, regex=False)
        .str.replace("\\", "/", regex=False)
    )

    return df

train_df = load_split_csv(TRAIN_CSV)
val_df   = load_split_csv(VAL_CSV)

print(f"Loaded train_df={len(train_df)} | val_df={len(val_df)}")

# ============================================================
# 4. Dataset / Collate
# ============================================================
class CleanWeakPairDataset(Dataset):
    def __init__(self, df, processor, prompt_by_dataset, system_prompt=None):
        self.df = df.copy()
        self.processor = processor
        self.prompt_by_dataset = prompt_by_dataset
        self.system_prompt = system_prompt

        self.df["severity_norm"] = self.df["severity"].astype(str).str.lower()
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

        print(f"Dataset - found {len(self.pairs)} clean–weak pairs.")

    def _make_sample(self, row):
        img_path = row["filepath"]
        img = Image.open(img_path).convert("RGB")
        modality = row["dataset_norm"]
        label = int(row["binarylabel"])

        task_prompt = self.prompt_by_dataset.get(
            modality,
            "This is a medical image.\nQuestion: Does this image show normal anatomy or signs of disease?\n\n",
        )

        full_text = (self.system_prompt + "\n\n" + task_prompt) if self.system_prompt else task_prompt
        return {"image": img, "input_text": full_text, "label": label}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        clean_row, weak_row = self.pairs[idx]
        return {"clean": self._make_sample(clean_row), "weak": self._make_sample(weak_row)}

def make_clean_weak_collate_fn(processor, backend):
    def collate(batch):
        def build_inputs(which):
            images, texts, labels_int = [], [], []
            for item in batch:
                s = item[which]
                images.append(s["image"])
                texts.append(s["input_text"])
                labels_int.append(s["label"])

            if backend == "lingshu":
                messages_list = []
                for img, txt in zip(images, texts):
                    messages_list.append(
                        [{"role": "user",
                          "content": [{"type": "image", "image": img},
                                      {"type": "text", "text": txt}]}]
                    )
                chat_texts = [
                    processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                    for messages in messages_list
                ]
                model_inputs = processor(text=chat_texts, images=images, padding=True, return_tensors="pt")

            elif backend == "internvl":
                image_token = getattr(processor, "image_token", "<image>")
                inline_texts = [f"{image_token}\n{txt}" for txt in texts]
                model_inputs = processor(text=inline_texts, images=images, return_tensors="pt", padding=True)

            elif backend == "qwenvl":
                model_inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)

            else:
                model_inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)

            out = dict(model_inputs)
            out["labels_cls"] = torch.tensor(labels_int, dtype=torch.long)
            return out

        return {"clean": build_inputs("clean"), "weak": build_inputs("weak")}
    return collate

# ============================================================
# 5. helper: device 이동 + 한 epoch
# ============================================================
def to_device(batch_dict):
    out = {}
    for k, v in batch_dict.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out

def run_epoch(vlm_adapt: VLMAdapterWrapper, loader, optimizer=None):
    train = optimizer is not None
    vlm_adapt.train() if train else vlm_adapt.eval()

    total_loss = total_ce = total_cons = 0.0
    correct = total = 0
    n_steps = 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            clean = to_device(batch["clean"])
            weak  = to_device(batch["weak"])

            y_c = clean["labels_cls"]
            y_w = weak["labels_cls"]

            def forward_base(inputs):
                d = dict(inputs)
                d.pop("labels_cls", None)
                d.pop("labels_token", None)
                try:
                    outputs = vlm_adapt.base_model(**d, output_hidden_states=True)
                except TypeError:
                    outputs = vlm_adapt.base_model(**d)
                return vlm_adapt.extract_features(outputs)  # (B, H)

            h_clean = forward_base(clean)
            h_weak_base = forward_base(weak)

            # weak만 adapter
            h_weak = vlm_adapt.adapter(h_weak_base)

            logits_c = vlm_adapt.classifier(h_clean)
            logits_w = vlm_adapt.classifier(h_weak)

            with torch.no_grad():
                preds_c = torch.argmax(logits_c, dim=1)
                preds_w = torch.argmax(logits_w, dim=1)
                correct += (preds_c == y_c).sum().item()
                correct += (preds_w == y_w).sum().item()
                total   += y_c.numel() + y_w.numel()

            L_c = F.cross_entropy(logits_c, y_c)
            L_w = F.cross_entropy(logits_w, y_w)
            L_ce = w_c * L_c + w_w * L_w

            L_cons = F.mse_loss(h_weak, h_clean.detach())
            loss = L_ce + lambda_w * L_cons

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_ce   += L_ce.item()
            total_cons += L_cons.item()
            n_steps += 1

    if n_steps == 0:
        return 0, 0, 0, 0
    acc = correct / total if total > 0 else 0.0
    return total_loss / n_steps, total_ce / n_steps, total_cons / n_steps, acc

# ============================================================
# 6. BACKEND별 학습
# ============================================================
SAVE_DIR = "./adapter_cls_ckpt"
os.makedirs(SAVE_DIR, exist_ok=True)

for BACKEND in BACKENDS:
    print("\n==============================")
    print(f"🚀 Training backend (adapter): {BACKEND}")
    print("==============================")

    model_id = MODEL_ID_BY_BACKEND[BACKEND]

    base_model, processor = load_backend(BACKEND, model_id)
    vlm_adapt = VLMAdapterWrapper(base_model).to(device)

    optimizer = AdamW(
        list(vlm_adapt.adapter.parameters()) + list(vlm_adapt.classifier.parameters()),
        lr=LR,
    )

    train_ds = CleanWeakPairDataset(
        df=train_df,
        processor=processor,
        prompt_by_dataset=PROMPT_BY_DATASET,
        system_prompt=SYSTEM_PROMPT_SHORT,
    )
    val_ds = CleanWeakPairDataset(
        df=val_df,
        processor=processor,
        prompt_by_dataset=PROMPT_BY_DATASET,
        system_prompt=SYSTEM_PROMPT_SHORT,
    )

    collate_fn = make_clean_weak_collate_fn(processor, BACKEND)

    bs = BATCH_SIZE_BY_BACKEND.get(BACKEND, BATCH_SIZE_DEFAULT)
    print(f"[{BACKEND}] Using batch_size = {bs}")
    print(f"[{BACKEND}] #train_pairs = {len(train_ds)}, #val_pairs = {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, collate_fn=collate_fn)

    best_val_acc = -1.0
    BEST_CKPT = os.path.join(SAVE_DIR, f"{BACKEND}_adapter_cls_best.pt")

    for epoch in range(EPOCHS):
        tr_loss, tr_ce, tr_cons, tr_acc = run_epoch(vlm_adapt, train_loader, optimizer)
        val_loss, val_ce, val_cons, val_acc = run_epoch(vlm_adapt, val_loader, optimizer=None)

        print(
            f"[{BACKEND}][ADAPTER] Epoch {epoch+1}/{EPOCHS} | "
            f"train: total={tr_loss:.4f}, CE={tr_ce:.4f}, Cons={tr_cons:.4f}, Acc={tr_acc*100:.2f}% | "
            f"val: total={val_loss:.4f}, CE={val_ce:.4f}, Cons={val_cons:.4f}, Acc={val_acc*100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "adapter": vlm_adapt.adapter.state_dict(),
                    "classifier": vlm_adapt.classifier.state_dict(),
                    "hidden_dim": vlm_adapt.classifier.in_features,
                    "backend": BACKEND,
                    "model_id": model_id,
                    "best_val_acc": best_val_acc,
                    "epoch": epoch + 1,
                },
                BEST_CKPT,
            )
            print(f"💾 New BEST checkpoint saved! val_acc={best_val_acc*100:.2f}% → {BEST_CKPT}")

    del vlm_adapt, base_model, processor, optimizer, train_ds, val_ds, train_loader, val_loader
    torch.cuda.empty_cache()
