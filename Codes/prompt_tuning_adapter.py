import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from typing import List, Dict, Any, Tuple, Optional, Union

# ============================================================
# 0. 공통 설정
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

TEST_ONE_ROW = True

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

SYSTEM_PROMPT_SHORT = (
    "Answer with ONE WORD: \"normal\" or \"disease\"."
)


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
        # h: (B, H)
        delta = self.up(self.act(self.down(h)))
        return h + delta          # residual


class CLSHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, h):
        return self.fc(h)         # (B, 2)


class VLMAdapterWrapper(nn.Module):
    def __init__(self, base_model, backend: str):
        super().__init__()
        self.base_model = base_model
        self.backend = backend

        # backbone freeze
        for p in self.base_model.parameters():
            p.requires_grad = False

        emb = self._get_text_embeddings(self.base_model)
        hidden_dim = emb.weight.shape[1]
        embed_dtype = emb.weight.dtype
        embed_device = emb.weight.device

        self.hidden_dim = hidden_dim

        self.adapter = VisualAdapter(hidden_dim).to(device=embed_device, dtype=embed_dtype)
        self.classifier = nn.Linear(hidden_dim, 2).to(device=embed_device, dtype=embed_dtype)
        self.base_model.eval()
        
    def _get_text_embeddings(self, m):
        """
        모델마다 text embedding이 숨어있는 위치가 달라서 fallback 탐색.
        """
        # 0) 표준
        if hasattr(m, "get_input_embeddings") and callable(m.get_input_embeddings):
            emb = m.get_input_embeddings()
            if emb is not None:
                return emb

        # 1) language_model 계열
        for attr in ["language_model", "lm", "model", "text_model"]:
            if hasattr(m, attr):
                sub = getattr(m, attr)
                if hasattr(sub, "get_input_embeddings") and callable(sub.get_input_embeddings):
                    emb = sub.get_input_embeddings()
                    if emb is not None:
                        return emb

        # 2) 더 깊은 중첩 탐색 (흔한 케이스)
        # ex) m.language_model.model.embed_tokens 같은 구조
        candidates = [
            ("language_model", "model", "embed_tokens"),
            ("model", "embed_tokens"),
            ("text_model", "embed_tokens"),
            ("language_model", "embed_tokens"),
        ]
        for path in candidates:
            cur = m
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

    def extract_features(self, outputs, attention_mask: Optional[torch.Tensor] = None):
        """
        (B, H) feature로 만드는 공통 루틴.
        가능한 마지막 hidden을 찾아 평균 풀링.
        """
        H = self.hidden_dim

        def masked_mean(hs, attn_mask):
            m = attn_mask.unsqueeze(-1).to(hs.dtype)       # (B,T,1)
            denom = m.sum(dim=1).clamp_min(1.0)            # (B,1)
            return (hs * m).sum(dim=1) / denom             # (B,H)

        def reduce_tensor(t: torch.Tensor):
            if not isinstance(t, torch.Tensor):
                return None
            if t.dim() == 3:
                # (B, T, H) or (B, H, T)
                if t.shape[-1] == H:
                    if attention_mask is not None and attention_mask.dim() == 2:
                        T = min(attention_mask.shape[1], t.shape[1])
                        return masked_mean(t[:, :T, :], attention_mask[:, :T])
                    return t.mean(dim=1)
                if t.shape[1] == H:
                    return t.mean(dim=2)
            if t.dim() == 2:
                # (B, H) or (H, B)
                if t.shape[-1] == H:
                    return t
                if t.shape[0] == H:
                    return t.transpose(0, 1)
            return None

        def get(obj, name):
            if hasattr(obj, name) and getattr(obj, name) is not None:
                return getattr(obj, name)
            if isinstance(obj, dict) and name in obj and obj[name] is not None:
                return obj[name]
            return None

        # 1) hidden_states (가장 확실)
        hss = get(outputs, "hidden_states")
        if isinstance(hss, (list, tuple)) and len(hss) > 0 and isinstance(hss[-1], torch.Tensor):
            feat = reduce_tensor(hss[-1])
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

        # 4) 일부 VLM은 nested output에 들어있음
        for key in ["language_model_output", "lm_output", "text_outputs", "vision_outputs"]:
            nested = get(outputs, key)
            if nested is not None:
                try:
                    return self.extract_features(nested, attention_mask=attention_mask)
                except Exception:
                    pass

        # 5) tuple/list fallback: 텐서 중 hidden_dim 맞는 것 선택
        if isinstance(outputs, (list, tuple)):
            for x in outputs:
                feat = reduce_tensor(x) if isinstance(x, torch.Tensor) else None
                if feat is not None:
                    return feat

        raise ValueError("No usable hidden representations in outputs. (Need hidden_states/last_hidden_state)")

# ============================================================
# 2. Backend 로더
# ============================================================
def load_backend(backend, model_id):
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    if backend == "qwen3":
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_id)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=(torch.float16 if device == "cuda" else torch.float32),
        ).to(device)
        model.eval()
        return model, processor

    if backend in ["medgemma", "internvl"]:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_id)

        # 가능한 한 AutoModelForVision2Seq 쪽이 안정적인 경우가 많음
        try:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=dtype,
            ).to(device)
        except Exception:
            from transformers import AutoModelForVision2Seq
            model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                torch_dtype=dtype,
            ).to(device)

        model.eval()
        return model, processor

    if backend == "lingshu":
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_id)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to(device)
        model.eval()
        return model, processor

    raise ValueError(f"Unknown backend: {backend}")



# 지금은 lingshu, internvl만 학습
BACKENDS = ["qwen3", "medgemma", "internvl", "lingshu"]

# CE 가중치 (weak에 살짝 더 큰 weight)
w_c = 1.0   # clean
w_w = 1.3   # weak

# consistency 가중치
lambda_w = 0.2  # clean vs weak

LR = 2e-4
EPOCHS = 15
BATCH_SIZE_DEFAULT = 15
BATCH_SIZE_BY_BACKEND = {
    "internvl": 1,
    "lingshu": 1,
    "qwen3":  1,
    "medgemma": 1,
}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if device == "cuda":
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 3. train/val CSV 로드 (clean/weak만)
# ============================================================
base_out_dir = os.path.expanduser("~/Scratch/vlm_prompt_dataset")

def load_split_csv(path, base_out_dir):
    df = pd.read_csv(path)

    # binarylab → binarylabel
    if "binarylab" in df.columns and "binarylabel" not in df.columns:
        df = df.rename(columns={"binarylab": "binarylabel"})

    # severity / dataset 정규화
    df["severity_norm"] = df["severity"].astype(str).str.lower()
    df["dataset_norm"]  = df["dataset"].astype(str).str.lower()

    # strong 제외
    df = df[df["severity_norm"].isin(["clean", "weak"])].copy()

    # Windows path → Myriad path 변환
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

train_df = load_split_csv(TRAIN_CSV, base_out_dir)
val_df   = load_split_csv(VAL_CSV, base_out_dir)


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

        if self.system_prompt is not None:
            full_text = self.system_prompt + "\n\n" + task_prompt
        else:
            full_text = task_prompt

        return {
            "image": img,
            "input_text": full_text,
            "label": label,
        }

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        clean_row, weak_row = self.pairs[idx]
        clean_sample = self._make_sample(clean_row)
        weak_sample  = self._make_sample(weak_row)
        return {
            "clean": clean_sample,
            "weak":  weak_sample,
        }


def make_clean_weak_collate_fn(processor, backend):
    def collate(batch):
        def build_inputs(which):
            images, texts, labels_int = [], [], []
            for item in batch:
                s = item[which]
                images.append(s["image"])
                texts.append(s["input_text"])
                labels_int.append(s["label"])

            # ✅ Qwen3-VL도 chat template 사용
            if backend in ["lingshu", "qwen3"]:
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
                    processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    for messages in messages_list
                ]

                model_inputs = processor(
                    text=chat_texts,
                    images=images,
                    padding=True,
                    return_tensors="pt",
                )
                print("[qwen3 inputs keys]", model_inputs.keys())

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

            labels_cls = torch.tensor(labels_int, dtype=torch.long)
            out = dict(model_inputs)
            out["labels_cls"] = labels_cls
            return out

        return {"clean": build_inputs("clean"), "weak": build_inputs("weak")}
    return collate



# ============================================================
# 4. helper: device 이동 + 한 epoch
# ============================================================
def to_device(x):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_device(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v) for v in x)
    return x


def run_epoch(vlm_adapt: VLMAdapterWrapper, loader, optimizer=None):
    train = optimizer is not None
    if train:
        vlm_adapt.train()
    else:
        vlm_adapt.eval()

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

                with torch.no_grad():  # ✅ base freeze니까 no_grad
                    outputs = None
                    try:
                        outputs = vlm_adapt.base_model(**d, output_hidden_states=True)
                    except TypeError:
                        outputs = vlm_adapt.base_model(**d)

                attn = d.get("attention_mask", None)
                h = vlm_adapt.extract_features(outputs, attention_mask=attn)
                return h

            # 1) base feature
            h_clean = forward_base(clean)         # (B, H)
            h_weak_base = forward_base(weak)      # (B, H)

            # 2) adapter 적용 (weak만)
            h_weak = vlm_adapt.adapter(h_weak_base)

            # 3) classifier
            logits_c = vlm_adapt.classifier(h_clean)
            logits_w = vlm_adapt.classifier(h_weak)

            # accuracy
            with torch.no_grad():
                preds_c = torch.argmax(logits_c, dim=1)
                preds_w = torch.argmax(logits_w, dim=1)
                correct += (preds_c == y_c).sum().item()
                correct += (preds_w == y_w).sum().item()
                total   += y_c.numel() + y_w.numel()

            # 4) loss 구성
            L_c = F.cross_entropy(logits_c, y_c)
            L_w = F.cross_entropy(logits_w, y_w)
            L_ce = w_c * L_c + w_w * L_w

            # consistency: clean feature 쪽은 gradient 끊기
            L_cons = F.mse_loss(h_weak, h_clean.detach())

            loss = L_ce + lambda_w * L_cons

            if train:
                optimizer.zero_grad()
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
# 5. BACKEND별 학습
# ============================================================
SAVE_DIR = "./soft_prompt_ckpt"   # 폴더 이름만 재활용
os.makedirs(SAVE_DIR, exist_ok=True)

for BACKEND in BACKENDS:


    print("\n==============================")
    print(f"🚀 Training backend (adapter): {BACKEND}")
    print("==============================")

    model_id = MODEL_ID_BY_BACKEND[BACKEND]

    # 1) base_model, processor 로드
    base_model, processor = load_backend(BACKEND, model_id)

    # 2) adapter wrapper 생성
    vlm_adapt = VLMAdapterWrapper(base_model, backend=BACKEND).to(device)




    optimizer = None
    if not TEST_ONE_ROW:
        optimizer = AdamW(
            list(vlm_adapt.adapter.parameters()) + list(vlm_adapt.classifier.parameters()),
            lr=LR,
        )

    if TEST_ONE_ROW:
        # ✅ CSV에서 row 1개만
        one = train_df.iloc[[0]].copy()

        # CleanWeakPairDataset은 clean/weak 둘 다 있어야 pair가 생김.
        # 그래서 같은 fileindex의 clean+weak를 같이 가져오도록 하는게 안전함.
        fid0 = one["fileindex"].iloc[0]
        one = train_df[train_df["fileindex"] == fid0].copy()
        one = one[one["severity_norm"].isin(["clean", "weak"])].copy()

        test_ds = CleanWeakPairDataset(
            df=one,
            processor=processor,
            prompt_by_dataset=PROMPT_BY_DATASET,
            system_prompt=SYSTEM_PROMPT_SHORT,
        )

        if len(test_ds) == 0:
            print(f"[{BACKEND}] No clean-weak pair for fileindex={fid0}. Skip.")
            del vlm_adapt, base_model, processor, optimizer, test_ds
            gc.collect(); torch.cuda.empty_cache()
            continue

        collate_fn = make_clean_weak_collate_fn(processor, BACKEND)

        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
        )
    else:

        # 3) Dataset / DataLoader
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

    best_val_acc = -1.0
    BEST_CKPT = os.path.join(
        SAVE_DIR,
        f"{BACKEND}_adapter_cls_best.pt"
    )

    if TEST_ONE_ROW:
        batch = next(iter(test_loader))
        clean = to_device(batch["clean"])
        weak  = to_device(batch["weak"])

        def forward_base(inputs):
            d = dict(inputs)
            d.pop("labels_cls", None)
            d.pop("labels_token", None)
            with torch.no_grad():
                try:
                    out = vlm_adapt.base_model(**d, output_hidden_states=True)
                except TypeError:
                    out = vlm_adapt.base_model(**d)
            attn = d.get("attention_mask", None)
            h = vlm_adapt.extract_features(out, attention_mask=attn)
            return h

        with torch.no_grad():
            h_c = forward_base(clean)
            h_w0 = forward_base(weak)
            h_w = vlm_adapt.adapter(h_w0)

            logits_c = vlm_adapt.classifier(h_c)
            logits_w = vlm_adapt.classifier(h_w)

        print(f"[{BACKEND}] OK | h_clean {tuple(h_c.shape)} | h_weak {tuple(h_w.shape)} | logits_c {tuple(logits_c.shape)} | logits_w {tuple(logits_w.shape)}")

        # 다음 모델로
        del vlm_adapt, base_model, processor, optimizer, test_ds, test_loader
        gc.collect()
        torch.cuda.empty_cache()
        continue


    batch0 = next(iter(train_loader))
    clean0 = to_device(batch0["clean"])
    d0 = dict(clean0); d0.pop("labels_cls", None); d0.pop("labels_token", None)

    with torch.no_grad():
        try:
            out0 = vlm_adapt.base_model(**d0, output_hidden_states=True)
        except TypeError:
            out0 = vlm_adapt.base_model(**d0)
    hs = getattr(out0, "hidden_states", None) if not isinstance(out0, dict) else out0.get("hidden_states", None)
    print(f"[{BACKEND}] hidden_states:", None if hs is None else (type(hs), len(hs)))

    attn0 = d0.get("attention_mask", None)
    h0 = vlm_adapt.extract_features(out0, attention_mask=attn0)
    print(f"[{BACKEND}] h0 shape:", tuple(h0.shape))

    

    # 4) 학습 루프
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
    gc.collect()
    torch.cuda.empty_cache()
