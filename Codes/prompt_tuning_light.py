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

from typing import List, Dict, Any, Tuple, Optional, Union

# ============================================================
# 0. 공통 설정
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

TRAIN_CSV = "prompt_tuning_train_750.csv"
VAL_CSV   = "prompt_tuning_val_150.csv"

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

SYSTEM_PROMPT_NORMAL = (
    "You are a medical image classifier.\n"
    "You must answer using ONLY ONE WORD:\n"
    "either \"normal\" or \"disease\".\n\n"
    "Do NOT include any other text, explanation, punctuation,\n"
    "formatting, or symbols. Output exactly one token."
)


def load_backend(backend, model_id):
    """
    backend 타입에 따라 base VLM과 processor 로드
    """
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


# 지금은 InternVL만 학습
BACKENDS = ["internvl","lingshu"]
NUM_VIRTUAL_TOKENS = 20

# CE 가중치 (weak에 더 큰 weight)
w_c = 0.5   # clean
w_w = 1.5  # weak

# consistency 가중치
lambda_w = 0.5  # clean vs weak

LR = 1e-3
EPOCHS = 10
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


# ============================================================
# 1. SoftPrompt 래퍼
# ============================================================
class SoftPromptVLM(nn.Module):
    """
    - base_model: frozen VLM
    - soft_prompt: 앞에 붙는 가상 토큰 임베딩
    - classifier: hidden_size -> 2 (normal / disease)
    """
    def __init__(self, base_model, num_virtual_tokens: int = 20, keep_input_ids: bool = False):
        super().__init__()
        self.base_model = base_model
        self.num_virtual_tokens = num_virtual_tokens
        self.keep_input_ids = keep_input_ids

        # 입력 임베딩에서 hidden size 추출
        input_embeddings = self.base_model.get_input_embeddings()
        hidden_size = input_embeddings.weight.shape[1]
        embed_dtype = input_embeddings.weight.dtype
        embed_device = input_embeddings.weight.device

        # soft prompt embedding
        self.soft_prompt = nn.Embedding(
            num_virtual_tokens,
            hidden_size,
            dtype=embed_dtype,
            device=embed_device,
        )

        # hidden -> 2-class classifier
        self.classifier = nn.Linear(
            hidden_size,
            2,
            device=embed_device,
            dtype=embed_dtype,
        )

        # base_model 파라미터는 freeze
        for p in self.base_model.parameters():
            p.requires_grad = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        output_hidden_states: bool = False,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("SoftPromptVLM.forward requires input_ids")

        # 원본 토큰 임베딩
        input_embeds = self.base_model.get_input_embeddings()(input_ids)
        B, L, H = input_embeds.shape
        dev = input_embeds.device

        # dtype 맞추기
        target_dtype = self.soft_prompt.weight.dtype
        if input_embeds.dtype != target_dtype:
            input_embeds = input_embeds.to(target_dtype)

        # soft prompt 임베딩
        virtual_token_ids = torch.arange(
            self.num_virtual_tokens, device=dev
        ).unsqueeze(0).expand(B, -1)  # (B, V)
        soft_embeds = self.soft_prompt(virtual_token_ids)  # (B, V, H)

        # [soft] + [original]
        inputs_embeds = torch.cat([soft_embeds, input_embeds], dim=1)  # (B, V+L, H)

        # attention mask 확장
        if attention_mask is not None:
            soft_mask = torch.ones(
                (B, self.num_virtual_tokens),
                device=dev,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([soft_mask, attention_mask], dim=1)

        # labels 확장 (CausalLM loss는 안 쓰지만, 혹시 labels 들어오면 안전하게)
        if labels is not None:
            pad = torch.full(
                (B, self.num_virtual_tokens),
                fill_value=-100,
                device=dev,
                dtype=labels.dtype,
            )
            labels = torch.cat([pad, labels], dim=1)

        # 일부 backend는 kwargs에 input_ids 필요 없음
        if "input_ids" in kwargs and not self.keep_input_ids:
            kwargs.pop("input_ids")

        # hidden_states 전체는 안 쓰고,
        # last_hidden_state / encoder_last_hidden_state 위주로 쓸 거라
        # output_hidden_states=False 유지 (OOM 방지)
        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            output_hidden_states=True,  # 👈 추가
            **kwargs,
        )
        return outputs

    def extract_features(self, outputs: Any) -> torch.Tensor:
        """
        다양한 output 타입에 대응해서 (B, H) feature 뽑기.

        우선순위:
          1) outputs.last_hidden_state / outputs["last_hidden_state"]
          2) outputs.encoder_last_hidden_state / outputs["encoder_last_hidden_state"]
          3) outputs.hidden_states[-1]
          4) InternVL류 nested output (language_model_output 등)
          5) tuple/list 에서 (B, T, H) 텐서 추론
        """

        def _has_attr_or_key(obj, name: str):
            if hasattr(obj, name) and getattr(obj, name) is not None:
                return getattr(obj, name)
            if isinstance(obj, dict) and name in obj and obj[name] is not None:
                return obj[name]
            return None

        # ---------- 1) last_hidden_state ----------
        hs = _has_attr_or_key(outputs, "last_hidden_state")
        if isinstance(hs, torch.Tensor):
            if hs.dim() == 3:
                return hs[:, -1, :]  # (B, H)
            elif hs.dim() == 2:
                return hs            # (B, H)

        # ---------- 2) encoder_last_hidden_state ----------
        enc = _has_attr_or_key(outputs, "encoder_last_hidden_state")
        if isinstance(enc, torch.Tensor):
            if enc.dim() == 3:
                return enc.mean(dim=1)  # (B, H)
            elif enc.dim() == 2:
                return enc

        # ---------- 3) hidden_states[-1] ----------
        hidden_states = _has_attr_or_key(outputs, "hidden_states")
        if isinstance(hidden_states, (list, tuple)) and len(hidden_states) > 0:
            hs_last = hidden_states[-1]
            if isinstance(hs_last, torch.Tensor):
                if hs_last.dim() == 3:
                    return hs_last[:, -1, :]
                elif hs_last.dim() == 2:
                    return hs_last

        # ---------- 4) InternVL 계열 nested output ----------
        #   예: outputs.language_model_output.last_hidden_state
        for key in ["language_model_output", "lm_output", "text_outputs"]:
            nested = _has_attr_or_key(outputs, key)
            if nested is not None:
                try:
                    return self.extract_features(nested)
                except ValueError:
                    pass  # 못 뽑으면 다음 후보로

        # ---------- 5) tuple/list fallback ----------
        # return_dict=False 인 모델이나 remote code에서 자주 발생.
        # (loss, logits, hidden_states) / (logits, hidden_states) / (logits,) 등 다양하므로
        # (B, T, H) 형태면서 hidden_size 쪽이 상대적으로 작은 텐서를 골라 씀.
        if isinstance(outputs, (list, tuple)):
            candidate = None
            for x in outputs:
                if isinstance(x, torch.Tensor) and x.dim() == 3:
                    # vocab_size(보통 > 10k) 말고 hidden_size(보통 <= 4096) 쪽을 선택
                    if x.size(-1) <= 4096:
                        candidate = x
                        break
            if candidate is not None:
                return candidate[:, -1, :]

        raise ValueError("No usable hidden representations in outputs.")



# ============================================================
# 2. train/val CSV 로드 (clean/weak만)
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
    """
    fileindex 기준으로 clean / weak 페어를 가져오는 Dataset.
    """
    def __init__(self, df, processor, prompt_by_dataset, system_prompt=None):
        self.df = df.copy()
        self.processor = processor
        self.prompt_by_dataset = prompt_by_dataset
        self.system_prompt = system_prompt  # 지금은 사용 안 함

        self.df["severity_norm"] = self.df["severity"].astype(str).str.lower()
        self.df["dataset_norm"]  = self.df["dataset"].astype(str).str.lower()

        self.pairs = []  # (clean_row, weak_row)

        for fid, g in self.df.groupby("fileindex"):
            g = g.reset_index(drop=True)
            clean_rows = g[g["severity_norm"] == "clean"]
            weak_rows  = g[g["severity_norm"] == "weak"]

            if len(clean_rows) == 0 or len(weak_rows) == 0:
                continue

            # clean은 한 장(anchor)만 쓰고
            clean_row = clean_rows.iloc[0]

            # weak 개수만큼 pair 생성
            for _, weak_row in weak_rows.iterrows():
                self.pairs.append((clean_row, weak_row))
       
        if len(self.pairs) == 0:
            raise RuntimeError("clean/weak 페어 없음. fileindex/severity 확인 필요.")
        print(f"Dataset - found {len(self.pairs)} clean–weak pairs.")

    def _make_sample(self, row):
        img_path = row["filepath"]
        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)

        img = Image.open(img_path).convert("RGB")
        modality = row["dataset_norm"]
        label = int(row["binarylabel"])

        prompt = self.prompt_by_dataset.get(
            modality,
            "This is a medical image.\nQuestion: Does this image show normal anatomy or signs of disease?\n\n",
        )

        # system prompt 제거, 그대로 사용
        full_text = prompt

        label_text = "normal" if label == 0 else "disease"

        return {
            "image": img,
            "input_text": full_text,
            "label_text": label_text,
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

            # Lingshu(Qwen2.5-VL) 전용
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

            # InternVL 전용
            elif backend == "internvl":
                image_token = getattr(processor, "image_token", "<image>")
                inline_texts = [f"{image_token}\n{txt}" for txt in texts]

                model_inputs = processor(
                    text=inline_texts,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                )

            # 기타 backend (나중 확장용)
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

        return {
            "clean": build_inputs("clean"),
            "weak":  build_inputs("weak"),
        }

    return collate


# ============================================================
# 4. 공통 helper (train/val 한 epoch)
# ============================================================
def to_device(batch_dict):
    out = {}
    for k, v in batch_dict.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def last_hidden(sp_model: SoftPromptVLM, outputs):
    """
    SoftPromptVLM.extract_features 사용해서 (B, H) feature 추출
    """
    return sp_model.extract_features(outputs)


def run_epoch(sp_model, loader, optimizer=None):
    train = optimizer is not None
    if train:
        sp_model.train()
    else:
        sp_model.eval()

    total_loss = 0.0
    total_ce   = 0.0
    total_cons = 0.0
    n_steps = 0

    # ---- accuracy 계산용 ----
    correct = 0
    total   = 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            clean = to_device(batch["clean"])
            weak  = to_device(batch["weak"])

            labels_cls_clean = clean["labels_cls"]
            labels_cls_weak  = weak["labels_cls"]

            def forward_one(mode_dict):
                mode_dict = dict(mode_dict)
                mode_dict.pop("labels_token", None)
                mode_dict.pop("labels_cls", None)
                outputs = sp_model(**mode_dict)
                return outputs

            out_c = forward_one(clean)
            out_w = forward_one(weak)

            # 1) representation 추출
            h_c = last_hidden(sp_model, out_c)  # (B, H)
            h_w = last_hidden(sp_model, out_w)  # (B, H)

            # 2) classifier를 통한 CE loss
            logits_c = sp_model.classifier(h_c)  # (B, 2)
            logits_w = sp_model.classifier(h_w)  # (B, 2)

            # ---- accuracy update ----
            with torch.no_grad():
                preds_c = torch.argmax(logits_c, dim=1)
                preds_w = torch.argmax(logits_w, dim=1)

                correct += (preds_c == labels_cls_clean).sum().item()
                correct += (preds_w == labels_cls_weak).sum().item()

                total   += labels_cls_clean.numel()
                total   += labels_cls_weak.numel()

            # ---- loss 계산 ----
            L_c = F.cross_entropy(logits_c, labels_cls_clean)
            L_w = F.cross_entropy(logits_w, labels_cls_weak)
            L_ce = w_c * L_c + w_w * L_w

            L_cons = lambda_w * F.mse_loss(h_c, h_w)
            loss = L_ce + L_cons

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_ce   += L_ce.item()
            total_cons += L_cons.item()
            n_steps += 1

    # ---- 여기부터 return 부분 수정 ----
    if n_steps == 0:
        return 0.0, 0.0, 0.0, 0.0

    acc = correct / total if total > 0 else 0.0
    return (
        total_loss / n_steps,
        total_ce   / n_steps,
        total_cons / n_steps,
        acc,
    )



# ============================================================
# 5. BACKEND별로 순차 학습 (여기서는 internvl만)
# ============================================================
SAVE_DIR = "./soft_prompt_ckpt"
os.makedirs(SAVE_DIR, exist_ok=True)

for BACKEND in BACKENDS:
    print("\n==============================")
    print(f"🚀 Training backend: {BACKEND}")
    print("==============================")

    model_id = MODEL_ID_BY_BACKEND[BACKEND]

    # 1) base_model, processor 로드
    base_model, processor = load_backend(BACKEND, model_id)

    # 2) soft-prompt 래퍼 & optimizer
    keep_input_ids = (BACKEND == "lingshu")
    sp_model = SoftPromptVLM(
        base_model,
        num_virtual_tokens=NUM_VIRTUAL_TOKENS,
        keep_input_ids=keep_input_ids,
    )

    
    # 3) ckpt 경로 설정 (backend별로 다르게)
    ckpt_path = os.path.join(
        SAVE_DIR,
        f"{BACKEND}_soft_prompt_tuning_with_system_900.pt"
    )

    resume = False

    # 4) 체크포인트 있으면 soft_prompt + classifier 로드해서 이어 학습
    if os.path.exists(ckpt_path):
        print(f"🔄 Found checkpoint for {BACKEND} → {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)

        # num_virtual_tokens이 ckpt에 저장돼 있으면 맞춰주기 (옵션)
        n_ckpt = ckpt.get("num_virtual_tokens", NUM_VIRTUAL_TOKENS)
        if n_ckpt != sp_model.num_virtual_tokens:
            print(f"⚠ num_virtual_tokens mismatch: ckpt={n_ckpt}, current={sp_model.num_virtual_tokens}")
            print("   → 새로 SoftPromptVLM을 ckpt 설정에 맞춰 생성합니다.")
            sp_model = SoftPromptVLM(
                base_model,
                num_virtual_tokens=n_ckpt,
                keep_input_ids=keep_input_ids,
            )

        sp_model.soft_prompt.load_state_dict(ckpt["soft_prompt"])
        sp_model.classifier.load_state_dict(ckpt["classifier"])

        resume = True
        print(f"✅ Loaded soft_prompt + classifier for {BACKEND}")
    else:
        print(f"✨ No checkpoint for {BACKEND} — training from scratch")

    optimizer = AdamW(
        list(sp_model.soft_prompt.parameters()) +
        list(sp_model.classifier.parameters()),
        lr=LR,
    )

    # 3) Dataset / DataLoader
    train_ds = CleanWeakPairDataset(
        df=train_df,
        processor=processor,
        prompt_by_dataset=PROMPT_BY_DATASET,
        system_prompt=None,
    )
    val_ds = CleanWeakPairDataset(
        df=val_df,
        processor=processor,
        prompt_by_dataset=PROMPT_BY_DATASET,
        system_prompt=None,
    )

    collate_fn = make_clean_weak_collate_fn(processor, BACKEND)

    bs = BATCH_SIZE_BY_BACKEND.get(BACKEND, BATCH_SIZE_DEFAULT)
    print(f"[{BACKEND}] Using batch_size = {bs}")

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

    print(f"[{BACKEND}] #train_pairs = {len(train_ds)}, #val_pairs = {len(val_ds)}")


    # 4) 학습 루프
    for epoch in range(EPOCHS):
        tr_loss, tr_ce, tr_cons, tr_acc = run_epoch(sp_model, train_loader, optimizer)
        val_loss, val_ce, val_cons, val_acc = run_epoch(sp_model, val_loader, optimizer=None)
        tag = "RESUME" if resume else "FRESH"
        print(
            f"[{BACKEND}][{tag}] Epoch {epoch+1}/{EPOCHS} | "
            f"train: total={tr_loss:.4f}, CE={tr_ce:.4f}, Cons={tr_cons:.4f}, Acc={tr_acc*100:.2f}% | "
            f"val: total={val_loss:.4f}, CE={val_ce:.4f}, Cons={val_cons:.4f}, Acc={val_acc*100:.2f}%"
        )



    # 8) soft_prompt + classifier 저장 (internvl / lingshu 둘 다 공통)
    torch.save(
        {
            "soft_prompt": sp_model.soft_prompt.state_dict(),
            "classifier": sp_model.classifier.state_dict(),
            "num_virtual_tokens": sp_model.num_virtual_tokens,
            "backend": BACKEND,
            "model_id": model_id,
        },
        ckpt_path,
    )
    print(f"✅ Saved soft_prompt for {BACKEND} → {ckpt_path}")

    del sp_model, base_model, processor, optimizer, train_ds, val_ds, train_loader, val_loader
    torch.cuda.empty_cache()
