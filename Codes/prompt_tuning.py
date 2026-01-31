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


from typing import List, Dict, Any, Tuple, Optional,Union

# ============================================================
# 0. 공통 설정
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

TRAIN_CSV = "prompt_tuning_train_750.csv"
VAL_CSV   = "prompt_tuning_val_150.csv"

# 이 딕셔너리/프롬프트는 네 기존 코드에서 이미 정의돼 있다고 가정
MODEL_ID_BY_BACKEND = { "qwen3": "Qwen/Qwen3-VL-2B-Instruct", "medgemma": "google/medgemma-4b-it", "internvl": "OpenGVLab/InternVL3_5-8B-HF", "lingshu": "lingshu-medical-mllm/Lingshu-7B", }
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
    "Respond using ONLY valid JSON.\n"
    "Output exactly one JSON object with one key called \"label\".\n"
    "The value of \"label\" MUST be either \"normal\" or \"disease\".\n"
    "Do NOT include any explanation, text, or formatting outside the JSON."
)



def load_backend(backend, model_id):
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


BACKENDS = ["lingshu", "internvl"]
NUM_VIRTUAL_TOKENS = 20

# CE 가중치 (weak에 더 큰 weight)
w_c = 1.0   # clean
w_w = 2.0   # weak

# consistency 가중치
lambda_w = 0.5  # clean vs weak

LR = 1e-3
EPOCHS = 5
BATCH_SIZE = 4

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
    def __init__(self, base_model, num_virtual_tokens: int = 20, keep_input_ids: bool = False):
        super().__init__()
        self.base_model = base_model
        self.num_virtual_tokens = num_virtual_tokens
        self.keep_input_ids = keep_input_ids

        # base_model 임베딩 정보 가져오기
        input_embeddings = self.base_model.get_input_embeddings()
        hidden_size = input_embeddings.weight.shape[1]
        embed_dtype = input_embeddings.weight.dtype
        embed_device = input_embeddings.weight.device

        # 🔥 base_model 임베딩과 동일한 dtype/device로 soft_prompt 생성
        self.soft_prompt = nn.Embedding(
            num_virtual_tokens,
            hidden_size,
            dtype=embed_dtype,
            device=embed_device,
        )

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
        device = input_embeds.device

        # 🔥 dtype 맞추기 (혹시라도 다르면 soft_prompt 쪽에 맞춰줌)
        target_dtype = self.soft_prompt.weight.dtype
        if input_embeds.dtype != target_dtype:
            input_embeds = input_embeds.to(target_dtype)

        # soft prompt 임베딩
        virtual_token_ids = torch.arange(
            self.num_virtual_tokens, device=device
        ).unsqueeze(0).expand(B, -1)
        soft_embeds = self.soft_prompt(virtual_token_ids)  # (B, V, H)

        # [soft] + [original]
        inputs_embeds = torch.cat([soft_embeds, input_embeds], dim=1)  # (B, V+L, H)

        # attention mask 확장
        if attention_mask is not None:
            soft_mask = torch.ones(
                (B, self.num_virtual_tokens),
                device=device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([soft_mask, attention_mask], dim=1)

        # labels 확장
        if labels is not None:
            pad = torch.full(
                (B, self.num_virtual_tokens),
                fill_value=-100,
                device=device,
                dtype=labels.dtype,
            )
            labels = torch.cat([pad, labels], dim=1)

        # internvl 같은 애들은 input_ids 제거, lingshu는 유지
        if "input_ids" in kwargs and not self.keep_input_ids:
            kwargs.pop("input_ids")

        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        return outputs





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

    # 🔥 strong 제외
    df = df[df["severity_norm"].isin(["clean", "weak"])].copy()

    # 🔥 Windows path → Myriad path 변환
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
        self.system_prompt = system_prompt

        self.df["severity_norm"] = self.df["severity"].astype(str).str.lower()
        self.df["dataset_norm"]  = self.df["dataset"].astype(str).str.lower()

        self.pairs = []  # (clean_row, weak_row)

        for fid, g in self.df.groupby("fileindex"):
            g = g.reset_index(drop=True)
            clean_rows = g[g["severity_norm"] == "clean"]
            weak_rows  = g[g["severity_norm"] == "weak"]
            if len(clean_rows) == 0 or len(weak_rows) == 0:
                continue
            self.pairs.append(
                (clean_rows.iloc[0], weak_rows.iloc[0])
            )

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

        prompt = PROMPT_BY_DATASET.get(
            modality,
            "This is a medical image.\nQuestion: Does this image show normal anatomy or signs of disease?\n\n",
        )

        full_text = prompt
        if SYSTEM_PROMPT_NORMAL:
            full_text = SYSTEM_PROMPT_NORMAL + "\n" + prompt

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
        # 라벨 텍스트 인코딩 헬퍼
        def encode_labels(label_texts):
            if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
                enc = processor.tokenizer(
                    text=label_texts,
                    return_tensors="pt",
                    padding=True,
                )
                return enc["input_ids"]
            else:
                enc = processor(
                    text=label_texts,
                    return_tensors="pt",
                    padding=True,
                )
                return enc["input_ids"]

        def build_inputs(which):
            images, texts, label_texts, labels_int = [], [], [], []
            for item in batch:
                s = item[which]
                images.append(s["image"])
                texts.append(s["input_text"])
                label_texts.append(s["label_text"])
                labels_int.append(s["label"])

            # 🔹 Lingshu(Qwen2.5-VL) 전용 처리
            if backend == "lingshu":
                # 각 샘플마다 messages 만들고 chat 텍스트 생성
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

                # chat_texts 길이 == batch_size, images 길이 == batch_size
                model_inputs = processor(
                    text=chat_texts,
                    images=images,
                    padding=True,
                    return_tensors="pt",
                )

            # 🔹 InternVL 등 일반 케이스
            else:
                model_inputs = processor(
                    text=texts,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                )

            labels_token = encode_labels(label_texts)
            labels_cls   = torch.tensor(labels_int, dtype=torch.long)

            out = dict(model_inputs)
            out["labels_token"] = labels_token
            out["labels_cls"]   = labels_cls
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

def last_hidden(outputs):
    hs = outputs.hidden_states[-1]  # (B, T, H)
    return hs[:, -1, :]             # (B, H)

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

    with torch.set_grad_enabled(train):
        for batch in loader:
            clean = to_device(batch["clean"])
            weak  = to_device(batch["weak"])

            labels_clean = clean["labels_token"]
            labels_weak  = weak["labels_token"]

            def forward_one(mode_dict, labels_token):
                mode_dict = dict(mode_dict)
                
                # 1. 라벨 설정
                mode_dict["labels"] = labels_token
                
                # 2. 불필요한 키 제거 (DataLoader에서 올라온 임시 키들)
                mode_dict.pop("labels_token", None)
                mode_dict.pop("labels_cls", None)
                
                # 🔥 3. Qwen/InternVL 에러 방지: 
                # SoftPromptVLM 내부에서 inputs_embeds를 만들 것이므로, 
                # 원본 input_ids는 arguments로 직접 넘기되, 
                # 모델(sp_model)이 내부에서 input_ids를 None으로 처리하게 해야 합니다.
                # (위의 SoftPromptVLM 수정사항이 적용되었다면 여기서는 그대로 넘겨도 됩니다.)
                
                outputs = sp_model(**mode_dict, output_hidden_states=True)
                return outputs

            out_c = forward_one(clean, labels_clean)
            out_w = forward_one(weak,  labels_weak)

            L_c = out_c.loss
            L_w = out_w.loss
            L_ce = w_c * L_c + w_w * L_w

            h_c = last_hidden(out_c)
            h_w = last_hidden(out_w)
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

    if n_steps == 0:
        return 0.0, 0.0, 0.0
    return total_loss / n_steps, total_ce / n_steps, total_cons / n_steps


# ============================================================
# 5. BACKEND별로 순차 학습 (lingshu, internvl)
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
    base_model.to(device)

    # 2) soft-prompt 래퍼 & optimizer
    keep_input_ids = (BACKEND == "lingshu")  # lingshu(Qwen2.5-VL)만 True
    sp_model = SoftPromptVLM(
        base_model,
        num_virtual_tokens=NUM_VIRTUAL_TOKENS,
        keep_input_ids=keep_input_ids,
    )
    sp_model.to(device)
    optimizer = AdamW(sp_model.soft_prompt.parameters(), lr=LR)

    # 3) Dataset / DataLoader (backend별 processor 사용)
    train_ds = CleanWeakPairDataset(
        df=train_df,
        processor=processor,
        prompt_by_dataset=PROMPT_BY_DATASET,
        system_prompt=SYSTEM_PROMPT_NORMAL,
    )
    val_ds = CleanWeakPairDataset(
        df=val_df,
        processor=processor,
        prompt_by_dataset=PROMPT_BY_DATASET,
        system_prompt=SYSTEM_PROMPT_NORMAL,
    )

    # 기존: collate_fn = make_clean_weak_collate_fn(processor)
    collate_fn = make_clean_weak_collate_fn(processor, BACKEND)


    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # 4) 학습 루프
    for epoch in range(EPOCHS):
        tr_loss, tr_ce, tr_cons = run_epoch(sp_model, train_loader, optimizer)
        val_loss, val_ce, val_cons = run_epoch(sp_model, val_loader, optimizer=None)

        print(
            f"[{BACKEND}] Epoch {epoch+1}/{EPOCHS} | "
            f"train: total={tr_loss:.4f}, CE={tr_ce:.4f}, Cons={tr_cons:.4f} | "
            f"val: total={val_loss:.4f}, CE={val_ce:.4f}, Cons={val_cons:.4f}"
        )

    # 5) soft_prompt 저장
    ckpt_path = os.path.join(
        SAVE_DIR,
        f"{BACKEND}_soft_prompt_prompt_tuning_900.pt"
    )
    torch.save(
        {
            "state_dict": sp_model.soft_prompt.state_dict(),
            "num_virtual_tokens": sp_model.num_virtual_tokens,
            "backend": BACKEND,
            "model_id": model_id,
        },
        ckpt_path,
    )
    print(f"✅ Saved soft_prompt for {BACKEND} → {ckpt_path}")

    # 메모리 정리
    del sp_model, base_model, processor, optimizer, train_ds, val_ds, train_loader, val_loader
    torch.cuda.empty_cache()
