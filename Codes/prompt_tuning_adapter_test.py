import os
import gc
import json
import re
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score


# ========================
# 0. 실행 설정
# ========================

SAVE_FORMAT = "csv"  # "csv" / "json" / "both" (지금은 csv만 사용)
RUN_NAME = "adapter_behaviour_shift_1"

BACKENDS = [
    "internvl",
    "lingshu",
]

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

# ✅ 학습 때 사용했던 system prompt (clean/weak 전부 여기에 포함해서 사용)
SYSTEM_PROMPT_SHORT = (
    "Answer with ONE WORD: \"normal\" or \"disease\"."
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

GEN_KWARGS_BY_BACKEND = {
    "default": dict(max_new_tokens=50, do_sample=False),
}

MODEL_ID_BY_BACKEND = {
    "qwen3":     "Qwen/Qwen3-VL-2B-Instruct",
    "medgemma":  "google/medgemma-4b-it",
    "internvl":  "OpenGVLab/InternVL3_5-2B-HF",
    "lingshu":   "lingshu-medical-mllm/Lingshu-7B",
}

ADAPTER_CKPT_DIR = "./soft_prompt_ckpt"  # adapter ckpt 저장해 둔 폴더

CONSIST_JOIN_KEYS = [
    "fileindex",
    "filename",
    "filepath",
    "IDX",
]


# ========================
# 유틸 함수들
# ========================

def normalize_dataset_name(x: str) -> str:
    if not isinstance(x, str):
        return ""
    s = x.strip().lower()
    mapping = {
        "retina": "fundus",
        "fundus": "fundus",
        "retinamnist": "fundus",
        "xray": "xray",
        "chestxray": "xray",
        "pneumoniamnist": "xray",
        "oct": "oct",
        "octmnist": "oct",
        "mri": "mri",
    }
    return mapping.get(s, s)


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def label_to_binary(label: Optional[str]) -> Optional[int]:
    if label == "disease":
        return 1
    if label == "normal":
        return 0
    return None


def strip_code_fence(t: str) -> str:
    t = (t or "").strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if len(lines) >= 2 and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if len(lines) >= 1 and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def extract_label(text: str) -> Optional[str]:
    """
    공통 라벨 추출:
    - 원래 strong에서 JSON + 'distorted'도 지원했지만
      지금은 weak/clean만 쓰더라도 그대로 두어도 무방.
    """
    t = strip_code_fence(text).strip()
    low = t.lower()

    # 1) 전체 문자열 JSON 파싱
    try:
        obj = json.loads(t)
        label = obj.get("label", None)
        if isinstance(label, str):
            label = label.strip().lower()
        if label in ["normal", "disease", "distorted"]:
            return label
    except Exception:
        pass

    # 2) 안에 있는 첫 { ... } 파싱
    m = re.search(r"\{.*?\}", t, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            label = obj.get("label", None)
            if isinstance(label, str):
                label = label.strip().lower()
            if label in ["normal", "disease", "distorted"]:
                return label
        except Exception:
            pass

    # 3) 텍스트 기반 fallback
    if "distorted" in low or "ungradable" in low or "not gradable" in low:
        return "distorted"

    if re.search(r"\bdisease\b", low) or re.search(r"\babnormal\b", low) or re.search(r"\blesion\b", low):
        return "disease"

    if re.search(r"\bnormal\b", low) or re.search(r"\bno abnormal\b", low):
        if not re.search(r"\bdisease\b|\babnormal\b|\blesion\b", low):
            return "normal"

    # 4) 진짜 한 단어만 왔을 때
    token = re.findall(r"[a-z]+", low)
    if token:
        tok = token[0]
        if tok == "normal":
            return "normal"
        if tok == "disease":
            return "disease"

    return None


def get_image_token(processor) -> str:
    for obj in [processor, getattr(processor, "tokenizer", None), getattr(processor, "processor", None)]:
        if obj is None:
            continue
        for attr in ["image_token", "image_placeholder", "img_token", "image_token_str"]:
            if hasattr(obj, attr):
                v = getattr(obj, attr)
                if isinstance(v, str) and v:
                    return v
    return "<image>"


def can_chat_template(processor) -> bool:
    return hasattr(processor, "apply_chat_template")


def safe_apply_chat_template(processor, messages):
    if not can_chat_template(processor):
        return None
    try:
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return None


# ========================
# 1) BACKEND 로더
# ========================

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


# ========================
# 2) Adapter 래퍼 (eval용)
# ========================

class VisualAdapter(nn.Module):
    def __init__(self, hidden_dim: int, bottleneck: int = 256):
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck)
        self.act  = nn.ReLU()
        self.up   = nn.Linear(bottleneck, hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        delta = self.up(self.act(self.down(h)))
        return h + delta  # residual


class AdapterVLM(nn.Module):
    """
    frozen base_model + adapter + classifier
    eval에서 hidden feature 뽑고 adapter + classifier로 예측
    """
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

        self.adapter = VisualAdapter(hidden_dim).to(
            device=embed_device,
            dtype=embed_dtype,
        )
        self.classifier = nn.Linear(hidden_dim, 2).to(
            device=embed_device,
            dtype=embed_dtype,
        )

    def extract_features(self, outputs: Any) -> torch.Tensor:
        """
        training에서 쓰던 hidden_dim-aware feature 추출 버전과 동일한 형태
        """
        def get(obj, name):
            if hasattr(obj, name) and getattr(obj, name) is not None:
                return getattr(obj, name)
            if isinstance(obj, dict) and name in obj and obj[name] is not None:
                return obj[name]
            return None

        H = self.hidden_dim

        def reduce_tensor(t: torch.Tensor):
            # t: 2D or 3D 텐서에서 hidden dim이 어느 축인지 보고 맞춰줌
            if t.dim() == 3:
                # (B, T, H) 또는 (B, H, T)
                if t.shape[-1] == H:
                    return t.mean(dim=1)          # (B, H)
                if t.shape[1] == H:
                    return t.mean(dim=2)          # (B, H)
            elif t.dim() == 2:
                # (B, H) 혹은 (H, B)
                if t.shape[-1] == H:
                    return t                      # (B, H)
                if t.shape[0] == H:
                    return t.transpose(0, 1)      # (B, H)
            return None

        # 1) hidden_states 우선
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

        # 4) nested (InternVL / Qwen 계열)
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

        # 6) logits fallback (hidden_dim 안 맞으면 무시)
        logits = get(outputs, "logits")
        if isinstance(logits, torch.Tensor):
            feat = reduce_tensor(logits)
            if feat is not None:
                return feat

        raise ValueError("No usable hidden representations in outputs.")


def load_adapter_module(backend: str, base_model) -> Tuple[Optional[AdapterVLM], Optional[str]]:
    """
    training 코드에서 저장한
    { adapter, classifier, hidden_dim, backend, model_id, ... } ckpt 로드.
    """
    ckpt_path = os.path.join(
        ADAPTER_CKPT_DIR,
        f"{backend}_adapter_cls_best.pt",
    )
    if not os.path.exists(ckpt_path):
        print(f"⚠ No adapter ckpt for {backend}: {ckpt_path}")
        return None, None

    print(f"🔄 Using adapter + classifier checkpoint for {backend}: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    ad_model = AdapterVLM(base_model)
    # hidden_dim 체크 (다르면 경고만)
    ckpt_hidden = ckpt.get("hidden_dim", None)
    if ckpt_hidden is not None and ckpt_hidden != ad_model.hidden_dim:
        print(f"⚠ hidden_dim mismatch: ckpt={ckpt_hidden}, model={ad_model.hidden_dim}")

    ad_model.adapter.load_state_dict(ckpt["adapter"])
    ad_model.classifier.load_state_dict(ckpt["classifier"])
    ad_model.to(device)
    ad_model.eval()

    return ad_model, ckpt_path


# ========================
# 3) baseline용 input 빌더 (gen)
# ========================

def build_inputs_for_backend(
    backend: str,
    model,
    processor,
    pil_img: Image.Image,
    img_path: str,
    prompt_text: str,
    system_prompt: Optional[str] = None,
) -> Tuple[Dict[str, torch.Tensor], str]:
    """
    baseline generate용 processor input.
    여기서는 system_prompt를 실제 system role로 안 쓰고
    prompt_text 안에 이미 SYSTEM_PROMPT_SHORT가 포함되어 있다고 가정해도 됨.
    다만 구조는 유지해둘게.
    """
    use_system = bool(system_prompt and system_prompt.strip())

    if backend in ["qwen3", "lingshu"]:
        from qwen_vl_utils import process_vision_info

        messages = []
        if use_system:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": prompt_text},
                ],
            }
        )

        prompt_final = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        imgs, vids = process_vision_info(messages)

        inputs = processor(
            text=[prompt_final],
            images=imgs,
            videos=vids,
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        return inputs, prompt_final

    if backend in ["medgemma", "internvl"]:
        messages = []
        if use_system:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        )

        prompt_final = safe_apply_chat_template(processor, messages)

        if prompt_final is not None:
            inputs = processor(
                text=[prompt_final],
                images=[pil_img],
                return_tensors="pt",
                padding=True,
            ).to(model.device)
            return inputs, prompt_final

        img_tok = get_image_token(processor)
        if use_system:
            prompt_final = f"{img_tok}\n{system_prompt}\n\n{prompt_text}"
        else:
            prompt_final = f"{img_tok}\n{prompt_text}"

        inputs = processor(
            text=[prompt_final],
            images=[pil_img],
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        return inputs, prompt_final

    raise ValueError(f"Unknown backend in build_inputs_for_backend: {backend}")


# ========================
# 4) adapter eval용 input 빌더
# ========================

def build_inputs_for_adapter(
    backend: str,
    processor,
    pil_img: Image.Image,
    prompt_text: str,
):
    """
    adapter 학습 때 사용한 collate_fn 로직을 eval용으로 재현:
    - lingshu: chat_template(add_generation_prompt=False)
    - internvl: "<image_token>\\n{prompt_text}"
    """
    if backend == "lingshu":
        messages_list = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
        ]

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
            images=[pil_img],
            padding=True,
            return_tensors="pt",
        )
        return model_inputs

    elif backend == "internvl":
        image_token = getattr(processor, "image_token", "<image>")
        inline_texts = [f"{image_token}\n{prompt_text}"]

        model_inputs = processor(
            text=inline_texts,
            images=[pil_img],
            return_tensors="pt",
            padding=True,
        )
        return model_inputs

    else:
        # 기타 대비용
        model_inputs = processor(
            text=[prompt_text],
            images=[pil_img],
            return_tensors="pt",
            padding=True,
        )
        return model_inputs


# ========================
# 5) baseline generate (한 단어 / JSON 혼합, 여기서는 SYSTEM_PROMPT_SHORT 포함 텍스트 사용)
# ========================

def run_vlm_baseline_gen(
    backend: str,
    model,
    processor,
    pil_img: Image.Image,
    img_path: str,
    prompt_text: str,
):
    """
    prompt_text 안에 이미 SYSTEM_PROMPT_SHORT + task_prompt가 포함되어 있다고 가정.
    system_prompt는 별도로 쓰지 않고 None으로 둠.
    """
    gen_kwargs = GEN_KWARGS_BY_BACKEND.get(backend, GEN_KWARGS_BY_BACKEND["default"])

    inputs, prompt_final = build_inputs_for_backend(
        backend, model, processor, pil_img, img_path, prompt_text, system_prompt=None
    )

    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)

    if getattr(model.config, "is_encoder_decoder", False):
        gen_ids = out
    else:
        input_len = inputs["input_ids"].shape[1]
        gen_ids = out[:, input_len:]

    output_only = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    full_decoded = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
    return output_only, full_decoded, prompt_final


# ========================
# 6) 데이터 로드
# ========================

base_out_dir = os.path.expanduser("~/Scratch/vlm_prompt_dataset")
os.makedirs(base_out_dir, exist_ok=True)

clean_meta_csv_path = os.path.join(base_out_dir, "clean_test_image_300.csv")
weak_meta_csv_path  = os.path.join(base_out_dir, "weak_test_image_300.csv")

df_clean_base = pd.read_csv(clean_meta_csv_path)
df_weak_base  = pd.read_csv(weak_meta_csv_path)

for df_ in [df_clean_base, df_weak_base]:
    df_["filepath"] = (
        df_["filepath"]
        .astype(str)
        .str.replace(
            r"C:\Users\hanna\Lectures\Research_Project\Codes\Dataset\vlm_prompt_dataset",
            base_out_dir,
            regex=False,
        )
        .str.replace("\\", "/", regex=False)
    )

# weak 메타에서 strong 섞여있을 수 있으니 여기서 weak만 필터링
df_weak_base["severity_norm"] = df_weak_base["severity"].astype(str).str.lower()
df_weak_base = df_weak_base[df_weak_base["severity_norm"] == "weak"].copy()


# ========================
# 7) 실행 + metrics 계산 (clean + weak only)
# ========================

metrics_rows = []

for BACKEND in BACKENDS:
    print("\n==============================")
    print(f"🚀 RUNNING BACKEND (adapter eval): {BACKEND}")
    print("==============================")

    model_id = MODEL_ID_BY_BACKEND[BACKEND]
    try:
        model, processor = load_backend(BACKEND, model_id)
    except Exception as e:
        print(f"❌ FAILED to load backend={BACKEND}, model_id={model_id}")
        print(f"   {type(e).__name__}: {e}")
        cleanup()
        continue

    # adapter + classifier 로드
    ad_model, ckpt_path = load_adapter_module(BACKEND, model)
    has_adapter = ad_model is not None
    if has_adapter:
        print(f"✅ Adapter classifier variant enabled for {BACKEND}")
    else:
        print(f"⚠ Adapter classifier variant DISABLED for {BACKEND}, baseline only.")

    # ------------------------
    # 7-1) CLEAN SET 평가
    # ------------------------
    df_clean = df_clean_base.copy()

    df_clean["model_pred_baseline"] = None
    df_clean["pred_binary_baseline"] = None

    df_clean["model_pred_adapter"] = None
    df_clean["pred_binary_adapter"] = None

    for i, row in df_clean.iterrows():
        img_path = row.get("filepath", None)
        if not isinstance(img_path, str) or not os.path.exists(img_path):
            continue

        try:
            with Image.open(img_path) as im:
                pil_img = im.convert("RGB")

            ds = normalize_dataset_name(row.get("dataset", ""))

            base_prompt = PROMPT_BY_DATASET.get(
                ds,
                "This is a medical image.\nQuestion: Does this image show normal anatomy or signs of disease?\n\n",
            )

            # 학습 때와 동일: SYSTEM_PROMPT_SHORT + task_prompt
            full_prompt = SYSTEM_PROMPT_SHORT + "\n\n" + base_prompt

            # baseline(one-word) — full_prompt를 user 텍스트에 넣어서 gen
            try:
                out_text_base, _, _ = run_vlm_baseline_gen(
                    BACKEND, model, processor, pil_img, img_path, full_prompt
                )
                pred_base = extract_label(out_text_base)
                pred_bin_base = label_to_binary(pred_base)

                df_clean.at[i, "model_pred_baseline"] = pred_base
                df_clean.at[i, "pred_binary_baseline"] = pred_bin_base
            except Exception as e:
                df_clean.at[i, "model_pred_baseline"] = None
                df_clean.at[i, "pred_binary_baseline"] = None

            # adapter classifier (clean)
            if has_adapter:
                try:
                    inputs_ad = build_inputs_for_adapter(
                        BACKEND, processor, pil_img, full_prompt
                    )
                    inputs_ad = {k: v.to(device) for k, v in inputs_ad.items()}

                    with torch.inference_mode():
                        try:
                            outputs_ad = ad_model.base_model(
                                **inputs_ad,
                                output_hidden_states=True,
                                return_dict=True,
                            )
                        except TypeError:
                            outputs_ad = ad_model.base_model(
                                **inputs_ad,
                                return_dict=True,
                            )

                        h = ad_model.extract_features(outputs_ad)
                        h_ad = ad_model.adapter(h)
                        logits = ad_model.classifier(h_ad)
                        probs = torch.softmax(logits, dim=-1)
                        pred_cls = torch.argmax(probs, dim=-1).item()

                    pred_ad = "normal" if pred_cls == 0 else "disease"
                    df_clean.at[i, "model_pred_adapter"] = pred_ad
                    df_clean.at[i, "pred_binary_adapter"] = int(pred_cls)
                except Exception as e:
                    df_clean.at[i, "model_pred_adapter"] = None
                    df_clean.at[i, "pred_binary_adapter"] = None

        except Exception:
            df_clean.at[i, "model_pred_baseline"] = None
            df_clean.at[i, "pred_binary_baseline"] = None
            if has_adapter:
                df_clean.at[i, "model_pred_adapter"] = None
                df_clean.at[i, "pred_binary_adapter"] = None

    df_clean["modality"] = df_clean["dataset"].apply(normalize_dataset_name).astype(str).str.lower()
    if "binarylabel" in df_clean.columns:
        def _to_int01_clean(x):
            try:
                if pd.isna(x):
                    return None
                xi = int(x)
                return xi if xi in [0, 1] else None
            except Exception:
                return None
        df_clean["gt_binary"] = df_clean["binarylabel"].apply(_to_int01_clean)
    else:
        df_clean["gt_binary"] = None

    # baseline clean acc
    baseline_acc: Dict[Tuple[str, str], float] = {}

    print(f"\n--- [CLEAN ACCURACY] backend={BACKEND}, variant=baseline_gen, severity=clean ---")
    for modality_key, g0 in df_clean.groupby("modality"):
        g = g0.copy()
        eval_g = g[
            g["gt_binary"].isin([0, 1]) &
            g["pred_binary_baseline"].isin([0, 1])
        ].copy()
        n_total = int(len(g))
        n_eval = int(len(eval_g))

        if n_eval > 0:
            y_true = eval_g["gt_binary"].astype(int).values
            y_pred = eval_g["pred_binary_baseline"].astype(int).values
            acc_clean = float((y_true == y_pred).mean())
        else:
            acc_clean = float("nan")

        baseline_acc[(BACKEND, str(modality_key).lower())] = acc_clean

        if n_eval > 0:
            print(
                f"  modality={modality_key:8s} | n_total={n_total:3d}, n_eval={n_eval:3d} | acc={acc_clean:.4f}"
            )
        else:
            print(
                f"  modality={modality_key:8s} | n_total={n_total:3d}, n_eval={n_eval:3d} | acc=NaN"
            )

    # consistency 계산용 clean_df 준비 (baseline 기준)
    clean_df_for_consistency = df_clean.copy()
    clean_df_for_consistency["pred_binary"] = clean_df_for_consistency["pred_binary_baseline"]

    # ------------------------
    # 7-2) WEAK SET 평가 (strong은 아예 안 함)
    # ------------------------
    df = df_weak_base.copy()

    df["model_prompt_baseline"] = None
    df["model_raw_output_baseline"] = None
    df["model_pred_baseline"] = None
    df["pred_binary_baseline"] = None

    df["model_prompt_adapter"] = None
    df["model_raw_output_adapter"] = None
    df["model_pred_adapter"] = None
    df["pred_binary_adapter"] = None

    n_ok_base, n_err_base = 0, 0
    n_ok_ad, n_err_ad = 0, 0

    for i, row in df.iterrows():
        img_path = row.get("filepath", None)
        if not isinstance(img_path, str) or not os.path.exists(img_path):
            continue

        try:
            with Image.open(img_path) as im:
                pil_img = im.convert("RGB")

            ds = normalize_dataset_name(row.get("dataset", ""))

            base_prompt = PROMPT_BY_DATASET.get(
                ds,
                "This is a medical image.\nQuestion: Does this image show normal anatomy or signs of disease?\n\n",
            )

            # weak도 clean과 동일하게 SYSTEM_PROMPT_SHORT 포함
            full_prompt = SYSTEM_PROMPT_SHORT + "\n\n" + base_prompt

            # ----- baseline gen -----
            try:
                out_text_base, full_dec_base, prompt_final_base = run_vlm_baseline_gen(
                    BACKEND, model, processor, pil_img, img_path, full_prompt
                )
                pred_base = extract_label(out_text_base)
                pred_bin_base = label_to_binary(pred_base)

                df.at[i, "model_prompt_baseline"] = prompt_final_base
                df.at[i, "model_raw_output_baseline"] = out_text_base
                df.at[i, "model_pred_baseline"] = pred_base
                df.at[i, "pred_binary_baseline"] = pred_bin_base
                n_ok_base += 1
            except Exception as e:
                df.at[i, "model_raw_output_baseline"] = f"__ERROR__ {type(e).__name__}: {e}"
                df.at[i, "model_pred_baseline"] = None
                df.at[i, "pred_binary_baseline"] = None
                n_err_base += 1

            # ----- adapter classifier -----
            if has_adapter:
                try:
                    inputs_ad = build_inputs_for_adapter(
                        BACKEND,
                        processor,
                        pil_img,
                        full_prompt,
                    )
                    inputs_ad = {k: v.to(device) for k, v in inputs_ad.items()}

                    with torch.inference_mode():
                        try:
                            outputs_ad = ad_model.base_model(
                                **inputs_ad,
                                output_hidden_states=True,
                                return_dict=True,
                            )
                        except TypeError:
                            outputs_ad = ad_model.base_model(
                                **inputs_ad,
                                return_dict=True,
                            )

                        h = ad_model.extract_features(outputs_ad)
                        h_ad = ad_model.adapter(h)
                        logits = ad_model.classifier(h_ad)
                        probs = torch.softmax(logits, dim=-1)
                        pred_cls = torch.argmax(probs, dim=-1).item()

                    pred_ad = "normal" if pred_cls == 0 else "disease"
                    pred_bin_ad = int(pred_cls)

                    df.at[i, "model_prompt_adapter"] = full_prompt
                    df.at[i, "model_raw_output_adapter"] = f"__CLS__:{pred_ad}"
                    df.at[i, "model_pred_adapter"] = pred_ad
                    df.at[i, "pred_binary_adapter"] = pred_bin_ad
                    n_ok_ad += 1
                except Exception as e:
                    df.at[i, "model_raw_output_adapter"] = f"__ERROR__ {type(e).__name__}: {e}"
                    df.at[i, "model_pred_adapter"] = None
                    df.at[i, "pred_binary_adapter"] = None
                    n_err_ad += 1

        except Exception as e:
            df.at[i, "model_raw_output_baseline"] = f"__ERROR__ {type(e).__name__}: {e}"
            df.at[i, "model_pred_baseline"] = None
            df.at[i, "pred_binary_baseline"] = None

            if has_adapter:
                df.at[i, "model_raw_output_adapter"] = f"__ERROR__ {type(e).__name__}: {e}"
                df.at[i, "model_pred_adapter"] = None
                df.at[i, "pred_binary_adapter"] = None

    out_csv = os.path.join(base_out_dir, f"{BACKEND}_{RUN_NAME}.csv")
    df.to_csv(out_csv, index=False)

    # ========================
    # Metrics 계산 (weak만)
    # ========================

    df["modality"] = df["dataset"].apply(normalize_dataset_name).astype(str).str.lower()
    df["severity_norm"] = df["severity"].astype(str).str.lower()

    def _to_int01(x):
        try:
            if pd.isna(x):
                return None
            xi = int(x)
            return xi if xi in [0, 1] else None
        except Exception:
            return None

    if "binarylabel" in df.columns:
        df["gt_binary"] = df["binarylabel"].apply(_to_int01)
    else:
        df["gt_binary"] = None

    # consistency join key 자동 선택
    #  - clean_df: clean 이미지 메타 + 각 variant의 clean 예측 포함
    #  - df      : weak/strong 메타 + 각 variant의 weak/strong 예측 포함
    clean_df = clean_df_for_consistency.copy()


    consistency_join_key = None
    for k in CONSIST_JOIN_KEYS:
        if k in df.columns and k in clean_df.columns:
            consistency_join_key = k
            break
    if consistency_join_key is None:
        print(f"⚠ Consistency join key not found for {BACKEND}. Checked: {CONSIST_JOIN_KEYS}")
    else:
        print(f"🔗 Consistency join key for {BACKEND}: {consistency_join_key}")


    # 디버그: adapter classifier 결과 확인 (weak only)
    if BACKEND in ["internvl", "lingshu"] and has_adapter:
        debug_cols = [
            "dataset", "severity", "severity_norm",
            "model_raw_output_adapter",
            "model_pred_adapter",
            "pred_binary_adapter",
            "gt_binary",
        ]
        weak_mask = df["severity_norm"] == "weak"

        print(f"\n[DEBUG] --- {BACKEND} adapter CLS (weak only, head 20) ---")
        print(
            df.loc[weak_mask, debug_cols]
              .head(20)
              .to_string(index=False)
        )

        print(f"\n[DEBUG] value_counts(model_pred_adapter) for weak ({BACKEND}):")
        print(
            df.loc[weak_mask, "model_pred_adapter"]
              .value_counts(dropna=False)
        )

        print(f"\n[DEBUG] value_counts(pred_binary_adapter) for weak ({BACKEND}):")
        print(
            df.loc[weak_mask, "pred_binary_adapter"]
              .value_counts(dropna=False)
        )

    variants = ["baseline"]
    if has_adapter:
        variants.append("adapter")

    for variant in variants:
        pred_col = f"pred_binary_{variant}"
        label_col = "gt_binary"

        variant_name = "baseline_gen" if variant == "baseline" else "adapter_cls"

        # ---------- WEAK ----------
        weak_df = df[df["severity_norm"] == "weak"].copy()

        per_mod_eval_n = {}
        per_mod_base_acc = {}
        per_mod_acc = {}
        per_mod_consistency_base = {}
        per_mod_cons_n_base = {}
        per_mod_consistency_own = {}
        per_mod_cons_n_own = {}


        print(f"\n--- [ACCURACY SUMMARY] backend={BACKEND}, variant={variant_name}, severity=weak ---")

        for modality_key, g0 in weak_df.groupby("modality"):
            g = g0.copy()

            eval_g = g[
                g[label_col].isin([0, 1]) &
                g[pred_col].isin([0, 1])
            ].copy()

            n_total = int(len(g))
            n_eval = int(len(eval_g))

            if n_eval > 0:
                y_true = eval_g[label_col].astype(int).values
                y_pred = eval_g[pred_col].astype(int).values

                acc = float((y_true == y_pred).mean())
                prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
                rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
                f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
            else:
                acc = prec = rec = f1 = float("nan")

            base_acc = baseline_acc.get((BACKEND, str(modality_key).lower()), float("nan"))
            perf_pct_change = float("nan")
            if pd.notna(base_acc) and base_acc != 0 and pd.notna(acc):
                perf_pct_change = (acc - base_acc) / base_acc

            # clean vs weak consistency
            # 🔹 Consistency with clean
            #  - (A) baseline clean 기준: untuned VLM의 clean 예측을 anchor로 사용
            #  - (B) own clean 기준: 해당 variant의 clean 예측을 anchor로 사용
            consistency_base = float("nan")
            cons_n_eval_base = 0
            consistency_own = float("nan")
            cons_n_eval_own = 0

            if clean_df is not None and consistency_join_key is not None:
                key = consistency_join_key

                # ----- (A) baseline clean anchor -----
                base_anchor_col = "pred_binary_baseline"
                if base_anchor_col in clean_df.columns:
                    merged_base = pd.merge(
                        g[[key, pred_col]].copy(),
                        clean_df[[key, base_anchor_col]].copy().rename(
                            columns={base_anchor_col: f"{pred_col}_clean_base"}
                        ),
                        on=key,
                        how="inner",
                    )
                    if not merged_base.empty:
                        valid_base = (
                            merged_base[pred_col].isin([0, 1]) &
                            merged_base[f"{pred_col}_clean_base"].isin([0, 1])
                        )
                        cons_n_eval_base = int(valid_base.sum())
                        if cons_n_eval_base > 0:
                            consistency_base = float(
                                (
                                    merged_base.loc[valid_base, pred_col].astype(int).values ==
                                    merged_base.loc[valid_base, f"{pred_col}_clean_base"].astype(int).values
                                ).mean()
                            )

                # ----- (B) own clean anchor (각 variant의 자기 clean 예측 기준) -----
                own_anchor_col = f"pred_binary_{variant}"
                if own_anchor_col in clean_df.columns:
                    merged_own = pd.merge(
                        g[[key, pred_col]].copy(),
                        clean_df[[key, own_anchor_col]].copy().rename(
                            columns={own_anchor_col: f"{pred_col}_clean_own"}
                        ),
                        on=key,
                        how="inner",
                    )
                    if not merged_own.empty:
                        valid_own = (
                            merged_own[pred_col].isin([0, 1]) &
                            merged_own[f"{pred_col}_clean_own"].isin([0, 1])
                        )
                        cons_n_eval_own = int(valid_own.sum())
                        if cons_n_eval_own > 0:
                            consistency_own = float(
                                (
                                    merged_own.loc[valid_own, pred_col].astype(int).values ==
                                    merged_own.loc[valid_own, f"{pred_col}_clean_own"].astype(int).values
                                ).mean()
                            )


            per_mod_eval_n[modality_key] = n_eval
            per_mod_base_acc[modality_key] = base_acc
            per_mod_acc[modality_key] = acc
            # 모달리티별 consistency 저장
            per_mod_consistency_base[modality_key] = consistency_base
            per_mod_cons_n_base[modality_key]      = cons_n_eval_base
            per_mod_consistency_own[modality_key]  = consistency_own
            per_mod_cons_n_own[modality_key]       = cons_n_eval_own


            # 출력
            if n_eval > 0:
                line = (
                    f"  modality={modality_key:8s} | "
                    f"n_total={n_total:3d}, n_eval={n_eval:3d} | "
                    f"acc={acc:.4f}"
                )
                if pd.notna(base_acc):
                    line += f" | baseline={base_acc:.4f}"
                    if pd.notna(perf_pct_change):
                        line += f" | Δ={(perf_pct_change * 100):+.2f}%"
                if not pd.isna(consistency_base):
                    line += f" | cons_base={consistency_base:.4f} (n={cons_n_eval_base})"
                if not pd.isna(consistency_own):
                    line += f" | cons_own={consistency_own:.4f} (n={cons_n_eval_own})"
                print(line)

            else:
                print(
                    f"  modality={modality_key:8s} | "
                    f"n_total={n_total:3d}, n_eval={n_eval:3d} | acc=NaN"
                )

            metrics_rows.append({
                "run_name": RUN_NAME,
                "model": BACKEND,
                "variant": variant_name,
                "severity": "weak",
                "modality": modality_key,
                "n_total": n_total,
                "n_eval": n_eval,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "baseline_accuracy": base_acc,
                "performance_pct_change": perf_pct_change,
                "distorted_rate": float("nan"),
                # 기존: untuned(baseline) clean 기준 consistency (behaviour shift 관점)
                "consistency_with_clean": consistency_base,
                "consistency_n_eval": cons_n_eval_base,
                # 새로 추가: 각 variant 자기 clean 기준 consistency (robustness 관점)
                "consistency_with_own_clean": consistency_own,
                "consistency_own_n_eval": cons_n_eval_own,
            })


        # 전체 weak
        eval_all = weak_df[
            weak_df[label_col].isin([0, 1]) &
            weak_df[pred_col].isin([0, 1])
        ].copy()

        n_total_all = int(len(weak_df))
        n_eval_all = int(len(eval_all))

        if n_eval_all > 0:
            y_true = eval_all[label_col].astype(int).values
            y_pred = eval_all[pred_col].astype(int).values

            acc_all = float((y_true == y_pred).mean())
            prec_all = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
            rec_all = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
            f1_all = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
        else:
            acc_all = prec_all = rec_all = f1_all = float("nan")

        base_num = 0.0
        base_den = 0.0
        for mod, n_eval_mod in per_mod_eval_n.items():
            b = per_mod_base_acc.get(mod, float("nan"))
            if n_eval_mod > 0 and pd.notna(b):
                base_num += float(n_eval_mod) * float(b)
                base_den += float(n_eval_mod)

        base_acc_all = (base_num / base_den) if base_den > 0 else float("nan")

        perf_pct_change_all = float("nan")
        if pd.notna(base_acc_all) and base_acc_all != 0 and pd.notna(acc_all):
            perf_pct_change_all = (acc_all - base_acc_all) / base_acc_all

        # overall consistency
        # 🔹 overall consistency (weak)
        consistency_all_base = float("nan")
        cons_n_eval_all_base = 0
        consistency_all_own = float("nan")
        cons_n_eval_all_own = 0

        if (
            clean_df is not None and
            consistency_join_key is not None
        ):
            key = consistency_join_key

            # (A) baseline clean anchor
            if "pred_binary_baseline" in clean_df.columns:
                merged_all_base = pd.merge(
                    weak_df[[key, pred_col]].copy(),
                    clean_df[[key, "pred_binary_baseline"]].copy().rename(
                        columns={"pred_binary_baseline": f"{pred_col}_clean_base"}
                    ),
                    on=key,
                    how="inner",
                )
                if not merged_all_base.empty:
                    valid_all_base = (
                        merged_all_base[pred_col].isin([0, 1]) &
                        merged_all_base[f"{pred_col}_clean_base"].isin([0, 1])
                    )
                    cons_n_eval_all_base = int(valid_all_base.sum())
                    if cons_n_eval_all_base > 0:
                        consistency_all_base = float(
                            (
                                merged_all_base.loc[valid_all_base, pred_col].astype(int).values ==
                                merged_all_base.loc[valid_all_base, f"{pred_col}_clean_base"].astype(int).values
                            ).mean()
                        )

            # (B) own clean anchor
            own_anchor_col_all = f"pred_binary_{variant}"
            if own_anchor_col_all in clean_df.columns:
                merged_all_own = pd.merge(
                    weak_df[[key, pred_col]].copy(),
                    clean_df[[key, own_anchor_col_all]].copy().rename(
                        columns={own_anchor_col_all: f"{pred_col}_clean_own"}
                    ),
                    on=key,
                    how="inner",
                )
                if not merged_all_own.empty:
                    valid_all_own = (
                        merged_all_own[pred_col].isin([0, 1]) &
                        merged_all_own[f"{pred_col}_clean_own"].isin([0, 1])
                    )
                    cons_n_eval_all_own = int(valid_all_own.sum())
                    if cons_n_eval_all_own > 0:
                        consistency_all_own = float(
                            (
                                merged_all_own.loc[valid_all_own, pred_col].astype(int).values ==
                                merged_all_own.loc[valid_all_own, f"{pred_col}_clean_own"].astype(int).values
                            ).mean()
                        )


        if n_eval_all > 0 and pd.notna(base_acc_all) and pd.notna(acc_all):
            line = (
                f"  >>> OVERALL (weak) | n_total={n_total_all}, n_eval={n_eval_all} | "
                f"acc={acc_all:.4f} | baseline={base_acc_all:.4f} | "
                f"Δ={(perf_pct_change_all * 100):+.2f}%"
            )
            if not pd.isna(consistency_all_base):
                line += f" | cons_base={consistency_all_base:.4f} (n={cons_n_eval_all_base})"
            if not pd.isna(consistency_all_own):
                line += f" | cons_own={consistency_all_own:.4f} (n={cons_n_eval_all_own})"
            print(line)
        else:
            print(
                f"  >>> OVERALL (weak) | n_total={n_total_all}, n_eval={n_eval_all} | acc=NaN"
            )

        metrics_rows.append({
            "run_name": RUN_NAME,
            "model": BACKEND,
            "variant": variant_name,
            "severity": "weak",
            "modality": "__overall__",
            "n_total": n_total_all,
            "n_eval": n_eval_all,
            "accuracy": acc_all,
            "precision": prec_all,
            "recall": rec_all,
            "f1": f1_all,
            "baseline_accuracy": base_acc_all,
            "performance_pct_change": perf_pct_change_all,
            "distorted_rate": float("nan"),
            "consistency_with_clean": consistency_all_base,
            "consistency_n_eval": cons_n_eval_all_base,
            "consistency_with_own_clean": consistency_all_own,
            "consistency_own_n_eval": cons_n_eval_all_own,
        })

        # ========================
    # 7-3) REAL RETINA DATASET 평가 (매칭 없음 — accuracy만)
    # ========================

    real_csv_path = os.path.join(base_out_dir, "real_retina_40.csv")

    if os.path.exists(real_csv_path):
        print(f"\n==============================")
        print(f"🩺 REAL RETINA DATASET EVAL — backend={BACKEND}")
        print(f"==============================")

        df_real = pd.read_csv(real_csv_path)

        # 경로 변환 (Windows → Myriad)
        df_real["filepath"] = (
            df_real["filepath"]
            .astype(str)
            .str.replace(
                r"C:\Users\hanna\Lectures\Research_Project\Codes\Dataset\vlm_prompt_dataset",
                base_out_dir,
                regex=False,
            )
            .str.replace("\\", "/", regex=False)
        )

        # GT 변환
        def _to_int01_real(x):
            try:
                if pd.isna(x):
                    return None
                xi = int(x)
                return xi if xi in [0, 1] else None
            except:
                return None

        df_real["gt_binary"] = df_real["binarylabel"].apply(_to_int01_real)

        df_real["pred_binary_baseline"] = None
        df_real["pred_binary_adapter"] = None

        for i, row in df_real.iterrows():
            img_path = row.get("filepath", None)
            if not isinstance(img_path, str) or not os.path.exists(img_path):
                continue

            try:
                with Image.open(img_path) as im:
                    pil_img = im.convert("RGB")

                # retina → fundus prompt 사용
                base_prompt = PROMPT_BY_DATASET.get(
                    "fundus",
                    "This is a retinal fundus photograph.\n"
                    "Question: Does this image show normal anatomy or signs of disease?\n\n",
                )

                full_prompt = SYSTEM_PROMPT_SHORT + "\n\n" + base_prompt

                # --- baseline ---
                try:
                    out_text_base, _, _ = run_vlm_baseline_gen(
                        BACKEND, model, processor, pil_img, img_path, full_prompt
                    )
                    pred_base = extract_label(out_text_base)
                    df_real.at[i, "pred_binary_baseline"] = label_to_binary(pred_base)
                except:
                    df_real.at[i, "pred_binary_baseline"] = None

                # --- adapter ---
                if has_adapter:
                    try:
                        inputs_ad = build_inputs_for_adapter(
                            BACKEND, processor, pil_img, full_prompt
                        )
                        inputs_ad = {k: v.to(device) for k, v in inputs_ad.items()}

                        with torch.inference_mode():
                            try:
                                outputs_ad = ad_model.base_model(
                                    **inputs_ad,
                                    output_hidden_states=True,
                                    return_dict=True,
                                )
                            except TypeError:
                                outputs_ad = ad_model.base_model(
                                    **inputs_ad,
                                    return_dict=True,
                                )

                            h = ad_model.extract_features(outputs_ad)
                            h_ad = ad_model.adapter(h)
                            logits = ad_model.classifier(h_ad)
                            probs = torch.softmax(logits, dim=-1)
                            pred_cls = torch.argmax(probs, dim=-1).item()

                        df_real.at[i, "pred_binary_adapter"] = int(pred_cls)

                    except:
                        df_real.at[i, "pred_binary_adapter"] = None

            except:
                continue


        # ===== metrics 계산 =====
        variants = ["baseline"]
        if has_adapter:
            variants.append("adapter")

        for variant in variants:
            col = f"pred_binary_{variant}"
            variant_name = "baseline_gen" if variant == "baseline" else "adapter_cls"

            eval_df = df_real[
                df_real["gt_binary"].isin([0, 1]) &
                df_real[col].isin([0, 1])
            ].copy()

            n_total = len(df_real)
            n_eval = len(eval_df)

            if n_eval > 0:
                y_true = eval_df["gt_binary"].astype(int).values
                y_pred = eval_df[col].astype(int).values
                acc = float((y_true == y_pred).mean())
                prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
                rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
                f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
            else:
                acc = prec = rec = f1 = float("nan")

            print(
                f"  REAL-RETINA | variant={variant_name:12s} | "
                f"n_total={n_total}, n_eval={n_eval} | "
                f"acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}"
            )

            metrics_rows.append({
                "run_name": RUN_NAME,
                "model": BACKEND,
                "variant": variant_name,
                "severity": "real_retina",
                "modality": "fundus",
                "n_total": n_total,
                "n_eval": n_eval,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "baseline_accuracy": float("nan"),
                "performance_pct_change": float("nan"),
                "distorted_rate": float("nan"),
                "consistency_with_clean": float("nan"),
                "consistency_n_eval": 0,
                "consistency_with_own_clean": float("nan"),
                "consistency_own_n_eval": 0,
            })

    else:
        print("\n⚠ real_retina_40.csv not found — skipping real dataset eval.")



    print(f"✅ DONE: {BACKEND}")
    print(f"   baseline_gen: ok={n_ok_base}, errors={n_err_base}")
    if has_adapter:
        print(f"   adapter_cls: ok={n_ok_ad}, errors={n_err_ad}")
    print(f"   CSV : {out_csv}")

    del model, processor
    if ad_model is not None:
        del ad_model
    cleanup()


# ========================
# 8) metrics 전체 CSV 저장
# ========================

if len(metrics_rows) > 0:
    new_df = pd.DataFrame(metrics_rows)

    preferred_order = [
        "run_name",
        "model",
        "variant",
        "severity",
        "modality",
        "n_total",
        "n_eval",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "baseline_accuracy",
        "performance_pct_change",
        "distorted_rate",
        "consistency_with_clean",
        "consistency_n_eval",
    ]
    ordered_cols = [c for c in preferred_order if c in new_df.columns]
    remaining_cols = [c for c in new_df.columns if c not in ordered_cols]
    new_df = new_df[ordered_cols + remaining_cols]

    out_metrics_csv = os.path.join(base_out_dir, "all_metrics_adapter_behaviour_shift.csv")

    if os.path.exists(out_metrics_csv):
        old_df = pd.read_csv(out_metrics_csv)
        all_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        all_df = new_df

    all_df.to_csv(out_metrics_csv, index=False)
    print(f"📄 ALL METRICS CSV (updated): {out_metrics_csv}")
else:
    print("⚠️ No metrics collected.")
