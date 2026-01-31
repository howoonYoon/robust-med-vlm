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
RUN_NAME = "cls_behaviour_shift_1"  # 이름만 살짝 변경 (원하면 다시 json으로)

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

# ✅ weak/clean 에서 사용할 한 단어 system prompt
SYSTEM_PROMPT_NORMAL = (
    "You are a medical image classifier.\n"
    "You must answer using ONLY ONE WORD:\n"
    "either \"normal\" or \"disease\".\n\n"
    "Do NOT include any other text, explanation, punctuation,\n"
    "formatting, or symbols. Output exactly one token."
)

# strong 에서는 여전히 JSON + distorted 사용 (그대로 유지)
SYSTEM_PROMPT_STRONG = (
    "You are a medical image classifier.\n"
    "Respond using ONLY valid JSON.\n"
    "Output exactly one JSON object with one key called \"label\".\n"
    "The value of \"label\" MUST be one of: \"normal\", \"disease\", or \"distorted\".\n"
    "Use \"distorted\" only if the image quality is too poor to make a reliable decision.\n"
    "Do NOT include any explanation, text, or formatting outside the JSON."
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

SOFTPROMPT_CKPT_DIR = "./soft_prompt_ckpt"


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
    공통 라벨 추출 함수.
    - strong: JSON + 'distorted' 포함
    - weak/clean: SYSTEM_PROMPT_NORMAL 때문에 보통 "normal" 또는 "disease" 한 단어.
      → 그래도 안전하게 자연어 fallback까지 유지.
    """
    t = strip_code_fence(text).strip()
    low = t.lower()

    # 1) JSON 파싱
    try:
        obj = json.loads(t)
        label = obj.get("label", None)
        if isinstance(label, str):
            label = label.strip().lower()
        if label in ["normal", "disease", "distorted"]:
            return label
    except Exception:
        pass

    # 2) 텍스트 안 첫 {...}
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

    # 3) 자연어 fallback
    if "distorted" in low or "ungradable" in low or "not gradable" in low:
        return "distorted"

    if re.search(r"\bdisease\b", low) or re.search(r"\babnormal\b", low) or re.search(r"\blesion\b", low):
        return "disease"

    if re.search(r"\bnormal\b", low) or re.search(r"\bno abnormal\b", low):
        if not re.search(r"\bdisease\b|\babnormal\b|\blesion\b", low):
            return "normal"

    # SYSTEM_PROMPT_NORMAL에서 진짜로 한 단어만 왔을 때 처리
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
# 2) Soft-prompt + classifier 모듈 (eval용)
# ========================

class SoftPromptVLM(nn.Module):
    """
    eval 시에 사용할 래퍼:
    - base_model: frozen VLM
    - soft_prompt: num_virtual_tokens x hidden_size
    - classifier: hidden_size -> 2 (normal / disease)
    """
    def __init__(self, base_model, num_virtual_tokens: int = 20, keep_input_ids: bool = False):
        super().__init__()
        self.base_model = base_model
        self.num_virtual_tokens = num_virtual_tokens
        self.keep_input_ids = keep_input_ids

        input_embeddings = self.base_model.get_input_embeddings()
        hidden_size = input_embeddings.weight.shape[1]
        embed_dtype = input_embeddings.weight.dtype
        embed_device = input_embeddings.weight.device

        self.soft_prompt = nn.Embedding(
            num_virtual_tokens,
            hidden_size,
            dtype=embed_dtype,
            device=embed_device,
        )

        self.classifier = nn.Linear(
            hidden_size,
            2,
            device=embed_device,
            dtype=embed_dtype,
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

        input_embeds = self.base_model.get_input_embeddings()(input_ids)
        B, L, H = input_embeds.shape
        dev = input_embeds.device

        target_dtype = self.soft_prompt.weight.dtype
        if input_embeds.dtype != target_dtype:
            input_embeds = input_embeds.to(target_dtype)

        virtual_token_ids = torch.arange(
            self.num_virtual_tokens, device=dev
        ).unsqueeze(0).expand(B, -1)
        soft_embeds = self.soft_prompt(virtual_token_ids)

        inputs_embeds = torch.cat([soft_embeds, input_embeds], dim=1)

        if attention_mask is not None:
            soft_mask = torch.ones(
                (B, self.num_virtual_tokens),
                device=dev,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([soft_mask, attention_mask], dim=1)

        if labels is not None:
            pad = torch.full(
                (B, self.num_virtual_tokens),
                fill_value=-100,
                device=dev,
                dtype=labels.dtype,
            )
            labels = torch.cat([pad, labels], dim=1)

        if "input_ids" in kwargs and not self.keep_input_ids:
            kwargs.pop("input_ids")

        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            output_hidden_states=True,
            **kwargs,
        )
        return outputs

    def extract_features(self, outputs: Any) -> torch.Tensor:
        """
        training 코드와 동일한 feature 추출 로직.
        """
        def _has_attr_or_key(obj, name: str):
            if hasattr(obj, name) and getattr(obj, name) is not None:
                return getattr(obj, name)
            if isinstance(obj, dict) and name in obj and obj[name] is not None:
                return obj[name]
            return None

        hs = _has_attr_or_key(outputs, "last_hidden_state")
        if isinstance(hs, torch.Tensor):
            if hs.dim() == 3:
                return hs[:, -1, :]
            elif hs.dim() == 2:
                return hs

        enc = _has_attr_or_key(outputs, "encoder_last_hidden_state")
        if isinstance(enc, torch.Tensor):
            if enc.dim() == 3:
                return enc.mean(dim=1)
            elif enc.dim() == 2:
                return enc

        hidden_states = _has_attr_or_key(outputs, "hidden_states")
        if isinstance(hidden_states, (list, tuple)) and len(hidden_states) > 0:
            hs_last = hidden_states[-1]
            if isinstance(hs_last, torch.Tensor):
                if hs_last.dim() == 3:
                    return hs_last[:, -1, :]
                elif hs_last.dim() == 2:
                    return hs_last

        for key in ["language_model_output", "lm_output", "text_outputs"]:
            nested = _has_attr_or_key(outputs, key)
            if nested is not None:
                try:
                    return self.extract_features(nested)
                except ValueError:
                    pass

        if isinstance(outputs, (list, tuple)):
            candidate = None
            for x in outputs:
                if isinstance(x, torch.Tensor) and x.dim() == 3:
                    if x.size(-1) <= 4096:
                        candidate = x
                        break
            if candidate is not None:
                return candidate[:, -1, :]

        raise ValueError("No usable hidden representations in outputs.")


def load_softprompt_module(backend: str, base_model) -> Tuple[Optional[SoftPromptVLM], Optional[str]]:
    """
    training 코드에서 저장한
    { soft_prompt, classifier, num_virtual_tokens, backend, model_id } ckpt 로드.
    """
    ckpt_path = os.path.join(
        SOFTPROMPT_CKPT_DIR,
        f"{backend}_soft_prompt_tuning_with_system_prompt_900.pt",
    )
    if not os.path.exists(ckpt_path):
        print(f"⚠ No soft-prompt ckpt for {backend}: {ckpt_path}")
        return None, None

    print(f"🔄 Using soft-prompt + classifier checkpoint for {backend}: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    num_virtual_tokens = ckpt.get("num_virtual_tokens", 20)
    keep_input_ids = (backend == "lingshu")

    sp_model = SoftPromptVLM(
        base_model,
        num_virtual_tokens=num_virtual_tokens,
        keep_input_ids=keep_input_ids,
    )

    sp_model.soft_prompt.load_state_dict(ckpt["soft_prompt"])
    sp_model.classifier.load_state_dict(ckpt["classifier"])
    sp_model.to(device)
    sp_model.eval()

    return sp_model, ckpt_path


# ========================
# 3) Backend별 input 빌더
# ========================

def build_inputs_for_backend(
    backend: str,
    model,
    processor,
    pil_img: Image.Image,
    img_path: str,
    prompt_text: str,
    system_prompt: str,
) -> Tuple[Dict[str, torch.Tensor], str]:
    """
    baseline/softprompt 공통 processor input.
    """
    if backend in ["qwen3", "lingshu"]:
        from qwen_vl_utils import process_vision_info

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": prompt_text},
            ]},
        ]
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
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ]},
        ]
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
        prompt_final = f"{img_tok}\n{system_prompt}\n\n{prompt_text}"

        inputs = processor(
            text=[prompt_final],
            images=[pil_img],
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        return inputs, prompt_final

    raise ValueError(f"Unknown backend in build_inputs_for_backend: {backend}")


# ========================
# 4) baseline generate (한 단어 / JSON 혼합)
# ========================

def run_vlm_baseline_gen(
    backend: str,
    model,
    processor,
    pil_img: Image.Image,
    img_path: str,
    prompt_text: str,
    system_prompt: str,
):
    """
    튜닝 안 한 VLM baseline:
    - weak/clean: SYSTEM_PROMPT_NORMAL 로 한 단어 출력 유도
    - strong: SYSTEM_PROMPT_STRONG + JSON (distorted 포함)
    """
    gen_kwargs = GEN_KWARGS_BY_BACKEND.get(backend, GEN_KWARGS_BY_BACKEND["default"])

    inputs, prompt_final = build_inputs_for_backend(
        backend, model, processor, pil_img, img_path, prompt_text, system_prompt
    )

    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[1]
    gen_ids = out[:, input_len:]
    output_only = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    full_decoded = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
    return output_only, full_decoded, prompt_final


# ========================
# 5) 데이터 로드
# ========================

meta_csv_path = os.path.expanduser("~/Scratch/vlm_prompt_dataset/vlm_pg_subset_100.csv")
base_out_dir = os.path.expanduser("~/Scratch/vlm_prompt_dataset")
os.makedirs(base_out_dir, exist_ok=True)

df_base = pd.read_csv(meta_csv_path)

baseline_csv = os.path.join(base_out_dir, "baseline_performance.csv")
if not os.path.exists(baseline_csv):
    raise FileNotFoundError(f"baseline_performance.csv not found: {baseline_csv}")

baseline_df = pd.read_csv(baseline_csv)

baseline_acc = {}
tmp = baseline_df[baseline_df["metric"].astype(str).str.lower() == "accuracy"].copy()
for _, r in tmp.iterrows():
    m = str(r["model"]).strip()
    mod = str(r["modality"]).strip().lower()
    try:
        v = float(r["value"])
    except Exception:
        v = float("nan")
    baseline_acc[(m, mod)] = v

df_base["filepath"] = (
    df_base["filepath"]
    .astype(str)
    .str.replace(
        r"C:\Users\hanna\Lectures\Research_Project\Codes\Dataset\vlm_prompt_dataset",
        base_out_dir,
        regex=False,
    )
    .str.replace("\\", "/", regex=False)
)


# ========================
# 6) 실행 + metrics 계산
# ========================

metrics_rows = []

for BACKEND in BACKENDS:
    print("\n==============================")
    print(f"🚀 RUNNING BACKEND: {BACKEND}")
    print("==============================")

    model_id = MODEL_ID_BY_BACKEND[BACKEND]
    try:
        model, processor = load_backend(BACKEND, model_id)
    except Exception as e:
        print(f"❌ FAILED to load backend={BACKEND}, model_id={model_id}")
        print(f"   {type(e).__name__}: {e}")
        cleanup()
        continue

    # soft-prompt + classifier 로드
    sp_model, ckpt_path = load_softprompt_module(BACKEND, model)
    has_softprompt = sp_model is not None
    if has_softprompt:
        print(f"✅ Soft-prompt classifier variant enabled for {BACKEND}")
    else:
        print(f"⚠ Soft-prompt classifier variant DISABLED for {BACKEND}, baseline only.")

    df = df_base.copy()

    df["model_prompt_baseline"] = None
    df["model_raw_output_baseline"] = None
    df["model_pred_baseline"] = None
    df["pred_binary_baseline"] = None

    df["model_prompt_softprompt"] = None
    df["model_raw_output_softprompt"] = None  # 여기선 classifier만 사용하므로 optional
    df["model_pred_softprompt"] = None
    df["pred_binary_softprompt"] = None

    n_ok_base, n_err_base = 0, 0
    n_ok_sp, n_err_sp = 0, 0

    for i, row in df.iterrows():
        img_path = row.get("filepath", None)
        if not isinstance(img_path, str) or not os.path.exists(img_path):
            continue

        try:
            with Image.open(img_path) as im:
                pil_img = im.convert("RGB")

            ds = normalize_dataset_name(row.get("dataset", ""))

            prompt = PROMPT_BY_DATASET.get(
                ds,
                "This is a medical image.\nQuestion: Does this image show normal anatomy or signs of disease?\n\n",
            )

            severity = str(row.get("severity", "")).lower()
            if severity == "strong":
                system_prompt = SYSTEM_PROMPT_STRONG
            else:
                system_prompt = SYSTEM_PROMPT_NORMAL

            # ----- baseline (generate + 한 단어/JSON) -----
            try:
                out_text_base, full_dec_base, prompt_final_base = run_vlm_baseline_gen(
                    BACKEND, model, processor, pil_img, img_path, prompt, system_prompt
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

            # ----- soft-prompt + classifier -----
            # strong 은 classifier가 distorted를 못 내니까, weak/clean만 평가하는게 자연스러움
            if has_softprompt and severity != "strong":
                try:
                    inputs, prompt_final_sp = build_inputs_for_backend(
                        BACKEND, model, processor, pil_img, img_path, prompt, SYSTEM_PROMPT_NORMAL
                    )
                    # sp_model 은 base_model wrapper라 model 대신 sp_model 사용
                    with torch.inference_mode():
                        outputs = sp_model(**inputs)
                        h = sp_model.extract_features(outputs)
                        logits = sp_model.classifier(h.float())
                        probs = torch.softmax(logits, dim=-1)
                        pred_cls = torch.argmax(probs, dim=-1).item()

                    if pred_cls == 0:
                        pred_sp = "normal"
                    else:
                        pred_sp = "disease"

                    pred_bin_sp = int(pred_cls)

                    df.at[i, "model_prompt_softprompt"] = prompt_final_sp
                    df.at[i, "model_raw_output_softprompt"] = f"__CLS__:{pred_sp}"
                    df.at[i, "model_pred_softprompt"] = pred_sp
                    df.at[i, "pred_binary_softprompt"] = pred_bin_sp
                    n_ok_sp += 1
                except Exception as e:
                    df.at[i, "model_raw_output_softprompt"] = f"__ERROR__ {type(e).__name__}: {e}"
                    df.at[i, "model_pred_softprompt"] = None
                    df.at[i, "pred_binary_softprompt"] = None
                    n_err_sp += 1

        except Exception as e:
            df.at[i, "model_raw_output_baseline"] = f"__ERROR__ {type(e).__name__}: {e}"
            df.at[i, "model_pred_baseline"] = None
            df.at[i, "pred_binary_baseline"] = None

            if has_softprompt:
                df.at[i, "model_raw_output_softprompt"] = f"__ERROR__ {type(e).__name__}: {e}"
                df.at[i, "model_pred_softprompt"] = None
                df.at[i, "pred_binary_softprompt"] = None

    out_csv = os.path.join(base_out_dir, f"{BACKEND}_{RUN_NAME}.csv")
    df.to_csv(out_csv, index=False)

    # ========================
    # Metrics 계산
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

    # 디버그: softprompt classifier 결과 확인
    if BACKEND in ["internvl", "lingshu"] and has_softprompt:
        debug_cols = [
            "dataset", "severity", "severity_norm",
            "model_raw_output_softprompt",
            "model_pred_softprompt",
            "pred_binary_softprompt",
            "gt_binary",
        ]
        weak_mask = df["severity_norm"] == "weak"

        print(f"\n[DEBUG] --- {BACKEND} softprompt CLS (weak only, head 20) ---")
        print(
            df.loc[weak_mask, debug_cols]
              .head(20)
              .to_string(index=False)
        )

        print(f"\n[DEBUG] value_counts(model_pred_softprompt) for weak ({BACKEND}):")
        print(
            df.loc[weak_mask, "model_pred_softprompt"]
              .value_counts(dropna=False)
        )

        print(f"\n[DEBUG] value_counts(pred_binary_softprompt) for weak ({BACKEND}):")
        print(
            df.loc[weak_mask, "pred_binary_softprompt"]
              .value_counts(dropna=False)
        )

    variants = ["baseline"]
    if has_softprompt:
        variants.append("softprompt")

    for variant in variants:
        pred_col = f"pred_binary_{variant}"
        label_col = "gt_binary"
        pred_label_col = f"model_pred_{variant}"

        # 이름만 바꿈: baseline_gen vs softprompt_cls
        variant_name = "baseline_gen" if variant == "baseline" else "softprompt_cls"

        # ---------- WEAK ----------
        weak_df = df[df["severity_norm"] == "weak"].copy()

        per_mod_eval_n = {}
        per_mod_base_acc = {}
        per_mod_acc = {}

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

            print(
                f"  modality={modality_key:8s} | "
                f"n_total={n_total:3d}, n_eval={n_eval:3d} | "
                f"acc={acc:.4f} | baseline={base_acc:.4f} | "
                f"Δ={(perf_pct_change * 100):+.2f}%"
                if n_eval > 0 and pd.notna(base_acc) and pd.notna(acc)
                else f"  modality={modality_key:8s} | n_total={n_total:3d}, n_eval={n_eval:3d} | acc=NaN"
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
            })

            per_mod_eval_n[modality_key] = n_eval
            per_mod_base_acc[modality_key] = base_acc
            per_mod_acc[modality_key] = acc

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

        if n_eval_all > 0 and pd.notna(base_acc_all) and pd.notna(acc_all):
            print(
                f"  >>> OVERALL (weak) | n_total={n_total_all}, n_eval={n_eval_all} | "
                f"acc={acc_all:.4f} | baseline={base_acc_all:.4f} | "
                f"Δ={(perf_pct_change_all * 100):+.2f}%"
            )
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
        })

        # ---------- STRONG ----------
        strong_df = df[df["severity_norm"] == "strong"].copy()

        print(f"\n--- [DISTORTED RATE] backend={BACKEND}, variant={variant_name}, severity=strong ---")
        for modality_key, g0 in list(strong_df.groupby("modality")) + [("__overall__", strong_df)]:
            g = g0.copy()
            n_total = int(len(g))

            if n_total > 0:
                pred_dist = (g[pred_label_col].astype(str).str.lower() == "distorted")
                distorted_rate = float(pred_dist.mean())
            else:
                distorted_rate = float("nan")

            if n_total > 0 and pd.notna(distorted_rate):
                print(
                    f"  modality={modality_key:8s} | n_total={n_total:3d} | distorted_rate={distorted_rate:.4f}"
                )
            else:
                print(
                    f"  modality={modality_key:8s} | n_total={n_total:3d} | distorted_rate=NaN"
                )

            metrics_rows.append({
                "run_name": RUN_NAME,
                "model": BACKEND,
                "variant": variant_name,
                "severity": "strong",
                "modality": modality_key,
                "n_total": n_total,
                "n_eval": 0,
                "accuracy": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "baseline_accuracy": float("nan"),
                "performance_pct_change": float("nan"),
                "distorted_rate": distorted_rate,
            })

    print(f"✅ DONE: {BACKEND}")
    print(f"   baseline_gen: ok={n_ok_base}, errors={n_err_base}")
    if has_softprompt:
        print(f"   softprompt_cls: ok={n_ok_sp}, errors={n_err_sp}")
    print(f"   CSV : {out_csv}")

    del model, processor
    if sp_model is not None:
        del sp_model
    cleanup()


# ========================
# 7) metrics 전체 CSV 저장
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
    ]
    ordered_cols = [c for c in preferred_order if c in new_df.columns]
    remaining_cols = [c for c in new_df.columns if c not in ordered_cols]
    new_df = new_df[ordered_cols + remaining_cols]

    out_metrics_csv = os.path.join(base_out_dir, "all_metrics_json_behaviour_shift.csv")

    if os.path.exists(out_metrics_csv):
        old_df = pd.read_csv(out_metrics_csv)
        all_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        all_df = new_df

    all_df.to_csv(out_metrics_csv, index=False)
    print(f"📄 ALL METRICS CSV (updated): {out_metrics_csv}")
else:
    print("⚠️ No metrics collected.")
