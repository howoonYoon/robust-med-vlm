import os
import gc
import json
import re
import torch
import pandas as pd
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score


# ========================
# 0. 실행 설정
# ========================
SAVE_FORMAT = "csv"  # "csv" / "json" / "both" 확장 가능
RUN_NAME = "clean_baseline"

BACKENDS = [
    "qwen3",
    "medgemma",
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

ONE_WORD_SYSTEM_PROMPT = (
    "You are a medical image classifier.\n"
    "Respond using ONLY valid JSON.\n"
    "Output exactly one JSON object with one key called \"label\".\n"
    "The value of \"label\" MUST be either \"normal\" or \"disease\".\n"
    "Do NOT include any explanation, text, or formatting outside the JSON."
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

GEN_KWARGS_BY_BACKEND = {
    "default":   dict(max_new_tokens=24, do_sample=False),
}

MODEL_ID_BY_BACKEND = {
    "qwen3":     "Qwen/Qwen3-VL-2B-Instruct",
    "medgemma":  "google/medgemma-4b-it",
    "internvl":  "OpenGVLab/InternVL3_5-2B-HF",
    "lingshu":   "lingshu-medical-mllm/Lingshu-7B",
}

# ========================
# 유틸
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

def label_to_binary(label):
    if label == "disease":
        return 1
    if label == "normal":
        return 0
    return None

def strip_code_fence(t: str) -> str:
    t = (t or "").strip()
    if t.startswith("```"):
        lines = t.splitlines()
        # ```json / ```javascript 같은 언어 라벨 제거
        if len(lines) >= 2 and lines[0].strip().startswith("```"):
            lines = lines[1:]
        # 마지막 ``` 제거
        if len(lines) >= 1 and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t

def extract_label(text: str):
    """
    JSON만 허용하되, 모델이 문장을 붙여도
    텍스트에서 첫 JSON 객체를 뽑아서 파싱한다.
    """
    t = strip_code_fence(text).strip()
    low = t.lower()

    # 1) 통째로 JSON 파싱
    try:
        obj = json.loads(t)
        label = obj.get("label", None)
        if isinstance(label, str):
            label = label.strip().lower()
        if label in ["normal", "disease", "distorted"]:
            return label
    except Exception:
        pass

    # 2) 텍스트 안 첫 {...} 파싱
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

    # 3) 자연어 fallback (보수적으로)
    if "distorted" in low or "ungradable" in low or "not gradable" in low:
        return "distorted"

    # disease 먼저(조금이라도 있으면 disease)
    if re.search(r"\bdisease\b", low) or re.search(r"\babnormal\b", low) or re.search(r"\blesion\b", low):
        return "disease"

    # 마지막에 normal
    if re.search(r"\bnormal\b", low) or re.search(r"\bno abnormal\b", low):
        # 단, disease/abnormal/lesion이 아예 없을 때만
        if not re.search(r"\bdisease\b|\babnormal\b|\blesion\b", low):
            return "normal"

    return None

def get_image_token(processor):
    """
    chat_template 사용 불가한 경우를 대비해 이미지 토큰 후보를 최대한 탐색.
    """
    for obj in [processor, getattr(processor, "tokenizer", None), getattr(processor, "processor", None)]:
        if obj is None:
            continue
        for attr in ["image_token", "image_placeholder", "img_token", "image_token_str"]:
            if hasattr(obj, attr):
                v = getattr(obj, attr)
                if isinstance(v, str) and v:
                    return v
    # 많이 쓰는 후보들
    return "<image>"

def can_chat_template(processor) -> bool:
    return hasattr(processor, "apply_chat_template")

def safe_apply_chat_template(processor, messages):
    """
    apply_chat_template이 모델/processor마다 깨지는 경우가 있어서 try/except로 보호.
    """
    if not can_chat_template(processor):
        return None
    try:
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return None

# ========================
# 1) BACKEND 로더
# ========================

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

# ========================
# 2) 실행 (핵심)
# ========================

def run_vlm(backend, model, processor, pil_img, img_path, prompt_text):
    gen_kwargs = GEN_KWARGS_BY_BACKEND.get(backend, GEN_KWARGS_BY_BACKEND["default"])

    # ------------------------
    # Qwen3 / Lingshu: messages에 image는 "경로"로 넣는 게 제일 안전
    # ------------------------
    if backend in ["qwen3", "lingshu"]:
        from qwen_vl_utils import process_vision_info

        messages = [
            {"role": "system", "content": [{"type": "text", "text": ONE_WORD_SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": img_path},  # ✅ path 기반 (PIL보다 안전)
                {"type": "text", "text": prompt_text},
            ]},
        ]

        prompt_final = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(messages)

        inputs = processor(
            text=[prompt_final],
            images=imgs,
            videos=vids,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        gen_ids = out[:, input_len:]
        output_only = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        full_decoded = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        return output_only, full_decoded, prompt_final

    # ------------------------
    # MedGemma / InternVL:
    # 1) 가능하면 chat_template + {"type":"image"} placeholder
    # 2) 안 되면 fallback: 이미지 토큰 1개 강제 삽입
    # ------------------------
    if backend in ["medgemma", "internvl"]:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": ONE_WORD_SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image"},  # ✅ placeholder
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

            with torch.inference_mode():
                out = model.generate(**inputs, **gen_kwargs)

            input_len = inputs["input_ids"].shape[1]
            gen_ids = out[:, input_len:]
            output_only = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
            full_decoded = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
            return output_only, full_decoded, prompt_final

        # ---- fallback (chat_template 깨질 때) ----
        img_tok = get_image_token(processor)
        prompt_final = f"{img_tok}\n{ONE_WORD_SYSTEM_PROMPT}\n\n{prompt_text}"

        inputs = processor(
            text=[prompt_final],
            images=[pil_img],
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        gen_ids = out[:, input_len:]
        output_only = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        full_decoded = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        return output_only, full_decoded, prompt_final

    raise ValueError(f"Unknown backend in run_vlm: {backend}")

# ========================
# 3) 데이터 로드
# ========================


data_root = os.path.expanduser("~/Scratch/vlm_prompt_dataset")   # ✅ 이미지 실제 위치
result_root = os.path.expanduser("~/Scratch/prompt_test_result") # ✅ 결과 저장 위치
os.makedirs(result_root, exist_ok=True)

meta_csv_path = os.path.join(data_root, "clean_test_image_300.csv")
df_base = pd.read_csv(meta_csv_path)

# windows path -> myriad path
df_base["filepath"] = (
    df_base["filepath"]
    .astype(str)
    .str.replace(
        r"C:\Users\hanna\Lectures\Research_Project\Codes\Dataset\vlm_prompt_dataset",
        data_root,      # ✅ 여기! data_root로
        regex=False,
    )
    .str.replace("\\", "/", regex=False)
)


# ========================
# 4) 실행
# ========================

baseline_rows = []  # ✅ 모델별/모달리티별 clean baseline 성능 저장용

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

    df = df_base.copy()
    df["model_prompt"] = None
    df["model_raw_output"] = None
    df["model_pred"] = None
    df["pred_binary"] = None
    df["gt_binary"] = df["binarylabel"]   # ✅ GT
    df["is_correct"] = None               # ✅ 맞/틀

    n_ok, n_missing, n_err = 0, 0, 0

    for i, row in df.iterrows():
        img_path = row.get("filepath", None)
        if not isinstance(img_path, str) or not os.path.exists(img_path):
            n_missing += 1
            continue

        try:
            # PIL은 llava/medgemma/internvl에서 안전하게 필요하니까 항상 만들어두자
            with Image.open(img_path) as im:
                pil_img = im.convert("RGB")

            ds = normalize_dataset_name(row.get("dataset", ""))
            prompt = PROMPT_BY_DATASET.get(ds)
            if prompt is None:
                prompt = (
                    "This is a medical image.\n"
                    "Question: Does this medical image show normal anatomy or signs of disease?\n\n"
                )

            output_text, full_decoded, prompt_final = run_vlm(
                BACKEND, model, processor, pil_img, img_path, prompt
            )

            pred = extract_label(output_text)
            pred_bin = label_to_binary(pred)

            gt = row.get("binarylabel", None)
            try:
                gt = int(gt) if pd.notna(gt) else None
            except Exception:
                gt = None

            df.at[i, "gt_binary"] = gt

            if (gt in [0, 1]) and (pred_bin in [0, 1]):
                df.at[i, "is_correct"] = int(pred_bin == gt)
            else:
                df.at[i, "is_correct"] = None

            df.at[i, "model_prompt"] = prompt_final
            df.at[i, "model_raw_output"] = output_text
            df.at[i, "model_pred"] = pred
            df.at[i, "pred_binary"] = pred_bin

            n_ok += 1

        except Exception as e:
            n_err += 1
            df.at[i, "model_raw_output"] = f"__ERROR__ {type(e).__name__}: {e}"
            df.at[i, "model_pred"] = None
            df.at[i, "pred_binary"] = None

    out_csv = os.path.join(result_root, f"{BACKEND}_{RUN_NAME}.csv")
    df.to_csv(out_csv, index=False)

    print(f"✅ DONE: {BACKEND}")
        # ========================
    # 전체 예측 & GT 출력
    # ========================
    display_cols = [
        "dataset",
        "filepath",
        "model_pred",
        "pred_binary",
        "gt_binary",
        "is_correct",
    ]

    safe_cols = [c for c in display_cols if c in df.columns]
    view_df = df[safe_cols].copy()

    print("\n🧾 FULL PREDICTION RESULTS")
    print(view_df.to_string(index=False))

    print(f"   ok={n_ok}, missing_file={n_missing}, errors={n_err}")
    print(f"   CSV : {out_csv}")

    # ========================
    # 예측 개수 점검 로그 출력
    # ========================
    total_samples = len(df)
    predicted_samples = df["model_pred"].notna().sum()
    skipped_samples = total_samples - predicted_samples

    print(f"\n📊 Prediction coverage summary for backend = {BACKEND}")
    print(f"   Total samples        : {total_samples}")
    print(f"   Predicted samples    : {predicted_samples}")
    print(f"   Skipped / Empty pred : {skipped_samples}")

    # 모달리티별도 확인
    for modality_key, g in df.groupby(df["dataset"].astype(str)):
        total = len(g)
        pred = g["model_pred"].notna().sum()
        print(f"   [{normalize_dataset_name(modality_key)}] "
              f"total={total}, predicted={pred}, skipped={total - pred}")



    # ========================
    # 5) baseline_performance.csv 용 accuracy 계산 (모델별/모달리티별)
    # ========================
    eval_df = df.copy()

    # GT/Pred 둘 다 0/1인 행만 평가
    eval_df = eval_df[
        eval_df["gt_binary"].isin([0, 1]) &
        eval_df["pred_binary"].isin([0, 1])
    ].copy()

    for ds_name, g in eval_df.groupby("dataset", dropna=False):
        modality = normalize_dataset_name(ds_name)

        y_true = g["gt_binary"].astype(int).values
        y_pred = g["pred_binary"].astype(int).values

        acc = float((y_true == y_pred).mean()) if len(g) > 0 else float("nan")
        prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec  = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1   = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

        baseline_rows.extend([
            {
                "model": BACKEND,
                "modality": modality,
                "metric": "accuracy",
                "value": acc,
                "n": len(g),
                "run_name": RUN_NAME,
            },
            {
                "model": BACKEND,
                "modality": modality,
                "metric": "precision",
                "value": prec,
                "n": len(g),
                "run_name": RUN_NAME,
            },
            {
                "model": BACKEND,
                "modality": modality,
                "metric": "recall",
                "value": rec,
                "n": len(g),
                "run_name": RUN_NAME,
            },
            {
                "model": BACKEND,
                "modality": modality,
                "metric": "f1",
                "value": f1,
                "n": len(g),
                "run_name": RUN_NAME,
            },
        ])


    # 콘솔 출력(선택)
    overall_n = len(eval_df)
    overall_acc = float((eval_df["gt_binary"] == eval_df["pred_binary"]).mean()) if overall_n > 0 else float("nan")
    print(f"📊 Accuracy (overall) = {overall_acc:.4f}  (n={overall_n})")

    del model, processor
    cleanup()

# ========================
# 6) 모든 모델 baseline을 하나의 CSV로 저장
# ========================
if len(baseline_rows) > 0:
    baseline_df = pd.DataFrame(baseline_rows)

    # 예시 포맷처럼 딱 4컬럼만 원하면 아래 줄로 축소 가능
    # baseline_df = baseline_df[["model", "modality", "metric", "value"]]

    out_baseline_csv = os.path.join(result_root, "baseline_performance.csv")
    baseline_df.to_csv(out_baseline_csv, index=False)
    print(f"📄 BASELINE CSV : {out_baseline_csv}")
else:
    print("⚠️ No baseline rows collected. baseline_performance.csv not written.")

    
