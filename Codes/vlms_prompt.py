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
RUN_NAME = "disease-cue-reason-1-prompts"

BACKENDS = [
    "qwen3",
    "medgemma",
    "internvl",
    "lingshu",
]

PROMPT_BY_DATASET = {
    "mri": (
        "This is a brain MRI scan.\n"
        "Focus on brain tissue structure, symmetry, and signal characteristics.\n"
        "Determine whether there are true abnormalities such as tumors, edema, "
        "mass effect, or focal signal changes.\n\n"
        "Question: Does this image show normal anatomy or signs of disease?\n\n"
    ),
    "oct": (
        "This is a retinal OCT scan.\n"
        "Focus on retinal layer integrity and the foveal contour.\n"
        "Determine whether there are pathological features such as drusen-like "
        "elevations, intraretinal or subretinal fluid, or abnormal layer "
        "thickening or disruption.\n\n"
        "Question: Does this image show normal anatomy or signs of disease?\n\n"
    ),
    "xray": (
        "This is a chest X-ray image.\n"
        "Focus on the lung fields, heart borders, and costophrenic angles.\n"
        "Determine whether there are pneumonia-related abnormalities such as "
        "focal opacity, consolidation, or interstitial infiltrates.\n\n"
        "Question: Does this image show normal anatomy or signs of disease?\n\n"
    ),
    "fundus": (
        "This is a retinal fundus photograph.\n"
        "Focus on the retinal vessels, macula, and optic disc.\n"
        "Determine whether there are disease-related findings such as "
        "microaneurysms, hemorrhages, exudates, or vessel irregularities.\n\n"
        "Question: Does this image show normal anatomy or signs of disease?\n\n"
    ),
}

# SYSTEM_PROMPT_NORMAL = (
#     "You are a medical image classifier.\n"
#     "Respond using ONLY valid JSON.\n"
#     "Output exactly one JSON object with one key called \"label\".\n"
#     "The value of \"label\" MUST be either \"normal\" or \"disease\".\n"
#     "Do NOT include any explanation, text, or formatting outside the JSON."
# )

# SYSTEM_PROMPT_STRONG = (
#     "You are a medical image classifier.\n"
#     "Respond using ONLY valid JSON.\n"
#     "Output exactly one JSON object with one key called \"label\".\n"
#     "The value of \"label\" MUST be one of: \"normal\", \"disease\", or \"distorted\".\n"
 
#     "Use \"distorted\" only if the image quality is too poor to make a reliable decision.\n"
#     "Do NOT include any explanation, text, or formatting outside the JSON."
# )


SYSTEM_PROMPT_NORMAL = (
    "You are a medical image classifier.\n"
    "Respond using ONLY valid JSON.\n"
    "Output exactly one JSON object with TWO keys: \"label\" and \"reason\".\n"
    "The value of \"label\" MUST be either \"normal\" or \"disease\".\n"
    "For ANY label, you MUST explain the reasoning behind the decision.\n"
    "The value of \"reason\" MUST be ONE short sentence (max 20 words)\n"
    "describing the key visual evidence supporting the label.\n"
    "Keep the reason strictly under 20 words.\n"
    "Do NOT include any text or formatting outside the JSON."
)


SYSTEM_PROMPT_STRONG = (
    "You are a medical image classifier.\n"
    "Respond using ONLY valid JSON.\n"
    "Output exactly one JSON object with TWO keys: \"label\" and \"reason\".\n"
    "The value of \"label\" MUST be one of: \"normal\", \"disease\", or \"distorted\".\n"
    "First assess image quality and artefacts.\n"
    "Use \"distorted\" ONLY if image quality or artefacts make diagnosis unreliable.\n"
    "For ANY label, you MUST explain the reasoning behind the decision.\n"
    "The value of \"reason\" MUST be ONE short sentence (max 20 words)\n"
    "describing the key visual evidence supporting the label.\n"
    "Keep the reason strictly under 20 words.\n"
    "Do NOT include any text or formatting outside the JSON."
)




device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

GEN_KWARGS_BY_BACKEND = {
    "default":   dict(max_new_tokens=50, do_sample=False)
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
    텍스트에서 VLM의 라벨을 robust하게 추출한다.
    1) JSON 전체 파싱
    2) 텍스트 내 첫 번째 {...} JSON 파싱
    3) 자연어 fallback (distorted → normal(negs) → disease → normal 순)
    """
    t = strip_code_fence(text or "").strip()
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

    # 3) 자연어 fallback

    # 3-1) distorted / ungradable 계열 먼저
    if re.search(
        r"\b(distorted|ungradable|not gradable|non[- ]diagnostic|"
        r"cannot be assessed|too (noisy|blurred|blurry)|severely degraded)\b",
        low,
    ):
        return "distorted"

    # 3-2) 'no evidence of disease/abnormality/lesion...' + normal/unremarkable 표현
    neg_disease_pattern = re.compile(
        r"no (evidence of )?(significant )?(acute )?"
        r"(disease|abnormalit(?:y|ies)|lesion|pneumonia|pathology|finding[s]?)"
    )
    neg_env_pattern = re.compile(
        r"(normal (study|scan|examination|chest x[- ]ray|x[- ]ray|mri|oct|image))"
        r"|(\bunremarkable( study| scan| appearance)?\b)"
        r"|(\bwithin normal limits\b|\bwnl\b)"
    )
    hedge_pattern = re.compile(
        r"cannot (exclude|rule out)|suspicious for|concern(ing)? for|suggestive of"
    )

    # hedge 표현이 없으면서 'no disease/abnormality' 또는 'normal study' 류면 normal
    if (neg_disease_pattern.search(low) or neg_env_pattern.search(low)) and not hedge_pattern.search(low):
        return "normal"

    # 3-3) disease 쪽 키워드
    disease_keywords = [
        "disease", "lesion", "abnormal", "opacity", "consolidation",
        "nodule", "mass", "effusion", "infiltrate", "infiltration",
        "edema", "oedema", "hemorrhage", "haemorrhage",
        "stroke", "infarct", "ischemia", "ischaemia",
        "plaque", "drusen", "fluid", "thickening",
        "collapse", "atelectasis", "pneumonia", "tumour", "tumor",
        "metastasis", "infection", "pathology"
    ]

    for kw in disease_keywords:
        if kw in low:
            # "no evidence of <kw>"는 위에서 normal로 처리했으니 여기선 제외
            if re.search(rf"no (evidence of )?(significant )?(acute )?{kw}", low):
                continue
            return "disease"

    # 3-4) 남은 normal 패턴들
    if re.search(r"\bno (significant )?abnormalit(?:y|ies)\b", low) and not hedge_pattern.search(low):
        return "normal"

    # disease/abnormal/lesion 언급 없이 그냥 normal만 있을 때
    if re.search(r"\bnormal\b", low):
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

def run_vlm(backend, model, processor, pil_img, img_path, prompt_text, system_prompt):
    gen_kwargs = GEN_KWARGS_BY_BACKEND.get(backend, GEN_KWARGS_BY_BACKEND["default"])

    # ------------------------
    # Qwen3 / Lingshu: messages에 image는 "경로"로 넣는 게 제일 안전
    # ------------------------
    if backend in ["qwen3", "lingshu"]:
        from qwen_vl_utils import process_vision_info

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "image", "image": img_path},
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
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
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
        prompt_final = f"{img_tok}\n{system_prompt}\n\n{prompt_text}"

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

data_root   = os.path.expanduser("~/Scratch/vlm_prompt_dataset")   # ✅ 이미지 위치
result_root = os.path.expanduser("~/Scratch/prompt_test_result")   # ✅ 결과 저장 위치
os.makedirs(result_root, exist_ok=True)

meta_csv_path = os.path.join(data_root, "weak_strong_test_image_600.csv")
df_base = pd.read_csv(meta_csv_path)

# baseline_performance.csv 로드 (clean baseline)
baseline_csv = os.path.join(result_root, "baseline_performance.csv")  # ✅ baseline은 결과 폴더에 있음
if not os.path.exists(baseline_csv):
    raise FileNotFoundError(f"baseline_performance.csv not found: {baseline_csv}")

baseline_df = pd.read_csv(baseline_csv)

# baseline accuracy만 사용
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


# windows path -> myriad path
df_base["filepath"] = (
    df_base["filepath"]
    .astype(str)
    .str.replace(
        r"C:\Users\hanna\Lectures\Research_Project\Codes\Dataset\vlm_prompt_dataset",
        data_root,   # ✅ 여기 data_root로!
        regex=False,
    )
    .str.replace("\\", "/", regex=False)
)

# ========================
# 4) 실행
# ========================

metrics_rows = []  # ✅ 모든 모델/모달리티/시버리티 metrics 모아서 한 파일로 저장

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

    # ✅ 여기서 미리 만들어 두기
    df["modality"] = df["dataset"].apply(normalize_dataset_name).astype(str).str.lower()
    df["severity_norm"] = df["severity"].astype(str).str.lower()

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

            # 🔴 severity가 strong일 때만 distorted 규칙 추가
            severity = str(row.get("severity", "")).lower()

            # if severity == "strong":
            #     prompt = prompt + "\nIf the image is severely distorted by artefacts, reply with {\"label\":\"distorted\"}.\n"


            if severity == "strong":
                system_prompt = SYSTEM_PROMPT_STRONG
            else:
                system_prompt = SYSTEM_PROMPT_NORMAL
            
            
            output_text, full_decoded, prompt_final = run_vlm(
                BACKEND, model, processor, pil_img, img_path, prompt, system_prompt
            )

            pred = extract_label(output_text)
            pred_bin = label_to_binary(pred)

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

    # ========================
    # 예측 개수 점검 로그 출력
    # ========================
    total_samples = len(df)
    predicted_samples = df["model_pred"].notna().sum()
    skipped_samples = total_samples - predicted_samples

    print(f"\n📊 Prediction coverage summary for backend = {BACKEND}")
    for sev in ["weak", "strong"]:
        sub = df[df["severity_norm"] == sev]
        total = len(sub)
        pred = sub["model_pred"].notna().sum()
        print(f"   [{sev}] total={total}, predicted={pred}, skipped={total-pred}")


    # ========================
    # 5) Metrics 계산
    # - weak: binarylabel 기반 acc/prec/rec/f1 + performance_pct_change(accuracy 기준)
    # - strong: distorted_rate (=distorted_rate = 모델이 strong set에서 distorted를 선택한 비율)
    # ========================

    # ---- helper: safe binarylabel ----
    def _to_int01(x):
        try:
            if pd.isna(x):
                return None
            xi = int(x)
            return xi if xi in [0, 1] else None
        except Exception:
            return None

    # df에 GT 추가
    if "binarylabel" in df.columns:
        df["gt_binary"] = df["binarylabel"].apply(_to_int01)
    else:
        df["gt_binary"] = None

    # ---------- WEAK ----------
    weak_df = df[df["severity_norm"] == "weak"].copy()

    # 모달리티별 baseline 가중평균을 만들기 위해
    # modality별 (n_eval, acc, baseline_acc) 모아두기
    per_mod_eval_n = {}
    per_mod_base_acc = {}
    per_mod_acc = {}

    # 1) modality별 metrics (baseline 비교 포함)
    for modality_key, g0 in weak_df.groupby("modality"):
        g = g0.copy()

        eval_g = g[
            g["gt_binary"].isin([0, 1]) &
            g["pred_binary"].isin([0, 1])
        ].copy()

        n_total = int(len(g))
        n_eval = int(len(eval_g))

        if n_eval > 0:
            y_true = eval_g["gt_binary"].astype(int).values
            y_pred = eval_g["pred_binary"].astype(int).values

            acc = float((y_true == y_pred).mean())
            prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
            rec  = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
            f1   = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
        else:
            acc = prec = rec = f1 = float("nan")

        base_acc = baseline_acc.get((BACKEND, str(modality_key).lower()), float("nan"))
        perf_pct_change = float("nan")
        if pd.notna(base_acc) and base_acc != 0 and pd.notna(acc):
            perf_pct_change = (acc - base_acc) / base_acc

        metrics_rows.append({
            "run_name": RUN_NAME,
            "model": BACKEND,
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

    # 2) 모델별 전체(overall) metrics + overall baseline(가중평균) + overall pct change
    # overall accuracy는 전체 eval sample 기준으로 계산
    eval_all = weak_df[ 
        weak_df["gt_binary"].isin([0, 1]) &
        weak_df["pred_binary"].isin([0, 1])
    ].copy()

    n_total_all = int(len(weak_df))
    n_eval_all = int(len(eval_all))

    if n_eval_all > 0:
        y_true = eval_all["gt_binary"].astype(int).values
        y_pred = eval_all["pred_binary"].astype(int).values

        acc_all = float((y_true == y_pred).mean())
        prec_all = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        rec_all  = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        f1_all   = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    else:
        acc_all = prec_all = rec_all = f1_all = float("nan")

    # baseline overall = modality baseline의 n_eval 가중평균
    base_num = 0.0
    base_den = 0.0
    for mod, n_eval in per_mod_eval_n.items():
        b = per_mod_base_acc.get(mod, float("nan"))
        if n_eval > 0 and pd.notna(b):
            base_num += float(n_eval) * float(b)
            base_den += float(n_eval)

    base_acc_all = (base_num / base_den) if base_den > 0 else float("nan")

    perf_pct_change_all = float("nan")
    if pd.notna(base_acc_all) and base_acc_all != 0 and pd.notna(acc_all):
        perf_pct_change_all = (acc_all - base_acc_all) / base_acc_all

    metrics_rows.append({
        "run_name": RUN_NAME,
        "model": BACKEND,
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

    for modality_key, g0 in list(strong_df.groupby("modality")) + [("__overall__", strong_df)]:
        g = g0.copy()

        n_total = int(len(g))                     # strong 전체 샘플 수
        has_pred = g["model_pred"].notna()       # 예측이 실제로 있는 샘플만
        n_eval = int(has_pred.sum())             # distorted_rate 분모

        if n_eval > 0:
            pred_dist = (
                g.loc[has_pred, "model_pred"]
                .astype(str)
                .str.lower() == "distorted"
            )
            distorted_rate = float(pred_dist.mean())
        else:
            distorted_rate = float("nan")

        metrics_rows.append({
            "run_name": RUN_NAME,
            "model": BACKEND,
            "severity": "strong",
            "modality": modality_key,
            "n_total": n_total,
            "n_eval": n_eval,          # 0이 아니라 실제 eval 샘플 수
            "accuracy": float("nan"),  # 원하면 여기 = distorted_rate 로 바꿔도 됨
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "baseline_accuracy": float("nan"),
            "performance_pct_change": float("nan"),
            "distorted_rate": distorted_rate,
        })

    # ========================

    print(f"✅ DONE: {BACKEND}")
    print(f"   ok={n_ok}, missing_file={n_missing}, errors={n_err}")
    print(f"   CSV : {out_csv}")

    del model, processor
    cleanup()

# ========================
# 6) 모든 모델 / 모든 실험 metrics 하나의 CSV에 누적 저장
# ========================
if len(metrics_rows) > 0:
    new_df = pd.DataFrame(metrics_rows)

    # run_name 맨 앞으로 정렬
    preferred_order = [
        "run_name",
        "model",
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

    out_metrics_csv = os.path.join(result_root, "all_metrics.csv")

    if os.path.exists(out_metrics_csv):
        old_df = pd.read_csv(out_metrics_csv)
        all_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        all_df = new_df

    all_df.to_csv(out_metrics_csv, index=False)
    print(f"📄 ALL METRICS CSV (updated): {out_metrics_csv}")
else:
    print("⚠️ No metrics collected.")
