import os
import json
import torch
import pandas as pd
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# ========================
# 0. 저장 형식 설정
#    "json" 또는 "csv" 선택
# ========================
SAVE_FORMAT = "csv"  # "json" 또는 "csv" 로 바꿔서 사용


# ========================
# 1. 모델 / 프로세서 로드
# ========================

model_id = "Qwen/Qwen3-VL-2B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

if device == "cuda":
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
else:  # CPU
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
    )

processor = AutoProcessor.from_pretrained(model_id)


# ========================
# 2. VLM 호출 함수
# ========================

def run_vlm_on_image(pil_img: Image.Image, prompt_text: str) -> str:
    """
    한 장의 이미지를 Qwen3-VL에 넣고, 생성된 텍스트(모델 답변 전체)를 문자열로 반환.
    """
    system_prompt = (
        "You are an expert medical AI assistant. "
        "Follow the instructions exactly and respond in the requested JSON format."
    )

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]

    # 1) 멀티모달 chat 템플릿 → 텍스트 프롬프트 생성
    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 2) 텍스트 + 이미지 → 모델 입력 텐서
    inputs = processor(
        text=[chat_text],
        images=[pil_img],
        return_tensors="pt",
    ).to(model.device)

    # 3) 생성
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.2,
            top_p=0.9,
        )

    # 프롬프트 부분 잘라내기
    output_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
    output_text = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return output_text


# ========================
# 3. Baseline 프롬프트
# ========================

BASELINE_PROMPT = (
    'Is there disease in this medical image?\n\n'
    'Answer only in JSON:\n'
    '{"label": "disease"} or {"label": "normal"}'
)


# ========================
# 4. 메타데이터 / 출력 경로 설정
# ========================

meta_csv_path = r"C:\Users\hanna\Lectures\Research_Project\Codes\Dataset\vlm_prompt_dataset\vlm_prompt_dataset_metadata_VLMTEST.csv"

base_out_dir = r"C:\Users\hanna\Lectures\Research_Project\Codes\Dataset\vlm_prompt_dataset"
os.makedirs(base_out_dir, exist_ok=True)

if SAVE_FORMAT.lower() == "json":
    out_path = os.path.join(base_out_dir, "qwen3vl_baseline_results.json")
elif SAVE_FORMAT.lower() == "csv":
    out_path = os.path.join(base_out_dir, "qwen3vl_baseline_results.csv")
else:
    raise ValueError("SAVE_FORMAT 은 'json' 또는 'csv' 중 하나여야 합니다.")

df = pd.read_csv(meta_csv_path)
print("총 샘플 수:", len(df))
print("컬럼:", df.columns.tolist())


# ========================
# 5. 출력에서 label만 뽑는 함수
# ========================

def extract_label_from_output(output_text: str):
    """
    모델 출력 문자열에서 {"label": "disease"} 또는 {"label": "normal"} JSON을 파싱해서
    label 문자열만 리턴. 실패하면 None.
    """
    text = output_text.strip()

    # 1) 전체를 JSON으로 파싱 시도
    try:
        obj = json.loads(text)
        label = obj.get("label", None)
        if label in ["disease", "normal"]:
            return label
    except Exception:
        pass

    # 2) 안 되면 단순 문자열 검색 (fallback)
    lower = text.lower()
    if '"label"' in lower:
        if "disease" in lower and "normal" not in lower:
            return "disease"
        if "normal" in lower and "disease" not in lower:
            return "normal"

    return None


# ========================
# 6. 전체 루프 돌면서 VLM 실행
# ========================

results = []

for i, row in df.iterrows():
    img_path = row["filepath"]

    if not isinstance(img_path, str) or not os.path.exists(img_path):
        print(f"[WARN] 이미지 없음: {img_path}")
        continue

    try:
        pil_img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] 이미지 로드 실패: {img_path}, {e}")
        continue

    # Qwen3-VL 호출
    try:
        raw_output = run_vlm_on_image(pil_img, BASELINE_PROMPT)
    except Exception as e:
        print(f"[ERROR] Qwen3-VL 호출 실패: {img_path}, {e}")
        continue

    pred_label = extract_label_from_output(raw_output)

    rec = {
        # 기존 메타데이터
        "dataset": row.get("dataset", None),
        "corruption": row.get("corruption", None),
        "severity": row.get("severity", None),
        "global_index": row.get("global_index", None),
        "clean_index": row.get("clean_index", None),
        "label_gt": row.get("label", None),
        "filepath": img_path,

        # 모델 출력
        "model_raw_output": raw_output,
        "model_label": pred_label,
    }
    results.append(rec)

    if (i + 1) % 20 == 0:
        print(f"{i+1}/{len(df)} 개 처리 완료...")

print("총 inference 결과 개수:", len(results))


# ========================
# 7. JSON or CSV 로 저장
# ========================

if SAVE_FORMAT.lower() == "json":
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n=== 완료! JSON으로 저장됨 ===\n경로: {out_path}")

elif SAVE_FORMAT.lower() == "csv":
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_path, index=False)
    print(f"\n=== 완료! CSV로 저장됨 ===\n경로: {out_path}")
