import os
import pandas as pd
import numpy as np

# ===== MRI 데이터 경로 =====
mri_root = r"C:\Users\hanna\Lectures\Research_Project\Codes\Dataset\mri\Challenging Datasets\Challenging Datasets"

mri_corruptions = {
    "Blurred": os.path.join(mri_root, "Blurred"),
    "Motion": os.path.join(mri_root, "Motion"),
    "Noisy": os.path.join(mri_root, "Noisy"),
}

# MRI 라벨 매핑
mri_label_map = {
    "No-tumor": 0,
    "Glioma": 1,
    "Meningioma": 2,
    "Pituitary": 3,
}

# 기존 메타데이터 CSV
meta_csv_path = r"C:\Users\hanna\Lectures\Research_Project\Codes\Dataset\vlm_prompt_dataset\vlm_prompt_dataset_metadata.csv"

# 기존 CSV 로드
if os.path.exists(meta_csv_path):
    meta_df = pd.read_csv(meta_csv_path)
else:
    meta_df = pd.DataFrame(columns=[
        "dataset", "corruption", "severity", "global_index", "clean_index",
        "label", "filepath"
    ])

mri_meta_rows = []

# ===== MRI 메타데이터 수집 =====

for corruption_name, corruption_path in mri_corruptions.items():
    print(f"\n===== MRI Corruption: {corruption_name} =====")

    for label_text, numeric_label in mri_label_map.items():
        label_dir = os.path.join(corruption_path, label_text)

        if not os.path.exists(label_dir):
            print(f"[WARN] Missing label path: {label_dir}")
            continue

        imgs = [f for f in os.listdir(label_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        if len(imgs) == 0:
            print(f"[WARN] No images under: {label_dir}")
            continue

        print(f"  - {label_text} ({numeric_label}): {len(imgs)} images")

        for idx, img_name in enumerate(imgs):
            img_path = os.path.join(label_dir, img_name)

            mri_meta_rows.append({
                "dataset": "mri",
                "corruption": corruption_name,
                "severity": None,              # MRI는 severity 없음
                "global_index": idx,           # MedMNIST의 global_index 대응
                "clean_index": idx,            # 동일 값으로 맞춰서 구조 통일
                "label": numeric_label,
                "filepath": img_path,
            })

# ===== CSV 업데이트 =====

mri_df = pd.DataFrame(mri_meta_rows)
final_df = pd.concat([meta_df, mri_df], ignore_index=True)

final_df.to_csv(meta_csv_path, index=False)

print("\n=== MRI 메타데이터 추가 완료 ===")
print("총 추가된 항목:", len(mri_df))
print("CSV 저장 위치:", meta_csv_path)
print(final_df.tail())
