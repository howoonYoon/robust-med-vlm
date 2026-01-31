from pathlib import Path
import csv
# --------mri--------------
# # ===== 설정 =====
# ROOT = Path(r"C:\Users\hanna\Lectures\Research_Project\Codes\Dataset\vlm_prompt_dataset\pg_all")
# DATASET_NAME = "MRI"
# SPLITS = ["Training", "Testing"]
# CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
# OUT_CSV = ROOT / "mri_clean_index.csv"

# # ===== 실행 =====
# rows = []
# idx = 1

# for split in SPLITS:
#     for cls in CLASSES:
#         cls_dir = ROOT / DATASET_NAME / split / cls
#         if not cls_dir.exists():
#             print(f"[WARN] 폴더 없음: {cls_dir}")
#             continue

#         exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
#         files = [p for p in cls_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
#         files.sort(key=lambda p: str(p).lower())

#         for p in files:
#             label = cls
#             binarylabel = 0 if label.lower() == "notumor" else 1

#             rows.append({
#                 "IDX": idx,
#                 "dataset": DATASET_NAME,
#                 "dataset_detail": split,          # ⭐ 추가
#                 "corruption": "clean",
#                 "severity": "",
#                 "fileindex": p.stem,
#                 "filename": p.name,
#                 "label": label,
#                 "filepath": str(p),
#                 "binarylabel": binarylabel,
#             })
#             idx += 1

# # ===== CSV 저장 =====
# fieldnames = [
#     "IDX", "dataset", "dataset_detail",
#     "corruption", "severity",
#     "fileindex", "filename",
#     "label", "filepath", "binarylabel"
# ]

# with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
#     writer = csv.DictWriter(f, fieldnames=fieldnames)
#     writer.writeheader()
#     writer.writerows(rows)

# print(f"Saved: {OUT_CSV} ({len(rows)} rows)")


#--------OCT--------------

from pathlib import Path
import csv

# ===== 설정 =====
# ROOT = Path(r"C:\Users\hanna\Lectures\Research_Project\Codes\Dataset\vlm_prompt_dataset\pg_all")
# DATASET_NAME = "OCT"
# SPLITS = ["train", "test"]
# CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
# OUT_CSV = ROOT / "oct_clean_index.csv"

# # ===== 실행 =====
# rows = []
# idx = 1

# for split in SPLITS:
#     for cls in CLASSES:
#         cls_dir = ROOT / DATASET_NAME / split / cls
#         if not cls_dir.exists():
#             print(f"[WARN] 폴더 없음: {cls_dir}")
#             continue

#         exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
#         files = [p for p in cls_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
#         files.sort(key=lambda p: str(p).lower())

#         for p in files:
#             label = cls
#             binarylabel = 0 if label.upper() == "NORMAL" else 1

#             rows.append({
#                 "IDX": idx,
#                 "dataset": DATASET_NAME,
#                 "dataset_detail": split,     # train / test
#                 "corruption": "clean",
#                 "severity": "",
#                 "fileindex": p.stem,
#                 "filename": p.name,
#                 "label": label,
#                 "filepath": str(p),
#                 "binarylabel": binarylabel,
#             })
#             idx += 1

# # ===== CSV 저장 =====
# fieldnames = [
#     "IDX", "dataset", "dataset_detail",
#     "corruption", "severity",
#     "fileindex", "filename",
#     "label", "filepath", "binarylabel"
# ]

# with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
#     writer = csv.DictWriter(f, fieldnames=fieldnames)
#     writer.writeheader()
#     writer.writerows(rows)

# print(f"Saved: {OUT_CSV} ({len(rows)} rows)")


#--------ChestXray--------------
from pathlib import Path
import csv

# ===== 설정 =====
ROOT = Path(r"C:\Users\hanna\Lectures\Research_Project\Codes\Dataset\vlm_prompt_dataset\pg_all")
DATASET_NAME = "Xray"
SPLITS = ["NonAugmentedTrain", "TrainData", "ValData"]
CLASSES = ["BacterialPneumonia", "COVID-19", "Normal", "ViralPneumonia"]
OUT_CSV = ROOT / "xray_clean_index.csv"

# ===== 실행 =====
rows = []
idx = 1

for split in SPLITS:
    for cls in CLASSES:
        cls_dir = ROOT / DATASET_NAME / split / cls
        if not cls_dir.exists():
            print(f"[WARN] 폴더 없음: {cls_dir}")
            continue

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        files = [p for p in cls_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
        files.sort(key=lambda p: str(p).lower())

        for p in files:
            label = cls
            binarylabel = 0 if label.lower() == "normal" else 1

            rows.append({
                "IDX": idx,
                "dataset": DATASET_NAME,
                "dataset_detail": split,
                "corruption": "clean",
                "severity": "",
                "fileindex": p.stem,
                "filename": p.name,
                "label": label,
                "filepath": str(p),
                "binarylabel": binarylabel,
            })
            idx += 1

# ===== CSV 저장 =====
fieldnames = [
    "IDX", "dataset", "dataset_detail",
    "corruption", "severity",
    "fileindex", "filename",
    "label", "filepath", "binarylabel"
]

with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved: {OUT_CSV} ({len(rows)} rows)")
