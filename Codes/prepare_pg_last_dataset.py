import os
import re
import pandas as pd

ROOT = r"C:\Users\hanna\Lectures\Research_Project\Codes\Dataset\vlm_prompt_dataset\pg_last\Benchmarks"

DATASET_CONFIG = {
    "Retina": {"path": "Color fundus", "type": "retina"},
    "MRI": {"path": "MRI", "type": "standard"},
    "OCT": {"path": "OCT", "type": "standard"},
    "Xray": {"path": "X-ray", "type": "standard"},
}

IMG_EXT = (".jpg", ".png", ".jpeg", ".bmp")

rows = []
idx = 1

def extract_index(fname: str) -> str:
    m = re.search(r"\d+", fname)
    return m.group() if m else ""

def binary_label(label: str):
    # label이 normal이면 0, 그 외는 1로 처리(원하면 여기서 더 엄격하게 바꿀 수 있음)
    return 0 if label.lower() == "normal" else 1

for dataset, cfg in DATASET_CONFIG.items():
    base_path = os.path.join(ROOT, cfg["path"])

    if cfg["type"] == "retina":
        # Retina: higher_qual_dataset (label subfolders), weak_artefacts_dataset (label subfolders),
        # strong_artefacts_dataset (NO subfolders, only images)
        for subset in os.listdir(base_path):
            subset_path = os.path.join(base_path, subset)
            if not os.path.isdir(subset_path):
                continue

            if "higher" in subset:
                corruption = "higher_qual"
                severity = ""
            elif "weak" in subset:
                corruption = "weak_artefacts"
                severity = "weak"
            elif "strong" in subset:
                corruption = "strong_artefacts"
                severity = "strong"
            else:
                corruption = subset
                severity = ""

            subdirs = [d for d in os.listdir(subset_path) if os.path.isdir(os.path.join(subset_path, d))]

            if len(subdirs) > 0:
                for label in subdirs:
                    label_path = os.path.join(subset_path, label)
                    for f in os.listdir(label_path):
                        if f.lower().endswith(IMG_EXT):
                            rows.append({
                                "IDX": idx,
                                "dataset": "Retina",
                                "corruption": corruption,
                                "severity": severity,
                                "fileindex": extract_index(f),
                                "filename": f,
                                "label": label,
                                "filepath": os.path.join(label_path, f),
                                "binarylabel": 0 if label.lower() == "normal" else 1
                            })
                            idx += 1
            else:
                for f in os.listdir(subset_path):
                    if f.lower().endswith(IMG_EXT):
                        rows.append({
                            "IDX": idx,
                            "dataset": "Retina",
                            "corruption": corruption,
                            "severity": severity,
                            "fileindex": extract_index(f),
                            "filename": f,
                            "label": "ungradable",
                            "filepath": os.path.join(subset_path, f),
                            "binarylabel": ""
                        })
                        idx += 1

    else:
        # MRI / OCT / Xray
        for artefact in os.listdir(base_path):
            artefact_path = os.path.join(base_path, artefact)
            if not os.path.isdir(artefact_path):
                continue

            # ========= FIXED: "original images" 아래에 라벨 폴더가 있는 케이스 지원 =========
            if artefact == "original images":
                # case A) original images 바로 밑에 label 폴더들이 존재
                label_dirs = [d for d in os.listdir(artefact_path)
                              if os.path.isdir(os.path.join(artefact_path, d))]

                if len(label_dirs) > 0:
                    for label in label_dirs:
                        label_path = os.path.join(artefact_path, label)
                        for f in os.listdir(label_path):
                            if f.lower().endswith(IMG_EXT):
                                rows.append({
                                    "IDX": idx,
                                    "dataset": dataset,
                                    "corruption": "clean",
                                    "severity": "",
                                    "fileindex": extract_index(f),
                                    "filename": f,
                                    "label": label,  # ex) brain tumor / normal 등
                                    "filepath": os.path.join(label_path, f),
                                    "binarylabel": binary_label(label)
                                })
                                idx += 1
                else:
                    # case B) original images 폴더 바로 밑에 이미지가 있는 기존 구조도 유지
                    for f in os.listdir(artefact_path):
                        if f.lower().endswith(IMG_EXT):
                            label = "normal" if "normal" in f.lower() else "disease"
                            rows.append({
                                "IDX": idx,
                                "dataset": dataset,
                                "corruption": "clean",
                                "severity": "",
                                "fileindex": extract_index(f),
                                "filename": f,
                                "label": label,
                                "filepath": os.path.join(artefact_path, f),
                                "binarylabel": binary_label(label)
                            })
                            idx += 1
            # ===========================================================================
            else:
                for sev in ["strong", "weak"]:
                    sev_path = os.path.join(artefact_path, sev)
                    if not os.path.isdir(sev_path):
                        continue

                    for label in os.listdir(sev_path):
                        label_path = os.path.join(sev_path, label)
                        if not os.path.isdir(label_path):
                            continue

                        for f in os.listdir(label_path):
                            if f.lower().endswith(IMG_EXT):
                                rows.append({
                                    "IDX": idx,
                                    "dataset": dataset,
                                    "corruption": artefact,
                                    "severity": sev,
                                    "fileindex": extract_index(f),
                                    "filename": f,
                                    "label": label,
                                    "filepath": os.path.join(label_path, f),
                                    "binarylabel": binary_label(label)
                                })
                                idx += 1

df = pd.DataFrame(rows)
out_csv = os.path.join(ROOT, "vlm_benchmark_metadata.csv")
df.to_csv(out_csv, index=False, encoding="utf-8-sig")

print("Saved:", out_csv)
print("Total images:", len(df))
