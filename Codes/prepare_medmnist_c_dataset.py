import os
import numpy as np
from PIL import Image
import pandas as pd

# ===== 1. 경로 & 출력 폴더 설정 =====

# 각 데이터셋의 npz가 들어있는 폴더 (네가 준 경로로 교체)
base_dirs = [
    r"C:\Users\hanna\Lectures\Research_Project\Codes\Dataset\breastmnist_c\breastmnist",
    r"C:\Users\hanna\Lectures\Research_Project\Codes\Dataset\octmnist_c\octmnist",
    r"C:\Users\hanna\Lectures\Research_Project\Codes\Dataset\pneumoniamnist_c\pneumoniamnist",
    r"C:\Users\hanna\Lectures\Research_Project\Codes\Dataset\retinamnist_c\retinamnist",
]

# PNG랑 CSV를 저장할 폴더 (원하면 경로 바꿔도 됨)
output_root = r"C:\Users\hanna\Lectures\Research_Project\Codes\Dataset\vlm_prompt_dataset"
os.makedirs(output_root, exist_ok=True)

csv_path = os.path.join(output_root, "vlm_prompt_dataset_metadata.csv")

np.random.seed(0)  # 재현성 (지금은 랜덤 안 쓰지만 놔둬도 됨)

meta_rows = []  # csv로 저장할 메타데이터


def save_all_images_from_npz(dataset_root):
    """
    dataset_root: 예) C:/.../octmnist_c/octmnist

    해당 폴더 안의 모든 corruption npz에 대해
    test_images 전체(= 모든 severity * 모든 index)를 PNG로 저장하고,
    메타데이터를 meta_rows에 추가.
    """
    dataset_name = os.path.basename(dataset_root)  # breastmnist / octmnist / ...

    print(f"\n=== Dataset: {dataset_name} (root: {dataset_root}) ===")

    # 이 데이터셋 전용 출력 폴더
    out_dir = os.path.join(output_root, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    # npz 파일들 (각 corruption 별)
    npz_files = sorted([f for f in os.listdir(dataset_root) if f.endswith(".npz")])
    if not npz_files:
        print(f"[WARN] No npz files in {dataset_root}")
        return

    for npz_name in npz_files:
        npz_path = os.path.join(dataset_root, npz_name)
        corruption_name = os.path.splitext(npz_name)[0]

        print(f"  - Loading {npz_name} ...", end=" ")

        data = np.load(npz_path)
        imgs = data["test_images"]
        labels = data["test_labels"].squeeze()

        total = imgs.shape[0]

        # MedMNIST-C: test set이 severity 5단계로 복제된 구조 → total = N * 5
        if total % 5 != 0:
            print(f"\n[WARN] {npz_path}: total={total} 이 5로 안 나눠떨어짐, 스킵")
            continue

        N = total // 5  # 원래 clean test set 크기
        print(f" total={total}, per_severity={N}")

        # 🔴 여기서부터: 랜덤 1장 말고, 모든 이미지 순회
        for idx in range(total):
            img_arr = imgs[idx]
            label = labels[idx]
            if isinstance(label, np.ndarray):
                label = int(label.squeeze())
            else:
                label = int(label)

            img_arr = np.array(img_arr)

            # [0,1] 스케일이면 0~255로 변환
            if img_arr.max() <= 1.0:
                img_arr = (img_arr * 255).astype(np.uint8)
            else:
                img_arr = img_arr.astype(np.uint8)

            # 채널/shape 처리
            if img_arr.ndim == 2:
                pil_img = Image.fromarray(img_arr)  # grayscale
            elif img_arr.ndim == 3 and img_arr.shape[2] == 1:
                pil_img = Image.fromarray(img_arr[:, :, 0])
            elif img_arr.ndim == 3 and img_arr.shape[2] in (3, 4):
                pil_img = Image.fromarray(img_arr[:, :, :3])
            else:
                raise ValueError(f"Unexpected image shape: {img_arr.shape}")

            # severity & clean_index 계산
            # severity: 0~4, clean_index: 0~(N-1)
            severity = idx // N
            clean_index = idx % N

            # 파일 이름:
            #   {dataset}__{corruption}__sev{severity}__idx{clean_index}__label{label}.png
            filename = (
                f"{dataset_name}__{corruption_name}__"
                f"sev{severity}__idx{clean_index:05d}__label{label}.png"
            )
            save_path = os.path.join(out_dir, filename)
            pil_img.save(save_path)

            meta_rows.append({
                "dataset": dataset_name,
                "corruption": corruption_name,
                "severity": int(severity),
                "global_index": int(idx),        # test_images 전체에서의 index
                "clean_index": int(clean_index), # 원래 test set index
                "label": label,
                "filepath": save_path,
            })


# ===== 2. 각 base_dir에 대해 실행 =====

for b in base_dirs:
    print(f"\n######## Processing base dir: {b} ########")
    save_all_images_from_npz(b)

# ===== 3. CSV 파일 있으면 append, 없으면 새로 생성 =====

if os.path.exists(csv_path):
    print(f"\n[INFO] 기존 CSV 발견 → append 모드로 병합: {csv_path}")
    existing_df = pd.read_csv(csv_path)

    new_df = pd.DataFrame(meta_rows)

    # concat
    final_df = pd.concat([existing_df, new_df], ignore_index=True)

else:
    print(f"\n[INFO] 기존 CSV 없음 → 새로 생성")
    final_df = pd.DataFrame(meta_rows)

final_df.to_csv(csv_path, index=False)

print("\n=== Done! CSV saved ===")
print("CSV path:", csv_path)
print(final_df.tail())
