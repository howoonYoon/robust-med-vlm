import pandas as pd
import numpy as np

META_CSV = r"C:\Users\hanna\Lectures\Research_Project\Codes\Dataset\vlm_prompt_dataset\vlm_pg_metadata.csv"
OUT_CSV  = r"C:\Users\hanna\Lectures\Research_Project\Codes\Dataset\vlm_prompt_dataset\vlm_pg_metadata_subset_100.csv"

RANDOM_SEED = 42
rng = np.random.RandomState(RANDOM_SEED)

STRICT = True  # 부족하면 에러로 중단 (False면 replace=True로 채움 -> 중복 가능)

# ========================
# 1) Load & filter
# ========================
df = pd.read_csv(META_CSV)

df["severity"]   = df["severity"].fillna("").astype(str).str.strip().str.lower()
df["corruption"] = df["corruption"].fillna("").astype(str).str.strip().str.lower()

# clean 제외 + severity null/빈값 제외
df = df[
    (df["corruption"] != "clean") &
    (df["severity"].isin(["weak", "strong"]))
].copy()

df["binarylabel_num"] = pd.to_numeric(df["binarylabel"], errors="coerce")

# ========================
# 2) helpers
# ========================
def pick_exact(df_pool, n, replace_ok=False):
    if len(df_pool) >= n:
        return df_pool.sample(n=n, random_state=RANDOM_SEED)
    if STRICT and not replace_ok:
        raise ValueError(f"Not enough samples: need {n}, have {len(df_pool)}")
    return df_pool.sample(n=n, random_state=RANDOM_SEED, replace=True)

def rr_sample_by_corruption(df_pool, n, corruption_col="corruption", replace_ok=False):
    """
    corruption별로 round-robin(1개씩 돌아가며) 뽑아서 다양화.
    - df_pool이 충분하면 최대한 중복 없이 corruption을 골고루
    - 부족하면 STRICT에 따라 에러 or replacement
    """
    if n <= 0:
        return df_pool.iloc[0:0].copy()

    # corruption별로 섞어서 리스트화
    groups = {}
    for cor, g in df_pool.groupby(corruption_col):
        g = g.sample(frac=1, random_state=RANDOM_SEED)  # group 내 shuffle
        groups[cor] = list(g.index)

    cors = list(groups.keys())
    rng.shuffle(cors)

    chosen = []
    while len(chosen) < n:
        progressed = False
        for cor in cors:
            if len(chosen) >= n:
                break
            if groups[cor]:
                chosen.append(groups[cor].pop())
                progressed = True

        if not progressed:
            # 더 뽑을 게 없음(풀 부족)
            if STRICT and not replace_ok:
                raise ValueError(f"Not enough unique samples to RR-sample {n} (have {len(chosen)})")
            # replacement로 채우기
            remaining = df_pool.index.tolist()
            if not remaining:
                raise ValueError("Empty pool")
            extra = rng.choice(remaining, size=(n - len(chosen)), replace=True).tolist()
            chosen.extend(extra)

    return df_pool.loc[chosen]

def sample_label_split_with_corruption(df_pool, n1, n0, use_corruption=True, replace_ok=False):
    """
    label 1/0 개수 강제 + (선택) corruption 다양화
    """
    df_pool = df_pool[df_pool["binarylabel_num"].isin([0, 1])].copy()

    p1 = df_pool[df_pool["binarylabel_num"] == 1]
    p0 = df_pool[df_pool["binarylabel_num"] == 0]

    if use_corruption:
        s1 = rr_sample_by_corruption(p1, n1, replace_ok=replace_ok)
        s0 = rr_sample_by_corruption(p0, n0, replace_ok=replace_ok)
    else:
        s1 = pick_exact(p1, n1, replace_ok=replace_ok)
        s0 = pick_exact(p0, n0, replace_ok=replace_ok)

    out = pd.concat([s1, s0]).sample(frac=1, random_state=RANDOM_SEED)
    return out

# ========================
# 3) Sampling per modality
# ========================
subset_rows = []

# --- MRI/OCT/Xray: corruption 다양화 + 비율 강제 ---
for modality in ["MRI", "OCT", "Xray"]:
    df_m = df[df["dataset"] == modality].copy()

    weak_pool   = df_m[df_m["severity"] == "weak"]
    strong_pool = df_m[df_m["severity"] == "strong"]

    weak = sample_label_split_with_corruption(
        weak_pool, n1=8, n0=7, use_corruption=True, replace_ok=not STRICT
    )
    strong = sample_label_split_with_corruption(
        strong_pool, n1=5, n0=5, use_corruption=True, replace_ok=not STRICT
    )

    subset_rows.append(pd.concat([weak, strong]))

# --- Retina: corruption 없음(무시) ---
df_r = df[df["dataset"] == "Retina"].copy()

# weak: 라벨 비율 맞추기 (corruption 무시)
retina_weak_pool = df_r[df_r["severity"] == "weak"]
retina_weak = sample_label_split_with_corruption(
    retina_weak_pool, n1=8, n0=7, use_corruption=False, replace_ok=not STRICT
)

# strong: 라벨 없을 수 있으니 그냥 랜덤 10장
retina_strong_pool = df_r[df_r["severity"] == "strong"]
retina_strong = pick_exact(retina_strong_pool, 10, replace_ok=not STRICT)

subset_rows.append(pd.concat([retina_weak, retina_strong]))

# ========================
# 4) Save & checks
# ========================
df_subset = (
    pd.concat(subset_rows)
    .sample(frac=1, random_state=RANDOM_SEED)
    .reset_index(drop=True)
)

df_subset.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print("✅ Saved:", OUT_CSV)
print("Total images:", len(df_subset))

print("\n=== Check: per-modality severity counts ===")
print(df_subset.groupby(["dataset", "severity"]).size())

print("\n=== Check: per-modality severity label counts (Retina strong may be NaN) ===")
chk = df_subset.copy()
chk["binarylabel_num"] = pd.to_numeric(chk["binarylabel"], errors="coerce")
print(chk.groupby(["dataset", "severity", "binarylabel_num"]).size())

print("\n=== Check: corruption diversity (MRI/OCT/Xray) ===")
for modality in ["MRI", "OCT", "Xray"]:
    for sev in ["weak", "strong"]:
        sub = df_subset[(df_subset["dataset"] == modality) & (df_subset["severity"] == sev)]
        print(modality, sev, "unique_corruptions:", sub["corruption"].nunique())
