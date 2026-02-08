# from PIL import Image
# import numpy as np
# import pandas as pd


# def image_blank_score(path, white_thr=250):
#     # white_thr: 0~255에서 "거의 흰색" 기준
#     im = Image.open(path).convert("L")  # grayscale
#     x = np.asarray(im, dtype=np.uint8)

#     white_ratio = (x >= white_thr).mean()     # 거의 흰색 픽셀 비율
#     std = x.std()                             # 대비/텍스쳐 정도
#     mean = x.mean()
#     return white_ratio, std, mean

# df = pd.read_csv(r"C:\Users\hanna\Lectures\Research_Project\Codes\Dataset\vlm_prompt_dataset\vlm_clean_train_2520.csv")

# before = len(df)
# bad = []
# for i, row in df.iterrows():
#     p = row["filepath"]
#     try:
#         white_ratio, std, mean = image_blank_score(p)
#         # ✅ 기준은 데이터 보고 조절:
#         # - white_ratio가 너무 높고
#         # - std가 너무 낮으면 (거의 한 색)
#         if (white_ratio > 0.98 and std < 3.0):
#             bad.append(i)
#     except Exception as e:
#         # 파일 깨짐/읽기 실패도 학습 제외 추천
#         bad.append(i)

# df_clean = df.drop(index=bad).reset_index(drop=True)
# print("before:", before)
# print("after :", len(df_clean))

# # 저장
# df_clean.to_csv("your_filtered.csv", index=False)

import random, numpy as np, pandas as pd
from PIL import Image

def blank_stats_fast(path, white_thr=250):
    im = Image.open(path).convert("L")
    x = np.asarray(im, dtype=np.uint8)
    white_ratio = (x >= white_thr).mean()
    std = x.std()

    gx = np.abs(x[:, 1:].astype(np.int16) - x[:, :-1].astype(np.int16))
    gy = np.abs(x[1:, :].astype(np.int16) - x[:-1, :].astype(np.int16))
    edge_density = (gx > 10).mean() * 0.5 + (gy > 10).mean() * 0.5
    return white_ratio, std, edge_density

df = pd.read_csv(r"C:\Users\hanna\Lectures\Research_Project\Codes\Dataset\vlm_prompt_dataset\vlm_clean_train_2520.csv")
paths = df["filepath"].tolist()

sample_paths = random.sample(paths, k=min(200, len(paths)))

rows = []
for p in sample_paths:
    try:
        wr, sd, ed = blank_stats_fast(p)
        rows.append((p, wr, sd, ed))
    except:
        pass

rows = np.array([(wr, sd, ed) for _,wr,sd,ed in rows], dtype=np.float64)
wr_all, sd_all, ed_all = rows[:,0], rows[:,1], rows[:,2]

print("white_ratio quantiles:", np.quantile(wr_all, [0.5, 0.9, 0.95, 0.99]))
print("std         quantiles:", np.quantile(sd_all, [0.01, 0.05, 0.1, 0.5]))
print("edge_density quantiles:", np.quantile(ed_all, [0.01, 0.05, 0.1, 0.5]))

# 터진 파일도 같이 찍기
bad_wr, bad_sd, bad_ed = blank_stats_fast("DME-15307-5.jpeg")
print("BAD:", bad_wr, bad_sd, bad_ed)

