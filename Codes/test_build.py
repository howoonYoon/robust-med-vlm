#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CPU-only probe: just build processor inputs (keys + shapes) without loading the model.

Usage examples:
  python probe_processor_shapes.py --backend internvl --n 3
  python probe_processor_shapes.py --backend qwen3 --n 3
  python probe_processor_shapes.py --backend medgemma --n 3
  python probe_processor_shapes.py --backend lingshu --n 3
"""

import os
import argparse
import pandas as pd
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor

os.environ.setdefault("HF_HUB_DISABLE_MMAP", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_ID_BY_BACKEND = {
    "qwen3":    "Qwen/Qwen3-VL-8B-Instruct",
    "medgemma": "google/medgemma-1.5-4b-it",
    "internvl": "OpenGVLab/InternVL3_5-8B-HF",
    "lingshu":  "lingshu-medical-mllm/Lingshu-7B",
}

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

SYSTEM_PROMPT_SHORT = 'Answer with ONE WORD: "normal" or "disease".'


def load_df(csv_path: str, base_out_dir: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "binarylab" in df.columns and "binarylabel" not in df.columns:
        df = df.rename(columns={"binarylab": "binarylabel"})

    if "dataset" not in df.columns:
        raise ValueError("CSV must contain 'dataset' column.")
    if "filepath" not in df.columns:
        raise ValueError("CSV must contain 'filepath' column.")
    if "binarylabel" not in df.columns:
        raise ValueError("CSV must contain 'binarylabel' column.")

    df["dataset_norm"] = df["dataset"].astype(str).str.lower()

    # Windows path -> SAN path (keep your original replacement)
    df["filepath"] = (
        df["filepath"]
        .astype(str)
        .str.replace(
            r"C:\Users\hanna\Lectures\Research_Project\Codes\Dataset\vlm_prompt_dataset",
            base_out_dir,
            regex=False,
        )
        .str.replace("\\", "/", regex=False)
    )

    return df.reset_index(drop=True)


def load_processor(backend: str, model_id: str):
    from transformers import AutoProcessor

    if backend == "internvl":
        # InternVL needs trust_remote_code
        return AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # qwen3 / lingshu / medgemma
    return AutoProcessor.from_pretrained(model_id)


def build_text(row) -> str:
    modality = str(row["dataset_norm"])
    task_prompt = PROMPT_BY_DATASET.get(
        modality,
        "This is a medical image.\nQuestion: Does this image show normal anatomy or signs of disease?\n\n",
    )
    return SYSTEM_PROMPT_SHORT + "\n\n" + task_prompt


def collate_like_training(processor, backend: str, images, texts):
    print("processor.image_token:", getattr(processor, "image_token", None))
    if hasattr(processor, "tokenizer"):
        print("tokenizer.image_token:", getattr(processor.tokenizer, "image_token", None))
        print("additional_special_tokens:", getattr(processor.tokenizer, "additional_special_tokens", None))

    # This mirrors your collate logic, but CPU-only.
    if backend in ["lingshu", "qwen3"]:
        messages_list = []
        for img, txt in zip(images, texts):
            messages_list.append([{
                "role": "user",
                "content": [{"type": "image", "image": img}, {"type": "text", "text": txt}],
            }])
        chat_texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in messages_list
        ]
        model_inputs = processor(text=chat_texts, images=images, padding=True, return_tensors="pt")
        return dict(model_inputs)

    if backend == "internvl":
        image_tok = getattr(processor, "image_token", None) or "<image>"
        inline_texts = [f"{image_tok}\n{txt}" for txt in texts]
        model_inputs = processor(text=inline_texts, images=images, padding=True, return_tensors="pt")
        return dict(model_inputs)

    elif backend == "medgemma":
        # 1) messages -> chat template (여기서 gemma3가 인식하는 image token이 삽입됨)
        messages_list = []
        for img, txt in zip(images, texts):
            messages_list.append([{
                "role": "user",
                "content": [
                    {"type": "image"},                 # 중요: gemma3는 이 형태를 기대
                    {"type": "text", "text": txt},
                ],
            }])

        chat_texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in messages_list
        ]

        # 2) images는 샘플별 list로 감싸서 1:1 매칭 강제
        images_nested = [[img] for img in images]

        # 디버그(필요하면 잠깐만)
        print("chat_texts[0] head:", repr(chat_texts[0][:120]))

        model_inputs = processor(
            text=chat_texts,
            images=images_nested,
            padding=True,
            return_tensors="pt",
        )

        return dict(model_inputs)


def print_batch(out: dict, backend: str):
    print("\n====================")
    print("BACKEND:", backend)
    print("keys:", list(out.keys()))
    for k, v in out.items():
        if torch.is_tensor(v):
            print(f"{k:20s} -> shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")
        elif isinstance(v, list):
            print(f"{k:20s} -> list(len={len(v)}) type={type(v[0]) if len(v)>0 else None}")
        else:
            print(f"{k:20s} -> {type(v)}")

    if "attention_mask" in out and torch.is_tensor(out["attention_mask"]):
        am = out["attention_mask"]
        uniq = torch.unique(am)
        print(f"attention_mask uniq: {uniq[:10].tolist()} (showing up to 10 values)")

    # vision tensor key guess
    for vk in ["pixel_values", "images", "image", "vision_x"]:
        if vk in out and torch.is_tensor(out[vk]):
            print(f"vision key detected: {vk} -> {tuple(out[vk].shape)}")
            break
    print("====================\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", type=str, required=True, choices=list(MODEL_ID_BY_BACKEND.keys()))
    ap.add_argument("--train_csv", type=str, default="/SAN/ioo/HORIZON/howoon/vlm_clean_weak_train_2520.csv")
    ap.add_argument("--base_out_dir", type=str, default="/SAN/ioo/HORIZON/howoon")
    ap.add_argument("--n", type=int, default=3, help="number of samples to probe")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)

    model_id = MODEL_ID_BY_BACKEND[args.backend]
    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)

    df = load_df(args.train_csv, args.base_out_dir)
    if len(df) == 0:
        raise RuntimeError("Empty CSV after loading.")

    # sample n rows
    n = min(args.n, len(df))
    sample_df = df.sample(n=n, random_state=args.seed).reset_index(drop=True)

    images, texts = [], []
    for i in range(n):
        row = sample_df.iloc[i]
        img_path = row["filepath"]
        img = Image.open(img_path).convert("RGB")
        images.append(img)
        texts.append(build_text(row))

    out = collate_like_training(processor, args.backend, images, texts)
    print_batch(out, args.backend)

    # Also print token lengths if available
    if "input_ids" in out and torch.is_tensor(out["input_ids"]):
        lens = (out["input_ids"] != 0).sum(dim=1) if out["input_ids"].dim() == 2 else None
        if lens is not None:
            print("approx token lengths per sample:", lens.tolist())


if __name__ == "__main__":
    main()
