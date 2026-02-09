#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import gc
import json
from typing import Optional, Any, Dict, Tuple
from datetime import datetime
import argparse

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import torchvision.transforms as T
import torchvision.models as tvm


# -------------------------
# Args
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", type=str, default="effv2_l",
                   choices=["effv2_s", "effv2_m", "effv2_l", "eff_b0", "eff_b1", "eff_b2", "eff_b3", "eff_b4", "eff_b5", "eff_b6", "eff_b7"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--val_csv", type=str, required=True)

    # fairness knobs
    p.add_argument("--freeze_backbone", action="store_true", help="Freeze EfficientNet backbone, train head only (recommended for fair compare vs VLM frozen).")
    p.add_argument("--target_trainable_params", type=int, default=0,
                   help="If >0, set head hidden dim so trainable params ~= target. (Works best with --freeze_backbone)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="./effnet_ckpt")
    return p.parse_args()


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_params(m: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable

def compute_binary_metrics(y_true: torch.Tensor, y_score: torch.Tensor, thr: float = 0.5):
    y_true = y_true.detach().cpu().to(torch.int64)
    y_score = y_score.detach().cpu().to(torch.float32)

    y_pred = (y_score >= thr).to(torch.int64)

    tp = int(((y_pred == 1) & (y_true == 1)).sum().item())
    tn = int(((y_pred == 0) & (y_true == 0)).sum().item())
    fp = int(((y_pred == 1) & (y_true == 0)).sum().item())
    fn = int(((y_pred == 0) & (y_true == 1)).sum().item())

    eps = 1e-12
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    # AUROC (same style as your code; tie-handling not perfect but OK)
    pos = (y_true == 1)
    neg = (y_true == 0)
    if int(pos.sum()) == 0 or int(neg.sum()) == 0:
        auroc = float("nan")
    else:
        order = torch.argsort(y_score)  # ascending
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(1, len(y_score) + 1, dtype=torch.float32)
        sum_ranks_pos = ranks[pos].sum()
        n_pos = float(pos.sum().item())
        n_neg = float(neg.sum().item())
        u = sum_ranks_pos - n_pos * (n_pos + 1.0) / 2.0
        auroc = float((u / (n_pos * n_neg)).item())

    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(auroc),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# -------------------------
# CSV loader (same idea as yours)
# -------------------------
def load_split_csv(path: str, base_out_dir: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "binarylab" in df.columns and "binarylabel" not in df.columns:
        df = df.rename(columns={"binarylab": "binarylabel"})

    # normalize severity
    sev = df.get("severity", None)
    if sev is None:
        df["severity_norm"] = "clean"
    else:
        df["severity_norm"] = df["severity"].fillna("clean").astype(str).str.lower()

    # filepath rewrite if needed
    if base_out_dir is not None and "filepath" in df.columns:
        df["filepath"] = (
            df["filepath"].astype(str)
            .str.replace(
                r"C:\Users\hanna\Lectures\Research_Project\Codes\Dataset\vlm_prompt_dataset",
                base_out_dir,
                regex=False,
            )
            .str.replace("\\", "/", regex=False)
        )

    if "filepath" not in df.columns:
        raise ValueError(f"{path}: 'filepath' column missing")
    if "binarylabel" not in df.columns:
        raise ValueError(f"{path}: 'binarylabel' column missing")

    return df.reset_index(drop=True)


# -------------------------
# Dataset
# -------------------------
class ImageClsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform
        print(f"Dataset - found {len(self.df)} samples.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row["filepath"]
        y = int(row["binarylabel"])
        sev = row.get("severity_norm", "unknown")

        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return {"x": img, "y": torch.tensor(y, dtype=torch.long), "severity": sev, "path": path}


def make_transforms(img_size: int, train: bool):
    if train:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])


# -------------------------
# Model: EfficientNet + MLP head
# -------------------------
class EfficientNetBinary(nn.Module):
    def __init__(self, variant: str, hidden: int = 0, freeze_backbone: bool = False):
        super().__init__()

        # backbone
        if variant == "effv2_s":
            self.backbone = tvm.efficientnet_v2_s(weights=tvm.EfficientNet_V2_S_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
        elif variant == "effv2_m":
            self.backbone = tvm.efficientnet_v2_m(weights=tvm.EfficientNet_V2_M_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
        elif variant == "effv2_l":
            self.backbone = tvm.efficientnet_v2_l(weights=tvm.EfficientNet_V2_L_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
        elif variant == "eff_b0":
            self.backbone = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
        elif variant == "eff_b1":
            self.backbone = tvm.efficientnet_b1(weights=tvm.EfficientNet_B1_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
        elif variant == "eff_b2":
            self.backbone = tvm.efficientnet_b2(weights=tvm.EfficientNet_B2_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
        elif variant == "eff_b3":
            self.backbone = tvm.efficientnet_b3(weights=tvm.EfficientNet_B3_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
        elif variant == "eff_b4":
            self.backbone = tvm.efficientnet_b4(weights=tvm.EfficientNet_B4_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
        elif variant == "eff_b5":
            self.backbone = tvm.efficientnet_b5(weights=tvm.EfficientNet_B5_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
        elif variant == "eff_b6":
            self.backbone = tvm.efficientnet_b6(weights=tvm.EfficientNet_B6_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
        elif variant == "eff_b7":
            self.backbone = tvm.efficientnet_b7(weights=tvm.EfficientNet_B7_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
        else:
            raise ValueError(f"Unknown variant: {variant}")

        # remove original classifier -> expose features
        self.backbone.classifier = nn.Identity()

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # head: either linear or 2-layer MLP
        if hidden and hidden > 0:
            self.head = nn.Sequential(
                nn.Linear(feat_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 2),
            )
        else:
            self.head = nn.Linear(feat_dim, 2)

        self.feat_dim = feat_dim

    def forward(self, x):
        feat = self.backbone(x)          # (B, feat_dim)
        logits = self.head(feat)         # (B, 2)
        return logits


def choose_hidden_for_target(feat_dim: int, target_trainable_params: int) -> int:
    """
    Trainable params for head MLP:
      W1: feat_dim*H, b1: H
      W2: H*2,       b2: 2
    total = H*(feat_dim + 1 + 2) + 2 = H*(feat_dim+3) + 2
    => H ~ (target-2)/(feat_dim+3)
    """
    if target_trainable_params <= 0:
        return 0
    H = int((target_trainable_params - 2) / (feat_dim + 3))
    return max(H, 0)


# -------------------------
# Train / Eval
# -------------------------
def run_epoch(model, loader, optimizer=None, device="cuda"):
    train = optimizer is not None
    model.train() if train else model.eval()

    all_y, all_score, all_sev = [], [], []
    total_loss, n_steps = 0.0, 0

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        sev = batch["severity"]

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            prob = torch.softmax(logits.float(), dim=1)[:, 1]
            all_y.append(y.detach().cpu())
            all_score.append(prob.detach().cpu())
            all_sev.extend(list(sev))

        total_loss += float(loss.item())
        n_steps += 1

    y_true = torch.cat(all_y, dim=0)
    y_score = torch.cat(all_score, dim=0)
    m_all = compute_binary_metrics(y_true, y_score)

    all_sev = np.array(all_sev)
    clean_mask = all_sev == "clean"
    weak_mask  = all_sev == "weak"

    m_clean = compute_binary_metrics(y_true[clean_mask], y_score[clean_mask]) if clean_mask.sum() > 0 else {}
    m_weak  = compute_binary_metrics(y_true[weak_mask],  y_score[weak_mask])  if weak_mask.sum() > 0 else {}

    return (total_loss / max(n_steps, 1)), {"overall": m_all, "clean": m_clean, "weak": m_weak}


def main():
    args = parse_args()
    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_dir = os.path.join(args.out_dir, "metrics_json")
    os.makedirs(metrics_dir, exist_ok=True)

    # load csv
    base_out_dir = "/SAN/ioo/HORIZON/howoon"
    train_df = load_split_csv(args.train_csv, base_out_dir=base_out_dir)
    val_df   = load_split_csv(args.val_csv,   base_out_dir=base_out_dir)
    print("train_df shape:", train_df.shape, "val_df shape:", val_df.shape)

    # datasets
    tr_tf = make_transforms(args.img_size, train=True)
    va_tf = make_transforms(args.img_size, train=False)

    train_ds = ImageClsDataset(train_df, transform=tr_tf)
    val_ds   = ImageClsDataset(val_df, transform=va_tf)

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,  num_workers=4, pin_memory=(device=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=(device=="cuda"))

    # create model with optional param matching
    # build a temporary model to get feat_dim
    tmp = EfficientNetBinary(args.variant, hidden=0, freeze_backbone=args.freeze_backbone)
    feat_dim = tmp.feat_dim
    del tmp

    hidden = 0
    if args.target_trainable_params and args.target_trainable_params > 0:
        hidden = choose_hidden_for_target(feat_dim, args.target_trainable_params)

    model = EfficientNetBinary(args.variant, hidden=hidden, freeze_backbone=args.freeze_backbone).to(device)

    total_p, train_p = count_params(model)
    print(f"[EffNet] variant={args.variant} feat_dim={feat_dim} hidden={hidden} "
          f"total_params={total_p:,} trainable_params={train_p:,}")

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    best_val = -1.0
    BEST_CKPT = os.path.join(args.out_dir, f"{RUN_ID}_{args.variant}_best.pt")
    LAST_CKPT = os.path.join(args.out_dir, f"{RUN_ID}_{args.variant}_last.pt")
    metrics_path = os.path.join(metrics_dir, f"{RUN_ID}_{args.variant}_metrics.json")

    logs = []

    for epoch in range(args.epochs):
        tr_loss, tr_m = run_epoch(model, train_loader, optimizer=optimizer, device=device)
        va_loss, va_m = run_epoch(model, val_loader,   optimizer=None,      device=device)

        tr_auroc = tr_m["overall"].get("auroc", float("nan"))
        va_auroc = va_m["overall"].get("auroc", float("nan"))
        va_clean = va_m.get("clean", {}).get("auroc", float("nan"))
        va_weak  = va_m.get("weak",  {}).get("auroc", float("nan"))

        print(
            f"[EffNet] Epoch {epoch+1}/{args.epochs}\n"
            f"Train: loss={tr_loss:.4f} AUROC={tr_auroc:.3f}\n"
            f"Val  : loss={va_loss:.4f} AUROC={va_auroc:.3f} | Clean={va_clean:.3f} | Weak={va_weak:.3f}"
        )

        logs.append({
            "epoch": epoch + 1,
            "train_loss": float(tr_loss),
            "val_loss": float(va_loss),
            "train": tr_m,
            "val": va_m,
        })

        score = va_auroc
        if not np.isnan(score) and score > best_val:
            best_val = score
            torch.save({"model": model.state_dict(),
                        "variant": args.variant,
                        "hidden": hidden,
                        "freeze_backbone": args.freeze_backbone,
                        "best_val_auroc": best_val,
                        "epoch": epoch + 1}, BEST_CKPT)
            print(f"BEST saved: {BEST_CKPT} (best_val_auroc={best_val:.4f})")

        # always save last + metrics
        torch.save({"model": model.state_dict(),
                    "variant": args.variant,
                    "hidden": hidden,
                    "freeze_backbone": args.freeze_backbone,
                    "epoch": epoch + 1,
                    "best_val_auroc": best_val}, LAST_CKPT)

        payload = {
            "run_id": RUN_ID,
            "variant": args.variant,
            "hidden": hidden,
            "freeze_backbone": args.freeze_backbone,
            "img_size": args.img_size,
            "seed": args.seed,
            "epochs": args.epochs,
            "best_val_auroc": best_val,
            "logs": logs,
        }
        tmp_path = metrics_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, metrics_path)

    del model, optimizer, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
