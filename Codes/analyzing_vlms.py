#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from PIL import Image
import numpy as np


# ----------------------------
# Backend config
# ----------------------------
MODEL_ID_BY_BACKEND = {
    "qwen3":    "Qwen/Qwen3-VL-8B-Instruct",
    "medgemma": "google/medgemma-1.5-4b-it",
    "internvl": "OpenGVLab/InternVL3_5-8B-HF",
    "lingshu":  "lingshu-medical-mllm/Lingshu-7B",
}

SYSTEM_PROMPT_SHORT = 'Answer with ONE WORD: "normal" or "disease".'
TASK_PROMPT = (
    "This is a medical image.\n"
    "Question: Does this image show normal anatomy or signs of disease?\n\n"
)

# heuristic keywords
KW_VISION = ["vision", "visual", "image", "vit", "clip", "resampler", "perceiver"]
KW_PROJECTOR = ["projector", "mm_projector", "vision_proj", "visual_proj", "projection", "proj", "mapper"]
KW_ADAPTER = ["adapter", "lora", "ia3", "prompt", "mlp_adapter"]  # base 모델 내부 어댑터 후보 (있을 수도/없을 수도)
KW_FUSE = ["fusion", "multimodal", "mm", "cross", "qformer", "connector"]


# ----------------------------
# utils
# ----------------------------
def set_env_like_you():
    # 너 코드 스타일 유지 (optional)
    assert os.environ.get("HF_HOME"), "HF_HOME not set (use TMPDIR/hf_cache etc.)"
    hf_home = os.environ["HF_HOME"]
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(hf_home, "hub"))


def make_dummy_image(size=224) -> Image.Image:
    # deterministic dummy image
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:, :, 0] = 128
    arr[::8, :, 1] = 255
    arr[:, ::8, 2] = 255
    return Image.fromarray(arr, mode="RGB")


def print_tree(model: nn.Module, depth: int = 2) -> List[Tuple[str, str]]:
    # returns a list of (name, class)
    rows = []
    def rec(m, prefix, d):
        if d < 0:
            return
        for name, child in m.named_children():
            rows.append((prefix + name, child.__class__.__name__))
            rec(child, prefix + name + ".", d - 1)
    rec(model, "", depth)
    return rows


def list_named_modules(model: nn.Module, keywords: Optional[List[str]] = None, max_items=500) -> List[Tuple[str, str]]:
    out = []
    for name, m in model.named_modules():
        if name == "":
            continue
        low = name.lower()
        if keywords is None or any(k in low for k in keywords):
            out.append((name, m.__class__.__name__))
            if len(out) >= max_items:
                break
    return out


def tensor_brief(x: torch.Tensor) -> Dict[str, Any]:
    x = x.detach()
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype).replace("torch.", ""),
        "device": str(x.device),
        "min": float(torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0).min().item()) if x.numel() else None,
        "max": float(torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0).max().item()) if x.numel() else None,
        "nan": int(torch.isnan(x.float()).sum().item()) if x.numel() else 0,
        "inf": int(torch.isinf(x.float()).sum().item()) if x.numel() else 0,
    }


def summarize_output(obj: Any) -> Dict[str, Any]:
    # hook output can be Tensor / tuple / dict / other
    if torch.is_tensor(obj):
        return {"type": "tensor", **tensor_brief(obj)}
    if isinstance(obj, (tuple, list)):
        # summarize first 2 tensors only
        summ = {"type": type(obj).__name__, "len": len(obj), "items": []}
        for i, it in enumerate(obj[:2]):
            if torch.is_tensor(it):
                summ["items"].append({"i": i, **tensor_brief(it)})
            else:
                summ["items"].append({"i": i, "type": type(it).__name__})
        return summ
    if isinstance(obj, dict):
        keys = list(obj.keys())
        summ = {"type": "dict", "keys": keys[:20], "tensors": {}}
        for k in keys[:20]:
            v = obj[k]
            if torch.is_tensor(v):
                summ["tensors"][k] = tensor_brief(v)
        return summ
    return {"type": type(obj).__name__}


# ----------------------------
# build inputs per backend
# ----------------------------
def build_inputs(backend: str, processor, image: Image.Image, text: str, device: str) -> Dict[str, Any]:
    # follow your collate logic (single sample)
    if backend in ["lingshu", "qwen3"]:
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": text}],
        }]
        chat_text = processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=False)
        model_inputs = processor(text=chat_text, images=[image], padding=True, return_tensors="pt")

    elif backend == "internvl":
        image_tok = getattr(processor, "image_token", None) or "<image>"
        inline_text = f"{image_tok}\n{text}"
        model_inputs = processor(text=[inline_text], images=[image], padding=True, return_tensors="pt")

    elif backend == "medgemma":
        if not hasattr(processor, "apply_chat_template"):
            raise RuntimeError("medgemma needs processor.apply_chat_template for image placeholder.")
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": text}],
        }]
        chat_text = processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=False)
        model_inputs = processor(text=chat_text, images=[image], padding=True, return_tensors="pt")

    else:
        model_inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")

    # move tensors
    out = {}
    for k, v in dict(model_inputs).items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


# ----------------------------
# load backend
# ----------------------------
def load_backend(backend: str, model_id: str, device: str):
    import transformers
    from transformers import AutoProcessor

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    if backend in ["medgemma", "internvl"] and device == "cuda":
        torch_dtype = torch.bfloat16

    common = dict(torch_dtype=torch_dtype, device_map=None, low_cpu_mem_usage=True)

    if backend == "qwen3":
        from transformers import Qwen3VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_id)
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **common).to(device).eval()
        return model, processor

    if backend == "lingshu":
        from transformers import Qwen2_5_VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_id)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **common).to(device).eval()
        return model, processor

    if backend == "internvl":
        from transformers import AutoModel, AutoProcessor as AP2
        processor = AP2.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True, **common).to(device).eval()
        return model, processor

    if backend == "medgemma":
        processor = AutoProcessor.from_pretrained(model_id)
        AutoImageTextToText = getattr(transformers, "AutoModelForImageTextToText", None)
        if AutoImageTextToText is not None:
            model = AutoImageTextToText.from_pretrained(model_id, **common).to(device).eval()
            return model, processor
        AutoVision2Seq = getattr(transformers, "AutoModelForVision2Seq", None)
        if AutoVision2Seq is not None:
            model = AutoVision2Seq.from_pretrained(model_id, **common).to(device).eval()
            return model, processor
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_id, **common).to(device).eval()
        return model, processor

    raise ValueError(f"Unknown backend: {backend}")


# ----------------------------
# Hook probing
# ----------------------------
class HookProbe:
    def __init__(self):
        self.calls: Dict[str, int] = {}
        self.outputs: Dict[str, Any] = {}

    def make_hook(self, name: str):
        def hook(module, inputs, output):
            self.calls[name] = self.calls.get(name, 0) + 1
            if name not in self.outputs:
                self.outputs[name] = summarize_output(output)
        return hook


def pick_candidates(model: nn.Module, max_each=30) -> Dict[str, List[str]]:
    # pick module names likely related to vision/projector/adapter/fusion
    modmap = dict(model.named_modules())
    names = [n for n in modmap.keys() if n != ""]

    def filt(keywords):
        picked = []
        for n in names:
            low = n.lower()
            if any(k in low for k in keywords):
                picked.append(n)
                if len(picked) >= max_each:
                    break
        return picked

    return {
        "vision": filt(KW_VISION),
        "projector": filt(KW_PROJECTOR),
        "adapter": filt(KW_ADAPTER),
        "fusion": filt(KW_FUSE),
    }


def run_forward(model: nn.Module, backend: str, inputs: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    # Some models require special dtype for vision input; you had a workaround for internvl
    forward_kwargs = dict(inputs)

    if backend == "internvl" and torch.cuda.is_available():
        # keep vision tensor in fp32 to avoid inf/nan issues
        for k in ["pixel_values", "images", "image", "vision_x"]:
            if k in forward_kwargs and torch.is_tensor(forward_kwargs[k]):
                forward_kwargs[k] = forward_kwargs[k].to(dtype=torch.float32)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            return _forward_try(model, forward_kwargs)
    else:
        return _forward_try(model, forward_kwargs)


def _forward_try(model: nn.Module, forward_kwargs: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    # try with output_hidden_states/attentions and return_dict
    # not all models accept these kwargs
    tried = {}
    for config in [
        dict(output_hidden_states=True, output_attentions=True, return_dict=True),
        dict(output_hidden_states=True, return_dict=True),
        dict(output_hidden_states=True),
        dict(),
    ]:
        try:
            out = model(**forward_kwargs, **config)
            tried["used_kwargs"] = config
            return out, tried
        except TypeError as e:
            tried[str(config)] = f"TypeError: {e}"
        except Exception as e:
            tried[str(config)] = f"{type(e).__name__}: {e}"
            # if forward itself fails for real reasons, stop
            raise
    return model(**forward_kwargs), tried


def extract_top_level_outputs(outputs: Any) -> Dict[str, Any]:
    # summarize common output fields for locating vision/text representations
    info = {"type": type(outputs).__name__}

    def get(obj, name: str):
        if hasattr(obj, name) and getattr(obj, name) is not None:
            return getattr(obj, name)
        if isinstance(obj, dict) and name in obj and obj[name] is not None:
            return obj[name]
        return None

    for k in ["last_hidden_state", "hidden_states", "attentions", "logits"]:
        v = get(outputs, k)
        if v is None:
            continue
        if torch.is_tensor(v):
            info[k] = summarize_output(v)
        elif isinstance(v, (tuple, list)):
            info[k] = {"type": type(v).__name__, "len": len(v)}
            # include last element summary if tensor
            if len(v) > 0 and torch.is_tensor(v[-1]):
                info[k]["last"] = summarize_output(v[-1])
        else:
            info[k] = {"type": type(v).__name__}
    return info


def inspect_backend(backend: str, device: str, depth_tree: int = 3, max_hooks_each: int = 25) -> Dict[str, Any]:
    model_id = MODEL_ID_BY_BACKEND[backend]
    model, processor = load_backend(backend, model_id, device)

    # build dummy inputs
    img = make_dummy_image(224)
    text = (SYSTEM_PROMPT_SHORT + "\n\n" + TASK_PROMPT)
    inputs = build_inputs(backend, processor, img, text, device)

    # structure scan
    tree = print_tree(model, depth=depth_tree)
    vision_like = list_named_modules(model, KW_VISION, max_items=300)
    proj_like   = list_named_modules(model, KW_PROJECTOR, max_items=300)
    fuse_like   = list_named_modules(model, KW_FUSE, max_items=300)
    adapt_like  = list_named_modules(model, KW_ADAPTER, max_items=300)

    # pick hook candidates
    cand = pick_candidates(model, max_each=max_hooks_each)
    modmap = dict(model.named_modules())

    probe = HookProbe()
    handles = []

    def register_many(names: List[str], tag: str):
        ok = []
        for n in names:
            m = modmap.get(n, None)
            if m is None:
                continue
            # skip huge container modules sometimes (optional)
            try:
                h = m.register_forward_hook(probe.make_hook(f"{tag}:{n}"))
                handles.append(h)
                ok.append(n)
            except Exception:
                pass
        return ok

    registered = {
        "vision": register_many(cand["vision"], "vision"),
        "projector": register_many(cand["projector"], "projector"),
        "adapter": register_many(cand["adapter"], "adapter"),
        "fusion": register_many(cand["fusion"], "fusion"),
    }

    # run forward once
    with torch.no_grad():
        outputs, tried = run_forward(model, backend, inputs)

    # remove hooks
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

    # summarize which hooks actually fired
    fired = []
    for name, cnt in sorted(probe.calls.items(), key=lambda x: -x[1]):
        fired.append({
            "name": name,
            "calls": cnt,
            "first_output": probe.outputs.get(name, None),
        })

    # heuristics: do we "see" likely vision encoder / projector?
    # if any fired hook in "vision:*" -> likely vision path is hookable
    vision_hookable = any(x["name"].startswith("vision:") for x in fired)
    projector_hookable = any(x["name"].startswith("projector:") for x in fired)

    report = {
        "backend": backend,
        "model_id": model_id,
        "device": device,
        "dtype_first_param": str(next(model.parameters()).dtype).replace("torch.", ""),
        "tree_depth": depth_tree,
        "tree_top": tree[:200],  # keep small
        "module_hits": {
            "vision_like": vision_like[:200],
            "projector_like": proj_like[:200],
            "fusion_like": fuse_like[:200],
            "adapter_like": adapt_like[:200],
        },
        "hook_candidates_registered": registered,
        "hook_fired_summary": fired[:200],
        "hookable": {
            "vision_path_hookable": bool(vision_hookable),
            "projector_path_hookable": bool(projector_hookable),
        },
        "forward_kwargs_tried": tried,
        "top_level_outputs": extract_top_level_outputs(outputs),
        "notes": {
            "vision_encoder_extractable": (
                "LIKELY" if vision_hookable else
                "UNCERTAIN (no vision-related hooks fired; may be inside custom ops or name heuristics missed)"
            ),
            "projector_extractable": (
                "LIKELY" if projector_hookable else
                "UNCERTAIN (no projector-related hooks fired; module may be named differently)"
            ),
        }
    }

    # cleanup
    del model, processor, outputs, inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return report


# ----------------------------
# main
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", type=str, default="all",
                   choices=["all", "qwen3", "medgemma", "internvl", "lingshu"])
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--depth_tree", type=int, default=3)
    p.add_argument("--max_hooks_each", type=int, default=25)
    p.add_argument("--out_json", type=str, default=None)
    return p.parse_args()


def main():
    set_env_like_you()
    args = parse_args()

    backends = ["qwen3", "medgemma", "internvl", "lingshu"] if args.backend == "all" else [args.backend]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = args.out_json or f"./vlm_hook_report_{run_id}.json"

    all_reports = []
    for b in backends:
        print("\n" + "=" * 80)
        print(f"[Inspect] backend={b} device={args.device}")
        print("=" * 80)
        try:
            rep = inspect_backend(b, device=args.device, depth_tree=args.depth_tree, max_hooks_each=args.max_hooks_each)
            all_reports.append(rep)

            print(f"- dtype(first param): {rep['dtype_first_param']}")
            print(f"- vision_like modules: {len(rep['module_hits']['vision_like'])}")
            print(f"- projector_like modules: {len(rep['module_hits']['projector_like'])}")
            print(f"- hook fired count: {len(rep['hook_fired_summary'])}")
            print(f"- vision hookable: {rep['hookable']['vision_path_hookable']}")
            print(f"- projector hookable: {rep['hookable']['projector_path_hookable']}")
            print(f"- notes: {rep['notes']}")

        except Exception as e:
            print(f"[ERROR] backend={b} -> {type(e).__name__}: {e}")
            all_reports.append({
                "backend": b,
                "model_id": MODEL_ID_BY_BACKEND[b],
                "error": f"{type(e).__name__}: {e}",
            })

    payload = {
        "run_id": run_id,
        "device": args.device,
        "reports": all_reports,
    }
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n[Saved]", out_json)


if __name__ == "__main__":
    main()
