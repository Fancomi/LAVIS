#!/usr/bin/env python3
"""
Diagnose 'Cannot copy out of meta tensor' by loading each component
independently and checking for meta-device parameters.

Usage:
    python check_meta.py [siglip2|dinov3|radio|qformer|bert|blip2|all] [vit_name]
    python check_meta.py blip2 siglip2
    python check_meta.py blip2 dinov3
    python check_meta.py blip2 radio_so400m
"""
import sys
import argparse
import torch
import torch.nn as nn

LAVIS = "/root/paddlejob/workspace/env_run/penghaotian/vision_encoders/LAVIS"
BERT  = "/root/paddlejob/workspace/env_run/penghaotian/models/google-bert/bert-base-uncased"
if LAVIS not in sys.path:
    sys.path.insert(0, LAVIS)


# ── helpers ────────────────────────────────────────────────────────────────────

def meta_report(model: nn.Module, tag: str) -> bool:
    meta = [(n, tuple(p.shape)) for n, p in model.named_parameters() if p.is_meta]
    buf  = [(n, tuple(b.shape)) for n, b in model.named_buffers()    if b.is_meta]
    if meta or buf:
        print(f"[{tag}] FAIL — {len(meta)} meta params, {len(buf)} meta buffers")
        for n, s in (meta + buf)[:5]:
            print(f"         {n}: {s}")
        if len(meta) + len(buf) > 5:
            print(f"         ... and {len(meta)+len(buf)-5} more")
        return False
    print(f"[{tag}] OK — no meta tensors")
    return True


def to_device_test(model: nn.Module, tag: str, device: str = "cuda:0") -> bool:
    try:
        model.to(device)
        print(f"[{tag}] .to({device}) OK")
        return True
    except Exception as e:
        print(f"[{tag}] .to({device}) FAIL: {e}")
        return False


# ── individual tests ───────────────────────────────────────────────────────────

def test_siglip2():
    print("\n=== SigLIP2 ===")
    from lavis.models.external_vit import create_siglip2
    enc = create_siglip2("fp16")
    ok = meta_report(enc, "SigLIP2-after-init")
    if ok and torch.cuda.is_available():
        to_device_test(enc, "SigLIP2")


def test_dinov3():
    print("\n=== DINOv3 ===")
    from lavis.models.external_vit import create_dinov3
    enc = create_dinov3("fp16")
    ok = meta_report(enc, "DINOv3-after-init")
    if ok and torch.cuda.is_available():
        to_device_test(enc, "DINOv3")


def test_radio():
    print("\n=== C-RADIOv4 ===")
    from lavis.models.external_vit import create_radio
    enc = create_radio("fp16")
    ok = meta_report(enc, "RADIO-after-init")
    if ok and torch.cuda.is_available():
        to_device_test(enc, "RADIO")


def test_bert():
    print("\n=== Raw BertModel ===")
    from transformers import BertModel
    model = BertModel.from_pretrained(BERT)
    ok = meta_report(model, "BertModel-raw")
    if ok and torch.cuda.is_available():
        to_device_test(model, "BertModel")


def test_qformer():
    print("\n=== Qformer (BERT backbone) ===")
    from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
    encoder_config = BertConfig.from_pretrained(BERT)
    encoder_config.encoder_width = 768
    encoder_config.add_cross_attention = True
    encoder_config.cross_attention_freq = 2
    encoder_config.query_length = 32
    qformer = BertLMHeadModel.from_pretrained(BERT, config=encoder_config)
    ok = meta_report(qformer, "Qformer-after-init")
    if ok and torch.cuda.is_available():
        to_device_test(qformer, "Qformer")


def test_blip2_stage1(vit: str = "siglip2"):
    """
    Instantiate full Blip2Qformer (Stage 1) then .to('cuda:0') —
    exactly what runner_base.py:86 does.
    """
    print(f"\n=== Full BLIP2 Stage1 (vit={vit}) ===")
    from lavis.models.blip2_models.blip2_qformer import Blip2Qformer
    model = Blip2Qformer(
        vit_model=vit,
        img_size=384,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
    )
    ok = meta_report(model, f"Blip2Qformer[{vit}]-after-init")
    if ok and torch.cuda.is_available():
        to_device_test(model, f"Blip2Qformer[{vit}]")


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", nargs="?", default="all",
                        choices=["siglip2", "dinov3", "radio", "qformer",
                                 "bert", "blip2", "all"])
    parser.add_argument("vit", nargs="?", default="siglip2",
                        help="vit name for blip2 target (siglip2|dinov3|radio_so400m)")
    args = parser.parse_args()

    dispatch = {
        "siglip2": test_siglip2,
        "dinov3":  test_dinov3,
        "radio":   test_radio,
        "bert":    test_bert,
        "qformer": test_qformer,
    }

    if args.target == "all":
        for fn in dispatch.values():
            try:
                fn()
            except Exception as e:
                print(f"  [ERROR] {e}")
        for vit in ("siglip2", "dinov3", "radio_so400m"):
            try:
                test_blip2_stage1(vit)
            except Exception as e:
                print(f"  [ERROR blip2/{vit}] {e}")
    elif args.target == "blip2":
        try:
            test_blip2_stage1(args.vit)
        except Exception as e:
            print(f"  [ERROR] {e}")
    else:
        try:
            dispatch[args.target]()
        except Exception as e:
            print(f"  [ERROR] {e}")

    print("\nDone.")
