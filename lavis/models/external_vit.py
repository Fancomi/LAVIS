"""
External vision encoder wrappers for BLIP2 pipeline.

Supported: SigLIP2, DINOv3, C-RADIOv4

Each wrapper exposes the EVA-ViT compatible interface expected by Blip2Base:
  encoder(x: [B,3,H,W]) -> [B, N, C]   patch tokens only
  encoder.num_features: int             hidden dim C
  encoder.get_num_layer(name) -> int    layer index for layerwise LR decay
"""
import importlib.util
import logging
import os
import sys

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

log = logging.getLogger(__name__)

_DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def _strip_hooks(module: nn.Module) -> None:
    """Remove accelerate dispatch hooks so LAVIS can call .to(device) normally."""
    try:
        from accelerate.hooks import remove_hook_from_module
        remove_hook_from_module(module, recurse=True)
    except ImportError:
        pass


# ─── shared base ──────────────────────────────────────────────────────────────

class _ExternalViTBase(nn.Module):
    num_features: int
    _depth: int

    def get_num_layer(self, var_name: str = "") -> int:
        """Layer index for layerwise LR decay (mirrors eva_vit convention)."""
        for part in var_name.split("."):
            if part.isdigit():
                return int(part)
        if any(k in var_name for k in ("head", "norm", "ln_post", "layernorm")):
            return self._depth
        return 0


# ─── SigLIP2 ──────────────────────────────────────────────────────────────────

class SigLIP2Encoder(_ExternalViTBase):
    """
    google/siglip2-so400m-patch14-384
    Output: [B, 729, 1152]  (no CLS token, all patch tokens)
    """
    def __init__(self, model_path: str, precision: str = "fp16"):
        super().__init__()
        dtype = _DTYPE.get(precision, torch.float16)
        cfg = AutoConfig.from_pretrained(model_path)
        full_model = AutoModel.from_pretrained(
            model_path, torch_dtype=dtype, device_map={"": "cpu"}
        )
        _strip_hooks(full_model)
        self.encoder    = full_model.vision_model
        self.num_features = cfg.vision_config.hidden_size    # 1152
        self._depth       = cfg.vision_config.num_hidden_layers  # 27
        log.info(f"[SigLIP2] num_features={self.num_features}, depth={self._depth}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # last_hidden_state: [B, N, C]  — SigLIP2 has no CLS token
        return self.encoder(pixel_values=x).last_hidden_state


# ─── DINOv3 ───────────────────────────────────────────────────────────────────

class DINOv3Encoder(_ExternalViTBase):
    """
    facebook/dinov3-vitl16-pretrain-lvd1689m
    Tokens: [CLS, reg×4, patch×196] -> strip prefix, return patch tokens only
    Output: [B, 196, 1024]
    """
    def __init__(self, model_path: str, precision: str = "fp16"):
        super().__init__()
        import json
        cfg = json.load(open(os.path.join(model_path, "config.json")))
        # DINOv2/v3 attention is numerically unstable in fp16; always load fp32
        self.encoder = AutoModel.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.float32, device_map={"": "cpu"}
        )
        _strip_hooks(self.encoder)
        self.num_features   = cfg["hidden_size"]        # 1024
        self._depth         = cfg["num_hidden_layers"]  # 24
        self._num_prefix    = 1 + cfg.get("num_register_tokens", 0)  # CLS + regs = 5
        log.info(
            f"[DINOv3] num_features={self.num_features}, depth={self._depth}, "
            f"prefix_tokens={self._num_prefix}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Disable outer fp16 autocast: DINOv2/v3 overflows in fp16
        with torch.autocast(device_type="cuda", enabled=False):
            tokens = self.encoder(pixel_values=x.float()).last_hidden_state
        return tokens[:, self._num_prefix:, :]


# ─── C-RADIOv4 ────────────────────────────────────────────────────────────────

class RadioEncoder(_ExternalViTBase):
    """
    C-RADIOv4-SO400M (NVIDIA RADIO)
    Uses local package import; input expected in [0,1], conditioner is internal.
    Output: [B, N, 1152]  (patch features from RadioOutput.features)
    """
    def __init__(self, model_path: str, precision: str = "fp16"):
        super().__init__()
        dtype = _DTYPE.get(precision, torch.float16)

        # Load as a package: insert parent dir, treat folder as a module
        parent   = os.path.dirname(model_path)
        pkg_name = os.path.basename(model_path).replace("-", "_")
        if parent not in sys.path:
            sys.path.insert(0, parent)

        spec = importlib.util.spec_from_file_location(
            pkg_name,
            os.path.join(model_path, "hf_model.py"),
            submodule_search_locations=[model_path],
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[pkg_name] = mod
        spec.loader.exec_module(mod)

        hf_model = mod.RADIOModel.from_pretrained(
            model_path, torch_dtype=dtype, device_map={"": "cpu"}
        )
        _strip_hooks(hf_model)
        self.radio        = hf_model.radio_model
        self.num_features = self.radio.embed_dim   # 1152
        self._depth       = (
            len(list(self.radio.model.blocks))
            if hasattr(self.radio.model, "blocks") else 27
        )
        log.info(
            f"[C-RADIOv4] num_features={self.num_features}, depth={self._depth}, "
            f"preferred_res={self.radio.preferred_resolution}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.radio(x)
        # out may be RadioOutput or dict (when adaptors are present)
        if isinstance(out, dict):
            out = out["backbone"]
        return out.features   # [B, N, C]


# ─── factory functions ────────────────────────────────────────────────────────

_MODEL_PATHS = {
    "siglip2":      "/root/paddlejob/workspace/env_run/penghaotian/models/siglip2-so400m-patch14-384",
    "dinov3":       "/root/paddlejob/workspace/env_run/penghaotian/models/dinov3-vitl16-pretrain-lvd1689m",
    "radio_so400m": "/root/paddlejob/workspace/env_run/penghaotian/models/C-RADIOv4/C-RADIOv4-SO400M",
}

def create_siglip2(precision: str = "fp16") -> SigLIP2Encoder:
    return SigLIP2Encoder(_MODEL_PATHS["siglip2"], precision=precision)

def create_dinov3(precision: str = "fp16") -> DINOv3Encoder:
    return DINOv3Encoder(_MODEL_PATHS["dinov3"], precision=precision)

def create_radio(precision: str = "fp16") -> RadioEncoder:
    return RadioEncoder(_MODEL_PATHS["radio_so400m"], precision=precision)
