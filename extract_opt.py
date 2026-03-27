import json, shutil, os
from pathlib import Path
from safetensors.torch import load_file, save_file

src = Path("/root/paddlejob/workspace/env_run/penghaotian/models/blip2-opt-2.7b")
dst = Path("/root/paddlejob/workspace/env_run/penghaotian/models/opt-2.7b")
dst.mkdir(exist_ok=True)

# 读取 shard index
index = json.loads((src / "model.safetensors.index.json").read_text())
weight_map = index["weight_map"]

# 找出哪些 shard 包含 language_model 权重
lm_shards = set(v for k, v in weight_map.items() if k.startswith("language_model."))
print(f"Language model shards: {lm_shards}")

# 提取并去掉 "language_model." 前缀
lm_weights = {}
for shard in lm_shards:
    print(f"Loading {shard}...")
    tensors = load_file(src / shard)
    for k, v in tensors.items():
        if k.startswith("language_model."):
            lm_weights[k[len("language_model."):]] = v

print(f"Saving {len(lm_weights)} tensors...")
save_file(lm_weights, dst / "model.safetensors", metadata={"format": "pt"})

# 复制 tokenizer 相关文件
for f in ["tokenizer.json", "tokenizer_config.json", "vocab.json",
          "merges.txt", "special_tokens_map.json"]:
    if (src / f).exists():
        shutil.copy(src / f, dst / f)

# 写 config.json（取 text_config 部分）
full_cfg = json.loads((src / "config.json").read_text())
text_cfg = full_cfg.get("text_config", {})
text_cfg["architectures"] = ["OPTForCausalLM"]
text_cfg["model_type"] = "opt"
(dst / "config.json").write_text(json.dumps(text_cfg, indent=2))

print(f"Done -> {dst}")
