"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
from lavis.common.registry import registry
from lavis.datasets.data_utils import prepare_sample
from lavis.tasks.base_task import BaseTask


@registry.register_task("image_text_pretrain")
class ImageTextPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()

    def evaluation(self, model, data_loader, cuda_enabled=True):
        total_loss = 0.0
        num_batches = 0
        for samples in data_loader:
            if "text_input" not in samples:
                continue
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            with torch.cuda.amp.autocast():
                out = model(samples)
                loss = out["loss"] if isinstance(out, dict) else out
            total_loss += loss.item()
            num_batches += 1
        if num_batches == 0:
            # val split 不含文本（纯图像），用固定占位值触发 checkpoint_best 保存
            return {"val_loss": 0.0, "agg_metrics": 1.0}
        avg_loss = total_loss / num_batches
        return {"val_loss": round(avg_loss, 6), "agg_metrics": -avg_loss}

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        return val_result
