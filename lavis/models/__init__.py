"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import torch
from omegaconf import OmegaConf
from lavis.common.registry import registry

from lavis.models.base_model import BaseModel

# ── BLIP2 core (required) ────────────────────────────────────
from lavis.models.blip2_models.blip2 import Blip2Base
from lavis.models.blip2_models.blip2_opt import Blip2OPT
from lavis.models.blip2_models.blip2_t5 import Blip2T5
from lavis.models.blip2_models.blip2_qformer import Blip2Qformer
from lavis.models.blip2_models.blip2_image_text_matching import Blip2ITM

# ── Optional models (graceful degradation on missing deps) ───
def _try_import(import_fn, name):
    try:
        return import_fn()
    except (ImportError, ModuleNotFoundError) as e:
        logging.getLogger(__name__).debug(f"Optional model {name} not loaded: {e}")
        return None

AlbefClassification    = _try_import(lambda: __import__('lavis.models.albef_models.albef_classification',    fromlist=['AlbefClassification']).AlbefClassification,    'AlbefClassification')
AlbefFeatureExtractor  = _try_import(lambda: __import__('lavis.models.albef_models.albef_feature_extractor',  fromlist=['AlbefFeatureExtractor']).AlbefFeatureExtractor,  'AlbefFeatureExtractor')
AlbefNLVR              = _try_import(lambda: __import__('lavis.models.albef_models.albef_nlvr',              fromlist=['AlbefNLVR']).AlbefNLVR,                          'AlbefNLVR')
AlbefPretrain          = _try_import(lambda: __import__('lavis.models.albef_models.albef_pretrain',          fromlist=['AlbefPretrain']).AlbefPretrain,                  'AlbefPretrain')
AlbefRetrieval         = _try_import(lambda: __import__('lavis.models.albef_models.albef_retrieval',         fromlist=['AlbefRetrieval']).AlbefRetrieval,                'AlbefRetrieval')
AlbefVQA               = _try_import(lambda: __import__('lavis.models.albef_models.albef_vqa',               fromlist=['AlbefVQA']).AlbefVQA,                            'AlbefVQA')
AlproQA                = _try_import(lambda: __import__('lavis.models.alpro_models.alpro_qa',                fromlist=['AlproQA']).AlproQA,                              'AlproQA')
AlproRetrieval         = _try_import(lambda: __import__('lavis.models.alpro_models.alpro_retrieval',         fromlist=['AlproRetrieval']).AlproRetrieval,                'AlproRetrieval')
BlipBase               = _try_import(lambda: __import__('lavis.models.blip_models.blip',                     fromlist=['BlipBase']).BlipBase,                            'BlipBase')
BlipCaption            = _try_import(lambda: __import__('lavis.models.blip_models.blip_caption',             fromlist=['BlipCaption']).BlipCaption,                      'BlipCaption')
BlipClassification     = _try_import(lambda: __import__('lavis.models.blip_models.blip_classification',      fromlist=['BlipClassification']).BlipClassification,        'BlipClassification')
BlipFeatureExtractor   = _try_import(lambda: __import__('lavis.models.blip_models.blip_feature_extractor',   fromlist=['BlipFeatureExtractor']).BlipFeatureExtractor,    'BlipFeatureExtractor')
BlipITM                = _try_import(lambda: __import__('lavis.models.blip_models.blip_image_text_matching', fromlist=['BlipITM']).BlipITM,                              'BlipITM')
BlipNLVR               = _try_import(lambda: __import__('lavis.models.blip_models.blip_nlvr',                fromlist=['BlipNLVR']).BlipNLVR,                            'BlipNLVR')
BlipPretrain           = _try_import(lambda: __import__('lavis.models.blip_models.blip_pretrain',            fromlist=['BlipPretrain']).BlipPretrain,                    'BlipPretrain')
BlipRetrieval          = _try_import(lambda: __import__('lavis.models.blip_models.blip_retrieval',           fromlist=['BlipRetrieval']).BlipRetrieval,                  'BlipRetrieval')
BlipVQA                = _try_import(lambda: __import__('lavis.models.blip_models.blip_vqa',                 fromlist=['BlipVQA']).BlipVQA,                              'BlipVQA')
Blip2T5Instruct        = _try_import(lambda: __import__('lavis.models.blip2_models.blip2_t5_instruct',       fromlist=['Blip2T5Instruct']).Blip2T5Instruct,              'Blip2T5Instruct')
Blip2VicunaInstruct    = _try_import(lambda: __import__('lavis.models.blip2_models.blip2_vicuna_instruct',   fromlist=['Blip2VicunaInstruct']).Blip2VicunaInstruct,      'Blip2VicunaInstruct')
Blip2VicunaXInstruct   = _try_import(lambda: __import__('lavis.models.blip2_models.blip2_vicuna_xinstruct',  fromlist=['Blip2VicunaXInstruct']).Blip2VicunaXInstruct,    'Blip2VicunaXInstruct')
BlipDiffusion          = _try_import(lambda: __import__('lavis.models.blip_diffusion_models.blip_diffusion', fromlist=['BlipDiffusion']).BlipDiffusion,                  'BlipDiffusion')
PNPVQA                 = _try_import(lambda: __import__('lavis.models.pnp_vqa_models.pnp_vqa',               fromlist=['PNPVQA']).PNPVQA,                                'PNPVQA')
PNPUnifiedQAv2FiD      = _try_import(lambda: __import__('lavis.models.pnp_vqa_models.pnp_unifiedqav2_fid',   fromlist=['PNPUnifiedQAv2FiD']).PNPUnifiedQAv2FiD,          'PNPUnifiedQAv2FiD')
Img2PromptVQA          = _try_import(lambda: __import__('lavis.models.img2prompt_models.img2prompt_vqa',     fromlist=['Img2PromptVQA']).Img2PromptVQA,                  'Img2PromptVQA')
XBertLMHeadDecoder     = _try_import(lambda: __import__('lavis.models.med',                                  fromlist=['XBertLMHeadDecoder']).XBertLMHeadDecoder,        'XBertLMHeadDecoder')
VisionTransformerEncoder = _try_import(lambda: __import__('lavis.models.vit',                                fromlist=['VisionTransformerEncoder']).VisionTransformerEncoder, 'VisionTransformerEncoder')
CLIP                   = _try_import(lambda: __import__('lavis.models.clip_models.model',                    fromlist=['CLIP']).CLIP,                                    'CLIP')
GPTDialogue            = _try_import(lambda: __import__('lavis.models.gpt_models.gpt_dialogue',              fromlist=['GPTDialogue']).GPTDialogue,                      'GPTDialogue')

from lavis.processors.base_processor import BaseProcessor


__all__ = [
    "load_model",
    "AlbefClassification",
    "AlbefFeatureExtractor",
    "AlbefNLVR",
    "AlbefVQA",
    "AlbefPretrain",
    "AlbefRetrieval",
    "AlproQA",
    "AlproRetrieval",
    "BaseModel",
    "BlipBase",
    "BlipFeatureExtractor",
    "BlipCaption",
    "BlipClassification",
    "BlipDiffusion",
    "BlipITM",
    "BlipNLVR",
    "BlipPretrain",
    "BlipRetrieval",
    "BlipVQA",
    "Blip2Qformer",
    "Blip2Base",
    "Blip2ITM",
    "Blip2OPT",
    "Blip2T5",
    "Blip2T5Instruct",
    "Blip2VicunaInstruct",
    "Blip2VicunaXInstruct",
    "PNPVQA",
    "Img2PromptVQA",
    "PNPUnifiedQAv2FiD",
    "CLIP",
    "VisionTransformerEncoder",
    "XBertLMHeadDecoder",
    "GPTDialogue",
]


def load_model(name, model_type, is_eval=False, device="cpu", checkpoint=None):
    """
    Load supported models.

    To list all available models and types in registry:
    >>> from lavis.models import model_zoo
    >>> print(model_zoo)

    Args:
        name (str): name of the model.
        model_type (str): type of the model.
        is_eval (bool): whether the model is in eval mode. Default: False.
        device (str): device to use. Default: "cpu".
        checkpoint (str): path or to checkpoint. Default: None.
            Note that expecting the checkpoint to have the same keys in state_dict as the model.

    Returns:
        model (torch.nn.Module): model.
    """

    model = registry.get_model_class(name).from_pretrained(model_type=model_type)

    if checkpoint is not None:
        model.load_checkpoint(checkpoint)

    if is_eval:
        model.eval()

    if device == "cpu":
        model = model.float()

    return model.to(device)


def load_preprocess(config):
    """
    Load preprocessor configs and construct preprocessors.

    If no preprocessor is specified, return BaseProcessor, which does not do any preprocessing.

    Args:
        config (dict): preprocessor configs.

    Returns:
        vis_processors (dict): preprocessors for visual inputs.
        txt_processors (dict): preprocessors for text inputs.

        Key is "train" or "eval" for processors used in training and evaluation respectively.
    """

    def _build_proc_from_cfg(cfg):
        return (
            registry.get_processor_class(cfg.name).from_config(cfg)
            if cfg is not None
            else BaseProcessor()
        )

    vis_processors = dict()
    txt_processors = dict()

    vis_proc_cfg = config.get("vis_processor")
    txt_proc_cfg = config.get("text_processor")

    if vis_proc_cfg is not None:
        vis_train_cfg = vis_proc_cfg.get("train")
        vis_eval_cfg = vis_proc_cfg.get("eval")
    else:
        vis_train_cfg = None
        vis_eval_cfg = None

    vis_processors["train"] = _build_proc_from_cfg(vis_train_cfg)
    vis_processors["eval"] = _build_proc_from_cfg(vis_eval_cfg)

    if txt_proc_cfg is not None:
        txt_train_cfg = txt_proc_cfg.get("train")
        txt_eval_cfg = txt_proc_cfg.get("eval")
    else:
        txt_train_cfg = None
        txt_eval_cfg = None

    txt_processors["train"] = _build_proc_from_cfg(txt_train_cfg)
    txt_processors["eval"] = _build_proc_from_cfg(txt_eval_cfg)

    return vis_processors, txt_processors


def load_model_and_preprocess(name, model_type, is_eval=False, device="cpu"):
    """
    Load model and its related preprocessors.

    List all available models and types in registry:
    >>> from lavis.models import model_zoo
    >>> print(model_zoo)

    Args:
        name (str): name of the model.
        model_type (str): type of the model.
        is_eval (bool): whether the model is in eval mode. Default: False.
        device (str): device to use. Default: "cpu".

    Returns:
        model (torch.nn.Module): model.
        vis_processors (dict): preprocessors for visual inputs.
        txt_processors (dict): preprocessors for text inputs.
    """
    model_cls = registry.get_model_class(name)

    # load model
    model = model_cls.from_pretrained(model_type=model_type)

    if is_eval:
        model.eval()

    # load preprocess
    cfg = OmegaConf.load(model_cls.default_config_path(model_type))
    if cfg is not None:
        preprocess_cfg = cfg.preprocess

        vis_processors, txt_processors = load_preprocess(preprocess_cfg)
    else:
        vis_processors, txt_processors = None, None
        logging.info(
            f"""No default preprocess for model {name} ({model_type}).
                This can happen if the model is not finetuned on downstream datasets,
                or it is not intended for direct use without finetuning.
            """
        )

    if device == "cpu" or device == torch.device("cpu"):
        model = model.float()

    return model.to(device), vis_processors, txt_processors


class ModelZoo:
    """
    A utility class to create string representation of available model architectures and types.

    >>> from lavis.models import model_zoo
    >>> # list all available models
    >>> print(model_zoo)
    >>> # show total number of models
    >>> print(len(model_zoo))
    """

    def __init__(self) -> None:
        self.model_zoo = {
            k: list(v.PRETRAINED_MODEL_CONFIG_DICT.keys())
            for k, v in registry.mapping["model_name_mapping"].items()
        }

    def __str__(self) -> str:
        return (
            "=" * 50
            + "\n"
            + f"{'Architectures':<30} {'Types'}\n"
            + "=" * 50
            + "\n"
            + "\n".join(
                [
                    f"{name:<30} {', '.join(types)}"
                    for name, types in self.model_zoo.items()
                ]
            )
        )

    def __iter__(self):
        return iter(self.model_zoo.items())

    def __len__(self):
        return sum([len(v) for v in self.model_zoo.values()])


model_zoo = ModelZoo()
