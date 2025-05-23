# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Modified by Jian Ding from: https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py
import fvcore.nn.weight_init as weight_init
import torch
from detectron2.data import MetadataCatalog
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .model import Aggregator
from baselines.third_party import clip
from baselines.third_party import imagenet_templates
import numpy as np
import open_clip


class CATSegPredictor(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        clip_pretrained: str,
        prompt_ensemble_type: str,
        text_guidance_dim: int,
        text_guidance_proj_dim: int,
        appearance_guidance_dim: int,
        appearance_guidance_proj_dim: int,
        prompt_depth: int,
        prompt_length: int,
        decoder_dims: list,
        decoder_guidance_dims: list,
        decoder_guidance_proj_dims: list,
        num_heads: int,
        num_layers: tuple,
        hidden_dims: tuple,
        pooling_sizes: tuple,
        feature_resolution: tuple,
        window_sizes: tuple,
        attention_type: str,
        train_dataset: str,
        test_dataset: str,
        bg_on: bool,
        prompt_learner: None,
    ):
        """
        Args:

        """
        super().__init__()

        self.train_class_texts = MetadataCatalog.get(train_dataset).stuff_classes
        self.train_obj_classes = MetadataCatalog.get(train_dataset).obj_classes

        self.test_class_texts = MetadataCatalog.get(test_dataset).stuff_classes
        self.test_obj_classes = MetadataCatalog.get(test_dataset).obj_classes

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = None
        if clip_pretrained == "ViT-G" or clip_pretrained == "ViT-H":
            # for OpenCLIP models
            name, pretrain = ('ViT-H-14', 'laion2b_s32b_b79k') if clip_pretrained == 'ViT-H' else (
                'ViT-bigG-14', 'laion2b_s39b_b160k')
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
                name,
                pretrained=pretrain,
                device=device,
                force_image_size=336,)

            self.tokenizer = open_clip.get_tokenizer(name)
        else:
            # for OpenAI models
            clip_model, clip_preprocess = clip.load(
                clip_pretrained, device=device, jit=False, prompt_depth=prompt_depth, prompt_length=prompt_length)

        self.prompt_ensemble_type = prompt_ensemble_type

        if self.prompt_ensemble_type == "imagenet_select":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES_SELECT
        elif self.prompt_ensemble_type == "imagenet":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES
        elif self.prompt_ensemble_type == "single":
            prompt_templates = ['A photo of a {} in the scene',]
        else:
            raise NotImplementedError
        self.bg_on = bg_on

        if self.bg_on:
            self.non_object_embedding = nn.Parameter(
                torch.empty(1, clip_model.text_projection.shape[-1])
            )
            nn.init.normal_(
                self.non_object_embedding.data,
                std=clip_model.transformer.width ** -0.5,
            )
        self.text_features = self.class_embeddings(
            self.train_class_texts, prompt_templates, clip_model).permute(1, 0, 2).float()
        self.text_features_test = self.class_embeddings(
            self.test_class_texts, prompt_templates, clip_model).permute(1, 0, 2).float()

        self.clip_model = clip_model.float()
        self.clip_preprocess = clip_preprocess

        transformer = Aggregator(
            text_guidance_dim=text_guidance_dim,
            text_guidance_proj_dim=text_guidance_proj_dim,
            appearance_guidance_dim=appearance_guidance_dim,
            appearance_guidance_proj_dim=appearance_guidance_proj_dim,
            decoder_dims=decoder_dims,
            decoder_guidance_dims=decoder_guidance_dims,
            decoder_guidance_proj_dims=decoder_guidance_proj_dims,
            num_layers=num_layers,
            nheads=num_heads,
            hidden_dim=hidden_dims,
            pooling_size=pooling_sizes,
            feature_resolution=feature_resolution,
            window_size=window_sizes,
            attention_type=attention_type,
            prompt_channel=len(prompt_templates)
        )
        self.transformer = transformer

    @classmethod
    def from_config(cls, cfg):  # , in_channels, mask_classification):
        ret = {}

        ret["clip_pretrained"] = cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED
        ret["prompt_ensemble_type"] = cfg.MODEL.PROMPT_ENSEMBLE_TYPE

        # Aggregator parameters:
        ret["text_guidance_dim"] = cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM
        ret["text_guidance_proj_dim"] = cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_PROJ_DIM
        ret["appearance_guidance_dim"] = cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_DIM
        ret["appearance_guidance_proj_dim"] = cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_PROJ_DIM

        ret["decoder_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_DIMS
        ret["decoder_guidance_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_DIMS
        ret["decoder_guidance_proj_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_PROJ_DIMS

        ret["prompt_depth"] = cfg.MODEL.SEM_SEG_HEAD.PROMPT_DEPTH
        ret["prompt_length"] = cfg.MODEL.SEM_SEG_HEAD.PROMPT_LENGTH

        ret["num_layers"] = cfg.MODEL.SEM_SEG_HEAD.NUM_LAYERS
        ret["num_heads"] = cfg.MODEL.SEM_SEG_HEAD.NUM_HEADS
        ret["hidden_dims"] = cfg.MODEL.SEM_SEG_HEAD.HIDDEN_DIMS
        ret["pooling_sizes"] = cfg.MODEL.SEM_SEG_HEAD.POOLING_SIZES
        ret["feature_resolution"] = cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION
        ret["window_sizes"] = cfg.MODEL.SEM_SEG_HEAD.WINDOW_SIZES
        ret["attention_type"] = cfg.MODEL.SEM_SEG_HEAD.ATTENTION_TYPE
        ret["train_dataset"] = cfg.DATASETS.TRAIN[0]
        ret["test_dataset"] = cfg.DATASETS.TEST[0]
        ret["bg_on"] = cfg.MODEL.SEM_SEG_HEAD.BG_ON
        ret["prompt_learner"] = None
        return ret

    def get_class_name_list(self, dataset_name):
        class_names = [
            c.strip() for c in MetadataCatalog.get(dataset_name).stuff_classes
        ]
        return class_names

    def forward(self, x, vis_guidance, test_text=None):
        vis = [vis_guidance[k] for k in vis_guidance.keys()][::-1]

        if self.training:
            text = self.text_features
        elif test_text is not None:
            text = self.text_features_test[[
                self.test_class_texts.index(t) for t in test_text]]
        else:
            text = self.text_features_test
        text = text.repeat(x.shape[0], 1, 1, 1)
        out = self.transformer(x, text, vis)
        return out

    @torch.no_grad()
    def class_embeddings(self, classnames, templates, clip_model):
        zeroshot_weights = []
        for classname in classnames:
            if ', ' in classname:
                classname_splits = classname.split(', ')
                texts = []
                for template in templates:
                    for cls_split in classname_splits:
                        texts.append(template.format(cls_split))
            else:
                # format with class
                texts = [template.format(classname) for template in templates]

            if self.tokenizer is not None:
                texts = self.tokenizer(texts).cuda()
            else:
                texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            if len(templates) != class_embeddings.shape[0]:
                class_embeddings = class_embeddings.reshape(
                    len(templates), -1, class_embeddings.shape[-1]).mean(dim=1)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        if self.bg_on:
            non_object_text_features = (
                self.non_object_embedding
                / self.non_object_embedding.norm(dim=-1, keepdim=True)
            )
            zeroshot_weights = torch.cat([zeroshot_weights, non_object_text_features.unsqueeze(
                0).repeat(len(zeroshot_weights), 1, 1).cuda()], dim=-2)
        return zeroshot_weights
