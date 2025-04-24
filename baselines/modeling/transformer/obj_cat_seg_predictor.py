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

from .obj_cat_seg_model import Aggregator
from baselines.third_party import clip
from baselines.third_party import imagenet_templates
from collections import defaultdict
import numpy as np
import open_clip


class ObjCATSegPredictor(nn.Module):
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

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.ignore_label = MetadataCatalog.get(test_dataset).ignore_label
        self.init_metadata(train_dataset, "train")
        self.init_metadata(test_dataset, "test")

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

        self.text_part_obj_features = self.class_embeddings(
            self.train_part_classes + self.train_obj_in_part_classes, 
            prompt_templates, clip_model
        ).permute(1, 0, 2).float()

        self.text_obj_features = self.class_embeddings(
            self.train_obj_in_part_classes,
            prompt_templates, clip_model
        ).permute(1, 0, 2).float()

        self.text_specific_part_features = self.class_embeddings(
            self.train_ori_text_classes,
            prompt_templates, clip_model
        ).permute(1, 0, 2).float()

        self.text_part_obj_features_test = self.class_embeddings(
            self.test_part_classes + self.test_obj_in_part_classes,
            prompt_templates, clip_model
        ).permute(1, 0, 2).float()

        self.text_obj_features_test = self.class_embeddings(
            self.test_obj_in_part_classes,
            prompt_templates, clip_model
        ).permute(1, 0, 2).float()

        self.text_specific_part_features_test = self.class_embeddings(
            self.test_ori_text_classes,
            prompt_templates, clip_model
        ).permute(1, 0, 2).float()

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

    def init_metadata(self, dataset_name, prefix):
        text_classes = MetadataCatalog.get(dataset_name).stuff_classes
        setattr(self, f"{prefix}_ori_text_classes", text_classes)

        # Create generalized and object-specific classes
        setattr(self, f"{prefix}_text_classes", [c.replace("'s", "") for c in text_classes])
        setattr(self, f"{prefix}_obj_classes", MetadataCatalog.get(dataset_name).obj_classes)
        part_classes = sorted(list(set([c.split("'s")[1].strip() for c in text_classes])))
        setattr(self, f"{prefix}_part_classes", part_classes)
        obj_in_part_classes = sorted(list(set([c.split("'s")[0].strip() for c in text_classes])))
        setattr(self, f"{prefix}_obj_in_part_classes", obj_in_part_classes)

        # Maps for text to part, object to object-in-part, etc.
        text_to_part_map = torch.full(
            (self.ignore_label + 1,),
            self.ignore_label,
            dtype=torch.long,
            device=self.device
        )
        setattr(self, f"{prefix}_text_to_part_map", text_to_part_map)

        obj_to_obj_in_part_map = torch.full(
            (self.ignore_label + 1,),
            self.ignore_label,
            dtype=torch.long,
            device=self.device
        )
        setattr(self, f"{prefix}_obj_to_obj_in_part_map", obj_to_obj_in_part_map)

        text_to_obj_in_part_map = torch.full(
            (len(getattr(self, f"{prefix}_text_classes")),),
            len(getattr(self, f"{prefix}_text_classes")),
            dtype=torch.long,
            device=self.device
        )
        setattr(self, f"{prefix}_text_to_obj_in_part_map", text_to_obj_in_part_map)

        # Create class to part and object to object-in-part mappings
        class_to_part = {
            index: part_classes.index(class_text.split("'s")[1].strip())
            for index, class_text in enumerate(text_classes)
        }
        setattr(self, f"{prefix}_class_to_part", class_to_part)
        for index, part_index in class_to_part.items():
            getattr(self, f"{prefix}_text_to_part_map")[index] = part_index

        obj_to_obj_in_part = {}
        for index, class_text in enumerate(getattr(self, f"{prefix}_obj_classes")):
            if class_text in obj_in_part_classes:
                obj_to_obj_in_part[index] = obj_in_part_classes.index(class_text)
            else:
                obj_to_obj_in_part[index] = self.ignore_label
        setattr(self, f"{prefix}_obj_to_obj_in_part", obj_to_obj_in_part)
        for index, obj_in_part_index in obj_to_obj_in_part.items():
            getattr(self, f"{prefix}_obj_to_obj_in_part_map")[index] = obj_in_part_index

        # Create object-in-part to text mapping
        obj_in_part_to_text = defaultdict(list)
        for index, class_text in enumerate(text_classes):
            obj_class, _ = class_text.split("'s", maxsplit=1)
            obj_in_part_index = obj_in_part_classes.index(obj_class)
            obj_in_part_to_text[obj_in_part_index].append(index)
            getattr(self, f"{prefix}_text_to_obj_in_part_map")[index] = obj_in_part_index
        setattr(self, f"{prefix}_obj_in_part_to_text", obj_in_part_to_text)

        return None

    def forward(self, x, vis_guidance, test_text=None):

        # TODO: 2024.11.04
        # vis_guidance
        vis = [vis_guidance[k] for k in vis_guidance.keys()][::-1]
        # vis[0].shape: torch.Size([4, 1024, 24, 24])
        # vis[1].shape: torch.Size([4, 512, 48, 48])
        # vis[2].shape: torch.Size([4, 256, 96, 96])

        if self.training:
            part_obj_text = self.text_part_obj_features
            obj_text = self.text_obj_features
            specific_part_text = self.text_specific_part_features

            part_index_map = self.train_text_to_part_map[None, None, None, :len(self.train_text_classes)]
            obj_index_map = self.train_text_to_obj_in_part_map[None, None, None, :]
            # num_part_classes = len(self.train_part_classes)
            num_obj_classes = len(self.train_obj_classes)
        elif test_text is not None:
            part_obj_text = self.text_part_obj_features_test[[
                (self.test_part_classes + self.test_obj_in_part_classes).index(t) for t in test_text]]
            obj_text = self.text_obj_features_test[[
                (self.test_obj_in_part_classes).index(t) for t in test_text]]
            specific_part_text = self.text_specific_part_features_test[[
                self.test_ori_text_classes.index(t) for t in test_text]]
            
            part_index_map = self.test_text_to_part_map[None, None, None, :len(self.test_text_classes)]
            obj_index_map = self.test_text_to_obj_in_part_map[None, None, None, :]
            # num_part_classes = len(self.test_part_classes)
            num_obj_classes = len(self.test_obj_classes)
        else:
            part_obj_text = self.text_part_obj_features_test
            obj_text = self.text_obj_features_test
            specific_part_text = self.text_specific_part_features_test
            
            part_index_map = self.test_text_to_part_map[None, None, None, :len(self.test_text_classes)]
            obj_index_map = self.test_text_to_obj_in_part_map[None, None, None, :]
            # num_part_classes = len(self.test_part_classes)
            num_obj_classes = len(self.test_obj_classes)

        part_obj_text = part_obj_text.repeat(x.shape[0], 1, 1, 1)
        obj_text = obj_text.repeat(x.shape[0], 1, 1, 1)

        specific_part_text = specific_part_text.repeat(x.shape[0], 1, 1, 1)
        obj_out, costs = self.transformer(
            x, 
            obj_text,  # part_obj_text, 
            specific_part_text, 
            vis,  # appearance guidance
            part_index_map, 
            obj_index_map,
            num_obj_classes
        )
        return obj_out, costs

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
