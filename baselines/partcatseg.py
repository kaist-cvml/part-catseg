# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple
import gc
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from collections import defaultdict

@META_ARCH_REGISTRY.register()
class PartCATSeg(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        clip_pixel_mean: Tuple[float],
        clip_pixel_std: Tuple[float],
        clip_finetune: str,
        clip_pretrained: str,
        train_dataset: str,
        test_dataset: str,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
        """
        super().__init__()
        
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_mean", torch.Tensor(clip_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_std", torch.Tensor(clip_pixel_std).view(-1, 1, 1), False)

        self.ignore_label = MetadataCatalog.get(test_dataset).ignore_label
        self.init_metadata(train_dataset, "train")
        self.init_metadata(test_dataset, "test")
        self.device = "cuda"

        self.clip_finetune = clip_finetune
        for name, params in self.sem_seg_head.predictor.clip_model.named_parameters():
            if "visual" in name:
                if clip_finetune == "prompt":
                    params.requires_grad = True if "prompt" in name else False
                elif clip_finetune == "attention":
                    # if "attn" in name or "position" in name:
                    #     print(name,params.shape)
                    params.requires_grad = True if "attn" in name or "position" in name else False
                elif clip_finetune == "full":
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            else:
                params.requires_grad = False

        self.sem_seg_head.predictor.transformer.conv2.load_state_dict(self.sem_seg_head.predictor.transformer.conv1.state_dict())
        self.sem_seg_head.predictor.transformer.layers_object.load_state_dict(self.sem_seg_head.predictor.transformer.layers.state_dict())
        self.sem_seg_head.predictor.transformer.layers_specific_part.load_state_dict(self.sem_seg_head.predictor.transformer.layers.state_dict())

        self.sem_seg_head.predictor.transformer.conv2.requires_grad_(True)
        self.sem_seg_head.predictor.transformer.layers_object.requires_grad_(False)
        self.sem_seg_head.predictor.transformer.layers_specific_part.requires_grad_(True)

        self.clip_resolution = (384, 384) if clip_pretrained == "ViT-B/16" else (336, 336)
        self.sequential = False
        self.lambda_part = 1.0
        self.lambda_obj = 1.0
        self.lambda_jsd = 1.0

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_pixel_mean": cfg.MODEL.CLIP_PIXEL_MEAN,
            "clip_pixel_std": cfg.MODEL.CLIP_PIXEL_STD,
            "clip_finetune": cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE,
            "clip_pretrained": cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED,
            "train_dataset": cfg.DATASETS.TRAIN[0],
            "test_dataset": cfg.DATASETS.TEST[0],
        }

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

    @property
    def device(self):
        return self.pixel_mean.device

    @device.setter
    def device(self, value):
        self.pixel_mean = self.pixel_mean.to(value)
        self.pixel_std = self.pixel_std.to(value)
        self.clip_pixel_mean = self.clip_pixel_mean.to(value)
        self.clip_pixel_std = self.clip_pixel_std.to(value)
        for prefix in ["train", "test"]:
            setattr(self, f"{prefix}_text_to_part_map", getattr(self, f"{prefix}_text_to_part_map").to(value))
            setattr(self, f"{prefix}_obj_to_obj_in_part_map", getattr(self, f"{prefix}_obj_to_obj_in_part_map").to(value))
            setattr(self, f"{prefix}_text_to_obj_in_part_map", getattr(self, f"{prefix}_text_to_obj_in_part_map").to(value))

    def jensen_shannon_divergence(self, p, q, dim, eps=1e-10):
        p = p + eps
        q = q + eps
        m = 0.5 * (p + q)
        jsd = 0.5 * (p * (p.log() - m.log())).sum(dim=dim) + 0.5 * (q * (q.log() - m.log())).sum(dim=dim)
        return jsd.mean()

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        gts = [x["obj_part_sem_seg"].to(self.device) for x in batched_inputs]
        obj_gts = [x["sem_seg"].to(self.device) for x in batched_inputs]

        clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
        clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)  # [[3, 384, 384], ...]
        clip_images = F.interpolate(clip_images.tensor, size=self.clip_resolution, mode='bilinear', align_corners=False, )
        clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images, dense=True)

        # ------
        # DINOv2
        # ------
        features_dino = self.backbone(images)

        if self.training:
            num_text_classes = len(self.train_text_classes)
            num_part_classes = len(self.train_part_classes)
            num_obj_classes = len(self.train_obj_in_part_classes)
            num_part_obj_classes = num_part_classes + num_obj_classes

            targets = torch.stack([gt for gt in gts], dim=0).long().squeeze(1).squeeze(1)
            obj_targets = torch.stack([gt for gt in obj_gts], dim=0).long().squeeze(1).squeeze(1)
            part_targets = self.train_text_to_part_map[targets]
            obj_in_part_targets = self.train_obj_to_obj_in_part_map[obj_targets]

            part_obj_outputs, specific_part_outputs, costs = self.sem_seg_head(clip_features, features_dino)
            part_obj_outputs = F.interpolate(part_obj_outputs, size=(targets.shape[-2], targets.shape[-1]), mode="bilinear", align_corners=False)
            specific_part_outputs = F.interpolate(specific_part_outputs, size=(targets.shape[-2], targets.shape[-1]), mode="bilinear", align_corners=False)

            bs, num_classes, h, w = part_obj_outputs.shape
            assert num_classes == num_part_obj_classes + 1, f"{num_classes} != {num_part_obj_classes + 1}"

            mask = targets != self.sem_seg_head.ignore_value
            part_mask = part_targets != self.ignore_label
            obj_mask = obj_in_part_targets != self.ignore_label

            part_obj_outputs = part_obj_outputs.permute(0,2,3,1)
            specific_part_outputs = specific_part_outputs.permute(0,2,3,1)

            part_obj_targets = torch.zeros(part_obj_outputs.shape, device=self.device)
            specific_part_targets = torch.zeros((bs, h, w, num_text_classes + 1), device=self.device)

            part_onehot = F.one_hot(part_targets[part_mask], num_classes=num_part_classes).float()
            obj_onehot = F.one_hot(obj_in_part_targets[obj_mask], num_classes=num_obj_classes).float()
            specific_part_onehot = F.one_hot(targets[mask], num_classes=num_text_classes).float()

            part_obj_targets[..., :num_part_classes][part_mask] = part_onehot
            part_obj_targets[..., num_part_classes:-1][obj_mask] = obj_onehot
            specific_part_targets[..., :num_text_classes][mask] = specific_part_onehot

            part_obj_weight = torch.ones(num_classes).cuda()
            specific_part_weight = torch.ones(num_text_classes + 1).cuda()

            if self.sem_seg_head.predictor.bg_on:
                part_obj_targets[..., -1][~obj_mask] = 1
                specific_part_targets[..., -1][~obj_mask] = 1

                part_obj_weight[-1] = 0.05
                specific_part_weight[-1] = 0.05

            specific_part_loss, part_loss, obj_loss = 0.0, 0.0, 0.0

            if part_mask.sum() > 0:
                specific_part_loss = F.binary_cross_entropy_with_logits(
                    specific_part_outputs[part_mask],
                    specific_part_targets[part_mask],
                    weight=specific_part_weight,
                )
                part_loss = F.binary_cross_entropy_with_logits(
                    part_obj_outputs[..., :num_part_classes][part_mask],
                    part_obj_targets[..., :num_part_classes][part_mask],
                    weight=part_obj_weight[:num_part_classes],
                )

            if obj_mask.sum() > 0 and (~obj_mask).sum() > 0:
                specific_part_loss += F.binary_cross_entropy_with_logits(
                    specific_part_outputs[~obj_mask],
                    specific_part_targets[~obj_mask],
                    weight=specific_part_weight,
                )

            obj_loss = F.binary_cross_entropy_with_logits(
                part_obj_outputs[..., num_part_classes:],
                part_obj_targets[..., num_part_classes:],
                weight=part_obj_weight[num_part_classes:],
            )

            eps = 1e-10
            bs, _, _, ch, cw = costs["part_obj_cost"].shape
            obj_costs = costs["part_obj_cost"].permute(0, 2, 1, 3, 4)[:, num_part_classes:] # B P T H W -> B T P H W (include background)
            obj_costs = obj_costs.mean(dim=2, keepdim=True).view(bs, num_obj_classes + 1, -1) # B T P H W -> B T 1 H W -> B T (H * W)
            
            specific_part_costs = costs["specific_part_cost"].permute(0, 2, 1, 3, 4) # B P T H W -> B T P H W (include background)
            specific_part_costs = specific_part_costs.mean(dim=2, keepdim=True).view(bs, num_text_classes + 1, -1) # B T P H W -> B T 1 H W -> B T (H * W)

            specific_part_costs = specific_part_costs.softmax(dim=1) + eps
            obj_costs = obj_costs.softmax(dim=1) + eps

            mapping = torch.cat([self.train_text_to_obj_in_part_map, torch.tensor(num_obj_classes).to(self.device).unsqueeze(0)], dim=-1) # T -> T+1 (include background)

            indices = mapping.unsqueeze(0).unsqueeze(2).expand(bs, -1, ch * cw).long()
            mapped_specific_part_costs = torch.zeros(obj_costs.shape, device=self.device)

            mapped_specific_part_costs = mapped_specific_part_costs.scatter_reduce(
                dim=1,
                index=indices,
                src=specific_part_costs,
                reduce='sum',
                include_self=False 
            )
        
            jsd_loss = self.jensen_shannon_divergence(mapped_specific_part_costs, obj_costs, dim=1)

            loss = specific_part_loss + self.lambda_part * part_loss + self.lambda_obj * obj_loss + self.lambda_jsd * jsd_loss

            losses = {"loss_sem_seg" : loss}
            return losses
        else:
            num_text_classes = len(self.test_text_classes)
            num_part_classes = len(self.test_part_classes)
            num_obj_classes = len(self.test_obj_in_part_classes)
            num_part_obj_classes = num_part_classes + num_obj_classes

            with torch.no_grad():
                _, outputs, costs = self.sem_seg_head(clip_features, features_dino)
                costs = {k: v.cpu() for k, v in costs.items()}

                obj_instances = [x["instances"].to(self.device) for x in batched_inputs]
                obj_class = self.sem_seg_head.predictor.test_obj_classes[obj_instances[0].gt_classes[0].item()]
                obj_part_classes = self.sem_seg_head.predictor.test_ori_text_classes
                select_mask = [i for i, name in enumerate(obj_part_classes) if obj_class not in name] 

                if self.sem_seg_head.predictor.bg_on:
                    outputs_all = outputs.sigmoid()
                    outputs = outputs.sigmoid()[:,:-1,:,:].cpu()
                else:
                    outputs_all = outputs.sigmoid()
                    outputs = outputs.sigmoid().cpu()

                outputs[:,select_mask,:,:] = -1.0

                image_size = images.image_sizes[0]
                height = batched_inputs[0].get("height", image_size[0])
                width = batched_inputs[0].get("width", image_size[1])
                output = sem_seg_postprocess(outputs[0].cpu(), image_size, height, width)
                output_all = sem_seg_postprocess(outputs_all[0].cpu(), image_size, height, width)
                processed_results = [
                    {'sem_seg': output.detach().cpu(), 'sem_seg_all': output_all.detach().cpu(), 'costs': costs}
                ]

            gc.collect()
            torch.cuda.empty_cache()

            return processed_results