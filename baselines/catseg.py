# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple
import gc
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import _ignore_torch_cuda_oom

from einops import rearrange

@META_ARCH_REGISTRY.register()
class CATSeg(nn.Module):
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
        sliding_window: bool,
        clip_finetune: str,
        backbone_multiplier: float,
        clip_pretrained: str,
        use_dino: bool = False,
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

        self.clip_finetune = clip_finetune
        for name, params in self.sem_seg_head.predictor.clip_model.named_parameters():
            if "visual" in name:
                if clip_finetune == "prompt":
                    params.requires_grad = True if "prompt" in name else False
                elif clip_finetune == "attention":
                    params.requires_grad = True if "attn" in name or "position" in name else False
                elif clip_finetune == "full":
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            else:
                params.requires_grad = False
        finetune_backbone = backbone_multiplier > 0.

        # ResNet
        for name, params in self.backbone.named_parameters():
            if "norm0" in name:
                params.requires_grad = False
            else:
                params.requires_grad = finetune_backbone

        self.sliding_window = sliding_window
        self.clip_resolution = (384, 384) if clip_pretrained == "ViT-B/16" else (336, 336)
        self.sequential = False
        self.use_dino = use_dino

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
            "sliding_window": cfg.TEST.SLIDING_WINDOW,
            "clip_finetune": cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE,
            "backbone_multiplier": cfg.SOLVER.BACKBONE_MULTIPLIER,
            "clip_pretrained": cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED,
            "use_dino": cfg.MODEL.BACKBONE.NAME == "DINOv2Backbone"
        }

    @property
    def device(self):
        return self.pixel_mean.device

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
        if not self.training and self.sliding_window:
            if not self.sequential:
                with _ignore_torch_cuda_oom():
                    return self.inference_sliding_window(batched_inputs)
            return self.inference_sliding_window(batched_inputs)

        clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
        clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        clip_images = F.interpolate(clip_images.tensor, size=self.clip_resolution, mode='bilinear', align_corners=False, )
        clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images, dense=True)

        # images_resized = F.interpolate(images.tensor, size=(384, 384), mode='bilinear', align_corners=False,)
        # features_resnet = self.backbone(images_resized)
        if self.use_dino:
            features = self.backbone(images)
        else:
            images_resized = F.interpolate(images.tensor, size=(384, 384), mode='bilinear', align_corners=False,)
            features = self.backbone(images_resized)


        if self.training:
            outputs = self.sem_seg_head(clip_features, features)
            targets = torch.stack([x["obj_part_sem_seg"].to(self.device) for x in batched_inputs], dim=0)
            
            outputs = F.interpolate(outputs, size=(targets.shape[-2], targets.shape[-1]), mode="bilinear", align_corners=False)
            num_classes = outputs.shape[1]
            
            mask = targets != self.sem_seg_head.ignore_value
            outputs = outputs.permute(0,2,3,1)

            _targets = torch.zeros(outputs.shape, device=self.device)
            class_weight = torch.ones(num_classes).cuda()

            if self.sem_seg_head.predictor.bg_on:
                _targets[:,:,:,-1] = 1
                class_weight[-1] = 0.05
            _onehot = F.one_hot(targets[mask], num_classes=num_classes).float()
            _targets[mask] = _onehot

            loss = F.binary_cross_entropy_with_logits(outputs, _targets, weight=class_weight)
            losses = {"loss_sem_seg" : loss}
            return losses
        else:
            with torch.no_grad():
                outputs = self.sem_seg_head(clip_features, features)
                obj_instances = [x["instances"].to(self.device) for x in batched_inputs]
                obj_class = self.sem_seg_head.predictor.test_obj_classes[obj_instances[0].gt_classes[0].item()]
                obj_part_classes = self.sem_seg_head.predictor.test_class_texts
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
                
                processed_results = [{'sem_seg': output, 'sem_seg_all': output_all}]
            
            gc.collect()
            torch.cuda.empty_cache()
            
            return processed_results


    @torch.no_grad()
    def inference_sliding_window(self, batched_inputs, kernel=384, overlap=0.333, out_res=[640, 640]):
        images = [x["image"].to(self.device, dtype=torch.float32) for x in batched_inputs]
        stride = int(kernel * (1 - overlap))
        unfold = nn.Unfold(kernel_size=kernel, stride=stride)
        fold = nn.Fold(out_res, kernel_size=kernel, stride=stride)

        image = F.interpolate(images[0].unsqueeze(0), size=out_res, mode='bilinear', align_corners=False).squeeze()
        image = rearrange(unfold(image), "(C H W) L-> L C H W", C=3, H=kernel)
        global_image = F.interpolate(images[0].unsqueeze(0), size=(kernel, kernel), mode='bilinear', align_corners=False)
        image = torch.cat((image, global_image), dim=0)

        images = (image - self.pixel_mean) / self.pixel_std
        clip_images = (image - self.clip_pixel_mean) / self.clip_pixel_std
        clip_images = F.interpolate(clip_images, size=self.clip_resolution, mode='bilinear', align_corners=False, )
        clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images, dense=True)
        
        if self.sequential:
            outputs = []
            for clip_feat, image in zip(clip_features, images):
                feature = self.backbone(image.unsqueeze(0))
                output = self.sem_seg_head(clip_feat.unsqueeze(0), feature)
                outputs.append(output[0])
            outputs = torch.stack(outputs, dim=0)
        else:
            features = self.backbone(images)
            outputs = self.sem_seg_head(clip_features, features)

        outputs = F.interpolate(outputs, size=kernel, mode="bilinear", align_corners=False)
        outputs = outputs.sigmoid()
        
        global_output = outputs[-1:]
        global_output = F.interpolate(global_output, size=out_res, mode='bilinear', align_corners=False,)
        outputs = outputs[:-1]
        outputs = fold(outputs.flatten(1).T) / fold(unfold(torch.ones([1] + out_res, device=self.device)))
        outputs = (outputs + global_output) / 2.

        height = batched_inputs[0].get("height", out_res[0])
        width = batched_inputs[0].get("width", out_res[1])
        output = sem_seg_postprocess(outputs, out_res, height, width)
        return [{'sem_seg': output}]