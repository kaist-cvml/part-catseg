# Import necessary modules
from detectron2.modeling import Backbone, ShapeSpec, BACKBONE_REGISTRY
from detectron2.utils.file_io import PathManager

import torch
import torch.nn.functional as F

@BACKBONE_REGISTRY.register()
class DINOv2Backbone(Backbone):
    def __init__(self, cfg, input_shape):
        """
        Initialize the DINOv2 backbone.

        Args:
            cfg: detectron2 configuration.
            input_shape: ShapeSpec of the input image.
        """
        super().__init__()
        # Load DINOv2 model
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')  # Patch 14x14 -> Emb 384
         # Set to evaluation mode
        self.appearance_dim = cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_DIM

        # Load pretrained weights if provided
        weights = cfg.MODEL.WEIGHTS if hasattr(cfg.MODEL, "WEIGHTS") else None
        if weights:
            with PathManager.open(cfg.MODEL.WEIGHTS, "rb") as f:
                checkpoint = torch.load(f, map_location="cpu")
            # load only the model parameters starting with "backbone.dinov2." from the checkpoint
            model_dict = self.dinov2.state_dict().copy()
            for k, v in checkpoint["model"].items():
                if k.startswith("backbone.dinov2."):
                    model_dict[k.replace("backbone.dinov2.", "")] = v
            self.dinov2.load_state_dict(model_dict)

        self.dinov2.eval()
        for name, params in self.dinov2.named_parameters():
            if "norm0" in name:
                params.requires_grad = False
            else:
                params.requires_grad = cfg.SOLVER.BACKBONE_MULTIPLIER > 0.

        # Define the output shapes of the features
        self._out_features = ["res2", "res3", "res4"]
        self._out_feature_channels = {"res2": 256, "res3": 512, "res4": 1024}
        self._out_feature_strides = {"res2": 4, "res3": 8, "res4": 16}

    def forward(self, images):
        """
        Forward pass of the backbone.

        Args:
            images (Tensor): Input images tensor of shape (B, C, H, W).

        Returns:
            Dict[str, Tensor]: Dictionary of feature maps.
        """
        images_resized_dino = F.interpolate(images.tensor, size=(448, 448), mode='bilinear', align_corners=False)
        embeddings = self.dinov2.forward_features(images_resized_dino)
        x_norm_patchtokens = embeddings["x_norm_patchtokens"]

        # print("images_resized_dino", images_resized_dino.shape)  # [4, 3, 448, 448]
        # print("x_norm_patchtokens", x_norm_patchtokens.shape)    # [4, 1024, 384]

        # Reshape and permute to match expected dimensions
        x_norm_patchtokens = x_norm_patchtokens.view(-1, 32, 32, self.appearance_dim)
        x_norm_patchtokens = x_norm_patchtokens.permute(0, 3, 1, 2)  # (B, C, H, W)

        # Return the features as a dictionary
        features_dino = {}
        sizes = {'res2': 96, 'res3': 48, 'res4': 24}
        for key in ['res2', 'res3', 'res4']:
            x_resized = F.interpolate(x_norm_patchtokens, size=(sizes[key], sizes[key]), mode='bilinear', align_corners=False)
            # Only DINO
            features_dino[key] = torch.cat([x_resized], dim=1)

        return features_dino

    def output_shape(self):
        """
        Returns the output shape of each feature map produced by the backbone.

        Returns:
            Dict[str, ShapeSpec]: Mapping from feature name to ShapeSpec.
        """
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name]
            ) for name in self._out_features
        }
