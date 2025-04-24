# Import necessary modules
from detectron2.modeling import Backbone, ShapeSpec, BACKBONE_REGISTRY
from detectron2.utils.file_io import PathManager

import torch
import torch.nn.functional as F

@BACKBONE_REGISTRY.register()
class DINOv1Backbone(Backbone):
    # DINO v1 의 중간값 출력을 위한 Hook


    def __init__(self, cfg, input_shape):

        def get_intermediate_outputs(module, input, output):
            global intermediate_output
            intermediate_output = output

        """
        Initialize the DINOv1 backbone.

        Args:
            cfg: detectron2 configuration.
            input_shape: ShapeSpec of the input image.
        """
        super().__init__()

        # Load DINOv2 model
        # self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')  # Patch 14x14 -> Emb 384

        # Load DINOv1 model
        self.dinov1 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')  # Patch 16x16 -> Emb 384

         # Set to evaluation mode
        self.appearance_dim = cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_DIM

        # Load pretrained weights if provided
        weights = cfg.MODEL.WEIGHTS if hasattr(cfg.MODEL, "WEIGHTS") else None
        if weights:
            with PathManager.open(cfg.MODEL.WEIGHTS, "rb") as f:
                checkpoint = torch.load(f, map_location="cpu")
            # load only the model parameters starting with "backbone.dinov2." from the checkpoint
            model_dict = self.dinov1.state_dict().copy()
            for k, v in checkpoint["model"].items():
                if k.startswith("backbone.dinov1."):
                    model_dict[k.replace("backbone.dinov1.", "")] = v
            self.dinov1.load_state_dict(model_dict)

        self.dinov1.eval()
        for name, params in self.dinov1.named_parameters():
            if "norm0" in name:
                params.requires_grad = False
            else:
                params.requires_grad = cfg.SOLVER.BACKBONE_MULTIPLIER > 0.

        # Define the output shapes of the features
        self._out_features = ["res2", "res3", "res4"]
        self._out_feature_channels = {"res2": 256, "res3": 512, "res4": 1024}
        self._out_feature_strides = {"res2": 4, "res3": 8, "res4": 16}

        hook = self.dinov1.blocks[-1].register_forward_hook(get_intermediate_outputs)



    def forward(self, images):
        """
        Forward pass of the backbone.

        Args:
            images (Tensor): Input images tensor of shape (B, C, H, W).

        Returns:
            Dict[str, Tensor]: Dictionary of feature maps.
        """
        # DINOv2
        # images_resized_dino = F.interpolate(images.tensor, size=(448, 448), mode='bilinear', align_corners=False)
        # embeddings = self.dinov2.forward_features(images_resized_dino)
        # x_norm_patchtokens = embeddings["x_norm_patchtokens"]

        # print("images_resized_dino", images_resized_dino.shape)  # [4, 3, 448, 448]
        # print("x_norm_patchtokens", x_norm_patchtokens.shape)    # [4, 1024, 384]


        # DINOv1
        images_resized_dino = F.interpolate(images.tensor, size=(224, 224), mode='bilinear', align_corners=False)
        # embeddings = self.dinov1.forward_features(images_resized_dino)


        # TODO: patch token 생성 및 dimension 맞추기
        # intermediate_output [4, 197, 384]
        # patch_tokens [4, 196, 384]

        embeddings = self.dinov1(images_resized_dino)
        patch_tokens = intermediate_output[:, 1:, :]
        x_norm_patchtokens = F.normalize(patch_tokens, dim=-1)

        # print(images_resized_dino.shape)  # torch.Size([4, 3, 224, 224])
        # print(embeddings.shape)           # torch.Size([4, 384])


        # --------------------------------------------------------------
        # 모델     | 입력 해상도 (HxW) | 패치 크기 (PxP) | 패치 토큰 개수 (N) | 출력 크기 (B, N, D) |
        # --------------------------------------------------------------
        # DINO v1 |   224 x 224     |   16 x 16     | 14 x 14 = 196   | [4, 196, 384]     |
        # DINO v2 |   448 x 448     |   14 x 14     | 32 x 32 = 1024  | [4, 1024, 384]    |
        # --------------------------------------------------------------



        # images.tensor [4, 3, 384, 384]
        # images_resized_dino [4, 3, 384, 384]

        # DINOv2

        # # Reshape and permute to match expected dimensions
        # x_norm_patchtokens = x_norm_patchtokens.view(-1, 32, 32, self.appearance_dim)

        # DINOv1
        # x_norm_patchtokens = embeddings.view(-1, 14, 14, self.appearance_dim)  # Assuming patch size 16 -> 14x14 tokens
        # x_norm_patchtokens = x_norm_patchtokens.permute(0, 3, 1, 2)  # (B, C, H, W)

        x_norm_patchtokens = x_norm_patchtokens.view(-1, 14, 14, self.appearance_dim)
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
