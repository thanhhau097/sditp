from typing import Tuple

import timm
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
        model_name: str = "resnet50",
        objective: str = "cosine",
        **kwargs
    ):
        super().__init__()

    def feature(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        features = self.linear(features)
        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature(images)
        return features


class TimmModel(Model):
    def __init__(self, model_name: str = "resnet50", objective: str = "cosine"):
        super().__init__(model_name, objective)
        self.backbone = timm.create_model(
            model_name, pretrained=True, in_chans=3, drop_path_rate=0.2
        )
        self.backbone.reset_classifier(0, "avg")

        self.objective = objective
        self.linear = nn.Linear(self.backbone.num_features, 384)

    def feature(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        features = self.linear(features)
        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature(images)
        return features


# How to get encoder pretrained weights from stable diffusion model?
# from diffusers import StableDiffusionPipeline
# import torch

# class CFG:
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     seed = 42
#     generator = torch.Generator(device).manual_seed(seed)
#     model_id = "stabilityai/stable-diffusion-2"

# CFG = CFG()

# model_pipe = StableDiffusionPipeline.from_pretrained(CFG.model_id, torch_dtype=torch.float32)
# torch.save(model_pipe.components["vae"], "autoencoder_module.pth")


class SDModel(Model):
    def __init__(self, weights_path, objective: str = "cosine", freeze_backbone=True, **kwargs):
        super().__init__()
        self.backbone = torch.load(weights_path, map_location="cpu")
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.linear = nn.Linear(16384, 384)

        self.objective = objective

    def feature(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone.encoder(images)
        features = self.backbone.quant_conv(features)
        features, _ = torch.chunk(features, 2, dim=1)

        features = features.view(features.size(0), -1)
        features = self.linear(features)
        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature(images)
        return features
