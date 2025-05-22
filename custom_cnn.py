# custom_cnn.py

import torch
import torch.nn as nn
import torchvision.models as models
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()  # Fully connected 제거

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 + observation_space["vector"].shape[0], features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        image = observations["image"].float() / 255.0
        vector = observations["vector"]
        image_feat = self.resnet(image)
        concat = torch.cat([image_feat, vector], dim=1)
        return self.linear(concat)