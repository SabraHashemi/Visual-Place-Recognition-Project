# Model from "MixVPR: Feature Mixing for Visual Place Recognition" - https://arxiv.org/abs/2303.02190
# Parts of this code are from https://github.com/amaralibey/MixVPR

import os

import gdown
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

MODELS_INFO = {
    128: (
        "https://drive.google.com/file/d/1DQnefjk1hVICOEYPwE4-CZAZOvi1NSJz/view",
        "resnet50_MixVPR_128_channels(64)_rows(2)",
        64,
        2,
    ),
    512: (
        "https://drive.google.com/file/d/1khiTUNzZhfV2UUupZoIsPIbsMRBYVDqj/view",
        "resnet50_MixVPR_512_channels(256)_rows(2)",
        256,
        2,
    ),
    4096: (
        "https://drive.google.com/file/d/1vuz3PvnR7vxnDDLQrdHJaOA04SQrtk5L/view",
        "resnet50_MixVPR_4096_channels(1024)_rows(4)",
        1024,
        4,
    ),
}


class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.mix(x)


class MixVPR(nn.Module):
    def __init__(
        self,
        in_channels=1024,
        in_h=20,
        in_w=20,
        out_channels=512,
        mix_depth=1,
        mlp_ratio=1,
        out_rows=4,
    ) -> None:
        super().__init__()

        self.in_h = in_h  # height of input feature maps
        self.in_w = in_w  # width of input feature maps
        self.in_channels = in_channels  # depth of input feature maps

        self.out_channels = out_channels  # depth wise projection dimension
        self.out_rows = out_rows  # row wise projection dimesion

        self.mix_depth = mix_depth  # L the number of stacked FeatureMixers
        self.mlp_ratio = mlp_ratio  # ratio of the mid projection layer in the mixer block

        hw = in_h * in_w
        self.mix = nn.Sequential(*[FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio) for _ in range(self.mix_depth)])
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x):
        x = x.flatten(2)
        x = self.mix(x)
        x = x.permute(0, 2, 1)
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)
        x = self.row_proj(x)
        x = F.normalize(x.flatten(1), p=2, dim=-1)
        return x

### Implement ResNet-50 here for MixVPR model,
### otherwise `self.backbone = ResNet()` will fail
### (academic purpose)
# This is the minimal ResNet wrapper used by MixVPR (from the original repo).
# It instantiates torchvision's ResNet variants, removes avgpool/fc, supports freezing
# initial layers and cropping layer3/layer4 if you want a lower-channel output.
class ResNet(nn.Module):
    def __init__(self,
                 model_name='resnet50',
                 pretrained=True,
                 layers_to_freeze=2,
                 layers_to_crop=[],
                 ):
        """
        Args:
            model_name (str): e.g. 'resnet50'
            pretrained (bool): use ImageNet weights if True
            layers_to_freeze (int): freeze first N residual blocks (0..4)
            layers_to_crop (list): indices of residual layers to remove (e.g. [4] to remove layer4)
        """
        super().__init__()
        self.model_name = model_name.lower()
        self.layers_to_freeze = layers_to_freeze

        # torchvision new weights API uses strings such as 'IMAGENET1K_V1'
        if pretrained:
            weights = 'IMAGENET1K_V1'
        else:
            weights = None

        if 'swsl' in model_name or 'ssl' in model_name:
            # optional alternative weight source (facebook semi-supervised)
            self.model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', model_name)
        else:
            if 'resnext50' in model_name:
                self.model = torchvision.models.resnext50_32x4d(weights=weights)
            elif 'resnet50' in model_name:
                self.model = torchvision.models.resnet50(weights=weights)
            elif '101' in model_name:
                self.model = torchvision.models.resnet101(weights=weights)
            elif '152' in model_name:
                self.model = torchvision.models.resnet152(weights=weights)
            elif '34' in model_name:
                self.model = torchvision.models.resnet34(weights=weights)
            elif '18' in model_name:
                self.model = torchvision.models.resnet18(weights=weights)
            elif 'wide_resnet50_2' in model_name:
                self.model = torchvision.models.wide_resnet50_2(weights=weights)
            else:
                raise NotImplementedError('Backbone architecture not recognized!')

        # freeze only if the model is pretrained
        if pretrained:
            if layers_to_freeze >= 0:
                self.model.conv1.requires_grad_(False)
                self.model.bn1.requires_grad_(False)
            if layers_to_freeze >= 1:
                self.model.layer1.requires_grad_(False)
            if layers_to_freeze >= 2:
                self.model.layer2.requires_grad_(False)
            if layers_to_freeze >= 3:
                self.model.layer3.requires_grad_(False)

        # remove the avgpool and fc layers (we want the conv feature maps)
        self.model.avgpool = None
        self.model.fc = None

        return x

class MixVPRModel(torch.nn.Module):
    def __init__(self, agg_config={},  backbone_cfg=None):
        super().__init__()
        if backbone_cfg is None:
            backbone_cfg = {'model_name': 'resnet50', 'pretrained': True, 'layers_to_freeze': 2, 'layers_to_crop': [4]}
        self.backbone = ResNet(**backbone_cfg)
        self.aggregator = MixVPR(**agg_config)

    def forward(self, x):
        x = transforms.Resize([320, 320], antialias=True)(x)
        x = self.backbone(x)
        x = self.aggregator(x)
        return x


def get_mixvpr(descriptors_dimension):
    url, filename, out_channels, out_rows = MODELS_INFO[descriptors_dimension]
    model_config = {
        "in_channels": 1024,
        "in_h": 20,
        "in_w": 20,
        "out_channels": out_channels,
        "mix_depth": 4,
        "mlp_ratio": 1,
        "out_rows": out_rows,
    }
    backbone_cfg = {'model_name': 'resnet50', 'pretrained': True, 'layers_to_freeze': 2, 'layers_to_crop': [4]}
    model = MixVPRModel(agg_config=model_config, backbone_cfg=backbone_cfg)
    file_path = f"trained_models/mixvpr/{filename}"
    if not os.path.exists(file_path):
        os.makedirs("trained_models/mixvpr", exist_ok=True)
        gdown.download(url=url, output=file_path, fuzzy=True)
    state_dict = torch.load(file_path)
    model.load_state_dict(state_dict)
    model = model.eval()

    return model
