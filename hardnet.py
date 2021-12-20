from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

urls: Dict[str, str] = {}
urls["hardnet++"] = "https://github.com/DagnyT/hardnet/raw/master/pretrained/pretrained_all_datasets/HardNet++.pth"
urls[
    "liberty_aug"
] = "https://github.com/DagnyT/hardnet/raw/master/pretrained/train_liberty_with_aug/checkpoint_liberty_with_aug.pth"  # noqa pylint: disable
urls["hardnet8v2"] = "http://cmp.felk.cvut.cz/~mishkdmy/hardnet8v2.pt"  # pylint: disable
urls["ps2"] = "https://github.com/DagnyT/hardnet/blob/master/pretrained/3rd_party/HardNetPS/HardNetPS.pth?raw=true"


class HardNet(nn.Module):
    r"""Module, which computes HardNet descriptors of given grayscale patches of 32x32.

    This is based on the original code from paper "Working hard to know your neighbor's
    margins: Local descriptor learning loss". See :cite:`HardNet2017` for more details.

    Args:
        pretrained: Download and set pretrained weights to the model.

    Returns:
        torch.Tensor: HardNet descriptor of the patches.

    Shape:
        - Input: :math:`(B, 1, 32, 32)`
        - Output: :math:`(B, 128)`

    Examples:
        >>> input = torch.rand(16, 1, 32, 32)
        >>> hardnet = HardNet()
        >>> descs = hardnet(input) # 16x128
    """
    patch_size = 32

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )

        # use torch.hub to load pretrained model
        if pretrained:
            # pretrained_dict = torch.hub.load_state_dict_from_url(
            #     urls['ps2'], map_location=lambda storage, loc: storage
            # )
            self.load_state_dict(torch.load("data/HardNetPS.pth"), strict=True)
        self.eval()

    @staticmethod
    def _normalize_input(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Utility function that normalizes the input by batch."""
        sp, mp = torch.std_mean(x, dim=(-3, -2, -1), keepdim=True)
        # WARNING: we need to .detach() input, otherwise the gradients produced by
        # the patches extractor with F.grid_sample are very noisy, making the detector
        # training totally unstable.
        return (x - mp.detach()) / (sp.detach() + eps)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_norm: torch.Tensor = self._normalize_input(input)
        x_features: torch.Tensor = self.features(x_norm)
        x_out = x_features.view(x_features.size(0), -1)
        return F.normalize(x_out, dim=1)


class HardNet2(nn.Module):

    patch_size = 32

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=8, bias=True)
        )

        # use torch.hub to load pretrained model
        if pretrained:
            self.load_state_dict(torch.load("data/HardNetPS.pth"), strict=True)
        self.eval()

    @staticmethod
    def _normalize_input(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Utility function that normalizes the input by batch."""
        sp, mp = torch.std_mean(x, dim=(-3, -2, -1), keepdim=True)
        # WARNING: we need to .detach() input, otherwise the gradients produced by
        # the patches extractor with F.grid_sample are very noisy, making the detector
        # training totally unstable.
        return (x - mp.detach()) / (sp.detach() + eps)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_norm: torch.Tensor = self._normalize_input(input)
        x_features: torch.Tensor = self.features(x_norm)
        x_out = x_features.view(x_features.size(0), -1)
        return F.normalize(x_out, dim=1)


if __name__ == '__main__':
    n = HardNet2(pretrained=True)