import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BG_Rejection(nn.Module):
    """
    Background Rejection Model.

    This model is based on ring-average(fix) and denoise parts.
    """

    def __init__(self, in_channels, out_channels, f_maps=32, gsize=8, infer=False):
        super().__init__()

        radius_min = int(1.5 * gsize)
        radius_max = radius_min + 2
        self.padding_size = radius_max

        self.infer = infer

        # -----------------background branch-----------------
        # ring kernel generation
        xx = np.arange(-radius_max, radius_max + 1)
        yy = np.arange(-radius_max, radius_max + 1)
        [XX, YY] = np.meshgrid(xx, yy)
        R = np.sqrt(XX ** 2 + YY ** 2)
        R[R < radius_min] = 0
        R[R > radius_max] = 0
        R[R > 0] = 1

        R = R / R.sum()  # normalization

        self.ring_weight = torch.as_tensor(R, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # ------------------neuron branch--------------------
        self.first_conv = nn.Conv3d(in_channels, f_maps, kernel_size=3, stride=1, padding=1)
        self.enc1 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(f_maps, f_maps, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(f_maps, 2 * f_maps, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(2 * f_maps, 2 * f_maps, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(2 * f_maps, 4 * f_maps, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.up_sample = nn.Upsample(scale_factor=(1, 2, 2))
        self.dec2 = nn.Sequential(
            nn.Conv3d(2 * f_maps + 4 * f_maps, 2 * f_maps, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(2 * f_maps, 2 * f_maps, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.Conv3d(f_maps + 2 * f_maps, f_maps, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(f_maps, f_maps, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.final_conv = nn.Conv3d(f_maps, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # input shape
        B, _, T, H, W = x.shape

        # background branch
        bg = x.permute(0, 2, 1, 3, 4).flatten(0, 1)  # (B*T, 1, H, W)
        bg = F.pad(bg, (self.padding_size, self.padding_size, self.padding_size, self.padding_size), mode='reflect')
        bg = F.conv2d(bg, self.ring_weight.to(bg.device)) # convolve with the ring
        bg = bg.reshape(B, T, -1, H, W).permute(0, 2, 1, 3, 4)  # (B, 1, T, H, W)

        if self.infer:
            neuron = self.first_conv(x - bg)  # (B, C, T, H, W), infer goes all the sequences
        else:
            neuron = self.first_conv((x - bg)[:, :, ::2, :, :])  # (B, C, T/2, H, W), training goes half of the sequences, since we are going to do odd and even mapping

        # neuron branch
        neuron_enc1 = self.enc1(neuron)  # (B, 2C, T/2, H/2, W/2)
        neuron_enc2 = self.enc2(neuron_enc1)  # (B, 4C, T/2, H/4, W/4)
        neuron_dec2 = self.dec2(torch.cat([neuron_enc1, self.up_sample(neuron_enc2)], dim=1))  # (B, 2C, T/2, H/2, W/2)
        neuron_dec1 = self.dec1(torch.cat([neuron, self.up_sample(neuron_dec2)], dim=1))  # (B, C, T/2, H, W)
        neuron = self.final_conv(neuron_dec1)  # (B, 1, T/2, H, W)

        return bg, neuron
