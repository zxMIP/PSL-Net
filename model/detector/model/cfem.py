import torch
import torch.nn as nn

class CFEM(nn.Module):
    def __init__(self, in_channels):
        super(CFEM, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=2),
            nn.ReLU(inplace=True)
        )

        self.conv = nn.ModuleList([
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.Conv2d(in_channels, 256, kernel_size=1),
            nn.Conv2d(in_channels, 512, kernel_size=1)
        ])

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        self.conv_norm = nn.Conv2d(64 + 128 + 256 + 512, 4, kernel_size=1)

        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)

        self.conv_out = nn.Conv2d(4, in_channels, kernel_size=1)

    def forward(self, x):
        features = []
        out = x

        for _ in range(4):
            out = self.downsample(out)
            features.append(out)

        feature_maps = [conv(feat) for conv, feat in zip(self.conv, features)]

        gap_maps = [self.gap(fm) for fm in feature_maps]
        gmp_maps = [self.gmp(fm) for fm in feature_maps]

        fused_gap = torch.cat(gap_maps, dim=1)
        fused_gmp = torch.cat(gmp_maps, dim=1)

        fgap = self.conv_norm(fused_gap)
        fgmp = self.conv_norm(fused_gmp)

        fgap_relu = torch.relu(fgap)
        fgmp_relu = torch.relu(fgmp)

        fused_features = torch.sigmoid(fgap_relu + fgmp_relu)

        new_feature_maps = [fused_features[:, i:i+1] * fm for i, fm in enumerate(feature_maps)]
        upsampled_maps = [self.upsample(fm) for fm in new_feature_maps]

        final_features = torch.cat(upsampled_maps, dim=1)
        final_features = self.conv_out(final_features)

        return final_features
