from modules import PointNetSetAbstraction
from feature_propagation import PointNetFeaturePropagation
import torch.nn as nn
import torch

class PointNet2SemSeg(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(npoint=1024, radius=0.1, nsample=32, in_channel=0, mlp_channels=[32, 32, 64])
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=0.2, nsample=32, in_channel=64, mlp_channels=[64, 64, 128])
        self.sa3 = PointNetSetAbstraction(npoint=64, radius=0.4, nsample=32, in_channel=128, mlp_channels=[128, 128, 256])

        self.fp3 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp2 = PointNetFeaturePropagation(in_channel=192, mlp=[128, 64])
        self.fp1 = PointNetFeaturePropagation(in_channel=64, mlp=[64, 64, 64])

        self.classifier = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.Dropout(0.5),
            nn.Conv1d(64, num_classes, 1)
        )

    def forward(self, xyz):  # xyz: (B, N, 3)
        B, N, _ = xyz.shape
        l0_points = None
        l0_xyz = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        out = self.classifier(l0_points.permute(0, 2, 1))  # (B, num_classes, N)
        return out.permute(0, 2, 1)  # (B, N, num_classes)
