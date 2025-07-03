import torch
import torch.nn as nn
import torch.nn.functional as F

def three_nn_interpolate(xyz1, xyz2, points2):
    # interpolate from xyz2 → xyz1
    dists = torch.cdist(xyz1, xyz2, p=2)  # (B, N1, N2)
    dists, idx = torch.topk(dists, 3, dim=-1, largest=False, sorted=False)  # (B, N1, 3)
    dist_recip = 1.0 / (dists + 1e-8)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm  # (B, N1, 3)

    interpolated = torch.sum(index_points(points2, idx) * weight.unsqueeze(-1), dim=2)
    return interpolated  # (B, N1, C)

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        layers = []
        last_channel = in_channel
        for out_channel in mlp:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz1, xyz2, points1, points2):
        interpolated_points = three_nn_interpolate(xyz1, xyz2, points2)
        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        return self.mlp(new_points.permute(0, 2, 1)).permute(0, 2, 1)
