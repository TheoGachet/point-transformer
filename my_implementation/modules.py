import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, channels):
        super().__init__()
        layers = []
        for i in range(len(channels)-1):
            layers += [
                nn.Linear(channels[i], channels[i+1]),
                nn.BatchNorm1d(channels[i+1]),
                nn.ReLU()
            ]
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        # x: (B, N, C) → fusionne batch et points pour BatchNorm1d
        B, N, C = x.shape
        x = self.model(x.view(B*N, C))
        return x.view(B, N, -1)

def square_distance(src, dst):
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)

def farthest_point_sample(xyz, npoint):
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long)
    distance = torch.ones(B, N) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B), farthest].unsqueeze(1)  # (B,1,3)
        dist = torch.sum((xyz - centroid) ** 2, -1)  # (B,N)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    B = points.shape[0]
    S = idx.shape[1]
    return points[torch.arange(B)[:, None], idx]

def query_ball_point(radius, nsample, xyz, new_xyz):
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    sqrdists = square_distance(new_xyz, xyz)  # (B, S, N)
    group_idx = sqrdists.argsort()[:, :, :nsample]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points):
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # (B, npoint)
    new_xyz = index_points(xyz, fps_idx)  # (B, S, 3)
    group_idx = query_ball_point(radius, nsample, xyz, new_xyz)  # (B, S, nsample)
    grouped_xyz = index_points(xyz, group_idx) - new_xyz.unsqueeze(2)  # Local coords

    if points is not None:
        grouped_points = index_points(points, group_idx)
        new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)  # (B, S, nsample, C+3)
    else:
        new_points = grouped_xyz

    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp_channels):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        layers = []
        last_channel = in_channel + 3
        for out_channel in mlp_channels:
            layers.append(nn.Conv2d(last_channel, out_channel, 1))
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, points):
        new_xyz, grouped_points = sample_and_group(
            self.npoint, self.radius, self.nsample, xyz, points
        )  # grouped_points: (B, S, nsample, C)

        new_points = grouped_points.permute(0, 3, 2, 1)  # (B, C, nsample, S)
        new_points = self.mlp(new_points)  # (B, C', nsample, S)
        new_points = torch.max(new_points, 2)[0]  # (B, C', S)
        return new_xyz, new_points.permute(0, 2, 1)  # (B, S, 3), (B, S, C')
