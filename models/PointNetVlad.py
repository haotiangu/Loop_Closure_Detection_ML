
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import torch.nn.functional as F
from config import cfg
import math



class GatingContext(nn.Module):
    def __init__(self, dim, add_batch_norm=True):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        self.gating_weights = torch.randn(dim, dim) * 1 / math.sqrt(dim)
        self.sigmoid = nn.Sigmoid()
        self.gating_biases = None
        self.bn1 = nn.BatchNorm1d(dim)

    def forward(self, x):
        gates = torch.matmul(x, self.gating_weights)
        gates = self.bn1(gates)
        gates = gates.sigmoid()
        activation = torch.mul(x, gates)
        return activation

class STN3d(nn.Module):
    def __init__(self, num_points=2500, k=3, use_bn=True):
        super(STN3d, self).__init__()
        self.k = k
        self.kernel_size = 3 if k == 3 else 1
        self.channels = 1 if k == 3 else k
        self.num_points = num_points
        self.use_bn = use_bn

        self.seq_layers1 = nn.Sequential(
            torch.nn.Conv2d(self.channels, 64, (1, self.kernel_size)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            torch.nn.Conv2d(64, 128, (1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            torch.nn.Conv2d(128, 1024, (1,1)),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            torch.nn.MaxPool2d((num_points, 1), 1)

        )

        self.seq_layers2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, k * k,)

        )

        self.seq_layers2[-1].weight.data.zero_()
        self.seq_layers2[-1].bias.data.zero_()

    def forward(self, x):
        batchsize = x.size()[0]

        x = self.seq_layers1(x)
        x = x.view(-1, 1024)
        x = self.seq_layers2(x)

        iden = torch.from_numpy(np.eye(self.k).astype(np.float32)).view(
            1, self.k*self.k).repeat(batchsize, 1)
        iden.requires_grad = True
        if x.is_cuda:
            iden = iden.to('cuda')

        x = torch.add(x, iden)
        x = x.view(-1, self.k, self.k)
        return x

class Model(nn.Module):
    def __init__(self, num_points=4096, global_feat=True, feature_transform=False, max_pool=True, output_dim=256):
        super(Model, self).__init__()
        self.num_points = num_points
        self.global_feat = global_feat
        self.max_pool = max_pool
        self.feature_transform = feature_transform
        self.output_dim = output_dim

        # point_net
        self._init_layers_pointnet()
        # net_vlad
        self._init_layers_vlad()

    def _init_layers_pointnet(self):
        self.stn = STN3d(num_points=self.num_points, k=3, use_bn=False).to(cfg.device)
        self.feature_trans = STN3d(num_points=self.num_points, k=64, use_bn=False).to(cfg.device)
        self.apply_feature_trans = self.feature_transform

        self.in_layer1 = nn.Sequential(
            torch.nn.Conv2d(1, 64, (1, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, (1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ).to(cfg.device)
        self.in_layer2 = nn.Sequential(
            torch.nn.Conv2d(64, 64, (1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 128, (1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 1024, (1, 1)),
            nn.BatchNorm2d(1024)
        ).to(cfg.device)

    def _init_layers_vlad(self):
        self.feature_size = 1024
        self.max_samples = self.num_points
        self.gating = True
        self.add_batch_norm = True
        self.cluster_size = 64
        self.cluster_weights = torch.randn(
            self.feature_size, self.cluster_size).to(cfg.device) * 1 / (self.feature_size)**0.5
        self.cluster_weights2 = torch.randn(
            1, self.feature_size, self.cluster_size).to(cfg.device) * 1 / (self.feature_size)**0.5
        self.hidden1_weights = torch.randn(self.cluster_size * self.feature_size, self.output_dim).to(cfg.device) * 1 / math.sqrt(self.feature_size)

        if self.add_batch_norm:
            self.cluster_biases = None
            self.bn1 = nn.BatchNorm1d(self.cluster_size).to(cfg.device)
        else:
            self.cluster_biases = nn.Parameter(torch.randn(
                self.cluster_size) * 1 / math.sqrt(self.feature_size))
            self.bn1 = None

        self.bn2 = nn.BatchNorm1d(self.output_dim).to(cfg.device)

        if self.gating:
            self.context_gating = GatingContext(
                self.output_dim, add_batch_norm=self.add_batch_norm).to(cfg.device)

    def forward(self, x):
        x = self.forward_by_pointnet(x)
        out = self.forward_by_vlad(x)
        return out

    def forward_by_pointnet(self, x):
        batchsize = x.size(0)
        trans = self.stn(x)
        x = torch.matmul(torch.squeeze(x), trans)
        x = x.view(batchsize, 1, -1, 3)
        x = self.in_layer1(x)
        pointfeat = x
        if self.apply_feature_trans:
            f_trans = self.feature_trans(x)
            x = torch.squeeze(x)
            if batchsize == 1:
                x = torch.unsqueeze(x, 0)
            x = torch.matmul(x.transpose(1, 2), f_trans)
            x = x.transpose(1, 2).contiguous()
            x = x.view(batchsize, 64, -1, 1)
        x = self.in_layer2(x)
        if not self.max_pool:
            return x
        else:
            x = F.max_pool2d(x, kernel_size=(self.num_points, 1), stride=1)
            x = x.view(-1, 1024)
            if self.global_feat:
                return x, trans
            else:
                x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans

    def forward_by_vlad(self, x):
        x = x.transpose(1, 3).contiguous()
        x = x.view((-1, self.max_samples, self.feature_size))
        activation = torch.matmul(x, self.cluster_weights)
        if self.add_batch_norm:
            activation = activation.view(-1, self.cluster_size)
            activation = self.bn1(activation)
            activation = activation.view(-1,
                                         self.max_samples, self.cluster_size)
            # activation = activation.transpose(1,2).contiguous()
        else:
            activation = activation + self.cluster_biases
        activation = activation.softmax(-1)
        activation = activation.view((-1, self.max_samples, self.cluster_size))

        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weights2

        activation = torch.transpose(activation, 2, 1)
        x = x.view((-1, self.max_samples, self.feature_size))
        vlad = torch.matmul(activation, x)
        vlad = torch.transpose(vlad, 2, 1)
        vlad = vlad - a

        vlad = F.normalize(vlad, dim=1, p=2)
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)
        vlad = F.normalize(vlad, dim=1, p=2)

        vlad = torch.matmul(vlad, self.hidden1_weights)

        vlad = self.bn2(vlad)

        if self.gating:
            vlad = self.context_gating(vlad)
        return vlad

if '__main__' == '__name__':

    m = Model(global_feat=True, feature_transform=True,
                                 max_pool=False, output_dim=256, num_points=4096)
