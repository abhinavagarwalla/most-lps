# -*- coding:utf-8 -*-
# author: Abhinav, Modified from Cylinder3D
# @file: segmentator_3d_asymm_spconv.py
from dataclasses import replace
import numpy as np
try:
    import spconv.pytorch as spconv 
    from spconv.pytorch import ops
    from spconv.pytorch import SparseConv3d, SubMConv3d
except: 
    import spconv 
    from spconv import ops
    from spconv import SparseConv3d, SubMConv3d

import torch
from torch import nn

from ..registry import BACKBONES
from ..utils import build_norm_layer

def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out


def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    return SubMConv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


def conv1x3(in_planes, out_planes, stride=1, indice_key=None):
    return SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride,
                             padding=(0, 1, 1), bias=False, indice_key=indice_key)


def conv1x1x3(in_planes, out_planes, stride=1, indice_key=None):
    return SubMConv3d(in_planes, out_planes, kernel_size=(1, 1, 3), stride=stride,
                             padding=(0, 0, 1), bias=False, indice_key=indice_key)


def conv1x3x1(in_planes, out_planes, stride=1, indice_key=None):
    return SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 1), stride=stride,
                             padding=(0, 1, 0), bias=False, indice_key=indice_key)


def conv3x1x1(in_planes, out_planes, stride=1, indice_key=None):
    return SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride,
                             padding=(1, 0, 0), bias=False, indice_key=indice_key)


def conv3x1(in_planes, out_planes, stride=1, indice_key=None):
    return SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 3), stride=stride,
                             padding=(1, 0, 1), bias=False, indice_key=indice_key)


def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    return SubMConv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None, norm_cfg=None):
        super(ResContextBlock, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.conv1 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0 = build_norm_layer(norm_cfg, out_filters)[1]
        self.act1 = nn.LeakyReLU()

        self.conv1_2 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef3x1")
        self.bn0_2 = build_norm_layer(norm_cfg, out_filters)[1]
        self.act1_2 = nn.LeakyReLU()

        self.conv2 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef3x1")
        self.act2 = nn.LeakyReLU()
        self.bn1 = build_norm_layer(norm_cfg, out_filters)[1]

        self.conv3 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act3 = nn.LeakyReLU()
        self.bn2 = build_norm_layer(norm_cfg, out_filters)[1]

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = replace_feature(shortcut, self.act1(shortcut.features))
        shortcut = replace_feature(shortcut, self.bn0(shortcut.features))

        shortcut = self.conv1_2(shortcut)
        shortcut = replace_feature(shortcut, self.act1_2(shortcut.features))
        shortcut = replace_feature(shortcut, self.bn0_2(shortcut.features))

        resA = self.conv2(x)
        resA = replace_feature(resA, self.act2(resA.features))
        resA = replace_feature(resA, self.bn1(resA.features))

        resA = self.conv3(resA)
        resA = replace_feature(resA, self.act3(resA.features))
        resA = replace_feature(resA, self.bn2(resA.features))
        resA = replace_feature(resA, resA.features + shortcut.features)

        return resA


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3, 3), stride=1,
                 pooling=True, drop_out=True, height_pooling=False, indice_key=None, norm_cfg=None):
        super(ResBlock, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pooling = pooling
        self.drop_out = drop_out

        self.conv1 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.act1 = nn.LeakyReLU()
        self.bn0 = build_norm_layer(norm_cfg, out_filters)[1]

        self.conv1_2 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef1x3")
        self.act1_2 = nn.LeakyReLU()
        self.bn0_2 = build_norm_layer(norm_cfg, out_filters)[1]

        self.conv2 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef1x3")
        self.act2 = nn.LeakyReLU()
        self.bn1 = build_norm_layer(norm_cfg, out_filters)[1]

        self.conv3 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act3 = nn.LeakyReLU()
        self.bn2 = build_norm_layer(norm_cfg, out_filters)[1]

        if pooling:
            if height_pooling:
                self.pool = SparseConv3d(out_filters, out_filters, kernel_size=3, stride=2,
                                                padding=1, indice_key=indice_key, bias=False)
            else:
                self.pool = SparseConv3d(out_filters, out_filters, kernel_size=3, stride=(2, 2, 1),
                                                padding=1, indice_key=indice_key, bias=False)
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = replace_feature(shortcut, self.act1(shortcut.features))
        shortcut = replace_feature(shortcut, self.bn0(shortcut.features))

        shortcut = self.conv1_2(shortcut)
        shortcut = replace_feature(shortcut, self.act1_2(shortcut.features))
        shortcut = replace_feature(shortcut, self.bn0_2(shortcut.features))

        resA = self.conv2(x)
        resA = replace_feature(resA, self.act2(resA.features))
        resA = replace_feature(resA, self.bn1(resA.features))

        resA = self.conv3(resA)
        resA = replace_feature(resA, self.act3(resA.features))
        resA = replace_feature(resA, self.bn2(resA.features))
        resA = replace_feature(resA, resA.features + shortcut.features)

        if self.pooling:
            resB = self.pool(resA)
            return resB, resA
        else:
            return resA


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), indice_key=None, up_key=None, norm_cfg=None):
        super(UpBlock, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.trans_dilao = conv3x3(in_filters, out_filters)
        self.trans_act = nn.LeakyReLU()
        self.trans_bn = build_norm_layer(norm_cfg, out_filters)[1]

        self.conv1 = conv1x3(out_filters, out_filters)
        self.act1 = nn.LeakyReLU()
        self.bn1 = build_norm_layer(norm_cfg, out_filters)[1]

        self.conv2 = conv3x1(out_filters, out_filters)
        self.act2 = nn.LeakyReLU()
        self.bn2 = build_norm_layer(norm_cfg, out_filters)[1]

        self.conv3 = conv3x3(out_filters, out_filters)
        self.act3 = nn.LeakyReLU()
        self.bn3 = build_norm_layer(norm_cfg, out_filters)[1]

        self.up_subm = spconv.SparseInverseConv3d(out_filters, out_filters, kernel_size=3, indice_key=up_key,
                                                  bias=False)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, skip):
        upA = self.trans_dilao(x)
        upA = replace_feature(upA, self.trans_act(upA.features))
        upA = replace_feature(upA, self.trans_bn(upA.features))

        ## upsample
        upA = self.up_subm(upA)
        upA = replace_feature(upA, upA.features + skip.features)

        upE = self.conv1(upA)
        upE = replace_feature(upE, self.act1(upE.features))
        upE = replace_feature(upE, self.bn1(upE.features))

        upE = self.conv2(upE)
        upE = replace_feature(upE, self.act2(upE.features))
        upE = replace_feature(upE, self.bn2(upE.features))

        upE = self.conv3(upE)
        upE = replace_feature(upE, self.act3(upE.features))
        upE = replace_feature(upE, self.bn3(upE.features))

        return upE


class ReconBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None, norm_cfg=None):
        super(ReconBlock, self).__init__()
        
        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)
        
        self.conv1 = conv3x1x1(in_filters, out_filters)
        self.bn0 = build_norm_layer(norm_cfg, out_filters)[1]
        self.act1 = nn.Sigmoid()

        self.conv1_2 = conv1x3x1(in_filters, out_filters)
        self.bn0_2 = build_norm_layer(norm_cfg, out_filters)[1]
        self.act1_2 = nn.Sigmoid()

        self.conv1_3 = conv1x1x3(in_filters, out_filters)
        self.bn0_3 = build_norm_layer(norm_cfg, out_filters)[1]
        self.act1_3 = nn.Sigmoid()

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = replace_feature(shortcut, self.bn0(shortcut.features))
        shortcut = replace_feature(shortcut, self.act1(shortcut.features))

        shortcut2 = self.conv1_2(x)
        shortcut = replace_feature(shortcut, self.bn0_2(shortcut.features))
        shortcut = replace_feature(shortcut, self.act1_2(shortcut.features))

        shortcut3 = self.conv1_3(x)
        shortcut3 = replace_feature(shortcut3, self.bn0_3(shortcut3.features))
        shortcut3 = replace_feature(shortcut3, self.act1_3(shortcut3.features))

        shortcut = replace_feature(shortcut, shortcut.features + shortcut2.features + shortcut3.features)
        shortcut = replace_feature(shortcut, shortcut.features * x.features)

        return shortcut

@BACKBONES.register_module
class Asymm_3d_spconv(nn.Module):
    def __init__(self,
                 num_input_features=128, norm_cfg=None,
                 name="Asymm_3d_spconv",
                 nclasses=17, init_size=16, **kwargs):
        super(Asymm_3d_spconv, self).__init__()
        self.nclasses = nclasses
        self.name = name
        
        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre", norm_cfg=norm_cfg)
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2", norm_cfg=norm_cfg)
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3", norm_cfg=norm_cfg)
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down4", norm_cfg=norm_cfg)
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down5", norm_cfg=norm_cfg)

        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5", norm_cfg=norm_cfg)
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4", norm_cfg=norm_cfg)
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3", norm_cfg=norm_cfg)

        self.logits = SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)

        self.bev_conv = spconv.SparseSequential(
            SparseConv3d(
                256, 256, (1, 1, 3), (1, 1, 2), padding=(0, 0, 1), bias=False
            ),  # [180, 180, 10] -> [180, 180, 5]
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            SparseConv3d(
                256, 128, (1, 1, 3), (1, 1, 2), bias=False
            ),  # [180, 180, 5] -> [180, 180, 2]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size, input_shape):
        sparse_shape = np.array(input_shape)

        coors = coors[:, [0, 3, 2, 1]].int()
        ret_sp = spconv.SparseConvTensor(voxel_features, coors, sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret_sp)
        down1c, down1b = self.resBlock2(ret)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down4c, down4b = self.resBlock5(down3c)

        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        
        logits = self.logits(up2e)
        
        feat_bev = self.bev_conv(down4b)
        out_bev = feat_bev.dense().permute((0, 1, 4, 3, 2)).contiguous()
        N, C, D, H, W = out_bev.shape
        out_bev = out_bev.view(N, C * D, H, W)

        return out_bev, logits
