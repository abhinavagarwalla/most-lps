import numpy as np
try:
    import spconv.pytorch as spconv 
    from spconv.pytorch import ops
    from spconv.pytorch import SparseConv3d, SubMConv3d
except: 
    import spconv 
    from spconv import ops
    from spconv import SparseConv3d, SubMConv3d

from torch import nn
from torch.nn import functional as F

from ..registry import BACKBONES
from ..utils import build_norm_layer

def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out

def conv3x3(in_planes, out_planes, stride=1, indice_key=None, bias=True):
    """3x3 convolution with padding"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=bias,
        indice_key=indice_key,
    )


def conv1x1(in_planes, out_planes, stride=1, indice_key=None, bias=True):
    """1x1 convolution"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=1,
        bias=bias,
        indice_key=indice_key,
    )


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_cfg=None,
        downsample=None,
        indice_key=None,
    ):
        super(SparseBasicBlock, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None

        self.conv1 = conv3x3(inplanes, planes, stride, indice_key=indice_key, bias=bias)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, indice_key=indice_key, bias=bias)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


@BACKBONES.register_module
class SpMiddleResNetFHD(nn.Module):
    def __init__(
        self, num_input_features=128, norm_cfg=None, name="SpMiddleResNetFHD", **kwargs
    ):
        super(SpMiddleResNetFHD, self).__init__()
        self.name = name

        self.dcn = None
        self.zero_init_residual = False

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        # input: # [1440, 1440, 41]
        self.conv_input = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, bias=False, indice_key="res0"),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(inplace=True)
        )

        self.conv1 = spconv.SparseSequential(        
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
        )

        self.conv2 = spconv.SparseSequential(
            SparseConv3d(
                16, 32, 3, 2, padding=1, bias=False
            ),  # [1440, 1440, 41] -> [720, 720, 21]
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(inplace=True),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv3 = spconv.SparseSequential(
            SparseConv3d(
                32, 64, 3, 2, padding=1, bias=False
            ),  # [720, 720, 21] -> [360, 360, 11]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(inplace=True),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv3d(
                64, 128, 3, 2, padding=[0, 1, 1], bias=False
            ),  # [360, 260, 11] -> [180, 180, 5]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(inplace=True),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
        )


        self.extra_conv = spconv.SparseSequential(
            SparseConv3d(
                128, 128, (3, 1, 1), (2, 1, 1), bias=False
            ),  # [200, 150, 5] -> [200, 150, 2]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size, input_shape):

        # input: # [41, 1600, 1408]
        sparse_shape = np.array(input_shape[::-1]) + [0, 0, 1]

        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)

        x = self.conv_input(ret)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        ret = self.extra_conv(x_conv4)

        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)

        multi_scale_voxel_features = {
            'conv1': x_conv1,
            'conv2': x_conv2,
            'conv3': x_conv3,
            'conv4': x_conv4,
        }

        return ret, multi_scale_voxel_features

class SparseUpBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_cfg=None,
        downsample=None,
        indice_key=None,
        up_key=None,
    ):
        super(SparseUpBlock, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None

        self.conv1 = conv3x3(inplanes, planes, stride, indice_key=indice_key, bias=bias)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, indice_key=indice_key, bias=bias)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.downsample = downsample
        self.stride = stride

        # self.up_seq = SparseBasicBlock(inplanes, planes, norm_cfg=norm_cfg, indice_key=up_key+"bef")
        # self.up_seq = spconv.SparseConv3d(inplanes, planes, kernel_size=3, stride=1,
        #                                         padding=1, indice_key=indice_key, bias=False)
        self.up_subm = spconv.SparseInverseConv3d(planes, planes, kernel_size=3, indice_key=up_key,
                                                  bias=False)

    def forward(self, x, skip):
        identity = x

        # up = self.up_seq(x)
        up = self.up_subm(x)
        x = replace_feature(up, up.features + skip.features) # TODO: the UNet way is to concatenate.. but Cylinder3D does addition

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out

class SparseDownBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        pool_kernel_size=3,
        pool_stride=2,
        pool_pad=1,
        stride=1,
        norm_cfg=None,
        downsample=None,
        indice_key=None,
    ):
        super(SparseDownBlock, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None

        self.block1 = SparseBasicBlock(inplanes, inplanes, stride, norm_cfg=norm_cfg, downsample=downsample, indice_key=indice_key)
        self.block2 = SparseBasicBlock(inplanes, inplanes, stride, norm_cfg=norm_cfg, downsample=downsample, indice_key=indice_key)

        self.pool = SparseConv3d(
                inplanes, planes, pool_kernel_size, pool_stride, padding=pool_pad, bias=False
            )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out_pool = self.pool(out)
        return out_pool, out


@BACKBONES.register_module
class SpEncoderDecoderFHD(nn.Module):
    def __init__(
        self, num_input_features=128, norm_cfg=None, name="SpEncoderDecoderFHD", **kwargs
    ):
        super(SpEncoderDecoderFHD, self).__init__()
        self.name = name

        self.dcn = None
        self.zero_init_residual = False

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        # input: # [1440, 1440, 41]
        self.conv_input = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, bias=False, indice_key="res0"),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(inplace=True)
        )

        self.conv1 = SparseDownBlock(inplanes=16, planes=16,
                                    norm_cfg=norm_cfg, indice_key="res0")

        self.conv2 = SparseDownBlock(inplanes=16, planes=32,
                                    norm_cfg=norm_cfg, indice_key="res1")

        self.conv3 = SparseDownBlock(inplanes=32, planes=64,
                                    pool_pad=[0, 1, 1], norm_cfg=norm_cfg, indice_key="res2")

        self.conv4 = SparseDownBlock(inplanes=64, planes=128,
                                    pool_kernel_size=(3, 1, 1), pool_stride=(2, 1, 1),
                                    pool_pad=0,
                                    norm_cfg=norm_cfg, indice_key="res3")


        self.up_conv4 = SparseUpBlock(128, 128, norm_cfg=norm_cfg, indice_key="up3", up_key="res3")
        self.up_conv3 = SparseUpBlock(128, 64, norm_cfg=norm_cfg, indice_key="up2", up_key="res2")
        self.up_conv2 = SparseUpBlock(64, 32, norm_cfg=norm_cfg, indice_key="up1", up_key="res1"),
        self.up_conv1 = SparseUpBlock(32, 16, norm_cfg=norm_cfg, indice_key="up0", up_key="res0")

    def forward(self, voxel_features, coors, batch_size, input_shape):

        # input: # [41, 1600, 1408]
        sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]

        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)

        x = self.conv_input(ret)

        x_conv1p, x_conv1 = self.conv1(x)
        x_conv2p, x_conv2 = self.conv2(x_conv1p)
        x_conv3p, x_conv3 = self.conv3(x_conv2p)
        x_conv4p, x_conv4 = self.conv4(x_conv3p)

        # ret = self.extra_conv(x_conv4)

        up_conv4 = self.up_conv4(x_conv4p, x_conv4) #ideally first should be ret. need to set key to None?
        up_conv3 = self.up_conv3(up_conv4, x_conv3)
        up_conv2 = self.up_conv2(up_conv3, x_conv2)
        up_conv1 = self.up_conv1(up_conv2, x_conv1)

        ret_bev = x_conv4p.dense()

        N, C, D, H, W = ret_bev.shape
        ret_bev = ret_bev.view(N, C * D, H, W)

        multi_scale_voxel_features = {
            'conv1': x_conv1,
            'conv2': x_conv2,
            'conv3': x_conv3,
            'conv4': x_conv4,
        }

        return ret_bev, up_conv1, multi_scale_voxel_features
