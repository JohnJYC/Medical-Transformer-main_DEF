import math

from .utils import *
import pdb
import matplotlib.pyplot as plt
from DCNv4.modules.dcnv4 import DCNv4  # 导入新的 DCNv4
from DCNv4 import ext
import torch.nn as nn
from DCNv4.functions.dcnv4_func import DCNv4Function


import random
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def save_feature_map(feature_map, name, save_dir="output_visualizations"):
    """
    Save or visualize the feature maps.
    :param feature_map: The tensor of feature maps.
    :param name: Name prefix for saved images.
    :param save_dir: Directory to save the images.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    feature_map = feature_map.detach().cpu().numpy()[0]  # Get the first batch
    for i, channel in enumerate(feature_map):
        plt.imshow(channel, cmap='viridis')
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f"{name}_channel_{i}.png"))
        plt.close()


# SEFusion module to replace torch.add(
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)

        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        # pdb.set_trace()
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H),
                              [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2,
                                                                                       self.kernel_size,
                                                                                       self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings,
                                                            [self.group_planes // 2, self.group_planes // 2,
                                                             self.group_planes], dim=0)

        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)

        qk = torch.einsum('bgci, bgcj->bgij', q, k)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        # stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        # nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))


class AxialAttention_dynamic(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention_dynamic, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1,
                                           stride=1, padding=0, bias=False)
        self.bn_qkv        = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output     = nn.BatchNorm1d(out_planes * 2)

        # static weights for fusion
        self.f_qr  = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.f_kr  = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.f_sve = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.f_sv  = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        # positional embedding table (max size = 2*K−1)
        self.relative = nn.Parameter(torch.randn(self.group_planes*2, kernel_size*2 - 1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        # rearrange to (N, W, C, H) or (N, H, C, W)
        if self.width:
            x = x.permute(0,2,1,3)
        else:
            x = x.permute(0,3,1,2)
        N, W, C, H = x.shape
        x = x.contiguous().view(N*W, C, H)

        # qkv projection + BN
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(
            qkv.view(N*W, self.groups, self.group_planes*2, H),
            [self.group_planes//2, self.group_planes//2, self.group_planes],
            dim=2
        )

        # --- dynamic position embedding for current H ---
        device = x.device
        idx_q = torch.arange(H, device=device).view(1, -1)
        idx_k = torch.arange(H, device=device).view(-1, 1)
        rel_idx = (idx_k - idx_q + H - 1).long().view(-1)           # (H*H,)
        all_emb = self.relative.index_select(1, rel_idx)           # (2Cg, H*H)
        all_emb = all_emb.view(self.group_planes*2, H, H)          # (2Cg, H, H)
        q_emb, k_emb, v_emb = torch.split(
            all_emb,
            [self.group_planes//2, self.group_planes//2, self.group_planes],
            dim=0
        )

        # content + position
        qr = torch.einsum('bgci,cij->bgij', q, q_emb)
        kr = torch.einsum('bgci,cij->bgij', k, k_emb).transpose(2,3)
        qk = torch.einsum('bgci,bgcj->bgij', q, k)

        qr = qr * self.f_qr
        kr = kr * self.f_kr

        # similarity and attention
        sim = torch.cat([qk, qr, kr], dim=1)
        sim = self.bn_similarity(sim).view(N*W, 3, self.groups, H, H).sum(dim=1)
        attn = F.softmax(sim, dim=3)

        sv  = torch.einsum('bgij,bgcj->bgci', attn, v)
        sve = torch.einsum('bgij,cij->bgci', attn, v_emb)

        sv  = sv  * self.f_sv
        sve = sve * self.f_sve

        out = torch.cat([sv, sve], dim=2).view(N*W, self.out_planes*2, H)
        out = self.bn_output(out)
        out = out.view(N, W, self.out_planes, 2, H).sum(dim=3)

        if self.width:
            out = out.permute(0,2,1,3)
        else:
            out = out.permute(0,2,3,1)

        if self.stride > 1:
            out = self.pooling(out)

        return out

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1./self.in_planes))
        nn.init.normal_(self.relative, 0., math.sqrt(1./self.group_planes))



class AxialAttention_wopos(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention_wopos, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups)

        self.bn_output = nn.BatchNorm1d(out_planes * 1)

        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H),
                              [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        qk = torch.einsum('bgci, bgcj->bgij', q, k)

        stacked_similarity = self.bn_similarity(qk).reshape(N * W, 1, self.groups, H, H).sum(dim=1).contiguous()

        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)

        sv = sv.reshape(N * W, self.out_planes * 1, H).contiguous()
        output = self.bn_output(sv).reshape(N, W, self.out_planes, 1, H).sum(dim=-2).contiguous()

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        # nn.init.uniform_(self.relative, -0.1, 0.1)
        # nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))


class AxialBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(AxialBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size, stride=stride,
                                          width=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print(out.shape)
        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AxialBlock_dynamic(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(AxialBlock_dynamic, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention_dynamic(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention_dynamic(width, width, groups=groups, kernel_size=kernel_size, stride=stride,
                                                  width=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AxialBlock_wopos(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(AxialBlock_wopos, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # print(kernel_size)
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.conv1 = nn.Conv2d(width, width, kernel_size=1)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention_wopos(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention_wopos(width, width, groups=groups, kernel_size=kernel_size, stride=stride,
                                                width=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # pdb.set_trace()

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print(out.shape)
        out = self.hight_block(out)
        out = self.width_block(out)

        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class DeformConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(DeformConvBlock, self).__init__()
        # 如果 in_channels 不能被 groups 整除，则将 groups 设为 1
        if in_channels % groups != 0:
            groups = 1
        self.groups = groups
        self.offset_channels = 2 * self.groups * kernel_size * kernel_size
        self.offset_conv = nn.Conv2d(
            in_channels,
            self.offset_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.deform_conv = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=self.groups,
            bias=False
        )

    def forward(self, x):
        offset = self.offset_conv(x)
        out = self.deform_conv(x, offset)
        return out


class ResAxialAttentionUNet(nn.Module):
    def __init__(self, block, layers, num_classes=2, zero_init_residual=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.125, img_size=256, imgchan=3):
        super(ResAxialAttentionUNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.img_size = img_size  # Update image size
        self.inplanes = int(64 * s)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group

        # Encoder with reduced DeformConv2d usage
        self.conv1 = nn.Conv2d(imgchan, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(self.inplanes, 128, kernel_size=3, stride=1, padding=1, bias=False)  # Standard Conv
        self.conv3 = nn.Conv2d(128, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)  # Standard Conv

        self.bn1 = norm_layer(self.inplanes)
        self.bn2 = norm_layer(128)
        self.bn3 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # Encoder layers
        self.layer1 = self.DEF_make_layer(block, int(128 * s), layers[0], kernel_size=(img_size // 2))
        self.layer2 = self.DEF_make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=(img_size // 2),
                                          dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, int(512 * s), layers[2], stride=2, kernel_size=(img_size // 4),
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self.DEF_make_layer(block, int(1024 * s), layers[3], stride=2, kernel_size=(img_size // 8),
                                          dilate=replace_stride_with_dilation[2])

        # Decoder with partial DeformConv2d usage
        self.decoder1 = DeformConvBlock(int(1024 * 2 * s), int(1024 * 2 * s), kernel_size=3, stride=1, padding=1)
        self.decoder2 = DeformConvBlock(int(1024 * 2 * s), int(1024 * s), kernel_size=3, stride=1, padding=1)
        self.decoder3 = DeformConvBlock(int(1024 * s), int(512 * s), kernel_size=3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(int(512 * s), int(256 * s), kernel_size=3, stride=1, padding=1)  # Standard Conv
        self.decoder5 = nn.Conv2d(int(256 * s), int(128 * s), kernel_size=3, stride=1, padding=1)  # Standard Conv
        self.decoderf = nn.Conv2d(int(128 * s), int(128 * s), kernel_size=3, stride=1, padding=1)  # Standard Conv

        self.adjust = nn.Conv2d(int(128 * s), num_classes, kernel_size=1, stride=1, padding=0)
        self.soft = nn.Softmax(dim=1)

    def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, padding=0, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation,
                            norm_layer=norm_layer, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def DEF_make_layer(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            small_kernel_size = 3
            downsample = nn.Sequential(
                DeformConvBlock(
                    in_channels=self.inplanes,
                    out_channels=planes * block.expansion,
                    kernel_size=small_kernel_size,
                    stride=stride,
                    padding=small_kernel_size // 2,
                ),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                groups=self.groups,
                base_width=self.base_width,
                dilation=previous_dilation,
                norm_layer=norm_layer,
                kernel_size=kernel_size
            )
        )
        self.inplanes = planes * block.expansion

        if stride != 1:
            kernel_size = kernel_size // 2  # Adjust kernel_size based on downsampling

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    kernel_size=kernel_size
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x1 = self.layer1(x)  # Size: 128x128
        x2 = self.layer2(x1)  # Size: 64x64
        x3 = self.layer3(x2)  # Size: 32x32
        x4 = self.layer4(x3)  # Size: 16x16

        # Decoder
        x = F.relu(
            F.interpolate(self.decoder1(x4), scale_factor=2, mode='bilinear', align_corners=False))  # Size: 32x32
        x4_upsampled = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)  # Size: 32x32
        x = x + x4_upsampled
        x = self.relu(x)

        x = F.relu(F.interpolate(self.decoder2(x), scale_factor=2, mode='bilinear', align_corners=False))  # Size: 64x64
        x3_upsampled = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)  # Size: 64x64
        x = x + x3_upsampled
        x = self.relu(x)

        x = F.relu(
            F.interpolate(self.decoder3(x), scale_factor=2, mode='bilinear', align_corners=False))  # Size: 128x128
        x2_upsampled = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)  # Size: 128x128
        x = x + x2_upsampled
        x = self.relu(x)

        x = F.relu(
            F.interpolate(self.decoder4(x), scale_factor=2, mode='bilinear', align_corners=False))  # Size: 256x256
        x1_upsampled = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=False)  # Size: 256x256
        x = x + x1_upsampled
        x = self.relu(x)

        x = F.relu(self.decoder5(x))  # Size: 256x256
        x = self.decoderf(x)
        x = self.adjust(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)

class SEFusion(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.gp = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch*2, ch//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch//reduction, ch*2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        b, c, _, _ = x1.size()
        y = torch.cat([x1, x2], dim=1)
        y = self.gp(y).view(b, -1)
        y = self.fc(y).view(b, 2*c, 1, 1)
        s1, s2 = torch.split(y, c, dim=1)
        return x1 * s1 + x2 * s2


import torch
import torch.nn as nn
import torch.nn.functional as F


class SEFusion(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEFusion, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels * 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        b, c, h, w = x1.size()
        x = torch.cat([x1, x2], dim=1)  # Concatenate along channel dimension
        y = self.global_pool(x).view(b, -1)
        y = self.fc(y).view(b, 2 * c, 1, 1)
        s1, s2 = torch.split(y, c, dim=1)
        x1 = x1 * s1
        x2 = x2 * s2
        out = x1 + x2
        return out


class DCNv4Block(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1,
                 groups=1, dilation=1,
                 offset_scale=1.0, remove_center=False):
        super().__init__()
        if in_channels % groups != 0: groups = 1
        self.groups = groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.offset_scale = offset_scale
        self.remove_center = remove_center
        self.group_channels = in_channels // groups
        pts = kernel_size*kernel_size - (1 if remove_center else 0)
        mask_ch = pts * groups * 3
        self.offset_mask_conv = nn.Conv2d(in_channels, mask_ch,
                                          kernel_size, stride, padding)
        nn.init.constant_(self.offset_mask_conv.weight, 0)
        nn.init.constant_(self.offset_mask_conv.bias, 0)
        self.im2col_step = 64
        self.output_proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        offset_mask = self.offset_mask_conv(x)
        out = DCNv4Function.apply(
            x, offset_mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.padding, self.padding,
            self.dilation, self.dilation,
            self.groups, self.group_channels,
            self.offset_scale, self.im2col_step,
            self.remove_center
        )
        return self.output_proj(out)

class medt_net(nn.Module):
    def __init__(self, block, block_2, layers, num_classes=2, zero_init_residual=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.125, img_size=128, imgchan=3):
        super(medt_net, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = int(64 * s)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group

        # Encoder
        self.conv1 = nn.Conv2d(imgchan, self.inplanes, kernel_size=7, stride=1, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = DCNv4Block(self.inplanes, 128, kernel_size=3, stride=1, padding=1, groups=self.groups)
        self.conv3 = nn.Conv2d(128, self.inplanes, kernel_size=3, stride=1, padding=1)

        self.bn1 = norm_layer(self.inplanes)
        self.bn2 = norm_layer(128)
        self.bn3 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # Main Path Layers
        self.layer1 = self._make_layer(block, int(128 * s), layers[0], kernel_size=(img_size // 2))
        self.layer2 = self._make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=(img_size // 2),
                                       dilate=replace_stride_with_dilation[0])

        # Decoder
        self.decoder4 = DCNv4Block(int(512 * s), int(256 * s), kernel_size=3, stride=1, padding=1, groups=self.groups)
        self.decoder5 = DCNv4Block(int(256 * s), int(128 * s), kernel_size=3, stride=1, padding=1, groups=self.groups)
        self.adjust = nn.Conv2d(int(128 * s), num_classes, kernel_size=1, stride=1, padding=0)

        # Local Path
        self.conv1_p = DCNv4Block(imgchan, self.inplanes, kernel_size=7, stride=2, padding=3, groups=self.groups)
        self.conv2_p = nn.Conv2d(self.inplanes, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_p = nn.Conv2d(128, self.inplanes, kernel_size=3, stride=1, padding=1)

        self.bn1_p = norm_layer(self.inplanes)
        self.bn2_p = norm_layer(128)
        self.bn3_p = norm_layer(self.inplanes)
        self.relu_p = nn.ReLU(inplace=True)

        img_size_p = img_size // 4

        # Local Path Layers
        self.layer1_p = self.DEF_make_layer(block_2, int(128 * s), layers[0], kernel_size=3)
        self.layer2_p = self.DEF_make_layer(block_2, int(256 * s), layers[1], stride=2, kernel_size=3,
                                            dilate=replace_stride_with_dilation[0])
        self.layer3_p = self._make_layer(block_2, int(512 * s), layers[2], stride=2, kernel_size=3,
                                         dilate=replace_stride_with_dilation[1])
        self.layer4_p = self.DEF_make_layer(block_2, int(1024 * s), layers[3], stride=2, kernel_size=3,
                                            dilate=replace_stride_with_dilation[2])

        self.decoder1_p = DCNv4Block(int(1024 * 2 * s), int(1024 * 2 * s), kernel_size=3, stride=2, padding=1,
                                     groups=self.groups)
        self.decoder2_p = DCNv4Block(int(1024 * 2 * s), int(1024 * s), kernel_size=3, stride=1, padding=1,
                                     groups=self.groups)
        self.decoder3_p = DCNv4Block(int(1024 * s), int(512 * s), kernel_size=3, stride=1, padding=1,
                                     groups=self.groups)
        self.decoder4_p = nn.Conv2d(int(512 * s), int(256 * s), kernel_size=3, stride=1, padding=1)
        self.decoder5_p = nn.Conv2d(int(256 * s), int(128 * s), kernel_size=3, stride=1, padding=1)

        self.decoderf = nn.Conv2d(int(128 * s), int(128 * s), kernel_size=3, stride=1, padding=1)
        self.adjust_p = nn.Conv2d(int(128 * s), num_classes, kernel_size=1, stride=1, padding=0)

        # SE Fusion Module
        self.sefusion_final = SEFusion(channels=int(128 * s))

    def DEF_make_layer(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            small_kernel_size = 3
            downsample = nn.Sequential(
                DCNv4Block(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=small_kernel_size,
                    stride=stride,
                    padding=small_kernel_size // 2,
                    groups=self.groups
                ),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                groups=self.groups,
                base_width=self.base_width,
                dilation=previous_dilation,
                norm_layer=norm_layer,
                kernel_size=kernel_size
            )
        )
        self.inplanes = planes * block.expansion

        if stride != 1:
            kernel_size = kernel_size // 2  # Adjust kernel_size based on downsampling

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    kernel_size=kernel_size
                )
            )

        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, padding=0, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation,
                            norm_layer=norm_layer, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        xin = x.clone()

        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)

        # Decoder
        x = F.relu(F.interpolate(self.decoder4(x2), scale_factor=2, mode='bilinear', align_corners=False))
        x = F.relu(F.interpolate(self.decoder5(x), scale_factor=2, mode='bilinear', align_corners=False))

        x_loc = x.clone()

        # Local Path Processing
        for i in range(0, 4):
            for j in range(0, 4):
                x_p = xin[:, :, 32 * i:32 * (i + 1), 32 * j:32 * (j + 1)]
                x_p = self.conv1_p(x_p)
                x_p = self.bn1_p(x_p)
                x_p = self.relu_p(x_p)
                x_p = self.conv2_p(x_p)
                x_p = self.bn2_p(x_p)
                x_p = self.relu_p(x_p)
                x_p = self.conv3_p(x_p)
                x_p = self.bn3_p(x_p)
                x_p = self.relu_p(x_p)
                x1_p = self.layer1_p(x_p)
                x2_p = self.layer2_p(x1_p)
                x3_p = self.layer3_p(x2_p)
                x4_p = self.layer4_p(x3_p)

                x_p = F.relu(F.interpolate(self.decoder1_p(x4_p), scale_factor=2, mode='bilinear', align_corners=False))
                x_p = F.relu(F.interpolate(self.decoder2_p(x_p), scale_factor=2, mode='bilinear', align_corners=False))
                x_p = F.relu(F.interpolate(self.decoder3_p(x_p), scale_factor=2, mode='bilinear', align_corners=False))
                x_p = F.relu(F.interpolate(self.decoder4_p(x_p), scale_factor=2, mode='bilinear', align_corners=False))
                x_p = F.relu(F.interpolate(self.decoder5_p(x_p), scale_factor=2, mode='bilinear', align_corners=False))

                x_loc[:, :, 32 * i:32 * (i + 1), 32 * j:32 * (j + 1)] = x_p

        # Fusion
        x = self.sefusion_final(x, x_loc)
        x = F.relu(self.decoderf(x))
        x = self.adjust(F.relu(x))

        return x

    def forward(self, x):
        return self._forward_impl(x)


def forward(self, x):
    return self._forward_impl(x)


def axialunet(pretrained=False, **kwargs):
    model = ResAxialAttentionUNet(AxialBlock, [1, 2, 4, 1], s=0.125, **kwargs)
    return model


def gated(pretrained=False, **kwargs):
    model = ResAxialAttentionUNet(AxialBlock_dynamic, [1, 2, 4, 1], s=0.125, **kwargs)
    return model


def MedT(pretrained=False, **kwargs):
    model = medt_net(AxialBlock_dynamic, AxialBlock_wopos, [1, 2, 4, 1], s=0.125, **kwargs)
    return model


def logo(pretrained=False, **kwargs):
    model = medt_net(AxialBlock, AxialBlock, [1, 2, 4, 1], s=0.125, **kwargs)
    return model

# EOF
