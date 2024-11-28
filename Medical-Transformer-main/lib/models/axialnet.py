import pdb
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .utils import *
import pdb
import matplotlib.pyplot as plt

import random
import os

# Function to save feature maps
def save_feature_map(tensor, name, save_dir="visualizations"):
    """
    Save feature maps to the specified directory.
    :param tensor: Feature map tensor to save.
    :param name: Prefix for the filename.
    :param save_dir: Directory to save the images.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Move tensor to CPU and extract the first sample
    feature_map = tensor.detach().cpu().numpy()[0]

    # Generate images for each channel
    for i, channel in enumerate(feature_map):
        plt.imshow(channel, cmap='viridis')  # Use Viridis colormap
        plt.axis('off')  # Hide axes
        plt.savefig(os.path.join(save_dir, f"{name}_channel_{i}.png"))
        plt.close()

# SEFusion module to replace torch.add()
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

class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, dilation=2, bias=None, modulation=False,
                 adaptive_d=True):
        super(DeformConv2d, self).__init__()
        self.outc = outc
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.adaptive_d = adaptive_d

        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0.5)
            self.m_conv.register_backward_hook(self._set_lr)

        if self.adaptive_d:
            self.ad_conv = nn.Conv2d(inc, kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.ad_conv.weight, 1)
            self.ad_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.adaptive_d:
            ad_base = self.ad_conv(x)
            ad_base = 1 - torch.sigmoid(ad_base)
            ad = ad_base.repeat(1, 2 * self.kernel_size, 1, 1) * self.dilation

        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        if self.adaptive_d:
            p = self._get_p(offset, dtype, ad)
        else:
            p = self._get_p(offset, dtype, None)

        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = Variable(p.data, requires_grad=False).floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)

        mask = torch.cat([p[..., :N].lt(self.padding) + p[..., :N].gt(x.size(2) - 1 - self.padding),
                          p[..., N:].lt(self.padding) + p[..., N:].gt(x.size(3) - 1 - self.padding)], dim=-1).type_as(p)
        mask = mask.detach()
        floor_p = p - (p - torch.floor(p))
        p = p * (1 - mask) + floor_p * mask
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p(self, offset, dtype, ad_offset):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        p_n = self._get_p_n(N, dtype)
        p_0 = self._get_p_0(h, w, N, dtype)
        if self.adaptive_d:
            p = p_0 + p_n + offset + ad_offset * p_n
        else:
            p = p_0 + p_n + offset
        return p

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        p_n_x = p_n_x.type(dtype)
        p_n_y = p_n_y.type(dtype)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)
        index = q[..., :N] * padded_w + q[..., N:]
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset


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
        #stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
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
        #nn.init.uniform_(self.relative, -0.1, 0.1)
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
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Priority on encoding

        ## Initial values

        self.f_qr = nn.Parameter(torch.tensor(0.1),  requires_grad=False)
        self.f_kr = nn.Parameter(torch.tensor(0.1),  requires_grad=False)
        self.f_sve = nn.Parameter(torch.tensor(0.1),  requires_grad=False)
        self.f_sv = nn.Parameter(torch.tensor(1.0),  requires_grad=False)


        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()
        # self.print_para()

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2, self.kernel_size, self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)


        # multiply by factors
        qr = torch.mul(qr, self.f_qr)
        kr = torch.mul(kr, self.f_kr)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        #stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)

        # multiply by factors
        sv = torch.mul(sv, self.f_sv)
        sve = torch.mul(sve, self.f_sve)

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
        #nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))

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
        self.bn_similarity = nn.BatchNorm2d(groups )

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
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        qk = torch.einsum('bgci, bgcj->bgij', q, k)

        stacked_similarity = self.bn_similarity(qk).reshape(N * W, 1, self.groups, H, H).sum(dim=1).contiguous()

        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)

        sv = sv.reshape(N*W,self.out_planes * 1, H).contiguous()
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
        #nn.init.uniform_(self.relative, -0.1, 0.1)
        # nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))

#end of attn definition

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
        self.width_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)
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
        self.width_block = AxialAttention_dynamic(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)
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
        self.conv1 = nn.Conv2d(width, width, kernel_size = 1)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention_wopos(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention_wopos(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)
        self.conv_up = DeformConv2d(width, planes * self.expansion, kernel_size=3, stride=1, padding=1, modulation=True)
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


#end of block definition


import torch
import torch.nn as nn
import torch.nn.functional as F

# 请确保 DeformConv2d 类已正确定义
# 如果尚未定义，请参考之前的回答或使用 torchvision 的 deform_conv2d

class ResAxialAttentionUNet(nn.Module):

    def __init__(self, block, layers, num_classes=2, zero_init_residual=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.125, img_size=256, imgchan=3):
        super(ResAxialAttentionUNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.img_size = img_size  # 更新 img_size
        self.inplanes = int(64 * s)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # 编码器部分，使用 DeformConv2d
        self.conv1 = nn.Conv2d(imgchan, self.inplanes, kernel_size=7, stride=1, padding=3,
                                  bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(self.inplanes, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.bn2 = norm_layer(128)
        self.bn3 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # 编码器层
        self.layer1 = self._make_layer(block, int(128 * s), layers[0], kernel_size=(img_size // 2))
        self.layer2 = self.DEF_make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=(img_size // 2),
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, int(512 * s), layers[2], stride=2, kernel_size=(img_size // 4),
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self.DEF_make_layer(block, int(1024 * s), layers[3], stride=2, kernel_size=(img_size // 8),
                                       dilate=replace_stride_with_dilation[2])

        # 解码器部分，调整 stride
        # 解码器部分，部分使用 DeformConv2d
        self.decoder1 = DeformConv2d(int(1024 * 2 * s), int(1024 * 2 * s), kernel_size=3, stride=1, padding=1)
        self.decoder2 = DeformConv2d(int(1024 * 2 * s), int(1024 * s), kernel_size=3, stride=1, padding=1)
        self.decoder3 = DeformConv2d(int(1024 * s), int(512 * s), kernel_size=3, stride=1, padding=1)
        self.decoder4 = DeformConv2d(int(512 * s), int(256 * s), kernel_size=3, stride=1, padding=1)
        self.decoder5 = DeformConv2d(int(256 * s), int(128 * s), kernel_size=3, stride=1, padding=1)
        self.decoderf = DeformConv2d(int(128 * s), int(128 * s), kernel_size=3, stride=1, padding=1)

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
            # Use a smaller kernel size for downsampling, e.g., kernel_size=3
            small_kernel_size = 3
            downsample = nn.Sequential(
                DeformConv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=small_kernel_size,
                    stride=stride,
                    padding=small_kernel_size // 2,
                    bias=False,
                    modulation=True
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

    import time

    def _forward_impl(self, x):
        # 字典存储每一部分的时间消耗
        layer_times = {}

        # 编码器部分
        start_time = time.time()
        x = self.conv1(x)


        start_time = time.time()
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        start_time = time.time()
        x = self.conv2(x)


        start_time = time.time()
        x = self.bn2(x)
        x = self.relu(x)


        start_time = time.time()
        x = self.conv3(x)


        start_time = time.time()
        x = self.bn3(x)
        x = self.relu(x)


        start_time = time.time()
        x1 = self.layer1(x)  # 尺寸：128 x 128


        start_time = time.time()
        x2 = self.layer2(x1)  # 尺寸：64 x 64


        start_time = time.time()
        x3 = self.layer3(x2)  # 尺寸：32 x 32


        start_time = time.time()
        x4 = self.layer4(x3)  # 尺寸：16 x 16


        # 解码器部分，调整 stride 并确保尺寸匹配
        start_time = time.time()
        x = F.relu(F.interpolate(self.decoder1(x4), scale_factor=2, mode='bilinear', align_corners=False))  # 尺寸：32 x 32
        x4_upsampled = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)  # 尺寸：32 x 32
        x = x + x4_upsampled
        x = self.relu(x)
        layer_times["decoder1"] = time.time() - start_time

        start_time = time.time()
        x = F.relu(F.interpolate(self.decoder2(x), scale_factor=2, mode='bilinear', align_corners=False))  # 尺寸：64 x 64
        x3_upsampled = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)  # 尺寸：64 x 64
        x = x + x3_upsampled
        x = self.relu(x)
        layer_times["decoder2"] = time.time() - start_time

        start_time = time.time()
        x = F.relu(
            F.interpolate(self.decoder3(x), scale_factor=2, mode='bilinear', align_corners=False))  # 尺寸：128 x 128
        x2_upsampled = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)  # 尺寸：128 x 128
        x = x + x2_upsampled
        x = self.relu(x)
        layer_times["decoder3"] = time.time() - start_time

        start_time = time.time()
        x = F.relu(
            F.interpolate(self.decoder4(x), scale_factor=2, mode='bilinear', align_corners=False))  # 尺寸：256 x 256
        x1_upsampled = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=False)  # 尺寸：256 x 256
        x = x + x1_upsampled
        x = self.relu(x)
        layer_times["decoder4"] = time.time() - start_time

        start_time = time.time()
        x = F.relu(self.decoder5(x))  # 尺寸：256 x 256
        layer_times["decoder5"] = time.time() - start_time

        start_time = time.time()
        x = self.decoderf(x)
        layer_times["decoderf"] = time.time() - start_time

        start_time = time.time()
        x = self.adjust(x)
        layer_times["adjust"] = time.time() - start_time

        # 打印每一部分的时间消耗
        for layer_name, time_taken in layer_times.items():
            print(f"{layer_name}: {time_taken:.6f} seconds")

        return x

    def forward(self, x):
        return self._forward_impl(x)


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

        # Deformable Convolutions
        self.conv1 = DeformConv2d(imgchan, self.inplanes, kernel_size=7, stride=2, padding=3, modulation=True)
        self.conv2 = DeformConv2d(self.inplanes, 128, kernel_size=3, stride=1, padding=1, modulation=True)
        self.conv3 = DeformConv2d(128, self.inplanes, kernel_size=3, stride=1, padding=1, modulation=True)

        self.bn1 = norm_layer(self.inplanes)
        self.bn2 = norm_layer(128)
        self.bn3 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # 主路径中的层
        self.layer1 = self._make_layer(block, int(128 * s), layers[0], kernel_size=(img_size // 2))
        self.layer2 = self._make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=(img_size // 2),
                                       dilate=replace_stride_with_dilation[0])

        # 解码器
        self.decoder4 = DeformConv2d(int(512 * s), int(256 * s), kernel_size=3, stride=1, padding=1, modulation=True)
        self.decoder5 = DeformConv2d(int(256 * s), int(128 * s), kernel_size=3, stride=1, padding=1, modulation=True)
        self.adjust = nn.Conv2d(int(128 * s), num_classes, kernel_size=1, stride=1, padding=0)

        # 本地路径（Local Path）
        self.conv1_p = DeformConv2d(imgchan, self.inplanes, kernel_size=7, stride=2, padding=3, modulation=True)
        self.conv2_p = DeformConv2d(self.inplanes, 128, kernel_size=3, stride=1, padding=1, modulation=True)
        self.conv3_p = DeformConv2d(128, self.inplanes, kernel_size=3, stride=1, padding=1, modulation=True)

        self.bn1_p = norm_layer(self.inplanes)
        self.bn2_p = norm_layer(128)
        self.bn3_p = norm_layer(self.inplanes)
        self.relu_p = nn.ReLU(inplace=True)

        img_size_p = img_size // 4

        # 本地路径中的层
        # 本地路径中的层
        self.layer1_p = self.DEF_make_layer(block_2, int(128 * s), layers[0], kernel_size=3)
        self.layer2_p = self.DEF_make_layer(block_2, int(256 * s), layers[1], stride=2, kernel_size=3,
                                            dilate=replace_stride_with_dilation[0])
        self.layer3_p = self.DEF_make_layer(block_2, int(512 * s), layers[2], stride=2, kernel_size=3,
                                            dilate=replace_stride_with_dilation[1])
        self.layer4_p = self.DEF_make_layer(block_2, int(1024 * s), layers[3], stride=2, kernel_size=3,
                                            dilate=replace_stride_with_dilation[2])

        self.decoder1_p = DeformConv2d(int(1024 * 2 * s), int(1024 * 2 * s), kernel_size=3, stride=2, padding=1,
                                       modulation=True)
        self.decoder2_p = DeformConv2d(int(1024 * 2 * s), int(1024 * s), kernel_size=3, stride=1, padding=1,
                                       modulation=True)
        self.decoder3_p = DeformConv2d(int(1024 * s), int(512 * s), kernel_size=3, stride=1, padding=1,
                                       modulation=True)
        self.decoder4_p = DeformConv2d(int(512 * s), int(256 * s), kernel_size=3, stride=1, padding=1,
                                       modulation=True)
        self.decoder5_p = DeformConv2d(int(256 * s), int(128 * s), kernel_size=3, stride=1, padding=1,
                                       modulation=True)

        self.decoderf = DeformConv2d(int(128 * s), int(128 * s), kernel_size=3, stride=1, padding=1,
                                     modulation=True)
        self.adjust_p = nn.Conv2d(int(128 * s), num_classes, kernel_size=1, stride=1, padding=0)

        # SE Fusion模块，用于替换torch.add()
        self.sefusion1 = SEFusion(channels=int(256 * s))
        self.sefusion_p1 = SEFusion(channels=int(1024 * 2 * s))
        self.sefusion_p2 = SEFusion(channels=int(1024 * s))
        self.sefusion_p3 = SEFusion(channels=int(512 * s))
        self.sefusion_p4 = SEFusion(channels=int(256 * s))
        self.sefusion_final = SEFusion(channels=int(128 * s))

    def DEF_make_layer(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            # Use a smaller kernel size for downsampling, e.g., kernel_size=3
            small_kernel_size = 3
            downsample = nn.Sequential(
                DeformConv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=small_kernel_size,
                    stride=stride,
                    padding=small_kernel_size // 2,
                    bias=False,
                    modulation=True
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
        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)

        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)

        x = self.bn3(x)
        x = self.relu(x)

        x1 = self.layer1(x)

        x2 = self.layer2(x1)

        # 全局路径的上采样和可视化
        x = F.relu(F.interpolate(self.decoder4(x2), scale_factor=(2, 2), mode='bilinear'))

        # 使用 SEFusion 替代 torch.add(x, x1)
        x = self.sefusion1(x, x1)
        x = F.relu(F.interpolate(self.decoder5(x), scale_factor=(2, 2), mode='bilinear'))

        x_loc = x.clone()
        for i in range(0, 4):
            for j in range(0, 4):
                x_p = xin[:, :, 32 * i:32 * (i + 1), 32 * j:32 * (j + 1)]
                x_p = self.conv1_p(x_p)
                # 保存本地分支的特征图

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


                # 本地分支的上采样和可视化
                x_p = F.relu(F.interpolate(self.decoder1_p(x4_p), scale_factor=(2, 2), mode='bilinear'))

                # 使用 SEFusion 替代 torch.add(x_p, x4_p)
                x_p = self.sefusion_p1(x_p, x4_p)
                x_p = F.relu(F.interpolate(self.decoder2_p(x_p), scale_factor=(2, 2), mode='bilinear'))

                x_p = self.sefusion_p2(x_p, x3_p)
                x_p = F.relu(F.interpolate(self.decoder3_p(x_p), scale_factor=(2, 2), mode='bilinear'))

                x_p = self.sefusion_p3(x_p, x2_p)
                x_p = F.relu(F.interpolate(self.decoder4_p(x_p), scale_factor=(2, 2), mode='bilinear'))

                x_p = self.sefusion_p4(x_p, x1_p)
                x_p = F.relu(F.interpolate(self.decoder5_p(x_p), scale_factor=(2, 2), mode='bilinear'))

                x_loc[:, :, 32 * i:32 * (i + 1), 32 * j:32 * (j + 1)] = x_p
        # 使用 SEFusion 替代 torch.add(x, x_loc)
        x = self.sefusion_final(x, x_loc)
        save_feature_map(x, "fusion_output")
        x = F.relu(self.decoderf(x))
        save_feature_map(x, "decoderf_output")

        x = self.adjust(F.relu(x))
        save_feature_map(x, "final_output")

        return x

    def forward(self, x):
        return self._forward_impl(x)


def axialunet(pretrained=False, **kwargs):
    model = ResAxialAttentionUNet(AxialBlock, [1, 2, 4, 1], s= 0.125, **kwargs)
    return model

def gated(pretrained=False, **kwargs):
    model = ResAxialAttentionUNet(AxialBlock_dynamic, [1, 2, 4, 1], s= 0.125, **kwargs)
    return model

def MedT(pretrained=False, **kwargs):
    model = medt_net(AxialBlock_dynamic,AxialBlock_wopos, [1, 2, 4, 1], s= 0.125,  **kwargs)
    return model

def logo(pretrained=False, **kwargs):
    model = medt_net(AxialBlock,AxialBlock, [1, 2, 4, 1], s= 0.125, **kwargs)
    return model

# EOF