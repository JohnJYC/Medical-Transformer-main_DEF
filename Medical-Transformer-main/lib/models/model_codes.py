import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 定义可变形卷积层
class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=stride, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        nn.init.constant_(self.p_conv.bias, 0)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            nn.init.constant_(self.m_conv.bias, 0)

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        p = self._get_p(offset, dtype)

        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.clamp(q_lt, 0, x.size(2) - 1)
        q_rb = torch.clamp(q_rb, 0, x.size(2) - 1)
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        p = torch.clamp(p, 0, x.size(2) - 1)

        # 双线性插值权重
        g_lt = (1 + (q_lt[..., :N] - p[..., :N])) * (1 + (q_lt[..., N:] - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N] - p[..., :N])) * (1 - (q_rb[..., N:] - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N] - p[..., :N])) * (1 - (q_lb[..., N:] - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N] - p[..., :N])) * (1 + (q_rt[..., N:] - p[..., N:]))

        # 获取四个邻近点的值
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # 加权求和
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            x_offset = x_offset * m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1)
        )
        p_n = torch.cat((p_n_x.flatten(), p_n_y.flatten()), 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x = torch.arange(0, h * self.stride, self.stride)
        p_0_y = torch.arange(0, w * self.stride, self.stride)
        p_0_x = p_0_x.view(-1, 1).repeat(1, w)
        p_0_y = p_0_y.view(1, -1).repeat(h, 1)
        p_0 = torch.cat((p_0_x.flatten(), p_0_y.flatten()), 0)
        p_0 = p_0.view(1, 2 * N, h, w).type(dtype)
        return p_0

    def _get_p(self, offset, dtype):
        N = offset.size(1) // 2
        h, w = offset.size(2), offset.size(3)
        p_n = self._get_p_n(N, dtype)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, c, h, w = x.size()
        q = q.long()
        x = x.contiguous().view(b, c, -1)

        index = q[..., :N] * w + q[..., N:]
        index = index.contiguous().view(b, -1)
        x_offset = x.gather(dim=2, index=index.unsqueeze(1).expand(-1, c, -1))
        x_offset = x_offset.view(b, c, q.size(2), q.size(3), N)
        return x_offset

    def _reshape_x_offset(self, x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = x_offset.permute(0, 1, 4, 2, 3).contiguous()
        x_offset = x_offset.view(b, c * N, h, w)
        return x_offset

# 定义1x1卷积的快捷函数
def conv1x1(in_planes, out_planes, stride=1):
    """1x1卷积"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# 定义轴向注意力机制
class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56, stride=1, bias=False, width=False):
        super(AxialAttention, self).__init__()
        self.width = width
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_planes = in_planes
        self.out_planes = out_planes

        # 添加可变形卷积层
        self.deform_conv = DeformConv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)

        # QKV变换仍然使用1x1卷积
        self.qkv_transform = nn.Conv1d(in_planes, out_planes * 2, kernel_size=1, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups)
        self.bn_output = nn.BatchNorm1d(out_planes)

        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=2)

        self.reset_parameters()

    def forward(self, x):
        # 可变形卷积增强特征
        x = self.deform_conv(x)

        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # QKV变换
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H),
                              [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # 计算相似度
        qk = torch.einsum('bgci,bgcj->bgij', q, k)
        stacked_similarity = self.bn_similarity(qk).reshape(N * W, 1, self.groups, H, H).sum(dim=1).contiguous()
        similarity = self.softmax(stacked_similarity)

        # 注意力加权
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sv = sv.reshape(N * W, self.out_planes, H).contiguous()
        output = self.bn_output(sv).reshape(N, W, self.out_planes, 1, H).sum(dim=-2).contiguous()

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        nn.init.normal_(self.qkv_transform.weight, 0, math.sqrt(1. / self.in_planes))

# 定义轴向残差块
class AxialBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=8,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(AxialBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))

        # 保持1x1卷积不变
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.relu = nn.ReLU(inplace=True)

        # 添加可变形卷积层
        self.deform_conv = DeformConv2d(width, width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_deform = norm_layer(width)

        # 轴向注意力层
        self.height_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)

        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 可变形卷积
        out = self.deform_conv(out)
        out = self.bn_deform(out)
        out = self.relu(out)

        # 轴向注意力
        out = self.height_block(out)
        out = self.width_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# 定义 ResAxialAttentionUNet
class ResAxialAttentionUNet(nn.Module):

    def __init__(self, block, layers, num_classes=2, zero_init_residual=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.125, img_size=128, imgchan=3):
        super(ResAxialAttentionUNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = int(64 * s)
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        self.groups = groups
        self.base_width = width_per_group

        # 初始可变形卷积层
        self.conv1 = DeformConv2d(imgchan, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # 编码器
        self.layer1 = self._make_layer(block, int(128 * s), layers[0], kernel_size=(img_size // 2))
        self.layer2 = self._make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=(img_size // 2),
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, int(512 * s), layers[2], stride=2, kernel_size=(img_size // 4),
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, int(1024 * s), layers[3], stride=2, kernel_size=(img_size // 8),
                                       dilate=replace_stride_with_dilation[2])

        # 解码器
        self.decoder1 = DeformConv2d(int(1024 * 2 * s), int(1024 * 2 * s), kernel_size=3, stride=1, padding=1)
        self.decoder2 = DeformConv2d(int(1024 * 2 * s), int(1024 * s), kernel_size=3, stride=1, padding=1)
        self.decoder3 = DeformConv2d(int(1024 * s), int(512 * s), kernel_size=3, stride=1, padding=1)
        self.decoder4 = DeformConv2d(int(512 * s), int(256 * s), kernel_size=3, stride=1, padding=1)
        self.decoder5 = DeformConv2d(int(256 * s), int(128 * s), kernel_size=3, stride=1, padding=1)

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
            # 保持1x1卷积不变
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
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

    def forward(self, x):
        # 编码器
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)  # [N, C1, H/2, W/2]
        x2 = self.layer2(x1)  # [N, C2, H/4, W/4]
        x3 = self.layer3(x2)  # [N, C3, H/8, W/8]
        x4 = self.layer4(x3)  # [N, C4, H/16, W/16]0

        # 解码器
        x = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.decoder1(x)
        x = torch.add(x, x4)
        x = self.relu(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.decoder2(x)
        x = torch.add(x, x3)
        x = self.relu(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.decoder3(x)
        x = torch.add(x, x2)
        x = self.relu(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.decoder4(x)
        x = torch.add(x, x1)
        x = self.relu(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.decoder5(x)
        x = self.relu(x)

        x = self.adjust(x)
        x = self.soft(x)
        return x

# 模型构造函数
def resxialunet128s(pretrained=False, **kwargs):
    model = ResAxialAttentionUNet(AxialBlock, [1, 2, 4, 1], s=0.125, img_size=128, imgchan=3, **kwargs)
    return model

# EOF
