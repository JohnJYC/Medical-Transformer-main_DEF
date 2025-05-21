import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_fwd, custom_bwd
from .table import TABLE, BWDTABLE
from DCNv4 import ext  # CUDA extension

# Tuning helpers
def factors(N): return [i for i in range(1, N+1) if N % i == 0]

def findspec(B, H, W, G, C):
    key = f"{B}x{H}x{W}x{G}x{C}"
    if key in TABLE: return TABLE[key]
    d_stride = 8
    ms = factors(B * H * W)
    m = max([m for m in ms if m <= 64 and (m * G * C // d_stride) <= 512] or [1])
    n_thread = m * G * C // d_stride
    TABLE[key] = (d_stride, n_thread)
    return d_stride, n_thread

def find_spec_bwd(B, H, W, G, C):
    key = f"{B}x{H}x{W}x{G}x{C}"
    if key in BWDTABLE: return BWDTABLE[key]
    d_stride = 2 if C >= 64 else 1
    ms = factors(B * H * W)
    m = max([m for m in ms if m <= 64 and (m * G * C // d_stride) <= 256] or [1])
    n_thread = m * G * C // d_stride
    BWDTABLE[key] = (d_stride, n_thread)
    return d_stride, n_thread

class DCNv4Function(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, offset_mask,
                kernel_h, kernel_w,
                stride_h, stride_w,
                pad_h, pad_w,
                dilation_h, dilation_w,
                group, group_channels,
                offset_scale, im2col_step,
                remove_center):
        # shapes
        N, C, H, W = input.shape
        _, mask_c, H_out, W_out = offset_mask.shape
        # flatten
        input_seq = input.permute(0,2,3,1).contiguous().view(N, H*W, C)
        offset_seq = offset_mask.permute(0,2,3,1).contiguous().view(N, H_out*W_out, mask_c)
        # tuning
        fs, ft = findspec(N, H, W, group, C)
        bs, bt = find_spec_bwd(N, H, W, group, C)
        # save
        ctx.save_for_backward(input_seq, offset_seq)
        # save channels for backward reshape
        ctx.C = C
        ctx.H, ctx.W, ctx.H_out, ctx.W_out = H, W, H_out, W_out
        ctx.params = (
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w,
            dilation_h, dilation_w,
            group, group_channels,
            offset_scale, im2col_step,
            fs, ft, bs, bt,
            remove_center
        )
                                # extension call to CUDA binding (18 args expected)
        out_seq = ext.dcnv4_forward(
            input_seq, offset_seq,
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w,
            dilation_h, dilation_w,
            group, group_channels,
            offset_scale, im2col_step,
            fs, ft, bs,
            remove_center
        )
        out_seq = ext.dcnv4_forward(
            input_seq, offset_seq,
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w,
            dilation_h, dilation_w,
            group, group_channels,
            offset_scale, im2col_step,
            fs, ft, bs,
            remove_center
        )
        # reshape back
        out = out_seq.view(N, H_out, W_out, -1).permute(0,3,1,2).contiguous()
        return out

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        input_seq, offset_seq = ctx.saved_tensors
        (kernel_h, kernel_w,
         stride_h, stride_w,
         pad_h, pad_w,
         dilation_h, dilation_w,
         group, group_channels,
         offset_scale, im2col_step,
         fs, ft, bs, bt,
         remove_center) = ctx.params
        H, W, H_out, W_out = ctx.H, ctx.W, ctx.H_out, ctx.W_out
        # retrieve channel count for reshape
        C = ctx.C
        # grad flatten
        grad_seq = grad_output.permute(0,2,3,1).contiguous().view(grad_output.shape[0], H_out*W_out, grad_output.shape[1])
                        # backward call to CUDA binding (19 args expected)
        grad_input_seq, grad_offset_seq = ext.dcnv4_backward(
            input_seq, offset_seq,
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w,
            dilation_h, dilation_w,
            group, group_channels,
            offset_scale, im2col_step,
            grad_seq,
            fs, ft, bs,
            remove_center
        )
        # unflatten
        grad_input = grad_input_seq.view(grad_input_seq.shape[0], C, H, W)
        grad_offset = grad_offset_seq.view(grad_offset_seq.shape[0], -1, H_out, W_out)
        return grad_input, grad_offset, *(None,)*15