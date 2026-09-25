"""Discrete Wavelet Transform and Frequency Attention modules for MSDANet."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import pywt


class LH_DWT_Function_attation(Function):
    @staticmethod
    def forward(ctx, x, w_lh):
        x = x.contiguous()
        ctx.save_for_backward(w_lh)
        ctx.shape = x.shape
        dim = x.shape[1]
        x = F.pad(x, (1, 0, 1, 0))
        x_lh = F.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=1, padding=0, groups=dim)
        return x_lh

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_lh = ctx.saved_tensors[0]
            B, C, H, W = ctx.shape
            dx = dx.view(B, 1, -1, H, W)
            dx = dx.transpose(1, 2).reshape(B, -1, H, W)
            filters = w_lh.repeat(C, 1, 1, 1).to(dtype=dx.dtype)
            dx = F.conv_transpose2d(dx, filters, stride=1, groups=C)
            dx = torch.cat((dx[:, :, :0], dx[:, :, 1:]), dim=2)
            dx = torch.cat((dx[:, :, :, :0], dx[:, :, :, 1:]), dim=3)
            return dx, None
        return None, None


class LH_DWT_2D_attation(nn.Module):
    def __init__(self, wave='haar'):
        super().__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0).float())

    def forward(self, x):
        return LH_DWT_Function_attation.apply(x, self.w_lh)


class HL_DWT_Function_attation(Function):
    @staticmethod
    def forward(ctx, x, w_hl):
        x = x.contiguous()
        ctx.save_for_backward(w_hl)
        ctx.shape = x.shape
        dim = x.shape[1]
        x = F.pad(x, (1, 0, 1, 0))
        x_hl = F.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=1, padding=0, groups=dim)
        return x_hl

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_hl = ctx.saved_tensors[0]
            B, C, H, W = ctx.shape
            dx = dx.view(B, 1, -1, H, W)
            dx = dx.transpose(1, 2).reshape(B, -1, H, W)
            filters = w_hl.repeat(C, 1, 1, 1).to(dtype=dx.dtype)
            dx = F.conv_transpose2d(dx, filters, stride=1, groups=C)
            dx = torch.cat((dx[:, :, :0], dx[:, :, 1:]), dim=2)
            dx = torch.cat((dx[:, :, :, :0], dx[:, :, :, 1:]), dim=3)
            return dx, None
        return None, None


class HL_DWT_2D_attation(nn.Module):
    def __init__(self, wave='haar'):
        super().__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0).float())

    def forward(self, x):
        return HL_DWT_Function_attation.apply(x, self.w_hl)


class HH_DWT_Function_attation(Function):
    @staticmethod
    def forward(ctx, x, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_hh)
        ctx.shape = x.shape
        dim = x.shape[1]
        x = F.pad(x, (1, 0, 1, 0))
        x_hh = F.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=1, padding=0, groups=dim)
        return x_hh

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_hh = ctx.saved_tensors[0]
            B, C, H, W = ctx.shape
            dx = dx.view(B, 1, -1, H, W)
            dx = dx.transpose(1, 2).reshape(B, -1, H, W)
            filters = w_hh.repeat(C, 1, 1, 1).to(dtype=dx.dtype)
            dx = F.conv_transpose2d(dx, filters, stride=1, groups=C)
            dx = torch.cat((dx[:, :, :0], dx[:, :, 1:]), dim=2)
            dx = torch.cat((dx[:, :, :, :0], dx[:, :, :, 1:]), dim=3)
            return dx, None
        return None, None


class HH_DWT_2D_attation(nn.Module):
    def __init__(self, wave='haar'):
        super().__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0).float())

    def forward(self, x):
        return HH_DWT_Function_attation.apply(x, self.w_hh)


class LL_DWT_Function_attation(Function):
    @staticmethod
    def forward(ctx, x, w_ll):
        x = x.contiguous()
        ctx.save_for_backward(w_ll)
        ctx.shape = x.shape
        dim = x.shape[1]
        x = F.pad(x, (1, 0, 1, 0))
        x_ll = F.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=1, padding=0, groups=dim)
        return x_ll

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll = ctx.saved_tensors[0]
            B, C, H, W = ctx.shape
            dx = dx.view(B, 1, -1, H, W)
            dx = dx.transpose(1, 2).reshape(B, -1, H, W)
            filters = w_ll.repeat(C, 1, 1, 1).to(dtype=dx.dtype)
            dx = F.conv_transpose2d(dx, filters, stride=1, groups=C)
            dx = torch.cat((dx[:, :, :0], dx[:, :, 1:]), dim=2)
            dx = torch.cat((dx[:, :, :, :0], dx[:, :, :, 1:]), dim=3)
            return dx, None
        return None, None


class LL_DWT_2D_attation(nn.Module):
    def __init__(self, wave='haar'):
        super().__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])
        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0).float())

    def forward(self, x):
        return LL_DWT_Function_attation.apply(x, self.w_ll)


class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape
        dim = x.shape[1]

        x_lh = F.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, padding=0, groups=dim)
        x_hl = F.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, padding=0, groups=dim)
        x_hh = F.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, padding=0, groups=dim)
        return torch.cat([x_lh, x_hl, x_hh], dim=1)

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 3, -1, H // 2, W // 2)
            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            filters = torch.cat([w_lh, w_hl, w_hh], dim=0)
            filters = filters.repeat(C, 1, 1, 1).to(dtype=dx.dtype)
            dx = F.conv_transpose2d(dx, filters, stride=2, groups=C)
            return dx, None, None, None, None
        return None, None, None, None, None


class DWT_2D(nn.Module):
    def __init__(self, wave='haar'):
        super().__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0).float())
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0).float())
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0).float())
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0).float())

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)


class Hfrequencyfeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.DWT_2D = DWT_2D('haar')

    def forward(self, x):
        return self.DWT_2D(x)


class Hfrequency(nn.Module):
    def __init__(self):
        super().__init__()
        self.Hfrequencyfeature = Hfrequencyfeature()

    def forward(self, x, out_2):
        out_1 = self.Hfrequencyfeature(x)
        return torch.cat([out_1, out_2], dim=1)
