"""Color and PAN meet inside the scan.

Each pixel stays one token. A linear layer builds that token from the color
and PAN channels, then RWKV scans the ordinary 8x8 window order and Mamba
scans ordinary full-resolution rows and columns. Official RwkvBlock and
official mamba_ssm.Mamba are used. No branch starts at zero.
"""
import os
import shutil
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from transformers import RwkvConfig
from transformers.models.rwkv import modeling_rwkv as rwkv_mod
from transformers.models.rwkv.modeling_rwkv import RwkvBlock, RwkvPreTrainedModel


def _rwkv_backward(ctx, g_output, g_state=None):
    """Official WKV backward with one gradient slot per sequence.

    The transformers 4.40 kernel writes ``gw[batch, channel]``. Its wrapper
    allocates only ``[channel]``. Summing over the batch is the gradient of
    the shared decay.
    """
    input_dtype = ctx.input_dtype
    time_decay, time_first, key, value, output = ctx.saved_tensors
    batch, _, channels = key.shape
    grad_decay = torch.empty(batch, channels, device=key.device, dtype=torch.float32)
    grad_first = torch.empty(batch, channels, device=key.device, dtype=torch.float32)
    grad_key = torch.empty_like(key, memory_format=torch.contiguous_format)
    grad_value = torch.empty_like(value, memory_format=torch.contiguous_format)
    if input_dtype == torch.float16:
        g_output = g_output.float()
    kernel = rwkv_mod.rwkv_cuda_kernel
    backward_func = kernel.backward_bf16 if input_dtype == torch.bfloat16 else kernel.backward
    backward_func(
        time_decay, time_first, key, value, output, g_output.contiguous(),
        grad_decay, grad_first, grad_key, grad_value,
    )
    return (
        grad_decay.sum(dim=0).to(input_dtype),
        grad_first.sum(dim=0).to(input_dtype),
        grad_key.to(input_dtype),
        grad_value.to(input_dtype),
        None,
        None,
    )


rwkv_mod.RwkvLinearAttention.backward = _rwkv_backward

if shutil.which("ninja") is None:
    _bin = os.path.dirname(sys.executable)
    if os.path.isfile(os.path.join(_bin, "ninja")) or os.path.isfile(os.path.join(_bin, "ninja.exe")):
        os.environ["PATH"] = _bin + os.pathsep + os.environ.get("PATH", "")


def _rwkv_block(dim, layer_id, n_layers):
    config = RwkvConfig(
        vocab_size=1,
        context_length=128,
        hidden_size=dim,
        num_hidden_layers=n_layers,
        intermediate_size=dim * 4,
        attention_hidden_size=dim,
        bos_token_id=0,
        eos_token_id=0,
        rescale_every=0,
    )
    block = RwkvBlock(config, layer_id=layer_id)
    block.apply(lambda module: RwkvPreTrainedModel._init_weights(block, module))
    return block


def _mamba(dim):
    return Mamba(d_model=dim, d_state=16, d_conv=4, expand=2, use_fast_path=False)


class ResStem(nn.Module):
    def __init__(self, in_ch, dim):
        super(ResStem, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.conv(x)


class WindowFuseRwkv(nn.Module):
    """Official RWKV on one joint token per pixel, in ordinary window order."""

    def __init__(self, dim, layer_id, n_layers, win=8):
        super(WindowFuseRwkv, self).__init__()
        self.win = win
        self.token = nn.Linear(dim * 2, dim)
        self.forward_block = _rwkv_block(dim, layer_id, n_layers)
        self.backward_block = _rwkv_block(dim, layer_id, n_layers)
        self.to_color = nn.Linear(dim, dim)
        self.to_pan = nn.Linear(dim, dim)
        self.local_color = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.local_pan = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def _windows(self, x):
        batch, channels, height, width = x.shape
        win = self.win
        n_h, n_w = height // win, width // win
        seq = x.view(batch, channels, n_h, win, n_w, win)
        seq = seq.permute(0, 2, 4, 3, 5, 1).reshape(batch * n_h * n_w, win * win, channels)
        return seq, n_h, n_w

    def _unwindows(self, seq, batch, channels, height, width, n_h, n_w):
        win = self.win
        mixed = seq.reshape(batch, n_h, n_w, win, win, channels)
        return mixed.permute(0, 5, 1, 3, 2, 4).reshape(batch, channels, height, width)

    def forward(self, color, pan):
        batch, channels, height, width = color.shape
        color_w, n_h, n_w = self._windows(color)
        pan_w, _, _ = self._windows(pan)
        seq = self.token(torch.cat([color_w, pan_w], dim=-1)).contiguous()
        forward_out = self.forward_block(seq)[0]
        backward_out = self.backward_block(seq.flip(1).contiguous())[0].flip(1)
        mixed = forward_out + backward_out - seq
        color_o = self._unwindows(self.to_color(mixed), batch, channels, height, width, n_h, n_w)
        pan_o = self._unwindows(self.to_pan(mixed), batch, channels, height, width, n_h, n_w)
        return color + color_o + self.local_color(color), pan + pan_o + self.local_pan(pan)


class AxisFuseMamba(nn.Module):
    """Official selective scan of a token built from both streams."""

    def __init__(self, dim, axis):
        super(AxisFuseMamba, self).__init__()
        self.axis = axis
        self.token = nn.Linear(dim * 2, dim)
        self.norm_f = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)
        self.scan_f = _mamba(dim)
        self.scan_b = _mamba(dim)
        self.to_color = nn.Linear(dim, dim)
        self.to_pan = nn.Linear(dim, dim)

    def _to_seq(self, x):
        batch, channels, height, width = x.shape
        if self.axis == "w":
            return x.permute(0, 2, 3, 1).reshape(batch * height, width, channels)
        return x.permute(0, 3, 2, 1).reshape(batch * width, height, channels)

    def _from_seq(self, seq, batch, channels, height, width):
        if self.axis == "w":
            return seq.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        return seq.reshape(batch, width, height, channels).permute(0, 3, 2, 1)

    def forward(self, color, pan):
        batch, channels, height, width = color.shape
        color_seq = self._to_seq(color).contiguous()
        pan_seq = self._to_seq(pan).contiguous()
        seq = self.token(torch.cat([color_seq, pan_seq], dim=-1)).contiguous()
        forward_out = seq + self.scan_f(self.norm_f(seq))
        backward = seq.flip(1).contiguous()
        backward_out = (backward + self.scan_b(self.norm_b(backward))).flip(1)
        mixed = forward_out + backward_out - seq
        color = color + self._from_seq(self.to_color(mixed), batch, channels, height, width)
        pan = pan + self._from_seq(self.to_pan(mixed), batch, channels, height, width)
        return color, pan


class FuseStage(nn.Module):
    """Window RWKV crosses the two streams, then full-resolution Mamba does."""

    def __init__(self, dim, n_layers):
        super(FuseStage, self).__init__()
        self.local = WindowFuseRwkv(dim, layer_id=0, n_layers=n_layers, win=8)
        self.long_w = AxisFuseMamba(dim, axis="w")
        self.long_h = AxisFuseMamba(dim, axis="h")

    def forward(self, color, pan):
        color, pan = self.local(color, pan)
        color, pan = self.long_w(color, pan)
        color, pan = self.long_h(color, pan)
        return color, pan


class Net(nn.Module):
    def __init__(self, num_channels, base_filter, args):
        super(Net, self).__init__()
        dim = 32
        self.color_stem = ResStem(num_channels, dim)
        self.pan_stem = ResStem(1, dim)
        self.stages = nn.ModuleList([FuseStage(dim, n_layers=2) for _ in range(5)])
        self.head = nn.Conv2d(dim, num_channels, kernel_size=3, padding=1)

    def forward(self, lms, bms, pan):
        image = F.interpolate(lms, scale_factor=4)
        color = self.color_stem(image)
        pan_f = self.pan_stem(pan)
        for stage in self.stages:
            color, pan_f = stage(color, pan_f)
        return image + self.head(color)
