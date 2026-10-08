"""Ablations of synfuse. The full variant is that network.

Set algorithm: synfuse_ablate and synfuse_ablate.variant to one of:

  full          joint token, 8x8 RWKV, row Mamba, column Mamba, both ways, five stages
  one_stage     the same stage, used once
  no_window     drop the 8x8 RWKV
  no_rowcol     drop the row and column Mamba
  forward_only  drop the reversed scan
  color_write   PAN still enters the token; PAN features are not updated
  no_joint      color and PAN are scanned as two sequences
  pan_head      a second 3x3 adds the PAN features onto the image

Width stays 32. Official RwkvBlock and official Mamba, ordinary initialization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from research2v1.models.synfuse import (
    AxisFuseMamba,
    FuseStage,
    ResStem,
    WindowFuseRwkv,
    _mamba,
    _rwkv_block,
)

VARIANTS = {
    "full": dict(stages=5, window=True, rowcol=True, bidirectional=True, joint=True, write_pan=True, pan_head=False),
    "one_stage": dict(stages=1, window=True, rowcol=True, bidirectional=True, joint=True, write_pan=True, pan_head=False),
    "no_window": dict(stages=5, window=False, rowcol=True, bidirectional=True, joint=True, write_pan=True, pan_head=False),
    "no_rowcol": dict(stages=5, window=True, rowcol=False, bidirectional=True, joint=True, write_pan=True, pan_head=False),
    "forward_only": dict(stages=5, window=True, rowcol=True, bidirectional=False, joint=True, write_pan=True, pan_head=False),
    "color_write": dict(stages=5, window=True, rowcol=True, bidirectional=True, joint=True, write_pan=False, pan_head=False),
    "no_joint": dict(stages=5, window=True, rowcol=True, bidirectional=True, joint=False, write_pan=True, pan_head=False),
    "pan_head": dict(stages=5, window=True, rowcol=True, bidirectional=True, joint=True, write_pan=True, pan_head=True),
}


def _spec(args):
    cfg = {}
    if isinstance(args, dict):
        raw = args.get("synfuse_ablate", {})
        if isinstance(raw, dict):
            cfg = raw
    name = str(cfg.get("variant", "full"))
    if name not in VARIANTS:
        known = ", ".join(sorted(VARIANTS))
        raise ValueError("unknown synfuse ablation %r (choose from %s)" % (name, known))
    spec = dict(VARIANTS[name])
    spec["name"] = name
    return spec


def _uses_original_blocks(spec):
    return spec["bidirectional"] and spec["joint"] and spec["write_pan"]


class _WindowMix(nn.Module):
    def __init__(self, dim, bidirectional, joint, write_pan, win=8):
        super(_WindowMix, self).__init__()
        self.win = win
        self.bidirectional = bidirectional
        self.joint = joint
        self.write_pan = write_pan
        in_dim = dim * 2 if joint else dim
        self.token = nn.Linear(in_dim, dim)
        self.forward_block = _rwkv_block(dim, layer_id=0, n_layers=2)
        self.backward_block = _rwkv_block(dim, layer_id=0, n_layers=2) if bidirectional else None
        self.to_color = nn.Linear(dim, dim)
        self.to_pan = nn.Linear(dim, dim) if write_pan else None
        self.local_color = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.local_pan = nn.Conv2d(dim, dim, kernel_size=3, padding=1) if write_pan else None

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

    def _mix(self, seq):
        forward_out = self.forward_block(seq)[0]
        if not self.bidirectional:
            return forward_out
        backward_out = self.backward_block(seq.flip(1).contiguous())[0].flip(1)
        return forward_out + backward_out - seq

    def forward(self, color, pan):
        batch, channels, height, width = color.shape
        color_w, n_h, n_w = self._windows(color)
        pan_w, _, _ = self._windows(pan)
        if self.joint:
            seq = self.token(torch.cat([color_w, pan_w], dim=-1)).contiguous()
            mixed = self._mix(seq)
            color_o = self._unwindows(self.to_color(mixed), batch, channels, height, width, n_h, n_w)
            color = color + color_o + self.local_color(color)
            if self.write_pan:
                pan_o = self._unwindows(self.to_pan(mixed), batch, channels, height, width, n_h, n_w)
                pan = pan + pan_o + self.local_pan(pan)
            return color, pan
        mixed_c = self._mix(self.token(color_w).contiguous())
        mixed_p = self._mix(self.token(pan_w).contiguous())
        color_o = self._unwindows(self.to_color(mixed_c), batch, channels, height, width, n_h, n_w)
        pan_o = self._unwindows(self.to_pan(mixed_p), batch, channels, height, width, n_h, n_w)
        return color + color_o + self.local_color(color), pan + pan_o + self.local_pan(pan)


class _AxisMix(nn.Module):
    def __init__(self, dim, axis, bidirectional, joint, write_pan):
        super(_AxisMix, self).__init__()
        self.axis = axis
        self.bidirectional = bidirectional
        self.joint = joint
        self.write_pan = write_pan
        in_dim = dim * 2 if joint else dim
        self.token = nn.Linear(in_dim, dim)
        self.norm_f = nn.LayerNorm(dim)
        self.scan_f = _mamba(dim)
        self.norm_b = nn.LayerNorm(dim) if bidirectional else None
        self.scan_b = _mamba(dim) if bidirectional else None
        self.to_color = nn.Linear(dim, dim)
        self.to_pan = nn.Linear(dim, dim) if write_pan else None

    def _to_seq(self, x):
        batch, channels, height, width = x.shape
        if self.axis == "w":
            return x.permute(0, 2, 3, 1).reshape(batch * height, width, channels)
        return x.permute(0, 3, 2, 1).reshape(batch * width, height, channels)

    def _from_seq(self, seq, batch, channels, height, width):
        if self.axis == "w":
            return seq.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        return seq.reshape(batch, width, height, channels).permute(0, 3, 2, 1)

    def _mix(self, seq):
        forward_out = seq + self.scan_f(self.norm_f(seq))
        if not self.bidirectional:
            return forward_out
        backward = seq.flip(1).contiguous()
        backward_out = (backward + self.scan_b(self.norm_b(backward))).flip(1)
        return forward_out + backward_out - seq

    def forward(self, color, pan):
        batch, channels, height, width = color.shape
        color_seq = self._to_seq(color).contiguous()
        pan_seq = self._to_seq(pan).contiguous()
        if self.joint:
            mixed = self._mix(self.token(torch.cat([color_seq, pan_seq], dim=-1)).contiguous())
            color = color + self._from_seq(self.to_color(mixed), batch, channels, height, width)
            if self.write_pan:
                pan = pan + self._from_seq(self.to_pan(mixed), batch, channels, height, width)
            return color, pan
        mixed_c = self._mix(self.token(color_seq).contiguous())
        mixed_p = self._mix(self.token(pan_seq).contiguous())
        color = color + self._from_seq(self.to_color(mixed_c), batch, channels, height, width)
        pan = pan + self._from_seq(self.to_pan(mixed_p), batch, channels, height, width)
        return color, pan


class _AblateStage(nn.Module):
    def __init__(self, dim, spec):
        super(_AblateStage, self).__init__()
        self.local = None
        self.long_w = None
        self.long_h = None
        if spec["window"]:
            self.local = _WindowMix(dim, spec["bidirectional"], spec["joint"], spec["write_pan"])
        if spec["rowcol"]:
            self.long_w = _AxisMix(dim, "w", spec["bidirectional"], spec["joint"], spec["write_pan"])
            self.long_h = _AxisMix(dim, "h", spec["bidirectional"], spec["joint"], spec["write_pan"])

    def forward(self, color, pan):
        if self.local is not None:
            color, pan = self.local(color, pan)
        if self.long_w is not None:
            color, pan = self.long_w(color, pan)
            color, pan = self.long_h(color, pan)
        return color, pan


class _OriginalStage(nn.Module):
    """synfuse blocks with the window, or the row and column scans, left out."""

    def __init__(self, dim, window, rowcol):
        super(_OriginalStage, self).__init__()
        self.local = WindowFuseRwkv(dim, layer_id=0, n_layers=2, win=8) if window else None
        self.long_w = AxisFuseMamba(dim, axis="w") if rowcol else None
        self.long_h = AxisFuseMamba(dim, axis="h") if rowcol else None

    def forward(self, color, pan):
        if self.local is not None:
            color, pan = self.local(color, pan)
        if self.long_w is not None:
            color, pan = self.long_w(color, pan)
            color, pan = self.long_h(color, pan)
        return color, pan


def _make_stage(dim, spec):
    if _uses_original_blocks(spec):
        if spec["window"] and spec["rowcol"]:
            return FuseStage(dim, n_layers=2)
        return _OriginalStage(dim, spec["window"], spec["rowcol"])
    return _AblateStage(dim, spec)


class Net(nn.Module):
    def __init__(self, num_channels, base_filter, args):
        super(Net, self).__init__()
        spec = _spec(args)
        dim = 32
        self.variant = spec["name"]
        self.color_stem = ResStem(num_channels, dim)
        self.pan_stem = ResStem(1, dim)
        self.stages = nn.ModuleList([_make_stage(dim, spec) for _ in range(spec["stages"])])
        self.head = nn.Conv2d(dim, num_channels, kernel_size=3, padding=1)
        self.pan_head = nn.Conv2d(dim, num_channels, kernel_size=3, padding=1) if spec["pan_head"] else None

    def forward(self, lms, bms, pan):
        image = F.interpolate(lms, scale_factor=4)
        color = self.color_stem(image)
        pan_f = self.pan_stem(pan)
        for stage in self.stages:
            color, pan_f = stage(color, pan_f)
        out = image + self.head(color)
        if self.pan_head is not None:
            out = out + self.pan_head(pan_f)
        return out
