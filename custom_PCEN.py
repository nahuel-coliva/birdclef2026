"""
The following code is (almost) a copy-paste from https://github.com/daemon/pytorch-pcen as of 28/03/2026
It was the easiest way to change the 16kHz sampling rate to a custom one (32kHz in our application)
I do not own any rights on it, contact me if anything is off to peacefully solve the question
"""

"""
BSD 2-Clause License

Copyright (c) 2017 Facebook Inc. (Soumith Chintala), 
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class F2M(nn.Module):
    """This turns a normal STFT into a MEL Frequency STFT, using a conversion
       matrix.  This uses triangular filter banks.
    Args:
        n_mels (int): number of MEL bins
        sr (int): sample rate of audio signal
        f_max (float, optional): maximum frequency. default: sr // 2
        f_min (float): minimum frequency. default: 0
    """
    def __init__(self, n_mels=40, sr=32000, f_max=None, f_min=0., n_fft=40, onesided=True):
        super().__init__()
        self.n_mels = n_mels
        self.sr = sr
        self.f_max = f_max if f_max is not None else sr // 2
        self.f_min = f_min
        self.n_fft = n_fft
        if onesided:
            self.n_fft = self.n_fft // 2 + 1
        self._init_buffers()

    def _init_buffers(self):
        m_min = 0. if self.f_min == 0 else 2595 * np.log10(1. + (self.f_min / 700))
        m_max = 2595 * np.log10(1. + (self.f_max / 700))

        m_pts = torch.linspace(m_min, m_max, self.n_mels + 2)
        f_pts = (700 * (10**(m_pts / 2595) - 1))

        bins = torch.floor(((self.n_fft - 1) * 2) * f_pts / self.sr).long()

        fb = torch.zeros(self.n_fft, self.n_mels)
        for m in range(1, self.n_mels + 1):
            f_m_minus = bins[m - 1].item()
            f_m = bins[m].item()
            f_m_plus = bins[m + 1].item()

            if f_m_minus != f_m:
                fb[f_m_minus:f_m, m - 1] = (torch.arange(f_m_minus, f_m) - f_m_minus) / (f_m - f_m_minus)
            if f_m != f_m_plus:
                fb[f_m:f_m_plus, m - 1] = (f_m_plus - torch.arange(f_m, f_m_plus)) / (f_m_plus - f_m)
        self.register_buffer("fb", fb)

    def forward(self, spec_f):
        # OLD
        #spec_m = torch.matmul(spec_f, self.fb)  # (c, l, n_fft) dot (n_fft, n_mels) -> (c, l, n_mels)

        # NEW: updated to use a layer
        # (c, l, n_fft) dot (n_fft, n_mels) == F.linear((c, l, n_fft), (n_mels, n_fft))
        spec_m = F.linear(spec_f, self.fb.T)
        return spec_m

@torch.compile
def compiled_M(x, s, T, device):
    s_x = x.mul(s)
    s_x[:,0,:] = x[:,0,:]

    powers = (1 - s) ** -torch.arange(T, device=device)  # [T]
    
    # reshape per broadcasting: [1, T, 1]
    powers = powers.view(1, T, 1)
    
    M = s_x * powers
    M = torch.cumsum(M, dim=1) / powers
    return M

def pcen(x, eps=1E-6, s=0.025, alpha=0.98, delta=2, r=0.5, training=False, last_state=None, empty=True):
    """
    # OLD
    frames = x.split(1, -2)
    m_frames = []
    if empty:
        last_state = None
    for frame in frames:
        if last_state is None:
            last_state = frame
            m_frames.append(frame)
            continue
        if training:
            m_frame = ((1 - s) * last_state).add_(s * frame)
        else:
            m_frame = (1 - s) * last_state + s * frame
        last_state = m_frame
        m_frames.append(m_frame)
    M = torch.cat(m_frames, 1)
    """
    # DEBUG: proviamo a ricostruire M in maniera più rapida, tanto di last_state non ci importa
    #print(x.shape) = print(M.shape) = [64, 501, 200] = [B, T, mel]
    # NEW
    """
    device = x.device
    s_x = x.mul(s)
    s_x[:,0,:] = x[:,0,:]

    _, T, _ = s_x.shape
    
    powers = (1 - s) ** -torch.arange(T, device=device)  # [T]
    
    # reshape per broadcasting: [1, T, 1]
    powers = powers.view(1, T, 1)
    
    M = s_x * powers
    M = torch.cumsum(M, dim=1) / powers
    """
    device = x.device
    _, T, _ = x.shape
    M = compiled_M(x, s, T, device)
    # FINE DEBUG

    if training:
        pcen_ = (x / (M + eps).pow(alpha) + delta).pow(r) - delta ** r
    else:
        pcen_ = x.div_(M.add_(eps).pow_(alpha)).add_(delta).pow_(r).sub_(delta ** r)
    return pcen_, last_state


class StreamingPCENTransform(nn.Module):

    def __init__(self, eps=1E-6, s=0.05, alpha=0.9, delta=2, r=0.5, trainable=False, 
            use_cuda_kernel=False, **stft_kwargs):
        super().__init__()
        self.use_cuda_kernel = use_cuda_kernel
        if trainable:
            self.s = nn.Parameter(torch.Tensor([s]))
            self.alpha = nn.Parameter(torch.Tensor([alpha]))
            self.delta = nn.Parameter(torch.Tensor([delta]))
            self.r = nn.Parameter(torch.Tensor([r]))
        else:
            self.s = s
            self.alpha = alpha
            self.delta = delta
            self.r = r
        self.eps = eps
        self.trainable = trainable
        self.stft_kwargs = stft_kwargs
        self.register_buffer("last_state", torch.zeros(stft_kwargs["n_mels"]))
        mel_keys = {"n_mels", "sr", "f_max", "f_min", "n_fft"}
        mel_keys = set(stft_kwargs.keys()).intersection(mel_keys)
        mel_kwargs = {k: stft_kwargs[k] for k in mel_keys}
        stft_keys = set(stft_kwargs.keys()) - mel_keys
        self.n_fft = stft_kwargs["n_fft"]
        self.stft_kwargs = {k: stft_kwargs[k] for k in stft_keys}
        # ADDED: window length for explicit window, see below
        win_length = self.stft_kwargs.get("win_length", self.n_fft)
        self.register_buffer("window", torch.hann_window(win_length))
        self.f2m = F2M(**mel_kwargs)
        self.reset()

    def reset(self):
        self.empty = True

    def forward(self, x):
        # OLD
        #x = torch.stft(x, self.n_fft, **self.stft_kwargs).norm(dim=-1, p=2)
        # NEW:
        # 1) stft without "return_complex=True" is deprecated, so we add it and change the norm operator accordingly
        # 2) stft without a specified window uses the rectangle window: best suited for periodic signals, it introduces heavy spectral leakage otherwise
        #    therefore, we change to Hanning window
        window = self.window.to(device=x.device, dtype=x.dtype)
        x = torch.stft(x, self.n_fft, **self.stft_kwargs, window=window, return_complex=True).abs()
        x = self.f2m(x.permute(0, 2, 1))
        if self.use_cuda_kernel:
            x, ls = pcen(x, self.eps, self.s, self.alpha, self.delta, self.r, self.trainable, self.last_state, self.empty) #pcen_cuda_kernel
        else:
            x, ls = pcen(x, self.eps, self.s, self.alpha, self.delta, self.r, self.training and self.trainable, self.last_state, self.empty)
        self.last_state = ls.detach()
        self.empty = False
        return x

    