import os
import torch
import librosa
import pandas as pd
from torchvision.models import mobilenet_v2

"""
The following code is (almost) copy-paste from https://github.com/daemon/pytorch-pcen as of 28/03/2026
It was the easiest way to change the 16kHz sampling rate to a custom one (32kHz in our application)
I do not own any rights on it, contact me if anything is off to peacefully solve the question :)
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

def pcen(x, eps=1E-6, s=0.025, alpha=0.98, delta=2, r=0.5, training=False, last_state=None, empty=True):
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

"""
# END OF (ALMOST) COPY-PASTE CODE
"""

# === MODEL ===
class PCENFrontend(nn.Module):
    def __init__(
        self,
        sample_rate=32000,
        n_fft=1024,
        hop_length=320,
        n_mels=128,
        trainable_pcen=True
    ):
        super().__init__()

        # PCEN preso dal tuo codice copiato
        self.pcen = StreamingPCENTransform(
            trainable=trainable_pcen,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            sr=sample_rate
        ).to("cpu")
    
    def forward(self, x):
        """
        x: [B, T]
        """

        # [B, T]

        # PCEN
        pcen = self.pcen(x)

        self.pcen.reset()

        # normalizzazione
        #pcen = (pcen - pcen.mean(dim=-1, keepdim=True)) / (pcen.std(dim=-1, keepdim=True) + 1e-6)
        #ora trasormato in torch.nn.functional.layer_norm(pcen, (pcen.shape[-1],), eps=1e-6) per fonderlo nel CUDA graph

        # Lascio come appunto se servisse ma alla fine abbiamo modificato il primo layer di MobileNet
        #pcen = pcen.repeat(1, 3, 1, 1)  # 1 → 3 canali per input a MobileNetV2

        return torch.nn.functional.layer_norm(pcen, (pcen.shape[-1],), eps=1e-6)
    



class BirdModel(nn.Module):
    def __init__(self,
                num_classes,
                sample_rate=32000,
                n_fft=1024,
                hop_length=320,
                n_mels=128,
                trainable_pcen=True,
    ):
        super().__init__()
        self.num_classes=num_classes
        self.sample_rate=sample_rate
        self.n_fft=n_fft
        self.hop_length=hop_length
        self.n_mels=n_mels
        self.trainable_pcen=trainable_pcen

        # frontend PCEN
        self.frontend = PCENFrontend(sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            trainable_pcen=trainable_pcen
        )

        # backbone pre-trained
        #self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        # Pesi impostati None per submission
        self.backbone = mobilenet_v2(weights=None)

        # parameters freezing
        for p in self.backbone.features.parameters():
            p.requires_grad = False

        # modifica primo layer per 1 canale (invece di 3)
        first_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False
        )

        # testa di classificazione
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        """
        x: [B, T]
        """

        x = self.frontend(x)   # [B, mel, time]
        
        # aggiungi channel dim
        #x = x.unsqueeze(1)     # [B, 1, mel, time]
        x = self.backbone(x.unsqueeze(1))   # [B, num_classes]

        return x # previously torch.sigmoid(x), wrong with BCEWithLogitsLoss

# === CONFIG ===
MODEL_PATH = "./results/session_bigger_dataset_1/320_200/model_checkpoint.pth"
TEST_DIR = "./data/validation_soundscapes"
TAXONOMY_PATH = "./data/taxonomy.csv"
SR = 32000
WINDOW = 5  # secondi
DEVICE = "cpu"
HOP_LENGTH = 320
N_FFT = HOP_LENGTH*4
N_MELS = 200

# === load class names ===
taxonomy = pd.read_csv(TAXONOMY_PATH)
class_names = taxonomy["primary_label"].tolist()
num_classes = len(class_names)

# === load model ===
model = BirdModel(num_classes=num_classes,
                    sample_rate=SR,
                    n_fft=N_FFT,
                    hop_length=HOP_LENGTH,
                    n_mels=N_MELS,
                    trainable_pcen=True
                ).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
state_dict = checkpoint["model_state_dict"]
# Set BirdModel.last_state to by-pass loading issues
checkpoint["model_state_dict"]["frontend.pcen.last_state"] = torch.zeros(N_MELS)

model.load_state_dict(state_dict)
starting_epoch = checkpoint["epoch"]
model.eval()
print("Successfully loaded model "+str(MODEL_PATH))

# === inference ===
rows = []

for fname in os.listdir(TEST_DIR):
    if not fname.endswith(".ogg"):
        continue

    fpath = os.path.join(TEST_DIR, fname)

    audio, _ = librosa.load(fpath, sr=SR)

    total_duration = len(audio) / SR
    n_windows = int(total_duration // WINDOW)

    base_name = fname.replace(".ogg", "")

    for i in range(n_windows):
        start = int(i * WINDOW * SR)
        end = int((i + 1) * WINDOW * SR)

        chunk = torch.tensor(audio[start:end]).float().to(DEVICE)
        chunk = chunk.unsqueeze(0)

        with torch.no_grad():
            logits = model(chunk)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        row_id = f"{base_name}_{(i+1)*WINDOW}"

        row = {"row_id": row_id}
        row.update({cls: prob for cls, prob in zip(class_names, probs)})

        rows.append(row)

# === salva submission ===
df = pd.DataFrame(rows)
df.to_csv("./results/submission.csv", index=False)