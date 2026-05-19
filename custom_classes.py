import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import librosa
from collections import OrderedDict

import custom_PCEN
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import matplotlib.pyplot as plt

class SoundscapeDataset(Dataset):
    def __init__(
        self,
        root_dir,
        df,
        species_to_idx,
        sample_rate=32000,
        chunk_duration=5.0,
        overlap=0.0
    ):
        self.cache = OrderedDict()
        self.cache_size = 10  # max file in RAM
        self.root_dir = root_dir
        self.df = df
        self.species_to_idx = species_to_idx
        self.num_classes = len(species_to_idx)
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self.hop_size = int(self.chunk_size * (1 - overlap))
        # ogni riga = un sample
        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []

        for _, row in self.df.iterrows():
            if row["filename"].count("BC2026")>1:
                actual_root_dir = self.root_dir["synthetic"]
            else:
                actual_root_dir = self.root_dir["real"]
            filepath = os.path.join(actual_root_dir, row["filename"])
            start_time = row["start"]
            start_sample = int(start_time * self.sample_rate)
            if str(row["primary_label"])=="nan":
                labels = []
            else:
                labels = row["primary_label"].split(";")

            samples.append((filepath, start_sample, labels))

        return samples

    def __getitem__(self, idx):
        filepath, start, labels = self.samples[idx]

        # carica audio già resamplato a 32kHz
        # DEBUG: performance test SENZA cache
        """
        # OLD: with cache
        if filepath not in self.cache:
            if len(self.cache) >= self.cache_size:
                self.cache.popitem(last=False)

            self.cache[filepath], _ = librosa.load(filepath, sr=self.sample_rate)

        audio = self.cache[filepath]
        """
        # NEW: no cache
        audio, _ = librosa.load(filepath, sr=self.sample_rate)
        
        # FINE DEBUG

        chunk = audio[start:start + self.chunk_size]

        # padding se necessario : NON SEMBRA ESSERE MAI NECESSARIO, TOLGO
        """
        if len(chunk) < self.chunk_size:
            pad = self.chunk_size - len(chunk)

            #DEBUG
            print("Ho paddato "+int(str(pad/self.sample_rate*1000))+"ms")
            #FINE DEBUG

            chunk = torch.nn.functional.pad(
                torch.tensor(chunk), (0, pad)
            )
        else:
            chunk = torch.tensor(chunk)
        """
        chunk = torch.tensor(chunk)

        label = self.build_label(labels)

        return chunk.float(), label, filepath, start
        
    def build_label(self, labels):
        y = torch.zeros(self.num_classes, dtype=torch.float32)

        for label in labels:
            if label in self.species_to_idx:
                y[self.species_to_idx[label]] = 1.0

        return y
    
    def __len__(self):
        return len(self.samples)

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

        self.pcen = custom_PCEN.StreamingPCENTransform(
            trainable=trainable_pcen,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            sr=sample_rate,
            # DEBUG: dobbiamo passare device anche qua e sotto, intanto metto una pezza
            use_cuda_kernel=True
            # FINE DEBUG
        ).cuda()
    
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

        return torch.nn.functional.layer_norm(pcen, (pcen.shape[-1],), eps=1e-6)
    
    def get_pcen_parameters(self):
        return self.pcen.get_parameters()
    

@torch.compile(mode="reduce-overhead")
def change_dimensions(x):
    return x.unsqueeze(1).repeat(1, 3, 1, 1)


class BirdModel(nn.Module):
    def __init__(self,
                num_classes,
                sample_rate=32000,
                n_fft=1024,
                hop_length=320,
                n_mels=128,
                trainable_pcen=True,
                #DEBUG
                epoch=0,
                last_layer_flag=True #set to False to analyze extracted features
                #FINE DEBUG
    ):
        super().__init__()
        self.num_classes=num_classes
        self.sample_rate=sample_rate
        self.n_fft=n_fft
        self.hop_length=hop_length
        self.n_mels=n_mels
        self.trainable_pcen=trainable_pcen
        #DEBUG
        self.epoch=epoch
        self.epoch_spectrogram_printing_flag = {0: 1, 20: 1}
        #FINE DEBUG

        # frontend PCEN
        self.frontend = PCENFrontend(sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            trainable_pcen=trainable_pcen
        )

        # backbone pre-trained
        self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        # parameters freezing
        for p in self.backbone.features.parameters():
            p.requires_grad = False

        # Modifica primo layer per 1 canale (invece di 3)
        """
        # OPTION A: new trainable layer with 1 channel in, same channels out
        # OPTION B: see forward() method
        first_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False
        )
        """

        # testa di classificazione
        if last_layer_flag:
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Linear(in_features, num_classes)
            )

    def forward(self, x):
        """
        input x: [B, T]
        post-frontend: [B, mel, time]
        pre-backbone:
            OPTION A: [B, 1, mel, time]
            OPTION B: [B, 3, mel, time] ACTIVE NOW!
        post-backbone: [B, num_classes]
        """

        x = self.frontend(x)   # [B, mel, time]

        #DEBUG
        """
        if self.epoch_spectrogram_printing_flag.get(self.epoch, 0):
            print("Vorrei plottare: "+str(x[0].shape))
            self.plot_spectrogram(x[0])
            self.epoch_spectrogram_printing_flag[self.epoch] = 0
        """
        #FINE DEBUG
        
        """
        # OPTION A: see __init__() method
        x = self.backbone(x.unsqueeze(1))
        # OPTION B: repeat the spectrogram over 3 channels
        """
        # aggiungi channel dim
        #x = self.backbone(x.unsqueeze(1).repeat(1, 3, 1, 1))   # [B, num_classes]
        x = self.backbone(change_dimensions(x))   # [B, num_classes]

        return x
    
    #DEBUG
    def set_epoch(self, epoch):
        self.epoch=epoch
    
    def plot_spectrogram(self, x):
        plt.figure()
        _, axes = plt.subplots(1, 1, figsize=(14, 6), sharex=True) # "_" sarebbe fig, boh
        
        # --- Spettrogramma ---
        img = librosa.display.specshow(
            x.detach().cpu().numpy().T,
            x_axis='time',
            y_axis='hz',
            sr=self.sample_rate,
            hop_length=self.hop_length,
            ax=axes
        )
        axes.set_title("hop_length: "+ str(self.hop_length))
    
        plt.colorbar(img, ax=axes, label="PCEN value")
        plt.tight_layout()
        plt.savefig(str("spectrogram_at_epoch_"+str(self.epoch)))

    def get_pcen_parameters(self):
        return self.frontend.get_pcen_parameters()
    #FINE DEBUG


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        output = self.model(x)

        if class_idx is None:
            class_idx = output.argmax(dim=1)

        print("Looking at: "+str(class_idx))
        loss = output[:, class_idx]
        self.model.zero_grad()
        loss.backward(torch.ones_like(loss))

        # pesi = media dei gradienti
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        return cam