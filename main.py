import warnings
from sklearn.exceptions import UndefinedMetricWarning
import json
import time

import os
import pandas as pd
import torch
import librosa
from torch.utils.data import Dataset
from collections import OrderedDict
import torch.nn as nn
import custom_PCEN
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score
import birdCLEF_ROCAUC
import matplotlib.pyplot as plt
from pathlib import Path

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
        
        # OLD: with cache
        if filepath not in self.cache:
            if len(self.cache) >= self.cache_size:
                self.cache.popitem(last=False)

            self.cache[filepath], _ = librosa.load(filepath, sr=self.sample_rate)

        audio = self.cache[filepath]
        """
        # NEW: no cache
        audio, _ = librosa.load(filepath, sr=self.sample_rate)
        """
        # FINE DEBUG

        chunk = audio[start:start + self.chunk_size]

        # padding se necessario
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
            sr=sample_rate
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
                #DEBUG
                epoch=0
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

        #DEBUG: da capire se il primo layer è trainable così
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

        #DEBUG
        """
        if self.epoch_spectrogram_printing_flag.get(self.epoch, 0):
            print("Vorrei plottare: "+str(x[0].shape))
            self.plot_spectrogram(x[0])
            self.epoch_spectrogram_printing_flag[self.epoch] = 0
        """
        #FINE DEBUG
        
        # aggiungi channel dim
        x = self.backbone(x.unsqueeze(1))   # [B, num_classes]

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
    #FINE DEBUG


def batch_f1(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred > threshold).int().cpu()
    y_true_int = y_true.int().cpu()
    return f1_score(y_true_int, y_pred_bin, average="macro", zero_division=0)

def batch_recall(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred > threshold).int().cpu()
    y_true_int = y_true.int().cpu()
    
    #Keeping only classes with at least one occurence in y_true_int
    valid_classes = (y_true_int.sum(dim=0) > 0)

    # Return 0 when no classes are present
    if valid_classes.sum() == 0:
        return 0.0

    y_true_filtered = y_true_int[:, valid_classes]
    y_pred_filtered = y_pred_bin[:, valid_classes]

    return recall_score(y_true_filtered, y_pred_filtered, average="macro", zero_division=0)

def plot_metric(train_values, val_values, name, path):
    plt.figure()
    plt.plot(train_values, label="train")
    plt.plot(val_values, label="validation")
    plt.xlabel("epoch")
    plt.ylabel(name)
    plt.legend()
    plt.title(f"Train vs Validation {name}")
    plt.savefig(path)
    plt.close()

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_total_params(model):
    return sum(p.numel() for p in model.parameters())

def experimental_campaign(results_path, sample_rate, hop_length, n_fft, n_mels, num_epochs):
    #warning policy
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    train_root_dir_dict = {"real":"./data/train_soundscapes", "synthetic": "./data/synthetic_train_soundscapes"}
    val_root_dir_dict = {"real":"./data/train_soundscapes", "synthetic": "./data/synthetic_validation_soundscapes"}

    #DF CREATION
    species_list = []
    # Train
    df_train = pd.read_csv("./data/bigger_train_soundscapes_labels.csv").drop_duplicates()
    df_train["start"] = pd.to_timedelta(df_train["start"]).dt.total_seconds()
    df_train["end"] = pd.to_timedelta(df_train["end"]).dt.total_seconds()

    species_list_train = []
    for item in df_train["primary_label"].str.split(";").explode().unique():
        if str(item)!="nan":
            species_list_train.append(item)
    num_classes = len(species_list_train)
    print("Train number of classes: "+str(num_classes))
    species_list.extend(species_list_train)
    #print(str(sorted(species_list)))

    # Validation
    df_validation = pd.read_csv("./data/validation_soundscapes_labels.csv").drop_duplicates()
    df_validation["start"] = pd.to_timedelta(df_validation["start"]).dt.total_seconds()
    df_validation["end"] = pd.to_timedelta(df_validation["end"]).dt.total_seconds()

    species_list_validation = []
    for item in df_validation["primary_label"].str.split(";").explode().unique():
        if str(item)!="nan":
            species_list_validation.append(item)
    num_classes = len(species_list_validation)
    print("Validation number of classes: "+str(num_classes))

    species_list.extend(species_list_validation)
    species_list = sorted(list(set(species_list)))
    #print(str(species_list))
    species_to_idx = {s: i for i, s in enumerate(species_list)}
    num_classes = len(species_list)
    print("Total number of classes: "+str(num_classes))

    # Nel training manca(va)no '47158son02', '47158son14' mentre '43435' ha(veva) un solo record
    #input("Partiamo?")

    #DATASET CREATION
    # Train
    train_dataset = SoundscapeDataset(
    	root_dir=train_root_dir_dict,
    	df=df_train,
    	species_to_idx=species_to_idx,
    	sample_rate=sample_rate
    )

    # Validation
    val_dataset = SoundscapeDataset(
    	root_dir=val_root_dir_dict,
    	df=df_validation,
    	species_to_idx=species_to_idx,
    	sample_rate=sample_rate
    )

    # TRAIN LOOP: setup
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # pos_weights to correct imbalanced dataset
    pos_weights = torch.zeros(num_classes, device=device)

    for _, _, labels in train_dataset.samples:
        for label in labels:
            if label in species_to_idx:
                pos_weights[species_to_idx[label]] += 1

    print("Count")
    print(str(pos_weights))

    total_samples = len(train_dataset.samples)
    for i in range(len(pos_weights)):
        pos_weights[i] += 1e-6
        # Previously: pos_weights[i] = min(int((total_samples-pos_weights[i])/pos_weights[i]), 50)
        # pos_weights[i] = max(min(int(total_samples/(pos_weights[i]*len(pos_weights))), 1000), 1)
        pos_weights[i] = min(total_samples/(pos_weights[i]*len(pos_weights)), 100)

    for i in range(len(pos_weights)):
        pos_weights[i] /= min(pos_weights)

    print("Weigths")
    print(str(pos_weights))
    print(min(pos_weights))
    print(max(pos_weights))

    
    train_workers = 1
    batch_size=64
    lr = 0.01*batch_size/256

    model = BirdModel(num_classes=num_classes,
                    sample_rate=sample_rate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=n_mels,
                    trainable_pcen=True
                ).to(device)
    optimizer = torch.optim.Adam([
        {"params": model.backbone.features.parameters(), "lr": lr/10},
        {"params": model.backbone.classifier.parameters(), "lr": lr},
        {"params": model.frontend.pcen.parameters(), "lr": lr/10}
    ])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(pos_weights)).to(device)  # preferibile a BCE(sigmoid) per stabilità

    # Model complexity vs dataset size
    trainable = count_trainable_params(model)
    total = count_total_params(model)

    print(f"Trainable: {trainable}")
    print(f"Total: {total}")
    print(f"Ratio (expected < 0.1 since MobileNet is a feature extractor): {trainable/total:.3f}")
    print(f"Sample per param (expected 10-100): {df_train.shape[0]/trainable}")
    input("Quindi? Come siam messi?")

    # dataset: train_dataset e val_dataset già istanziati
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=train_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True, pin_memory=True)

    # TRAIN LOOP: execution
    try:
        with open(results_path+"/list_train_loss.json", "r") as a, open(results_path+"/list_train_recall.json", "r") as b, open(results_path+"/list_train_rocauc.json", "r") as c:
            list_train_loss = json.load(a)
            list_train_recall = json.load(b)
            list_train_rocauc = json.load(c)

        with open(results_path+"/list_val_loss.json", "r") as d, open(results_path+"/list_val_recall.json", "r") as e, open(results_path+"/list_val_rocauc.json", "r") as f:
            list_val_loss = json.load(d)
            list_val_recall = json.load(e)
            list_val_rocauc = json.load(f)

        checkpoint = torch.load(results_path+"/model_checkpoint.pth", map_location=device)
        
        state_dict = checkpoint["model_state_dict"]
        # Set BirdModel.last_state to by-pass loading issues
        checkpoint["model_state_dict"]["frontend.pcen.last_state"] = torch.zeros(n_mels)
        
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        starting_epoch = checkpoint["epoch"]
        print("Successfully loaded model "+str(results_path+"/model_checkpoint.pth"))

    except Exception as e:
        print(e)
        print("No previous run detected: setting up a new one")
        list_train_loss = [0 for i in range(num_epochs)]
        list_train_recall = [0 for i in range(num_epochs)]
        list_train_rocauc = [0 for i in range(num_epochs)]
        list_val_loss = [0 for i in range(num_epochs)]
        list_val_recall = [0 for i in range(num_epochs)]
        list_val_rocauc = [0 for i in range(num_epochs)]
        starting_epoch = 0

    fine_tuning_epoch_threshold = int(num_epochs*0.9)

    for epoch in range(starting_epoch, num_epochs):
        #DEBUG
        model.set_epoch(epoch)
        #FINE DEBUG

        if epoch == starting_epoch:
            print("Training started")
        
        # Final backbone fine tuning
        if epoch>fine_tuning_epoch_threshold:
            for p in model.backbone.features[-2:].parameters():
                p.requires_grad = True
            print("Fine tuning attivo")
        
        # -------------------
        # TRAIN
        # -------------------
        model.train()
        running_loss = 0.0
        running_f1 = 0.0
        running_ROCAUC = 0.0
        running_recall = 0.0
        total_batches = df_train.shape[0]
        batch = 0

        true_start = time.perf_counter()
        for x, y, _, _ in train_loader:
            batch += 1
            if batch%500==0:
                print("Batch "+str(batch)+"/"+str(int(total_batches/batch_size)))
            x, y = x.to(device), y.to(device)

            y_true = pd.DataFrame(y.cpu().numpy(), columns=species_list)

            """
            # DEBUG
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True) #parameter suggested by pytorch performance tuning guide
            end = time.perf_counter()
            print(f"Tempo optimizer: {end - start:.6f} secondi")

            start = time.perf_counter()
            outputs = model(x)  # logits (sigmoid in forward opzionale quindi la applichiamo dopo)
            end = time.perf_counter()
            print(f"Tempo forward: {end - start:.6f} secondi")

            start = time.perf_counter()
            loss = criterion(outputs, y)
            end = time.perf_counter()
            print(f"Tempo loss: {end - start:.6f} secondi")

            start = time.perf_counter()
            loss.backward()
            end = time.perf_counter()
            print(f"Tempo backward: {end - start:.6f} secondi")
            
            start = time.perf_counter()
            optimizer.step()
            end = time.perf_counter()
            print(f"Tempo step: {end - start:.6f} secondi")
            # FINE DEBUG

            """
            optimizer.zero_grad(set_to_none=True) #parameter suggested by pytorch performance tuning guide
            outputs = model(x)  # logits (sigmoid in forward opzionale quindi la applichiamo dopo)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            

            running_loss += loss.item() * x.size(0)
            running_f1 += batch_f1(y, torch.sigmoid(outputs)) * x.size(0)
            running_recall += batch_recall(y, torch.sigmoid(outputs)) * x.size(0)

            #DEBUG: vanno sistemate le row_ids (che al momento non stanno venendo usate ma potrebbero)
            row_ids = ["{}_{}".format(f"soundscape_{i}", 5) for i in range(y.size(0))]  # esempio row_id
            y_true = pd.DataFrame(y.cpu().numpy(), columns=species_list)
            y_true.insert(0, "row_id", row_ids)

            y_pred_df = pd.DataFrame(torch.sigmoid(outputs).detach().cpu().numpy(), columns=species_list)
            y_pred_df.insert(0, "row_id", row_ids)
            
            running_ROCAUC += birdCLEF_ROCAUC.score(solution=y_true, submission=y_pred_df, row_id_column_name="row_id") * x.size(0)
            #FINE DEBUG

        true_end = time.perf_counter()
        print(f"Epoch running time: {(true_end - true_start)/60:.6f} minutes")
            

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_f1 = running_f1 / len(train_loader.dataset)
        epoch_ROCAUC = running_ROCAUC / len(train_loader.dataset)
        epoch_recall = running_recall / len(train_loader.dataset)

        # -------------------
        # VALIDATION
        # -------------------
        model.eval()
        val_loss = 0.0
        val_f1 = 0.0
        val_ROCAUC = 0.0
        val_recall = 0.0
        with torch.no_grad():
            for x_val, y_val, _, _ in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                outputs_val = model(x_val)
                loss_val = criterion(outputs_val, y_val)
                val_loss += loss_val.item() * x_val.size(0)
                val_f1 += batch_f1(y_val, torch.sigmoid(outputs_val)) * x_val.size(0)
                val_recall += batch_recall(y_val, torch.sigmoid(outputs_val)) * x_val.size(0)

                #DEBUG: vanno sistemate le row_ids (che al momento non stanno venendo usate ma potrebbero)
                row_ids = ["{}_{}".format(f"soundscape_{i}", 5) for i in range(y.size(0))]  # esempio row_id
                y_val = pd.DataFrame(y.cpu().numpy(), columns=species_list)
                y_val.insert(0, "row_id", row_ids)

                y_val_pred_df = pd.DataFrame(torch.sigmoid(outputs_val).detach().cpu().numpy(), columns=species_list)
                y_val_pred_df.insert(0, "row_id", row_ids)

                val_ROCAUC += birdCLEF_ROCAUC.score(y_val, y_val_pred_df, row_id_column_name="row_id") * x.size(0)
                #FINE DEBUG

        val_loss /= len(val_loader.dataset)
        val_f1 /= len(val_loader.dataset)
        val_ROCAUC /= len(val_loader.dataset)
        val_recall /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{num_epochs} "
            f"- Train Loss: {epoch_loss:.4f}, Train F1: {epoch_f1:.4f} "
            f"- Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
        
        print(f"Train recall: {epoch_recall:.4f} "
            f"- Val recall: {val_recall:.4f}")

        print(f"Train ROCAUC: {epoch_ROCAUC:.4f} "
            f"- Val ROCAUC: {val_ROCAUC:.4f}")

        list_train_loss[epoch] += epoch_loss
        list_train_recall[epoch] += epoch_recall
        list_train_rocauc[epoch] += epoch_ROCAUC
        list_val_loss[epoch] += val_loss
        list_val_recall[epoch] += val_recall
        list_val_rocauc[epoch] += val_ROCAUC

        with open(results_path+"/list_train_loss.json", "w") as a, open(results_path+"/list_train_recall.json", "w") as b, open(results_path+"/list_train_rocauc.json", "w") as c:
            json.dump(list_train_loss, a)
            json.dump(list_train_recall, b)
            json.dump(list_train_rocauc, c)

        with open(results_path+"/list_val_loss.json", "w") as d, open(results_path+"/list_val_recall.json", "w") as e, open(results_path+"/list_val_rocauc.json", "w") as f:
            json.dump(list_val_loss, d)
            json.dump(list_val_recall, e)
            json.dump(list_val_rocauc, f)


        # Save final model and training graphs
        if os.path.exists(results_path+"/model_checkpoint.pth"):
            os.remove(results_path+"/model_checkpoint.pth")
            print("Deleted checkpoint")
        else:
            print("No previous checkpoint to be deleted")
        
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        }, results_path+"/model_checkpoint.pth")

    plot_metric(list_train_loss, list_val_loss, "loss", results_path+"/train_vs_validation_loss.png")
    plot_metric(list_train_recall, list_val_recall, "recall", results_path+"/train_vs_validation_recall.png")
    plot_metric(list_train_rocauc, list_val_rocauc, "rocauc", results_path+"/train_vs_validation_rocauc.png")

if __name__ == "__main__":
    torch.cuda.memory.set_per_process_memory_fraction(1.0)
    
    sample_rate=32000
    hops = [320]
    n_fft = [320*4] #il default presente in documentazione è n_hops = floor(n_fft / 4)
    n_mels = [200]
    session_ID = "bigger_dataset_2"

    num_epochs = 30

    # Enables automatic kernel selection for performance: does not seem to improve performances
    #torch.backends.cudnn.benchmark = True

    for i in range(len(hops)):
        for j in range(len(n_mels)):
            results_path = "./results/session_"+str(session_ID)+"/"+str(hops[i])+"_"+str(n_mels[j])
            Path(results_path).mkdir(parents=True, exist_ok=True)
            experimental_campaign(results_path, sample_rate, hops[i], n_fft[i], n_mels[j], num_epochs=num_epochs)