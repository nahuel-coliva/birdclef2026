import warnings
from sklearn.exceptions import UndefinedMetricWarning
import json

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score
import matplotlib.pyplot as plt
from pathlib import Path
import custom_classes
import custom_PCEN
import matplotlib.pyplot as plt
import librosa


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

def load_audio(path, sr, offset=0):
    return librosa.core.load(path, sr=sr, duration=5, offset=offset)[0]

def load_spectrogram(path, sr, n_mels, n_fft, hop_length, s, alpha, delta, offset):
    transform = custom_PCEN.StreamingPCENTransform(n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
                                        s=s, alpha=alpha, delta=delta
                ).cuda()

    x = torch.tensor(load_audio(path, sr, offset)).unsqueeze(0).cuda() #duration=5 : loads first 5 seconds

    spectrogram = transform(x)

    """
    if i==1:
        start = int(200/(hops[i]/sr*1000)) # taglio i primi hop in modo da partire da 200ms: ogni hop è hops/sr*1000 ms, nel nostro caso 256/32 ms; avrò bisogno di 200/(256/32) hops
    else:
        start=0
    #print("Start: "+str(start))
    spectrogram = y[0].cpu().numpy().T[:,start:]
    
    # Normalizzazione
    spec_no_streaming.append((spectrogram - spectrogram.mean(axis=1, keepdims=True)) / (spectrogram.std(axis=1, keepdims=True) + 1e-6))
    """
    spectrogram = (spectrogram - spectrogram.mean(axis=1, keepdims=True)) / (spectrogram.std(axis=1, keepdims=True) + 1e-6)
    
    return spectrogram[0].cpu().numpy().T

#
# Keep the same validation code as the main loop, add per-class metrics
#
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

    """
    #DATASET CREATION
    # Train
    train_dataset = custom_classes.SoundscapeDataset(
    	root_dir=train_root_dir_dict,
    	df=df_train,
    	species_to_idx=species_to_idx,
    	sample_rate=sample_rate
    )

    # Validation
    val_dataset = custom_classes.SoundscapeDataset(
    	root_dir=val_root_dir_dict,
    	df=df_validation,
    	species_to_idx=species_to_idx,
    	sample_rate=sample_rate
    )

    print("Dataset objects ready")
    """
    # TRAIN LOOP: setup
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        with open(results_path+"/pos_weights.json", "r") as a:
            pos_weights_serial = json.load(a)
        pos_weights = torch.Tensor(pos_weights_serial)
        print("Loaded weights from "+results_path+"/pos_weights.json")
    
    except Exception as e:
        print(e)
        """
        print("Computing weights")
        # pos_weights to correct imbalanced dataset
        pos_weights = torch.zeros(num_classes, device=device)

        for _, _, labels in train_dataset.samples:
            for label in labels:
                if label in species_to_idx:
                    pos_weights[species_to_idx[label]] += 1

        total_samples = len(train_dataset.samples)
        for i in range(len(pos_weights)):
            pos_weights[i] += 1e-6
            pos_weights[i] = min(total_samples/(pos_weights[i]*len(pos_weights)), 100)

        for i in range(len(pos_weights)):
            pos_weights[i] /= min(pos_weights)

        pos_weights_serial = pos_weights.tolist()
        with open(results_path+"/pos_weights.json", "w") as a:
            json.dump(pos_weights_serial, a)
        """


    model = custom_classes.BirdModel(num_classes=num_classes,
                    sample_rate=sample_rate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=n_mels,
                    trainable_pcen=True
                ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(pos_weights)).to(device)  # preferibile a BCE(sigmoid) per stabilità

    # TRAIN LOOP: execution
    try:
        checkpoint = torch.load(results_path+"/model_checkpoint.pth", map_location=device)
        
        state_dict = checkpoint["model_state_dict"]
        # Set BirdModel.last_state to by-pass loading issues
        checkpoint["model_state_dict"]["frontend.pcen.last_state"] = torch.zeros(n_mels)
        
        model.load_state_dict(state_dict)
        print("Successfully loaded model "+str(results_path+"/model_checkpoint.pth"))

    except Exception as e:
        print(e)
        print("No previous run detected: setting up a new one")

    model.eval()
    target_layer = model.backbone.features[-1]

    # -------------------
    # EXPLAINABILITY
    # -------------------
    #path = r".\data\synthetic_validation_soundscapes\BC2026_Train_0006_S09_00_BC2026_Train_0006_S09_00_BC2026_Train_0006_S09_45_23154.ogg"
    path = r".\data\train_audio\22967\iNat911059.ogg" # 9 ma argmax 207
    path = r".\data\train_audio\strowl1\iNat16636.ogg"# 207 ma argmax 67
    sr=32000
    hop_length = 320
    n_fft = 320*4
    n_mels = 200
    s = 0.05
    alpha = 0.9
    delta = 2
    offset=10
    spectrogram = load_spectrogram(path=path, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, s=s, alpha=alpha, delta=delta, offset=offset) # [1,H,W]
    
    gradcam = custom_classes.GradCAM(model, target_layer)

    cam = gradcam(torch.Tensor(load_audio(path, sr, offset=offset)).float().to(device).unsqueeze(0))  # [1,1,H,W]
    
    # resize alla dimensione input
    cam = F.interpolate(cam, size=spectrogram.shape, mode='bilinear', align_corners=False)

    # normalizzazione
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    cam = cam.squeeze().cpu().detach().numpy()


    # PRINT
    highlighted = spectrogram * cam
    vmin = min(x.min() for x in [spectrogram, highlighted])
    vmax = max(x.max() for x in [spectrogram, highlighted])

    fig, axes = plt.subplots(1, 2, figsize=(10,4))

    im0 = axes[0].imshow(spectrogram, aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap="jet")
    im1 = axes[1].imshow(highlighted, aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap="jet")

    fig.subplots_adjust(right=0.88)

    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im1, cax=cax)
    plt.show()

if __name__ == "__main__":
    torch.cuda.memory.set_per_process_memory_fraction(1.0)
    
    sample_rate=32000
    hops = [320]
    n_fft = [320*4] #il default presente in documentazione è n_hops = floor(n_fft / 4)
    n_mels = [200]
    session_ID = "1_point_5M_dataset_3_channels"

    num_epochs = 30

    for i in range(len(hops)):
        for j in range(len(n_mels)):
            results_path = "./results/session_"+str(session_ID)+"/"+str(hops[i])+"_"+str(n_mels[j])
            Path(results_path).mkdir(parents=True, exist_ok=True)
            experimental_campaign(results_path, sample_rate, hops[i], n_fft[i], n_mels[j], num_epochs=num_epochs)