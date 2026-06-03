import warnings
from sklearn.exceptions import UndefinedMetricWarning
import json

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score
import birdCLEF_ROCAUC
import matplotlib.pyplot as plt
from pathlib import Path
import custom_classes


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

def build_metric_dataframes(y_true_batches, y_pred_batches, species_list):
    y_true_tensor = torch.cat(y_true_batches, dim=0)
    y_pred_tensor = torch.cat(y_pred_batches, dim=0)
    row_ids = [f"soundscape_{i}_5" for i in range(y_true_tensor.size(0))]

    y_true_df = pd.DataFrame(y_true_tensor.numpy(), columns=species_list)
    y_true_df.insert(0, "row_id", row_ids)

    y_pred_df = pd.DataFrame(y_pred_tensor.numpy(), columns=species_list)
    y_pred_df.insert(0, "row_id", row_ids)

    return y_true_df, y_pred_df

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

    # TRAIN LOOP: setup
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        with open(results_path+"/pos_weights.json", "r") as a:
            pos_weights_serial = json.load(a)
        pos_weights = torch.Tensor(pos_weights_serial)
        print("Loaded weights from "+results_path+"/pos_weights.json")
    
    except Exception as e:
        print(e)
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
            # Previously: pos_weights[i] = min(int((total_samples-pos_weights[i])/pos_weights[i]), 50)
            # pos_weights[i] = max(min(int(total_samples/(pos_weights[i]*len(pos_weights))), 1000), 1)
            pos_weights[i] = min(total_samples/(pos_weights[i]*len(pos_weights)), 100)

        for i in range(len(pos_weights)):
            pos_weights[i] /= min(pos_weights)

        pos_weights_serial = pos_weights.tolist()
        with open(results_path+"/pos_weights.json", "w") as a:
            json.dump(pos_weights_serial, a)

    """print("Count")
    print(str(pos_weights))

    print("Weigths")
    print(str(pos_weights))
    print(min(pos_weights))
    print(max(pos_weights))"""

    """
    # LOAD TRAINING METRICS
    with open(results_path+"/list_train_loss.json", "r") as a, open(results_path+"/list_train_recall.json", "r") as b, open(results_path+"/list_train_rocauc.json", "r") as c:
        list_train_loss = json.load(a)
        list_train_recall = json.load(b)
        list_train_rocauc = json.load(c)

    with open(results_path+"/list_val_loss.json", "r") as d, open(results_path+"/list_val_recall.json", "r") as e, open(results_path+"/list_val_rocauc.json", "r") as f:
        list_val_loss = json.load(d)
        list_val_recall = json.load(e)
        list_val_rocauc = json.load(f)
    
    plot_metric(list_train_loss, list_val_loss, "loss", results_path+"/train_vs_validation_loss.png")
    plot_metric(list_train_recall, list_val_recall, "recall", results_path+"/train_vs_validation_recall.png")
    plot_metric(list_train_rocauc, list_val_rocauc, "rocauc", results_path+"/train_vs_validation_rocauc.png")

    input("Immagini plottate, vuoi continuare?")
    """

    batch_size=32

    model = custom_classes.BirdModel(num_classes=num_classes,
                    sample_rate=sample_rate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=n_mels,
                    trainable_pcen=True
                ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(pos_weights)).to(device)  # preferibile a BCE(sigmoid) per stabilità
    model.compile(mode="reduce-overhead")
    print("Model compiled")

    # dataset: train_dataset e val_dataset già istanziati
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True, pin_memory=True)

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

    print(model.get_pcen_parameters())

    # -------------------
    # VALIDATION
    # -------------------
    model.eval()
    val_loss = 0.0
    val_f1 = 0.0
    val_recall = 0.0
    spec_to_index_dict = {
        species_list[i]:i for i in range(len(species_list))
    }
    val_y_true_batches = []
    val_y_pred_batches = []
    with torch.no_grad():
        batch = 0
        print("Batch "+str(batch+1)+"/"+str(int(df_validation.shape[0]/batch_size)))
        for x_val, y_val, _, _ in val_loader:
            batch += 1
            if batch%100==0:
                print("Batch "+str(batch)+"/"+str(int(df_validation.shape[0]/batch_size)))
            
            x_val, y_val = x_val.to(device), y_val.to(device)
            outputs_val = model(x_val)
            loss_val = criterion(outputs_val, y_val)
            val_loss += loss_val.item() * x_val.size(0)
            probs_val = torch.sigmoid(outputs_val)
            val_f1 += batch_f1(y_val, probs_val) * x_val.size(0)
            val_recall += batch_recall(y_val, probs_val) * x_val.size(0)
            val_y_true_batches.append(y_val.detach().cpu())
            val_y_pred_batches.append(probs_val.detach().cpu())

    val_loss /= len(val_loader.dataset)
    val_f1 /= len(val_loader.dataset)
    val_recall /= len(val_loader.dataset)
    y_val_df, y_val_pred_df = build_metric_dataframes(val_y_true_batches, val_y_pred_batches, species_list)
    val_ROCAUC = birdCLEF_ROCAUC.score(y_val_df.copy(), y_val_pred_df.copy(), row_id_column_name="row_id")
    scored_per_class_val_ROCAUC, scored_columns = birdCLEF_ROCAUC.per_class_score(
        y_val_df.copy(),
        y_val_pred_df.copy(),
        row_id_column_name="row_id"
    )
    per_class_val_ROCAUC = np.full(len(species_list), np.nan)
    for score, spec in zip(scored_per_class_val_ROCAUC, scored_columns):
        per_class_val_ROCAUC[spec_to_index_dict[spec]] = score

    print(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
    
    print(f"Val recall: {val_recall:.4f}")

    print(f"Val ROCAUC: {val_ROCAUC:.4f}")

    print("Per class val ROCAUC:")
    k = 0
    for spec in species_list:
        print(str(k)+" "+str(spec)+": "+str(per_class_val_ROCAUC[spec_to_index_dict[spec]].round(4)))
        k+=1

if __name__ == "__main__":
    torch.cuda.memory.set_per_process_memory_fraction(1.0)
    
    sample_rate=32000
    hops = [160]
    n_fft = [1280] #il default presente in documentazione è n_hops = floor(n_fft / 4)
    n_mels = [200]
    session_ID = "mobilenet_training_scheduler"
    num_epochs = 30

    for i in range(len(hops)):
        for j in range(len(n_mels)):
            results_path = "./results/session_"+str(session_ID)+"/"+str(hops[i])+"_"+str(n_mels[j])+"/model_checkpoint_6"
            Path(results_path).mkdir(parents=True, exist_ok=True)
            experimental_campaign(results_path, sample_rate, hops[i], n_fft[i], n_mels[j], num_epochs=num_epochs)
