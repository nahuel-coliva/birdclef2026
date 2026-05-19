import warnings
from sklearn.exceptions import UndefinedMetricWarning

import pandas as pd
import numpy as np
import torch
from pathlib import Path
import custom_classes
import librosa
import os
from torch.nn.functional import cosine_similarity
import json
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import time


def safe_isnan(item):
    try:
        return np.isnan(item)
    except Exception as e:
        return False

def lambda_function(x, item):
  if isinstance(x, list):
    return item in x
  return safe_isnan(item) and safe_isnan(x)

def heatmap(path):
    # Carica dizionario
    with open(path, "r") as f:
        d = json.load(f)

    labels = list(d.keys())
    labels_length = len(labels)

    intra = np.array([d[c][c] for c in labels]).mean()
    for r in range(labels_length-1):
        for c in range(r+1,labels_length):
            elements = [d[labels[r]][labels[c]]]
    inter = np.array(elements).mean()
    print("Intra: "+str(intra))
    print("Inter: "+str(inter))

    # Matrice simmetrica
    mat = np.array([[d[r][c] for c in labels] for r in labels])

    fig, ax = plt.subplots(figsize=(18, 16))

    levels = np.linspace(mat.min(), mat.max(), 6)  # 5 intervalli
    norm = BoundaryNorm(levels, ncolors=256)

    im = ax.imshow(mat, cmap="viridis", norm=norm)

    # Tick più leggibili
    step = 5
    ax.set_xticks(np.arange(0, len(labels), step))
    ax.set_yticks(np.arange(0, len(labels), step))

    ax.set_xticklabels(labels[::step], rotation=90, fontsize=8)
    ax.set_yticklabels(labels[::step], fontsize=8)

    fig.colorbar(im, ax=ax)

    plt.tight_layout()
    #plt.show()

def mean_cosine_similarity(species_1, species_2, embeddings):
    sim = 0
    if species_1==species_2:
        a_set = list(embeddings[species_1].keys())
        elements = len(a_set)
        magnitude = elements*(elements-1)/2
        
        for a in range(elements-1):
            for b in range(a+1,elements):
                sim += cosine_similarity(embeddings[species_1][a_set[a]], embeddings[species_2][a_set[b]], dim=0)

    else:
        a_set = set(embeddings[species_1].keys()).difference(embeddings[species_2].keys())
        b_set = set(embeddings[species_2].keys()).difference(embeddings[species_1].keys())
        magnitude = len(a_set)*len(b_set)
    
        for a in a_set:
            for b in b_set:
                sim += cosine_similarity(embeddings[species_1][a], embeddings[species_2][b], dim=0)
    
    if magnitude == 0:
        return 0
    return float(sim/magnitude)

def tensor_mean_cosine_similarity(species_1, species_2, embeddings, device):
    if species_1==species_2:
        a_set = list(embeddings[species_1].keys())
        elements = len(a_set)
        magnitude = elements*(elements-1)/2

        if magnitude == 0:
            return 0
        
        embeddings_a = embeddings[species_1][a_set[0]].repeat(elements-1, 1)
        embeddings_b = torch.Tensor().to(device)
        embeddings_b_work_tensor = torch.cat([embeddings[species_2][a].unsqueeze(0) for a in a_set], 0)
        for a in range(1, elements-1):
            embeddings_a = torch.cat([embeddings_a, embeddings[species_1][a_set[a]].repeat(elements-1-a, 1)], 0)

        for b in range(1,elements):
            embeddings_b = torch.cat([embeddings_b, embeddings_b_work_tensor[b:]], 0)

    else:
        a_set = set(embeddings[species_1].keys()).difference(embeddings[species_2].keys())
        b_set = set(embeddings[species_2].keys()).difference(embeddings[species_1].keys())
        magnitude = len(a_set)*len(b_set)

        if magnitude == 0:
            return 0

        embeddings_a = torch.cat([embeddings[species_1][a].unsqueeze(0) for a in a_set], 0).repeat(len(b_set), 1)
        embeddings_b = torch.cat([embeddings[species_2][b].unsqueeze(0) for b in b_set], 0).repeat(len(a_set), 1)
    
    sim = cosine_similarity(embeddings_a, embeddings_b, dim=1).mean()
    
    return float(sim)


#
# Keep the same validation code as the main loop, add per-class metrics
#
def experimental_campaign(results_path, model_path, sample_rate, hop_length, n_fft, n_mels):
    #warning policy
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    val_root_dir_dict = {"real":"./data/train_soundscapes", "synthetic": "./data/synthetic_validation_soundscapes"}

    #DF CREATION
    # Validation
    df_validation = pd.read_csv("./data/cosine_similarity_labels.csv").drop_duplicates()
    df_validation["start"] = pd.to_timedelta(df_validation["start"]).dt.total_seconds()
    df_validation["end"] = pd.to_timedelta(df_validation["end"]).dt.total_seconds()
    df_validation["primary_label"] = df_validation["primary_label"].str.split(";")

    el_per_class = 10
    new_df = pd.DataFrame()
    embeddings = {}
    for item in df_validation["primary_label"].explode().unique():
        item_rows = df_validation[df_validation["primary_label"].apply(lambda x: lambda_function(x, item))]
        new_df = pd.concat([new_df, item_rows.iloc[:el_per_class]])
        embeddings["silence" if safe_isnan(item) else item] = {}
    
    df_validation = new_df
    num_classes = len(df_validation["primary_label"].explode().unique())
    print("Validation number of classes: "+str(num_classes))


    # TRAIN LOOP: setup
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = custom_classes.BirdModel(num_classes=num_classes,
                    sample_rate=sample_rate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=n_mels,
                    trainable_pcen=True,
                    last_layer_flag=False
                ).to(device)
    #model.compile(mode="reduce-overhead")
    print("Model compiled")

    # Model loading
    try:
        checkpoint = torch.load(model_path+"/model_checkpoint.pth", map_location=device)
        
        state_dict = checkpoint["model_state_dict"]

        # Just load the feature extractor
        del state_dict["backbone.classifier.0.weight"]
        del state_dict["backbone.classifier.0.bias"]
        del state_dict["frontend.pcen.f2m.fb"]
        del state_dict["frontend.pcen.window"]
        
        # Set BirdModel.last_state to by-pass loading issues
        checkpoint["model_state_dict"]["frontend.pcen.last_state"] = torch.zeros(n_mels)
        
        model.load_state_dict(state_dict, strict=False)
        print("Successfully loaded model "+str(model_path+"/model_checkpoint.pth"))

    except Exception as e:
        print(e)
        print("No previous run detected: setting up a new one")

    # -------------------
    # VALIDATION
    # -------------------
    model.eval()
    with torch.no_grad():
        batch = 0
        for _, item in df_validation.iterrows():
            batch += 1
            if batch%500==0:
                print("Batch "+str(batch)+"/"+str(int(df_validation.shape[0])))
            
            if item["filename"].count("BC2026")>1:
                actual_root_dir = val_root_dir_dict["synthetic"]
            else:
                actual_root_dir = val_root_dir_dict["real"]
            filepath = os.path.join(actual_root_dir, item["filename"])
            audio, _ = librosa.load(filepath, sr=sample_rate)
            start = int(item["start"])*sample_rate
            # Manca la dimensione della batch size, probabilmente la aggiungeremo poi, intanto unsqueeze
            x_val = torch.tensor(audio[start:start + sample_rate*5]).float().to(device).unsqueeze(0)

            if isinstance(item["primary_label"], list):
                label_list = item["primary_label"]
            else:
                label_list = ["silence" if np.isnan(item["primary_label"]) else item["primary_label"]]

            for label in label_list:
                embeddings[label][item["filename"]+"_"+str(start)] = model(x_val).squeeze()
    
    
    
    species_list = list(embeddings.keys())
    species_list_length = len(species_list)

    cosine_similarity = {}
    for species in species_list:
        cosine_similarity[species] = {}

    avanzamento = 0
    start = time.perf_counter()
    
    print("Specie "+str(avanzamento)+"/"+str(species_list_length))
    for i in range(species_list_length):
        species_1 = species_list[i]
        avanzamento += 1
        if avanzamento%10==0:
            print("Specie "+str(avanzamento)+"/"+str(species_list_length))
        for j in range(i,species_list_length):
            species_2 = species_list[j]
            
            #cosine_similarity[species_1][species_2] = mean_cosine_similarity(species_1, species_2, embeddings)
            cosine_similarity[species_1][species_2] = tensor_mean_cosine_similarity(species_1, species_2, embeddings, device)

            cosine_similarity[species_2][species_1] = cosine_similarity[species_1][species_2]
    
    end = time.perf_counter()
    print(f"Tempo: {end - start:.6f} secondi")

    print()
    
    cosine_similarity_path = results_path+"/cosine_similarity_"+str(hop_length)+"_"+str(n_fft)+"_"+str(n_mels)+".json"
    with open(cosine_similarity_path, "w") as a:
        json.dump(cosine_similarity, a)
    
    #
    # VISUALIZZAZIONE
    #
    heatmap(cosine_similarity_path)


if __name__ == "__main__":
    torch.cuda.memory.set_per_process_memory_fraction(1.0)
    
    mode = "_"
    while mode not in ["ciclo", "ondemand"]:
        mode = input("ciclo o ondemand? ")

    sample_rate = 32000
    
    if mode=="ciclo":
        hops = [160, 313]
        n_fft = [1280, 3072] #il default presente in documentazione è n_hops = floor(n_fft / 4)
        n_mels = [128, 256]
        session_ID = "performance_test"

    elif mode=="ondemand":
        hops = [int(input("Hops: "))]
        n_fft = [int(input("n_fft: "))]
        n_mels = [int(input("Mels: "))]
        session_ID = input("Session ID (performance_test): ")
    

    for i in range(len(hops)):
        for j in range(len(n_mels)):
            results_path = "./results/session_"+str(session_ID)+"/"+str(hops[i])+"_"+str(n_mels[j])
            model_path = "./results/session_"+str(session_ID)
            Path(results_path).mkdir(parents=True, exist_ok=True)
            experimental_campaign(results_path, model_path, sample_rate, hops[i], n_fft[i], n_mels[j])