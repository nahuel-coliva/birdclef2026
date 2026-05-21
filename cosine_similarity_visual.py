import numpy as np
import torch
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

def heatmap(path, param_dict, summary_path):
    # Carica dizionario
    with open(path, "r") as f:
        d = json.load(f)

    labels = list(d.keys())
    labels_length = len(labels)

    intra = np.array([d[c][c] for c in labels]).mean()
    elements = []
    for r in range(labels_length-1):
        for c in range(r+1,labels_length):
            elements.append(d[labels[r]][labels[c]])

    inter = np.array(elements).mean()
    print("Intra: "+str(intra))
    print("Inter: "+str(inter))

    with open(summary_path+"/summary.txt", "a") as file1:
        file1.write(str(param_dict)+"\n")
        file1.write("Intra: "+str(intra)+"\nInter: "+str(inter)+"\n")

    
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
    plt.show()
    

if __name__ == "__main__":
    torch.cuda.memory.set_per_process_memory_fraction(1.0)
    
    mode = "_"
    while mode not in ["ciclo", "ondemand"]:
        mode = input("ciclo o ondemand? ")

    sample_rate = 32000
    
    if mode=="ciclo":
        hops = [120, 160, 320]
        n_fft = [640, 1280, 3072] #il default presente in documentazione è n_hops = floor(n_fft / 4)
        n_mels = [128, 200, 256]
        session_ID = "performance_test_02"

    elif mode=="ondemand":
        hops = [int(input("Hops: "))]
        n_fft = [int(input("n_fft: "))]
        n_mels = [int(input("Mels: "))]
        session_ID = input("Session ID (performance_test): ")
    

    for i in range(len(hops)):
        for j in range(len(n_fft)):
            for k in range(len(n_mels)):
                results_path = "./results/session_"+str(session_ID)+"/"+str(hops[i])+"_"+str(n_mels[k])
                model_path = "./results/session_"+str(session_ID)
                Path(results_path).mkdir(parents=True, exist_ok=True)
                print("Parameters: hops "+str(hops[i])+" - n_fft "+str(n_fft[j])+" - mels "+str(n_mels[k]))
                cosine_similarity_path = results_path+"/cosine_similarity_"+str(hops[i])+"_"+str(n_fft[j])+"_"+str(n_mels[k])+".json"
                param_dict = {
                    "hops": hops[i]
                    , "n_fft": n_fft[j]
                    , "n_mels": n_mels[k]
                }
                heatmap(cosine_similarity_path, param_dict, model_path)