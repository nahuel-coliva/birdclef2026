import numpy as np
import torch
from pathlib import Path
import json
import matplotlib.pyplot as plt

def histogram(path, param_dict, results_path):
    # Carica dizionario
    with open(path, "r") as f:
        d = json.load(f)

    labels = []
    values = []
    for key, value in d.items():
        labels.append(key)
        values.append(value)
    
    fig, ax = plt.subplots(figsize=(18, 16))
    plt.bar(labels, values)

    title = "KNN-Consistency"
    for key, item in param_dict.items():
        title += " - "+key+" "+str(item)

    plt.ylabel("Consistency")
    plt.title(title)
    #plt.xticks(rotation=45, ha="right")
    step = 5
    ax.set_xticks(np.arange(0, len(labels), step))
    ax.set_xticklabels(labels[::step], rotation=45, fontsize=8)
    plt.grid(axis="y", alpha=0.3)
    plt.ylim(0,1)

    plt.tight_layout()
    fig.savefig(results_path+"/"+title)
    #plt.show()
    

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
                knn_metric_path = results_path+"/knn_consistency_"+str(hops[i])+"_"+str(n_fft[j])+"_"+str(n_mels[k])+".json"
                param_dict = {
                    "hops": hops[i]
                    , "n_fft": n_fft[j]
                    , "n_mels": n_mels[k]
                }
                histogram(knn_metric_path, param_dict, model_path)