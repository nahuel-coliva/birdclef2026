import pandas as pd
import main
import torch
import numpy as np

sample_rate = 32000
root_dir_dict = {"real":"./data/train_soundscapes", "synthetic": "./data/synthetic_train_soundscapes"}
device = "cuda" if torch.cuda.is_available() else "cpu"
hop_length = 320
n_fft = hop_length*4
n_mels = 200

df_train = pd.read_csv("./data/bigger_train_soundscapes_labels.csv").drop_duplicates()
df_train["start"] = pd.to_timedelta(df_train["start"]).dt.total_seconds()
df_train["end"] = pd.to_timedelta(df_train["end"]).dt.total_seconds()

species_list_train = []
for item in df_train["primary_label"].str.split(";").explode().unique():
    if str(item)!="nan":
        species_list_train.append(item)
num_classes = len(species_list_train)
print("Train number of classes: "+str(num_classes))

species_to_idx = {s: i for i, s in enumerate(species_list_train)}

train_dataset = main.SoundscapeDataset(
    	root_dir=root_dir_dict,
    	df=df_train,
    	species_to_idx=species_to_idx,
    	sample_rate=sample_rate
    )

"""
pos_count = [0 for i in range(num_classes)]
pos_weights = [0 for i in range(num_classes)]
"""
pos_count = np.zeros(num_classes)
pos_weights = np.zeros(num_classes)
for _, _, labels in train_dataset.samples:
    for label in labels:
        if label in species_to_idx:
            pos_count[species_to_idx[label]] += 1


total_samples = len(train_dataset.samples)
for i in range(len(pos_count)):
    pos_weights[i] += 1e-6
    pos_weights[i] = total_samples/(pos_count[i]*len(pos_count))

pos_weights /= min(pos_weights)

for label, count, weight in zip(species_list_train, pos_count, pos_weights):
    print("Class "+label+" has count "+str(count)+" thus weigth "+str(weight))

print(pos_weights)

model = main.BirdModel(num_classes=num_classes,
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                trainable_pcen=True
            ).to(device)
optimizer = torch.optim.Adam([
    {"params": model.backbone.features.parameters(), "lr": 1e-5},
    {"params": model.backbone.classifier.parameters(), "lr": 1e-4},
    {"params": model.frontend.pcen.parameters(), "lr": 1e-5}
])
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(pos_weights)).to(device)  # preferibile a BCE(sigmoid) per stabilità

# Model complexity vs dataset size
trainable = main.count_trainable_params(model)
total = main.count_total_params(model)

print(f"Trainable: {trainable}")
print(f"Total: {total}")
print(f"Ratio (expected < 0.1 since MobileNet is a feature extractor): {trainable/total:.3f}")
print(f"Sample per param (expected 10-100): {df_train.shape[0]/trainable}")
input("Quindi? Come siam messi?")