import os
import pandas as pd
import re

def seconds_to_hhmmss(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

DIR = "./data/synthetic_train_soundscapes"
WINDOW = 5
DURATION = 5  # soundscapes da 1 minuto

rows = []

for fname in os.listdir(DIR):
    if not fname.endswith(".ogg"):
        continue

    name = fname.replace(".ogg", "")[42:]

    # specie = parte dopo prefisso fisso (split sull'ultimo "_")

    match = re.search(r"((?:_[a-zA-Z0-9]{3,})+)$", name)

    try:
        species_part = match.group(1).lstrip("_")
        species_list = species_part.split("_")
    except:
        species_list = []
    

    species_list_string = ""
    for specie in species_list:
        species_list_string = species_list_string+specie+";"
    species_list_string = species_list_string[:-1]

    # finestre 5s
    for start in range(0, DURATION, WINDOW):
        end = start + WINDOW

        rows.append({
            "filename": fname,
            "start": seconds_to_hhmmss(start),
            "end": seconds_to_hhmmss(end),
            "primary_label": species_list_string
        })

df = pd.DataFrame(rows)
df.to_csv("biggest_synthetic_labels_yet.csv", index=False)
#df.to_csv("synthetic_validation_labels.csv", index=False)