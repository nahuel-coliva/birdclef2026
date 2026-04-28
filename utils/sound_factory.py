import os
import subprocess
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# background_pool = lista di np.array (audio 5s)
# bird_mix = np.array già mixato da te (stessa lunghezza dei background)

def random_gain(min_gain=0.3, max_gain=1.0):
    return np.random.uniform(min_gain, max_gain)

def normalize_audio(x):
    max_val = np.max(np.abs(x))
    if max_val > 0:
        x = x / max_val
    return x

def create_and_save_sample(background_pool, bird_pool, sr=32000, out_path="sample.wav", process_ID=0):
    background_filenames_pool = background_pool["filepath"].unique()
    bird_labels_pool = bird_pool["label"].unique()

    # 1️⃣ scegli 2 background random
    background_row_1 = background_pool[background_pool["filepath"]==random.choice(background_filenames_pool)].sample(1).iloc[0]
    background_row_2 = background_pool[background_pool["filepath"]==random.choice(background_filenames_pool)].sample(1).iloc[0]
    background_row_3 = background_pool[background_pool["filepath"]==random.choice(background_filenames_pool)].sample(1).iloc[0]


    bg1, _ = librosa.load(background_row_1["filepath"] , sr=sr)
    bg1 = bg1[background_row_1["start"]:background_row_1["start"]+sr*5]
    bg2, _ = librosa.load(background_row_2["filepath"] , sr=sr)
    bg2 = bg2[background_row_2["start"]:background_row_2["start"]+sr*5]
    bg3, _ = librosa.load(background_row_3["filepath"] , sr=sr)
    bg3 = bg3[background_row_3["start"]:background_row_3["start"]+sr*5]
    

    birds_number = random.choice([0, 1, 2, 3, 4])
    
    birds = []
    bird_label = list(set([random.choice(bird_labels_pool) for i in range(birds_number)]))
    birds_number = len(bird_label)
    bird_rows = [bird_pool[bird_pool["label"]==bird_label[i]].sample(1).iloc[0] for i in range(birds_number)]
    for bird_row in bird_rows:
        bird, _ = librosa.load(bird_row["filepath"] , sr=sr)
        bird = bird[bird_row["start"]:bird_row["start"]+sr*5]
        birds.append(bird)
    
    if birds_number==0:
        bird_mix = np.zeros(len(bg1))
    else:
        bird_mix = birds[0]*random_gain(0.5, 1.5)
        for i in range(1,birds_number):
            bird_mix += birds[i]*random_gain(0.5, 1.5)

    # 2️⃣ mix background + birds
    mix = (
        bg1 * random_gain(0.3, 1.0) +
        bg2 * random_gain(0.3, 1.0) +
        bg3 * random_gain(0.3, 1.0) +
        bird_mix
    )

    # normalizza per evitare clipping
    noise = np.random.normal(0, random_gain(0, 0.005), size=len(bg1))
    mix = normalize_audio(mix+noise)

    # 3️⃣ salva
    name = os.path.splitext(os.path.basename(background_row_1["filepath"]))[0]
    bg1_name = name.rsplit("_", 2)[0]

    name = os.path.splitext(os.path.basename(background_row_2["filepath"]))[0]
    bg2_name = name.rsplit("_", 2)[0]

    name = os.path.splitext(os.path.basename(background_row_2["filepath"]))[0]
    bg3_name = name.rsplit("_", 2)[0]

    filename = bg1_name+"_"+str(int(background_row_1["start"]/sr)).zfill(2)+"_"+bg2_name+"_"+str(int(background_row_2["start"]/sr)).zfill(2)+"_"+bg3_name+"_"+str(int(background_row_3["start"]/sr)).zfill(2)
    for bird_row in bird_rows:
        filename += "_"+os.path.basename(os.path.dirname(bird_row["filepath"]))
    filename += ".ogg"

    out_path = os.path.join(out_path, filename)
    #print("Salvo in "+out_path)
    save_as_ogg(out_path=out_path, mix=mix, sr=sr, i=process_ID)

    """
    print("Creato synthetic audio")
    print("Background 1: "+background_row_1["filepath"]+" from "+str(background_row_1["start"]/sr))
    print("Background 2: "+background_row_2["filepath"]+" from "+str(background_row_2["start"]/sr))
    print("Birds: "+str(birds_number))
    for i in range(len(birds)):
        print("Bird "+str(i)+" : "+bird_rows[i]["filepath"]+" from "+str(bird_rows[i]["start"]/sr))
    """

    return mix

def create_and_save_background_sample(background_pool, sr=32000, out_path="sample.wav"):
    #
    # Analogo a create_and_save_sample, ma SENZA uccelli: l'idea è aumentare il numero di sample disponibili con registrazioni di insetti
    #
    background_filenames_pool = background_pool["filepath"].unique()

    # 1️⃣ scegli 2 background random
    background_row_1 = background_pool[background_pool["filepath"]==random.choice(background_filenames_pool)].sample(1).iloc[0]
    background_row_2 = background_pool[background_pool["filepath"]==random.choice(background_filenames_pool)].sample(1).iloc[0]

    bg1, _ = librosa.load(background_row_1["filepath"] , sr=sr)
    bg1 = bg1[background_row_1["start"]:background_row_1["start"]+sr*5]
    bg2, _ = librosa.load(background_row_2["filepath"] , sr=sr)
    bg2 = bg2[background_row_2["start"]:background_row_2["start"]+sr*5]

    # 2️⃣ mix background
    mix = (
        bg1 * random_gain(0.3, 1.0) +
        bg2 * random_gain(0.3, 1.0)
    )

    # normalizza per evitare clipping
    noise = np.random.normal(0, random_gain(0, 0.005), size=len(bg1))
    mix = normalize_audio(mix+noise)

    # 3️⃣ salva
    name = os.path.splitext(os.path.basename(background_row_1["filepath"]))[0]
    bg1_name = name.rsplit("_", 2)[0]

    name = os.path.splitext(os.path.basename(background_row_2["filepath"]))[0]
    bg2_name = name.rsplit("_", 2)[0]

    filename = bg1_name+"_"+str(int(background_row_1["start"]/sr))+"_"+bg2_name+"_"+str(int(background_row_2["start"]/sr))

    for label in background_row_1["label"]:
        filename += "_"+label

    filename += ".ogg"

    out_path = os.path.join(out_path, filename)
    #print("Salvo in "+out_path)
    save_as_ogg(out_path=out_path, mix=mix, sr=sr)

    """
    print("Creato synthetic audio")
    print("Background 1: "+background_row_1["filepath"]+" from "+str(background_row_1["start"]/sr))
    print("Background 2: "+background_row_2["filepath"]+" from "+str(background_row_2["start"]/sr))
    """

    return mix

def save_as_ogg(mix, sr, out_path="sample.ogg", i=0):
    tmp_wav = "temp_"+str(i)+".wav"

    # 1️⃣ salva WAV
    sf.write(tmp_wav, mix.astype("float32"), sr)

    # 2️⃣ converti in OGG Vorbis (come BirdCLEF)
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", tmp_wav,
            "-acodec", "libvorbis",
            "-qscale:a", "5",
            out_path
        ],
        capture_output=True,
        text=True
    )

    #print("STDOUT:\n", result.stdout)
    #print("STDERR:\n", result.stderr)

    # 3️⃣ cleanup
    os.remove(tmp_wav)

def build_samples_audio_df(root_dir, df, sr=32000, label_in_path_flag=False):
    #
    # Restituisce pd.DataFrame; se label_in_path_flag = True assume sia un df di sample con 0/1 sample ognuno
    #
    rows = []

    for _, row in df.iterrows():
        start_time = row["start"]
        start_sample = int(start_time * sr)
        
        if label_in_path_flag:
            labels = row["primary_label"]
            filepath = os.path.join(root_dir, labels, row["filename"])
        else:
            if str(row["primary_label"])=="nan":
                labels = []
            else:
                labels = row["primary_label"].split(";")
            filepath = os.path.join(root_dir, row["filename"])

        rows.append({
            "filepath": filepath,
            "start": start_sample,
            "label": labels
        })

    return pd.DataFrame(rows)

def seconds_to_hhmmss(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def generate_train_audio_df(root_dir):
    rows = []

    j = 0
    for species in os.listdir(root_dir):
        j += 1
        print(str(j)+"/"+str(len(os.listdir(root_dir)))+" directories processed")
        species_path = os.path.join(root_dir, species)

        if not os.path.isdir(species_path):
            continue

        for file in os.listdir(species_path):
            if not file.endswith(".ogg"):
                continue

            filepath = os.path.join(species_path, file)

            # durata audio
            duration = librosa.get_duration(path=filepath)

            # numero finestre da 5s
            num_chunks = int(duration // 5)

            for i in range(num_chunks):
                start = i * 5
                end = start + 5

                rows.append({
                    "filename": file,
                    "start": seconds_to_hhmmss(start),
                    "end": seconds_to_hhmmss(end),
                    "primary_label": species
                })

    return pd.DataFrame(rows)


sr = 32000
root_dir="./data/train_soundscapes"
train_validation = "validation"
print("Sound factory for: "+train_validation)

# Trovare tutte le finestre di soundscapes senza specie
df = pd.read_csv("./data/train_soundscapes_labels_OG.csv").drop_duplicates()
df["start"] = pd.to_timedelta(df["start"]).dt.total_seconds()
df["end"] = pd.to_timedelta(df["end"]).dt.total_seconds()

# Sezione "background senza labels"

no_labels_df = pd.DataFrame()
for filename in sorted(df["filename"].unique()):
    for start in range(0,60,5):
        
        if start not in list(df[df["filename"] == filename]["start"]):
            #print("Missing "+str(start))
            #new_record = pd.DataFrame.from_dict({"filename": [filename], "start": [str(start)], "end": [str(start+5)], "primary_label": [""]})
            
            #Prossima istruzione per soundscape_with_background_flag=False
            new_record = pd.DataFrame.from_dict({"filename": [filename], "start": [start], "end": [start+5], "primary_label": [""]})
            df = pd.concat([df, new_record], ignore_index=True)
            no_labels_df = pd.concat([no_labels_df, new_record], ignore_index=True)

"""
# Sezione "insetti"
insect_list = []
for _, row in df.iterrows():
    for label in row["primary_label"].split(";"):
        if label[:8]=="47158son":
            insect_list.append(row)
            break
insects_df = pd.DataFrame(insect_list)
"""

soundscape_with_background_flag = False
if soundscape_with_background_flag:
    # Conversion back to HH:MM:SS
    for _, record in df.iterrows():
        df.loc[(df["filename"]==record["filename"]) & (df["start"]==record["start"]), "start"] = seconds_to_hhmmss(int(record["start"]))
        df.loc[(df["filename"]==record["filename"]) & (df["end"]==record["end"]), "end"] = seconds_to_hhmmss(int(record["end"]))
        df.sort_values(by=["filename", "start"], inplace=True)

    print(df.shape)
    df.to_csv("./data/train_soundscapes_v2.csv", index=False)
    input("Fermooo")


# Sezione "background senza labels": teniamo BC2026_Train_0006_S09_20250828_000000.ogg per il validation set
if train_validation=="train":
    no_labels_df = no_labels_df[no_labels_df["filename"]!="BC2026_Train_0006_S09_20250828_000000.ogg"]
else:
    no_labels_df = no_labels_df[no_labels_df["filename"]=="BC2026_Train_0006_S09_20250828_000000.ogg"]
background_samples = build_samples_audio_df(root_dir=root_dir, df=no_labels_df)


# Sezione "insetti"
"""
background_samples = build_samples_audio_df(root_dir=root_dir, df=insects_df)
"""

print(no_labels_df.shape[0])
print("Background OK")

#
# BIRDS
#
birds_flag = True #whether we need or not bird samples
if birds_flag:
    """
    # In case ./data/train_audio_windows.csv was not available
    root_dir = "./data/train_audio"
    birds_df = generate_train_audio_df(root_dir=root_dir)

    print("Birds df OK")

    # salva csv
    birds_df.to_csv("./data/train_audio_windows.csv", index=False)

    # Generate train/validation split for single mono-bird audio
    birds_df = pd.read_csv("./data/train_audio_windows.csv")
    birds_df["train_validation_split"] = "train"

    split_birds_df = pd.DataFrame()
    for bird in birds_df["primary_label"].unique():
        single_bird_df = birds_df[birds_df["primary_label"]==bird].copy()
        i = 1
        validation_files = []
        for file in single_bird_df["filename"].unique():
            if i%5==0:
                validation_files.append(file)
            i += 1
        
        validation_mask = single_bird_df["filename"].isin(validation_files)
        single_bird_df.loc[validation_mask, "train_validation_split"] = "validation"

        split_birds_df = pd.concat([split_birds_df, single_bird_df], ignore_index=True)

    split_birds_df.to_csv("./data/train_validation_split_for_birds.csv", index=False)

    for bird in split_birds_df["primary_label"].unique():
        print("Bird: "+bird)
        print("Train samples: "+str(birds_df[(birds_df["primary_label"]==bird) & (birds_df["train_validation_split"]=="train")].shape[0])+ " / Validation samples: "+str(birds_df[(birds_df["primary_label"]==bird) & (birds_df["train_validation_split"]=="validation")].shape[0]))
        print()
    input(":)")

    """
    root_dir = "./data/train_audio"
    
    birds_df = pd.read_csv("./data/train_validation_split_for_birds.csv")

    birds_df = birds_df[birds_df["train_validation_split"]==train_validation]
    birds_df["start"] = pd.to_timedelta(birds_df["start"]).dt.total_seconds()
    birds_df["end"] = pd.to_timedelta(birds_df["end"]).dt.total_seconds()

    bird_samples = build_samples_audio_df(root_dir=root_dir, df=birds_df, label_in_path_flag=True)

    print(birds_df.shape[0])
    print("Birds samples OK")
else:
    print("No birds requested")

results_path = "./data/synthetic_"+train_validation+"_soundscapes"
#results_path = "./data/test_insect_audio"

Path(results_path).mkdir(parents=True, exist_ok=True)

# Sample count by bird
"""
for bird in bird_samples["label"].unique():
    print(bird+": "+str(bird_samples[bird_samples["label"]==bird].shape[0]))
"""

print("Destination folder: "+results_path)

n = 25000
process_ID = input("process_ID: ")

for i in range(n):
    if i%1000==0:
        print(str(i)+"/"+str(n))
    if birds_flag:
        create_and_save_sample(background_pool=background_samples, bird_pool=bird_samples, out_path=results_path, process_ID=process_ID)
    else:
        create_and_save_background_sample(background_pool=background_samples, out_path=results_path)
