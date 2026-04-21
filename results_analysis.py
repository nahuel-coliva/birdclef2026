import pandas as pd
import matplotlib.pyplot as plt

results_path = "./results/submission.csv"

prediction_df = pd.read_csv(results_path)

print(prediction_df.head())

truth_df = pd.read_csv("./data/validation_soundscapes_labels.csv").drop_duplicates()
truth_df["start"] = pd.to_timedelta(truth_df["start"]).dt.total_seconds().astype(int)
truth_df["end"] = pd.to_timedelta(truth_df["end"]).dt.total_seconds().astype(int)

truth_df["primary_label_split"] = truth_df["primary_label"].fillna("").str.split(";")
truth_df["temp"] = [i[0] for i in truth_df["filename"].str.split(".")]
truth_df["row_id"] = truth_df["temp"]+"_"+truth_df["end"].astype(str)
truth_df = truth_df.drop(columns=["temp"])

# Check whether the split was correct
for _, row in truth_df.iterrows():
    if len(row["primary_label_split"]) == 0:
        input(row)
    try:
        assert str(row["primary_label"].split(";")) == str(row["primary_label_split"])
    except Exception as e:
        print(row)
        print(e)
        break

for _, row in truth_df.iterrows():
    print(row["row_id"])
    prediction = prediction_df[prediction_df["row_id"] == row["row_id"]]
    print(prediction[row["primary_label_split"]])
    confident_labels = []
    for col in prediction.columns:
        if col=="row_id":
            continue
        if prediction[col].values[0]>0.8:
            confident_labels.append(col)
    print(prediction[confident_labels])

    plt.figure()
    # Istogramma
    plt.hist(prediction.values[0][1:], bins=100)

    plt.xlabel("Valori")
    plt.ylabel("Frequenza")
    plt.title("Istogramma dei valori della matrice")
    plt.show()