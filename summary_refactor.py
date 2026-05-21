import re
import ast
import pandas as pd

rows = []

with open("./results/session_performance_test_02/summary.txt", "r") as f:
    lines = [line.strip() for line in f if line.strip()]

for i in range(0, len(lines), 3):
    params = ast.literal_eval(lines[i])

    intra = float(re.search(r"Intra:\s*([\d.]+)", lines[i + 1]).group(1))
    inter = float(re.search(r"Inter:\s*([\d.]+)", lines[i + 2]).group(1))

    rows.append({
        "hops": params["hops"],
        "n_fft": params["n_fft"],
        "n_mels": params["n_mels"],
        "intra": intra,
        "inter": inter,
    })

df = pd.DataFrame(rows)
df.to_csv("output.csv", index=False)

print(df)