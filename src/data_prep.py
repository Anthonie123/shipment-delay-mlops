import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
PROC_PATH = DATA_DIR / "processed.csv"

def main():
    # ambil semua file yang namanya diawali 'train' dan berakhiran '.csv'
    files = sorted(DATA_DIR.glob("train*.csv"))

    dfs = [pd.read_csv(path) for path in files]
    df = pd.concat(dfs, ignore_index=True)

    if "ID" in df.columns:
        df = df.drop("ID", axis=1)

    if "Customer_care_calls" in df.columns:
        df["Customer_care_calls"] = df["Customer_care_calls"].clip(lower=1, upper=5)

    X = df.drop("Reached.on.Time_Y.N", axis=1)
    y = df["Reached.on.Time_Y.N"]

    X_enc = pd.get_dummies(X)
    df_proc = X_enc.copy()
    df_proc["Reached.on.Time_Y.N"] = y

    df_proc.to_csv(PROC_PATH, index=False)

if __name__ == "__main__":
    main()
