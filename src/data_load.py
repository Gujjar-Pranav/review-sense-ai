import pandas as pd
from src.config import DATA_PATH

def load_dataset(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = df.drop_duplicates().copy()
    df["label"] = df["label"].map({"pos": 1, "neg": 0})
    df["review"] = df["review"].astype(str)
    df["label"] = df["label"].astype(int)
    return df
