import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "real_data"
UCI = "https://archive.ics.uci.edu/static/public"


def save(df, target, subdir, name, extra=()):
    # extra: kolumny zapisywane pod oryginalną nazwą (nie są cechami)
    y = df.pop(target)
    meta = df[list(extra)]
    df = df.drop(columns=list(extra))
    df.columns = [f"f{i}" for i in range(df.shape[1])]
    df[meta.columns] = meta
    df["target"] = y
    out_dir = DATA_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / f"{name}.parquet", index=False)
    print(f"Saved {subdir}/{name}.parquet: {df.shape[0]} samples, {df.shape[1] - 1 - len(extra)} features")


def main():
    zip_bytes = urllib.request.urlopen(f"{UCI}/279/susy.zip").read()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        susy = pd.read_csv(z.open("SUSY.csv.gz"), compression="gzip", header=None)
    susy[0] = susy[0].astype(int)
    save(susy, 0, "binary", "susy")

    # Statlog Shuttle (wieloklasowa): nagłówek w pliku jest uszkodzony, ostatnia kolumna = klasa
    shuttle = pd.read_csv(f"{UCI}/148/data.csv", header=None, skiprows=1)
    save(shuttle, shuttle.columns[-1], "multiclass", "shuttle")

    # Gas Turbine (regresja): cel NOX; rok i drugi cel CO zapisane jako osobne kolumny
    gas = pd.read_csv(f"{UCI}/551/data.csv")
    save(gas, "NOX", "regression", "gas_turbine", extra=["year", "CO"])


if __name__ == "__main__":
    main()
