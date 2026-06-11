"""Quick inspector for the SACap-annotated SA-1B-5P parquet.

Run:
    python stage3/data/inspect_sacap_parquet.py
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

PARQUET_PATH = "/home/b5cz/simonz.b5cz/program/stage2/efficientsam3/data/sa-1b-5p-sacap/anno.parquet"
IMG_ROOT = "/home/b5cz/simonz.b5cz/program/stage2/efficientsam3/data/SA-1B-5P"


def main() -> None:
    df = pd.read_parquet(PARQUET_PATH)
    print("Columns:", list(df.columns))
    print("Rows:", len(df))
    print("Dtypes:\n", df.dtypes)
    print("\nFirst row:")
    row = df.iloc[0].to_dict()
    for k, v in row.items():
        s = repr(v)
        if len(s) > 400:
            s = s[:400] + "...<truncated>"
        print(f"  {k}: {s}")

    print("\nSecond row keys/types:")
    row2 = df.iloc[1].to_dict()
    for k, v in row2.items():
        print(f"  {k}: type={type(v).__name__}")
        if isinstance(v, (list, tuple)) and len(v) > 0:
            print(f"    len={len(v)}, first={repr(v[0])[:300]}")
        if isinstance(v, dict):
            print(f"    keys={list(v.keys())[:10]}")

    # Try resolve images for first 5 rows
    print("\nImage path resolution (first 5):")
    for i in range(min(5, len(df))):
        r = df.iloc[i].to_dict()
        # try several likely columns
        candidate_keys = [
            k for k in r.keys() if any(t in k.lower() for t in ["image", "file", "path", "name", "img"])
        ]
        print(f"  row {i}: candidate keys={candidate_keys}")
        for ck in candidate_keys:
            print(f"    {ck} = {r[ck]!r}")

    # Distribution of subset/split info if present
    for col in df.columns:
        try:
            uniq = df[col].nunique(dropna=True)
            if 1 <= uniq <= 20 and df[col].dtype != object or (df[col].dtype == object and uniq <= 20):
                print(f"\nColumn {col} values ({uniq} unique):")
                print(df[col].value_counts(dropna=False).head(20))
        except Exception:
            pass


if __name__ == "__main__":
    main()
