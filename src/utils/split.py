import ast
from collections import Counter

import pandas as pd

GENRES = ["Action", "Adventure", "Animation", "Comedy", "Drama", "Horror", "Romance"]


def parse_genres(value):
    """Parse kolom target_genres (bisa string list, list, atau pipe-separated)."""
    if isinstance(value, list):
        return value
    value = str(value)
    if value.startswith("["):
        return ast.literal_eval(value)
    return [g for g in value.split("|") if g]


def make_splits(df, seed=42, train_frac=0.70, val_frac=0.15):
    """
    Split train/val/test terstratifikasi untuk data multi-label.

    Stratifikasi memakai genre paling langka (secara global) dari tiap baris,
    supaya genre minoritas (Romance, Horror) tetap terwakili merata di tiap split.
    """
    df = df.copy()
    df["genres"] = df["target_genres"].apply(parse_genres)
    counts = Counter(g for gl in df["genres"] for g in gl)
    df["strat"] = df["genres"].apply(lambda gl: min(gl, key=lambda g: (counts[g], g)))

    parts = []
    for _, grp in df.groupby("strat"):
        grp = grp.sample(frac=1.0, random_state=seed)
        n = len(grp)
        n_train = round(train_frac * n)
        n_val = round(val_frac * n)
        split = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
        parts.append(grp.assign(split=split))

    out = pd.concat(parts, ignore_index=True)
    return out[["imdbId", "filename", "strat", "split"]]


def verify_splits(splits, df_metadata):
    """
    Sanity check hasil split:
    - Tidak ada imdbId yang sama antar split (no leakage).
    - Semua genre target muncul di train.
    - Proporsi split mendekati target.
    Melempar AssertionError jika ada yang gagal.
    """
    by_split = {name: set(splits.loc[splits["split"] == name, "imdbId"])
                for name in ("train", "val", "test")}
    assert not (by_split["train"] & by_split["val"]), "Leakage: imdbId muncul di train & val"
    assert not (by_split["train"] & by_split["test"]), "Leakage: imdbId muncul di train & test"
    assert not (by_split["val"] & by_split["test"]), "Leakage: imdbId muncul di val & test"
    assert len(by_split["train"]) + len(by_split["val"]) + len(by_split["test"]) == len(splits)

    df = df_metadata.copy()
    df["genres"] = df["target_genres"].apply(parse_genres)
    df = df.merge(splits[["imdbId", "split"]], on="imdbId")
    train_genres = {g for gl in df.loc[df["split"] == "train", "genres"] for g in gl}
    missing = set(GENRES) - train_genres
    assert not missing, f"Genre tidak muncul di train: {missing}"

    frac = splits["split"].value_counts(normalize=True)
    assert 0.60 < frac["train"] < 0.80, f"Proporsi train janggal: {frac['train']:.2f}"
    assert 0.10 < frac["val"] < 0.20, f"Proporsi val janggal: {frac['val']:.2f}"
    assert 0.10 < frac["test"] < 0.20, f"Proporsi test janggal: {frac['test']:.2f}"

    return {
        "n_train": len(by_split["train"]),
        "n_val": len(by_split["val"]),
        "n_test": len(by_split["test"]),
    }
