import argparse
import ast
import os
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from PIL import Image

GENRES = ["Action", "Adventure", "Animation", "Comedy", "Drama", "Horror", "Romance"]
PRIORITY = {g: i for i, g in enumerate(
    ["Animation", "Action", "Horror", "Adventure", "Romance", "Comedy", "Drama"]
)}


def parse_genres(value):
    if isinstance(value, list):
        return value
    value = str(value)
    if value.startswith("["):
        return ast.literal_eval(value)
    return [g for g in value.split("|") if g in GENRES]


def download_one(row, out_dir, timeout):
    imdb_id = str(row.imdbId)
    dest = os.path.join(out_dir, f"{imdb_id}.jpg")
    if os.path.exists(dest):
        return imdb_id, "skip_exists"
    temp = dest + ".part"
    try:
        response = requests.get(row.Poster, timeout=timeout)
        response.raise_for_status()
        with open(temp, "wb") as f:
            f.write(response.content)
        with Image.open(temp) as image:
            image.verify()
        shutil.move(temp, dest)
        return imdb_id, "ok"
    except Exception as exc:
        if os.path.exists(temp):
            os.remove(temp)
        return imdb_id, f"error_{type(exc).__name__}"


def count_existing(df, out_dir):
    existing = {
        os.path.splitext(name)[0]
        for name in os.listdir(out_dir)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    }
    counts = Counter()
    for row in df[df.imdbId.isin(existing)].itertuples():
        counts.update(row.target_genres)
    return existing, counts


def run(csv_path, out_dir, log_path, target=1500, workers=16, timeout=15):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    df = pd.read_csv(csv_path, encoding="latin1")
    df = df.dropna(subset=["imdbId", "Genre", "Poster"]).copy()
    if "Poster_Status" in df:
        df = df[df["Poster_Status"].ne("not_found")]
    df["imdbId"] = df["imdbId"].astype(str)
    df["target_genres"] = df["Genre"].apply(parse_genres)
    df = df[df.target_genres.apply(bool)].drop_duplicates("imdbId")
    df["priority"] = df.target_genres.apply(lambda gs: min(PRIORITY[g] for g in gs))
    df = df.sort_values(["priority", "imdbId"]).reset_index(drop=True)

    failed = set()
    logs = []
    round_no = 0
    while True:
        existing, counts = count_existing(df, out_dir)
        missing = {g: max(0, target - counts[g]) for g in GENRES}
        print(f"round {round_no}: files={len(existing)} counts={dict(counts)} missing={missing}")
        if not any(missing.values()):
            break

        candidates = []
        planned = Counter(counts)
        for row in df.itertuples():
            if row.imdbId in existing or row.imdbId in failed:
                continue
            if not any(planned[g] < target for g in row.target_genres):
                continue
            candidates.append(row)
            planned.update(row.target_genres)
            if all(planned[g] >= target for g in GENRES):
                break
        if not candidates:
            break

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(download_one, row, out_dir, timeout) for row in candidates]
            for future in as_completed(futures):
                imdb_id, status = future.result()
                logs.append({"imdbId": imdb_id, "status": status, "round": round_no})
                if status.startswith("error_"):
                    failed.add(imdb_id)
        round_no += 1

    pd.DataFrame(logs).to_csv(log_path, index=False)
    existing, counts = count_existing(df, out_dir)
    print(f"final files={len(existing)} counts={dict(counts)}")
    print(f"log={log_path}")
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--target", type=int, default=1500)
    args = parser.parse_args()
    run(args.csv, args.out, args.log, target=args.target)
