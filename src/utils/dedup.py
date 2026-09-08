import os
import hashlib
import shutil
import pandas as pd
from PIL import Image

def calculate_md5(file_path, chunk_size=65536):
    """Menghitung hash MD5 dari file untuk deteksi byte-identical."""
    md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                md5.update(chunk)
        return md5.hexdigest()
    except Exception:
        return None

def calculate_dhash(image_path, hash_size=8):
    """
    Menghitung difference hash (dHash) secara mandiri (pure Pillow)
    tanpa dependensi eksternal tambahan.
    """
    try:
        with Image.open(image_path) as img:
            # Resize ke (hash_size + 1, hash_size) dan konversi ke grayscale
            img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = list(img.getdata())

            # Hitung perbedaan horizontal antar pixel
            diff = []
            for row in range(hash_size):
                for col in range(hash_size):
                    left = pixels[row * (hash_size + 1) + col]
                    right = pixels[row * (hash_size + 1) + col + 1]
                    diff.append(left > right)

            # Konversi array boolean ke hexadecimal string
            decimal_val = 0
            hex_str = []
            for index, value in enumerate(diff):
                if value:
                    decimal_val += 2**(index % 8)
                if (index % 8) == 7:
                    hex_str.append(hex(decimal_val)[2:].rjust(2, "0"))
                    decimal_val = 0
            return "".join(hex_str)
    except Exception:
        return None

def verify_image(file_path):
    """
    Verifikasi integritas file gambar:
    - Cek ukuran file > 0
    - Cek bisa dibuka dan didecode oleh PIL
    - Cek mode warna dan dimensi
    """
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return False, "file_empty_or_missing", None
        with Image.open(file_path) as img:
            img.verify()
        with Image.open(file_path) as img:
            img_format = img.format
            width, height = img.size
            return True, "valid", {"format": img_format, "width": width, "height": height}
    except Exception as e:
        return False, f"corrupt_{type(e).__name__}", None

def run_stage3_pipeline(posters_dir, df_metadata, quarantine_dir, output_metadata_path, log_audit_path):
    """
    Menjalankan proses Tahap 3:
    1. Validasi integritas gambar di folder posters_dir.
    2. Deteksi duplikasi 3 lapis (imdbId, MD5 hash, dHash visual).
    3. Pindahkan file rusak / duplikat ke quarantine_dir.
    4. Simpan metadata bersih ke output_metadata_path dan log ke log_audit_path.
    """
    os.makedirs(quarantine_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_metadata_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_audit_path), exist_ok=True)

    files = [f for f in os.listdir(posters_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Total file gambar ditemukan: {len(files)}")

    audit_logs = []
    valid_records = []

    # Trackers untuk deduplikasi
    seen_imdb_ids = set()
    seen_md5 = {}    # md5 -> file_name
    seen_dhash = {}  # dhash -> file_name

    # Buat lookup metadata film dari df_metadata jika tersedia
    meta_lookup = {}
    if df_metadata is not None and not df_metadata.empty:
        for _, row in df_metadata.iterrows():
            imdb_id = str(row["imdbId"])
            if imdb_id not in meta_lookup:
                meta_lookup[imdb_id] = row

    for f in files:
        file_path = os.path.join(posters_dir, f)
        imdb_id = os.path.splitext(f)[0]

        # 1. Verifikasi Integritas File
        is_valid, reason, info = verify_image(file_path)
        if not is_valid:
            dst = os.path.join(quarantine_dir, f)
            shutil.move(file_path, dst)
            audit_logs.append({
                "imdbId": imdb_id,
                "filename": f,
                "status": "quarantined",
                "reason": reason,
                "duplicate_of": None
            })
            continue

        # 2. Deduplikasi Lapis 1: imdbId
        if imdb_id in seen_imdb_ids:
            dst = os.path.join(quarantine_dir, f)
            shutil.move(file_path, dst)
            audit_logs.append({
                "imdbId": imdb_id,
                "filename": f,
                "status": "quarantined",
                "reason": "duplicate_imdbId",
                "duplicate_of": f"{imdb_id}.jpg"
            })
            continue

        # 3. Deduplikasi Lapis 2: Byte Hash (MD5)
        file_md5 = calculate_md5(file_path)
        if file_md5 and file_md5 in seen_md5:
            dst = os.path.join(quarantine_dir, f)
            shutil.move(file_path, dst)
            audit_logs.append({
                "imdbId": imdb_id,
                "filename": f,
                "status": "quarantined",
                "reason": "duplicate_md5_byte",
                "duplicate_of": seen_md5[file_md5]
            })
            continue

        # 4. Deduplikasi Lapis 3: Perceptual dHash (Visual)
        file_dhash = calculate_dhash(file_path)
        if file_dhash and file_dhash in seen_dhash:
            dst = os.path.join(quarantine_dir, f)
            shutil.move(file_path, dst)
            audit_logs.append({
                "imdbId": imdb_id,
                "filename": f,
                "status": "quarantined",
                "reason": "duplicate_visual_dhash",
                "duplicate_of": seen_dhash[file_dhash]
            })
            continue

        # Lolos semua saringan -> Data Valid
        seen_imdb_ids.add(imdb_id)
        if file_md5:
            seen_md5[file_md5] = f
        if file_dhash:
            seen_dhash[file_dhash] = f

        rec = {
            "imdbId": imdb_id,
            "filename": f,
            "file_path": file_path,
            "width": info["width"],
            "height": info["height"],
            "format": info["format"],
            "md5": file_md5,
            "dhash": file_dhash
        }

        # Merge info label jika ada
        if imdb_id in meta_lookup:
            row_data = meta_lookup[imdb_id]
            rec["Title"] = row_data.get("Title", "")
            rec["Genre"] = row_data.get("Genre", "")
            rec["target_genres"] = str(row_data.get("target_genres", []))

        valid_records.append(rec)

        audit_logs.append({
            "imdbId": imdb_id,
            "filename": f,
            "status": "valid",
            "reason": "passed_all_checks",
            "duplicate_of": None
        })

    df_valid = pd.DataFrame(valid_records)
    df_audit = pd.DataFrame(audit_logs)

    df_valid.to_csv(output_metadata_path, index=False)
    df_audit.to_csv(log_audit_path, index=False)

    print("\n=== RINGKASAN TAHAP 3 ===")
    print(f"Total file diproses      : {len(files)}")
    print(f"Total poster valid       : {len(valid_records)}")
    print(f"Total dikarantina        : {len(audit_logs) - len(valid_records)}")
    print(f"Log audit disimpan di    : {log_audit_path}")
    print(f"Metadata bersih di       : {output_metadata_path}")

    return df_valid, df_audit
