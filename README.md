# Skripsi: Klasifikasi Multi-Label Genre Film dari Poster (CNN)

## Setup awal

```bash
git init
git add .
git commit -m "init: struktur project"
```

Buka folder ini di VSCode, lalu jalankan `claude` di terminal untuk mulai pakai
Claude Code CLI. Claude Code akan otomatis membaca `CLAUDE.md` untuk konteks project.

## Workflow harian

1. **Edit lokal** — buka/edit `.ipynb` di VSCode (extension Jupyter, Microsoft),
   dibantu Claude Code CLI untuk debugging/penulisan kode.
2. **Training di Colab** — upload notebook yang sudah difinalisasi ke
   [colab.research.google.com](https://colab.research.google.com), aktifkan GPU
   (Runtime → Change runtime type → GPU), lalu jalankan training.
3. **Sinkron balik** — setelah run selesai, download `.ipynb` dari Colab dan
   timpa file lokal di folder `notebooks/` supaya output tersimpan.
4. **Commit** — commit perubahan notebook dan `src/` ke git (data mentah & model
   besar sudah otomatis diabaikan lewat `.gitignore`).

## Struktur

```
data/          -> raw, processed, posters (tidak masuk git)
notebooks/     -> semua .ipynb
src/           -> kode Python reusable
models/        -> model terlatih
outputs/       -> figures & logs
docs/          -> draf bab skripsi
```

## Catatan
- Dataset poster tidak disertakan di repo (terlalu besar). Simpan di Google Drive
  dan mount saat training di Colab jika perlu, atau taruh lokal di `data/posters/`.
- Lihat `CLAUDE.md` untuk detail arsitektur model dan konvensi kode.
