"""
Download dataset UCI 'Predict Students Dropout and Academic Success' (ID 697).

Dataset berisi 4.424 mahasiswa dengan 36 fitur demografis, sosio-ekonomi,
dan performa akademik. Target: Dropout, Enrolled, Graduate.

Sumber: https://archive.ics.uci.edu/dataset/697
"""

from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

print("Mengunduh dataset dari UCI Machine Learning Repository...")
print("Dataset ID: 697 (Predict Students Dropout and Academic Success)")
print()

# Mengambil dataset resmi UCI dengan ID 697
dataset = fetch_ucirepo(id=697)

# Memisahkan fitur dan target, lalu menggabungkannya
X = dataset.data.features
y = dataset.data.targets

df = pd.concat([X, y], axis=1)

# Membuat folder data jika belum tersedia
Path("data").mkdir(exist_ok=True)

# Menyimpan dataset lokal
output_path = "data/student_data.csv"
df.to_csv(output_path, index=False)

print("=" * 55)
print("  DATASET BERHASIL DIUNDUH")
print("=" * 55)
print(f"  File     : {output_path}")
print(f"  Ukuran   : {df.shape[0]} baris × {df.shape[1]} kolom")
print(f"  Fitur    : {X.shape[1]} atribut")
print()
print("  Nama kolom:")
for i, col in enumerate(df.columns, 1):
    print(f"    {i:2d}. {col}")
print()
print("  Distribusi target:")
for label, count in df["Target"].value_counts().items():
    pct = count / len(df) * 100
    print(f"    {label:12s} : {count:5d}  ({pct:.1f}%)")
print()
print("  Lima data pertama:")
print(df.head().to_string())
print("=" * 55)
