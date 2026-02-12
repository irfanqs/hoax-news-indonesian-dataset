# Hoax News Indonesian Dataset

Dataset teks berita hoax dan valid dalam Bahasa Indonesia, dilengkapi dengan pipeline pembersihan teks otomatis.

## Deskripsi

Repository ini berisi dataset klasifikasi berita hoax dalam Bahasa Indonesia. Dataset dikumpulkan dari berbagai sumber dan digabungkan dengan dataset [hoaxnews-ind-classification](https://huggingface.co/datasets/kornwtp/hoaxnews-ind-classfication) dari Hugging Face.

### Statistik Dataset

| Item                                      | Jumlah        |
| ----------------------------------------- | ------------- |
| Total data                                | 282 baris     |
| Label BENAR                               | 159 (56.4%)   |
| Label SALAH                               | 116 (41.1%)   |
| Tanpa label                               | 7 (2.5%)      |
| Rata-rata panjang teks (setelah cleaning) | 2154 karakter |

### Struktur Kolom

**dataset.csv** (data mentah):

| Kolom       | Deskripsi                                  |
| ----------- | ------------------------------------------ |
| Pesan       | Teks berita atau pesan                     |
| Claim       | Klaim yang disampaikan                     |
| Klasifikasi | Label: `SALAH` (hoax) atau `BENAR` (valid) |
| Link        | Sumber referensi                           |

**data_clean.csv** (data bersih):

| Kolom       | Deskripsi                                  |
| ----------- | ------------------------------------------ |
| Pesan       | Teks berita yang sudah dibersihkan         |
| Klasifikasi | Label: `SALAH` (hoax) atau `BENAR` (valid) |

## Cara Pemakaian

### 1. Persiapan

```bash
# Clone repository
git clone https://github.com/irfanqs/hoax-news-indonesian-dataset.git
cd hoax-news-indonesian-dataset

# Buat virtual environment (opsional)
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install dependensi
pip install pandas pyarrow
```

### 2. Menggunakan Dataset

#### Membaca data bersih (direkomendasikan):

```python
import pandas as pd

df = pd.read_csv("data_clean.csv")
print(df.head())
print(df["Klasifikasi"].value_counts())
```

#### Membaca data mentah:

```python
import pandas as pd

df = pd.read_csv("dataset.csv")
print(df.columns.tolist())
print(df.head())
```

### 3. Menjalankan Cleaning Pipeline

Script `cleaning.py` digunakan untuk membersihkan teks dari noise seperti:

- Huruf kapital (lowercase)
- URL, mention, hashtag
- Tag HTML
- Emoji dan emotikon
- Teks alay / slang Indonesia (menggunakan kamus alay)
- Karakter berulang (`halooo` -> `halo`)
- Tanda baca berlebih
- Karakter spesial

```bash
# Jalankan cleaning pada dataset.csv, output ke data_clean.csv
python cleaning.py -i dataset.csv

# Atau tentukan output sendiri
python cleaning.py -i dataset.csv -o output_saya.csv
```

#### Argumen CLI

| Argumen          | Deskripsi            | Default          |
| ---------------- | -------------------- | ---------------- |
| `-i`, `--input`  | Path file CSV input  | `dataset.csv`    |
| `-o`, `--output` | Path file CSV output | `data_clean.csv` |

### 4. Menambahkan Data Baru dari Hugging Face

```bash
pip install huggingface_hub

python -c "
from huggingface_hub import snapshot_download
snapshot_download('kornwtp/hoaxnews-ind-classfication', repo_type='dataset', local_dir='hf_hoax')
"
```

Kemudian merge ke dataset lokal:

```python
import pandas as pd

# Load dataset HF
df_hf = pd.read_parquet("hf_hoax/data/train-00000-of-00001.parquet")

# Mapping kolom dan label
label_map = {"Hoax": "SALAH", "Valid": "BENAR"}
df_hf = df_hf.rename(columns={"texts": "Pesan", "labels": "Klasifikasi"})
df_hf["Klasifikasi"] = df_hf["Klasifikasi"].map(label_map)
df_hf = df_hf[["Pesan", "Klasifikasi"]]

# Load dataset lokal dan merge
df_local = pd.read_csv("dataset.csv")
merged = pd.concat([df_local, df_hf], ignore_index=True)
merged = merged.drop_duplicates(subset=["Pesan"])
merged.to_csv("dataset.csv", index=False)

# Jalankan cleaning ulang
# python cleaning.py -i dataset.csv
```

## Struktur File

```
.
├── README.md
├── cleaning.py                          # Script pembersihan teks
├── colloquial-indonesian-lexicon.csv    # Kamus alay / slang Indonesia
├── dataset.csv                          # Data mentah
├── data_clean.csv                       # Data bersih (hasil cleaning)
└── .gitignore
```

## Pipeline Cleaning

Urutan proses pembersihan teks pada `cleaning.py`:

1. Lowercase
2. Hapus URL
3. Hapus mention dan hashtag
4. Hapus tag HTML
5. Hapus emoji
6. Hapus emotikon
7. Normalisasi leet speak (4l4y -> alay)
8. Normalisasi slang menggunakan kamus alay (4330 entri)
9. Normalisasi karakter berulang
10. Normalisasi tanda baca
11. Hapus karakter spesial
12. Hapus spasi berlebih

## Kamus Alay

File `colloquial-indonesian-lexicon.csv` berisi 4330 entri kosakata informal / alay Bahasa Indonesia beserta padanan formalnya. Contoh:

| Slang | Formal |
| ----- | ------ |
| gw    | saya   |
| lu    | kamu   |
| yg    | yang   |
| dg    | dengan |
| ank   | anak   |

## Sumber Data

- Koleksi manual dari berbagai platform media sosial
- [kornwtp/hoaxnews-ind-classfication](https://huggingface.co/datasets/kornwtp/hoaxnews-ind-classfication) (Hugging Face)
