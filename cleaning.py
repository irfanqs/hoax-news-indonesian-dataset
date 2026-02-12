import re
import os
import sys
import argparse
import pandas as pd

def load_alay_dict(kamus_path="kamus-alay/colloquial-indonesian-lexicon.csv"):
    if not os.path.exists(kamus_path):
        print(f"[WARNING] Kamus alay tidak ditemukan di: {kamus_path}")
        print("          Normalisasi slang akan dilewati.")
        return {}

    df_kamus = pd.read_csv(kamus_path)
    alay_map = {}
    for _, row in df_kamus.iterrows():
        slang = str(row['slang']).strip().lower()
        formal = str(row['formal']).strip().lower()
        if slang and formal and slang != 'nan' and formal != 'nan':
            if slang not in alay_map:
                alay_map[slang] = formal

    print(f"[INFO] Loaded {len(alay_map)} entries dari kamus alay.")
    return alay_map

# Mapping angka
LEET_DICT = {
    '0': 'o', '1': 'i', '3': 'e', '4': 'a',
    '5': 's', '6': 'g', '7': 't', '8': 'b',
    '9': 'g'
}


def normalize_leet_speak(text):
    def replace_in_word(word):
        has_letter = any(c.isalpha() for c in word)
        has_digit = any(c.isdigit() for c in word)
        if has_letter and has_digit:
            return ''.join(LEET_DICT.get(c, c) for c in word)
        return word

    return ' '.join(replace_in_word(w) for w in text.split())


def normalize_slang(text, alay_map):
    if not alay_map:
        return text
    words = text.split()
    normalized = []
    for word in words:
        normalized.append(alay_map.get(word, word))
    return ' '.join(normalized)


def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"  # misc
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"
        "\u3030"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)


def remove_urls(text):
    return re.sub(r'http\S+|www\.\S+', '', text)


def remove_mentions_hashtags(text):
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    return text


def remove_html_tags(text):
    return re.sub(r'<[^>]+>', '', text)


def normalize_repeated_chars(text):
    return re.sub(r'(.)\1{2,}', r'\1\1', text)


def normalize_punctuation(text):
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[.]{3,}', '.', text)
    text = re.sub(r'[,]{2,}', ',', text)
    return text


def remove_special_chars(text):
    return re.sub(r'[^a-z0-9\s.,!?\-]', '', text)


def remove_extra_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()


def remove_standalone_numbers(text):
    return re.sub(r'\b\d+\b', '', text)


def remove_emoticons(text):
    emoticon_pattern = r'[:;=8][\-~]?[)(DPpOo3><\|/\\]|[)(DPp][\-~]?[:;=8]|\^\^|<3|:v|:V|:\'\(|;\)|XD|xD|-_-|T_T|>_<|o_O|O_o|\*_\*'
    return re.sub(emoticon_pattern, '', text)


def clean_text(text, alay_map):
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = remove_urls(text)
    text = remove_mentions_hashtags(text)
    text = remove_html_tags(text)
    text = remove_emojis(text)
    text = remove_emoticons(text)
    text = normalize_leet_speak(text)
    text = normalize_slang(text, alay_map)
    text = normalize_repeated_chars(text)
    text = normalize_punctuation(text)
    text = remove_special_chars(text)
    text = remove_extra_spaces(text)

    return text


TEXT_COL = "Pesan"
LABEL_COL = "Klasifikasi"
KAMUS_PATH = "colloquial-indonesian-lexicon.csv"
MIN_LENGTH = 10
PREVIEW = 5


def main():
    parser = argparse.ArgumentParser(
        description="Text Cleaning Pipeline untuk Dataset Hoax (Bahasa Indonesia)"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="dataset.csv",
        help="Path ke file CSV input (default: dataset.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path ke file CSV output (default: sama dengan input, overwrite)"
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = "data_clean.csv"

    print("TEXT CLEANING PIPELINE - HOAX DETECTION")
    print(f"\n[1/5] Loading dataset: {args.input}")

    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"[ERROR] File tidak ditemukan: {args.input}")
        sys.exit(1)

    print(f"       Jumlah baris   : {len(df)}")
    print(f"       Kolom          : {list(df.columns)}")

    if TEXT_COL not in df.columns:
        print(f"\n[ERROR] Kolom '{TEXT_COL}' tidak ditemukan!")
        print(f"        Kolom yang tersedia: {list(df.columns)}")
        sys.exit(1)

    if LABEL_COL in df.columns:
        label_filled = df[LABEL_COL].notna().sum()
        label_empty = df[LABEL_COL].isna().sum()
        print(f"\n       Distribusi label ({LABEL_COL}):")
        for label, count in df[LABEL_COL].value_counts().items():
            pct = count / len(df) * 100
            print(f"         {label}: {count} ({pct:.1f}%)")
        if label_empty > 0:
            print(f"         (kosong): {label_empty} ({label_empty / len(df) * 100:.1f}%)")

    print(f"\n[2/5] Loading kamus alay: {KAMUS_PATH}")
    alay_map = load_alay_dict(KAMUS_PATH)

    print(f"\n[3/5] Preview data SEBELUM cleaning:")
    print("-" * 60)
    for i, text in enumerate(df[TEXT_COL].head(PREVIEW)):
        preview = str(text)[:100] + ("..." if len(str(text)) > 100 else "")
        print(f"  [{i+1}] {preview}")

    print(f"\n[4/5] Menjalankan cleaning pipeline...")

    # Hapus baris yang kolom Pesan-nya NaN (tapi biarkan Klasifikasi kosong)
    df_before = len(df)
    df = df.dropna(subset=[TEXT_COL])
    if df_before - len(df) > 0:
        print(f"       Hapus {df_before - len(df)} baris Pesan kosong/NaN")

    df['__original'] = df[TEXT_COL].copy()

    def clean_if_needed(text):
        cleaned = clean_text(text, alay_map)
        return cleaned

    df['__cleaned'] = df[TEXT_COL].apply(clean_if_needed)
    already_clean = (df[TEXT_COL] == df['__cleaned']).sum()
    new_rows = len(df) - already_clean
    print(f"       Row sudah bersih: {already_clean} (skip)")
    print(f"       Row baru/kotor  : {new_rows} (diproses)")

    df[TEXT_COL] = df['__cleaned']
    df = df.drop(columns=['__cleaned'])

    df_after_clean = len(df)
    df = df[df[TEXT_COL].str.len() >= MIN_LENGTH]
    removed = df_after_clean - len(df)
    if removed > 0:
        print(f"       Hapus {removed} baris terlalu pendek (< {MIN_LENGTH} karakter)")

    df_before_dedup = len(df)
    df = df.drop_duplicates(subset=[TEXT_COL])
    removed_dup = df_before_dedup - len(df)
    if removed_dup > 0:
        print(f"       Hapus {removed_dup} baris duplikat")

    print(f"       Sisa data      : {len(df)} baris")

    print(f"\n       Preview data SESUDAH cleaning:")
    print("-" * 60)
    for i, (_, row) in enumerate(df.head(PREVIEW).iterrows()):
        original = str(row['__original'])[:80] + ("..." if len(str(row['__original'])) > 80 else "")
        cleaned = str(row[TEXT_COL])[:80] + ("..." if len(str(row[TEXT_COL])) > 80 else "")
        print(f"  [{i+1}] ASLI  : {original}")
        print(f"       BERSIH: {cleaned}")
        print()

    df = df.drop(columns=['__original'])
    # Hanya simpan kolom Pesan dan Klasifikasi
    output_cols = [TEXT_COL]
    if LABEL_COL in df.columns:
        output_cols.append(LABEL_COL)
    df_out = df[output_cols]
    print(f"[5/5] Menyimpan hasil ke: {args.output}")
    df_out.to_csv(args.output, index=False)

    print(f"  Input          : {args.input}")
    print(f"  Output         : {args.output}")
    print(f"  Data awal      : {df_before} baris")
    print(f"  Data akhir     : {len(df)} baris")
    print(f"  Data dihapus   : {df_before - len(df)} baris")
    avg_len = df[TEXT_COL].str.len().mean()
    print(f"  Rata-rata len  : {avg_len:.0f} karakter")

    if LABEL_COL in df.columns:
        print(f"\n  Distribusi label akhir ({LABEL_COL}):")
        for label, count in df[LABEL_COL].value_counts().items():
            pct = count / len(df) * 100
            print(f"    {label}: {count} ({pct:.1f}%)")
        label_empty = df[LABEL_COL].isna().sum()
        if label_empty > 0:
            print(f"    (kosong): {label_empty} ({label_empty / len(df) * 100:.1f}%)")

    print(f"\n Done")


if __name__ == "__main__":
    main()
