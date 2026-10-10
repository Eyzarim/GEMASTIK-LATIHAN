# 23 · Teks bahasa Indonesia: pembersihan, stemming, TF-IDF

**Kapan:** data teks Indonesia (ulasan, tweet, berita). **Kata kunci:** teks, NLP, Indonesia, stopword, stemming, Sastrawi, TF-IDF, normalisasi, slang, sentimen.

```python
import re, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.model_selection import cross_val_score, StratifiedKFold
from superdb.text import clean_text, ID_STOPWORDS            # pembersih ringan + stopword ringkas (negasi TIDAK dibuang)

df = pd.read_csv("data_latihan/text/sentiment/smsa_train.csv").sample(2000, random_state=0)
y = df["label"]

# 1. Pembersihan ringan: huruf kecil, buang URL/@mention, ringkas huruf berulang ("baguuus" -> "baguus")
print(clean_text("Mantaaap!!! cek https://t.co/abc @budi harganya 50rb"))

# 2. Normalisasi slang (kamus kecil; perluas dengan kamus Anda sendiri)
SLANG = {"gak": "tidak", "ga": "tidak", "nggak": "tidak", "tdk": "tidak", "yg": "yang", "dgn": "dengan", "bgt": "banget", "udh": "sudah", "krn": "karena", "bgs": "bagus"}
norm = lambda s: " ".join(SLANG.get(w, w) for w in clean_text(s).split())
print(norm("gak bgs bgt yg ini"))

# 3. Stemming Sastrawi (lambat; terapkan hanya bila terbukti membantu di CV) -- cache per kata agar cepat
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
stemmer = StemmerFactory().create_stemmer(); cache = {}
def stem(s): return " ".join(cache.setdefault(w, stemmer.stem(w)) for w in s.split())
print(stem("pengiriman barangnya sangat memuaskan dan penjualnya ramah"))

# 4. Bandingkan praproses lewat CV (jangan menebak!)
cv = StratifiedKFold(3, shuffle=True, random_state=0)
def skor(texts, vec):
    return cross_val_score(make_pipeline(vec, LogisticRegression(C=5, max_iter=2000)), texts, y, cv=cv, scoring="f1_macro").mean()
word = lambda **k: TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, **k)
char = lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
hasil = {
    "mentah, kata": skor(df["text"], word()),
    "bersih, kata": skor(df["text"].map(clean_text), word()),
    "bersih+slang, kata": skor(df["text"].map(norm), word()),
    "bersih, kata+stopword": skor(df["text"].map(clean_text), word(stop_words=ID_STOPWORDS)),
    "bersih, karakter (tahan typo)": skor(df["text"].map(clean_text), char()),
    "bersih, kata+karakter": skor(df["text"].map(clean_text), make_union(word(), char())),
    "stem Sastrawi, kata": skor(df["text"].map(clean_text).head(len(df)).map(stem), word()),
}
print(pd.Series(hasil).round(4).sort_values(ascending=False).to_frame("F1 macro CV"))
```

## Jebakan
- **Jangan membuang kata negasi** (`tidak`, `bukan`, `belum`, `jangan`, `tanpa`) pada sentimen: stopword list bawaan library sering membuangnya.
- Stemming sering **tidak** menaikkan skor pada TF-IDF + model linear dan lambat; ukur dulu. N-gram karakter menangani imbuhan dan typo tanpa stemming.
- Emoji dan tanda seru membawa sentimen: pertimbangkan menyimpannya (`clean_text` tidak membuangnya) atau menjadikannya fitur terpisah.
- Teks pendek/slang berat: model transformer Indonesia (IndoBERT) sering jauh lebih baik: `notebooks/text/sentiment/finetune_indobert`.
- Semua metode klasik dan transformer lengkap: `notebooks/text/`.
