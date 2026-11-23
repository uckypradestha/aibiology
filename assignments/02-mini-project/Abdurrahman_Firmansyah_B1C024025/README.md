# Klasifikasi Kultivar Anggur Berdasarkan Profil Kimia

**Mata Kuliah:** Data Sains dan AI  
**Kelompok:** 2 — Rombongan A  
**Fakultas:** Fakultas Biologi, Universitas Jenderal Soedirman  

## Anggota Kelompok

| Nama | NIM |
|---|---|
| Fatih Aiman | B1C024001 |
| Abdurrahman Firmansyah | B1C024025 |
| Winarno | B1C024030 |

## Dataset

Dataset yang digunakan adalah **Wine Dataset** dari `sklearn.datasets.load_wine` atau UCI Machine Learning Repository. Dataset ini terdiri dari 178 sampel anggur dari tiga kultivar berbeda yang dijelaskan menggunakan 13 fitur kimia numerik.

- **Jumlah sampel:** 178
- **Jumlah fitur:** 13
- **Jumlah kelas:** 3
- **Missing value:** 0
- **Duplikat data:** 0

**Google Colab Notebook:** [Buka Notebook](https://colab.research.google.com/drive/1M_GSsTGxHn2Y0oVSOgrAZt2fy0dNUDby?usp=sharing)

## Ringkasan Proyek

Proyek ini menerapkan pendekatan *Supervised Learning* dan *Unsupervised Learning* untuk mengklasifikasikan tiga kultivar anggur dari Italia berdasarkan profil senyawa kimianya. Selain melakukan klasifikasi, proyek ini juga bertujuan mengidentifikasi fitur kimia yang paling berpengaruh sebagai penanda biologis utama pembeda antar kultivar.

## Eksplorasi Data dan Preprocessing

### Karakteristik Dataset

Distribusi kelas pada dataset adalah sebagai berikut:

| Kelas | Jumlah Sampel |
|---|---:|
| `class_0` | 59 |
| `class_1` | 71 |
| `class_2` | 48 |

Ditemukan korelasi tinggi antara beberapa fitur, khususnya:

- `flavanoids` dan `total_phenols` dengan koefisien korelasi \(r = 0.86\)
- `flavanoids` dan `od280/od315_of_diluted_wines` dengan koefisien korelasi \(r = 0.79\)

Korelasi tinggi tersebut menunjukkan adanya multikolinearitas antar fitur, terutama pada kelompok senyawa fenolik.

### Preprocessing

Tahapan preprocessing yang dilakukan meliputi:

- Pembagian data secara *stratified* dengan proporsi **80:20**
- Data latih sebanyak **142 sampel**
- Data uji sebanyak **36 sampel**
- Penggunaan `random_state=42` agar hasil eksperimen dapat direproduksi
- Normalisasi fitur menggunakan `StandardScaler`
- Penerapan scaling di dalam `Pipeline` untuk mencegah *data leakage*

## Hasil Supervised Learning

Perbandingan model dilakukan menggunakan **5-Fold Stratified Cross-Validation** pada data latih.

| Model | F1-Macro (mean ± std) | Accuracy (mean ± std) |
|---|---:|---:|
| Logistic Regression | 0.980 ± 0.027 | 0.979 ± 0.028 |
| Random Forest | 0.979 ± 0.029 | 0.979 ± 0.028 |
| SVM (RBF) | 0.986 ± 0.017 | 0.986 ± 0.017 |

### Model Terbaik

**Random Forest** dipilih sebagai model utama untuk dilakukan *hyperparameter tuning* karena mampu memberikan informasi *feature importance* yang berguna untuk interpretasi biologis.

Hyperparameter terbaik hasil `GridSearchCV` adalah:

- `n_estimators = 100`
- `max_depth = None`
- `min_samples_split = 2`

Pada 36 sampel data uji, model Random Forest menghasilkan performa sebagai berikut:

| Metrik | Nilai |
|---|---:|
| Accuracy | 1.000 |
| F1-Macro | 1.000 |

Model berhasil mengklasifikasikan seluruh sampel data uji dengan sempurna tanpa indikasi *data leakage*.

## Hasil Unsupervised Learning

### Principal Component Analysis

PCA dengan dua komponen utama digunakan untuk mereduksi dimensi data dan memvisualisasikan pola pemisahan antar kultivar.

| Komponen | Proporsi Variansi |
|---|---:|
| PC1 | 36.20% |
| PC2 | 19.21% |
| Total | 55.41% |

Visualisasi PCA menunjukkan pemisahan tiga kultivar anggur yang cukup jelas.

### Clustering

Clustering dilakukan dengan jumlah klaster \(K = 3\), sesuai dengan jumlah kultivar pada dataset.

| Metode Clustering | Silhouette Score | Adjusted Rand Index (ARI) |
|---|---:|---:|
| K-Means | 0.2849 | 0.8975 |
| Hierarchical Clustering (Ward) | 0.2774 | 0.7899 |

K-Means menghasilkan nilai ARI yang lebih tinggi dibandingkan Hierarchical Clustering, yang menunjukkan kesesuaian klaster dengan label asli yang lebih baik.

## Interpretasi Biologis

Berdasarkan *feature importance* dari model Random Forest, lima fitur paling berpengaruh dalam membedakan kultivar anggur adalah:

| Fitur | Feature Importance |
|---|---:|
| `color_intensity` | 0.188 |
| `flavanoids` | 0.160 |
| `proline` | 0.147 |
| `alcohol` | 0.118 |
| `hue` | 0.102 |

Secara biologis, fitur-fitur tersebut merepresentasikan:

- **`color_intensity`**: intensitas warna anggur yang berkaitan dengan kandungan pigmen antosianin pada kulit buah
- **`flavanoids`**: kelompok senyawa fenolik yang berperan dalam karakteristik rasa, warna, dan aktivitas antioksidan
- **`proline`**: asam amino yang dapat menjadi penanda akumulasi metabolit spesifik pada varietas tertentu
- **`alcohol`**: kandungan alkohol yang berhubungan dengan proses fermentasi dan kematangan buah
- **`hue`**: rona warna anggur yang dipengaruhi oleh komposisi pigmen dan tingkat kematangan

## Keterbatasan

Beberapa keterbatasan dari proyek ini antara lain:

- Ukuran sampel relatif terbatas, yaitu hanya 178 sampel
- Seluruh sampel berasal dari satu wilayah di Italia
- Tidak tersedia label nama kultivar asli
- Tidak tersedia informasi kondisi lingkungan, lokasi penanaman, maupun tahun panen
- Terdapat multikolinearitas antar fitur, terutama pada kelompok senyawa fenolik
- Performa model yang sempurna pada data uji perlu diinterpretasikan dengan hati-hati karena dataset berukuran kecil dan relatif mudah dipisahkan
