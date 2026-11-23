# Analisis Ekspresi Gen untuk Klasifikasi Kanker (AML) dan Pengelompokan Sampel Berbasis Machine Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1U728Wu5K4OJ76cVq4XcTlicOZQzrtyjc?usp=sharing#scrollTo=JdJc_0Sw0SCn)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange)
![Domain](https://img.shields.io/badge/Domain-Bioinformatics%20%2F%20Transcriptomics-green)

Proyek **Data Sains dan Artificial Intelligence (AI)** yang mengombinasikan pendekatan *Supervised Learning*, *Unsupervised Learning*, dan analisis ekspresi gen diferensial untuk mengklasifikasikan sampel **Leukemia Mieloid Akut (*Acute Myeloid Leukemia*/AML)** dari sel hematopoietik normal serta mengevaluasi struktur pengelompokan alami pada dataset transkriptomik berdimensi tinggi (**NCBI GEO: GSE9476**).

---

## Tim Penyusun — Kelompok 7-B
**Program Studi Biologi Terapan, Fakultas Biologi, Universitas Jenderal Soedirman (2026)**

| Nama | NIM |
| :--- | :--- |
| **Natasya Lunna Putri** | `B1C024032` |
| **Kholiana Faoziah** | `B1C024033` |
| **Sinar Lintang Sejati** | `B1C024035` |
| **Cindy Aulia Eka Putri** | `B1C024059` |

---

## Daftar Isi
- [Latar Belakang & Tujuan](#latar-belakang--tujuan)
- [Deskripsi Dataset](#deskripsi-dataset)
- [Metodologi & Desain Eksperimen](#metodologi--desain-eksperimen)
- [Ringkasan Hasil Utama](#ringkasan-hasil-utama)
- [Keterbatasan & Saran Pengembangan](#keterbatasan--saran-pengembangan)
- [Cara Menjalankan Proyek](#cara-menjalankan-proyek)

---

## Latar Belakang & Tujuan

Leukemia Mieloid Akut (AML) merupakan keganasan hematologis akibat proliferasi klonal sel prekursor mieloid imatur (*blast*) di dalam sumsum tulang dan darah tepi. Data profil ekspresi gen *microarray* memiliki karakteristik dimensi sangat tinggi ($p \gg n$: puluhan ribu *probe* pada puluhan sampel), sehingga rentan terhadap *overfitting* dan kebocoran data (*data leakage*).

**Tujuan Penelitian:**
1. Membangun dan mengevaluasi model klasifikasi biner **AML vs Normal** dari data ekspresi gen.
2. Menguji keandalan estimasi performa model menggunakan *Stratified Cross-Validation*, *Nested CV*, dan uji permutasi label.
3. Menganalisis struktur alami data menggunakan metode *unsupervised learning* serta membandingkannya dengan label penyakit dan subtipe sel.
4. Mengidentifikasi *probe* penanda utama (*biomarker* kandidat) dan menafsirkannya secara biologis.

---

## Deskripsi Dataset

* **Sumber Data:** NCBI Gene Expression Omnibus (GEO) — **GSE9476** (Stirewalt *et al.*, 2008)
* **Platform Microarray:** *Affymetrix Human Genome U133A* (`GPL96`)
* **Dimensi Data:** 64 sampel × 22.283 *probe* → **22.215 *probe*** setelah membuang 68 *probe* kontrol internal (`AFFX-*`).
* **Karakteristik:** Seluruh fitur numerik (`float64`), tanpa nilai hilang (*missing values*), dengan rentang intensitas berskala $\log_2$ sebesar `0.773 – 15.733`.

| Kelompok Utama | Subtipe Sel | Jumlah Sampel | Penggunaan dalam Analisis |
| :--- | :--- | :---: | :--- |
| **AML** | Pasien AML | **19** | *Supervised* (kelas `1`) & *Unsupervised* |
| **Normal** | *Normal-BM* (Sumsum tulang mononuklear) | **10** | *Supervised* (kelas `0`) & *Unsupervised* |
| **Normal** | *Normal-CD34+* (Sel progenitor CD34+) | **18** | *Supervised* (kelas `0`) & *Unsupervised* |
| **Normal** | *Normal-PBSC* (Sel punca darah tepi) | **10** | *Supervised* (kelas `0`) & *Unsupervised* |
| **MDS** | *Myelodysplastic Syndrome* | **7** | Hanya *Unsupervised* (dikeluarkan dari *Supervised*) |
| **Total** | | **64** | |

---

## Metodologi & Desain Eksperimen

### 1. *Supervised Learning* (Klasifikasi Biner: AML vs Normal)
* **Pembagian Data:** 57 sampel (19 AML, 38 Normal) dibagi secara *stratified* (`random_state=42`) menjadi **80% data latih** (45 sampel) dan **20% data uji** (12 sampel: 4 AML, 8 Normal).
* **Pipeline Anti-*Data Leakage* (`scikit-learn`):**
  1. `SimpleImputer(strategy="median")`
  2. `VarianceThreshold(1e-6)`
  3. `StandardScaler()`
  4. `SelectKBest(f_classif)` (seleksi $k$ fitur terbaik berdasarkan ANOVA *F-value*)
* **Model & Validasi:**
  * Perbandingan 4 model pada *5-Fold Stratified CV* ($k=200$): `DummyClassifier` (*baseline*), *Logistic Regression L2* ($C=0.1$), *Linear SVM* ($C=0.1$), dan *Random Forest* (300 pohon).
  * Optimasi hiperparameter dengan `GridSearchCV` (metrik utama: **ROC-AUC**) pada $k \in \{50, 200, 1000\}$.
  * Uji ketahanan model dengan **Nested CV** (5-*fold outer*, 3-*fold inner*) dan **Uji Permutasi Label** (50 kali pengacakan).

### 2. *Unsupervised Learning* (Eksplorasi Struktur Alami)
* Menggunakan seluruh **64 sampel** tanpa label pada **2.000 *probe* dengan variansi tertinggi** yang telah distandarisasi.
* Metode: **PCA**, **K-Means Clustering** ($K=2..8$, dievaluasi dengan *Elbow* & *Silhouette Score*), dan **Hierarchical Clustering** (*Ward linkage*).
* Kesesuaian klaster dievaluasi dengan **Adjusted Rand Index (ARI)** dan **Normalized Mutual Information (NMI)** terhadap status penyakit dan subtipe sel.

### 3. Analisis Ekspresi Diferensial (*Biomarker Discovery*)
* Dilakukan khusus pada **data latih** menggunakan uji-*t* Welch per *probe* dengan koreksi *Benjamini–Hochberg FDR* ($q < 0.05$ dan $|\log_2\text{FC}| > 1$), bobot koefisien *Logistic Regression L2*, serta *loading* PC1.

---

## Ringkasan Hasil Utama

### A. Perbandingan Model (5-Fold Stratified CV pada Data Latih)
| Model | Akurasi | Balanced Accuracy | F1-Score (AML) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Linear SVM** | `0.978 ± 0.044` | `0.967 ± 0.067` | `0.960 ± 0.080` | **`0.989 ± 0.022`** |
| **Random Forest** | `0.978 ± 0.044` | `0.967 ± 0.067` | `0.960 ± 0.080` | **`0.989 ± 0.022`** |
| **Logistic Regression L2** | `0.978 ± 0.044` | `0.967 ± 0.067` | `0.960 ± 0.080` | `0.978 ± 0.044` |
| **Baseline (Dummy)** | `0.667 ± 0.000` | `0.500 ± 0.000` | `0.000 ± 0.000` | `0.500 ± 0.000` |

### B. Tuning, Validasi Ketahanan, dan Evaluasi Data Uji (*Linear SVM*: $C=0.001, k=50$)
| Tahap Evaluasi | ROC-AUC | Akurasi / Balanced Acc | F1-Score (AML) |
| :--- | :---: | :---: | :---: |
| **GridSearchCV (5-Fold Train)** | `0.989` | — | — |
| **Nested Cross-Validation (Train)** | `1.000 ± 0.000` | `0.933 ± 0.082` (*bal.*) | `0.920 ± 0.098` |
| **Uji Permutasi (50x acak)** | `0.503 ± 0.130` ($p = 0.020$) | — | — |
| **Evaluasi Data Uji (*Test Set*, $n=12$)** | **`1.000`** | **`0.920` (92%)** | **`0.860`** |

* Pada data uji ($n=12$), seluruh 8 sampel **Normal** terklasifikasi tepat (*precision* `0.89`, *recall* `1.00`), sedangkan 3 dari 4 sampel **AML** terdeteksi (*precision* `1.00`, *recall* `0.75`; 1 *false negative*).

### C. Pengelompokan Tanpa Arahan (*Unsupervised*) & Temuan Biologis
* **PCA:** PC1 menjelaskan **39,9%** variansi dan PC2 menjelaskan **12,1%** variansi (kumulatif 2 PC = **52,0%**; 10 PC = **79,9%**).
* **K-Means ($K=4$, *Silhouette* = `0.330`):**
  * Kesesuaian terhadap **status penyakit** sangat rendah (`ARI = 0.062`, `NMI = 0.310`), sedangkan kesesuaian terhadap **subtipe sel** jauh lebih tinggi (`ARI = 0.455`; *Ward linkage* `ARI = 0.432`).
  * Sampel sel matang (`Normal-BM` 10/10 dan `Normal-PBSC` 10/10) terpisah menjadi klaster murni tersendiri, sementara seluruh 18 sampel progenitor imatur (`Normal-CD34+`) mengelompok bersama 10 sampel `AML` dan 6 sampel `MDS`. Ini menunjukkan variasi transkriptomik terbesar lebih mencerminkan **tingkat kematangan sel** dibandingkan status keganasan semata.
* **Gen Penanda (*Biomarker*):** Ditemukan **442 *probe* diferensial signifikan** ($q < 0.05$, $|\log_2\text{FC}| > 1$), dengan **22 *probe* konsisten** berada di peringkat 50 teratas uji-*t* Welch dan bobot *Logistic Regression* (antara lain `205632_s_at`, `205726_at`, `218711_s_at`, `215489_x_at`, `202478_at`, dan `201105_at`).

---

## Keterbatasan & Saran Pengembangan

* **Ukuran *Test Set* Kecil ($n=12$):** Satu kesalahan klasifikasi mengubah *recall* kelas AML sebesar 25 poin; diperlukan validasi eksternal pada kohort AML lain dari GEO serta penyesuaian ambang keputusan (*decision threshold*) untuk meminimalkan *false negative* klinis.
* **Efek *Confounding* Komposisi Sel & *Batch*:** Perlu dilakukan pembandingan khusus antara `Normal-CD34+` vs `AML` serta koreksi efek *batch* (misalnya dengan *ComBat*) karena pemindaian microarray dilakukan pada rentang tahun 2002–2004.
* **Anotasi Gen & Multi-kelas:** Melengkapi pemetaan simbol gen resmi serta memperluas model menjadi klasifikasi tiga kelas (`AML` / `MDS` / `Normal`).

---

## Cara Menjalankan Proyek

1. **Melalui Google Colab:**
   Klik lencana **Open In Colab** di bagian atas dokumen ini, lalu pilih `Runtime > Run all`.
2. **Secara Lokal (Jupyter Notebook):**
   ```bash
   pip install numpy pandas scipy scikit-learn matplotlib seaborn
   jupyter notebook GSE9476_ML_Biologi.ipynb