# Klasifikasi Status Diabetes pada Perempuan Suku Pima Menggunakan Machine Learning dan Interpretasi Biologis Fitur Klinis

## 1. Identitas

| | |
|---|---|
| **Mata kuliah** | Data Science and Artificial Intelligence for Biology |
| **Institusi** | Fakultas Biologi, Universitas Jenderal Soedirman, Purwokerto (2026) |
| **Kelompok** | 9 |
| **Anggota** | Yoel Hadi Prasetyo (B1C024013), Widia Nur Astuti (B1C024016), Yulia Tasiyah (B1C024027) |
| **Dataset** | Pima Indians Diabetes Database, OpenML ([openml.org/d/37](https://www.openml.org/d/37)) |
| **Tools** | Python, Google Colab, pandas, numpy, matplotlib, seaborn, scikit-learn |
| **Kode** | `Notebook_Pima_Diabetes.ipynb` |

## 2. Latar Belakang

Diabetes melitus adalah penyakit metabolik dengan hiperglikemia akibat gangguan sekresi atau kerja insulin. Pada diabetes tipe 2 gejala awal sering ringan, sehingga penyakit baru terdeteksi setelah komplikasi muncul. **Deteksi dini** karena itu sangat penting. Faktor risikonya (glukosa, BMI, usia, riwayat kehamilan, riwayat keluarga) dicatat dalam pemeriksaan rutin, sehingga dapat dipakai untuk model skrining berbasis *machine learning* (ML) sebagai alat bantu, bukan pengganti diagnosis.

Dataset Pima memuat 768 perempuan keturunan suku Pima (usia ≥ 21 tahun), populasi dengan prevalensi diabetes tipe 2 sangat tinggi. Dataset ini menarik karena (1) ada **nilai nol yang mustahil secara biologis**, (2) **kelas tidak seimbang** (268 diabetes : 500 tidak diabetes) sehingga akurasi saja menyesatkan, dan (3) fitur klinisnya **mudah dijelaskan secara biologis**.

**Pertanyaan penelitian:**

1. Bagaimana kualitas data, termasuk nilai tidak wajar secara biologis?
2. Seberapa baik Logistic Regression, Random Forest, dan SVM dibandingkan baseline?
3. Fitur apa yang paling berpengaruh, dan apakah sesuai teori biologis?
4. Apakah pasien dapat dikelompokkan secara alami tanpa label (PCA dan K-Means)?

## 3. Metode

**Data.** 768 baris, 8 fitur numerik (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age), target Outcome (1 = diabetes).

**Preprocessing.**

- Nilai 0 pada Glucose, BloodPressure, SkinThickness, Insulin, dan BMI dianggap *missing* (NaN). Nilai 0 pada Pregnancies sah dan dibiarkan.
- Pembagian data berstrata: **80% latih (614)** dan **20% uji (154)**, `random_state=42`. Data uji hanya dipakai sekali di akhir.
- **Imputasi median** dan **StandardScaler** di dalam `Pipeline` agar tidak terjadi kebocoran data.
- Tidak dilakukan oversampling dan penghapusan outlier (ketidakseimbangan sedang; nilai ekstrem klinis bisa informatif).

**Eksperimen.**

- *Supervised:* baseline (selalu kelas mayoritas), Logistic Regression, Random Forest (200 pohon), SVM-RBF. Validasi silang berstrata 5 lipatan; metrik utama **F1-score kelas diabetes dan ROC-AUC**. Model terbaik disetel dengan GridSearchCV (C = 0,01; 0,1; 1; 10).
- *Interpretasi:* koefisien Logistic Regression dan *feature importance* Random Forest (300 pohon).
- *Unsupervised:* PCA 2 komponen dan K-Means (K = 2–8, dipilih dengan elbow dan silhouette), dibandingkan dengan label memakai ARI.

## 4. Hasil

### 4.1 Kualitas data

Kolom **Insulin** memiliki nilai nol terbanyak (**374 baris; 48,7%**), diikuti **SkinThickness** (227; 29,6%), BloodPressure (35; 4,6%), BMI (11; 1,4%), dan Glucose (5; 0,7%). Hasil yang melibatkan Insulin dan SkinThickness perlu ditafsirkan hati-hati.

![Distribusi fitur](images/fig1_distribusi_fitur.png){width=88%}
*Gambar 1. Distribusi fitur klinis (nilai 0 masih disertakan). Insulin paling miring; maksimum 846 µU/mL.*

![Boxplot](images/fig2_boxplot.png){width=88%}
*Gambar 2. Perbandingan fitur antara pasien diabetes dan tidak diabetes. Glucose paling jelas berbeda.*

Korelasi tertinggi terhadap diabetes: **Glucose (r = 0,47)**, BMI (0,29), Age (0,24). Pregnancies berkorelasi kuat dengan Age (r = 0,54).

### 4.2 Perbandingan model (validasi silang 5 lipatan)

| Model | Akurasi | Precision | Recall | F1-score | ROC-AUC |
|---|:-:|:-:|:-:|:-:|:-:|
| Baseline | 0,651 | 0,000 | 0,000 | 0,000 | 0,500 |
| **Logistic Regression** | **0,788** | **0,764** | 0,575 | **0,655** | **0,843** |
| Random Forest | 0,774 | 0,710 | 0,608 | 0,653 | 0,820 |
| SVM | 0,780 | 0,742 | 0,584 | 0,648 | 0,833 |

Ketiga model jauh di atas baseline, tetapi selisih antarmodel (F1 hanya 0,007) lebih kecil daripada standar deviasi antar lipatan (0,019–0,025), sehingga **keunggulan Logistic Regression belum bermakna**. Penyetelan menghasilkan C = 0,1 dengan ROC-AUC 0,844, hanya beda 0,001 dari C = 1.

### 4.3 Evaluasi pada data uji (154 sampel)

| Kelas | Precision | Recall | F1-score | Jumlah |
|---|:-:|:-:|:-:|:-:|
| Tidak diabetes | 0,741 | 0,800 | 0,769 | 100 |
| Diabetes | 0,565 | 0,481 | 0,520 | 54 |
| Akurasi | | | 0,688 | 154 |

**ROC-AUC data uji = 0,810.** Confusion matrix: 80 TN, 20 FP, **28 FN**, 26 TP.

![Confusion matrix dan ROC](images/fig4_confusion_roc.png){width=80%}
*Gambar 3. Confusion matrix dan kurva ROC pada data uji.*

### 4.4 Pengaruh fitur

![Pengaruh fitur](images/fig5_pengaruh_fitur.png){width=88%}
*Gambar 4. Koefisien Logistic Regression (kiri) dan feature importance Random Forest (kanan).*

| Fitur | Koef. LR | Imp. RF | Sesuai teori biologis? |
|---|:-:|:-:|---|
| Glucose | 1,183 | 0,268 | Sesuai |
| BMI | 0,689 | 0,160 | Sesuai |
| DiabetesPedigreeFunction | 0,233 | 0,124 | Sesuai |
| Age | 0,148 | 0,121 | Sesuai |
| Insulin | −0,066 | 0,091 | Tidak sesuai arah |
| BloodPressure | −0,044 | 0,085 | Sebagian sesuai |
| Pregnancies | 0,377 | 0,079 | Sesuai arah; tumpang tindih dengan Age |
| SkinThickness | 0,028 | 0,072 | Sebagian sesuai |

### 4.5 PCA dan K-Means

PC1 menjelaskan 28,5% dan PC2 18,7% varians (**total 47,2%**). K = 2 dipilih (silhouette tertinggi, **0,196**; elbow tanpa tekukan tegas). **ARI = 0,161** terhadap status diabetes.

![Elbow dan silhouette](images/fig6_elbow_silhouette.png){width=62%}
*Gambar 5. Pemilihan K dengan metode elbow dan silhouette.*

![PCA](images/fig7_pca_kmeans.png){width=72%}
*Gambar 6. Proyeksi PCA berdasarkan status diabetes asli (kiri) dan klaster K-Means (kanan).*

| Klaster | Tidak diabetes | Diabetes | Jumlah | Glucose | BMI | Age | Preg. |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 0 | 156 (44,4%) | 195 (55,6%) | 351 | 138,0 | 35,4 | 41,1 | 5,8 |
| 1 | 344 (82,5%) | 73 (17,5%) | 417 | 107,9 | 30,0 | 26,6 | 2,2 |

## 5. Diskusi

**Performa model.** Semua model mencapai ROC-AUC sekitar 0,82–0,84 sehingga kemampuan membedakan kelas cukup baik. Namun pada ambang batas bawaan (0,5), **recall diabetes hanya 0,481**: 28 dari 54 pasien (±52%) terlewat. Untuk skrining ini belum memadai, karena kasus terlewat (FN) lebih merugikan daripada alarm palsu (FP). Menurunkan ambang batas dapat menaikkan recall dengan konsekuensi lebih banyak FP, tetapi belum diuji. Performa data uji lebih rendah daripada validasi silang, namun ROC-AUC hanya turun 0,843 → 0,810 sehingga tidak ada indikasi overfitting berat. Data uji kecil: satu pasien setara ±1,9 poin persentase recall.

**Kesesuaian dengan biologi.**

- **Glucose** (terkuat di kedua model) mencerminkan gangguan regulasi glukosa akibat resistensi insulin atau disfungsi sel beta pankreas, dan merupakan kriteria diagnosis diabetes.
- **BMI** (peringkat 2) sejalan dengan kaitan kelebihan lemak tubuh dan resistensi insulin.
- **Age, Pregnancies, DiabetesPedigreeFunction** berpengaruh positif, konsisten dengan faktor usia, riwayat diabetes gestasional, dan genetik. Pregnancies tumpang tindih dengan Age (r = 0,54), terlihat dari peringkatnya yang ke-3 pada LR tetapi ke-7 pada RF.
- **Insulin dan BloodPressure** berkoefisien negatif kecil mendekati nol, lebih tepat dibaca sebagai pengaruh lemah. Penyebab kemungkinan: korelasi antarfitur dan hampir 50% data Insulin diimputasi. **SkinThickness** paling lemah karena informasinya sebagian terwakili BMI dan sebagian datanya kosong.

**Pengelompokan.** Silhouette 0,196 dan ARI 0,161 menunjukkan klaster saling tumpang tindih dan hanya sedikit di atas acak terhadap status diabetes. Klaster 0 memang lebih berisiko (glukosa, BMI, usia, kehamilan lebih tinggi) dan memuat 72,8% pasien diabetes, tetapi pemisahannya banyak dipengaruhi **usia dan jumlah kehamilan** (PC2), bukan semata diabetes. PCA dua dimensi hanya mewakili 47,2% varians.

**Keterbatasan.** Dataset kecil (768); hanya perempuan Pima usia ≥ 21 tahun sehingga generalisasi terbatas; Insulin dan SkinThickness banyak kosong; belum ada validasi eksternal; perbandingan algoritma terbatas; model menunjukkan hubungan statistik, **bukan sebab-akibat**, dan bukan pengganti diagnosis medis.

## 6. Kesimpulan

1. Nilai nol tidak wajar pada 5 fitur klinis ditangani dengan imputasi median dalam `Pipeline` tanpa kebocoran data.
2. **Logistic Regression** menjadi model terbaik (CV: F1 0,655; ROC-AUC 0,843), meski selisihnya dengan model lain tidak bermakna. Pada data uji, ROC-AUC 0,810 dan recall diabetes 0,481 (28 dari 54 kasus terlewat).
3. **Glucose** adalah fitur paling berpengaruh, diikuti **BMI**, sesuai mekanisme biologis diabetes tipe 2.
4. PCA menjelaskan 47,2% varians; K-Means (silhouette 0,196; ARI 0,161) belum cukup menggambarkan status diabetes.

**Saran:** gunakan data yang lebih besar dan beragam (termasuk populasi Indonesia) dengan validasi eksternal; coba gradient boosting, `class_weight`, dan penyesuaian ambang batas untuk menaikkan recall; gunakan imputasi lebih canggih untuk Insulin dan SkinThickness; konsultasikan dengan tenaga kesehatan sebelum penggunaan praktis.

> **Disclaimer:** proyek ini untuk pembelajaran dan bukan pengganti diagnosis medis.
