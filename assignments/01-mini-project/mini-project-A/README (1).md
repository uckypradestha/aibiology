# Breast Cancer Classification Using Machine Learning

Praktikum klasifikasi tumor payudara (ganas/jinak) menggunakan **Logistic Regression** dan **Random Forest**, lalu membandingkan performa keduanya.

**Nama:** Laksita Widia Pratiwi
**NIM:** B1C024004
**Program Studi:** Biologi Terapan, Universitas Jenderal Soedirman

## Deskripsi

Notebook ini membangun model klasifikasi untuk membedakan tumor *malignant* (ganas) dan *benign* (jinak) berdasarkan fitur numerik hasil citra sel. Dua algoritma yang dibandingkan:

1. Logistic Regression
2. Random Forest Classifier

## Dataset

Breast Cancer Wisconsin (Diagnostic) Dataset dari Scikit-learn (`load_breast_cancer()`), berisi 569 sampel dan 30 fitur numerik.

## Tahapan

1. Import library
2. Load dataset
3. Exploratory Data Analysis (EDA)
4. Pembagian data (70% training, 30% testing, `random_state=42`, `stratify=y`)
5. Normalisasi fitur dengan `StandardScaler` dalam `Pipeline`
6. Prediksi model
7. Evaluasi: accuracy, precision, recall, F1-score, ROC-AUC
8. Confusion matrix
9. Prediksi satu sampel baru

## Hasil

| Metrik | Logistic Regression | Random Forest |
| :--- | :---: | :---: |
| Accuracy | 98,83% | 93,57% |
| ROC-AUC | 0,9981 | 0,9913 |
| Recall (malignant) | 0,98 | 0,91 |
| False Negative (ganas diprediksi jinak) | 1 | 6 |

**Kesimpulan:** Logistic Regression memberikan hasil lebih baik pada dataset ini, baik dari akurasi maupun jumlah kesalahan pada kelas tumor ganas. Pada kasus medis, evaluasi tidak cukup hanya dengan accuracy karena kesalahan *false negative* berisiko menunda penanganan pasien, sehingga recall dan F1-score juga perlu diperhatikan.

## Cara Menjalankan

1. Buka [Google Colab](https://colab.research.google.com).
2. Pilih **File → Upload notebook**, lalu unggah `breast_cancer_classification_revisi.ipynb`.
3. Jalankan semua cell lewat **Runtime → Run all**.

Library yang dipakai: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`.

## Struktur File

```
.
├── breast_cancer_classification_revisi.ipynb
└── README.md
```

## Catatan

Model ini dibuat hanya untuk pembelajaran dan demonstrasi Machine Learning. Hasil prediksi tidak boleh digunakan sebagai pengganti diagnosis medis profesional.
