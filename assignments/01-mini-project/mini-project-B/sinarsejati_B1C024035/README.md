# Mini Assignment — Logistic Regression vs Random Forest (Klasifikasi Kanker Payudara)

## Identitas Mahasiswa
- **Nama:** Sinar Lintang Sejati
- **NIM:** B1C024035
- **Program Studi:** S1 Biologi Terapan, Fakultas Biologi, Universitas Jenderal Soedirman
- **Kelas:** Biologi Terapan B

## Deskripsi Eksperimen
Eksperimen membandingkan **Logistic Regression** (dengan `StandardScaler`) dan **Random Forest Classifier** (100 pohon) pada *Breast Cancer Wisconsin Dataset* (scikit-learn; 569 sampel, 30 fitur; 212 malignant, 357 benign).

- Pembagian data: **70% training (398) / 30% testing (171)**, `stratify=y`, `random_state=42`
- Split yang sama dipakai untuk kedua algoritma
- Kelas positif untuk precision/recall/F1: **malignant (0)**, karena kelas ini yang paling penting dideteksi

## Hasil Evaluasi (data uji 30%)

| Metrik | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | **0,9883** | 0,9357 |
| Precision (malignant) | **0,9844** | 0,9206 |
| Recall (malignant) | **0,9844** | 0,9062 |
| F1-score (malignant) | **0,9844** | 0,9134 |
| ROC-AUC | **0,9981** | 0,9913 |
| False Negative | **1** | 6 |
| False Positive | **1** | 5 |

Confusion Matrix: `results/confusion_matrix.png` — Perbandingan metrik: `results/evaluation_results.png`

Pemeriksaan tambahan (Stratified 5-fold CV, seluruh data): accuracy Logistic Regression 0,9737 ± 0,017 vs Random Forest 0,9561 ± 0,012.

## Analisis Singkat
Logistic Regression lebih baik pada semua metrik di eksperimen ini, terutama recall malignant (melewatkan 1 dari 64 kasus ganas, versus 6 dari 64 pada Random Forest). Cross-validation menunjukkan arah yang sama dengan selisih yang lebih kecil, sehingga sebagian selisih pada satu split berasal dari variasi pembagian data. Dugaan penyebabnya: data kecil dengan fitur numerik berkorelasi tinggi yang hampir dapat dipisahkan secara linear sehingga cocok untuk model linear ter-*scale*; Random Forest memakai parameter bawaan tanpa *tuning*.

## Mengapa Accuracy Saja Tidak Cukup
Biaya kesalahan tidak setara: *False Negative* (kanker ganas diprediksi jinak) jauh lebih berbahaya daripada *False Positive*. Accuracy menyembunyikan jenis kesalahan dan bisa menyesatkan pada kelas tidak seimbang. Karena itu perlu Confusion Matrix, recall, precision, F1-score, dan ROC-AUC.

## Kesimpulan
1. Pembagian 70:30 diterapkan pada kedua algoritma dengan split identik.
2. Logistic Regression (accuracy 98,83%) mengungguli Random Forest (93,57%) pada dataset dan konfigurasi ini.
3. Perbedaan terpenting adalah jumlah False Negative (1 vs 6).
4. Kesimpulan berlaku untuk eksperimen ini, bukan klaim umum bahwa model linear selalu lebih baik.

## File
- `mini_assignment.ipynb` — source code beserta hasil eksekusi
- `results/confusion_matrix.png`, `results/evaluation_results.png`
