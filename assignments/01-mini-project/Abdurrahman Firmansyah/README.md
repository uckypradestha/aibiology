# Mini Assignment - Abdurrahman Firmansyah B1C024025

## Identitas
- Nama: Abdurrahman Firmansyah
- NIM: B1C024025
- Kelas: Bioter A

## Deskripsi
Perbandingan Logistic Regression dan Random Forest untuk klasifikasi kanker payudara
dengan pembagian data 70% training / 30% testing.

## Hasil
| Model | Accuracy | Precision (Malignant) | Recall (Malignant) | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.9883 | 0.9844 | 0.9844 | 0.9844 |
| Random Forest | 0.9357 | 0.9206 | 0.9062 | 0.9134 |

## Kesimpulan
Logistic Regression lebih baik di seluruh metrik, terutama Recall pada kelas
Malignant yang kritis dalam deteksi kanker.

## Mengapa Accuracy Saja Tidak Cukup?
Dataset tidak seimbang (212 Malignant vs 357 Benign). Model yang selalu menebak "Benign" akan mendapat akurasi ~63% tetapi tidak berguna.
False Negative berbahaya jika kanker (Malignant) diprediksi sebagai Benign, pasien bisa kehilangan nyawa.
Recall pada kelas Malignant menjadi metrik paling kritis (meminimalkan FN).
Precision juga penting untuk menghindari False Positive berlebihan yang menyebabkan kecemasan & biaya pemeriksaan tak perlu.
F1-Score menyeimbangkan Precision & Recall.
ROC-AUC menilai kemampuan diskriminasi model secara keseluruhan pada berbagai threshold.
