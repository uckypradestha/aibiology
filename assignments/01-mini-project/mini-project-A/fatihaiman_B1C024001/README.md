## Identitas Mahasiswa
- Nama: Fatih Aiman
- NIM: B1C024001
- Kelas: A

## Deskripsi Assignment
Implementasi dan perbandingan algoritma **Logistic Regression** dan **Random Forest Classifier** untuk klasifikasi kanker payudara menggunakan *Breast Cancer Wisconsin Dataset* dari Scikit-learn dengan pembagian data **70% training (398 sampel)** dan **30% testing (171 sampel)**.

## Eksperimen
- [x] Pembagian dataset 70% training dan 30% testing
- [x] Implementasi Logistic Regression
- [x] Implementasi Random Forest Classifier
- [x] Perbandingan accuracy, precision, recall, dan F1-score
- [x] Confusion Matrix kedua algoritma
- [x] Analisis hasil eksperimen
- [x] Kesimpulan

## Ringkasan Hasil & Analisis
### 1. Perbandingan Metrik Evaluasi (`random_state=42`)
| Algoritma | Accuracy (%) | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **98.83%** | **0.9907** | **0.9907** | **0.9907** | **0.9981** |
| **Random Forest** | 93.57% | 0.9444 | 0.9533 | 0.9488 | 0.9913 |

### 2. Ringkasan Confusion Matrix (171 Sampel Uji: 64 Malignant, 107 Benign)
- **Logistic Regression:** 63 TN, 1 FP (malignant diprediksi benign), 1 FN (benign diprediksi malignant), 106 TP — total hanya **2 kesalahan prediksi**.
- **Random Forest:** 58 TN, 6 FP (malignant diprediksi benign), 5 FN (benign diprediksi malignant), 102 TP — total **11 kesalahan prediksi**.

### 3. Analisis & Kesimpulan
- **Algoritma Terbaik:** **Logistic Regression** memberikan performa klasifikasi yang lebih tinggi di seluruh metrik dibandingkan Random Forest. Fitur-fitur pada dataset ini memiliki separabilitas linear yang sangat baik setelah dilakukan standardisasi (`StandardScaler`), sedangkan pada dataset berukuran relatif kecil (569 sampel), model ensemble pohon keputusan seperti Random Forest lebih rentan terhadap *overfitting* ringan dibandingkan model linear terregularisasi.
- **Urgensi Metrik Selain Accuracy:** Dalam konteks medis/kanker, *accuracy* saja tidak cukup karena adanya **risiko kesalahan yang tidak seimbang (*asymmetric cost of error*)** dan **ketidakseimbangan kelas (*class imbalance*)**. Kesalahan memprediksi tumor ganas (*malignant*) sebagai jinak (*benign*) jauh lebih berbahaya bagi keselamatan pasien dibandingkan sebaliknya, sehingga **Recall**, **Precision**, **F1-Score**, dan **Confusion Matrix** wajib dianalisis bersama-sama.

## File Pengumpulan
- Source code
- Hasil eksperimen
- Dokumentasi: `README.md` 

## Pernyataan
Saya menyatakan bahwa pekerjaan ini merupakan hasil pengerjaan saya sendiri dengan bantuan AI.
