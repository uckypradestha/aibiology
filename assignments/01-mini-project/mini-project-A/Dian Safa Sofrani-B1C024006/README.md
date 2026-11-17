# Mini Assignment - Breast Cancer Classification

## Identitas Mahasiswa
- **Nama:** [Isi Nama Lengkap Anda]
- **NIM:** [Isi NIM Anda]
- **Kelas:** [Isi Kelas Anda]

## Deskripsi Eksperimen
Eksperimen ini bertujuan untuk membandingkan performa dua algoritma klasifikasi Machine Learning, yaitu **Logistic Regression** dan **Random Forest Classifier**, pada *Breast Cancer Wisconsin Dataset*. Dataset dibagi dengan proporsi **70% data training (398 sampel)** dan **30% data testing (171 sampel)** menggunakan teknik *stratified split* dan standardisasi fitur (`StandardScaler`).

## Hasil Evaluasi
| Metrik Evaluasi | Logistic Regression | Random Forest Classifier |
| :--- | :---: | :---: |
| **Accuracy** | **98.83%** | 93.57% |
| **Precision (Malignant)** | **0.98** | 0.92 |
| **Recall (Malignant)** | **0.98** | 0.91 |
| **F1-Score (Malignant)** | **0.98** | 0.91 |
| **Precision (Benign)** | **0.99** | 0.94 |
| **Recall (Benign)** | **0.99** | 0.95 |
| **F1-Score (Benign)** | **0.99** | 0.95 |
| **ROC-AUC Score** | **0.9981** | 0.9913 |

## Kesimpulan Eksperimen
Berdasarkan hasil eksperimen yang telah dilakukan, dapat disimpulkan bahwa:
1. Dataset Breast Cancer Wisconsin memiliki 569 sampel dengan 30 fitur numerik dan tidak memiliki *missing values*.
2. Pembagian dataset 70% training dan 30% testing menghasilkan 398 sampel latih dan 171 sampel uji.
3. Algoritma **Logistic Regression** memberikan hasil klasifikasi yang jauh lebih baik dibandingkan **Random Forest Classifier** pada dataset ini, dengan akurasi **98.83%** (vs 93.57%) dan skor ROC-AUC **0.9981** (vs 0.9913).
4. Berdasarkan *Confusion Matrix*, Logistic Regression hanya melakukan **1 kesalahan** prediksi *False Negative* (tumor ganas diprediksi jinak), sedangkan Random Forest melakukan **6 kesalahan** *False Negative*.
5. Dalam konteks deteksi kanker, metrik **Accuracy saja tidak cukup** karena tidak menggambarkan jenis kesalahan prediksi secara spesifik. Kesalahan *False Negative* (pasien kanker ganas terdeteksi jinak) sangat fatal karena menunda penanganan medis. Oleh karena itu, metrik **Recall** (sensitivitas mendeteksi kelas ganas) dan **Precision** sangat krusial untuk dievaluasi bersamaan.
