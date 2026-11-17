## Identitas Mahasiswa
- Nama: Najwa Alifia Kusuma
- NIM: B1C024058
- Kelas: Biologi Terapan B

## Deskripsi Assignment
Implementasi dan perbandingan algoritma Logistic Regression dan Random Forest Classifier untuk klasifikasi kanker.

## Eksperimen
- [x] Pembagian dataset 70% training dan 30% testing
- [x] Implementasi Logistic Regression
- [x] Implementasi Random Forest Classifier
- [x] Perbandingan accuracy, precision, recall, dan F1-score
- [x] Confusion Matrix kedua algoritma
- [x] Analisis hasil eksperimen
- [x] Kesimpulan

## Hasil Evaluasi & Kesimpulan
| Metrik | Logistic Regression | Random Forest Classifier |
|---|---|---|
| Accuracy | 0.9883 | 0.9357 |
| Precision | 0.9844 | 0.9206 |
| Recall | 0.9844 | 0.9062 |
| F1-Score | 0.9844 | 0.9134 |

- **Algoritma Terbaik:** Logistic Regression memberikan hasil klasifikasi lebih baik dibandingkan Random Forest Classifier pada pembagian data 70:30.
- **Pentingnya Metrik Selain Accuracy:** Dalam konteks klasifikasi kanker, *Accuracy* saja tidak cukup karena kesalahan *False Negative* (pasien kanker terdeteksi sehat) berisiko fatal, sehingga metrik **Recall** dan **F1-Score** wajib diperhatikan.

## File Pengumpulan
- Source code: `assignments/mini-assignment/najwa-B1C024058/mini_assignment.ipynb`
- Hasil eksperimen: `assignments/mini-assignment/najwa-B1C024058/results/`
- Dokumentasi: `assignments/mini-assignment/najwa-B1C024058/README.md`

## Pernyataan
Saya menyatakan bahwa pekerjaan ini merupakan hasil pengerjaan dan pemahaman saya sendiri.
