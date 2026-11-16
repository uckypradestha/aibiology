# Mini Assignment – Breast Cancer Classification

Identitas
- Nama: Duara Karami Wahyudi
- NIM: B1C024057
- Program Studi: S1 Biologi Terapan
- Kelas: B

## Deskripsi Eksperimen
Eksperimen ini dilakukan untuk membandingkan kinerja dua algoritma klasifikasi, yaitu Logistic Regression dan Random Forest Classifier, pada dataset Breast Cancer Wisconsin.

Pembagian dataset menggunakan 70% data training dan 30% data testing. Evaluasi model dilakukan menggunakan accuracy, precision, recall, F1-score, dan confusion matrix.

## Hasil Evaluasi

| Metrik | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 98,83% | 93,57% |
| Precision (Malignant) | 98,44% | 92,06% |
| Recall (Malignant) | 98,44% | 90,62% |
| F1-score (Malignant) | 98,44% | 91,34% |
| False Negative | 1 | 6 |

## Kesimpulan
Berdasarkan hasil pengujian, Logistic Regression memperoleh nilai accuracy sebesar 98,83%, sedangkan Random Forest memperoleh 93,57%. Pada kelas malignant, Logistic Regression menghasilkan recall sebesar 98,44% dan hanya terdapat 1 kasus False Negative. Adapun Random Forest menghasilkan recall sebesar 90,62% dengan 6 kasus False Negative. Hasil tersebut menunjukkan bahwa Logistic Regression lebih baik dalam mengenali data malignant pada pengujian ini. Oleh karena itu, penilaian model klasifikasi kanker tidak cukup hanya berdasarkan accuracy, tetapi perlu memperhatikan nilai recall malignant dan jumlah False Negative. Hasil eksperimen ini hanya digunakan untuk keperluan pembelajaran machine learning dan bukan sebagai dasar diagnosis medis.
