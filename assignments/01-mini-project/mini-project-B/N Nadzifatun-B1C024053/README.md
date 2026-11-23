Nama : N Nadzifatun Fissilmi Kaffah 
NIM : B1C024053 
Mata Kuliah : Data Sains dan AI 


Deskripsi Eksperimen
Eksperimen ini bertujuan untuk membandingkan performa dua algoritma machine learning, yaitu Logistic Regression dan Random Forest Classifier, dalam mengklasifikasikan jenis kanker payudara (malignant / ganas vs benign / jinak) menggunakan dataset load_breast_cancer.

Tahapan utama yang dilakukan dalam kode tersebut meliputi:

Pembagian Data: Dataset dibagi menjadi 70% data latih dan 30% data uji menggunakan fungsi train_test_split dengan argumen stratify=y agar proporsi kelas antara data ganas dan jinak tetap seimbang pada kedua bagian.

Pelatihan Model:

Model A (Logistic Regression): Menggunakan pipeline yang diawali dengan StandardScaler untuk normalisasi skala data, diikuti oleh regresi logistik.

Model B (Random Forest): Menggunakan ensemble tree dengan jumlah 100 pohon (n_estimators=100) tanpa scaling khusus karena algoritma berbasis pohon tidak sensitif terhadap skala fitur.

Evaluasi: Kedua model diuji menggunakan data testing, kemudian performanya dievaluasi melalui nilai akurasi dan visualisasi confusion matrix.

Hasil Evaluasi Eksperimen
Berdasarkan hasil visualisasi confusion matrix dan perhitungan akurasi pada data uji (total 171 sampel), performa kedua model dijabarkan sebagai berikut:

Logistic Regression:

Akurasi: Sangat tinggi, yaitu 98.83%.

Confusion Matrix:

Benar memprediksi Ganas (True Malignant): 63

Benar memprediksi Jinak (True Benign): 106

Salah memprediksi Ganas sebagai Jinak (False Negative): 1

Salah memprediksi Jinak sebagai Ganas (False Positive): 1

Analisis: Model ini sangat handal dengan total kesalahan prediksi hanya 2 dari 171 data uji. Kasus False Negative yang hanya 1 sangat krusial dalam dunia medis karena meminimalisir pasien kanker yang tidak terdeteksi.

Random Forest Classifier:

Akurasi: 93.57%.

Confusion Matrix:

Benar memprediksi Ganas (True Malignant): 58

Benar memprediksi Jinak (True Benign): 102

Salah memprediksi Ganas sebagai Jinak (False Negative): 6

Salah memprediksi Jinak sebagai Ganas (False Positive): 5

Analisis: Performa model ini cukup baik, namun memiliki jumlah kesalahan yang lebih tinggi (total 11 kesalahan). Angka False Negative sebanyak 6 kasus patut diwaspadai karena lebih banyak pasien ganas yang keliru terdiagnosis sebagai jinak.

Kesimpulan
Secara keseluruhan, Logistic Regression menunjukkan performa yang lebih unggul dibandingkan Random Forest pada eksperimen ini, baik dari segi akurasi total maupun minimnya tingkat kesalahan prediksi (error rate).

Hal ini mengindikasikan bahwa hubungan antar fitur pada dataset breast cancer cenderung linier dan dapat dipisahkan secara efektif setelah melalui proses standarisasi (StandardScaler). Sementara itu, Random Forest dengan parameter default-nya justru sedikit tertinggal dan berpotensi memerlukan tuning hyperparameter lebih lanjut agar kinerjanya bisa optimal pada dataset ini.
