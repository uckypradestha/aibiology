# 🐧 Penguin Species Classification Using Machine Learning

## 1. Deskripsi Project

Project ini bertujuan untuk menganalisis karakteristik morfologi tiga spesies penguin, yaitu Adelie, Chinstrap, dan Gentoo, menggunakan pendekatan Machine Learning (ML). Analisis dilakukan dengan memanfaatkan dataset Palmer Penguins yang berisi pengukuran morfologi penguin.

Project ini menggabungkan pendekatan Supervised Learning untuk mengklasifikasikan spesies penguin dan Unsupervised Learning untuk mengeksplorasi pola pengelompokan berdasarkan karakteristik morfologi. Hasil analisis digunakan untuk mengetahui kemampuan model dalam membedakan spesies serta memahami pola morfologi antarspesies.

## 2. Tujuan Project

- Mengklasifikasikan tiga spesies penguin berdasarkan karakteristik morfologi.
- Membandingkan performa beberapa algoritma Supervised Learning.
- Mengidentifikasi pola pengelompokan morfologi menggunakan Unsupervised Learning.
- Mengetahui variabel morfologi yang berperan dalam membedakan spesies penguin.

## 3. Dataset

Dataset yang digunakan adalah **Palmer Penguins Dataset** yang dikumpulkan di sekitar Palmer Station, Antartika.

Dataset terdiri atas tiga spesies penguin:

- Adelie (*Pygoscelis adeliae*)
- Chinstrap (*Pygoscelis antarcticus*)
- Gentoo (*Pygoscelis papua*)

Dataset memiliki 344 data individu dengan variabel morfologi utama:

| Variabel | Keterangan | Satuan |
|---|---|---|
| Bill Length | Panjang paruh | mm |
| Bill Depth | Kedalaman paruh | mm |
| Flipper Length | Panjang sirip | mm |
| Body Mass | Massa tubuh | g |
| Species | Spesies penguin | Label |

Sumber dataset:  
https://github.com/allisonhorst/palmerpenguins

## 4. Metode Analisis

Project ini menggunakan dua pendekatan Machine Learning.

### A. Supervised Learning

Supervised Learning digunakan untuk memprediksi spesies penguin berdasarkan data morfologi yang telah memiliki label spesies.

Tahapan analisis meliputi:

- Pembagian data menjadi data training dan testing.
- Data preprocessing.
- Pelatihan model klasifikasi.
- Pengujian model menggunakan data testing.
- Evaluasi performa model.

Algoritma yang digunakan akan dibandingkan berdasarkan nilai Accuracy, Precision, Recall, dan F1-score.

### B. Unsupervised Learning

Unsupervised Learning digunakan untuk mengeksplorasi pola pengelompokan alami berdasarkan karakteristik morfologi tanpa menggunakan label spesies sebagai dasar pembentukan kelompok.

Analisis dilakukan menggunakan Principal Component Analysis (PCA) untuk mereduksi dimensi data morfologi menjadi dua komponen utama sehingga pola pengelompokan dapat divisualisasikan dan dibandingkan dengan spesies sebenarnya.

## 5. Tahapan Analisis Notebook

Notebook disusun secara berurutan agar setiap tahapan analisis dapat dipahami dan dijalankan.

1. **Import Library**  
   Mengimpor library Python yang diperlukan untuk pengolahan data, visualisasi, dan Machine Learning.

2. **Load Dataset**  
   Memuat dataset Palmer Penguins ke dalam lingkungan analisis.

3. **Exploratory Data Analysis (EDA)**  
   Mengeksplorasi struktur data, distribusi variabel, karakteristik morfologi, serta perbedaan antarspesies.

4. **Data Preprocessing**  
   Melakukan penanganan missing values, pemilihan variabel, dan standardisasi data jika diperlukan.

5. **Supervised Learning**  
   Melatih dan menguji beberapa algoritma klasifikasi untuk memprediksi spesies penguin.

6. **Evaluasi Model**  
   Membandingkan performa model menggunakan Accuracy, Precision, Recall, F1-score, dan Confusion Matrix.

7. **Unsupervised Learning (PCA)**  
   Melakukan reduksi dimensi dan visualisasi pola pengelompokan morfologi penguin.

8. **Interpretasi Hasil**  
   Menginterpretasikan hasil klasifikasi dan pengelompokan berdasarkan karakteristik morfologi serta sudut pandang biologi.

9. **Kesimpulan**  
   Merangkum hasil analisis, performa algoritma, dan pola morfologi yang ditemukan.

## 6. Tools dan Library

Project ini menggunakan Python dengan beberapa library berikut:

- Pandas: pengolahan dan manipulasi data.
- NumPy: perhitungan numerik.
- Matplotlib: visualisasi data.
- Seaborn: visualisasi statistik.
- Scikit-learn: implementasi algoritma Machine Learning, preprocessing, PCA, dan evaluasi model.

## 7. Cara Menjalankan Project

Project dapat dijalankan melalui Google Colab tanpa perlu instalasi Python secara lokal.

1. Buka file notebook `.ipynb` menggunakan Google Colab.
2. Pastikan dataset dapat diakses.
3. Jalankan setiap cell secara berurutan dari awal hingga akhir.
4. Perhatikan penjelasan Markdown pada setiap tahapan analisis.
5. Amati hasil visualisasi, evaluasi model, dan prediksi spesies.
6. Ulangi proses analisis jika diperlukan dengan mengikuti instruksi pada notebook.

## 8. Luaran Project

Luaran yang dihasilkan dari project ini meliputi:

- Notebook analisis Machine Learning.
- Hasil klasifikasi spesies penguin.
- Perbandingan performa algoritma.
- Confusion Matrix.
- Visualisasi PCA.
- Interpretasi hasil analisis berdasarkan karakteristik morfologi penguin.

## 9. Tim Pengembang

Project ini disusun oleh mahasiswa sebagai bagian dari Mini Project Machine Learning untuk bidang Biologi, dengan memanfaatkan pendekatan analisis data dan pemodelan Machine Learning untuk memahami karakteristik morfologi spesies penguin.
