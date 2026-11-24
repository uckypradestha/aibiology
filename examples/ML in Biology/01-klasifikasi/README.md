# Klasifikasi Penyakit Jantung (*Heart Disease*) dengan Machine Learning

Materi praktik penerapan Machine Learning pada **data biologi/medis**: memprediksi apakah seorang pasien menderita penyakit jantung berdasarkan data klinisnya.

## 🚀 Mulai di Google Colab

Secara lebih lengkap code dapat diakses melalui:

👉 [[Buka Notebook di Google Colab]((https://drive.google.com/file/d/1kFMfrEJkh6z0TADJWLXMoQs9J2ne67DH/view?usp=sharing))](https://drive.google.com/file/d/1kFMfrEJkh6z0TADJWLXMoQs9J2ne67DH/view?usp=sharing)

### Langkah-langkah

1. **Klik tautan Colab** di atas. Notebook akan terbuka di browser.
2. **Login** dengan akun Google Anda (akun pribadi atau akun kampus).
3. **Simpan salinan** ke Google Drive Anda sendiri: menu **File → Save a copy in Drive**. Ini penting agar pekerjaan Anda tidak hilang dan tidak mengubah notebook asli.
4. Pastikan **koneksi internet aktif**, karena dataset diunduh otomatis dari repositori UCI.
5. Jalankan seluruh sel secara berurutan: menu **Runtime → Run all** (atau tekan `Ctrl + F9`). Anda juga bisa menjalankan sel satu per satu dengan `Shift + Enter` sambil membaca penjelasannya.
6. Jika muncul peringatan *"This notebook was not authored by Google"*, pilih **Run anyway**.

---

## 📂 Isi Notebook

| No | Bagian | Isi |
|----|--------|-----|
| 1 | Import Library | Library yang digunakan (pandas, scikit-learn, seaborn, dll.) |
| 2 | Memuat Dataset | Unduh data UCI Heart Disease dan penjelasan tiap atribut |
| 3 | Pra-pemrosesan Awal | Mengubah label menjadi biner dan memeriksa *missing value* |
| 4 | Eksplorasi Data (EDA) | Distribusi kelas, histogram, matriks korelasi |
| 5 | Pembagian Data | *Train-test split* 80:20 dengan stratifikasi |
| 6 | Pipeline Preprocessing | Imputasi, standardisasi, dan *one-hot encoding* tanpa *data leakage* |
| 7 | Perbandingan Model | Logistic Regression, KNN, SVM, Naive Bayes, Random Forest dengan 5-fold CV |
| 8 | Hyperparameter Tuning | *Grid search* pada Random Forest |
| 9 | Evaluasi Akhir | Confusion matrix, kurva ROC, pengaruh *threshold* terhadap *recall* |
| 10 | Interpretasi Model | *Permutation importance* |
| 11 | Prediksi Pasien Baru | Simulasi prediksi satu pasien |
| 12 | Catatan Etika | Keterbatasan data dan penggunaan model secara bertanggung jawab |

---

## 🧬 Tentang Dataset

**UCI Heart Disease (Cleveland)**: 303 pasien dari Cleveland Clinic Foundation dengan 13 atribut klinis (usia, jenis kelamin, tipe nyeri dada, tekanan darah, kolesterol, hasil EKG, detak jantung maksimum, dan lainnya).

- Label asli `num` bernilai 0-4, diubah menjadi **biner**: 0 = sehat, 1 = penyakit jantung.
- Sumber: <https://archive.ics.uci.edu/dataset/45/heart+disease>

---

## 🛠️ Pemecahan Masalah

| Masalah | Solusi |
|---------|--------|
| `NameError: name '...' is not defined` | Ada sel yang terlewat. Jalankan ulang dari atas lewat **Runtime → Run all**. |
| Dataset gagal diunduh | Periksa koneksi internet. Notebook otomatis mencoba paket `ucimlrepo` sebagai alternatif. Jika masih gagal, coba lagi beberapa saat kemudian. |
| Sesi Colab terputus atau variabel hilang | Colab akan menghapus sesi jika lama tidak aktif. Sambungkan ulang, lalu jalankan **Run all**. |
| *Grid search* terasa lambat | Normal untuk Colab versi gratis. Kurangi kandidat pada `param_grid` jika diperlukan. |
| Tidak bisa mengedit notebook | Anda membuka versi asli. Lakukan **File → Save a copy in Drive** terlebih dahulu. |

---

## Cara Membagikan Tautan (untuk dosen)

**Opsi 1: lewat Google Drive**
1. Unggah file `klasifikasi_heart_disease.ipynb` ke Google Drive.
2. Klik kanan file → **Open with → Google Colaboratory** (jika belum ada, pilih *Connect more apps* dan cari "Colaboratory").
3. Di Colab, klik **Share** (kanan atas) → pada *General access* pilih **Anyone with the link** dengan peran **Viewer**.
4. Salin tautan, lalu tempel ke bagian "Buka Notebook di Google Colab" di atas.

**Opsi 2: lewat GitHub**
1. Unggah `klasifikasi_heart_disease.ipynb` dan `README.md` ke repositori GitHub publik.
2. Tautan Colab-nya berformat:  
   `https://colab.research.google.com/github/<username>/<repositori>/blob/main/klasifikasi_heart_disease.ipynb`
3. Tempel tautan tersebut ke README ini. Anda juga dapat memakai lencana:  
   `[![Buka di Colab](https://colab.research.google.com/assets/colab-badge.svg)](TAUTAN_COLAB_ANDA)`

---

## ⚠️ Catatan Penting

Notebook ini dibuat **untuk tujuan pembelajaran**, bukan sebagai alat diagnosis medis. Dataset berukuran kecil dan berasal dari satu institusi, sehingga hasilnya belum tentu berlaku untuk populasi lain. Hasil prediksi model tidak menggantikan pemeriksaan dan keputusan tenaga medis.

## Referensi

- Detrano, R. et al. (1989). *International application of a new probability algorithm for the diagnosis of coronary artery disease.* The American Journal of Cardiology, 64(5), 304-310.
- Dua, D. & Graff, C. UCI Machine Learning Repository: *Heart Disease Data Set*.
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12, 2825-2830.
