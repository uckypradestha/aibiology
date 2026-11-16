Nama : Putri Setyawati
NIM : B1C024047
MatKul : Data Sains dan AI

Deskripsi Eksperimen
Eksperimen ini membandingkan dua metode untuk mengklasifikasikan tumor payudara ganas atau jinak berupa Logistic Regression dan Random Forest. Data yang dipakai adalah dataset Breast Cancer Wisconsin dari scikit-learn, berisi 569 data dengan 30 angka yang menggambarkan ukuran dan bentuk sel, seperti radius, tekstur, luas, dan bentuk ketidak beraturan sel. Tidak ada data yang kosong, 212 kasus bersifat ganas dan 357 bersifat jinak.
Data dibagi menjadi 70% sebagai model dan 30% sebagai mengujinya, dengan pembagian berupa jumlah kasus ganas dan jinak tetap seimbang. Logistic Regression perlu data diskalakan terlebih dahulu, sedangkan Random Forest tidak. Kinerja keduanya diukur menggunakan akurasi, precision, recall, F1-score, confusion matrix, dan ROC-AUC.

Hasil Evaluasi
Dari 171 data uji, Logistic Regression benar sekitar 98,8% dan hanya salah 2 kali. Random Forest benar sekitar 93,6% dan salah 11 kali. Jadi Logistic Regression lebih baik di semua ukuran penilaian.
Yang terpenting yaitu Logistic Regression hanya melewatkan 1 tumor ganas (dikira jinak), sedangkan Random Forest melewatkan 6. Kemungkinan penyebab tersebut, datanya sedikit dan polanya cukup sederhana sehingga model yang lebih simpel justru bekerja lebih baik. 
Akurasi tidak dapat membedakan jenis kesalahan. Tumor ganas yang dikira jinak dapat lebih berbahaya karena pasien bisa terlambat ditangani, sedangkan tumor jinak yang dikira ganas biasanya hanya berujung pada pemeriksaan tambahan. Akurasi juga bisa menipu dalam dataset ini, menebak “jinak” terus-menerus saja sudah menghasilkan akurasi sekitar 62,7%. Karena itu dibutuhkan ukuran lain, terutama recall, yaitu seberapa banyak kasus ganas yang berhasil terdeteksi.

Kesimpulan
Dalam percobaan ini, Logistic Regression lebih baik daripada Random Forest, dengan akurasi 98,8% berbanding 93,6%, dan lebih aman untuk keperluan medis karena hanya melewatkan 1 tumor ganas, sedangkan Random Forest melewatkan 6. Menilai model kanker tidak cukup dari akurasi saja, tetapi harus melihat juga seberapa banyak kasus ganas yang berhasil terdeteksi. Walaupun begitu, pengujian hanya dilakukan satu kali dan Random Forest belum disetel secara optimal, sehingga perbandingannya belum sepenuhnya sama. Ke depannya, pengujian bisa diulang dengan cara yang lebih bervariasi, dan model harus diuji secara klinis sebelum dipakai di dunia nyata sebagai alat bantu keputusan. Dapat dilakukan juga dengan menambahkan fungsi precision_score, recall_score, dan f1_score ke daftar impor di awal notebook agar bisa dijalankan ulang tanpa error.
