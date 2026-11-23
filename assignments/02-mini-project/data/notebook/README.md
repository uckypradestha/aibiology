# Analisis Supervised dan Unsupervised Learning untuk Klasifikasi serta Segmentasi Diabetes pada Pima Indians Diabetes 

## Anggota Kelompok
- Najswa Mahfirotul Safitri	 (B1C024018)
- Ami Khusnul Amalia	 (B1C024021)
- Kayla Almasa Zahiyah	 (B1C024028)

## Deskripsi Masalah & Relevansi Biologi
Diabetes melitus merupakan gangguan metabolik kronis yang ditandai oleh hiperglikemia akibat terganggunya sekresi maupun kerja insulin, di mana variasi antarindividu pada parameter fisiologis dan antropometrik (seperti glukosa plasma, tekanan darah, ketebalan lipatan kulit, insulin serum, BMI, dan usia) membuat deteksi dini serta pemetaan risiko menjadi kurang optimal jika dianalisis secara terpisah. Oleh karena itu, analisis ini bertujuan untuk mengidentifikasi indikator klinis-biologis yang paling berpengaruh terhadap status diabetes serta mengelompokkan profil individu berdasarkan kemiripan karakteristik metabolik menggunakan pendekatan *supervised* dan *unsupervised machine learning* pada dataset `diabetes` (`version=1`). Dalam bidang Biologi Terapan, pendekatan komputasional berbasis data ini berperan penting untuk menerjemahkan interaksi parameter fisiologis yang kompleks menjadi pola prediktif dan stratifikasi risiko yang lebih terstruktur, sekaligus melatih evaluasi kritis terhadap potensi serta keterbatasan model kecerdasan buatan dalam studi sistem hayati dan kesehatan.

## Dataset
Dataset diakses menggunakan `fetch_openml("diabetes", version=1)` melalui pustaka `scikit-learn`.
- Tautan sumber: (https://www.openml.org/search?type=data&sort=runs&id=37&status=active)

## Tahapan & Metode
1. Identifikasi masalah
2. Pengumpulan data
3. Data understanding
4. Preprocessing (jelaskan langkah yang diterapkan maupun yang tidak diterapkan beserta alasannya)
5. Pemodelan AI (metode & pustaka yang digunakan)
6. Evaluasi
7. Interpretasi biologis

## Cara Menjalankan
1. Buka file `.ipynb` di Google Colab atau Jupyter Notebook.
2. Jalankan sel secara berurutan dari atas ke bawah.

## Hasil Utama & Keterbatasan
Pendekatan supervised dan unsupervised machine learning pada dataset diabetes (version=1) berhasil memetakan pola hubungan antarparameter klinis-biologis terhadap status diabetes sekaligus mengelompokkan profil individu berdasarkan kemiripan karakteristik metaboliknya, di mana indikator terkait regulasi glukosa darah dan kondisi metabolik-antropometrik tampil sebagai pembeda utama antar kelompok. Secara biologis, temuan ini sejalan dengan mekanisme patofisiologi diabetes melitus yang melibatkan gangguan homeostasis glukosa, resistensi insulin, serta variasi profil metabolik dan antropometrik antarindividu, sehingga pemodelan multivariat lebih informatif dibanding meninjau satu variabel secara terpisah. Meskipun demikian, model memiliki keterbatasan berupa ketergantungan pada karakteristik populasi dan fitur yang tersedia pada dataset diabetes (version=1)—termasuk adanya nilai nol pada beberapa fitur klinis dan tidak tercakupnya faktor genetik, riwayat keluarga, pola makan, atau biomarker molekuler yang lebih rinci—sehingga hasil evaluasi dan klasterisasi bersifat eksploratif-prediktif dan belum dapat dijadikan dasar diagnosis klinis tunggal tanpa validasi lanjutan.
