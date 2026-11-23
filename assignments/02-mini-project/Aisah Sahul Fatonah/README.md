# Klasifikasi & Clustering Spesies Penguin (Palmer Penguins) 

Proyek akhir mata kuliah **Machine Learning untuk Biologi** oleh **Kelompok 6**. Proyek ini menerapkan pendekatan *supervised* dan *unsupervised learning* untuk menganalisis perbedaan morfologi tiga spesies penguin (*Adelie*, *Chinstrap*, dan *Gentoo*) di Kepulauan Palmer, Antartika.

---

##  Anggota Kelompok 6
* **Aisah Sahul Fathonah**
* **Laksita Widia Pratiwi**
* **Dian Safa Sofrani**

---

##  Tujuan Proyek
1. **Supervised Learning (Klasifikasi):** Membangun model untuk memprediksi spesies penguin berdasarkan fitur morfologi fisik (*culmen length*, *culmen depth*, *flipper length*, dan *body mass*) serta lokasi pulau, menggunakan metrik **Macro F1-Score** untuk menangani ketidakseimbangan jumlah sampel antarspesies.
2. **Unsupervised Learning (Clustering):** Mengelompokkan data penguin tanpa label untuk mengevaluasi sejauh mana struktur alami morfologi sesuai dengan taksonomi spesies aslinya.

---

##  Dataset & Eksplorasi Data (EDA)
* **Sumber Dataset:** Palmer Penguins (`penguins_size.csv`), berisi 344 observasi awal yang dibersihkan menjadi **333 sampel valid** setelah menghapus data dengan nilai kosong (*missing values*).
* **Distribusi Spesies:** *Adelie* (146), *Gentoo* (119), dan *Chinstrap* (68).
* **Temuan Utama EDA:**
  * Terdapat pemisahan klaster yang cukup jelas pada kombinasi fitur morfologi (terutama antara spesies *Gentoo* dengan dua spesies lainnya).
  * Korelasi positif paling kuat ditemukan antara **panjang sirip (*flipper length*)** dan **massa tubuh (*body mass*)** sebesar **0.87**.

---

##  Teknologi & Pustaka (*Libraries*)
* **Bahasa:** Python 3
* **Manipulasi Data:** `pandas`, `numpy`
* **Visualisasi:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn`

---

## 📁 Struktur Repositori
```text
├── Kelompok6.ipynb       # Notebook utama (Pendefinisian Masalah, EDA, & Pemodelan)
├── penguins_size.csv     # Dataset Palmer Penguins
└── README.md             # Dokumentasi proyek
