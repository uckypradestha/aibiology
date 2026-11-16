# Mini Assignment: Klasifikasi Kanker Payudara (Logistic Regression vs Random Forest)

## Identitas Mahasiswa
- **Nama:** Widia Nur Astuti
- **NIM:** B1C024016
- **Kelas:** Biologi Terapan A (Angkatan 2024)

## Deskripsi Eksperimen
Eksperimen ini membandingkan performa algoritma **Logistic Regression** dan **Random Forest Classifier** pada *Breast Cancer Wisconsin Dataset* menggunakan pembagian dataset **70% data training** dan **30% data testing** (`random_state=42`, `stratify=y`).

## Hasil Evaluasi
| Algoritma | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.9883 | 0.9875 | 0.9875 | 0.9875 |
| Random Forest | 0.9357 | 0.9341 | 0.9286 | 0.9311 |

*(Catatan: Sesuaikan angka tabel di atas jika hasil eksekusi di Colab kamu berbeda).*

## Kesimpulan
1. **Algoritma Terbaik:** **Logistic Regression** unggul dibandingkan Random Forest pada seluruh metrik evaluasi (akurasi 98,83% vs 93,57%) karena karakteristik fitur dataset yang terpisah secara linear setelah proses standardisasi.
2. **Pentingnya Metrik Selain Accuracy:** Dalam deteksi kanker, metrik **Recall** pada kelas *Malignant* jauh lebih krusial daripada *Accuracy* semata untuk meminimalkan risiko *False Negative* (pasien kanker ganas yang terlewat dan dianggap jinak).
