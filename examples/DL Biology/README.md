# Perbandingan Fungsi Aktivasi pada Neural Network

Hasil percobaan memperlihatkan pengaruh fungsi aktivasi **linear (none), sigmoid, tanh, dan ReLU** terhadap proses pembelajaran jaringan saraf sederhana untuk klasifikasi data berbentuk dua lingkaran (*concentric circles*).

## Tujuan
1. Membuat dataset klasifikasi dua kelas berbentuk lingkaran.
2. Membangun *multilayer perceptron* (MLP) sederhana menggunakan NumPy.
3. Melatih empat jaringan dengan fungsi aktivasi tersembunyi yang berbeda.
4. Mengamati perubahan *decision boundary* dan akurasi selama pelatihan melalui animasi.

**Catatan:** Lapisan keluaran menggunakan sigmoid untuk seluruh model karena tugasnya adalah klasifikasi biner. Fungsi aktivasi yang dibandingkan digunakan pada dua lapisan tersembunyi.

Code secara lengkap dapat diakses melalui [Percobaan Fungsi Aktivasi Deep Learning](https://colab.research.google.com/drive/1wXDyuF9v624NtSrSrYmKbvmeIltgPmLN?usp=sharing)

## Perbandingan Hasil Eksperimen

Berdasarkan hasil eksperimen, diperoleh akurasi sebagai berikut:

| Activation Function | Accuracy |
|---|---:|
| None (Linear) | 35.6% |
| Sigmoid | 43.6% |
| Tanh | 96.4% |
| ReLU | 96.8% |

Hasil tersebut menunjukkan bahwa pemilihan activation function berpengaruh terhadap kemampuan neural network dalam mempelajari pola data.

<img width="1400" height="380" alt="image" src="https://github.com/user-attachments/assets/5aedca9c-8807-4d9e-b0af-32b350993a1b" />

## Analisis Hasil

1. **None (Linear):** Tanpa activation function, neural network hanya melakukan transformasi linear. Akibatnya, model kesulitan mempelajari pola non-linear pada dataset concentric circles.

2. **Sigmoid:** Menghasilkan akurasi lebih baik dibandingkan tanpa activation function, tetapi proses pembelajaran dapat terhambat oleh vanishing gradient, terutama ketika nilai input berada pada daerah saturasi.

3. **Tanh:** Menghasilkan akurasi yang tinggi karena mampu merepresentasikan hubungan non-linear. Fungsi ini memiliki rentang output antara -1 hingga 1 dan berpusat di sekitar nol (zero-centered).

4. **ReLU:** Menghasilkan akurasi tertinggi pada eksperimen ini. ReLU memiliki perhitungan sederhana dan membantu mengurangi masalah vanishing gradient pada sisi positif fungsi aktivasi.

## Kesimpulan

Berdasarkan eksperimen yang telah dilakukan, dapat disimpulkan bahwa:

- Activation function berperan penting dalam membentuk kemampuan neural network untuk mempelajari pola non-linear.
- Tanpa activation function, neural network dengan beberapa hidden layer tetap setara dengan transformasi linear.
- Sigmoid dapat mengalami vanishing gradient sehingga proses pembelajaran menjadi kurang optimal.
- Tanh dan ReLU menunjukkan performa yang lebih baik pada dataset concentric circles.
- ReLU memperoleh akurasi tertinggi sebesar 96.8%, sedangkan Tanh memperoleh akurasi sebesar 96.4%.

Perlu diperhatikan bahwa hasil eksperimen ini bergantung pada dataset, arsitektur neural network, parameter pelatihan, dan proses optimasi yang digunakan. Oleh karena itu, ReLU tidak selalu menghasilkan performa terbaik untuk semua permasalahan.
