# Lecture Note: Deep Learning untuk Data Sains dan AI dalam Biologi

**Mata Kuliah:** Data Sains dan AI untuk Biologi\
**Topik:** Deep Learning, Artificial Neural Network, Forward
Propagation, dan Backpropagation\
**Bahasa pemrograman:** Python\
**Library utama:** NumPy, Matplotlib, TensorFlow/Keras, Scikit-learn

## 1. Capaian Pembelajaran

Setelah mempelajari materi ini, mahasiswa diharapkan mampu:

1.  Menjelaskan konsep dan karakteristik deep learning serta
    perbedaannya dengan machine learning konvensional.
2.  Menjelaskan struktur Artificial Neural Network (ANN), fungsi
    aktivasi, fungsi loss, dan proses optimasi.
3.  Menghitung proses forward propagation dan backward propagation pada
    jaringan saraf sederhana.

## 2. Pengantar Deep Learning

Deep learning merupakan bagian dari machine learning yang menggunakan
jaringan saraf tiruan dengan banyak lapisan (*multiple layers*) untuk
mempelajari representasi atau pola dari data. Model deep learning dapat
mempelajari fitur secara bertingkat, mulai dari pola sederhana hingga
pola yang lebih kompleks.

Konsep deep learning sebenarnya sudah ada sejak sekitar seratus tahun yang lalu,
namun baru benar-benar berkembang pesat dalam beberapa tahun terakhir.
Untuk memahami alasannya, bayangkan sebuah grafik yang menunjukkan hubungan antara
jumlah data dan kinerja model. Pada grafik tersebut, sumbu-x mewakili jumlah data,
sedangkan sumbu-y menunjukkan tingkat kinerja (performance). Jika kita menggambarnya,
kurva tersebut akan terlihat menanjak (naik) seiring bertambahnya data, tetapi setelah
mencapai jumlah data yang sangat besar, kurva akan mulai mendatar. Artinya, pada
awalnya, penambahan data sangat membantu meningkatkan kinerja model. Namun, setelah
titik tertentu, penambahan data lebih banyak tidak lagi memberikan peningkatan besar,
kinerja model cenderung stabil atau konstan. Hal inilah yang menggambarkan bagaimana
perkembangan deep learning bergantung pada ketersediaan data dan kapasitas model
untuk memanfaatkannya secara efektif.

Contoh penerapan deep learning dalam biologi:

-   Klasifikasi citra sel normal dan sel abnormal.
-   Klasifikasi jenis jaringan berdasarkan citra histopatologi.
-   Identifikasi penyakit tanaman berdasarkan citra daun.
-   Klasifikasi spesies berdasarkan citra organisme.
-   Analisis citra mikroskopis untuk mengelompokkan mikroorganisme.
-   Analisis citra medis dan biologis lainnya.

### 2.1 Machine Learning dan Deep Learning

Pada machine learning konvensional, proses ekstraksi fitur sering
dilakukan secara manual atau menggunakan metode tertentu sebelum data
diberikan kepada algoritma klasifikasi.

Alur umum machine learning konvensional:

1.  Mengumpulkan data.
2.  Melakukan preprocessing.
3.  Mengekstraksi fitur (*feature extraction*).
4.  Melatih algoritma klasifikasi.
5.  Mengevaluasi model.

Pada deep learning, ekstraksi fitur dan klasifikasi dapat dipelajari
secara bersamaan oleh jaringan saraf.

Alur umum deep learning:

1.  Mengumpulkan data.
2.  Melakukan preprocessing.
3.  Memasukkan data ke jaringan saraf.
4.  Model mempelajari fitur dan hubungan antara fitur dengan label.
5.  Mengevaluasi model.

Perlu diperhatikan bahwa deep learning tidak selalu lebih baik daripada
machine learning konvensional. Kinerja model bergantung pada jumlah dan
kualitas data, kompleksitas masalah, representasi data, arsitektur,
serta strategi pelatihan.

## 3. Artificial Neural Network (ANN)

Artificial Neural Network (ANN) adalah model komputasi yang terinspirasi
dari cara neuron saling terhubung. ANN tersusun atas neuron-neuron
buatan yang menerima masukan, mengalikannya dengan bobot, menambahkan
bias, kemudian menerapkan fungsi aktivasi.

<img width="1120" height="542" alt="image" src="https://github.com/user-attachments/assets/87aef117-36ac-4efe-bf2b-c4707155e08e" />

Jaringan saraf sederhana umumnya terdiri dari:

-   **Input layer:** menerima fitur dari data.
-   **Hidden layer:** mempelajari representasi atau pola dari data.
-   **Output layer:** menghasilkan prediksi.

Ilustrasi konseptual:

<img width="1382" height="486" alt="image" src="https://github.com/user-attachments/assets/083df47d-82cd-4722-a585-0d22f6e5a03c" />

Setiap koneksi memiliki bobot (*weight*). Neuron juga memiliki bias
(*bias*).

### 3.1 Perhitungan pada Neuron

Untuk sebuah neuron dengan beberapa masukan, nilai sebelum aktivasi
dihitung dengan:

### Perhitungan pada Neuron

Nilai sebelum fungsi aktivasi dihitung menggunakan persamaan:

$$
z = \sum_{i=1}^{n} w_i x_i + b
$$

Keterangan:

- $$x_i$$: Nilai masukan ke-i (*input*).
- $$w_i$$: Bobot untuk masukan ke-i (*weight*).
- $$b$$: Bias.
- $$n$$: Jumlah masukan.
- $$z$$: Nilai sebelum fungsi aktivasi (*pre-activation value*).

Keluaran neuron diperoleh dengan menerapkan fungsi aktivasi:

$$
a = f(z)
$$

**Keterangan:**

- `f(z)`: Fungsi aktivasi.
- `a`: Keluaran neuron (*output*).
- `z`: Nilai sebelum fungsi aktivasi.

Keluaran neuron diperoleh dengan menerapkan fungsi aktivasi:

$$
a = f(z)
$$

Fungsi aktivasi diperlukan agar jaringan dapat mempelajari hubungan
nonlinier. Tanpa fungsi aktivasi nonlinier, beberapa lapisan linear yang
ditumpuk tetap setara dengan satu transformasi linear.

### 3.2 Fungsi Aktivasi

Beberapa fungsi aktivasi yang umum digunakan adalah:

**a. Sigmoid**

$$
\sigma(z) = \frac{1}{1+e^{-z}}
$$

Nilai keluaran sigmoid berada pada rentang 0 sampai 1. Fungsi ini dapat
digunakan pada output klasifikasi biner, terutama ketika keluaran
ditafsirkan sebagai probabilitas kelas positif.

Turunan sigmoid:

$$
\sigma'(z) = \sigma(z)(1-\sigma(z))
$$

**b. ReLU (Rectified Linear Unit)**

$$
ReLU(z) = \max(0,z)
$$

ReLU banyak digunakan pada hidden layer karena sederhana dan membantu
mengurangi masalah gradien yang sangat kecil pada wilayah positif.

Turunan ReLU:

$$
ReLU'(z) =
\begin{cases}
0, & z < 0 \\
1, & z > 0
\end{cases}
$$

Pada $z=0$, implementasi menggunakan subgradien tertentu.

**c. Softmax**

Untuk klasifikasi multikelas, softmax mengubah sejumlah skor menjadi
distribusi probabilitas:

$$
softmax(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K}e^{z_j}}
$$

Jumlah seluruh probabilitas keluaran softmax adalah 1.

Grafik fungsi aktivasi

<p align="center">
    <img width="553" height="370,5" alt="image" src="https://github.com/user-attachments/assets/78f14635-3a15-4bbc-9581-1da7d032a1ca" />
</p>

### 3.3 Fungsi Loss

Fungsi loss mengukur selisih antara prediksi model dan label sebenarnya.
Nilai loss menjadi dasar perhitungan gradien untuk memperbarui bobot.

Untuk klasifikasi biner dengan keluaran sigmoid, salah satu fungsi yang
umum digunakan adalah *Binary Cross-Entropy*:

$$
L(y,\hat{y}) =
-\left[y\log(\hat{y})+(1-y)\log(1-\hat{y})\right]
$$

Keterangan:

-   $y$: label sebenarnya, bernilai 0 atau 1.
-   $\hat{y}$: probabilitas prediksi untuk kelas 1.
-   $L$: nilai loss.

Untuk klasifikasi multikelas dengan softmax, dapat digunakan
*Categorical Cross-Entropy* atau *Sparse Categorical Cross-Entropy*,
bergantung pada format label.

## 4. Proses Pembelajaran Neural Network

Proses pembelajaran jaringan saraf terdiri dari dua tahap utama dalam
setiap iterasi pelatihan:

1.  **Forward propagation:** menghitung keluaran jaringan berdasarkan
    masukan, bobot, bias, dan fungsi aktivasi.
2.  **Backward propagation:** menghitung gradien loss terhadap bobot dan
    bias menggunakan aturan rantai (*chain rule*).

Setelah gradien diperoleh, optimizer memperbarui parameter jaringan.
Proses ini diulang selama beberapa iterasi atau *epoch*.

<img width="738" height="414" alt="image" src="https://github.com/user-attachments/assets/006b04ea-4629-4fed-90e3-4e49ba9bbcf7" />

### 4.1 Forward Propagation

Forward propagation adalah proses mengalirkan data dari input layer
menuju output layer.

Misalkan jaringan memiliki:

-   Dua input: $x_1$ dan $x_2$.
-   Dua neuron hidden: $h_1$ dan $h_2$.
-   Satu neuron output: $\hat{y}$.
-   Fungsi aktivasi sigmoid pada hidden layer dan output layer.

Persamaan hidden layer:

$$
z_{h1}=w_1x_1+w_2x_2+b_1
$$

$$
h_1=\sigma(z_{h1})
$$

$$
z_{h2}=w_3x_1+w_4x_2+b_2
$$

$$
h_2=\sigma(z_{h2})
$$

Persamaan output layer:

$$
z_o=w_5h_1+w_6h_2+b_3
$$

$$
\hat{y}=\sigma(z_o)
$$

Nilai $\hat{y}$ merupakan prediksi jaringan untuk satu data masukan.

### 4.2 Backward Propagation

Backward propagation adalah proses menghitung gradien loss terhadap
setiap parameter jaringan. Gradien menunjukkan arah dan besarnya
perubahan loss jika suatu parameter diubah.

Gradien dihitung dari output menuju lapisan sebelumnya menggunakan
*chain rule*.

Untuk klasifikasi biner dengan sigmoid pada output dan Binary
Cross-Entropy, gradien terhadap nilai sebelum aktivasi output memiliki
bentuk sederhana:

$$
\delta_o = \frac{\partial L}{\partial z_o} = \hat{y}-y
$$

Gradien bobot output:

$$
\frac{\partial L}{\partial w_5} = \delta_o h_1
$$

$$
\frac{\partial L}{\partial w_6} = \delta_o h_2
$$

Gradien bias output:

$$
\frac{\partial L}{\partial b_3} = \delta_o
$$

Gradien yang diteruskan ke hidden layer pertama:

$$
\delta_{h1} = \delta_o w_5 h_1(1-h_1)
$$

Untuk hidden layer kedua:

$$
\delta_{h2} = \delta_o w_6 h_2(1-h_2)
$$

Gradien bobot hidden layer pertama:

$$
\frac{\partial L}{\partial w_1}=\delta_{h1}x_1
$$

$$
\frac{\partial L}{\partial w_2}=\delta_{h1}x_2
$$

Gradien bobot hidden layer kedua:

$$
\frac{\partial L}{\partial w_3}=\delta_{h2}x_1
$$

$$
\frac{\partial L}{\partial w_4}=\delta_{h2}x_2
$$

Gradien bias hidden:

$$
\frac{\partial L}{\partial b_1}=\delta_{h1}
$$

$$
\frac{\partial L}{\partial b_2}=\delta_{h2}
$$

### 4.3 Pembaruan Bobot dengan Gradient Descent

Setelah gradien dihitung, bobot dan bias diperbarui menggunakan
*gradient descent*:

$$
w_{\text{baru}}=w_{\text{lama}}-\eta\frac{\partial L}{\partial w}
$$

$$
b_{\text{baru}}=b_{\text{lama}}-\eta\frac{\partial L}{\partial b}
$$

Keterangan:

-   $\eta$: *learning rate*.
-   $\frac{\partial L}{\partial w}$: gradien loss terhadap bobot.
-   $\frac{\partial L}{\partial b}$: gradien loss terhadap bias.

Learning rate mengatur besar langkah pembaruan parameter. Nilai terlalu
besar dapat membuat proses optimasi tidak stabil, sedangkan nilai
terlalu kecil dapat membuat pembelajaran berlangsung lambat.

### 4.4 Contoh Perhitungan Forward dan Backward Propagation

Gunakan jaringan dengan dua input, dua neuron hidden, dan satu output.
Seluruh neuron menggunakan sigmoid.

Data masukan dan target:

$$
x_1=1,\quad x_2=0,\quad y=1
$$

Parameter awal:

$$
w_1=0.1,\quad w_2=0.2
$$

$$
w_3=0.3,\quad w_4=0.4
$$

$$
w_5=0.5,\quad w_6=0.6
$$

$$
b_1=0,\quad b_2=0,\quad b_3=0
$$

Learning rate:

$$
\eta=0.1
$$

**Langkah 1. Forward propagation pada hidden layer**

Neuron hidden pertama:

$$
z_{h1}=(0.1)(1)+(0.2)(0)+0=0.1
$$

$$
h_1=\sigma(0.1)=0.52498
$$

Neuron hidden kedua:

$$
z_{h2}=(0.3)(1)+(0.4)(0)+0=0.3
$$

$$
h_2=\sigma(0.3)=0.57444
$$

**Langkah 2. Forward propagation pada output layer**

$$
z_o=(0.5)(0.52498)+(0.6)(0.57444)+0
$$

$$
z_o=0.60770
$$

$$
\hat{y}=\sigma(0.60770)\approx0.64740
$$

Dengan demikian, prediksi jaringan untuk data tersebut adalah sekitar
0.64740.

**Langkah 3. Menghitung loss**

Karena target $y=1$, Binary Cross-Entropy menjadi:

$$
L=-\log(\hat{y})
$$

$$
L=-\log(0.64740)\approx0.435
$$

**Langkah 4. Backward propagation pada output**

$$
\delta_o=\hat{y}-y
$$

$$
\delta_o=0.64740-1=-0.35260
$$

Gradien bobot output:

$$
\frac{\partial L}{\partial w_5}=(-0.35260)(0.52498)\approx-0.18511
$$

$$
\frac{\partial L}{\partial w_6}=(-0.35260)(0.57444)\approx-0.20258
$$

Gradien bias output:

$$
\frac{\partial L}{\partial b_3}=-0.35260
$$

**Langkah 5. Backward propagation pada hidden layer**

Untuk neuron hidden pertama:

$$
\delta_{h1}=\delta_o w_5 h_1(1-h_1)
$$

$$
\delta_{h1}=(-0.35260)(0.5)(0.52498)(1-0.52498)
$$

$$
\delta_{h1}\approx-0.04402
$$

Untuk neuron hidden kedua:

$$
\delta_{h2}=\delta_o w_6 h_2(1-h_2)
$$

$$
\delta_{h2}=(-0.35260)(0.6)(0.57444)(1-0.57444)
$$

$$
\delta_{h2}\approx-0.05198
$$

Karena $x_1=1$ dan $x_2=0$, gradien bobot hidden layer adalah:

$$
\frac{\partial L}{\partial w_1}=-0.04402,\quad
\frac{\partial L}{\partial w_2}=0
$$

$$
\frac{\partial L}{\partial w_3}=-0.05198,\quad
\frac{\partial L}{\partial w_4}=0
$$

Gradien bias hidden:

$$
\frac{\partial L}{\partial b_1}=-0.04402
$$

$$
\frac{\partial L}{\partial b_2}=-0.05198
$$

**Langkah 6. Memperbarui parameter**

Contoh pembaruan bobot $w_5$:

$$
w_5^{baru}=0.5-(0.1)(-0.18511)
$$

$$
w_5^{baru}\approx0.51851
$$

Contoh pembaruan bobot $w_1$:

$$
w_1^{baru}=0.1-(0.1)(-0.04402)
$$

$$
w_1^{baru}\approx0.10440
$$

Contoh pembaruan bias output:

$$
b_3^{baru}=0-(0.1)(-0.35260)=0.03526
$$

Parameter lainnya diperbarui dengan cara yang sama. Setelah parameter
diperbarui, data kembali dilewatkan melalui forward propagation. Nilai
loss kemudian dihitung kembali. Proses ini diulang agar model semakin
mendekati target.

Catatan: angka pada perhitungan manual dibulatkan sehingga hasil akhir
dapat sedikit berbeda jika dihitung menggunakan presisi penuh.

### 4.5 Implementasi Forward dan Backward Propagation Menggunakan NumPy

Kode berikut mengimplementasikan satu jaringan saraf sederhana tanpa
menggunakan TensorFlow atau PyTorch. Tujuannya agar mahasiswa dapat
melihat bahwa forward propagation, perhitungan loss, backward
propagation, dan pembaruan bobot dapat dilakukan secara eksplisit.

``` python

import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# 1. INISIALISASI
# ==================================================

np.set_printoptions(precision=5, suppress=True)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Data: dua fitur dan satu target biner
X = np.array([[1.0, 0.0]])
y = np.array([[1.0]])

# Inisialisasi bobot dan bias
W1 = np.array([
    [0.1, 0.2],
    [0.3, 0.4]
])

b1 = np.array([[0.0, 0.0]])

W2 = np.array([
    [0.5],
    [0.6]
])

b2 = np.array([[0.0]])

learning_rate = 0.1
epochs = 1000
epsilon = 1e-7


# ==================================================
# 2. MENAMPUNG HASIL TRAINING
# ==================================================

loss_history = []
accuracy_history = []


# ==================================================
# 3. TRAINING NEURAL NETWORK
# ==================================================

for epoch in range(epochs):

    # ----------------------------------------------
    # FORWARD PROPAGATION
    # ----------------------------------------------

    z1 = X @ W1 + b1
    a1 = sigmoid(z1)

    z2 = a1 @ W2 + b2
    y_pred = sigmoid(z2)

    # ----------------------------------------------
    # LOSS CALCULATION
    # Binary Cross-Entropy
    # ----------------------------------------------

    loss = -np.mean(
        y * np.log(y_pred + epsilon) +
        (1 - y) * np.log(1 - y_pred + epsilon)
    )

    # ----------------------------------------------
    # ACCURACY CALCULATION
    # ----------------------------------------------

    y_pred_class = (y_pred >= 0.5).astype(int)
    accuracy = np.mean(y_pred_class == y)

    loss_history.append(loss)
    accuracy_history.append(accuracy)

    # ----------------------------------------------
    # BACKWARD PROPAGATION
    # ----------------------------------------------

    # Output layer
    dz2 = y_pred - y

    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)

    # Hidden layer
    dz1 = (dz2 @ W2.T) * a1 * (1 - a1)

    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # ----------------------------------------------
    # UPDATE PARAMETERS
    # ----------------------------------------------

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # ----------------------------------------------
    # PRINT TRAINING PROGRESS
    # ----------------------------------------------

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch + 1:4d} | "
            f"Loss: {loss:.6f} | "
            f"Accuracy: {accuracy:.2%}"
        )


# ==================================================
# 4. EVALUASI MODEL
# ==================================================

# Forward propagation menggunakan parameter akhir
z1 = X @ W1 + b1
a1 = sigmoid(z1)

z2 = a1 @ W2 + b2
y_pred = sigmoid(z2)

y_pred_class = (y_pred >= 0.5).astype(int)

print("\n========== HASIL TRAINING ==========")

print("Hidden output:\n", a1)
print("Prediksi probabilitas:\n", y_pred)
print("Target aktual:\n", y)
print("Kelas prediksi:\n", y_pred_class)

print("\n========== PARAMETER AKHIR ==========")

print("W1:\n", W1)
print("b1:\n", b1)

print("W2:\n", W2)
print("b2:\n", b2)

print("\nFinal Loss:", loss_history[-1])
print("Final Accuracy:", f"{accuracy_history[-1]:.2%}")


# ==================================================
# 5. VISUALISASI LOSS DAN ACCURACY
# ==================================================

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# Training Loss
ax[0].plot(
    range(1, epochs + 1),
    loss_history,
    color='red',
    linewidth=2
)

ax[0].set_title("Training Loss")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Binary Cross-Entropy Loss")
ax[0].grid(True, alpha=0.3)

# Training Accuracy
ax[1].plot(
    range(1, epochs + 1),
    accuracy_history,
    color='blue',
    linewidth=2
)

ax[1].set_title("Training Accuracy")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Accuracy")
ax[1].set_ylim(-0.05, 1.05)
ax[1].grid(True, alpha=0.3)

plt.suptitle("Neural Network Training Performance")
plt.tight_layout()
plt.show()
```
Grafik akurasi dan loss training seperti pada gambar

<img width="1200" height="400" alt="Figure_11" src="https://github.com/user-attachments/assets/1e211709-a405-40c5-8ad8-b821bc493098" />

Beberapa hal yang perlu diperhatikan:
1. Loss: menunjukkan besarnya kesalahan prediksi. Idealnya menurun selama training.
2. Accuracy: menunjukkan proporsi prediksi yang benar. Idealnya meningkat.
3. Epoch: satu putaran training. Pada kode ini satu epoch terdiri dari satu kali forward propagation, backpropagation, dan update parameter.
4. Kode ini menggunakan satu sampel data saja. Oleh karena itu, accuracy hanya akan bernilai 0% atau 100%.

**Eksperimen:** menggunakan dataset XOR (4 sampel)

### Dataset XOR

| Input X1 | Input X2 | Target Y |
|:--------:|:--------:|:--------:|
| 0        | 0        | 0        |
| 0        | 1        | 1        |
| 1        | 0        | 1        |
| 1        | 1        | 0        |

Secara matematis, XOR menghasilkan kelas 1 apabila kedua input berbeda.

### Arsitektur Neural Network

Arsitektur yang digunakan adalah **2–4–1 (Input–Hidden–Output)**.

| Layer | Jumlah Neuron | Fungsi Aktivasi | Keterangan |
|:-----:|:-------------:|:---------------:|:-----------|
| Input | 2 | - | X1, X2 |
| Hidden | 4 | Sigmoid | Hidden layer |
| Output | 1 | Sigmoid | Prediksi kelas XOR |

``` python

import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# 1. INISIALISASI
# ==================================================

np.set_printoptions(precision=5, suppress=True)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Dataset XOR (4 sampel)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)


# Inisialisasi parameter
np.random.seed(42)

input_size = 2
hidden_size = 4
output_size = 1

W1 = np.random.randn(input_size, hidden_size) * 0.5
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * 0.5
b2 = np.zeros((1, output_size))

learning_rate = 1.0
epochs = 10000
epsilon = 1e-7


# ==================================================
# 2. MENAMPUNG HASIL TRAINING
# ==================================================

loss_history = []
accuracy_history = []


# ==================================================
# 3. TRAINING NEURAL NETWORK
# ==================================================

for epoch in range(epochs):

    # ----------------------------------------------
    # FORWARD PROPAGATION
    # ----------------------------------------------

    z1 = X @ W1 + b1
    a1 = sigmoid(z1)

    z2 = a1 @ W2 + b2
    y_pred = sigmoid(z2)

    # ----------------------------------------------
    # LOSS CALCULATION
    # ----------------------------------------------

    loss = -np.mean(
        y * np.log(y_pred + epsilon) +
        (1 - y) * np.log(1 - y_pred + epsilon)
    )

    # ----------------------------------------------
    # ACCURACY CALCULATION
    # ----------------------------------------------

    y_pred_class = (y_pred >= 0.5).astype(int)
    accuracy = np.mean(y_pred_class == y)

    loss_history.append(loss)
    accuracy_history.append(accuracy)

    # ----------------------------------------------
    # BACKWARD PROPAGATION
    # ----------------------------------------------

    # Output layer
    dz2 = (y_pred - y) / len(X)

    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)

    # Hidden layer
    dz1 = (dz2 @ W2.T) * a1 * (1 - a1)

    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # ----------------------------------------------
    # UPDATE PARAMETERS
    # ----------------------------------------------

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # ----------------------------------------------
    # MONITORING
    # ----------------------------------------------

    if (epoch + 1) % 1000 == 0:
        print(
            f"Epoch {epoch+1:5d} | "
            f"Loss: {loss:.6f} | "
            f"Accuracy: {accuracy:.2%}"
        )


# ==================================================
# 4. EVALUASI MODEL
# ==================================================

z1 = X @ W1 + b1
a1 = sigmoid(z1)

z2 = a1 @ W2 + b2
y_pred = sigmoid(z2)

y_pred_class = (y_pred >= 0.5).astype(int)

final_accuracy = np.mean(y_pred_class == y)

print("\n========== HASIL PREDIKSI XOR ==========")

for i in range(len(X)):
    print(
        f"Input: {X[i]} | "
        f"Target: {int(y[i, 0])} | "
        f"Probabilitas: {y_pred[i, 0]:.5f} | "
        f"Prediksi: {y_pred_class[i, 0]}"
    )

print("\nFinal Loss:", loss_history[-1])
print("Final Accuracy:", f"{final_accuracy:.2%}")


# ==================================================
# 5. VISUALISASI LOSS DAN ACCURACY
# ==================================================

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# Training Loss
ax[0].plot(loss_history, color='red', linewidth=2)
ax[0].set_title("Training Loss")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Binary Cross-Entropy Loss")
ax[0].grid(True, alpha=0.3)

# Training Accuracy
ax[1].plot(accuracy_history, color='blue', linewidth=2)
ax[1].set_title("Training Accuracy")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Accuracy")
ax[1].set_ylim(-0.05, 1.05)
ax[1].grid(True, alpha=0.3)

plt.suptitle("Neural Network XOR Training")
plt.tight_layout()
plt.show()


# ==================================================
# 6. VISUALISASI DECISION BOUNDARY
# ==================================================

gx, gy = np.meshgrid(
    np.linspace(-0.5, 1.5, 200),
    np.linspace(-0.5, 1.5, 200)
)

grid = np.c_[gx.ravel(), gy.ravel()]

# Forward propagation pada grid
Z1 = grid @ W1 + b1
A1 = sigmoid(Z1)

Z2 = A1 @ W2 + b2
P = sigmoid(Z2).reshape(gx.shape)

plt.figure(figsize=(6, 5))

plt.contourf(
    gx, gy, P,
    levels=np.linspace(0, 1, 21),
    cmap='RdBu_r',
    alpha=0.7
)

plt.contour(
    gx, gy, P,
    levels=[0.5],
    colors='black',
    linewidths=2
)

plt.scatter(
    X[:, 0], X[:, 1],
    c=y.ravel(),
    cmap='RdBu_r',
    s=180,
    edgecolors='black',
    linewidths=1.5
)

for i in range(len(X)):
    plt.annotate(
        f"Target: {int(y[i, 0])}",
        (X[i, 0], X[i, 1]),
        xytext=(7, 8),
        textcoords='offset points'
    )

plt.title("XOR Decision Boundary")
plt.xlabel("Input X1")
plt.ylabel("Input X2")
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.colorbar(label="Prediction Probability")
plt.grid(alpha=0.2)
plt.show()
```

### Interpretasi Hasil Training

<img width="1200" height="400" alt="Figure_12" src="https://github.com/user-attachments/assets/7ba94b04-1b87-42f1-8ef5-1f5cf63b63bf" />

Berdasarkan grafik Training Loss dan Training Accuracy, dapat diamati bahwa model Neural Network dengan arsitektur 2–4–1 berhasil mempelajari pola klasifikasi pada dataset XOR.

#### 1. Training Loss

Grafik Training Loss menunjukkan perubahan nilai kesalahan prediksi model selama 10.000 epoch menggunakan Binary Cross-Entropy.

- Pada awal training, nilai loss berada di sekitar 0.70.
- Pada sekitar epoch 500–1.000, nilai loss mengalami penurunan yang sangat signifikan.
- Setelah epoch 1.000, nilai loss terus menurun secara bertahap hingga mendekati 0.
- Pada akhir training, nilai loss mendekati 0, yang menunjukkan bahwa prediksi model semakin mendekati target aktual.

Penurunan nilai loss menunjukkan bahwa proses optimasi menggunakan Gradient Descent berhasil memperbarui bobot dan bias untuk mengurangi kesalahan prediksi.

#### 2. Training Accuracy

Grafik Training Accuracy menunjukkan tingkat ketepatan model dalam mengklasifikasikan empat sampel pada dataset XOR.

- Pada awal training, akurasi mengalami perubahan yang cukup fluktuatif.
- Akurasi meningkat secara bertahap dari 25%, 50%, hingga 75%.
- Sekitar epoch 600–1.000, akurasi mencapai 100%.
- Setelah mencapai 100%, akurasi tetap stabil hingga epoch terakhir.

Akurasi 100% menunjukkan bahwa model berhasil mengklasifikasikan seluruh empat sampel dataset XOR dengan benar.

#### 3. Analisis Hasil

Berdasarkan kedua grafik, dapat disimpulkan bahwa:

1. Model berhasil mempelajari pola nonlinier pada dataset XOR.
2. Nilai Training Loss yang semakin mendekati 0 menunjukkan berkurangnya kesalahan prediksi.
3. Training Accuracy mencapai 100%, yang berarti seluruh sampel training berhasil diklasifikasikan dengan benar.
4. Stabilnya akurasi setelah mencapai 100% menunjukkan bahwa model telah berhasil mempelajari pola XOR pada dataset yang digunakan.
5. Penggunaan hidden layer dengan fungsi aktivasi Sigmoid memungkinkan neural network mempelajari pola yang tidak dapat dipisahkan menggunakan satu garis lurus.

#### 4. Kesimpulan

Eksperimen menunjukkan bahwa Multi-Layer Perceptron (MLP) dengan arsitektur 2–4–1 mampu menyelesaikan permasalahan klasifikasi XOR melalui proses Forward Propagation, Backpropagation, dan Gradient Descent.

Hasil ini memperlihatkan bahwa hidden layer dan fungsi aktivasi nonlinier memiliki peran penting dalam kemampuan neural network mempelajari hubungan yang kompleks antara input dan target.

**Catatan:** Akurasi 100% pada eksperimen ini diperoleh dari empat sampel training. Hasil tersebut menunjukkan keberhasilan model mempelajari dataset XOR, tetapi belum cukup untuk membuktikan kemampuan generalisasi terhadap data baru.

## 5. Referensi

1.  Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*.
    MIT Press. https://www.deeplearningbook.org/
2.  LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*,
    521, 436--444. https://doi.org/10.1038/nature14539
3.  Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet
    classification with deep convolutional neural networks. *Advances in
    Neural Information Processing Systems*.
4.  Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning
    Publications.
5.  TensorFlow. Image classification tutorial.
    https://www.tensorflow.org/tutorials/images/classification
6.  TensorFlow API. `tf.keras.utils.image_dataset_from_directory`.
    https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
7.  Scikit-learn. Model evaluation: classification metrics.
    https://scikit-learn.org/stable/modules/model_evaluation.html

