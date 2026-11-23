# Lecture Note: Convolutional Neural Network (CNN)

**Mata Kuliah:** Data Sains dan AI untuk Biologi\
**Topik:** Convolutional Neural Network\
**Bahasa pemrograman:** Python\
**Library utama:** NumPy, Matplotlib, TensorFlow/Keras, Scikit-learn

## 1. Capaian Pembelajaran

Setelah mempelajari materi ini, mahasiswa diharapkan mampu:

1.  Menjelaskan peran Convolutional Neural Network (CNN) dalam
    pengolahan citra.
2.  Membangun dan mengevaluasi model CNN sederhana untuk klasifikasi
    citra biologis.
3.  Menginterpretasikan hasil evaluasi model dengan accuracy, precision,
    recall, F1-score, dan confusion matrix

## 2. Pengantar

Computer Vision adalah bidang ilmu interdisipliner yang bertujuan untuk
membuat komputer memproses, menganalisis gambar dan video, serta mengekstrak
detail dengan cara yang sama seperti yang dilakukan oleh pikiran manusia.
Sebagian besar tugas Computer Vision berkaitan dengan arsitektur CNN,
karena dasar dari sebagian besar masalah adalah mengklasifikasikan gambar
ke dalam label yang sudah diketahui.

Computer vision task antara lain classification, classification + localization,
object detection, dan Segmentation seperti yang terlihat pada gambar.

<img width="1824" height="769" alt="image" src="https://github.com/user-attachments/assets/9b9d4978-9394-416c-a852-85c8989337d0" />

Convolutional Neural Network (CNN) adalah arsitektur deep learning yang
banyak digunakan untuk mengolah data berbentuk grid, terutama citra. CNN
memanfaatkan operasi konvolusi untuk mempelajari pola lokal pada citra.

CNN sesuai untuk klasifikasi citra biologis karena informasi penting
sering muncul sebagai pola spasial, seperti tekstur, bentuk sel,
struktur jaringan, atau karakteristik permukaan daun.

### 2.1 Struktur Umum CNN

Arsitektur CNN sederhana umumnya terdiri atas:

1.  **Input:** menerima citra dengan dimensi tinggi, lebar, dan kanal
    warna.
2.  **Convolution layer:** menerapkan filter/kernel untuk mengekstraksi
    fitur.
3.  **Activation layer:** menerapkan fungsi aktivasi, biasanya ReLU.
4.  **Pooling layer:** mengurangi dimensi feature map.
5.  **Flatten atau Global Average Pooling:** mengubah feature map
    menjadi representasi yang dapat digunakan oleh lapisan klasifikasi.
6.  **Fully connected layer:** mempelajari hubungan fitur dengan kelas.
7.  **Output layer:** menghasilkan prediksi kelas.

Ilustrasi alur:

<img width="1225" height="617" alt="image" src="https://github.com/user-attachments/assets/55e7f8fe-ebe3-41f1-b4d2-c991093c0b88" />

### 2.2 Operasi Konvolusi

Konvolusi dilakukan dengan menggeser filter/kernel di atas citra atau
feature map. Pada setiap posisi, elemen kernel dikalikan dengan elemen
citra yang bersesuaian, kemudian seluruh hasil perkalian dijumlahkan.

Ilustrasinya seperti pada gambar berikut:

<img width="1374" height="516" alt="image" src="https://github.com/user-attachments/assets/4b5aa75e-99b4-45b3-a36a-25888bc14810" />

Contoh citra berukuran $3 \times 3$:

$$
I =
\begin{bmatrix}
1 & 2 & 0 \\
0 & 1 & 3 \\
2 & 1 & 0
\end{bmatrix}
$$

Contoh kernel berukuran $2 \times 2$:

$$
K =
\begin{bmatrix}
1 & 0 \\
0 & -1
\end{bmatrix}
$$

Untuk posisi pertama:

$$
(1)(1)+(2)(0)+(0)(0)+(1)(-1)=0
$$

Operasi tersebut diulangi pada posisi kernel berikutnya sehingga
menghasilkan *feature map*. Filter yang berbeda dapat mempelajari pola
yang berbeda pula.

Dalam CNN yang dilatih, nilai kernel bukan ditentukan secara manual
untuk seluruh proses, melainkan dipelajari melalui optimasi berdasarkan
loss.

<p align="center">
    <img width="600" height="239" alt="0_9y3bJ3EXof_KKTsj" src="https://github.com/user-attachments/assets/c45e8d1a-9847-4e33-9aed-3eaa1d1c14e7" />
</p>

### 2.3 Padding, Stride, dan Pooling

**Padding** adalah penambahan nilai di sekitar batas citra atau feature
map. Padding dapat membantu mempertahankan ukuran keluaran konvolusi.

<p align="center">
    <img width="800" height="400" alt="0_lLW_v6lWDC5C_txJ" src="https://github.com/user-attachments/assets/945911a2-b635-4155-83ca-c557d51ac75e" />
</p>

**Stride** adalah jarak pergeseran filter setiap kali operasi konvolusi
dilakukan. Stride lebih besar umumnya menghasilkan feature map yang
lebih kecil.

<p align="center">
    <img width="1280" height="720" alt="0_Vj63TlgKUetVUUYj" src="https://github.com/user-attachments/assets/a5ccd4ff-6070-4782-b61e-0dcd935c6f64" />
</p>

Untuk input berukuran $N \times N$, kernel $F \times F$, padding $P$,
dan stride $S$, ukuran keluaran pada masing-masing dimensi adalah:

$$
N_{out} = \left\lfloor \frac{N+2P-F}{S} \right\rfloor+1
$$

**Pooling** mengurangi dimensi feature map. Salah satu metode yang umum
adalah max pooling, yaitu mengambil nilai maksimum dari area tertentu.

<p align="center">
    <img width="726" height="550" alt="0_51NWG3HayLN3nNJ8" src="https://github.com/user-attachments/assets/fa693da4-604b-4dc0-a85a-03cf7e045c57" />
</p>

Contoh max pooling dengan jendela $2 \times 2$:

$$
\begin{bmatrix}
1 & 3 & 2 & 1 \\
4 & 6 & 0 & 2 \\
2 & 1 & 5 & 3 \\
0 & 2 & 4 & 1
\end{bmatrix}
\rightarrow
\begin{bmatrix}
6 & 2 \\
2 & 5
\end{bmatrix}
$$

Pooling membantu mengurangi jumlah komputasi dan membuat representasi
lebih ringkas.

### 2.4 Mengapa CNN Menggunakan Backpropagation?

CNN juga dilatih menggunakan prinsip forward propagation dan backward
propagation.

-   Forward propagation: citra diproses oleh convolution, activation,
    pooling, dan lapisan klasifikasi hingga menghasilkan prediksi.
-   Loss calculation: prediksi dibandingkan dengan label sebenarnya.
-   Backward propagation: gradien loss dihitung terhadap parameter yang
    dapat dilatih, termasuk kernel konvolusi dan bobot fully connected
    layer.
-   Optimizer: parameter diperbarui berdasarkan gradien.

Dengan demikian, CNN bukan metode yang terpisah dari neural network. CNN
merupakan salah satu arsitektur jaringan saraf yang dirancang khusus
untuk memanfaatkan struktur spasial data citra.

## 3. Praktikum: Klasifikasi Citra Biologis Menggunakan CNN

### 3.1 Tujuan Praktikum

Mahasiswa akan membangun CNN sederhana untuk mengklasifikasikan citra ke
dalam dua kelas:

-   `normal`: citra sampel dengan kategori normal.
-   `abnormal`: citra sampel dengan kategori abnormal.

Contoh ini menggunakan struktur folder dataset yang dapat diisi dengan
citra sel, jaringan, atau objek biologis lain. Nama kelas hanya contoh
dan harus disesuaikan dengan label dataset yang benar.

**Catatan penting:** contoh ini merupakan latihan klasifikasi citra,
bukan alat diagnosis medis. Hasil model tidak boleh digunakan untuk
keputusan klinis.

### 3.2 Persiapan Lingkungan

Instal library yang diperlukan:

``` bash
pip install tensorflow matplotlib numpy scikit-learn
```

Periksa instalasi TensorFlow:

``` python
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
```

Praktikum dapat dijalankan menggunakan Google Colab atau komputer lokal
yang telah memiliki Python dan TensorFlow.

### 3.3 Struktur Dataset

Siapkan dataset citra dengan struktur direktori berikut:

``` text
dataset/
|
|-- normal/
|   |-- image001.jpg
|   |-- image002.jpg
|   |-- image003.jpg
|   |-- ...
|
|-- abnormal/
    |-- image001.jpg
    |-- image002.jpg
    |-- image003.jpg
    |-- ...
```

Setiap folder merepresentasikan satu kelas. Nama folder akan digunakan
sebagai label oleh `image_dataset_from_directory`.

Jika menggunakan dataset publik, periksa lisensi, sumber data, definisi
label, dan cara pengambilan citra. Pastikan citra yang berasal dari satu
pasien, spesimen, preparat, atau objek yang sama tidak tersebar secara
sembarang antara data latih dan data uji. Pembagian berdasarkan subjek
atau spesimen lebih tepat untuk menghindari kebocoran data (*data
leakage*).

### 3.4 Memuat Dataset

Contoh berikut membagi data menjadi data latih dan validasi. Data uji
sebaiknya disiapkan secara terpisah dan tidak digunakan untuk memilih
hyperparameter.

``` python
import tensorflow as tf

DATA_DIR = "dataset"

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
SEED = 42

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

class_names = train_ds.class_names

print("Nama kelas:", class_names)
print("Jumlah kelas:", len(class_names))
```

Parameter `validation_split=0.2` berarti sekitar 20% data digunakan
sebagai validasi dan 80% sebagai data latih. Pembagian ini dilakukan
secara acak dengan seed tertentu. Untuk dataset biologis yang memiliki
beberapa citra dari subjek atau spesimen yang sama, pembagian acak per
citra belum tentu aman. Gunakan pembagian berbasis subjek/spesimen jika
metadata tersedia.

Dengan `label_mode="binary"`, label untuk dua kelas diberikan sebagai
nilai numerik 0 dan 1. Periksa `class_names` agar interpretasi label
tidak tertukar.

### 3.5 Menampilkan Contoh Citra

``` python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))

for images, labels in train_ds.take(1):
    for i in range(min(9, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        label_index = int(labels[i].numpy()[0])
        plt.title(class_names[label_index])
        plt.axis("off")

plt.tight_layout()
plt.show()
```

Tahap visualisasi penting untuk memeriksa apakah citra terbaca dengan
benar, label sesuai, dan tidak terdapat file yang rusak atau citra yang
tidak relevan.

### 3.6 Optimasi Input Pipeline

``` python
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(
    buffer_size=AUTOTUNE
)

val_ds = val_ds.cache().prefetch(
    buffer_size=AUTOTUNE
)
```

`cache()` membantu menyimpan data yang telah dibaca agar tidak selalu
dimuat ulang. `shuffle()` mengacak urutan data latih. `prefetch()`
membantu menyiapkan batch berikutnya ketika model sedang memproses batch
saat ini.

Untuk dataset yang terlalu besar untuk disimpan dalam memori,
pertimbangkan menghapus `cache()` atau menggunakan cache ke disk.

### 3.7 Membangun Arsitektur CNN

Model berikut menggunakan dua blok konvolusi dan max pooling, kemudian
diikuti fully connected layer.

``` python
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

    # Normalisasi piksel dari rentang 0-255 menjadi 0-1
    layers.Rescaling(1./255),

    # Blok konvolusi pertama
    layers.Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Blok konvolusi kedua
    layers.Conv2D(
        64,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Blok klasifikasi
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    # Output untuk klasifikasi biner
    layers.Dense(1, activation="sigmoid")
])

model.summary()
```

Penjelasan arsitektur:

-   `Input`: menerima citra RGB berukuran 128 × 128 piksel.
-   `Rescaling`: mengubah nilai piksel dari 0--255 menjadi 0--1.
-   `Conv2D(32)`: mempelajari 32 filter fitur pada lapisan konvolusi
    pertama.
-   `MaxPooling2D`: mengurangi ukuran feature map.
-   `Conv2D(64)`: mempelajari fitur yang lebih kompleks.
-   `Flatten`: mengubah feature map menjadi vektor.
-   `Dense(128)`: mempelajari kombinasi fitur untuk klasifikasi.
-   `Dropout(0.5)`: menonaktifkan sebagian unit secara acak selama
    pelatihan untuk membantu mengurangi overfitting.
-   `Dense(1, sigmoid)`: menghasilkan probabilitas untuk kelas dengan
    label 1.

### 3.8 Kompilasi Model

``` python
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

Alasan pemilihan konfigurasi:

-   Optimizer: Adam.
-   Learning rate awal: 0.001.
-   Loss: Binary Cross-Entropy karena klasifikasi memiliki dua kelas
    dengan output sigmoid.
-   Metric: accuracy untuk melihat proporsi prediksi yang benar.

Accuracy tidak cukup untuk mengevaluasi semua masalah biologis, terutama
jika jumlah data antar kelas tidak seimbang. Karena itu, evaluasi
lanjutan menggunakan precision, recall, F1-score, dan confusion matrix
juga diperlukan.

### 3.9 Melatih Model

``` python
EPOCHS = 20

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)
```

Satu epoch berarti model telah memproses seluruh batch data latih satu
kali. Selama pelatihan, model memperbarui bobot berdasarkan loss dan
optimizer.

### 3.10 Visualisasi Akurasi dan Loss

``` python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

# Grafik akurasi
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training dan Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Grafik loss
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training dan Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()
```

Interpretasi umum:

-   Training accuracy dan validation accuracy sama-sama meningkat: model
    sedang mempelajari pola yang berguna.
-   Training accuracy meningkat, tetapi validation accuracy stagnan atau
    menurun: model mungkin mengalami overfitting.
-   Training loss dan validation loss sama-sama menurun: proses
    pembelajaran cenderung berjalan baik.
-   Training loss menurun, tetapi validation loss meningkat: indikasi
    overfitting.
-   Kedua accuracy rendah: model mungkin mengalami underfitting, data
    kurang informatif, atau konfigurasi belum sesuai.

Grafik harus ditafsirkan bersama ukuran dataset, variasi citra, dan
hasil evaluasi pada data uji yang independen.

### 3.11 Evaluasi Model

Untuk contoh awal ini, validation set digunakan untuk memperlihatkan
cara menghitung metrik. Dalam eksperimen yang lebih kuat, siapkan test
set independen sejak awal dan gunakan hanya sekali untuk evaluasi akhir.

``` python
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

y_true = []
y_pred = []

for images, labels in val_ds:
    probabilities = model.predict(images, verbose=0)

    predictions = (probabilities >= 0.5).astype(int).flatten()

    y_true.extend(labels.numpy().astype(int).flatten())
    y_pred.extend(predictions)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("Accuracy:", accuracy_score(y_true, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)
```

Confusion matrix untuk klasifikasi biner dapat ditulis sebagai:

  Aktual / Prediksi     Prediksi 0   Prediksi 1
  ------------------- ------------ ------------
  Aktual 0                      TN           FP
  Aktual 1                      FN           TP

Keterangan:

-   True Negative (TN): data kelas 0 diprediksi sebagai kelas 0.
-   False Positive (FP): data kelas 0 diprediksi sebagai kelas 1.
-   False Negative (FN): data kelas 1 diprediksi sebagai kelas 0.
-   True Positive (TP): data kelas 1 diprediksi sebagai kelas 1.

Rumus metrik:

**Accuracy**

$$
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
$$

**Precision**

$$
Precision=\frac{TP}{TP+FP}
$$

**Recall / Sensitivity**

$$
Recall=\frac{TP}{TP+FN}
$$

**Specificity**

$$
Specificity=\frac{TN}{TN+FP}
$$

**F1-score**

$$
F1=2\times\frac{Precision\times Recall}{Precision+Recall}
$$

Untuk konteks biologis, perhatikan konsekuensi kesalahan klasifikasi.
Jika kelas positif merepresentasikan kondisi abnormal, False Negative
berarti sampel abnormal diprediksi normal. Oleh karena itu, recall kelas
abnormal dapat menjadi metrik penting. Namun, pemilihan metrik harus
mengikuti tujuan penelitian dan konsekuensi kesalahan.

### 3.12 Prediksi Satu Citra Baru

Pastikan citra baru memiliki format yang sesuai. Fungsi berikut membaca
citra, mengubah ukurannya, lalu menghasilkan prediksi.

``` python
import numpy as np
import tensorflow as tf

image_path = "contoh_citra.jpg"

img = tf.keras.utils.load_img(
    image_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH)
)

img_array = tf.keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

probability = model.predict(img_array, verbose=0)[0][0]

predicted_index = 1 if probability >= 0.5 else 0
predicted_class = class_names[predicted_index]

print("Prediksi kelas:", predicted_class)
print("Probabilitas kelas 1:", float(probability))
```

Perlu diingat bahwa probabilitas keluaran model bukan jaminan bahwa
prediksi benar. Model dapat menghasilkan prediksi dengan keyakinan
tinggi pada citra yang berbeda dari distribusi data pelatihan.

### 3.13 Menyimpan dan Memuat Model

Menyimpan model:

``` python
model.save("cnn_biologi.keras")
```

Memuat model:

``` python
loaded_model = tf.keras.models.load_model(
    "cnn_biologi.keras"
)
```

Simpan pula informasi nama kelas, ukuran citra, preprocessing, pembagian
dataset, dan konfigurasi eksperimen agar hasil dapat direproduksi.

## 4. Pengembangan Eksperimen

Setelah model dasar berjalan, lakukan eksperimen terkontrol. Ubah satu
atau beberapa konfigurasi dengan mencatat seluruh perubahan.

Contoh variabel eksperimen:

  Eksperimen   Konfigurasi          Tujuan
  ------------ -------------------- ------------------------------------------
  1            Citra 64 × 64        Mengamati pengaruh resolusi rendah
  2            Citra 128 × 128      Baseline resolusi
  3            Citra 224 × 224      Mengamati pengaruh resolusi lebih tinggi
  4            1 blok convolution   Model lebih sederhana
  5            2 blok convolution   Baseline CNN
  6            Dropout 0.2          Regularisasi lebih ringan
  7            Dropout 0.5          Regularisasi lebih kuat
  8            Data augmentation    Menguji ketahanan terhadap variasi citra

Contoh data augmentation:

``` python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])
```

Augmentasi hanya diterapkan pada data latih, bukan pada data validasi
atau data uji. Pastikan transformasi tidak mengubah makna biologis
citra. Misalnya, rotasi atau pembalikan tidak selalu valid untuk semua
jenis citra dan tugas.

Model dapat diperbarui dengan menambahkan `data_augmentation` setelah
input dan sebelum normalisasi:

``` python
model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    ),
    data_augmentation,
    tf.keras.layers.Rescaling(1./255),

    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", padding="same"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu", padding="same"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
```

## 5. Permasalahan Umum dalam Klasifikasi Citra Biologis

### 5.1 Overfitting

Overfitting terjadi ketika model terlalu menyesuaikan diri dengan data
latih sehingga performanya menurun pada data baru.

Strategi yang dapat dicoba:

-   Menambah jumlah data yang representatif.
-   Menggunakan augmentasi yang sesuai.
-   Menggunakan dropout atau regularisasi.
-   Mengurangi kompleksitas model.
-   Menggunakan early stopping.
-   Memastikan pembagian data tidak mengalami kebocoran.

Contoh early stopping:

``` python
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[early_stop]
)
```

### 5.2 Ketidakseimbangan Kelas

Jika jumlah citra kelas normal jauh lebih banyak daripada kelas
abnormal, accuracy dapat terlihat tinggi meskipun model gagal mengenali
kelas minoritas.

Strategi penanganan:

-   Melihat distribusi jumlah data setiap kelas.
-   Menggunakan precision, recall, F1-score, dan confusion matrix.
-   Mempertimbangkan class weight.
-   Menambah data kelas minoritas jika memungkinkan.
-   Menggunakan teknik sampling dengan hati-hati hanya pada data latih.

### 5.3 Data Leakage

Data leakage terjadi ketika informasi dari data validasi atau data uji
secara tidak sengaja memengaruhi pelatihan atau pemilihan model. Dalam
citra biologis, hal ini dapat terjadi jika citra dari spesimen yang sama
masuk ke data latih dan data uji.

Strategi pencegahan:

-   Membagi data berdasarkan pasien, spesimen, preparat, tanaman, atau
    objek biologis.
-   Menetapkan test set sebelum eksperimen.
-   Tidak menggunakan test set untuk memilih model atau threshold.
-   Mendokumentasikan proses preprocessing dan pembagian dataset.

### 5.4 Generalisasi Model

Model yang memiliki akurasi tinggi pada satu dataset belum tentu bekerja
baik pada dataset lain. Perbedaan perangkat mikroskop, pencahayaan,
pewarnaan, resolusi, laboratorium, dan prosedur pengambilan citra dapat
mengubah distribusi data.

Evaluasi yang lebih kuat dapat mencakup:

-   Pengujian pada data dari subjek yang berbeda.
-   Pengujian eksternal dari sumber atau laboratorium berbeda.
-   Analisis performa per kelas.
-   Analisis kesalahan prediksi.
-   Pengukuran variasi performa melalui beberapa pengulangan atau
    cross-validation yang sesuai.

## 6. Tugas Praktikum Mahasiswa

### Tugas 1: Memahami Forward dan Backward Propagation

Gunakan jaringan saraf dengan dua input, dua neuron hidden, dan satu
output seperti pada contoh perhitungan.

1.  Ubah nilai input menjadi $x_1=0.5$ dan $x_2=1.0$.
2.  Gunakan target $y=0$.
3.  Hitung nilai hidden layer menggunakan forward propagation.
4.  Hitung prediksi output.
5.  Hitung Binary Cross-Entropy.
6.  Hitung gradien pada output layer dan hidden layer.
7.  Perbarui bobot menggunakan learning rate 0.1.
8.  Bandingkan nilai loss sebelum dan sesudah pembaruan parameter.

Mahasiswa diminta menunjukkan tahapan perhitungan, bukan hanya
menampilkan hasil akhir program.

### Tugas 2: Klasifikasi Citra Biologis Menggunakan CNN

Bangun model CNN untuk klasifikasi dua kelas citra biologis.

Ketentuan:

1.  Gunakan dataset citra yang memiliki minimal dua kelas.
2.  Tampilkan minimal sembilan contoh citra dari dataset.
3.  Jelaskan jumlah citra pada setiap kelas.
4.  Lakukan preprocessing dan pembagian data latih serta validasi.
5.  Bangun CNN dengan minimal dua convolution layer.
6.  Latih model minimal 10 epoch atau sampai memenuhi kriteria
    penghentian yang dijelaskan.
7.  Tampilkan grafik training accuracy, validation accuracy, training
    loss, dan validation loss.
8.  Hitung confusion matrix, precision, recall, F1-score, dan accuracy.
9.  Tampilkan minimal lima contoh prediksi dan bandingkan dengan label
    sebenarnya.
10. Jelaskan apakah model mengalami overfitting atau underfitting.

### Tugas 3: Analisis Eksperimen

Lakukan minimal dua eksperimen dengan perbedaan konfigurasi, misalnya:

-   Resolusi citra 64 × 64 dibandingkan 128 × 128.
-   Jumlah filter convolution 16/32 dibandingkan 32/64.
-   Tanpa augmentasi dibandingkan dengan augmentasi.
-   Dropout 0.2 dibandingkan 0.5.

Buat tabel perbandingan hasil eksperimen.

  Konfigurasi      Accuracy   Precision   Recall   F1-score Catatan
  -------------- ---------- ----------- -------- ---------- ---------
  Eksperimen 1                                              
  Eksperimen 2                                              

Tuliskan konfigurasi terbaik berdasarkan tujuan klasifikasi, bukan hanya
berdasarkan accuracy tertinggi.

### Format Laporan

Laporan praktikum minimal memuat:

1.  Judul dan tujuan.
2.  Deskripsi dataset dan sumber data.
3.  Distribusi kelas.
4.  Preprocessing dan pembagian dataset.
5.  Arsitektur CNN.
6.  Proses pelatihan dan konfigurasi hyperparameter.
7.  Grafik accuracy dan loss.
8.  Confusion matrix dan metrik evaluasi.
9.  Analisis kesalahan prediksi.
10. Kesimpulan dan keterbatasan model.

## 7. Ringkasan

1.  CNN merupakan arsitektur neural network yang dirancang untuk
    mempelajari pola spasial pada citra.
2.  CNN memanfaatkan convolution, activation, pooling, dan lapisan
    klasifikasi.
3.  Evaluasi model harus mempertimbangkan confusion matrix dan metrik
    per kelas, bukan hanya accuracy.
4.  Pada penelitian biologis, pembagian data berdasarkan subjek atau
    spesimen, pencegahan data leakage, dan pengujian eksternal sangat
    penting.
5. Hasil klasifikasi merupakan keluaran model komputasi dan tidak
    otomatis menjadi kesimpulan biologis atau diagnosis klinis.

## 11. Referensi

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
