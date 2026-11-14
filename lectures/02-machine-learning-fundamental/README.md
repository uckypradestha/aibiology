# Lecture Note 2: Machine Learning Fundamental

## What is Machine Learning?

Machine Learning (ML) merupakan salah satu cabang dari Artificial Intelligence (AI) yang memungkinkan komputer mempelajari pola dari data dan menghasilkan prediksi atau keputusan tanpa harus diprogram secara eksplisit untuk setiap kemungkinan kondisi.

Berbeda dengan pemrograman konvensional yang menggunakan aturan yang ditentukan programmer, Machine Learning menggunakan algoritma untuk mempelajari hubungan dan pola dari data.

<p align="center">
  <img
    width="744"
    height="353"
    alt="image"
    src="https://github.com/user-attachments/assets/cdfa10a6-fa2e-47db-88dc-961ea1cf5abd"
  />
</p>

Dalam bidang biologi, Machine Learning dapat digunakan untuk:
- Klasifikasi spesies tanaman atau hewan.
- Prediksi penyakit berdasarkan karakteristik biologis.
- Analisis ekspresi gen.
- Identifikasi mikroorganisme.
- Pengelompokan sampel biologis.
- Analisis pertumbuhan tanaman.

## Relationship between AI, Machine Learning, and Deep Learning

Artificial Intelligence merupakan bidang yang lebih luas, sedangkan Machine Learning merupakan bagian dari AI. Deep Learning merupakan bagian dari Machine Learning yang menggunakan jaringan saraf tiruan dengan banyak lapisan.

```text
Artificial Intelligence (AI)
│
└── Machine Learning (ML)
    │
    ├── Supervised Learning
    │   ├── Classification
    │   └── Regression
    │
    ├── Unsupervised Learning
    │   ├── Clustering
    │   └── Dimensionality Reduction
    │
    └── Deep Learning
        ├── Artificial Neural Network
        ├── Convolutional Neural Network
        └── Recurrent Neural Network
```

## Types of Machine Learning

Secara umum, Machine Learning memiliki beberapa paradigma pembelajaran. Dua pendekatan utama yang akan dipelajari adalah Supervised Learning dan Unsupervised Learning.

### 1. Supervised Learning

Supervised Learning merupakan pendekatan Machine Learning yang menggunakan dataset berlabel (*labeled data*). Setiap data memiliki fitur (*features*) dan target atau label (*target*).

Model dilatih untuk mempelajari hubungan antara fitur dan target sehingga dapat melakukan prediksi terhadap data baru.

Contoh dalam bidang biologi:

- Klasifikasi spesies bunga berdasarkan karakteristik morfologi.
- Klasifikasi penyakit berdasarkan data biologis.
- Identifikasi jenis bakteri.
- Prediksi pertumbuhan tanaman berdasarkan kondisi lingkungan.

Supervised Learning dibagi menjadi dua jenis permasalahan:

#### A. Classification

Classification merupakan teknik untuk memprediksi kelas atau kategori suatu data.

Contoh:
- Klasifikasi spesies bunga: Setosa, Versicolor, dan Virginica.
- Klasifikasi kondisi jaringan: Normal dan Abnormal.
- Klasifikasi penyakit tanaman: Sehat dan Terinfeksi.

Beberapa algoritma Classification:

| Algorithm | Description |
|---|---|
| Logistic Regression | Memprediksi probabilitas suatu kelas |
| K-Nearest Neighbors (KNN) | Menentukan kelas berdasarkan mayoritas tetangga terdekat |
| Decision Tree | Membuat keputusan berdasarkan aturan berbentuk pohon |
| Random Forest | Menggabungkan beberapa Decision Tree |
| Support Vector Machine (SVM) | Menentukan hyperplane terbaik untuk memisahkan kelas |
| Naive Bayes | Menggunakan teorema Bayes untuk klasifikasi |

#### B. Regression

Regression merupakan teknik untuk memprediksi nilai numerik atau kontinu.

Contoh:
- Prediksi pertumbuhan tanaman.
- Prediksi konsentrasi suatu senyawa.
- Estimasi biomassa tanaman.
- Prediksi suhu optimal pertumbuhan mikroorganisme.

Beberapa algoritma Regression:

| Algorithm | Description |
|---|---|
| Linear Regression | Memodelkan hubungan linear antara fitur dan target |
| Polynomial Regression | Memodelkan hubungan polinomial |
| Decision Tree Regression | Melakukan prediksi menggunakan aturan pohon |
| Random Forest Regression | Menggabungkan beberapa pohon regresi |
| Support Vector Regression (SVR) | Melakukan regresi menggunakan konsep margin SVM |

### 2. Unsupervised Learning

Unsupervised Learning merupakan pendekatan Machine Learning yang menggunakan dataset tanpa label (*unlabeled data*).

Berbeda dengan Supervised Learning, model tidak diberikan informasi mengenai kelas atau target yang harus diprediksi. Model bertugas menemukan pola, struktur, atau hubungan tersembunyi dalam data.

Contoh dalam bidang biologi:

- Pengelompokan spesies berdasarkan karakteristik morfologi.
- Pengelompokan sampel berdasarkan ekspresi gen.
- Pengelompokan mikroorganisme.
- Analisis kemiripan karakteristik tanaman.

Unsupervised Learning memiliki beberapa pendekatan:

#### A. Clustering

Clustering merupakan teknik untuk mengelompokkan data berdasarkan kemiripan karakteristik.

Beberapa algoritma Clustering:

| Algorithm | Description |
|---|---|
| K-Means | Mengelompokkan data berdasarkan jarak terhadap centroid |
| Hierarchical Clustering | Membentuk kelompok secara bertingkat atau hierarkis |
| DBSCAN | Mengelompokkan data berdasarkan kepadatan (density) |

#### B. Dimensionality Reduction

Dimensionality Reduction merupakan teknik untuk mengurangi jumlah fitur dalam dataset dengan tetap mempertahankan informasi penting sebanyak mungkin.

Teknik ini bermanfaat ketika dataset memiliki banyak fitur, seperti data ekspresi gen yang dapat memiliki ribuan variabel.

Beberapa algoritma:

| Algorithm | Description |
|---|---|
| Principal Component Analysis (PCA) | Mengubah fitur menjadi sejumlah komponen utama |
| t-SNE | Memvisualisasikan struktur data berdimensi tinggi |
| UMAP | Memetakan data berdimensi tinggi ke dimensi lebih rendah |

## Comparison: Supervised vs Unsupervised Learning

| Parameter | Supervised Learning | Unsupervised Learning |
|---|---|---|
| Dataset | Labeled Data | Unlabeled Data |
| Target | Tersedia | Tidak tersedia |
| Tujuan | Prediksi | Menemukan pola |
| Proses | Training menggunakan label | Mempelajari struktur data |
| Output | Prediksi kelas atau nilai | Kelompok atau representasi data |
| Contoh | Klasifikasi spesies | Pengelompokan spesies |
| Algoritma | KNN, SVM, Decision Tree | K-Means, DBSCAN, PCA |

## Machine Learning Workflow

Pengembangan model Machine Learning umumnya dilakukan melalui beberapa tahapan.

```text
Problem Definition
        |
        v
Data Collection
        |
        v
Data Understanding
        |
        v
Data Preprocessing
        |
        v
Feature Selection
        |
        v
Model Development
        |
        v
Model Training
        |
        v
Model Evaluation
        |
        v
Interpretation & Deployment
```

### 1. Problem Definition

Menentukan permasalahan yang akan diselesaikan menggunakan Machine Learning.

Contoh:
- Apakah suatu sampel termasuk spesies Setosa, Versicolor, atau Virginica?
- Bagaimana mengelompokkan sampel biologis berdasarkan karakteristiknya?

### 2. Data Collection

Mengumpulkan data yang relevan dengan permasalahan.

Sumber dataset:
- Kaggle: https://www.kaggle.com/datasets
- UCI Machine Learning Repository: https://archive.ics.uci.edu/
- Scikit-learn Dataset: https://scikit-learn.org/stable/datasets.html
- NCBI: https://www.ncbi.nlm.nih.gov/

### 3. Data Understanding

Memahami karakteristik dataset, meliputi:
- Jumlah data.
- Jumlah fitur.
- Tipe data.
- Missing values.
- Distribusi data.
- Hubungan antarfitur.

### 4. Data Preprocessing

Preprocessing merupakan proses mempersiapkan data sebelum digunakan untuk melatih model.

| Technique | Description |
|---|---|
| Handling Missing Values | Menangani data yang kosong |
| Data Encoding | Mengubah data kategorikal menjadi numerik |
| Normalization | Mengubah nilai fitur ke rentang tertentu |
| Standardization | Menyamakan skala fitur berdasarkan mean dan standar deviasi |
| Outlier Handling | Menangani data dengan nilai ekstrem |

### 5. Feature Selection

Feature Selection merupakan proses memilih fitur yang relevan terhadap permasalahan.

Contoh pada dataset Iris:
- Sepal Length.
- Sepal Width.
- Petal Length.
- Petal Width.

### 6. Model Training

Pada tahap ini, algoritma Machine Learning mempelajari pola berdasarkan data yang diberikan.

### 7. Model Evaluation

Model dievaluasi untuk mengetahui kemampuan dalam menyelesaikan permasalahan.

Untuk Classification, metrik evaluasi yang umum digunakan:
- Accuracy.
- Precision.
- Recall.
- F1-Score.
- Confusion Matrix.

Untuk Regression:
- Mean Absolute Error (MAE).
- Mean Squared Error (MSE).
- Root Mean Squared Error (RMSE).
- R-squared (R²).

Untuk Clustering:
- Silhouette Score.
- Davies-Bouldin Index.
- Calinski-Harabasz Index.

---

# Supervised Learning: K-Nearest Neighbors (KNN)

## 1. Concept of KNN

K-Nearest Neighbors (KNN) merupakan algoritma Supervised Learning yang dapat digunakan untuk Classification maupun Regression.

Pada Classification, KNN menentukan kelas suatu data berdasarkan mayoritas kelas dari sejumlah tetangga terdekat.

Parameter utama dalam KNN adalah nilai K, yaitu jumlah tetangga yang dipertimbangkan.

Contoh:

Jika K = 3, kemudian dari tiga tetangga terdekat diperoleh:
- 2 data termasuk kelas A.
- 1 data termasuk kelas B.

Maka data baru akan diklasifikasikan sebagai kelas A.

## 2. Euclidean Distance

KNN umumnya menggunakan Euclidean Distance untuk menghitung jarak antara dua data.

Rumus:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?d(x%2Cy)%20%3D%20%5Csqrt%7B%5Csum_%7Bi%3D1%7D%5En(x_i-y_i)%5E2%7D" />
</p>

Keterangan:
- $d(x,y)$ : Jarak antara dua data.
- $x_i$ : Nilai fitur data pertama.
- $y_i$ : Nilai fitur data kedua.
- $n$ : Jumlah fitur.

## 3. KNN Implementation Using Python

Pada contoh ini digunakan dataset Iris dari Scikit-learn.

Dataset Iris memiliki:
- 150 sampel bunga.
- 4 fitur pengukuran.
- 3 kelas spesies: Setosa, Versicolor, dan Virginica.

### Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
```

### Load Dataset

```python
iris = load_iris()

X = iris.data
y = iris.target

df = pd.DataFrame(
    X,
    columns=iris.feature_names
)

df["species"] = [
    iris.target_names[i] for i in y
]

df.head()
```

### Dataset Information

```python
print("Jumlah data:", X.shape[0])
print("Jumlah fitur:", X.shape[1])
print("Kelas:", iris.target_names)

print("\nDistribusi kelas:")
print(df["species"].value_counts())
```

### Split Dataset

Dataset dibagi menjadi data training dan testing dengan rasio 80:20.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Data Training:", X_train.shape)
print("Data Testing:", X_test.shape)
```

### Feature Standardization

Standardization diperlukan karena KNN menggunakan perhitungan jarak. Fitur dengan rentang nilai besar dapat mendominasi perhitungan jarak jika tidak dilakukan scaling.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

### Build KNN Model

```python
knn = KNeighborsClassifier(
    n_neighbors=5,
    metric="euclidean"
)

knn.fit(X_train_scaled, y_train)
```

### Prediction

```python
y_pred = knn.predict(X_test_scaled)
```

### Model Evaluation

```python
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)
```

### Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - KNN")

plt.show()
```

## 4. Experiment: Effect of K Value

Nilai K dapat memengaruhi hasil klasifikasi. Untuk mengetahui pengaruhnya, lakukan eksperimen menggunakan beberapa nilai K.

```python
k_values = range(1, 16)

accuracies = []

for k in k_values:

    model = KNeighborsClassifier(
        n_neighbors=k
    )

    model.fit(X_train_scaled, y_train)

    prediction = model.predict(X_test_scaled)

    acc = accuracy_score(
        y_test,
        prediction
    )

    accuracies.append(acc)

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    accuracies,
    marker="o"
)

plt.xlabel("Nilai K")
plt.ylabel("Accuracy")
plt.title("Pengaruh Nilai K terhadap Accuracy")

plt.xticks(list(k_values))
plt.grid()

plt.show()
```

## 5. Analysis

Diskusikan hasil eksperimen berikut:

1. Berapa nilai accuracy yang diperoleh?
2. Apakah semua spesies berhasil diklasifikasikan dengan benar?
3. Kelas mana yang memiliki kesalahan klasifikasi paling banyak?
4. Bagaimana pengaruh nilai K terhadap performa model?
5. Mengapa standardization diperlukan pada KNN?

---

# Supervised Learning: Support Vector Machine (SVM)

## 1. Concept of SVM

Support Vector Machine (SVM) merupakan algoritma Supervised Learning yang digunakan untuk Classification dan Regression.

SVM bekerja dengan mencari hyperplane terbaik yang dapat memisahkan kelas-kelas data.

Hyperplane merupakan batas pemisah antar kelas.

Konsep penting dalam SVM:

- Hyperplane: Batas pemisah kelas.
- Margin: Jarak antara hyperplane dengan data terdekat dari setiap kelas.
- Support Vector: Data yang berada paling dekat dengan hyperplane.
- Kernel: Fungsi untuk menangani pola pemisahan yang tidak linear.

Tujuan utama SVM adalah memperoleh hyperplane dengan margin sebesar mungkin.

## 2. Types of SVM Kernel

| Kernel | Description |
|---|---|
| Linear | Memisahkan data menggunakan batas linear |
| Polynomial | Menangani hubungan polinomial |
| RBF | Menangani hubungan non-linear |
| Sigmoid | Menggunakan fungsi sigmoid |

## 3. SVM Implementation Using Python

Pada contoh ini digunakan dataset Iris dengan dua kelas dan dua fitur agar hasil klasifikasi dapat divisualisasikan dalam dua dimensi.

### Import Libraries

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns
```

### Load Dataset

```python
iris = load_iris()

# Menggunakan dua fitur
X = iris.data[:, [2, 3]]

# Menggunakan dua kelas
y = iris.target

mask = y < 2

X = X[mask]
y = y[mask]

print("Jumlah data:", X.shape[0])
print("Fitur:", iris.feature_names[2:4])
print("Kelas:", iris.target_names[:2])
```

### Split Dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### Standardization

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

### Build SVM Model

```python
svm_model = SVC(
    kernel="linear",
    C=1.0
)

svm_model.fit(
    X_train_scaled,
    y_train
)
```

### Prediction

```python
y_pred = svm_model.predict(X_test_scaled)
```

### Evaluation

```python
print(
    "Accuracy:",
    accuracy_score(y_test, y_pred)
)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names[:2]
    )
)
```

### Visualization

```python
plt.figure(figsize=(8, 6))

plt.scatter(
    X_test_scaled[:, 0],
    X_test_scaled[:, 1],
    c=y_test,
    cmap="coolwarm",
    marker="o"
)

plt.xlabel("Petal Length (Standardized)")
plt.ylabel("Petal Width (Standardized)")

plt.title("SVM Classification - Iris")

plt.show()
```

## 4. Analysis

Diskusikan:

1. Apa fungsi hyperplane pada SVM?
2. Mengapa margin yang lebih besar diharapkan menghasilkan pemisahan kelas yang lebih baik?
3. Apa perbedaan kernel Linear dan RBF?
4. Bagaimana pengaruh parameter C terhadap model?
5. Mengapa standardization diperlukan pada SVM?

---

# Unsupervised Learning: K-Means Clustering

## 1. Concept of K-Means

K-Means merupakan algoritma Unsupervised Learning yang digunakan untuk mengelompokkan data berdasarkan kemiripan karakteristik.

Algoritma ini membagi dataset menjadi sejumlah K kelompok (*clusters*).

Setiap kelompok memiliki titik pusat yang disebut centroid.

## 2. K-Means Algorithm

Tahapan algoritma K-Means:

1. Menentukan jumlah cluster (K).
2. Menginisialisasi centroid awal.
3. Menghitung jarak setiap data terhadap centroid.
4. Mengelompokkan data berdasarkan centroid terdekat.
5. Menghitung ulang posisi centroid.
6. Mengulangi proses hingga posisi centroid stabil atau mencapai kondisi berhenti.

### Euclidean Distance

K-Means menggunakan Euclidean Distance untuk menghitung jarak data terhadap centroid.

$$
d(x,c)=\sqrt{\sum_{i=1}^{n}(x_i-c_i)^2}
$$

Keterangan:
- $x$: Data.
- $c$: Centroid.
- $n$: Jumlah fitur.

### Centroid Calculation

Centroid baru dihitung berdasarkan rata-rata seluruh data dalam suatu cluster.

$$
C_j=\frac{1}{N_j}\sum_{i=1}^{N_j}x_i
$$

Keterangan:
- $C_j$: Centroid cluster ke-j.
- $N_j$: Jumlah data pada cluster ke-j.
- $x_i$: Data anggota cluster.

## 3. K-Means Implementation Using Python

Pada contoh ini digunakan dataset Iris tanpa menggunakan label spesies dalam proses clustering.

Tujuannya adalah melihat apakah algoritma dapat menemukan kelompok berdasarkan karakteristik morfologi bunga secara otomatis.

### Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
```

### Load Dataset

```python
iris = load_iris()

X = iris.data

df = pd.DataFrame(
    X,
    columns=iris.feature_names
)

df.head()
```

### Standardization

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

### Build K-Means Model

```python
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans.fit(X_scaled)

clusters = kmeans.labels_
```

### Clustering Results

```python
df["cluster"] = clusters

print(df.head(10))

print("\nJumlah anggota setiap cluster:")

print(
    df["cluster"].value_counts().sort_index()
)
```

## 4. Visualization Using PCA

Dataset Iris memiliki empat fitur. Agar hasil clustering dapat divisualisasikan dalam dua dimensi, digunakan Principal Component Analysis (PCA).

PCA digunakan untuk mereduksi empat fitur menjadi dua komponen utama.

```python
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    cmap="viridis",
    s=50
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title("K-Means Clustering - Iris Dataset")

plt.colorbar(label="Cluster")

plt.show()
```

## 5. Clustering Evaluation

Salah satu metrik evaluasi clustering adalah Silhouette Score.

Silhouette Score mengukur seberapa dekat suatu data dengan kelompoknya dibandingkan dengan kelompok lainnya.

Rumus:

$$
S(i)=\frac{b(i)-a(i)}{\max(a(i),b(i))}
$$

Keterangan:

- $a(i)$: Rata-rata jarak data terhadap anggota cluster yang sama.
- $b(i)$: Rata-rata jarak minimum terhadap cluster lainnya.

Rentang nilai:

$$
-1 \leq S(i) \leq 1
$$

Interpretasi:

| Value | Interpretation |
|---|---|
| Mendekati 1 | Data berada pada cluster yang sesuai |
| Mendekati 0 | Data berada di antara dua cluster |
| Mendekati -1 | Data kemungkinan masuk ke cluster yang kurang tepat |

### Implementation

```python
score = silhouette_score(
    X_scaled,
    clusters
)

print("Silhouette Score:", score)
```

## 6. Analysis

Diskusikan:

1. Mengapa K-Means tidak membutuhkan label spesies?
2. Apa fungsi centroid?
3. Mengapa nilai K harus ditentukan sebelum proses clustering?
4. Apakah hasil cluster selalu sama dengan kelompok spesies sebenarnya?
5. Apa fungsi PCA dalam visualisasi hasil clustering?
6. Bagaimana interpretasi Silhouette Score?

---

# Unsupervised Learning: Hierarchical Clustering

## 1. Concept of Hierarchical Clustering

Hierarchical Clustering merupakan algoritma Unsupervised Learning yang mengelompokkan data berdasarkan hubungan hierarkis atau bertingkat.

Berbeda dengan K-Means, Hierarchical Clustering dapat membentuk struktur kelompok yang digambarkan menggunakan dendrogram.

Terdapat dua pendekatan utama:

1. Agglomerative Clustering: Setiap data awalnya dianggap sebagai satu cluster, kemudian cluster digabungkan secara bertahap.
2. Divisive Clustering: Seluruh data awalnya berada dalam satu cluster, kemudian dibagi menjadi kelompok yang lebih kecil.

Pendekatan yang umum digunakan adalah Agglomerative Clustering.

## 2. Linkage Method

Linkage menentukan bagaimana jarak antara dua cluster dihitung.

| Linkage | Description |
|---|---|
| Single | Jarak minimum antaranggota dua cluster |
| Complete | Jarak maksimum antaranggota dua cluster |
| Average | Rata-rata jarak antaranggota dua cluster |
| Ward | Menggabungkan cluster dengan kenaikan variasi dalam cluster sekecil mungkin |

## 3. Hierarchical Clustering Implementation

### Import Libraries

```python
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

from scipy.cluster.hierarchy import (
    dendrogram,
    linkage
)
```

### Load Dataset

```python
iris = load_iris()

X = iris.data
```

### Standardization

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

### Dendrogram Visualization

```python
Z = linkage(
    X_scaled,
    method="ward"
)

plt.figure(figsize=(12, 6))

dendrogram(
    Z,
    truncate_mode="lastp",
    p=20
)

plt.title("Hierarchical Clustering Dendrogram")

plt.xlabel("Data / Cluster")
plt.ylabel("Distance")

plt.show()
```

### Build Agglomerative Clustering

```python
hierarchical = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

clusters = hierarchical.fit_predict(
    X_scaled
)

print("Hasil Clustering:")

print(clusters)
```

### Visualization Using PCA

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    cmap="viridis"
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title("Hierarchical Clustering - Iris")

plt.show()
```

## 4. Analysis

Diskusikan:

1. Apa fungsi dendrogram?
2. Apa perbedaan Agglomerative dan Divisive Clustering?
3. Apa perbedaan Single, Complete, Average, dan Ward Linkage?
4. Bagaimana menentukan jumlah cluster dari dendrogram?
5. Apa perbedaan K-Means dan Hierarchical Clustering?

---

# Model Evaluation

Setiap algoritma Machine Learning memiliki karakteristik dan keunggulan masing-masing.

## 1. Supervised Learning Evaluation

Untuk Classification, beberapa metrik evaluasi yang digunakan adalah:

### Accuracy

Mengukur proporsi prediksi yang benar terhadap seluruh data.

$$
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
$$

### Precision

Mengukur proporsi prediksi positif yang benar.

$$
Precision=\frac{TP}{TP+FP}
$$

### Recall

Mengukur proporsi data positif aktual yang berhasil dikenali.

$$
Recall=\frac{TP}{TP+FN}
$$

### F1-Score

Merupakan harmonic mean dari Precision dan Recall.

$$
F1=2\times\frac{Precision\times Recall}{Precision+Recall}
$$

Keterangan:

- TP: True Positive.
- TN: True Negative.
- FP: False Positive.
- FN: False Negative.

Dalam bidang kesehatan dan biologi, pemilihan metrik harus disesuaikan dengan tujuan penelitian. Misalnya, pada identifikasi penyakit, Recall dapat menjadi metrik penting untuk mengurangi jumlah sampel positif yang tidak terdeteksi.

## 2. Unsupervised Learning Evaluation

Evaluasi clustering berbeda dengan Classification karena data tidak memiliki label target.

Beberapa metrik yang umum digunakan:

| Metric | Description |
|---|---|
| Silhouette Score | Mengukur kekompakan dan pemisahan cluster |
| Davies-Bouldin Index | Mengukur kemiripan antarcluster |
| Calinski-Harabasz Index | Mengukur rasio dispersi antarcluster dan dalam cluster |

Interpretasi:

- Silhouette Score: Semakin mendekati 1 umumnya semakin baik.
- Davies-Bouldin Index: Semakin kecil umumnya semakin baik.
- Calinski-Harabasz Index: Semakin besar umumnya semakin baik.

Perlu diperhatikan bahwa hasil evaluasi internal clustering tidak selalu menunjukkan bahwa kelompok yang terbentuk memiliki makna biologis. Interpretasi biologis tetap memerlukan analisis domain dan, jika tersedia, validasi menggunakan informasi biologis independen.

---

# Important Considerations in Biological Data

## 1. High-Dimensional Data

Data biologis, khususnya ekspresi gen dan genomik, dapat memiliki jumlah fitur yang sangat besar.

Sebagai contoh:

- Jumlah sampel: 100.
- Jumlah fitur gen: 20.000.

Kondisi tersebut dapat menyebabkan kesulitan dalam proses training dan meningkatkan risiko overfitting.

Pendekatan yang dapat digunakan:

- Feature Selection.
- Principal Component Analysis.
- Regularization.
- Dimensionality Reduction.

## 2. Class Imbalance

Class imbalance terjadi ketika jumlah sampel pada setiap kelas tidak seimbang.

Contoh:

| Class | Number of Samples |
|---|---:|
| Normal | 900 |
| Disease | 100 |

Dalam kondisi tersebut, model dapat menghasilkan accuracy yang tinggi tetapi memiliki kemampuan deteksi kelas minoritas yang buruk.

Pendekatan yang dapat digunakan:

- Stratified Sampling.
- Class Weight.
- Oversampling.
- Undersampling.
- SMOTE.

## 3. Overfitting

Overfitting terjadi ketika model terlalu menyesuaikan diri dengan data training sehingga kemampuan generalisasi terhadap data baru menjadi buruk.

Cara mengurangi overfitting:

- Cross Validation.
- Regularization.
- Feature Selection.
- Pengaturan kompleksitas model.
- Penggunaan dataset yang representatif.

## 4. Data Leakage

Data leakage merupakan kondisi ketika informasi dari data testing secara tidak sengaja digunakan dalam proses training.

Contohnya, melakukan standardization pada seluruh dataset sebelum train-test split.

Pendekatan yang benar:

1. Membagi data menjadi training dan testing.
2. Melakukan fit preprocessing menggunakan data training.
3. Menerapkan transformasi yang sama pada data testing.

---

# Summary

Machine Learning merupakan salah satu pendekatan utama dalam pengembangan Artificial Intelligence untuk menganalisis data dan menemukan pola.

Dua paradigma utama yang telah dipelajari:

**Supervised Learning**

- Menggunakan data berlabel.
- Bertujuan melakukan prediksi.
- Terdiri atas Classification dan Regression.
- Contoh algoritma: KNN, SVM, Decision Tree, Logistic Regression, dan Random Forest.

**Unsupervised Learning**

- Menggunakan data tanpa label.
- Bertujuan menemukan pola atau struktur tersembunyi.
- Meliputi Clustering dan Dimensionality Reduction.
- Contoh algoritma: K-Means, Hierarchical Clustering, DBSCAN, dan PCA.

Dalam bidang biologi, Machine Learning dapat dimanfaatkan untuk klasifikasi spesies, analisis penyakit, pengelompokan ekspresi gen, analisis mikroorganisme, dan berbagai permasalahan biologis lainnya.

Pemilihan algoritma harus mempertimbangkan karakteristik data, tujuan analisis, kualitas dataset, serta interpretasi hasil secara biologis.

---

# Exercises and Assignments

## Exercise 1: Supervised Learning - KNN

Gunakan dataset Iris untuk melakukan klasifikasi menggunakan algoritma KNN.

Tugas:

1. Load dataset Iris menggunakan Scikit-learn.
2. Identifikasi jumlah sampel dan fitur.
3. Lakukan pembagian data training dan testing dengan rasio 80:20.
4. Lakukan standardization.
5. Bangun model KNN dengan K = 3, 5, 7, dan 9.
6. Bandingkan accuracy setiap model.
7. Buat confusion matrix untuk model terbaik.
8. Berikan interpretasi hasil klasifikasi.

Pertanyaan analisis:

- Bagaimana pengaruh nilai K terhadap hasil klasifikasi?
- Mengapa standardization diperlukan?
- Apakah accuracy merupakan satu-satunya indikator kualitas model?

## Exercise 2: Supervised Learning - SVM

Gunakan dataset Iris untuk melakukan klasifikasi menggunakan algoritma SVM.

Tugas:

1. Gunakan dua kelas dari dataset Iris.
2. Gunakan fitur Petal Length dan Petal Width.
3. Lakukan standardization.
4. Bangun model SVM dengan kernel Linear.
5. Ulangi eksperimen dengan kernel RBF.
6. Bandingkan hasil kedua model.
7. Analisis pengaruh parameter C dan kernel terhadap hasil klasifikasi.

Pertanyaan analisis:

- Apa perbedaan kernel Linear dan RBF?
- Apa fungsi support vector?
- Mengapa SVM menggunakan konsep margin?

## Exercise 3: Unsupervised Learning - K-Means

Gunakan dataset Iris tanpa menggunakan label spesies dalam proses clustering.

Tugas:

1. Load dataset Iris.
2. Gunakan empat fitur pengukuran bunga.
3. Lakukan standardization.
4. Bangun model K-Means dengan K = 2, 3, 4, dan 5.
5. Hitung Silhouette Score masing-masing model.
6. Visualisasikan hasil clustering menggunakan PCA.
7. Bandingkan hasil clustering berdasarkan jumlah cluster.

Pertanyaan analisis:

- Bagaimana pengaruh nilai K terhadap hasil clustering?
- Apakah cluster yang terbentuk selalu sama dengan spesies sebenarnya?
- Mengapa label spesies tidak boleh digunakan sebagai input clustering?
- Bagaimana interpretasi Silhouette Score?

## Exercise 4: Unsupervised Learning - Hierarchical Clustering

Gunakan dataset Iris untuk melakukan pengelompokan menggunakan Hierarchical Clustering.

Tugas:

1. Load dataset Iris.
2. Lakukan standardization.
3. Buat dendrogram menggunakan metode Ward.
4. Bangun Agglomerative Clustering dengan tiga cluster.
5. Visualisasikan hasil menggunakan PCA.
6. Bandingkan hasilnya dengan K-Means.

Pertanyaan analisis:

- Apa informasi yang dapat diperoleh dari dendrogram?
- Bagaimana cara menentukan jumlah cluster?
- Apa kelebihan dan kekurangan Hierarchical Clustering dibandingkan K-Means?

---

# Mini Project: Machine Learning for Biological Data

## Project: Biological Data Pattern Analysis

### Description

Mahasiswa diminta mengembangkan model Machine Learning sederhana untuk menganalisis dataset biologis menggunakan pendekatan Supervised Learning dan Unsupervised Learning.

Dataset dapat dipilih dari sumber publik seperti:

- UCI Machine Learning Repository.
- Kaggle.
- Scikit-learn Dataset.
- Dataset publik bidang bioinformatika.

### Project Topics

| Topic | Dataset / Object | Approach |
|---|---|---|
| Klasifikasi spesies bunga | Iris Dataset | Supervised |
| Klasifikasi spesies penguin | Palmer Penguins | Supervised |
| Pengelompokan spesies bunga | Iris Dataset tanpa label | Unsupervised |
| Pengelompokan sampel biologis | Gene Expression Dataset | Unsupervised |
| Pengelompokan karakteristik tanaman | Plant Dataset | Unsupervised |

### Project Requirements

Mahasiswa harus melakukan:

1. Identifikasi permasalahan.
2. Pemilihan dataset.
3. Eksplorasi dan visualisasi data.
4. Data preprocessing.
5. Implementasi minimal dua algoritma Machine Learning.
6. Evaluasi performa model.
7. Perbandingan hasil algoritma.
8. Interpretasi hasil berdasarkan konteks biologis.

### Project Deliverables

Laporan dan notebook Python yang berisi:

- Deskripsi permasalahan.
- Informasi dataset.
- Tahapan preprocessing.
- Implementasi algoritma.
- Hasil evaluasi.
- Visualisasi.
- Analisis dan kesimpulan.

---

# References

1. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
2. Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd Edition). O'Reilly Media.
3. James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023). *An Introduction to Statistical Learning: With Applications in Python*. Springer.
4. Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press.
5. Scikit-learn Documentation. https://scikit-learn.org/stable/
6. UCI Machine Learning Repository. https://archive.ics.uci.edu/
7. Kaggle Datasets. https://www.kaggle.com/datasets

---

**End of Lecture Note 2 - Machine Learning Fundamental**

**Course: Data Science and AI in Biology**
