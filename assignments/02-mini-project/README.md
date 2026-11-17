# Mini Project: Machine Learning untuk Biologi

## 1. Deskripsi

Merupakan tugas kelompok. Setiap kelompok merancang dan menyelesaikan **satu proyek ML end-to-end** pada dataset biologi/biomedis. Proyek harus mencakup **minimal satu pendekatan supervised** dan **satu pendekatan unsupervised**, serta interpretasi biologisnya.

## 2. Pilihan Dataset

| # | Dataset | Sumber | Deskripsi | Tugas Supervised | Tugas Unsupervised |
|---|---|---|---|---|---|
| 1 | **Palmer Penguins** | `sns.load_dataset("penguins")` | Morfologi 3 spesies penguin (panjang paruh, sirip, massa tubuh) | Klasifikasi spesies / jenis kelamin | Clustering dan PCA morfologi |
| 2 | **Pima Indians Diabetes** | `fetch_openml("diabetes", version=1)` | Data klinis perempuan Pima dan status diabetes | Klasifikasi diabetes | Clustering profil pasien |
| 3 | **Heart Disease (UCI)** | UCI Machine Learning Repository | Fitur klinis dan status penyakit jantung | Klasifikasi penyakit jantung | PCA + clustering |
| 4 | **Wine Quality / Wine** | `sklearn.datasets.load_wine` | Profil kimia anggur dari 3 kultivar | Klasifikasi kultivar | Clustering kimia |
| 5 | **Gene expression (GEO / OpenML)** | NCBI GEO / OpenML | Ekspresi gen kanker vs normal (misalnya leukemia, colon) | Klasifikasi kanker, seleksi gen | PCA dan hierarchical clustering sampel |
| 6 | **Dataset sendiri** | Penelitian / lab / Kaggle | Harus disetujui dosen/asisten sebelum dimulai | Sesuai data | Sesuai data |

> Dataset yang sudah dipakai di kelas (Breast Cancer Wisconsin) **tidak boleh** dijadikan dataset utama proyek.

## 3. Tahapan Pengerjaan

**Tahap 1: Pendefinisian Masalah**
- Tuliskan pertanyaan biologis/klinis dan mengapa penting.
- Tentukan jenis tugas dan metrik evaluasi utama (jelaskan alasannya).

**Tahap 2: EDA**
- Ukuran data, tipe fitur, missing value, distribusi, korelasi, proporsi kelas.
- Minimal 3 visualisasi yang informatif.

**Tahap 3: Preprocessing**
- Imputasi / encoding / scaling dengan `Pipeline` (**tanpa data leakage**).
- Split train/test dengan `stratify` (bila klasifikasi).

**Tahap 4: Supervised Learning**
- Bandingkan **minimal 3 algoritma** (termasuk satu baseline sederhana).
- Gunakan **cross-validation** dan laporkan rata-rata ± std.
- Lakukan *hyperparameter tuning* sederhana pada model terbaik (`GridSearchCV` / `RandomizedSearchCV`).
- Evaluasi akhir pada data test: confusion matrix, classification report, ROC curve (klasifikasi) atau MAE/RMSE/R² (regresi).

**Tahap 5: Unsupervised Learning**
- PCA untuk visualisasi (laporkan variansi yang dijelaskan).
- K-Means dan/atau Hierarchical Clustering, serta alasan pemilihan K (elbow/silhouette).
- Bandingkan cluster dengan label asli (jika ada) atau metadata biologis.

**Tahap 6: Interpretasi**
- Fitur/gen mana yang paling berpengaruh (feature importance / koefisien)?
- Apakah hasilnya masuk akal secara biologis? Rujuk literatur.

**Tahap 7: Diskusi dan Keterbatasan**
- Kelemahan data, risiko bias/leakage/overfitting, dan saran pengembangan.

## 4. Template Awal (Contoh dengan Palmer Penguins)

```python
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, silhouette_score, adjusted_rand_score

# 1. Load data
df = sns.load_dataset("penguins").dropna()
fitur = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
X, y = df[fitur], df["species"]

# 2. EDA
sns.pairplot(df, vars=fitur, hue="species")
plt.show()

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Supervised: bandingkan model dengan CV
cv = StratifiedKFold(5, shuffle=True, random_state=42)
models = {
    "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": make_pipeline(StandardScaler(), SVC()),
}
for nama, m in models.items():
    s = cross_validate(m, X_train, y_train, cv=cv, scoring=["accuracy", "f1_macro"])
    print(f"{nama:13s} acc={s['test_accuracy'].mean():.3f}  f1_macro={s['test_f1_macro'].mean():.3f}")

# 5. Tuning model terpilih
param = {"svc__C": [0.1, 1, 10], "svc__kernel": ["linear", "rbf"]}
gs = GridSearchCV(make_pipeline(StandardScaler(), SVC()), param, cv=cv, scoring="f1_macro")
gs.fit(X_train, y_train)
print("Terbaik:", gs.best_params_, round(gs.best_score_, 3))

# 6. Evaluasi akhir (SEKALI saja)
y_pred = gs.predict(X_test)
print(classification_report(y_test, y_pred))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
plt.show()

# 7. Unsupervised: PCA + K-Means (tanpa memakai label)
Xs = StandardScaler().fit_transform(X)
Xp = PCA(n_components=2).fit_transform(Xs)
km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(Xs)
print("Silhouette:", round(silhouette_score(Xs, km.labels_), 3))
print("ARI vs spesies:", round(adjusted_rand_score(y, km.labels_), 3))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.scatterplot(x=Xp[:, 0], y=Xp[:, 1], hue=y.values, ax=ax[0]); ax[0].set_title("PCA - label spesies asli")
sns.scatterplot(x=Xp[:, 0], y=Xp[:, 1], hue=km.labels_, palette="viridis", ax=ax[1]); ax[1].set_title("PCA - cluster K-Means")
plt.show()
```

## 5. Deliverables

| Deliverable | Format | Keterangan |
|---|---|---|
| **Notebook** | `.ipynb` | Kode bersih, berjalan dari awal sampai akhir (*Restart & Run All*), diberi komentar dan penjelasan markdown |
| **Laporan singkat** | PDF / README (3-5 halaman) | Latar belakang, metode, hasil, diskusi, kesimpulan |
| **Repositori GitHub** | Link | Struktur rapi: `data/`, `notebooks/`, `README.md`, `requirements.txt` |
| **Presentasi** | 5-7 menit | Ringkas dan fokus pada temuan serta interpretasi biologis |

## 6. Rubrik Penilaian

## Aspek Penilaian
| No. | Komponen Penilaian | Bobot |
|:---:|---|:---:|
| 1 | Kesesuaian hasil pengerjaan dengan luaran yang diminta | 50% |
| 2 | Kesesuaian instruksi dan format pengumpulan | 50% |
| | **Total** | **100%** |

## 7. Tips

- Mulai dari **baseline sederhana**. Model kompleks tidak selalu lebih baik.
- Selalu tetapkan `random_state` agar hasil bisa direproduksi.
- Jangan "mengintip" data test untuk memilih model atau fitur.
- Hasil yang **terlalu bagus (akurasi ~100%)** patut dicurigai: cek kemungkinan *data leakage*.
- Laporkan **apa yang tidak berhasil** juga. Itu bagian dari proses sains.

