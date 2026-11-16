Logistic Regression menunjukkan performa klasifikasi yang sangat baik dengan nilai akurasi sebesar 98,25%. Nilai precision dan recall yang sama-sama mencapai 97,62% menunjukkan bahwa algoritma ini memiliki kemampuan yang baik dalam mengidentifikasi kanker ganas sekaligus meminimalkan kesalahan prediksi. Selain itu, hanya terdapat satu kasus false negative, sehingga sebagian besar kasus kanker ganas berhasil terdeteksi.
<img width="571" height="470" alt="Logistic Regression Razidan" src="https://github.com/user-attachments/assets/36809794-a195-4b81-b8b7-b941b1be494b" />

Sedangkan pada Alogoritma Random Forest Classifier menghasilkan akurasi sebesar 93,57%. Nilai precision sebesar 92,06% menunjukkan bahwa sebagian besar prediksi kanker ganas yang dihasilkan algoritma ini memang benar merupakan kanker ganas. Sementara itu, nilai recall sebesar 90,63% menunjukkan bahwa algoritma mampu mendeteksi sekitar 90,63% dari seluruh kasus kanker ganas. Namun, terdapat enam kasus false negative, yaitu kanker ganas yang justru diklasifikasikan sebagai jinak. Hal ini perlu diperhatikan karena kesalahan tersebut berpotensi menyebabkan kasus kanker ganas tidak teridentifikasi.
<img width="579" height="470" alt="Random Forest Classifier Razidan" src="https://github.com/user-attachments/assets/6b00488b-b268-499b-93f7-b496c2a34ca5" />

Analisis dari perbandingan 2 algoritma tersebut adalah 
1. Accuracy: Logistic Regression memperoleh akurasi 98,25%, lebih tinggi dibandingkan Random Forest sebesar 93,57%. Artinya, berdasarkan hasil pengujian yang ditampilkan, Logistic Regression memiliki proporsi prediksi benar yang lebih tinggi.

2. Precision: Logistic Regression memiliki precision 97,62%, sedangkan Random Forest 92,06%. Hal ini menunjukkan bahwa Logistic Regression lebih baik dalam menghasilkan prediksi kanker ganas yang benar dan mengurangi kesalahan identifikasi kanker jinak sebagai ganas.

3. Recall: Logistic Regression memperoleh recall 97,62%, sedangkan Random Forest sebesar 90,63%. Dengan demikian, Logistic Regression lebih baik dalam mendeteksi kasus kanker ganas. Hal ini penting karena kasus kanker ganas yang tidak terdeteksi dapat berisiko terhadap penanganan pasien.

4. F1-Score: Logistic Regression memperoleh F1-Score 97,62%, lebih tinggi dibandingkan Random Forest sebesar 91,34%. Hasil ini menunjukkan keseimbangan precision dan recall yang lebih baik pada Logistic Regression.

Secara keseluruhan, Logistic Regression menunjukkan hasil evaluasi yang lebih unggul dibandingkan Random Forest berdasarkan keempat metrik yang dihitung.

Maka dapat diperoleh kesimpulan berdasarkan hasil pengujian, algoritma Logistic Regression memberikan performa klasifikasi yang lebih baik dibandingkan Random Forest Classifier. Logistic Regression memperoleh accuracy sebesar 98,25%, precision 97,62%, recall 97,62%, dan F1-Score 97,62%. Sementara itu, Random Forest memperoleh accuracy 93,57%, precision 92,06%, recall 90,63%, dan F1-Score 91,34%.

Keunggulan Logistic Regression terlihat dari jumlah kesalahan klasifikasi yang lebih sedikit, yaitu satu false positive dan satu false negative, dibandingkan Random Forest yang menghasilkan lima false positive dan enam false negative. Dengan demikian, berdasarkan hasil Confusion Matrix yang tersedia, Logistic Regression menjadi algoritma dengan performa terbaik untuk klasifikasi kanker payudara pada pengujian ini.
