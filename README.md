# 🏥 Sistem Prediksi Cerdas Klaim Asuransi BPJS

## 📌 Deskripsi Proyek
Sistem ini adalah prototipe aplikasi berbasis web yang dirancang untuk membantu dalam proses **verifikasi dan estimasi klaim asuransi kesehatan BPJS**. Dengan memanfaatkan **algoritma Random Forest**, sistem ini mampu:
1. Memprediksi **status klaim**: apakah akan diterima atau ditolak.
2. Mengestimasi **nilai klaim** jika klaim diterima.
---

## 🎯 Tujuan
- Meningkatkan efisiensi proses pengajuan klaim asuransi BPJS.
- Menyediakan **prediksi objektif berbasis data historis**.
- Mengurangi proses manual dan mempercepat pengambilan keputusan.
- Menyediakan estimasi nilai klaim yang akurat dan transparan.
---

## 🧠 Teknologi dan Metodologi
- **Python** sebagai bahasa pemrograman utama.
- **Streamlit** untuk antarmuka web.
- **Random Forest Classifier** untuk klasifikasi status klaim.
- **Random Forest Regressor** untuk estimasi nilai klaim.
- Model Machine Learning telah dilatih menggunakan dataset riil (10.000 baris, 65 kolom).
---

## 🔁 Alur Sistem
1. Pengguna memasukkan data klaim melalui form.
2. Sistem menggunakan **model klasifikasi** untuk memprediksi status klaim.
3. Jika klaim diterima, sistem melanjutkan ke model **regresi** untuk menghitung estimasi jumlah klaim.
4. Hasil ditampilkan secara real-time lengkap dengan visualisasi.
