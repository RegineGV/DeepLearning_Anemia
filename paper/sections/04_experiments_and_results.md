# 4. Hasil Eksperimen dan Evaluasi Komparatif

## 4.1 Performa Diagnostik Model Konjungtiva Mata
Model **EfficientNet-B0 + CBAM** dilatih menggunakan optimasi *AdamW* dan *Cosine Annealing Learning Rate*. Mekanisme *Early Stopping* secara otomatis memilih bobot terbaik pada **Epoch 14**, menghindari *overfitting* pada epoch-epoch selanjutnya.

Hasil evaluasi pada subset uji independen (*unseen internal test set*, 33 pasien) disajikan pada Tabel 1:

### Tabel 1. Kinerja Diagnostik Model Konjungtiva (EfficientNet-B0 + CBAM)
| Metrik Klinis | Nilai | Keterangan Medis |
| :--- | :---: | :--- |
| **Akurasi** | **81.82%** | Proporsi keseluruhan diagnosis yang tepat |
| **Sensitivitas (Recall)** | **92.86%** | Kemampuan mendeteksi pasien anemik (hanya 1 kasus terlewat dari 14) |
| **Spesifisitas** | **73.68%** | Kemampuan mengidentifikasi pasien non-anemik (14 dari 19 benar) |
| **Presisi (PPV)** | **72.22%** | Keandalan diagnosis positif anemia |
| **F1-Score** | **0.8125** | Keseimbangan harmonis antara presisi dan sensitivitas |
| **ROC-AUC** | **0.9662** | Tingkat diskriminasi diagnostik sangat tinggi |

### Matriks Konfusi
Dari 33 pasien uji independen:
- **True Positive (TP)**: 13 pasien anemik terdiagnosis dengan benar.
- **False Negative (FN)**: Hanya 1 pasien anemik yang tidak terdeteksi. Dalam skrining klinis, rendahnya nilai *False Negative* adalah prioritas tertinggi untuk mencegah pasien anemia lolos tanpa pengobatan.
- **True Negative (TN)**: 14 pasien normal diidentifikasi dengan benar.
- **False Positive (FP)**: 5 pasien normal terklasifikasi sebagai anemia.

File matriks konfusi grafis resolusi tinggi tersimpan di:
`results/figures/confusion_matrices/conjunctiva_confusion_matrix.png`

---

## 4.2 Analisis Kurva ROC dan Konvergensi Pelatihan
- **Kurva ROC (AUC = 0.966)** membuktikan bahwa model memiliki separabilitas fitur yang sangat tajam antara citra konjungtiva pucat (*pallor*) dan konjungtiva dengan vaskularisasi normal.
- **Kurva Loss dan Akurasi**: Menunjukkan penurunan *loss* yang stabil dan konvergensi cepat pada representasi fitur konjungtiva berkat kontribusi modul *Channel Attention* dan *Spatial Attention* (CBAM).

File kurva pelatihan dan ROC tersimpan di:
- `results/figures/curves/conjunctiva_training_curves.png`
- `results/figures/curves/conjunctiva_roc_curve.png`

---

## 4.3 Interpretabilitas Medis melalui Grad-CAM (XAI)
Untuk menjamin transparansi klinis, metode **Grad-CAM (Gradient-weighted Class Activation Mapping)** diaplikasikan pada lapisan konvolusi ber-atensi terakhir.

Peta panas visual (*heatmaps*) yang dihasilkan mengonfirmasi:
1. Model memusatkan atensi terkuat (warna merah-oranye) tepat pada **jaringan pembuluh darah konjungtiva palpebra**, bukan pada sklera, iris, atau derau latar belakang.
2. Pada kasus anemia, area kepucatan mikrovaskular memicu aktivasi tertinggi, memverifikasi bahwa model belajar fitur patologis yang relevan secara medis sesuai kaidah hematologi.

Sampel visualisasi Grad-CAM tersimpan di:
`results/figures/gradcam/conjunctiva/`
