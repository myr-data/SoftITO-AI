# Yapay Zeka Programcılığı

Eğitim sırasında yaptığım egzersizler, yazdığım kodlar, quizler burada bulunmaktadır.

## Hakkında

İstanbul Ticaret Odası (İTO) ve SoftİTO tarafından 2026 yılında verilen "4. Dönem Yapay Zeka Yazılımcılığı" eğitimi, SoftİTO bünyesinde verilen ilk yapay zeka odaklı eğitim olmaktadır. Bu programa başvuran yaklaşık 1000 kişi içerisinden ilk 24'e girerek başarı ile tamamladım.

## Konular

**Uzaktan Eğitim**

- Yapay Zeka Yazılım Geliştirmeye Giriş 	
- Dijital Dönüşüm
- Yazılım Proje Yönetimi
- Yazılım Analizi ve Tasarımı
- DevOps
- Programlama Sistematiği ve Kaynak Kod Yönetimi
- Veri Yapıları ve Temel Algoritmalar
- İşletim Sisteminin Temelleri
- Ağ ve Sistemin Temelleri
- Yazılım Mimarileri
- Güvenli Yazılım Geliştirmenin Temelleri
- Veritabanı Kullanımı ve Yönetimi
- Siber Güvenlik Farkındalığı
- Uluslararası IT Standartları
- İş Etiği ve Çalışma Kültürü
- Python Programlama Dili
- İstatistik ve Lineer Cebir Temelleri
- Makine Öğrenmesi Temelleri
- Derin Öğrenme ve Sinir Ağları
- Doğal Dil İşleme NLP

**Yüz Yüze Eğitim**
- Uygulamalı Spark ve Hadoop
- ETL/ELT Süreçleri ve Airflow ile Pipeline Yönetimi
- Vektör Veritabanları ve Semantic Search
- Veri Yönetişimi
- İleri RAG Mimarileri
- Otonom AI Ajan Geliştirme
- Fine-Tuning Atölyesi
- Model Değerlendirme Metrikleri ve RAGAS Framework
- Bilgisayarlı Görü
- Gerçek Zamanlı Video Analitiği ve Akış İşleme
- Generative Vision
- Docker ve Kubernetes (K8s) Temelleri
- API Geliştirme: FastAPI ile Yüksek Performanslı Inference Servisleri
- Model Deployment
- Canlı Model İzleme	
- Sektörel Senaryo Belirleme ve Veri Hazırlığı
- Uçtan Uca Proje Geliştirme
- Proje Sunumu ve Teknik Raporlama

# Notlarım

Bu dosyada genel notlar vardır. Detaylı ders notları için klasörlerin içerisine bakınız. (örn. ML içerisinde ML ile ilgili notlar içeren README.md)

> Detaylı notları İngilizce dilinde yazmak, araştırma yaparken kullandığım kaynakların da çoğunun İngilizce dilinde yazılmış olmasından dolayı kişisel tercihimdir.

## Dosya Yapısı

```
│   README.md
│   README_EN.md
│   sentetik-veri-uretimi.html
│   Sinem_Gencer_CV.html
│
├───0. Temeller
│       dsa.ipynb
│       python.ipynb
│
├───1. ML - Machine Learning
│   │   1_eda_linear_regression.ipynb
│   │   2_logistic_regression.ipynb
│   │   3_decision_tree.ipynb
│   │   4_knn.ipynb
│   │   5_naive_bayes.ipynb
│   │   6_svm.ipynb
│   │   7_unsupervised.ipynb
│   │   8_adaboost_xgboost_lightgbm.ipynb
│   │   meta.py
│
├───10. Görüntü İşleme
├───2. NLP - Natural Language Processing
│       1_tf_idf.ipynb
│       2_word2vec.ipynb
│       3_glove.ipynb
│       4_fasttext.ipynb
│
├───3. DL - Deep Learning
│       1_pytorch_ann.ipynb
│       2_keras_rnn.ipynb
│       3_lstm.ipynb
│       4_transformer.ipynb
│
├───4. LLM - Large Language Models
│       01_tokenization.ipynb
│       02_small_language_model.ipynb
│       03_lorem_ipsum.ipynb
│       04_distil_finetune.ipynb
│       05_peft.ipynb
│       06_quantization.ipynb
│       07_kv_cache.ipynb
│       08_flash_attention.ipynb
│       09_batching.ipynb
│       10_evaluation.ipynb
│       11_langchain.ipynb
│       12_llamaindex.ipynb
│
├───5. Gen AI - Generative AI
│       1_markov_chain.ipynb
│       2_gan.ipynb
│       3_rnn.ipynb
│
├───6. RAG - Retrieval Augmented Generation
├───7. Data Engineering
├───8. MLOps
├───9. Genetik
├───Datasets
│       StudentPerformanceFactors.csv
│       student_eda.csv
│
├───Projects
│   └───LSTM
│           lstm.ipynb
│           spam.csv
│
└───Quiz
```

## Makine Öğrenmesi

Öğrencilerin günlük hayatları, ders çalışma sıklıkları, okula uzaklıkları ve daha fazla özellikleri baz alınarak sınavdan alacakları not tahmin edilmiştir.

| Model | Accuracy | Precision | Recall | F1 Score | MAE | MSE | RMSE | R² |
|--------|:--------:|:---------:|:------:|:--------:|:---:|:---:|:----:|:--:|
| Linear Regression | — | — | — | — | **0.481** | **4.153** | **2.038** | **0.733** |
| Logistic Regression | **0.9835** | **0.9743** | **0.9589** | **0.9665** | — | — | — | — |
| K-Nearest Neighbors (KNN) | **0.8723** | **0.8462** | **0.5918** | **0.6965** | — | — | — | — |
| Naive Bayes | **0.9350** | **0.9331** | **0.7943** | **0.8581** | — | — | — | — |
| Support Vector Machine (SVM) | **0.9718** | **0.9575** | **0.9272** | **0.9421** | — | — | — | — |
| LightGBM | **0.9444** | **0.9359** | **0.8323** | **0.8811** | — | — | — | — |
| XGBoost | **0.9436** | **0.9266** | **0.8386** | **0.8804** | — | — | — | — |
| AdaBoost | **0.9373** | **0.9245** | **0.8133** | **0.8653** | — | — | — | — |

## Doğal Dil İşleme

**TF-IDF** ile yapay zeka hakkında yazılmış bir makaleden parçalar alınmış ve kelimelerin analizleri yapılmıştır.

Sıklık/nadirlik dengesi baz alınarak hesaplanan en önemli 3 kelime:
- olarak: 2.197
- ağırlıklı: 2.197
- kadar: 1.216

**IMDB** yorumları, hatalardan ve eksikliklerden arındılmıştır ve tokenlere çevrilmiştir. Sonra da **Word2Vec, GloVe, FastText** ile işlenmiştir.

**Word2Vec** çıktıları

```
Kelime çiftleri arasındaki benzerlik:
Çift                            Benzerlik
------------------------------------------
(movie, film)                0.8926  █████████████████
(good, great)                0.7586  ███████████████
(watch, see)                0.7186  ██████████████
(okay, fine)                0.5374  ██████████
(like, enjoy)                0.4894  █████████
(bad, cool)                0.4899  █████████


=== Vektör Aritmetiği ===
  (movie + action - film)
  → gunfights        (0.6361)
  → swordplay        (0.6330)
  → honk             (0.6239)
  ```


**GloVe** çıktıları, tüm kelimelerin istatistiklerine bakılarak hesaplanır:
- "movie" ve "film" yakınlık: 0.996
- "good" ve "bad" yakınlık: 0.991
- "actor" ve "actress" yakınlık: 0.983

**FastText** çıktıları:

```
'film' -> en benzer 3 kelime:
    file (0.8123)
    filmed (0.7942)
    films (0.7916)

'movie' -> en benzer 3 kelime:
    movies (0.8455)
    move (0.7924)
    pie (0.7691)

'action' -> en benzer 3 kelime:
    fraction (0.9407)
    function (0.9271)
    section (0.9250)

OOV Test: 'prehistoric' eğitimde var mı? Yok
  Ama FastText yine de vektör üretti (subword'ler sayesinde)!
  Vektör (ilk 5 değer): [ 0.03975208 -0.07775656  0.09691246  0.0873099  -0.1234142 ]
```

## Derin Öğrenme

Almanya'nın bir kentinin iklimsel/hava durumu bilgileri ile **çok değişkenli zaman serisi tahmini** projesi yapıldı. Öncelikle veriseti temizlendi ve eksikleri giderildi, sonra zamansal veriler sinüs ve kosinüs fonksiyonları kullanılarak dairesel bir şekilde sayısal değerlere çevrildi.

**Nem** ve **sıcaklık** verilerinin ikisinin de tahmin edilmesi için kullanılan farklı modellerin sonuçları:

ANN
Temperature
MAE  : 0.722 °C
RMSE : 0.927 °C
R²   : 0.9858

Humidity
MAE  : 1.890 %
RMSE : 2.881 %
R²   : 0.9657

RNN
Temperature
MAE  : 0.230 °C
RMSE : 0.296 °C
R²   : 0.9986

Humidity
MAE  : 0.856 %
RMSE : 1.171 %
R²   : 0.9943

LSTM
Temperature
MAE  : 0.633 °C
RMSE : 0.750 °C
R²   : 0.9907

Humidity
MAE  : 0.869 %
RMSE : 1.232 %
R²   : 0.9937

GRU
Temperature
MAE  : 0.633 °C
RMSE : 0.750 °C
R²   : 0.9907

Humidity
MAE  : 0.869 %
RMSE : 1.232 %
R²   : 0.9937

TRANSFORMER
