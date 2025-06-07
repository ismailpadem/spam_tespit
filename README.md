# Spam (Önemsiz E-posta) Tespit Sistemi

## Projenin Amacı

Bu projede, biz metin mesajlarını otomatik olarak "spam" (önemsiz/istenmeyen) veya "ham" (normal/istenen) olarak sınıflandıran gelişmiş bir makine öğrenmesi sistemi geliştirdik. PyTorch kullanarak derin öğrenme tekniklerini uygulayarak, e-posta ve SMS mesajlarını yüksek doğrulukla kategorize edebilen bir model oluşturduk.

### Temel Hedeflerimiz
- E-posta ve SMS mesajları için etkili bir spam tespit sistemi geliştirmek
- Yüksek doğruluk oranıyla istenmeyen mesajları otomatik filtrelemek
- CUDA GPU desteği ile hızlı eğitim ve tahmin sağlamak
- Gerçek zamanlı spam tespiti için optimize edilmiş model oluşturmak
- Kapsamlı sonuç analizi ve raporlama sistemi kurmak

## Proje Altyapısı

### Veri Seti
- **Veri Kaynağı**: `data/spam.csv` - Spam/Ham etiketli mesaj veri seti
- **Veri Formatı**: CSV dosyası (metin ve etiket sütunları)
- **Kategoriler**: 2 farklı sınıf
  - **Ham**: Normal, istenen mesajlar (0)
  - **Spam**: İstenmeyen, zararlı mesajlar (1)
- **Veri İşleme**: Otomatik metin temizleme ve normalizasyon

### Model Mimarisi
- **Ana Model**: PyTorch Embedding + Fully Connected Network
- **Embedding Boyutu**: 128 (yapılandırılabilir)
- **Maksimum Metin Uzunluğu**: 20 kelime
- **Dropout Oranı**: 0.3
- **Öğrenme Oranı**: 0.001
- **Optimizasyon**: Adam Optimizer
- **Kayıp Fonksiyonu**: Binary Cross Entropy Loss
- **Aktivasyon**: Sigmoid (çıkış katmanı)

### Test Ortamı ve Donanım
- **İşletim Sistemi**: Windows 11
- **Python Sürümü**: 3.12
- **GPU Desteği**: CUDA destekli GPU (Nvidia RTX 4060 Mobile 140W)
- **RAM**: 16GB
- **CPU**: Intel i7-13650HX

### Proje Yapısı
```
NLP_Final/
├── spam_tespit/
│   ├── spam_detection.py         # Ana model ve eğitim scripti
│   └── data/
│       └── spam.csv              # Spam veri seti
├── requirements.txt              # Gerekli kütüphaneler
├── spam_model_gpu.pth           # Eğitilmiş model
├── training_plots.png           # Eğitim grafikleri
├── confusion_matrix.png         # Karışıklık matrisi
├── spam_results/                # Detaylı sonuçlar
│   └── spam_results_*.json      # JSON formatında sonuçlar
└── README.md                    # Bu dosya
```

## Test Sonuçları ve Performans Metrikleri

Biz bu projede kapsamlı bir sonuç analizi gerçekleştirdik. Aşağıda elde ettiğimiz kesin metrikleri bulabilirsiniz:

### Ana Performans Metrikleri
- **Test Doğruluğu**: %97.13
- **F1-Score (Ham)**: 0.9836
- **F1-Score (Spam)**: 0.8841
- **Precision (Ham)**: 0.9726
- **Precision (Spam)**: 0.9606
- **Recall (Ham)**: 0.9948
- **Recall (Spam)**: 0.8188
- **Weighted Average F1**: 0.9703

### Eğitim Süreleri
- **Toplam Eğitim Süresi**: 0:00:05 (5.25 saniye)
- **Ortalama Epoch Süresi**: 0.34 saniye
- **Eğitim Epoch Sayısı**: 10 epoch
- **Batch Boyutu**: 32
- **GPU Kullanıldı**: NVIDIA GeForce RTX 4060 Laptop GPU
- **GPU Hızlanması**: 15-20x CPU'ya göre

### Veri Seti İstatistikleri
- **Toplam Örneklem**: 5,569 mesaj
- **Ham Mesajlar**: 4,822 (%86.59)
- **Spam Mesajlar**: 747 (%13.41)
- **Test Seti**: 1,114 mesaj
- **Eğitim Seti**: 4,455 mesaj
- **Doğru Tahmin**: 1,082 / 1,114

### Detaylı Sınıf Performansı

| Sınıf | Precision | Recall | F1-Score | Support |
|-------|-----------|---------|----------|---------|
| Ham   | 0.9726    | 0.9948  | 0.9836   | 965     |
| Spam  | 0.9606    | 0.8188  | 0.8841   | 149     |
| **Macro Avg** | **0.9666** | **0.9068** | **0.9338** | **1114** |
| **Weighted Avg** | **0.9710** | **0.9713** | **0.9703** | **1114** |

### Model Detayları
- **Kelime Dağarcığı Boyutu**: 3,960 kelime
- **Embedding Boyutu**: 128 boyut
- **Toplam Parametre**: 507,009
- **Eğitilebilir Parametre**: 507,009
- **Maksimum Metin Uzunluğu**: 20 kelime

### Epoch Bazında Gelişim

| Epoch | Train Loss | Train Acc | Test Loss | Test Acc |
|-------|------------|-----------|-----------|----------|
| 1     | 0.5521     | 0.8474    | 0.4509    | 0.8797   |
| 2     | 0.3849     | 0.8943    | 0.3311    | 0.9102   |
| 3     | 0.2824     | 0.9273    | 0.2533    | 0.9497   |
| 4     | 0.2110     | 0.9535    | 0.2053    | 0.9632   |
| 5     | 0.1673     | 0.9697    | 0.1762    | 0.9650   |
| 6     | 0.1369     | 0.9760    | 0.1580    | 0.9659   |
| 7     | 0.1208     | 0.9785    | 0.1442    | 0.9659   |
| 8     | 0.1019     | 0.9823    | 0.1350    | 0.9677   |
| 9     | 0.0904     | 0.9828    | 0.1275    | 0.9695   |
| 10    | 0.0812     | 0.9852    | 0.1224    | 0.9713   |

### Karışıklık Matrisi

|           | Ham (Tahmin) | Spam (Tahmin) |
|-----------|--------------|---------------|
| **Ham (Gerçek)**  | 960          | 5             |
| **Spam (Gerçek)** | 27           | 122           |

- **Doğru Ham Tespiti**: 960/965 (%99.48)
- **Doğru Spam Tespiti**: 122/149 (%81.88)
- **Yanlış Pozitif (Ham→Spam)**: 5 mesaj
- **Yanlış Negatif (Spam→Ham)**: 27 mesaj

### Örnek Tahminler

| Mesaj | Gerçek Tahmin | Güven Skoru |
|-------|---------------|-------------|
| "Free entry! Win a brand new iPhone now!" | SPAM | 0.945 |
| "I'll meet you at the library at 5pm" | HAM | 0.011 |
| "Congratulations! You've won a FREE holiday!" | SPAM | 0.953 |
| "Can you pick up some milk on your way home?" | HAM | 0.004 |
| "URGENT! Click here to claim your prize NOW!" | SPAM | 0.961 |
| "Thanks for the information, see you tomorrow" | HAM | 0.408 |

### Görsel Sonuçlar

Biz eğitim sürecinin ve sonuçların görsel analizini şu grafiklerde sunuyoruz:

#### 1. Eğitim Geçmişi
![Eğitim Geçmişi](training_plots.png)
*Eğitim ve test kayıpları ile doğruluk oranlarının gelişimi*

#### 2. Karışıklık Matrisi
![Karışıklık Matrisi](confusion_matrix.png)
*Gerçek ve tahmin edilen sınıflar arasındaki ilişki*

### Detaylı Sonuç Dosyaları

Biz kapsamlı sonuç analizi için şu dosyaları oluşturduk:

1. **`spam_results/spam_results_YYYYMMDD_HHMMSS.json`**
   - Tüm eğitim parametreleri ve sonuçları (JSON formatında)
   - Model konfigürasyonu ve hiperparametreler
   - Epoch bazında detaylı metrikler
   - Dataset istatistikleri

2. **`training_plots.png`**
   - Eğitim ve test kayıp grafikleri
   - Doğruluk oranları gelişimi
   - Model performansının görsel analizi

3. **`confusion_matrix.png`**
   - Karışıklık matrisi görselleştirmesi
   - Sınıf bazında tahmin doğruluğu
   - Ham/Spam ayrımının başarısı

4. **`spam_model_gpu.pth`**
   - Eğitilmiş PyTorch modeli
   - Kelime dağarcığı (vocabulary)
   - Model ağırlıkları ve parametreleri

## Kurulum ve Çalıştırma

### Gerekli Kütüphaneler

Biz bu projede aşağıdaki Python kütüphanelerini kullandık:

```bash
# Temel derin öğrenme kütüphaneleri
torch>=2.0.0
torchvision>=0.15.0

# Veri manipülasyonu
pandas>=1.5.0
numpy>=1.21.0

# Makine öğrenmesi
scikit-learn>=1.3.0

# Görselleştirme
matplotlib>=3.5.0
seaborn>=0.11.0



### ⚙️ Kurulum Adımları

1. **Depoyu klonlayın veya dosyaları indirin**
```bash
git clone <repository-url>
cd NLP_Final
```

2. **Gerekli kütüphaneleri yükleyin**
```bash
# Tüm gereksinimleri yükle
pip install -r requirements.txt

# VEYA sadece temel kütüphaneleri yükle
pip install torch pandas numpy scikit-learn matplotlib seaborn
```

3. **GPU desteği için (NVIDIA kullanıcıları)**
```bash
# CUDA destekli PyTorch yükle
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Diğer kütüphaneleri yükle
pip install pandas numpy scikit-learn matplotlib seaborn
```

4. **Veri setini hazırlayın**
```bash
# data/spam.csv dosyasının mevcut olduğundan emin olun
# Dosya formatı: metin ve etiket sütunları içermeli
```

### Modeli Eğitme

Biz modeli eğitmek için şu komutu kullanıyoruz:

```bash
# Ana klasöre gidin
cd NLP_Final

# Spam tespit modelini eğitin
python spam_tespit/spam_detection.py
```

Bu komut şunları gerçekleştirir:
- Veri setini yükler ve ön işleme tabi tutar
- Kelime dağarcığı oluşturur
- PyTorch modelini oluşturur ve GPU'ya taşır
- 10 epoch boyunca modeli eğitir
- Test sonuçlarını değerlendirir
- Grafikleri ve raporları oluşturur
- Eğitilmiş modeli kaydeder

### Sonuçları İnceleme

Eğitim tamamlandıktan sonra:

1. **Grafikleri kontrol edin**
   - `training_plots.png` - Eğitim geçmişi
   - `confusion_matrix.png` - Karışıklık matrisi

2. **Model dosyalarını inceleyin**
   - `spam_model_gpu.pth` - Eğitilmiş model
   - `spam_results/` - Detaylı JSON sonuçları

3. **İnteraktif test yapın**
   - Script otomatik olarak örnek mesajları test eder
   - Kendi mesajlarınızı test edebilirsiniz


## Özelleştirme Seçenekleri

Biz modeli özelleştirmek için şu parametreleri değiştirebilirsiniz:

```python
# spam_detection.py dosyasında main() fonksiyonunda
spam_detector = SpamDetectionSystem(
    max_len=20,                   # Maksimum kelime sayısı
    embedding_dim=128,            # Embedding boyutu
    learning_rate=0.001           # Öğrenme oranı
)

# Eğitim parametreleri
history = spam_detector.train_model(
    train_loader, 
    test_loader, 
    epochs=10                     # Epoch sayısı
)
```

## Teknik Detaylar

### Metin Ön İşleme Pipeline
1. **Küçük harfe çevirme** - Büyük/küçük harf tutarlılığı
2. **HTML etiketlerini kaldırma** - Web içeriği temizliği
3. **URL'leri kaldırma** - Bağlantı temizliği
4. **E-posta adreslerini kaldırma** - Kişisel bilgi koruması
5. **Telefon numaralarını kaldırma** - Numaraları filtreleme
6. **Noktalama işaretlerini kaldırma** - Simge temizliği
7. **Sayıları kaldırma** - Numerik değer temizliği
8. **Fazla boşlukları kaldırma** - Boşluk normalizasyonu


### Veri Akışı
1. **Ham Metin** → Temizleme → **Temiz Metin**
2. **Temiz Metin** → Tokenization → **Kelime Listesi**
3. **Kelime Listesi** → Encoding → **Sayısal Dizi**
4. **Sayısal Dizi** → Padding → **Sabit Uzunluk**
5. **Sabit Uzunluk** → Model → **Spam/Ham Tahmini**

