# GIS Geocode Hybrid (Nominatim + Metin Benzerliği)

**Amaç:** Test adreslerine en yakın coğrafi train adreslerini bulup, metin benzerliği ile birleştirerek **label** tahmini yapmak.

## Yöntem
- **Geokodlama:** `geopy.Nominatim` + `RateLimiter` + **JSON cache** (1.2s bekleme)
- **Yakın komşu:** Train noktaları üzerinde `KDTree` (k=10), gerçek mesafe için `haversine`
- **Metin benzerliği:** `rapidfuzz.fuzz.token_set_ratio`
- **Skor birleşimi:** `score = 0.6 * geo_score + 0.4 * text_score`

> Not: Nominatim hız limitine uyunuz. Büyük veri için özel/ücretli geokod servisleri veya **önceden geokodlanmış** koordinatlar önerilir.

## Kurulum
```bash
pip install -r requirements.txt
```

Gereken ek paketler:
- `geopandas`, `shapely`, `pyproj`, `rtree`, `geopy`, `haversine`, `rapidfuzz`

## Çalıştırma
```bash
python algorithms/gis-geocode-hybrid/main.py
# Özel yollar:
# TRAIN_PATH=data/train.csv TEST_PATH=data/test.csv OUT_PATH=submission.csv python algorithms/gis-geocode-hybrid/main.py
# KNN komşu sayısı:
# KNN_K=15 python algorithms/gis-geocode-hybrid/main.py
```

## Veri Beklentisi
- `train.csv` → kolonlar: `address`, `label`
- `test.csv` → kolonlar: `id`, `address`

## Çıktı
- Kök dizinde `submission.csv` (`id,label`)

## İpuçları
- Adresler il/ilçe/mahalle içermezse geokod başarısız olabilir. Metin normalize edin ve `, Turkey` ile bağlam verin.
- Üretim ortamında **öngeokodlanmış** train/test koordinatları ile çalışmak en sağlıklı ve hızlısıdır.
