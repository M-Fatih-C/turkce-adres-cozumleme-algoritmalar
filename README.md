# Türkçe Adres Çözümleme — 25+ Algoritma Koleksiyonu

Bu depo, adres normalizasyonu ve adres → **label/cluster** tahmini için denediğin farklı yaklaşımları **ayrı klasörler** halinde toplar. Her klasörde, ilgili kod ve açıklayıcı bir `README.md` bulunur.

## İçindekiler
| # | Yaklaşım | Klasör |
|---|---|---|
| 1 | BERT/Transformer + (Similarity: Fuzzy) | `algorithms/bert-fuzzy` |
| 2 | BERT/Transformer + (Similarity: Fuzzy) | `notebooks/bert-fuzzy-v2` |
| 3 | BERT/Transformer + FAISS + LightGBM+XGBoost + (Similarity: Fuzzy) | `notebooks/bert-faiss-xgb-fuzzy` |
| 4 | BERT/Transformer + PyTorch Model + (Similarity: Cosine+Fuzzy) | `notebooks/bert-torch-cosine` |
| 5 | BERT/Transformer + TF-IDF + FAISS + CatBoost+LightGBM+PyTorch Model+Random Forest+XGBoost + (Similarity: Fuzzy) | `notebooks/bert-faiss-rf-fuzzy` |
| 6 | BERT/Transformer + TF-IDF + FAISS + CatBoost+LightGBM+Random Forest+XGBoost + (Similarity: Fuzzy+Levenshtein) | `notebooks/bert-faiss-rf-fuzzy-v2` |
| 7 | BERT/Transformer + TF-IDF + FAISS + LightGBM + (Similarity: Cosine) | `notebooks/bert-faiss-lgbm-cosine` |
| 8 | BERT/Transformer + TF-IDF + FAISS + LightGBM + (Similarity: Cosine) | `notebooks/bert-faiss-lgbm-cosine-v2` |
| 9 | BERT/Transformer + TF-IDF + FAISS + LightGBM + (Similarity: Cosine+Fuzzy+Levenshtein) | `notebooks/bert-faiss-lgbm-cosine-v3` |
| 10 | BERT/Transformer + TF-IDF + FAISS + LightGBM + (Similarity: Cosine+Levenshtein) | `notebooks/bert-faiss-lgbm-cosine-v4` |
| 11 | GIS Geocode Hybrid (Nominatim + Metin Benzerliği) | `algorithms/gis-geocode-hybrid` |
| 12 | GIS Spatial Join Normalizer (İl/İlçe/Mahalle) | `algorithms/gis-spatial-join-normalizer` |
| 13 | Heuristic / Regex / Baseline | `algorithms/baseline` |
| 14 | Heuristic / Regex / Baseline | `algorithms/baseline-v2` |
| 15 | Heuristic / Regex / Baseline | `algorithms/baseline-v3` |
| 16 | Heuristic / Regex / Baseline | `notebooks/baseline-v4` |
| 17 | Heuristic / Regex / Baseline | `notebooks/baseline-v5` |
| 18 | TF-IDF | `notebooks/tfidf` |
| 19 | TF-IDF | `notebooks/tfidf-v2` |
| 20 | TF-IDF + (Similarity: Cosine+Fuzzy) | `notebooks/tfidf-cosine` |
| 21 | TF-IDF + CatBoost+LightGBM+Random Forest+XGBoost + (Similarity: Fuzzy+Levenshtein) | `notebooks/tfidf-rf-fuzzy` |
| 22 | TF-IDF + CatBoost+LightGBM+Random Forest+XGBoost + (Similarity: Fuzzy+Levenshtein) | `notebooks/tfidf-rf-fuzzy-v2` |
| 23 | TF-IDF + FAISS | `notebooks/tfidf-faiss` |
| 24 | TF-IDF + LightGBM+Logistic Regression+Naive Bayes | `notebooks/tfidf-logreg` |
| 25 | TF-IDF + PyTorch Model | `notebooks/tfidf-torch` |

## Kurulum
```bash
pip install -r requirements.txt
```

## Veri Düzeni
- `data/` klasörü örnek veri dosyaları için ayrılmıştır (gizli veri yüklemeyin).
- Her algoritma kendi klasöründe çalışır ve çıktı olarak kökte `submission.csv` üretmesi hedeflenir.

## Çoğaltılabilirlik (Reproducibility)
- Her algoritma klasöründeki `README.md`’de **girdi-çıktı**, **çalıştırma** ve **bileşenler** açıklanmıştır.
- Ortak bağımlılıklar kökteki `requirements.txt` dosyasındadır.

## Lisans
MIT — ayrıntılar için `LICENSE` dosyasına bakın.


🇹🇷 Türkçe — Problem Tanımı ve Kapsam
Amaç

Bu proje, serbest biçimli Türkçe adres metinlerinden doğru idari birimlere ve/veya proje özelindeki etiket/cluster alanına otomatik çıkarım yapmayı hedefler. İki temel alt görev ele alınır:

Adres Normalizasyonu: Farklı yazım biçimlerine sahip adresleri ortak bir şemaya dönüştürme (örn. mahalle/ilçe/il, cadde/sokak, kapı/daire no, posta kodu).

Etiket/Cluster Tahmini: Normalleştirilmiş (veya ham) adresten, yarışma/görev özelindeki sınıf/cluster etiketini tahmin etme.

Opsiyonel olarak GIS destekli iki yaklaşım da entegredir:

Geokod + Yakın Komşu (Hibrit): Adresi koordinata çevirip (Nominatim) coğrafi yakınlık + metin benzerliğiyle etiket seçmek.

Spatial Join Normalizer: Noktayı idari sınır çokgenleriyle (il/ilçe/mahalle) kesiştirip normalizasyonu mekânsal olarak doğrulamak.

Neden Zor?

Türkçe adresler, aşağıdaki nedenlerle makine tarafından yorumlanması güç serbest metinlerdir:

Tutarsız yazım ve kısaltmalar: “Cad./Cd.”, “Sok./Sk.”, “Mah./Mh.”, “No/No.”, “Daire/D.” vb.

Aksan/diakritik farklılıkları: “İ, I, Ğ, Ş, Ü, Ö, Ç” harfleri; “Üsküdar” → “Uskudar”.

Birden çok eş-isim: Türkiye genelinde aynı mahalle/sokak isimleri birden fazla il/ilçede bulunabilir.

Bölge ve idari değişiklikler: Yeni mahalleler, birleşmeler/ayrılmalar, posta kodu güncellemeleri.

Gürültü ve hatalar: Eksik alanlar, yazım hataları, karışık sıra (“İlçe/İl/Mahalle” gibi ters sıralar).

Veri Şeması (Örnek)

train.csv: address (str), label (int/str)

test.csv: id (str/int), address (str)

Opsiyonel: turkey_admin_boundaries.geojson (WGS84 / EPSG:4326) — kolonlar: il, ilce, mahalle.

Hedefler

Serbest biçimli address alanından tutarlı bir normal form üretmek.

Proje gereği label/cluster tahmininde yüksek doğruluk (Accuracy/F1).

Gerçek hayatta çalışabilirlik: hız, hataya dayanıklılık, kolay devreye alma (inference).

Yaklaşımlar
1) NLP Tabanlı

Ön-işleme: unidecode, lowercasing, regex ile temizlik, kısaltma genişletme (örn. “Cd.” → “Caddesi”).

Özellikler: TF-IDF/CountVectorizer, Transformer/BERT temelli gömmeler.

Modeller: Logistic Regression, SVM, XGBoost/LightGBM/CatBoost, ya da ince ayarlı BERT.

Benzerlik: Cosine similarity, Fuzzy (rapidfuzz), Levenshtein/Jaro-Winkler.

ANN/Arama: FAISS/Annoy ile hızlı en yakın örnek/vektör araması.

2) GIS Tabanlı (Opsiyonel/Hybrid)

Geokodlama: Nominatim (OSM) ile (lat, lon) çıkarımı; hız limiti ve caching şarttır.

Spatial Join: geopandas.sjoin ile noktayı idari polygonla eşleştirme → il|ilce|mahalle.

Hibrit Puanlama: Coğrafi yakınlık (haversine) + metin benzerliği puanlarını birleştirme.

Kural: Aynı idari anahtar altında train’de en sık görülen label’ı seçme (mode).

Değerlendirme

Sınıflandırma: Accuracy, Macro/Micro F1.

Normalizasyon Kalitesi: Bileşen eşleşme doğruluğu (il/ilçe/mahalle), edit distance.

Stabilite: K-Fold/Stratified CV; il/ilçe bazlı gruplu CV ile sızıntı (leakage) azaltma.

Gerçekçilik: Eksik/bozuk alanlara tolerans, hız (p99 gecikme), bellek.

Hata Sınıflandırması (Örnek)

Yanlış idari seviye: Mahalle doğru, ilçe/il hatalı.

Kısaltma hatası: “Cd/Sk” gibi türlerin karışması.

Coğrafi yakın hata: Yan mahalle/ilçe/il.

Metin eşleşme hatası: Benzerlik ölçütünün “yanlı” davranması (örn. yaygın isimler).

Veri İşleme İlkeleri

Normalize et: Unicode/aksan, boşluklar, noktalama, kısaltmalar.

Sözlükler: İl/ilçe/mahalle ad sözlüğü, sokak türü sözlüğü (Cd/Sk/Bulvar/Yol).

Önişleme sırası: Temizlik → kısaltma genişletme → token/şablon çıkarımı.

Önyargı ve Gizlilik: Gerçek kişisel adresleri anonimleştir; hassas veriyi repoya commit etme.

Üretim (Inference) Notları

Geokod Rate Limit: Nominatim için ~1–2 sn/islem → öngeokod veya ücretli servis düşünebilirsin.

Önbellek: Geokod/benzerlik sonuçlarını cache’le.

Hata toleransı: Fallback katmanları (örn. GIS yoksa metin bazlı).

Güncellik: İdari sınırlar ve posta kodları zamanla değişir; veri yenilemesi planlanmalı.

Önerilen Klasör Yapısı
algorithms/
  nlp-.../                 # TF-IDF/BERT + klasik modeller
  gis-geocode-hybrid/      # Nominatim + yakın komşu + metin skoru
  gis-spatial-join-normalizer/  # GeoJSON sınırlar + spatial join
data/
  train.csv / test.csv
  turkey_admin_boundaries.geojson

Gelecek Çalışmalar

Türkçe-lehçe/şive varyantları için daha zengin kural-tabanlı genişletmeler.

Adres sözlüklerinin (gazetteer) otomatik güncellenmesi.

BERT tabanlı sequence labeling ile bileşen çıkarımı (il/ilçe/mahalle/cadde/no).

Ensemble: NLP + GIS karışımı modellerin ağırlıklı oylaması.

🇬🇧 English — Problem Statement and Scope
Goal

This project aims to transform free-form Turkish address strings into canonical administrative units and/or a task-specific label/cluster. We focus on two core subtasks:

Address Normalization: Converting noisy, heterogeneous address text into a unified schema (province/district/neighbourhood, street type, building/apt, postal code).

Label/Cluster Prediction: Inferring the downstream class/cluster required by the task or competition from the (normalized or raw) address.

Optional GIS-assisted approaches are included:

Geocode + Nearest Neighbours (Hybrid): Obtain coordinates (Nominatim), then combine geographic proximity with text similarity to select a label.

Spatial Join Normalizer: Intersect the point with administrative polygons (province/district/neighbourhood) to “ground” normalization spatially.

Why It’s Hard

Turkish addresses are free text and often messy:

Inconsistent abbreviations: “Cad./Cd.” (Avenue), “Sok./Sk.” (Street), “Mah./Mh.” (Neighbourhood), “No/Daire”.

Diacritics: “İ, I, Ğ, Ş, Ü, Ö, Ç”; e.g., “Üsküdar” vs “Uskudar”.

Name collisions: Many neighbourhood/street names repeat across different districts/provinces.

Administrative drift: New neighbourhoods, splits/merges, postal-code changes.

Noise: Missing fields, typos, shuffled order (“District/Province/Neighbourhood”).

Data Schema (Example)

train.csv: address (str), label (int/str)

test.csv: id (str/int), address (str)

Optional GIS: turkey_admin_boundaries.geojson (WGS84 / EPSG:4326) with il, ilce, mahalle fields.

Objectives

Produce a consistent normalized form from free-form address.

Achieve high Accuracy/F1 in task-specific label/cluster prediction.

Ensure production viability: speed, robustness to noise, easy deployment.

Approaches
1) NLP-Driven

Preprocessing: unidecode, lowercasing, regex cleaning, abbreviation expansion (“Cd.” → “Caddesi”).

Features: TF-IDF/CountVectorizer, Transformer/BERT embeddings.

Models: Logistic Regression, SVM, XGBoost/LightGBM/CatBoost, fine-tuned BERT.

Similarity: Cosine, fuzzy matching, Levenshtein/Jaro-Winkler.

ANN/Search: FAISS/Annoy for fast nearest-vector lookups.

2) GIS-Driven (Optional/Hybrid)

Geocoding: Nominatim (OSM) → (lat, lon); respect rate limits and enable caching.

Spatial Join: geopandas.sjoin against polygons → il|ilce|mahalle key.

Hybrid Scoring: Combine geodesic distance (haversine) with text similarity.

Rule: Use the most frequent train label under the same admin key (mode).

Evaluation

Classification: Accuracy, Macro/Micro F1.

Normalization Quality: Component-level matching (province/district/neighbourhood), edit distance.

Stability: K-Fold/Stratified CV; grouped CV by admin units to reduce leakage.

Real-world constraints: Tolerance to missing/wrong fields, latency (p99), memory.

Error Taxonomy (Examples)

Wrong admin level: Neighbourhood correct, district/province wrong.

Abbreviation error: Street types confused (Cd vs Sk).

Geographic near miss: Adjacent neighbourhood/district/province.

Text match bias: Similarity overfits common names.

Processing Principles

Normalize: Unicode/diacritics, whitespace/punctuation, abbreviation expansion.

Dictionaries: Gazetteers for province/district/neighbourhood; street-type lexicon.

Order: Cleaning → expansion → token/pattern extraction.

Bias & Privacy: Anonymize real personal addresses; never commit sensitive data.

Inference Notes

Geocode Rate Limits: Nominatim ≈1–2s/call → consider pre-geocoding or paid services.

Caching: Store geocode/similarity results.

Fallbacks: Layered fallbacks (e.g., text-only path when GIS unavailable).

Freshness: Admin boundaries and postal codes evolve; plan data refresh.

Suggested Layout
algorithms/
  nlp-.../
  gis-geocode-hybrid/
  gis-spatial-join-normalizer/
data/
  train.csv / test.csv
  turkey_admin_boundaries.geojson

Future Work

Richer rule-based expansions for dialectal variations.

Automated gazetteer updates.

BERT-based sequence labeling to extract components (province/district/neighbourhood/street/no).

Ensembles combining NLP & GIS models.
