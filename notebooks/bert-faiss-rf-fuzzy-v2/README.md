# BERT/Transformer + TF-IDF + FAISS + CatBoost+LightGBM+Random Forest+XGBoost + (Similarity: Fuzzy+Levenshtein)

Bu klasör, `notebooks/bert-faiss-rf-fuzzy-v2/Untitled6.ipynb` dosyasındaki **Jupyter Notebook** çözümünün açıklamasını içerir.

## Özet
- **Amaç:** Adres metninden doğru **label/cluster** tahmini yapmak ve `submission.csv` üretmek.
- **Yaklaşım:** Aşağıdaki bileşenlerin birleşimi ile deneme yapılmıştır.
- **Ön-işleme:** Lowercase, Regex
- **Vektörleştirme:** TF-IDF
- **Gömme:** BERT/Transformer, Sentence-Transformers
- **Yakın Komşu (ANN):** FAISS
- **Model(ler):** CatBoost, LightGBM, Random Forest, XGBoost
- **Benzerlik Ölçüleri:** Fuzzy, Levenshtein

## Giriş/Çıkış Beklentisi
- **Girdi:** `train.csv` içinde `address`, `label`; `test.csv` içinde `id`, `address`.
- **Çıktı:** Proje kökünde `submission.csv` (`id,label`).

## Nasıl Çalıştırılır
> Not: Komut satırı argümanları yoksa dosyayı doğrudan çalıştırın veya notebook hücrelerini sırayla yürütün.

### Seçenek A — Python dosyası
```bash
python notebooks/bert-faiss-rf-fuzzy-v2/Untitled6.ipynb
```

### Seçenek B — Notebook
1. `notebooks/` altındaki `.ipynb` dosyasını açın.
2. Üstten aşağı hücreleri çalıştırın.

## Notlar
- Veri yolunuza göre dosya içindeki yolları güncelleyin.
- Gereken kütüphaneler için kökteki `requirements.txt` dosyasını yükleyin.
