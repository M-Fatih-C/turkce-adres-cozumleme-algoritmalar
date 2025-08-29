# GIS Spatial Join Normalizer (İl/İlçe/Mahalle)

**Amaç:** Adresleri Türkiye idari sınır çokgenleri (il/ilçe/mahalle) ile eşleştirip **normalize etmek** ve aynı idari anahtar altında **label modu** ile tahmin yapmak.

## Yöntem
1. **Adres normalize** (`unidecode`, regex)
2. **Geokodlama** (Nominatim) → Nokta
3. **Spatial Join** (`geopandas.sjoin`) → İdari birim (il/ilçe/mahalle)
4. **Admin key** = `il|ilce|mahalle` → Train'de **mode(label)** ile tahmin
5. **Fallback:** Geokod/Join başarısızsa, idari isimlere metin benzerliği

## Gereken Veri
- `data/turkey_admin_boundaries.geojson` (WGS84)
  - En azından şu kolonlarla: `il`, `ilce`, `mahalle`
- `data/train.csv` (kolonlar: `address`, `label`)
- `data/test.csv`  (kolonlar: `id`, `address`)

> İpucu: Türkiye idari sınır verisini (il/ilçe/mahalle) GeoJSON/SHAPE olarak bulup dönüştürün; çokgenlerin CRS'si **EPSG:4326** olmalı.

## Kurulum
```bash
pip install -r requirements.txt
```

Ek paketler:
- `geopandas`, `shapely`, `pyproj`, `rtree`, `geopy`, `rapidfuzz`

## Çalıştırma
```bash
python algorithms/gis-spatial-join-normalizer/main.py
```

## Çıktı
- Kök dizinde `submission.csv`

## Notlar
- Nominatim hız limitlerine uyunuz. Büyük veri için öngeokodlanmış centroid veya offline geocoder düşünün.
- `ADMIN_IL/ILCE/MAH` kolon adları farklı ise dosya başındaki sabitleri güncelleyin.
