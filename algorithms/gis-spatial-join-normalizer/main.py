#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GIS SPATIAL JOIN NORMALIZER
---------------------------------
Fikir: Türkiye idari sınırları (il/ilçe/mahalle) poligonları ile mekânsal
eşleştirme yaparak adresleri normalize etmek. Adreslerden (il, ilçe, mahalle)
çıkarımı yapılır; mümkünse geokodla nokta alınıp polygon ile sjoin; değilse
metin tabanlı eşleme ile en olası birim seçilir. En sonunda, aynı idari anahtar
için train'de en sık görülen label ile tahmin yapılır.
"""
import os, re, json
from pathlib import Path
import pandas as pd
from unidecode import unidecode
from tqdm import tqdm

import geopandas as gpd
from shapely.geometry import Point
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from rapidfuzz import process as rf_process, fuzz

DATA_DIR = Path("data")
BOUNDARIES = DATA_DIR / "turkey_admin_boundaries.geojson"  # il/ilce/mahalle polygonları
ADMIN_IL = "il"; ADMIN_ILCE = "ilce"; ADMIN_MAH = "mahalle"  # attribute adları

def normalize_text(s):
    if not isinstance(s, str):
        return ""
    s = unidecode(s).lower()
    s = re.sub(r'[^a-z0-9\s/.,-]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def guess_tokens(addr_norm):
    # çok basit heurstic: "mah" / "mahalle", "ilce" gibi anahtarları yakala
    # daha sağlamı için TR gazetteer listeleri önerilir
    tokens = {"il": None, "ilce": None, "mahalle": None}
    # virgül ve slash ile bölüp dene
    parts = re.split(r'[,/]', addr_norm)
    for p in parts:
        p = p.strip()
        # mahalle ipuçları
        if "mah" in p or "mahalle" in p:
            tokens["mahalle"] = p
        # ilçe/il için kabaca son parçaları dene (tam güvenilmez)
    # en sona il/ilçe tahmini (zayıf)
    words = addr_norm.split()
    if len(words) >= 2:
        tokens["il"] = words[-1]
        tokens["ilce"] = words[-2]
    return tokens

def sjoin_point_to_admin(gdf_admin, pt):
    gpt = gpd.GeoDataFrame(geometry=[pt], crs="EPSG:4326")
    join = gpd.sjoin(gpt, gdf_admin, how="left", predicate="within")
    if join.empty:
        return None
    row = join.iloc[0]
    return { "il": row.get(ADMIN_IL), "ilce": row.get(ADMIN_ILCE), "mahalle": row.get(ADMIN_MAH) }

def main():
    train = pd.read_csv(DATA_DIR/"train.csv")
    test  = pd.read_csv(DATA_DIR/"test.csv")
    assert all(c in train.columns for c in ["address","label"])
    assert all(c in test.columns for c in ["id","address"])

    # admin sınırlar
    gdf_admin = gpd.read_file(BOUNDARIES).to_crs("EPSG:4326")
    cols = [ADMIN_IL, ADMIN_ILCE, ADMIN_MAH]
    for c in cols:
        if c not in gdf_admin.columns:
            raise SystemExit(f"GeoJSON içinde '{c}' kolonu yok. Lütfen alan adlarını düzenleyin.")

    # geocoder (hız limiti)
    geolocator = Nominatim(user_agent="adres-cozumleme-gis")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.2)

    # train adres normalizasyonu
    train["addr_norm"] = train["address"].astype(str).map(normalize_text)
    test["addr_norm"]  = test["address"].astype(str).map(normalize_text)

    # idari anahtar üzerinden train label modu
    # önce train için yaklaşık admin eşleme (zayıf ama deneme amaçlı)
    admin_keys = []
    for _, row in tqdm(train.iterrows(), total=len(train), desc="Train admin guess"):
        toks = guess_tokens(row["addr_norm"])
        key = f"{toks.get('il') or ''}|{toks.get('ilce') or ''}|{toks.get('mahalle') or ''}"
        admin_keys.append(key)
    train["admin_key"] = admin_keys
    # mode(label)
    mode_df = train.groupby("admin_key")["label"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]).reset_index()
    key2label = dict(zip(mode_df["admin_key"], mode_df["label"]))

    # test: geokod + sjoin + admin_key -> label
    preds = []
    for _, row in tqdm(test.iterrows(), total=len(test), desc="Test spatial normalize"):
        addr = row["addr_norm"]
        # 1) geocode & spatial join
        try:
            loc = geocode(addr + ", Turkey")
            admin = None
            if loc:
                pt = Point(loc.longitude, loc.latitude)
                admin = sjoin_point_to_admin(gdf_admin, pt)
        except Exception:
            admin = None

        # 2) fallback: metin tabanlı en yakın mahalle/ilçe/il (hızlı & kaba)
        if admin is None:
            # tüm mahalle adları içinde metin benzerliği ile en yakın
            choices = (gdf_admin[ADMIN_MAH].astype(str) + ", " + gdf_admin[ADMIN_ILCE].astype(str) + ", " + gdf_admin[ADMIN_IL].astype(str)).tolist()
            match, score, idx = rf_process.extractOne(addr, choices, scorer=fuzz.token_set_ratio)
            row_admin = gdf_admin.iloc[idx]
            admin = { "il": row_admin[ADMIN_IL], "ilce": row_admin[ADMIN_ILCE], "mahalle": row_admin[ADMIN_MAH] }

        key = f"{normalize_text(str(admin.get('il')))}|{normalize_text(str(admin.get('ilce')))}|{normalize_text(str(admin.get('mahalle')))}"

        # 3) label seçimi: aynı admin_key altında train label modu, yoksa global metin en benzere dön
        if key in key2label:
            preds.append(int(key2label[key]))
        else:
            sims = train["addr_norm"].map(lambda s: fuzz.token_set_ratio(addr, s))
            best_idx = sims.idxmax()
            preds.append(int(train.loc[best_idx,"label"]))

    sub = pd.DataFrame({"id": test["id"], "label": preds})
    sub.to_csv("submission.csv", index=False)
    print("Saved: submission.csv")

if __name__ == "__main__":
    main()
