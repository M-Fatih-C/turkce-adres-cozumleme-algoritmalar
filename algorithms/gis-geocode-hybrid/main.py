#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GIS + NLP HİBRİT EŞLEŞTİRİCİ
---------------------------------------------------
Fikir: Train adreslerini ve test adreslerini geokodla (lat,lon). Test adresi için
coğrafi olarak en yakın K train adresini seç, sonra metin benzerliği ile skorla.
Son skoru (mesafe + metin benzerliği) birleştirerek label tahmini yap.

UYARI: Nominatim kullanımında hız limiti var (≥1 sn/gönderim). Bu betikte
RateLimiter ve basit JSON caching kullanıldı.
"""
import os, sys, time, json, math, re
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from unidecode import unidecode

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from rapidfuzz import fuzz
from haversine import haversine, Unit
from sklearn.neighbors import KDTree

DATA_DIR = Path("data")
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "geocode_cache.json"

def load_cache():
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_cache(cache):
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

def normalize_text(s):
    if not isinstance(s, str):
        return ""
    s = unidecode(s).lower()
    s = re.sub(r'[^a-z0-9\s/.,-]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def geocode_address(geocoder, txt, cache):
    key = normalize_text(txt)
    if key in cache:
        return cache[key]
    try:
        loc = geocoder(txt + ", Turkey")
        if loc:
            result = {"lat": loc.latitude, "lon": loc.longitude}
        else:
            result = None
    except Exception:
        result = None
    cache[key] = result
    return result

def build_kdtree(df):
    # expects radians for haversine if using BallTree; we use KDTree with euclidean on degrees for simplicity for local ranges.
    # For better accuracy on larger area, consider BallTree with Haversine metric.
    coords = df[["lat","lon"]].values
    return KDTree(coords, leaf_size=40)

def combined_score(dist_km, text_sim, w_dist=0.6, w_text=0.4):
    # dist smaller is better -> convert to [0,1] by 1/(1+dist)
    geo_score = 1.0 / (1.0 + dist_km)
    return w_dist*geo_score + w_text*(text_sim/100.0)

def main():
    train_path = os.environ.get("TRAIN_PATH", str(DATA_DIR/"train.csv"))
    test_path  = os.environ.get("TEST_PATH",  str(DATA_DIR/"test.csv"))
    out_path   = os.environ.get("OUT_PATH",   "submission.csv")

    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    # sanity
    assert "address" in train.columns and "label" in train.columns, "train.csv: 'address' ve 'label' kolonları olmalı"
    assert "address" in test.columns and "id" in test.columns, "test.csv: 'address' ve 'id' kolonları olmalı"

    # geocoder
    geolocator = Nominatim(user_agent="adres-cozumleme-gis")
    geocode_rl = RateLimiter(geolocator.geocode, min_delay_seconds=1.2)  # Nominatim'e saygı

    cache = load_cache()

    # geocode train
    train["address_norm"] = train["address"].astype(str).map(normalize_text)
    train["lat"] = None
    train["lon"] = None
    for i, row in tqdm(train.iterrows(), total=len(train), desc="Geocoding train"):
        res = geocode_address(geocode_rl, row["address"], cache)
        if res:
            train.at[i,"lat"] = res["lat"]
            train.at[i,"lon"] = res["lon"]
    save_cache(cache)
    train_geo = train.dropna(subset=["lat","lon"]).copy()
    if train_geo.empty:
        raise SystemExit("Geocode başarısız: Train konumları bulunamadı. Lütfen adres formatını iyileştirin.")

    # KDTree kur
    tree = build_kdtree(train_geo)

    # geocode test
    test["address_norm"] = test["address"].astype(str).map(normalize_text)
    test["lat"] = None
    test["lon"] = None
    for i, row in tqdm(test.iterrows(), total=len(test), desc="Geocoding test"):
        res = geocode_address(geocode_rl, row["address"], cache)
        if res:
            test.at[i,"lat"] = res["lat"]
            test.at[i,"lon"] = res["lon"]
    save_cache(cache)

    # tahmin
    preds = []
    coords_train = train_geo[["lat","lon"]].values
    K = int(os.environ.get("KNN_K", 10))
    for i, row in tqdm(test.iterrows(), total=len(test), desc="Scoring"):
        if pd.isna(row["lat"]) or pd.isna(row["lon"]):
            # fallback: en benzer metin (global)
            sims = train["address_norm"].map(lambda s: fuzz.token_set_ratio(row["address_norm"], s))
            best_idx = sims.idxmax()
            preds.append(int(train.loc[best_idx,"label"]))
            continue

        # KDTree: en yakın K komşu (euclidean on degrees ~ kabul edilebilir)
        dists, idxs = tree.query([[row["lat"], row["lon"]]], k=min(K, len(coords_train)), return_distance=True)
        dists = dists[0]; idxs = idxs[0]
        best_score = -1.0
        best_label = None
        for dist, idx in zip(dists, idxs):
            cand = train_geo.iloc[idx]
            # gerçek km için haversine
            dist_km = haversine((row["lat"], row["lon"]), (cand["lat"], cand["lon"]), unit=Unit.KILOMETERS)
            text_sim = fuzz.token_set_ratio(row["address_norm"], cand["address_norm"])
            score = combined_score(dist_km, text_sim)
            if score > best_score:
                best_score = score
                best_label = int(cand["label"])
        if best_label is None:
            # aşırı uç durum: yine metin benzerliği
            sims = train["address_norm"].map(lambda s: fuzz.token_set_ratio(row["address_norm"], s))
            best_idx = sims.idxmax()
            best_label = int(train.loc[best_idx,"label"])
        preds.append(best_label)

    sub = pd.DataFrame({"id": test["id"], "label": preds})
    sub.to_csv(out_path, index=False)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
