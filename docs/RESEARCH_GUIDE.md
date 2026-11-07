# Research methods and reproducibility guide

[Overview](../README.md) · [Türkçe](../README.tr.md) ·
[Notebook source inventory](EXPERIMENTS.md)

This guide preserves the earlier README's problem, method and evaluation detail.
Descriptions below distinguish source inventory from proposed evaluation and
future work. No current common benchmark or production deployment is asserted.

## Experiment directory index

All 25 historical directories are listed. Labels summarize the family; inspect
source for the actual model combination. Names such as `xgb` and `lgbm` are not
proof that those estimators are trained in that version.

| Directory | Family / distinction |
|---|---|
| [algorithms/baseline](../algorithms/baseline/) | Administrative dictionary / heuristic source |
| [algorithms/baseline-v2](../algorithms/baseline-v2/) | Test driver with unresolved `untitled13` import |
| [algorithms/baseline-v3](../algorithms/baseline-v3/) | Separate Python baseline implementation |
| [notebooks/baseline-v4](../notebooks/baseline-v4/) | Notebook baseline variant |
| [notebooks/baseline-v5](../notebooks/baseline-v5/) | Notebook baseline variant |
| [algorithms/bert-fuzzy](../algorithms/bert-fuzzy/) | Python preprocessing / transformer-fuzzy family |
| [notebooks/bert-fuzzy-v2](../notebooks/bert-fuzzy-v2/) | Separate preprocessing/matching notebook |
| [notebooks/bert-faiss-xgb-fuzzy](../notebooks/bert-faiss-xgb-fuzzy/) | Embedding retrieval; source inventory detects LightGBM |
| [notebooks/bert-torch-cosine](../notebooks/bert-torch-cosine/) | Embedding/neural similarity variant |
| [notebooks/bert-faiss-rf-fuzzy](../notebooks/bert-faiss-rf-fuzzy/) | Embeddings, TF-IDF and multiple classifier symbols |
| [notebooks/bert-faiss-rf-fuzzy-v2](../notebooks/bert-faiss-rf-fuzzy-v2/) | Separate hybrid variant; known indentation issue |
| [notebooks/bert-faiss-lgbm-cosine](../notebooks/bert-faiss-lgbm-cosine/) | Embedding/TF-IDF retrieval variant |
| [notebooks/bert-faiss-lgbm-cosine-v2](../notebooks/bert-faiss-lgbm-cosine-v2/) | Separate retrieval variant |
| [notebooks/bert-faiss-lgbm-cosine-v3](../notebooks/bert-faiss-lgbm-cosine-v3/) | Retrieval/feature variant with model artefact references |
| [notebooks/bert-faiss-lgbm-cosine-v4](../notebooks/bert-faiss-lgbm-cosine-v4/) | Multiple submission/analysis output paths |
| [notebooks/tfidf](../notebooks/tfidf/) | TF-IDF/LSTM notebook; known incomplete import |
| [notebooks/tfidf-v2](../notebooks/tfidf-v2/) | Separate LSTM notebook; inspect task and data before reuse |
| [notebooks/tfidf-cosine](../notebooks/tfidf-cosine/) | Sparse text similarity family |
| [notebooks/tfidf-rf-fuzzy](../notebooks/tfidf-rf-fuzzy/) | TF-IDF plus tree/boosting classifier symbols |
| [notebooks/tfidf-rf-fuzzy-v2](../notebooks/tfidf-rf-fuzzy-v2/) | Separate multi-classifier variant |
| [notebooks/tfidf-faiss](../notebooks/tfidf-faiss/) | Sparse-vector nearest-neighbour family |
| [notebooks/tfidf-logreg](../notebooks/tfidf-logreg/) | LogisticRegression detected; stacking output referenced |
| [notebooks/tfidf-torch](../notebooks/tfidf-torch/) | Neural variant with `.pth` artefact references |
| [algorithms/gis-geocode-hybrid](../algorithms/gis-geocode-hybrid/) | Geocoding, geographic candidates, text reranking |
| [algorithms/gis-spatial-join-normalizer](../algorithms/gis-spatial-join-normalizer/) | Administrative polygons and label lookup |

The notebook inventory records detected symbols and referenced filenames for all
19 notebooks. Some references are outputs, not prerequisites. Version numbers
are historical labels; these folders do not share an established benchmark.

## Inputs, outputs and preprocessing

| Artefact | Common contract | Caveat |
|---|---|---|
| Training CSV | `address` string, `label` task-specific class/cluster | Missing from checkout; labels are not a universal address ontology |
| Test CSV | `id`, `address` | Included data's rights/privacy remain unresolved |
| Submission CSV | Usually `id,label` | Filename/location varies by implementation |
| GIS boundaries | `turkey_admin_boundaries.geojson`, administrative fields `il`, `ilce`, `mahalle` | External data; confirm rights, field names and coordinate reference system |
| Models/caches | Per-notebook model names and generated files | May require downloads; caches can retain address text |

Text experiments vary in lowercasing, Unicode/diacritic normalization, regex
cleanup, abbreviation expansion, dictionaries and feature extraction. Do not
assume that removing diacritics is always beneficial: Turkish letter distinctions
and repeated administrative names can change matching behavior. Record the exact
preprocessing order with each result.

The historical objective of component normalization is broader than label
prediction. A correct competition label does not establish correct extraction of
province, district, neighbourhood, street, building number and apartment number.

## GIS paths in the current source

`gis-geocode-hybrid/main.py` geocodes addresses with Nominatim through a rate
limiter and JSON cache. It builds a KDTree over available training coordinates,
selects candidates, and combines geographic and fuzzy text scores (default
weights 0.6 / 0.4). Candidate search uses coordinate-space proximity; later
scoring uses haversine distance. Missing coordinates and fallback handling need
review before interpreting output quality.

`gis-spatial-join-normalizer/main.py` reads administrative boundary polygons,
geocodes test points and uses `geopandas.sjoin` to derive an administrative key.
It maps keys to training labels using the available training information.
Inspect its field validation, fallback behavior and geometry assumptions before
running it.

Both scripts can transmit addresses to a third party. The implemented delay is
not certification of compliance with a provider's current service policy. Obtain
permission and review current provider requirements before a run; do not use the
included address file as a default network smoke test.

## Proposed evaluation protocol

1. Document authorized dataset provenance/version and the meaning of `label`.
2. Fix a common split and seeds; keep near-duplicate addresses together and
   consider grouping by administrative unit to examine geographic leakage.
3. Fit learned transforms on training folds, then apply them to validation data.
   Record whether an experiment uses exact lookup, retrieval or a trained model.
4. Compare classification Accuracy and Macro/Micro-F1 where meaningful. Evaluate
   normalization separately with component accuracy and text/edit-distance errors.
5. Categorize errors: wrong administrative level, abbreviation confusion,
   neighbouring-area near misses, common-name bias and missing-field failures.
6. Record hardware, model/data downloads, runtime, memory and batch size. Measure
   latency percentiles only under a defined workload; no p99 claim is made here.

Saved notebook metrics and filenames such as `0.7plus` are historical artefacts,
not independently verified results. No performance leaderboard is supplied.

## Reproduction gaps and future work

Training data is absent, paths vary between Colab/local runs, dependencies are
unpinned, `baseline-v2` has an unresolved import, and two notebooks have static
syntax findings documented in the root README. Fix these in separately reviewed
experiment changes before presenting a reproducible end-to-end result.

Earlier ideas included richer Turkish rules, refreshed gazetteers, sequence
labeling of address components, NLP/GIS ensembles, caching and fallback layers.
They remain candidate research/engineering work, not implemented capabilities of
every experiment or a promise of production readiness.
