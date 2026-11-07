# Turkish address parsing and matching experiments

An archive of Turkish-address normalization and task-specific label prediction
experiments from the Teknofest/Hepsiburada hackathon. It preserves competing
approaches and historical notebooks, rather than offering one production parser.

## Experiment families

| Family | Where to start | Main distinction |
|---|---|---|
| Rules / baselines | `algorithms/baseline*`, `notebooks/baseline-*` | Administrative dictionaries and text normalization |
| TF-IDF | `notebooks/tfidf*` | Sparse text features with cosine/retrieval, classifiers or neural variants |
| Transformer retrieval | `algorithms/bert-fuzzy`, `notebooks/bert-*` | Embeddings plus FAISS, fuzzy/cosine similarity and optional boosted models |
| GIS | `algorithms/gis-geocode-hybrid`, `gis-spatial-join-normalizer` | Geocoding and geographic-neighbour/admin-boundary matching |

[Notebook inventory](docs/EXPERIMENTS.md) lists models and file requirements found
in the code. Version suffixes are separate experiments, not a verified ranking.
Some folder names overstate the exact model used; inspect source before selecting
a run. Existing per-folder READMEs provide historical context.

## Environment and input contracts

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

The broad, unpinned dependency list includes ML and GIS packages that may require
platform-specific builds. Not every experiment shares a tested environment.
Common inputs are `train.csv` (`address`, `label`) and `test.csv` (`id`, `address`);
individual code may expect Colab/local paths or additional boundary/model files.
The training CSV is not included. GIS normalization may also require authorized
administrative boundary data. Review a selected script's path constants and
README before running it; there is no universal root training command.

## Known execution gaps

- `algorithms/baseline-v2/test_pipeline.py` imports `untitled13`, absent from that
  directory. A differently named implementation exists in `baseline-v3`; it must
  not be silently substituted as an equivalent experiment.
- Static checks found an indentation error in `bert-faiss-rf-fuzzy-v2` notebook
  cell 3 and an incomplete import in `tfidf` notebook cell 8.
- Dataset/model downloads, GPU work and Nominatim calls were not executed in the
  audit. No current accuracy/F1 comparison was reproduced under common conditions.

## Data rights and privacy

The public checkout already contains `data/test.csv` with 217,241 address rows.
Its redistribution rights and personal-data status require owner review; the
audit does not certify those rights and did not copy row values into reports.
Notebook outputs can also contain address examples. Do not add new private
addresses or send them to geocoding services without permission. Apply provider
usage limits and review retained caches/outputs. Data files and notebook history
have not been deleted or rewritten by this documentation change.


## License and contact

See [LICENSE](LICENSE) for the existing MIT license. Author: [Muhammet Fatih Çetintaş](https://github.com/M-Fatih-C).
