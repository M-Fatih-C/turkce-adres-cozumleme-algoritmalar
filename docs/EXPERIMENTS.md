# Notebook experiment inventory

Static source inventory, 7 November 2026. No common benchmark was rerun. Model names below are symbols detected in source, not a claim that every model was trained. Data paths differ between experiments; listed files can include generated outputs as well as inputs. Inspect their read/write calls.

| Notebook | Model/feature symbols | Referenced local artefacts |
|---|---|---|
| `notebooks/baseline-v4/Adres çöümleme.ipynb` | rules/custom implementation; inspect source | paths defined dynamically/in code |
| `notebooks/baseline-v5/Untitled11.ipynb` | rules/custom implementation; inspect source | paths defined dynamically/in code |
| `notebooks/bert-faiss-lgbm-cosine/Untitled12.ipynb` | SentenceTransformer, TfidfVectorizer | `baseline_submission.csv`, `submission.csv`, `test.csv`, `train.csv` |
| `notebooks/bert-faiss-lgbm-cosine-v2/Untitled13.ipynb` | SentenceTransformer, TfidfVectorizer | `baseline_submission.csv`, `test.csv`, `train.csv` |
| `notebooks/bert-faiss-lgbm-cosine-v3/Untitled7.ipynb` | SentenceTransformer, TfidfVectorizer | `lgb_fold0.txt`, `submission.csv`, `test.csv`, `train.csv` |
| `notebooks/bert-faiss-lgbm-cosine-v4/adresçözümleme.ipynb` | SentenceTransformer, TfidfVectorizer | `baseline_submission.csv`, `embedding_submission.csv`, `final_submission.csv`, `prediction_analysis.csv`, `test.csv`, `train.csv` |
| `notebooks/bert-faiss-rf-fuzzy/Untitled3.ipynb` | AutoModel, CatBoostClassifier, LGBMClassifier, RandomForestClassifier, SentenceTransformer, TfidfVectorizer, XGBClassifier | `gpu_optimized_submission.csv`, `test.csv`, `train.csv` |
| `notebooks/bert-faiss-rf-fuzzy-v2/Untitled6.ipynb` | CatBoostClassifier, LGBMClassifier, RandomForestClassifier, SentenceTransformer, TfidfVectorizer, XGBClassifier | `hybrid_submission.csv`, `test.csv`, `train.csv` |
| `notebooks/bert-faiss-xgb-fuzzy/turkish_address_pipeline.py.ipynb` | LGBMClassifier, SentenceTransformer | `sample_submission.csv`, `test.csv`, `train.csv` |
| `notebooks/bert-fuzzy-v2/Untitled14.ipynb` | rules/custom implementation; inspect source | `address_label_map.csv`, `preprocessed_addresses.csv`, `sample_preprocessed_addresses.csv`, `sample_submission.csv`, `submission.csv`, `test.csv`, `test_preprocessed.csv`, `train.csv`, `train_preprocessed.csv`, `unmatched_test_preview.csv` |
| `notebooks/bert-torch-cosine/Untitled9.ipynb` | SentenceTransformer | `submission_gpu.csv`, `submission_memory_efficient.csv`, `submission_ultra_fast.csv`, `test.csv`, `train.csv` |
| `notebooks/tfidf/Turkce_TF_IDF_LSTM_Notebook.ipynb` | LSTM, TfidfVectorizer | paths defined dynamically/in code |
| `notebooks/tfidf-cosine/Untitled8.ipynb` | TfidfVectorizer | `submission.csv`, `test.csv`, `train.csv` |
| `notebooks/tfidf-faiss/Untitled10.ipynb` | TfidfVectorizer | `submission.csv`, `test.csv`, `train.csv` |
| `notebooks/tfidf-logreg/Notebook 2-2.ipynb` | LogisticRegression, TfidfVectorizer | `submission_stack.csv`, `test.csv`, `train.csv` |
| `notebooks/tfidf-rf-fuzzy/Untitled4.ipynb` | CatBoostClassifier, LGBMClassifier, RandomForestClassifier, TfidfVectorizer, XGBClassifier | `advanced_submission.csv`, `test.csv`, `train.csv` |
| `notebooks/tfidf-rf-fuzzy-v2/Untitled5.ipynb` | CatBoostClassifier, LGBMClassifier, RandomForestClassifier, TfidfVectorizer, XGBClassifier | `advanced_submission.csv`, `test.csv`, `train.csv` |
| `notebooks/tfidf-torch/Notebook 1.ipynb` | TfidfVectorizer | `best_model.pth`, `submission_0.7plus.csv`, `submission_optimized.csv`, `test.csv`, `train.csv` |
| `notebooks/tfidf-v2/Turkce_TF_IDF_LSTM_Notebook_2.ipynb` | LSTM, TfidfVectorizer | `duygu_modeli.h5` |

The v2/v3/v4 variants change retrieval/feature/model mixes and output handling. They are not duplicate runs with a shared evaluation protocol. In particular, TF-IDF neural notebooks, boosted retrieval notebooks and GIS paths solve different intermediate tasks. Historical output metrics are not directly comparable without matching data splits, seeds and hardware.
