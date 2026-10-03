# data/

- `raw/` and `cache/` are git-ignored (datasets are downloaded by `scripts/prepare_*.py`; never commit them).
- `splits/` (`pets_split.json`, `fs2k_split.json`) and `manifests/` (`pets_val_manifest.jsonl`, `pets_test_manifest.jsonl`, with `.sha256`) are **committed** so every device uses identical splits and corruptions.
