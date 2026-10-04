# Trained inference models — Generative AI Assignment 1

Seven trained ONNX models for Universal Restoration, Hard-Routed Restoration, Soft Mixture-of-Experts Restoration, and Face-to-Sketch Generator. Source branch: `dev`.

Download all seven `.onnx` assets into `models/onnx/`. File sizes and SHA-256 hashes are recorded in the attached `MANIFEST.json`, in each model's `onnx` field.

For a repository collaborator using authenticated GitHub CLI:

```powershell
gh release download models-v1 --repo AhmedLaiq34/genai-assignment-1 --pattern '*.onnx' --dir models/onnx
python scripts/fetch_models.py --verify-only
docker compose up --build
```

For a public release, the downloader can also use:

```powershell
python scripts/fetch_models.py --release-base https://github.com/AhmedLaiq34/genai-assignment-1/releases/download/models-v1
```

Open the application at http://localhost:8080. Private releases require repository access; anonymous HTTPS downloads are not available while the repository is private. No dataset or API credentials are included in these assets.
