# Run only after the student approves publication, as required by the audit prompt.
$ErrorActionPreference = 'Stop'
$modelRepoRoot = Split-Path -Parent $PSScriptRoot
$modelAssetNames = @(
    't1_universal_ae.onnx',
    't2_classifier.onnx',
    't2_ae_salt.onnx',
    't2_ae_blur.onnx',
    't2_ae_occlusion.onnx',
    't3_soft_moe.onnx',
    't4_generator.onnx'
)
$modelAssets = foreach ($modelAssetName in $modelAssetNames) {
    $modelAssetPath = Join-Path $modelRepoRoot "models/onnx/$modelAssetName"
    if (-not (Test-Path -LiteralPath $modelAssetPath -PathType Leaf)) {
        throw "Missing release asset: $modelAssetName"
    }
    $modelAssetPath
}
$modelAssets += Join-Path $modelRepoRoot 'models/MANIFEST.json'
$modelReleaseNotes = Join-Path $modelRepoRoot 'docs/MODEL_RELEASE_NOTES.md'
& gh release create models-v1 @modelAssets --repo AhmedLaiq34/genai-assignment-1 --target dev --title 'Assignment 1 trained ONNX models' --notes-file $modelReleaseNotes
if ($LASTEXITCODE -ne 0) { throw 'Model release creation failed.' }
