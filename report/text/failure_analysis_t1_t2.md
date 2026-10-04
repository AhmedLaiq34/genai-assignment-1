# Failure analysis, Tasks 1 and 2 (report-ready text)

All numbers come from the saved test-set evaluations (Task 1: `artifacts/eval/task1/20261004-1514_kaggle_t1_final_v2_test`, Task 2: `artifacts/eval/task2/20261004-234854_test`) and from `tools/failure_analysis_t1_t2.py`, which writes `report/tables/failure_analysis_t1_properties.csv`, `failure_analysis_t1_worst4.csv` and `failure_analysis_t2_misroutes.csv`. J = 0.5 L1 + 0.5 (1 - SSIM), lower is better; "J_input" is the J of leaving the corrupted input unchanged.
Two image properties were measured per test image: the **dark fraction** (share of pixels with mean RGB below 0.06) and the **texture** (variance of the Laplacian of the grey image, a measure of fine detail). A "dark image" below means a dark fraction of at least 0.30 (103 of 3,669 test images).
Figures: `report/figures/t1_test_worst_4.png` (Task 1, columns: target, corrupted input, output, absolute error) and `report/figures/task2/test/routing_failures_worst.png` (Task 2, columns: target, input, oracle output, predicted-route output, absolute error). Use the files in `report/figures/task2/test/` (test set); the same-named file one folder up is from validation data.

## Task 1: four failure cases

All four worst test cases are high-severity occlusion (three rectangles, about 35% of the image). In every one, the restored image is **worse than the occluded input** (J above J_input).

| Image | J | J_input | SSIM | Dark fraction | Texture |
|---|---|---|---|---|---|
| Egyptian_Mau_60 | 0.463 | 0.116 | 0.307 | 0.69 | 0.044 |
| Egyptian_Mau_98 | 0.416 | 0.176 | 0.304 | 0.52 | 0.023 |
| Bengal_19 | 0.406 | 0.260 | 0.317 | 0.32 | 0.040 |
| german_shorthaired_73 | 0.393 | 0.278 | 0.338 | 0.01 | 0.103 |

Test-set reference values: mean dark fraction 0.047; texture median 0.016, 95th percentile 0.049, 99th percentile 0.071.

**Cases 1 to 3 (cats on black backgrounds): the black occluder cannot be told apart from real black.** The occlusion corruption paints rectangles with value 0. When a large part of the photograph is truly black, the network cannot know which black pixels were erased. The outputs show it: the black background is overpainted with a brown or grey fill, and the error map is white over the former background (rows 1 to 3 of the figure). The clearest evidence is Egyptian_Mau_60: even **without any corruption** the model reconstructs it with J = 0.389 (SSIM 0.41), against a clean-set average of 0.105. The Task 2 classifier agrees: it gives this clean image p(occlusion) = 0.999999 (section below). This explanation fits all the measurements but was not tested directly (for example by repeating the occlusion with another colour).

**Case 4 (german_shorthaired_73): fine texture.** This image has almost no black (0.01) but very high detail (texture 0.103, above the 99th percentile). The output replaces the covered grass and flowers with smooth brown-grey blobs, and the error map is coloured high-frequency noise: that is the lost texture itself.

**Common cause: a compact latent keeps layout and colour, not detail.** The bottleneck is small, and the L1 + SSIM loss rewards the safest guess for anything the latent cannot predict, which is a smooth average colour. This is not limited to the four worst images: the per-image J correlates with texture at r = 0.74 (clean), 0.80 (high blur), 0.77 (medium occlusion) and 0.78 (high occlusion), while its correlation with the dark fraction stays below 0.1 in absolute value. The most textured quarter of the images has about twice the J of the least textured quarter (clean 0.153 vs 0.066; high occlusion 0.271 vs 0.171). Dark images are a smaller group of extreme cases, not a general trend: under high occlusion they average J = 0.243 against 0.217 for the other images.

What the section should conclude: Task 1's weakest area is large occlusion on textured or black-background pictures, because the model has to invent content that the latent does not carry. A possible remedy is limited skip connections or a larger latent (not tried, listed as future work), and a corruption value other than pure black would remove the black-background ambiguity.

## Task 2: classifier-caused routing failures

On the 36,690 test rows, the predicted route differs from the oracle route in **335 rows (0.91%)**. They fall into two groups with opposite effects on J (`J_drop` = predicted-route J minus oracle-route J; positive means the misroute hurt).

| True class | Predicted | Rows | Mean J_drop | Dark fraction | Texture | What happens |
|---|---|---|---|---|---|---|
| clean | occlusion | 37 | +0.215 | 0.339 | 0.029 | black areas read as occluders; the occlusion specialist overpaints real content |
| clean | blur | 64 | +0.085 | 0.026 | 0.008 | very smooth, low-detail photographs look blurred; the blur specialist smooths them further |
| blur | occlusion | 5 | +0.194 | 0.629 | 0.033 | same dark-background confusion |
| occlusion | gaussian_blur | 7 | +0.043 | 0.058 | 0.009 | small effect |
| occlusion | clean | 208 | -0.085 | 0.088 | 0.027 | 207 of the 208 are low severity (one rectangle, about 10% of the image); restoration is skipped |
| blur | clean | 10 | -0.123 | 0.090 | 0.060 | low severity, restoration skipped |
| salt_pepper | clean | 4 | -0.117 | 0.049 | 0.091 | low severity, restoration skipped |

Test images average a dark fraction of 0.047 and a texture of 0.020.

**Harmful errors: clean images sent to a specialist (101 of 3,669 clean images, 2.75%; mean J_drop +0.133, up to +0.43).** The identity bypass is only available when the classifier says "clean". Whenever the classifier calls a clean image corrupted, a specialist modifies a picture that needed nothing, so the system ends up worse than doing nothing. The worst eight cases (figure rows 1 to 8) are all cats on black or letterboxed backgrounds, which the classifier reads as occlusion with p(occlusion) between 0.82 and 0.999999 (mean 0.875 over the 37 clean-to-occlusion cases). The occlusion specialist then fills the black background with a brown or beige estimate, which is exactly the Task 1 failure of the black-background cases; here it is triggered by the classifier rather than by a real occluder. The mean dark fraction of these 37 images is 0.339, about seven times the test average. The clean-to-blur errors are milder (+0.085) and come from images with very little detail (texture 0.008, well below the 0.020 average).

**Skipped restoration (222 rows) does not raise J.** For these, the predicted-route J is *lower* than the oracle J (mean J_drop -0.088). The misses are nearly all low-severity corruptions (207 of the 208 occlusion-to-clean cases), and for low-severity damage the specialist's output is on average worse than the corrupted input itself (consistent with Task 1, where the output was worse than the input for 3,035 of 3,669 low-severity occlusion images). The classifier is therefore not failing in a way that matters for J here. One caveat: J measures fidelity, not visual completeness; a skipped occlusion leaves a black rectangle in the picture, which a viewer would call an unrestored image.

**Conclusion for the section.** The classifier is accurate (per-class recall between 97.2% for clean and 100% for salt-and-pepper), so routing errors are rare, but their cost is asymmetric: errors that send clean or dark-background images to a specialist cause the large losses, while missed low-severity corruptions cost nothing in J. The weak point is the shared ambiguity between true black content and a black occluder, which affects both the Task 1 model and the Task 2 classifier. This is a motivation for the soft routing of Task 3, where uncertain inputs can keep part of the identity branch.
