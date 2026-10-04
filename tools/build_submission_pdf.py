"""Generate the submission PDF directly, in IEEE-style two-column format.
Requires ReportLab and Pillow; reads saved evidence only.
"""
import argparse
import csv
import html
import json
from pathlib import Path
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Table, TableStyle, Image, Spacer, KeepTogether, NextPageTemplate, PageBreak
from reportlab.graphics.shapes import Drawing, Rect, Line, String

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'report'
FULL, COL = 504, 243
styles = {
 'body': ParagraphStyle('body',fontName='Times-Roman',fontSize=10,leading=12,alignment=4,spaceAfter=5),
 'head': ParagraphStyle('head',fontName='Times-Roman',fontSize=10,leading=12,alignment=1,spaceBefore=8,spaceAfter=6),
 'sub': ParagraphStyle('sub',fontName='Times-Italic',fontSize=10,leading=12,spaceBefore=6,spaceAfter=4),
 'small': ParagraphStyle('small',fontName='Times-Roman',fontSize=8,leading=9.5,spaceAfter=4),
 'caption': ParagraphStyle('caption',fontName='Times-Roman',fontSize=8,leading=9.5,alignment=1,spaceAfter=7),
 'cell': ParagraphStyle('cell',fontName='Times-Roman',fontSize=7.2,leading=8.3),
 'title': ParagraphStyle('title',fontName='Times-Roman',fontSize=22,leading=25,alignment=1),
}
story=[]
sources=[]
figures=0
tables=0

def P(text,style='body'):
 return Paragraph(text,styles[style])
def body(text): story.append(P(text))
def head(text): story.append(P(text,'head'))
def sub(text): story.append(P(text,'sub'))
def rows(path):
 with (ROOT/path).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def tab(title,headers,data,widths=None):
 global tables
 tables+=1
 story.append(P(f'TABLE {tables}<br/>{title.upper()}','caption'))
 data=[[P(html.escape(str(v)),'cell') for v in r] for r in [headers]+data]
 t=Table(data,colWidths=widths or [COL/len(headers)]*len(headers),repeatRows=1,hAlign='CENTER')
 t.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),.6,colors.black),('LINEBELOW',(0,0),(-1,0),.4,colors.black),('LINEBELOW',(0,-1),(-1,-1),.6,colors.black),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]))
 story.extend([t,Spacer(1,6)])
def image(path,width=COL,height=230):
 with PILImage.open(ROOT/path) as im: w,h=im.size
 s=min(width/w,height/h)
 return Image(str(ROOT/path),width=w*s,height=h*s,hAlign='CENTER')
def fig(path,caption,wide=False,height=230):
 global figures
 figures+=1; sources.append(path)
 story.append(KeepTogether([image(path,FULL if wide else COL,height),P(f'Fig. {figures}. {caption}','caption')]))
def eq(text): story.append(P(text,'caption'))

def main():
 global figures
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--youtube-url',default='')
 ap.add_argument('--model-url',default='')
 args=ap.parse_args()
 body('<b>Abstract—</b>Four image systems are implemented and deployed: a universal denoising autoencoder, a corruption classifier with hard-routed specialist autoencoders, a jointly trained soft mixture of experts, and a style-conditioned face-to-sketch conditional GAN. Oxford-IIIT Pet images support restoration and FS2K supports paired generation. On the saved 36,690-case restoration test protocol, composite reconstruction score J decreases from 0.1898 for unchanged inputs to 0.1330 for universal restoration, 0.1276 for predicted hard routing and 0.1142 for soft routing. These averages conceal failures on clean and mildly damaged images. The sketch generator achieves test L1 0.1102 and SSIM 0.460, with Style 2 the weakest category. Seven ONNX models are integrated into a React/FastAPI Docker application. This report presents methodology, hyperparameter selection, quantitative results, failure analysis and deployment evidence from the existing project records.')
 body('<b>Index Terms—</b>Image restoration, autoencoders, mixture of experts, conditional GAN, Optuna, ONNX.')
 head('I. INTRODUCTION AND RELATED WORK')
 body('Restoration must remove damage while preserving valid content. A universal model shares one representation across corruptions; specialists can improve focus but introduce dependence on classification. Soft routing combines identity and specialist outputs differentiably. Paired sketch generation adds learned categorical style conditioning to a different image-to-image task. The project compares these choices and exposes all four through one application.')
 body('Denoising autoencoders learn clean reconstructions from corrupted inputs [1]. SSIM supplements pixel error with structural fidelity [2]. Mixture-of-experts research motivates learned routing and utilization regularization [3]; the present system uses four dense image branches rather than sparse token dispatch. U-Net [4] supplies the encoder–decoder/skip pattern for paired generation, while pix2pix [5] motivates a conditional GAN and PatchGAN discriminator. Optuna [6] supports validation-based search, and ONNX [7] supports inference interchange.')
 body('A dense latent was investigated before selecting a convolutional grid. This decision follows validation diagnostics rather than the test set. Numerical claims below come from saved CSVs, final configurations and phase reports; no training, tuning or official-test evaluation was repeated for report preparation.')
 head('II. DATA PREPARATION AND EVALUATION')
 sub('A. Oxford-IIIT Pet')
 body('The official trainval collection is split 80/20 with seed 42 into 2,944 training and 736 validation images. The official 3,669-image test collection is reserved for final evaluation. Images are RGB, 128 × 128, and the same split is used across Tasks 1–3. Deterministic validation manifests contain four cases per image (2,944 rows); deterministic test manifests contain clean plus three severities for each corruption (36,690 rows). Split files and manifest SHA-256 sidecars are retained.')
 tab('Dataset split counts',['Dataset','Train','Val.','Test'],[['Oxford pets','2,944','736','3,669'],['FS2K','899','159','1,046']],[81,54,54,54])
 sub('B. Exact corruption configuration')
 body('Training applies a fresh corruption at each image load, with equal probability for clean, salt-and-pepper, Gaussian blur and rectangular occlusion. Salt probability is uniform in [0.02, 0.15]; selected pixels become black or white with equal probability. Blur kernels are sampled uniformly from {3, 5, 7}, with sigma uniform in [0.5, 2.5] and reflect padding. Occlusion uses one to three black rectangles at random positions; their union covers a uniformly sampled 10–35% area. Overlap is accounted for by union coverage. Corrupted copies are not permanently stored.')
 tab('Fixed official-test severities',['Setting','Low','Medium','High'],[['Salt probability','.03','.08','.15'],['Blur (kernel, sigma)','(3, .7)','(5, 1.5)','(7, 2.5)'],['Mask rectangles','1','2','3'],['Mask union coverage','~10%','~20%','~35%']],[90,51,51,51])
 body('Validation stores deterministic sampled settings from the training distributions; low/medium/high groups are parameter tertiles. The fixed settings in Table 2 apply to the official test manifest. Seeds, mask coordinates and blur settings make repeated evaluation comparable. The test average gives 10% weight to clean cases and 30% to each corrupted condition, not an equally weighted four-condition average.')
 fig('report/figures/corruption_examples.png','Existing corruption examples; training sampling is dynamic while evaluation settings are fixed.')
 sub('C. FS2K and paired augmentation')
 body('FS2K contains 2,104 photo/sketch pairs. Its official 1,058-pair training portion is split into 899 training and 159 validation pairs using 15% style-stratified validation with seed 42. The official test contains 1,046 pairs. Training crops paired 143 × 143 images to 128 × 128 and applies identical horizontal flips. Validation/test are resized to 128 × 128. Photo and grayscale sketch tensors use [-1, 1]; style indices 0/1/2 appear as Style 1/2/3 in the interface.')
 sub('D. Metrics, selection and test use')
 eq('J = 0.5 L1 + 0.5 (1 − SSIM)')
 body('L1/MAE and J are lower-is-better; SSIM and PSNR are higher-is-better. J is fixed across reconstruction comparisons, independent of the trained L1/SSIM weight. Classifier selection uses validation macro-F1, restoration uses validation J, and GAN selection uses validation L1 on [0, 1]-rescaled sketches. Identity outputs can yield infinite PSNR, so J/SSIM carry the main cross-system comparison. The saved access log records final evaluations on October 4, 2026 at 16:07:12 (Task 1), 18:48:54 (Task 2), 20:10:16 (Task 4) and 20:22:19 UTC (Task 3); phase reports attribute these to approved final runs. This build reads their saved outputs only.')
 head('III. TASK 1: UNIVERSAL RESTORATION')
 sub('A. Architecture and loss')
 body('The universal autoencoder has 1,322,507 parameters. Three stride-2 encoder convolutions reduce 128 → 64 → 32 → 16 spatial dimensions with 64, 128 and 256 channels. A 1 × 1 projection produces a 16 × 16 × 8 latent grid (2,048 values), giving 24:1 compression from 49,152 RGB input values. A projection and three transposed convolutions reconstruct RGB with a sigmoid. No skip connections bypass this genuine bottleneck.')
 eq('L_AE = alpha L1 + (1 − alpha)(1 − SSIM)')
 fig('report/figures/t1_architecture.png','Task 1 compressed convolutional autoencoder, without unrestricted skip connections.')
 sub('B. Optuna and final training')
 body('Search ranges: learning rate 10^-4–.003 (log), batch {32, 64, 128}, bottleneck {1,024, 2,048, 4,096}, base encoder channels {16, 32, 48, 64}, dropout [0, .3], and alpha [.5, .95]. TPE/median pruning runs 15 epochs per trial; 35 of 40 trials complete and five are pruned. Best trial 33 has validation J .1406. Final lr .00230389, batch 32, latent 2,048, base 64, dropout .147685 and alpha .502202 are used for 100 epochs with AdamW, weight decay 10^-4 and cosine scheduling. Best epoch 98 gives validation J .1237, SSIM .791 and PSNR 25.17 dB.')
 body('The earlier dense-vector model reached validation J .3134, near an 8 × 8 thumbnail baseline (.3131). Increasing its latent did not resolve the smooth reconstructions, while convolutional-grid prototypes reduced validation error substantially. This motivated the final grid architecture. Test performance is J .1330, SSIM .775 and PSNR 24.62 dB. Salt restoration improves strongly, but low blur, low occlusion and clean inputs can be unnecessarily modified.')
 names={'clean':'Clean','salt_pepper':'Salt','gaussian_blur':'Blur','occlusion':'Mask'}
 r1=rows('report/tables/t1_test_results.csv')
 tab('Task 1 test results',['Input','Level','MAE','SSIM','J'],[[names[r['cond']],r['severity'],f"{float(r['MAE']):.4f}",f"{float(r['SSIM']):.3f}",f"{float(r['J']):.4f}"] for r in r1 if r['severity']!='all'],[47,43,48,48,57])
 sub('C. Four discussed failures')
 body('All four documented worst cases have high occlusion: Egyptian_Mau_60 (J .463 versus input .116), Egyptian_Mau_98 (.416 versus .176), Bengal_19 (.406 versus .260), and german_shorthaired_73 (.393 versus .278). The first three have genuine black regions that are ambiguous with black masks; the fourth has very high texture (.103, above the saved 99th percentile). Reconstruction substitutes smooth estimates for detail. This explanation fits saved measurements but was not tested through a controlled mask-color intervention.')
 body('Per-image error correlates with texture (r = .78 for high occlusion); dark-background images form a smaller extreme group. A compact latent retains layout/color more reliably than fine detail. Larger latents, justified limited skips, and testing a different mask color are future work. Twelve representative examples with target/input/output/error panels and the four failure cases are included in the visual appendix.')
 head('IV. TASK 2: HARD-ROUTED SPECIALISTS')
 body('A four-class CNN learns corruption labels with cross-entropy and balanced batches. Its probabilities select argmax routing. Clean predictions use an identity bypass; three independently trained specialists restore salt, blur or occlusion. Each specialist trains only on its own corruption. A shared Optuna study selects common architecture/settings, while their trained parameters remain separate.')
 eq('r = argmax p; output = input for clean, otherwise A_r(input)')
 fig('report/figures/t2_pipeline.png','Hard classifier dispatch to identity or one specialist.')
 body('Classifier search covers lr 10^-4–.003 (log), batch {32,64,128}, channels {16-32-64,32-64-128,16-32-64-128,32-64-128-256}, dropout [0,.5], and weight decay 10^-6–.01 (log). Twelve eight-epoch trials produce 11 complete and one pruned. Best trial 10 has macro-F1 .98128. Final lr .00217503, batch 32, channels 16-32-64-128, dropout .399965 and weight decay 8.918 × 10^-5 are trained for 30 epochs; best epoch is 26. The classifier has 98,196 parameters.')
 body('Specialist search covers lr 10^-4–.003 (log), bottleneck {1,024,2,048,4,096}, base channels {16,32,48,64}, batch {32,64,128}, and L1 share [.5,.95]. Each trial trains all three corruptions for eight epochs. Six of ten complete and four are pruned; best trial 5 has mean J .16739. Each final specialist has 1,322,507 parameters and trains for 50 epochs with lr .00037508, latent 2,048, base 64, batch 64, alpha .589422 and dropout .03. Best epochs are 49/49/47.')
 tab('Classifier test metrics',['Class','Precision','Recall','F1'],[['Clean','.9414','.9725','.9567'],['Salt','1.0000','.9996','.9998'],['Blur','.9936','.9986','.9961'],['Occlusion','.9961','.9805','.9882'],['Macro','.9828','.9878','.9852']],[69,58,58,58])
 body('Test accuracy is .9909. Oracle and predicted routing evaluate identical tensors; their J values are .1277 and .1276 overall. The clean identity contribution helps the aggregate, but Task 1 remains better on the three corrupted conditions averaged separately. Predicted routing is not uniformly preferable to oracle routing.')
 r2=rows('report/tables/task2/test/results_predicted.csv')
 tab('Task 2 predicted test results',['Input','Level','MAE','SSIM','J'],[[names[r['cond']],r['severity'],f"{float(r['MAE']):.4f}",f"{float(r['SSIM']):.3f}",f"{float(r['J']):.4f}"] for r in r2 if r['severity']!='all'],[47,43,48,48,57])
 fig('report/figures/task2/test/confusion_normalised.png','Row-normalized test confusion matrix.')
 sub('Routing failure analysis')
 body('Predicted routing differs from oracle in 335 of 36,690 cases (.91%). The harmful clean-image errors send 37 images to occlusion (mean J increase .215) and 64 to blur (.085). Five blur→occlusion errors increase J .194; seven occlusion→blur errors increase it .043. Genuine dark content is a key ambiguity: clean-to-occlusion images have dark fraction .339 versus test average .047.')
 body('Skipped restoration affects 208 occluded, ten blurred and four salt images predicted clean. Of the 208 occlusion misses, 207 are low severity; identity can score better than the specialist, although it leaves a visible black rectangle. Thus fidelity metrics and visual completion differ. The largest damage comes from unnecessary specialist modification of clean images, motivating the soft identity contribution in Task 3. The full saved failure analysis is in report/text/failure_analysis_t1_t2.md.')
 head('V. TASK 3: SOFT MIXTURE OF EXPERTS')
 body('The gate initializes from the Task 2 classifier and experts from the three specialists; identity is retained. The model has 4,065,717 parameters. The Task 3 training report records Task 2 source-hash checks at startup and completion. Continuous routing blends outputs rather than choosing one branch.')
 eq('w = softmax(G(input) / tau)<br/>output = w0 input + w1 A_salt + w2 A_blur + w3 A_mask')
 eq('L_MoE = lambda1 L1 + lambdas(1 − SSIM)<br/>+ lambdac CE + lambdab sum_k(mean(wk) − 1/4)^2')
 fig('report/figures/task3/t3_architecture.png','Joint soft mixture with four outputs including identity; one ONNX graph returns image and weights.')
 body('Search ranges are joint lr [10^-5,2 × 10^-4] (log), tau [.5,5] (log), classification weight [.01,1] (log), balance weight [.001,.1] (log), and reconstruction L1 share [.3,.95]. Trials use one warm-up plus two joint epochs with 736 validation rows. Twelve stored trials comprise nine complete, one pruned and two failed; best trial 7 has J .09692. Final configuration: batch 64, tau 3.644554, lambda1 .403451, lambdas .596549, lambdac .019847, lambdab .013363.')
 body('Experts are frozen for one gate-only warm-up epoch at lr .0002, then all components jointly fine-tune for eight epochs at the smaller lr 7.3281 × 10^-5. AdamW uses weight decay 10^-4 and cosine scheduling. Best validation epoch six has J .0932 and SSIM .846. Trial-0 failed; a separate local PDF-start run gives a budget-matched J .1038 versus tuned .0969 and is not a study trial.')
 cmp=rows('report/tables/task3/test/comparison_cond_severity.csv')
 tab('Test J on identical cases',['Input','Level','Baseline','T1','T2 pred.','T3'],[[names.get(r['cond'],'All'),r['severity'],*[f"{float(r[k]):.4f}" for k in ['J_input','J_t1','J_t2_predicted','J_t3']]] for r in cmp if r['severity']!='all' or r['cond']=='overall'],[43,35,43,40,42,40])
 body('Task 3 test J .1142 is 10.5% lower than hard predicted J .1276; SSIM improves from .7869 to .8112. Improvements are not uniform: Task 1 is better on medium/high salt, and Task 3 high-occlusion J .2217 is slightly worse than hard predicted .2205. Clean J remains .0065 because the mixture is not an exact identity.')
 weights=rows('report/tables/task3/test/weights_by_class_severity.csv')
 tab('Mean routing weights by true class/severity',['Input','Level','Identity','Salt','Blur','Mask'],[[names[r['cond']],r['severity'],*[f"{float(r[k]):.3f}" for k in ['w_identity','w_salt','w_blur','w_occlusion']]] for r in weights],[45,36,44,39,39,40])
 body('Identity weight is .426 for low blur and .486 for low occlusion, falling to .037 for high occlusion. Salt dominates medium/high noise (.985/.989). Across all cases, mean weights are identity .258, salt .299, blur .196 and mask .247. Saved checks flag no inactive or unrelated-class-dominating branch under mean-weight .02 and foreign-class .9 thresholds. Gate argmax accuracy .904 is lower than Task 2 classification accuracy, but the blended reconstruction objective can favor identity for mild damage. Dominant/distributed examples and routing heatmaps document this behavior.')
 fig('report/figures/task3/test/routing_heatmap.png','Mean test routing weights by true condition and severity.')
 head('VI. TASK 4: STYLE-CONDITIONED FACE-TO-SKETCH')
 body('The U-Net generator uses base 64 channels and a learned 32-dimensional categorical style embedding. The embedding is tiled into the RGB input and 4 × 4 bottleneck; matched encoder features concatenate into the decoder. Grayscale output uses tanh. The 70 × 70 PatchGAN discriminator has its own style embedding concatenated with photo and sketch, and outputs 14 × 14 logits. Both networks therefore learn from the style condition.')
 d=Drawing(COL,195)
 for y,text in [(163,'RGB photo + G style embedding'),(124,'Encoder c,2c,4c,8c,8c → 4 × 4'),(85,'Bottleneck + style; decoder with skips'),(46,'Tanh sketch: 1 × 128 × 128'),(7,'D(photo, sketch, D style) → patch logits')]:
  d.add(Rect(2,y,COL-4,27,fillColor=colors.white,strokeWidth=.6)); d.add(String(COL/2,y+10,text,textAnchor='middle',fontName='Times-Roman',fontSize=8))
  if y>7: d.add(Line(COL/2,y,COL/2,y-12,strokeWidth=.6))
 figures+=1
 story.append(KeepTogether([d,P(f'Fig. {figures}. Task 4 architecture, drawn from models/cgan.py; c = 64. Separate embeddings condition G and D.','caption')]))
 eq('L_D = BCE(D(real),1) + BCE(D(fake),0)<br/>L_G = BCE(D(fake),1) + lambdaL1 L1(target,G(photo,style))')
 body('BCE operates on logits; generated images detach during discriminator training. Separate logging records D real/fake, G adversarial/reconstruction and validation metrics. Fixed validation photos are sampled at epoch one and every five epochs. The saved study has 26 trials: 14 complete, six pruned, six failed; best trial 25 selects lr_G .000231381, lr_D .000349229, batch 8, base 64, dropout .293383, embedding 32 and L1 weight 199.710142. Study source metadata is preserved.')
 body('Original Colab search ranges are not recoverable from the saved console record. The repository separately proposes lr_G/lr_D [10^-4,4 × 10^-4] (log), batch {8,16,32}, base {32,64}, dropout [0,.5], embedding {8,16,32}, and L1 weight [10,200] (log); these are not presented as original trial ranges. Final Kaggle training runs 100 epochs with Adam betas (.5,.999), constant then second-half linearly decaying learning rates. Best epoch is 57, validation L1 .0995; the saved training duration is 23.7 minutes. No new confirmatory study was run.')
 val=rows('report/tables/task4/val_by_style.csv'); test=rows('report/tables/task4/test_by_style.csv')
 tab('Sketch quality per style',['Split','Style','n','L1','SSIM','PSNR'],[[label,int(r['style'])+1,r['count'],f"{float(r['l1']):.4f}",f"{float(r['ssim']):.3f}",f"{float(r['psnr']):.2f}"] for label,group in [('Val.',val),('Test',test)] for r in group],[39,32,34,44,44,50])
 body('Overall test L1 is .1102, SSIM .460 and PSNR about 15.2 dB. Style 2 is weakest (L1 .1603, SSIM .371); Style 3 has only 46 test pairs, limiting conclusions from its stronger mean. Sample grids show different learned styles for the same photo. Training curves record no collapse but mild overfitting after epoch 57, motivating best-validation selection. Fine detail and paired-target disagreement remain limitations; webcam photos may differ from aligned training inputs. Failure cases and style variation examples are included below.')
 head('VII. APPLICATION, ONNX AND EXPERIMENT TRACKING')
 body('React/Vite and Tailwind CSS implement the Google Stitch designs in app/frontend_v2. The exact workspaces are Universal Restoration, Hard-Routed Restoration, Soft Mixture-of-Experts Restoration and Face-to-Sketch Generator. FastAPI validates uploaded files, preprocesses to training tensor conventions, uses ONNX Runtime, and returns PNG images with routing/timing fields. Runtime corruptions reuse the training functions. Restoration uses RGB [0,1]; sketches use [-1,1].')
 tab('Backend operations and visible information',['Endpoint','Displayed information'],[['/api/health','Model availability, hashes and runtime information'],['/api/universal','Input/output, corruption settings, inference time, download'],['/api/hard','Four probabilities, class/expert, identity bypass, output/time'],['/api/soft','Four weights, strongest contributions, output/time'],['/api/sketch','Upload/webcam/sample, Style 1/2/3, side-by-side sketch, download']],[80,163])
 body('Missing models appear in health and affected inference operations return HTTP 503. Docker Compose builds the frontend/backend, mounts models/onnx read-only, exposes the site at http://localhost:8080 and keeps backend port 8000 internal. One-command startup is <font name="Courier" size="8">docker compose up --build</font>. Historical phase reports and browser screenshots document real-model API calls, upload, routing, style selection and PNG downloads; no fresh app test was run for this report.')
 parity=rows('report/tables/onnx_parity.csv')
 tab('Saved PyTorch/ONNX parity; tolerance 10^-4',['Model/output','Cases','Max abs.','Pass'],[[r['model'].replace('t1_universal','T1').replace('t2_','T2 ').replace('t3_soft_moe','T3').replace('t4_generator','T4'),r['n_inputs'],f"{float(r['max_abs_diff']):.2e}",'Yes' if r['passed'].lower()=='true' else 'No'] for r in parity],[108,32,62,41])
 body('Seven ONNX files cover Task 1, the Task 2 classifier plus specialists, the full Task 3 mixture and the Task 4 generator. Task 3 includes image/weight parity; Task 4 covers 16 validation photos × three styles (48 cases). Only the generator is required for sketch inference. The model downloader verifies file sizes and SHA-256 against the manifest, accepts HTTPS release URLs or a local source folder, and does not replace existing artifacts.')
 body('W&amp;B project ahmedlaiq34/genai-a1 retains experiments and screenshots for all four tasks. Task 1 final run is 4wuthgxm and Task 4 is ugt4ngh7. Task 3/4 Kaggle records were logged offline and synchronized later according to their phase reports. Seed 42, saved configs, studies, deterministic manifests, evaluation CSVs and checkpoint hashes document reproducibility. Local ignored run folders and binaries are separate from repository source.')
 head('VIII. LIMITATIONS AND CONCLUSION')
 body('Compact restoration loses texture, mild damage may be worsened, and natural black backgrounds can resemble occlusion masks. Hard routing adds rare asymmetric failures; soft routing improves aggregate J while retaining useful identity contributions but is not best on every severity. Small paired data, style imbalance and domain shift limit sketch fidelity. Search/training budgets constrain the scope of conclusions. The experiments support the recorded evaluation protocol rather than universal superiority.')
 body('The saved test comparison favors soft restoration overall (J .1142), while face-to-sketch generation supports three learned styles (test L1 .1102). All four systems have model/export/application evidence. Baselines, per-severity tables, output grids and failures are essential for interpreting the aggregate scores.')
 head('IX. SUBMISSION LINKS')
 body('GitHub: <link href="https://github.com/AhmedLaiq34/genai-assignment-1/tree/dev" color="blue">AhmedLaiq34/genai-assignment-1 (dev)</link>. The link explicitly targets the populated branch; grader access and default-branch settings remain the author’s responsibility.')
 for label,url in [('YouTube demonstration',args.youtube_url),('Model download',args.model_url)]:
  body(f'{label}: <link href="{html.escape(url)}" color="blue">{html.escape(url)}</link>.' if url else f'<b>{label}: URL pending author upload.</b> Insert the final link before submission.')
 if args.model_url:
  base='https://github.com/AhmedLaiq34/genai-assignment-1/releases/download/models-v1/'
  for filename in ['t1_universal_ae.onnx','t2_classifier.onnx','t2_ae_salt.onnx','t2_ae_blur.onnx','t2_ae_occlusion.onnx','t3_soft_moe.onnx','t4_generator.onnx']:
   story.append(P(f'<link href="{base+filename}" color="blue">{filename}</link>','small'))
  body('Release assets require repository access while the repository is private. Each file’s SHA-256 and size are recorded in the release MANIFEST.json; all seven local files were verified before publication.')
 body('Submit this PDF through Google Classroom. Host the five-to-seven-minute demo on YouTube and include only its link in the report; do not upload the video directly to Classroom. The demo must show startup, upload, runtime corruption, all four tasks, probabilities/weights, download and tracking. A source package is recommended.')
 head('APPENDIX A. AI USE')
 body('Claude assistants supported planning, implementation, debugging, experiments, integration and documentation. Google Stitch supported interface design. Codex prepared the audit, downloader and this PDF from saved evidence. docs/AI_USE_LOG.md records tools, purposes and historical verification, including tests, smoke/resume runs, validation checks, model hashes, ONNX parity, HTTP calls and browser screenshots. During report preparation, existing files and primary research references were read; no training or official-test evaluation was repeated. The student must verify the narrative/citations, understand the system, acknowledge reused material and be able to explain or modify submitted components.')
 head('REFERENCES')
 refs=[
 ('[1] P. Vincent et al., “Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion,” JMLR, vol. 11, pp. 3371–3408, 2010.','https://www.jmlr.org/papers/v11/vincent10a.html'),
 ('[2] Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli, “Image quality assessment: From error visibility to structural similarity,” IEEE Trans. Image Process., vol. 13, no. 4, pp. 600–612, 2004.','https://ece.uwaterloo.ca/~z70wang/research/ssim/'),
 ('[3] N. Shazeer et al., “Outrageously large neural networks: The sparsely-gated mixture-of-experts layer,” arXiv:1701.06538, 2017.','https://arxiv.org/abs/1701.06538'),
 ('[4] O. Ronneberger, P. Fischer and T. Brox, “U-Net: Convolutional networks for biomedical image segmentation,” arXiv:1505.04597, 2015.','https://arxiv.org/abs/1505.04597'),
 ('[5] P. Isola, J.-Y. Zhu, T. Zhou and A. A. Efros, “Image-to-image translation with conditional adversarial networks,” arXiv:1611.07004, 2016.','https://arxiv.org/abs/1611.07004'),
 ('[6] T. Akiba, S. Sano, T. Yanase, T. Ohta and M. Koyama, “Optuna: A next-generation hyperparameter optimization framework,” arXiv:1907.10902, 2019.','https://arxiv.org/abs/1907.10902'),
 ('[7] ONNX contributors, “Open Neural Network Exchange,” official documentation, accessed Oct. 5, 2026.','https://onnx.ai/')]
 for text,url in refs: story.append(P(f'{html.escape(text)} <link href="{url}" color="blue">[Online]</link>','small'))
 story.extend([NextPageTemplate('Evidence'),PageBreak()])
 head('APPENDIX B. VISUAL RESULTS AND FAILURE CASES')
 fig('report/figures/t1_test_representative_12.png','Task 1: twelve representative test cases, with target, corrupted input, output and absolute-error panels.',True,540)
 story.append(PageBreak())
 for path,caption in [('report/figures/t1_test_worst_4.png','Task 1: four high-occlusion failures discussed in Section III.'),('report/figures/task2/test/routing_failures_worst.png','Task 2: classifier-caused failures, including harmful expert dispatch on clean dark backgrounds.')]: fig(path,caption,True,300)
 story.append(PageBreak())
 for path,caption in [('report/figures/task3/test/examples_dominant.png','Task 3: dominant-expert examples.'),('report/figures/task3/test/examples_distributed.png','Task 3: distributed-weight examples retaining multiple contributions.')]: fig(path,caption,True,300)
 story.append(PageBreak())
 for path,caption in [('report/figures/task4/results_test.png','Task 4: paired test photo/sketch outputs.'),('report/figures/task4/style_variations_test.png','Task 4: the same test photographs generated in all three learned styles.')]: fig(path,caption,True,300)
 story.append(PageBreak())
 fig('report/figures/task4/failures_test.png','Task 4: saved failure cases; fidelity to paired targets is limited.',True,330)
 fig('report/figures/task4/fs2k_pair_audit.png','FS2K pairing audit, previously reviewed by the student according to the phase report.',True,280)
 story.append(PageBreak())
 head('APPENDIX C. TRAINING AND SEARCH EVIDENCE')
 for path,caption in [('report/figures/t1_training_curves.png','Task 1 training/validation curves.'),('report/figures/task2/training_curves.png','Task 2 classifier and specialist learning curves.')]: fig(path,caption,True,295)
 story.append(PageBreak())
 for path,caption in [('report/figures/task3/training_curves.png','Task 3 warm-up and joint-training curves.'),('report/figures/task4/losses.png','Task 4 separate discriminator real/fake and generator adversarial/reconstruction losses.')]: fig(path,caption,True,295)
 story.append(PageBreak())
 for path,caption,height in [('report/figures/task4/val_metrics.png','Task 4 validation measurements; checkpoint 57 avoids later mild overfitting.',225),('report/figures/task4/d_probabilities.png','Task 4 discriminator real/fake probability history.',190),('report/figures/task4/sample_timeline.png','Fixed validation photos sampled over training intervals.',190)]: fig(path,caption,True,height)
 story.append(PageBreak())
 search=[('Task 1',['report/figures/t1_optuna_history.png','report/figures/t1_optuna_importance.png','report/figures/t1_optuna_parallel.png']),('Task 2 classifier',['report/figures/task2/cls_optimization_history.png','report/figures/task2/cls_param_importances.png','report/figures/task2/cls_parallel_coordinate.png']),('Task 2 specialists',['report/figures/task2/spec_optimization_history.png','report/figures/task2/spec_param_importances.png','report/figures/task2/spec_parallel_coordinate.png']),('Task 3',['report/figures/task3/t3_optuna_history.png','report/figures/task3/t3_optuna_importance.png','report/figures/task3/t3_optuna_parallel.png']),('Task 4',['report/figures/task4/study_log_history.png','report/figures/task4/study_log_importances.png','report/figures/task4/study_log_parallel.png'])]
 for label,paths in search:
  if label=='Task 3': story.append(PageBreak())
  figures+=1; sources.extend(paths)
  plot=Table([[image(path,FULL/3-5,155) for path in paths]],colWidths=[FULL/3]*3)
  story.append(KeepTogether([plot,P(f'Fig. {figures}. {label}: optimization history, parameter importance and parallel coordinates from saved study records.','caption'),Spacer(1,9)]))
 story.append(PageBreak())
 head('APPENDIX D. APPLICATION AND ORIGINAL STITCH DESIGN')
 groups=[
 [('report/figures/app/app_universal.png','Implemented Universal Restoration workspace.'),('report/figures/app/app_hard-gaussian-blur.png','Implemented Hard-Routed Restoration: class probabilities and selected expert.')],
 [('report/figures/app/app_soft-blur.png','Implemented Soft Mixture-of-Experts Restoration: weights and contributions.'),('app/frontend_v2/verification/sketch_real/sample-style1.png','Implemented Face-to-Sketch Generator; other style/upload evidence is retained in the verification folder.')],
 [('report/figures/stitch/universal_restoration.png','Original Google Stitch universal design.'),('report/figures/stitch/hard_routed_restoration.png','Original Google Stitch hard-routing design.')],
 [('report/figures/stitch/soft_MOE_restoration.png','Original Google Stitch soft mixture design.'),('report/figures/stitch/face_to_sketch.png','Original Google Stitch face-sketch design.')]]
 for i,group in enumerate(groups):
  if i: story.append(PageBreak())
  for path,caption in group: fig(path,caption,True,295)
 story.append(PageBreak())
 head('APPENDIX E. W&B EXPERIMENT TRACKING')
 body('Existing tracking screenshots are embedded directly. No live dashboard check was performed for this build. The screenshots’ folder is Git-ignored, so retain it in the source package.')
 fig('report/figures/wandb/task1_wandb.png','Task 1 W&amp;B record.',True,270)
 fig('report/figures/wandb/task2_wandb.png','Task 2 W&amp;B record.',True,270)
 for task in (3,4):
  story.append(PageBreak())
  for kind in ('train','val','system'): fig(f'report/figures/wandb/task{task}_wandb_{kind}.png',f'Task {task} W&amp;B {kind} record.',True,190)
 page_count=0
 def onpage(c,d):
  nonlocal page_count
  page_count=d.page; c.setFont('Times-Roman',8); c.drawCentredString(306,36,str(d.page))
  if d.page==1:
   title=P('Image Restoration with Autoencoders and Expert Routing,<br/>and Style-Conditioned Face-to-Sketch Generation','title')
   _,h=title.wrap(FULL,120); title.drawOn(c,54,743-h)
   c.setFont('Times-Roman',11); c.drawCentredString(306,730-h,'Ahmed Laiq'); c.drawCentredString(306,716-h,'Generative AI — Assignment #1')
 def frames(top):
  return [Frame(54,54,COL,top-54,id='left',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0),Frame(315,54,COL,top-54,id='right',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)]
 doc=BaseDocTemplate(str(OUT/'main.pdf'),pagesize=(612,792),title='Generative AI Assignment 1: Four Image Systems',author='Ahmed Laiq',pageCompression=1)
 doc.addPageTemplates([PageTemplate(id='First',frames=frames(615),onPage=onpage,autoNextPageTemplate='Paper'),PageTemplate(id='Paper',frames=frames(738),onPage=onpage),PageTemplate(id='Evidence',frames=[Frame(54,54,FULL,684,id='full',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=onpage)])
 doc.build(story)
 (OUT/'build_metadata.json').write_text(json.dumps({'pages':page_count,'figures':figures,'tables':tables,'youtube_url':args.youtube_url,'model_url':args.model_url,'figure_sources':sources,'format':'IEEE-style two-column US Letter, Times 10 pt; direct PDF'},indent=2),encoding='utf-8')
 print(f'Built report/main.pdf: {page_count} pages, {figures} figures, {tables} tables')

if __name__=='__main__': main()
