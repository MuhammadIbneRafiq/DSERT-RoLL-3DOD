# LiDAR lifts sparse radar mainly through distillation

Only one peer-reviewed paper tests what DSERT-RoLL's Livox-pretrain → radar fine-tune runs test: initialize a radar-only 3D detector from LiDAR weights, then fine-tune it. That paper is **LEROjD** (ECCV 2024, View-of-Delft), and its answer is split. A plain dense-LiDAR checkpoint **lowered short-range mAP by 1.8 points and raised mid-range mAP by 1.7**. The consistent gains of **+3 to +4.2 mAP** came only from pretraining on mixed radar+LiDAR, or from a curriculum that thins the LiDAR step by step toward radar density ([LEROjD](https://arxiv.org/html/2409.05564v1)). The biggest published gains come from distillation, not weight transfer. SCKD reports **+10.4 mAP on VoD (Car +4.4)** with a LiDAR+radar fusion teacher, and RadarDistill reports **+11.9 mAP on nuScenes (Car +12.2)**, though against a very weak 3D-radar baseline ([SCKD](https://arxiv.org/html/2412.14571v1), [RadarDistill](https://arxiv.org/html/2403.05061v3)). LiDAR-supervised densification, denoising and synthesis mostly help cyclists and pedestrians. Their **Car gains range from about 0 to 0.6 AP** (R2LDM, Song et al., 4D-RaDiff); L2RDaS (sedans, radar tensors) and CORENet (near-zero Dual-Radar baselines) are the exceptions. Across the roughly 75 papers catalogued below, none evaluates on DSERT-RoLL, none uses a solid-state (Livox) LiDAR as the source, and none breaks results down by standing vs. moving objects, height error or duplicate boxes. Those are exactly the four radar failure modes this project has measured. The project's own gain from Livox fine-tuning, **+3.3 Vehicle AP (0.320 ± 0.011 → 0.353 ± 0.005 over 3 seeds)**, matches LEROjD's effect size. That makes it a credible first result, but it is not yet a method. The best-supported next steps are to fix the radar input first (ego-compensated Doppler, full-model transfer, explicit handling of the 4th channel), then run LEROjD's thin-out curriculum, a SCKD-style fusion teacher with response/heatmap distillation aimed at score ranking, and LiDAR height supervision aimed at the z failure.

## Only LEROjD tests plain LiDAR-checkpoint fine-tuning, and its result splits

LEROjD keeps one PointPillars grid (0.16 × 0.16 × 5 m, x in [0, 51.2] m) and one point-cloud range for both LiDAR and radar. That shared setup lets weights pass unchanged from a LiDAR stage to a radar stage. **Radar from scratch scored 36.7 / 11.9 mAP** (short range 0–30 m / mid range 30–50 m). Fine-tuning from a **dense-LiDAR checkpoint scored 34.9 / 13.6**. A **mixed radar+LiDAR checkpoint scored 39.0 / 14.8**. The best recipe was multi-stage training with voxel-based LiDAR thinning down to 1/16 density, mixed with radar, at **39.7 / 15.4**. Every LiDAR-only thinning schedule hurt short range, and thinning past 1/32 fell below the radar baseline. The same transfer gave +3.2 / +2.6 on DSVT-P but only +1.0 / +0.5 on Voxel R-CNN ([LEROjD](https://arxiv.org/html/2409.05564v1)).

Three caveats limit how directly this carries over to DSERT-RoLL:
- LEROjD uses its own 80/20 re-split of the VoD train set and short-/mid-range mAP, not the official VoD metric.
- Per-class Car numbers for the transfer variants were not extracted; only the Car baseline (45.2 / 18.1) is known.
- The text does not say how radar RCS/Doppler and LiDAR intensity channels were reconciled. That detail lives in the repo configs ([LEROjD](https://arxiv.org/html/2409.05564v1)).

The rest of the weight-transfer literature is thin. RadarDistill initializes its radar backbone from the pretrained LiDAR backbone (an "inheriting strategy"), but folds this into its distillation recipe and never isolates its effect ([RadarDistill](https://arxiv.org/html/2403.05061v3)). The same TU Dortmund group published a cross-modal masked autoencoder (RaLi-AE, IEEE Access 2026) that reconstructs dense LiDAR from masked radar. Snippets claim "up to 3.1% mAP", but the full text was blocked ([IEEE Xplore](https://ieeexplore.ieee.org/document/11574675/)). They also list a LEROjD++ for RadarConf 2026 ([Google Scholar](https://scholar.google.com/citations?user=W8IXbsQAAAAJ&hl=de)).

Cross-LiDAR studies give one result relevant to choosing the source sensor. In the simulated LiDAR-CS benchmark, a CenterPoint trained on Livox patterns scores **68.6 mAP on Livox, 74.7 on dense VLD-64, and only 35.5 on sparse VLD-16**, and the authors tie the gap to how similar the ray distributions are ([LiDAR-CS](https://arxiv.org/html/2301.12515)). Transfer degrades with density mismatch. That predicts plain LiDAR→radar init will be weak unless the density gap is bridged, which is what LEROjD's thinning does.

No point-cloud foundation model (PTv3/PPT, Voxel-MAE, Sonata, Uni3D, AD-PT) was found evaluated on radar point clouds ([PTv3](https://arxiv.org/pdf/2312.10035), [Voxel-MAE](https://arxiv.org/pdf/2207.00531), [pretraining survey](https://arxiv.org/pdf/2512.24385)).

| Paper (venue, year) | Method: how LiDAR helps radar | Reported gain (metric, dataset) | Code | Verification | Baseline for DSERT-RoLL? |
|---|---|---|---|---|---|
| [LEROjD](https://arxiv.org/html/2409.05564v1) (ECCV 2024) | Same-architecture weight transfer: L→R, RL→R, multi-stage LiDAR thin-out (random / kNN-to-radar / voxel) | L→R −1.8 SR / +1.7 MR. Best MSTM +3.0 / +3.5 (+4.2 MR headline). PointPillars, VoD, custom split; **Car gains not extracted** | [Yes](https://github.com/rst-tu-dortmund/lerojd) | Full text | **Yes, top priority**: point clouds, OpenPCDet family |
| [LEROjD++](https://scholar.google.com/citations?user=W8IXbsQAAAAJ&hl=de) (RadarConf 2026) | Radar pseudo-feature augmentation, learnable thinning | Not extracted | Unknown | Listing only | Watch |
| [RaLi-AE / Cross-Modal MAE](https://ieeexplore.ieee.org/document/11574675/) (IEEE Access 2026) | Masked radar → reconstruct dense LiDAR; then fine-tune (DSVT-P primary) | "Up to 3.1% mAP" (snippet); datasets unverified | Listed N/A | Snippet only | Reimplement only |
| [RadarDistill](https://arxiv.org/html/2403.05061v3) (CVPR 2024) | LiDAR-backbone init plus KD (see KD table) | Init effect not isolated | [Yes](https://github.com/geonhobang/RadarDistill) | Full text | Via KD |
| [RAF](https://arxiv.org/html/2607.04587v1) (arXiv 2026) | Reuses a frozen LiDAR+radar fusion backbone when adding a camera, K-Radar | "Generally improves" | Unknown | Abstract | No (fusion) |
| [LiDAR-CS](https://arxiv.org/html/2301.12515) (arXiv 2023) | Cross-LiDAR benchmark incl. Livox pattern | Livox→VLD-64 transfers well; →VLD-16 collapses (CenterPoint 35.5) | [Project](https://opendriving.github.io/lidar-cs) | Full text | Informs Livox vs Ouster source |
| [CL3D](https://arxiv.org/abs/2212.00244) (arXiv 2022) | UDA from mechanical to solid-state LiDAR | Not read | Unknown | Abstract | No |
| [Reviewing 3D detectors for 3+1D radar](https://arxiv.org/pdf/2308.05478) (Palmer et al., 2023) | Adapts LiDAR detectors to radar; no checkpoint transfer found | n/a | Unknown | Snippet | Context |
| [K-Radar](https://arxiv.org/pdf/2206.08171) (NeurIPS 2022) | Cautions that radar→radar pretraining does not transfer across radar hardware | n/a | [Yes](https://github.com/kaist-avelab/K-Radar) | Snippet | Context |
| [PTv3](https://arxiv.org/pdf/2312.10035), [Voxel-MAE](https://arxiv.org/pdf/2207.00531) | LiDAR/point foundation and MAE pretraining; **never tested on radar** | Voxel-MAE +1.75 mAP (LiDAR, nuScenes) | Yes | Snippet | Novelty slot |

The project's own runs fit this picture. Livox-pretrained fine-tuning beat radar-from-scratch by **+3.3 Vehicle AP** with Doppler and **+1.9 without it**, and the seed ranges do not overlap. But only the encoder was transferred: 226 of 385 tensors, with all head tensors left at their initial values. The first layer's 4th-channel weights, learned on raw Livox intensity, were fed raw uncompensated Doppler. LEROjD's cleanest gains come from settings that avoid both mismatches: the whole architecture is transferred, and the density is bridged gradually.

## Distillation delivers the big numbers, but only with fusion teachers or weak baselines

Four papers distill a LiDAR (or LiDAR+radar) teacher into a radar-only 3D detection student: RadarDistill, LEROjD's KD branch, SCKD, and the ICRA 2024 cross-modal feature augmentation work. Their deltas look large, so read them against their baselines.

**RadarDistill takes nuScenes 3D radar from 8.6 to 20.5 mAP and Car AP from 41.8 to 54.0**. The baseline is so weak that the gain is easier to earn than on 4D radar. The often-quoted "+15.6 mAP" is measured against KPConvPillars, not against the same architecture ([RadarDistill](https://arxiv.org/html/2403.05061v3)). Its ablation shows the mechanism. Activation-masked low-level BEV distillation (AFD) does most of the work (6.4 → 10.9 mAP). Proposal-level distillation adds only +0.3. Removing the radar BEV densification module (CMA) drops NDS from 34.7 to 30.7. Uniform, unmasked mimicking gave nothing.

**SCKD is the strongest 4D-radar result: VoD 41.70 → 52.08 mAP (entire area) and Car 37.46 → 41.89**. It uses a SECOND fusion teacher trained with random modality dropout, distillation through conv adapters, and pseudo-labels from teacher boxes ([SCKD](https://arxiv.org/html/2412.14571v1)). Three of its findings bear directly on DSERT-RoLL:
- A fusion teacher beat a LiDAR-only teacher by **+3.84 mAP**.
- Naive MSE feature matching hurt.
- With only 1/4 of the labels, SCKD did not beat its baseline.

LEROjD's KD branch shows how this goes wrong:
- Logit distillation from a pure LiDAR teacher **dropped mAP to 32.8 / 12.6**.
- Feature KD *hurt* DSVT-P (34.7 / 11.3 vs 38.3 / 13.0).
- Its best result was feature KD from a teacher trained on mixed, quarter-density LiDAR (37.1 / 15.8) ([LEROjD](https://arxiv.org/html/2409.05564v1)).

Radar-camera work points the same way. CRKD found that distilling radar BEV features toward the teacher's *objectness heatmap* worked better than imitating LiDAR features (44.9 → 46.0 mAP), and that response distillation was its largest single term ([CRKD](https://arxiv.org/html/2403.19104v1)). RaSS found that adding logit distillation on top of feature distillation lowered segmentation mIoU ([RaSS](https://pmc.ncbi.nlm.nih.gov/articles/PMC12431280/)).

The pattern is consistent: **radar students learn best from teachers that have seen radar, through adapters and region masks, at low-level features or responses — not by copying LiDAR features or logits directly.**

| Paper (venue, year) | Teacher → student | Reported gain (metric, dataset) | Code | Verification | Baseline? |
|---|---|---|---|---|---|
| [RadarDistill](https://arxiv.org/html/2403.05061v3) (CVPR 2024) | LiDAR PillarNet → radar PillarNet; CMA densify + AFD + PFD; LiDAR-backbone init | 8.6 → 20.5 mAP, 34.7 → 43.7 NDS; Car 41.8 → 54.0 (nuScenes test, **3D radar, weak baseline**) | [Yes, pcdet + ckpts](https://github.com/geonhobang/RadarDistill) | Full text | **Yes** (AFD/CMA port) |
| [SCKD](https://arxiv.org/html/2412.14571v1) (AAAI 2025) | LiDAR+radar fusion SECOND (modality dropout) → radar SECOND; LRFD + FRFD adapters + pseudo-labels | VoD +10.38 / +6.21 mAP; **Car +4.43 / +5.50**; ZJUODset +1.08 (+5.12 with extra unlabeled data) | [Repo, possibly empty](https://github.com/Ruoyu-Xu/SCKD) | Full text | **Yes** (reimplement) |
| [LEROjD KD](https://arxiv.org/html/2409.05564v1) (ECCV 2024) | Same-arch logit / feature / label KD, student initialized from the teacher | Best +0.4 SR / +3.9 MR; pure-LiDAR logit KD −3.9 SR; hurt DSVT-P | Yes | Full text | Yes |
| [Cross-Modal Feature Augmentation / See Beyond Seeing](https://arxiv.org/pdf/2309.17336) (ICRA 2024) | Paired LiDAR/radar training, cross-modal feature hallucination, single-modality inference | "Outperforms SOTA" on VoD; numbers not checked | [Yes](https://github.com/DJNing/See_beyond_seeing) | Abstract | Possible |
| [MKFusion](https://www.sciencedirect.com/science/article/abs/pii/S0950705126003564) (KBS 2026) | LiDAR-radar teacher → radar segmentation, VoD/TJ4D | Not retrieved | Unknown | Snippet only | No (segmentation) |
| [CRKD](https://arxiv.org/html/2403.19104v1) (CVPR 2024) | LiDAR+camera BEVFusion → camera+radar | +3.5 mAP / +3.2 NDS (SwinT), +1.4 / +2.0 (R50), nuScenes | [Yes](https://github.com/Song-Jingyu/CRKD) | Full text | Ideas only (heatmap KD) |
| [RCTDistill](https://arxiv.org/html/2509.17712) (ICCV 2025) | LiDAR CenterPoint → radar-camera, range-azimuth elliptical masks, temporal KD | +4.7 mAP / +4.9 NDS (R50), nuScenes | None found | Full text | Ideas only (radar-uncertainty masks) |
| [IMKD](https://arxiv.org/html/2512.15581) (arXiv 2025) | LiDAR CenterPoint → camera-radar | 61.0 mAP / 67.0 NDS test; baseline unknown | Unknown | Snippet | No |
| [D3PD](https://www.sciencedirect.com/science/article/abs/pii/S0031320325010118) (Pattern Recognition 2025) | LiDAR features → radar and camera BEV | Not retrieved | Unknown | Snippet | No |
| [Camera-radar adverse-weather distillation](https://www.sciencedirect.com/science/article/abs/pii/S0141938225003579), [Autoencoder feature distillation](https://ietresearch.onlinelibrary.wiley.com/doi/abs/10.1049/ell2.70295), [Adaptive pillar propagation KD](https://www.sciencedirect.com/science/article/abs/pii/S092523122502867X) (2025) | Various LiDAR-guided radar-camera KD | Not retrieved | Unknown | Snippet | No |
| [RaSS](https://pmc.ncbi.nlm.nih.gov/articles/PMC12431280/) (Sensors 2025) | LiDAR segmentation teacher → radar, multi-scale BEV MSE + adaptive Doppler compensation | +5.4 / +5.6 mIoU, ZJUSSet | "Upon acceptance" | Full text | No (segmentation) |
| [LiCROcc](https://arxiv.org/abs/2407.16197) (arXiv 2024) | LiDAR-camera teacher → radar occupancy | "44.1%" over baseline (relative/absolute unclear), nuScenes-Occ | [Project page only](https://hr-zju.github.io/LiCROcc/) | Abstract | No |
| [4DRaL](https://arxiv.org/abs/2603.26206) (ICRA 2026) | LiDAR → 4D-radar place recognition KD | No numbers on abstract | None | Abstract | No |
| [CMFlow / Hidden Gems](https://openaccess.thecvf.com/content/CVPR2023/html/Ding_Hidden_Gems_4D_Radar_Scene_Flow_Learning_Using_Cross-Modal_Supervision_CVPR_2023_paper.html) (CVPR 2023) | LiDAR MOT pseudo scene-flow labels + odometry → radar scene flow, VoD | Not extracted | [Yes](https://github.com/Toytiny/CMFlow) | Abstract | Adjacent (motion cues) |

Several papers look like LiDAR→radar work but are not:
- "LiDAR-based All-weather 3D Object Detection via Prompting and Distilling 4D Radar" (ECCV 2024) goes the other way, radar into LiDAR ([ECVA](https://eccv.ecva.net/virtual/2024/poster/1725)).
- The multi-teacher K-Radar work (arXiv 2502.06114) distills between radar-tensor representations ([arXiv](https://arxiv.org/abs/2502.06114)).
- KD360-VoxelBEV and BEV-LGKD involve no radar ([arXiv](https://arxiv.org/abs/2512.15311), [arXiv](https://arxiv.org/pdf/2212.00623)).

## Self-supervised and synthetic pretraining stay unproven on 4D radar point clouds

Label-free pretraining with LiDAR as the target is almost absent for 4D radar point clouds. Apart from RaLi-AE (numbers unverifiable), the closest works pretrain radar-*camera* backbones:
- **CLLAP** converts LiDAR into pseudo-radar (GMM-density sampling, FlowNet3D velocity standing in for Doppler, elevation collapsed onto the radar plane) and runs BEV-column contrastive pretraining. It gains **+3.23 mAP on CRN, +3.52 on BEVFusion\*, and only +0.61 on RCBEVDet** on nuScenes ([CLLAP](https://arxiv.org/html/2604.24044v1)).
- **CRISP** pretrains a camera-radar BEV backbone by forecasting future LiDAR, with LiDAR "used only as privileged supervision during pretraining" ([CRISP](https://arxiv.org/abs/2607.04541)).

The image-to-LiDAR distillation family (SLidR, ST-SLidR, Seal, ScaLR, SuperFlow) is mature, but **no paper was found porting it to radar**, with LiDAR as teacher and radar as student. DSERT-RoLL's true radar–Livox pairs make that port straightforward ([ST-SLidR](https://arxiv.org/pdf/2301.05709), [ScaLR](https://arxiv.org/html/2310.17504)).

Radar-only SSL baselines exist, mostly on dense tensors, and they are the only works with label-fraction curves. The most transferable is Schwarzer et al. (2025). It runs on VoD point clouds with ego-compensated Doppler and RCS, and improves moving-object IoU from **41.1 → 46.0 at 1% labels and 60.2 → 63.5 at 100%** ([Schwarzer et al.](https://arxiv.org/html/2511.02395)). RiCL matches full supervision with 20% of labels on range-Doppler maps ([RiCL](https://arxiv.org/abs/2402.08427)). **No LiDAR→radar paper reports a label-fraction curve.**

| Paper (venue, year) | Pretraining signal | Reported gain | Code | Verification | Baseline? |
|---|---|---|---|---|---|
| [CLLAP](https://arxiv.org/html/2604.24044v1) (CVPR 2026 Findings) | LiDAR → pseudo-radar + local/global InfoNCE; radar-camera | +3.23 / +3.52 / +0.61 mAP (CRN / BEVFusion\* / RCBEVDet), nuScenes | None seen | Full text | Recipe for synthetic Doppler |
| [CRISP](https://arxiv.org/abs/2607.04541) (arXiv 2026) | Forecast future LiDAR from camera+radar | Multi-task; numbers not on abstract | [Project](https://umfieldrobotics.github.io/CRISP) | Abstract | No |
| [RLPR](https://arxiv.org/abs/2603.07920) (arXiv 2026), RaLF (2023) | Radar–LiDAR cross-modal alignment for place recognition | SOTA / zero-shot on 4 datasets | Unknown | Abstract | Alignment recipe only |
| [Radical](https://openaccess.thecvf.com/content/CVPR2024/html/Hao_Bootstrapping_Autonomous_Driving_Radars_with_Self-Supervised_Learning_CVPR_2024_paper.html) (CVPR 2024) | Radar-radar + radar-camera contrastive on heatmaps (no LiDAR) | +5.8 mAP over supervised | [Yes](https://github.com/yiduohao/Radical) | Abstract | No (tensors) |
| [Schwarzer et al.](https://arxiv.org/html/2511.02395) (arXiv 2025) | Radar-only EMA student-teacher, motion-aware contrastive, VoD/RadarScenes | +4.9 IoU (1%) → +3.3 (100%), moving-object segmentation | None | Full text | **Yes**, as the radar-only SSL control |
| [RiCL](https://arxiv.org/abs/2402.08427) (arXiv 2024) | Instance contrastive on range-Doppler maps | 20% labels ≈ 100% supervised | None | Abstract | No (tensors) |
| [SS-RODNet](https://ieeexplore.ieee.org/document/10353950/) (IEEE 2023), [MVRAE](https://api.crossref.org/works/10.1109%2FIV55156.2024.10588463) (IV 2024), [RADLER](https://arxiv.org/pdf/2504.12167) (2025) | Masked or multi-view radar-image SSL | SOTA-comparable; label efficiency; RADLER +15.5 mAP only at threshold 0.9 | Mixed | Snippet | No (tensors) |
| [RadarMAE](https://neurips.cc/virtual/2026/poster/152074), [Raw-spectra representation learning](https://neurips.cc/virtual/2026/poster/154355) (NeurIPS 2026) | Physics-aware radar MAE on tensors | Unverified | Unknown | Fragments | No |
| [GRT](https://arxiv.org/abs/2509.12482) (ICCV 2025), [RLM](https://arxiv.org/pdf/2511.21105) (2025) | Raw-radar foundation models (occupancy / text alignment) | ~20% gain per 10× data (GRT) | [GRT yes](https://wiselabcmu.github.io/grt/) | Abstract | No |

LiDAR-to-radar synthesis has the clearest empirical rule: **synthetic data helps when it is mixed with real data or pasted in as ground-truth objects, and training on synthetic data alone always loses**. Synthesis gains also land mostly on small classes:
- **L2RDaS** synthesizes K-Radar tensors from LiDAR with a conditional GAN. It lifts sedan AP_BEV by **+3.3 (RTNH), +2.6 (RadarPillar-Net) and +6.9 (RPFA-Net)**, but it does not synthesize Doppler and needs dense radar tensors that DSERT-RoLL's point-cloud release likely lacks ([L2RDaS](https://arxiv.org/html/2503.03637v2)).
- **4D-RaDiff** generates point clouds with LiDAR-conditioned backgrounds. It moves VoD CenterPoint Car AP only from **39.9 (real GT-sampling) to 40.5**, while cyclists and pedestrians gain more. Synthetic pretraining plus 10% real labels (47.0–47.6 mAP) beats 100% real without pretraining (46.0) ([4D-RaDiff](https://arxiv.org/html/2512.14235)).
- **Song et al. (IROS 2025)**: synthetic radar left VoD PointPillars **Car 3D AP flat (24.48 → 24.53)** while Cyclist rose 11.73 → 14.87 ([Song et al.](https://arxiv.org/html/2503.08068v2)).

| Paper (venue, year) | Synthesis route | Reported gain | Code | Verification | Baseline? |
|---|---|---|---|---|---|
| [L2RDaS](https://arxiv.org/html/2503.03637v2) (arXiv 2025) | LiDAR → 3D radar tensor cGAN + box-boundary/centre injection; GT-Aug | Sedan +3.75 AP_BEV / +4.03 AP_3D average (GT-Aug), K-Radar | [In K-Radar repo](https://github.com/kaist-avelab/K-Radar) | Full text | Only if tensors available |
| [L2RLDB](https://cvpr.thecvf.com/virtual/2026/poster/39723) (CVPR 2026) | LiDAR → 4D radar tensor via latent diffusion bridge, incl. Doppler | Ablation only (56.00 vs 54.84 mAP); real-vs-augmented not retrieved | Not public | Secondary | No |
| [LiDAR-to-4D Radar Synthesis for large-scale tensor datasets](https://openaccess.thecvf.com/content/CVPR2026F/papers/Jung_LiDAR-to-4D_Radar_Synthesis_for_Building_Large-Scale_Tensor_Datasets_CVPRF_2026_paper.pdf) (CVPR 2026 Findings) | KAIST follow-up to L2RDaS | Not read | Unknown | Listing | No |
| [4D-RaDiff](https://arxiv.org/html/2512.14235) (arXiv 2025) | Latent point diffusion; boxes → foreground, LiDAR → background; xyz + compensated Doppler + RCS | VoD mAP 52.4 → 53.3 over real GT-sampling (**Car +0.6**); 10% real + synthetic pretraining > 100% real; TruckScenes +0.6 to +1.0 mAP | Not stated | Full text | Yes, its "LiDAR downsampling" control is cheap |
| [Song et al.](https://arxiv.org/html/2503.08068v2) (IROS 2025) | Camera + LiDAR → radar point distribution and strength; Doppler from kinematics | **Car +0.05**, Cyclist +3.14 3D AP, VoD PointPillars | Not stated | Full text | No |
| [Lee et al.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10818739/) (Sensors 2024) | VFE-based LiDAR → radar point translation | Not retrieved | Unknown | Abstract | No |
| [L2R GAN](https://openaccess.thecvf.com/content/ACCV2020/html/Wang_L2R_GAN_LiDAR-to-Radar_Translation_ACCV_2020_paper.html) (ACCV 2020) | LiDAR BEV → radar spectrum GAN | Not retrieved | Unknown | Abstract | No |
| [Lee, radar-centric 3D detection](https://arxiv.org/abs/2003.00851) (arXiv 2020) | LiDAR → radar-like points + augmentation, Astyx | Not retrieved | Unknown | Abstract | Historical |
| [SHENRON](https://wcsng.ucsd.edu/shenron/) (RA-L 2023), [C-Shenron](https://wcsng.ucsd.edu/files/c-shenron_paper.pdf) (VTC 2025) | Physics MIMO radar from LiDAR + camera; CARLA | Driving metrics, not 3D AP | Yes | Abstract | No |
| [RadSimReal](https://openaccess.thecvf.com/content/CVPR2024/html/Bialer_RadSimReal_Bridging_the_Gap_Between_Synthetic_and_Real_Data_in_CVPR_2024_paper.html) (CVPR 2024), [Sim2Radar](https://arxiv.org/html/2602.13314) (2026), [RadarGen](https://arxiv.org/html/2512.17897v1) (ECCV 2026), [RF-Genesis](https://github.com/Asixa/RF-Genesis), [Midas](https://github.com/zihanjames/midas), [RadarSplat](https://openaccess.thecvf.com/content/ICCV2025/html/Kung_RadarSplat_Radar_Gaussian_Splatting_for_High-Fidelity_Data_Synthesis_and_3D_ICCV_2025_paper.html) | Non-LiDAR physics or camera-driven radar simulation | Sim2Radar +3.7 3D AP (indoor, IoU 0.3); others not detection | Mixed | Abstract | No |

## LiDAR as privileged supervision posts big deltas over weak baselines

The fastest-growing 2025–2026 branch uses LiDAR only at training time, as a target for denoising, densification, occupancy or labels, while inference stays radar-only. The 2026 review groups DenserRadar, CORENet, R2LDM, L2RDaS, Radar-Mamba, HyperDet, SCKD, RaSS, MKFusion and 4D-ROLLS under "cross-modal supervision". It warns that treating unmatched radar returns as noise "may suppress Doppler, RCS, multipath, and material-dependent cues" ([4D radar review](https://arxiv.org/html/2609.19216v1)).

The detection deltas are real but need context:
- **CORENet** turns LiDAR proximity into a per-point keep mask (any LiDAR point within 0.5 m). It lifts Dual-Radar CasA-V mean 3D AP **from 4.96 to 21.37** and Voxel R-CNN from 11.76 to 19.44. Those baselines are near zero ([CORENet](https://arxiv.org/html/2508.13485v1)).
- **R2LDM** densifies radar 6–10× with diffusion. It raises VoD PointPillars mAP from 24.92 to 30.37, but **Car gains only +0.04** in the entire area, against a raw baseline about 20 points below standard VoD radar results ([R2LDM](https://arxiv.org/html/2503.17097v2)).
- **HyperDet's** own table shows that most of its TruckScenes gain (VoxelNeXt 0.062 → 0.124 mAP) comes from multi-radar temporal aggregation before the LiDAR-supervised diffusion is added. The diffusion adds only +0.007 ([HyperDet](https://arxiv.org/html/2602.11554v2/)).
- **LiDAR auto-labels**: training RTNH on K-Radar with LiDAR-detector boxes matched training on manual labels. This supports teacher pseudo-labels on unlabeled DSERT-RoLL frames ([auto-labeling](https://arxiv.org/abs/2407.04709v1)).

| Paper (venue, year) | How LiDAR supervises radar | Reported gain | Code | Verification | Baseline? |
|---|---|---|---|---|---|
| [CORENet](https://arxiv.org/html/2508.13485v1) (IROS 2025 per review) | LiDAR-proximity voxel mask supervises a radar denoiser; plug-in | Mean 3D AP +16.4 (CasA-V), +7.7 (Voxel R-CNN), Dual-Radar, **near-zero baselines**; a secondary "Car +34.6" claim is unverified | [Yes](https://github.com/charlesuv/corenet) | Full text | **Yes**, cheap auxiliary head |
| [HyperDet](https://arxiv.org/html/2602.11554v2/) (arXiv 2026) | LiDAR in-box points with radar-mean RCS/Doppler → pseudo-radar foreground; one-step diffusion | VoxelNeXt 0.062 → 0.131 mAP, TruckScenes; mostly from aggregation | "Will release" | Full text | Pseudo-radar construction is cheap |
| [DenserRadar](https://arxiv.org/abs/2405.05131v1) (ITSC 2024) | Stitched-LiDAR occupancy GT supervises a tensor → point generator, K-Radar | Density/accuracy vs CFAR; detection not retrieved | Unknown | Abstract | No (tensors) |
| [R2LDM](https://arxiv.org/html/2503.17097v2) (IROS 2025 per review) | Latent voxel diffusion, radar → LiDAR | VoD PP mAP +5.45 / +10.97 (corridor); **Car +0.04 / +0.59** | Not stated | Full text | Low priority for Vehicle AP |
| [MSDNet](https://arxiv.org/html/2509.13149v1) (arXiv 2025) | LiDAR teacher, reconstruction- then diffusion-guided feature distillation | Chamfer 5.16 vs 6.07; odometry / place recognition. Full-text reading found no detection table, while an abstract summary claims detection gains (unresolved) | "Upon publication" | Full text | No |
| [SD4R](https://arxiv.org/html/2602.20653v1) (arXiv 2026) | Foreground point generator densifies radar; LiDAR use unconfirmed | 51.81 mAP VoD (listing) | [Yes](https://github.com/lancelot0805/SD4R) | Abstract | Possible radar baseline |
| [Radar-Mamba](https://arxiv.org/html/2609.19216v1) (ACM MM 2025) | Radar–LiDAR occupancy alignment for denoising | Not retrieved | Unknown | Review only | No |
| [4D-ROLLS](https://arxiv.org/html/2505.13905v1) (IROS 2025 per review) | LiDAR ray-cast occupancy + **LiDAR max-Z height map** supervision | Chamfer 2.79 vs 3.03 (MSC) | Announced | Full text | Height-supervision idea |
| [RadarOcc](https://arxiv.org/pdf/2405.14014) (NeurIPS 2024) | LiDAR-derived occupancy labels, K-Radar | n/a | [Yes](https://github.com/Toytiny/RadarOcc) | Abstract | No |
| [4D radar auto-labeling](https://arxiv.org/abs/2407.04709v1) (IV 2024) | LiDAR PVRCNN++ boxes as radar labels | ≈ manual GT for RTNH, K-Radar | Unknown | Abstract | Supports pseudo-labels |
| [RadarHD](https://arxiv.org/pdf/2206.09273.pdf) (ICRA 2023), [ZJU Radar-Diffusion](https://arxiv.org/abs/2403.08460) (RA-L 2024), [Luan Radar-diffusion](https://arxiv.org/pdf/2404.06012), [Range-image diffusion](https://arxiv.org/html/2503.02300v1), [R3D](https://arxiv.org/html/2601.06465v1), [RadarINV](https://arxiv.org/html/2505.09887v1), [Sem-RaDiff](https://arxiv.org/html/2509.02283v1) | LiDAR-like super-resolution of raw or single-chip radar | Geometry / odometry metrics only, **no 3D detection** | RadarHD, ZJU, RadarINV, R3D yes | Abstract | No |

Standard baselines and surveys round out the landscape, and two facts from them matter more for DSERT-RoLL than any transfer method.

First, **RadarPillars measures a −7.2 mAP drop on VoD when ego-motion-compensated radial velocity is removed, versus only −0.9 for raw relative velocity, and +3.8 from decomposing compensated velocity into x/y** ([RadarPillars](https://arxiv.org/abs/2408.05020)). This project has so far fed raw, uncompensated Doppler.

Second, the DSERT-RoLL paper itself reports radar-only Vehicle AP of only about 22–49 (RTNH, VoxelNeXt, HEDNet) against 44–90 for LiDAR. It notes that radar "fails on stationary objects" in snow, and it runs no LiDAR→radar transfer at all ([DSERT-RoLL](https://arxiv.org/html/2604.03685v1)).

| Paper (venue, year) | Role | Key fact | Code | Verification |
|---|---|---|---|---|
| [4D Radar Perception review](https://arxiv.org/html/2609.19216v1) (arXiv Sep 2026) | Survey; defines "cross-modal supervision" | Open problems: go beyond LiDAR-like completion, underused Doppler, velocity-stratified benchmarks, cross-hardware generalization | n/a | Full text |
| [4D mmWave radar in adverse environments](https://arxiv.org/html/2503.24091v1) (2025), [4D mmWave radar survey](https://arxiv.org/pdf/2306.04242) (2023), [Radar-vision fusion survey](https://arxiv.org/pdf/2406.00714) (2024), [Awesome 4D radar list](https://github.com/liuzengyun/Awesome-3D-Detection-with-4D-Radar), [awesome-mmwave](https://github.com/Armorhtk/awesome-mmwave-radar-perception) | Surveys and lists, mostly fusion-framed | Older surveys treat LiDAR as a fusion partner, not a teacher; the Awesome list's link column is partly wrong | n/a | Snippet |
| [DSERT-RoLL](https://arxiv.org/html/2604.03685v1) (CVPR 2026) | Target dataset | Radar VoxelNeXt Clear 25.03 vs LiDAR 86.06 Vehicle AP; no transfer experiments | [Dataset](https://github.com/jeongyh98/DSERT-RoLL-Dataset) | Full text |
| [RadarPillars](https://arxiv.org/abs/2408.05020) (ITSC 2024) | Radar-only baseline | Compensated v_r −7.2 mAP when removed; x/y decomposition +3.8 | [Third-party](https://github.com/fthbng77/RadarPillar) | Full text |
| [MAFF-Net](https://github.com/TRV-Lab/MAFF-Net) (RA-L 2025), [SMURF](https://arxiv.org/pdf/2307.10784) (T-IV 2023), [RPFA-Net](https://github.com/Nerdmust/astyx-pcdet-radar) (ITSC 2021), [RTNH](https://github.com/kaist-avelab/k-radar) (T-IV 2024), [RadarNeXt](https://arxiv.org/pdf/2501.02314v1), [RadarGaussianDet3D](https://arxiv.org/pdf/2509.16119) | Radar-only detectors | VoD mAP ~39–55 (RPFA 38.75, SMURF 50.97, MAFF 54.6) | MAFF, RPFA, RTNH yes | Snippet |
| [3D-LRF / RL3DOD](https://openaccess.thecvf.com/content/CVPR2024/papers/Chae_Towards_Robust_3D_Object_Detection_with_LiDAR_and_4D_Radar_CVPR_2024_paper.pdf) (CVPR 2024), [L4DR](https://arxiv.org/abs/2408.03677) (AAAI 2025), [DLRFusion](https://github.com/yujeong-star/DLRFusion) (ICCV 2025), [Bi-LRFusion](https://arxiv.org/abs/2306.01438), [LiRaFusion](https://arxiv.org/abs/2402.11735), [RaLiBEV](https://arxiv.org/abs/2211.06108), [MoRAL](https://arxiv.org/html/2505.09422v1), [ATN3D](https://arxiv.org/pdf/2606.09634) | LiDAR–radar fusion; candidate teachers | Bi-LRFusion enriches radar BEV from LiDAR; MoRAL compensates object motion during accumulation | RL3DOD, DLRFusion yes | Mixed |

## Vehicle gains shrink to near zero outside distillation

The user reports Vehicle AP only, so the published effect sizes need re-reading at the Car level. They sort into three tiers:
- **Distillation with radar-aware teachers moves cars**: SCKD gives +4.4 Car AP on VoD, and RadarDistill +12.2 on nuScenes 3D radar.
- **LiDAR-supervised input enhancement moves cars only when the baseline is broken**: CORENet on Dual-Radar, at about 5 mean AP before.
- **Synthesis and densification barely move cars at all**: +0.6 (4D-RaDiff), +0.05 (Song et al.), +0.04 (R2LDM). L2RDaS's sedan gains of +2.6 to +6.9 AP_BEV are the outlier, and they depend on radar tensors ([4D-RaDiff](https://arxiv.org/html/2512.14235), [Song et al.](https://arxiv.org/html/2503.08068v2), [R2LDM](https://arxiv.org/html/2503.17097v2), [L2RDaS](https://arxiv.org/html/2503.03637v2)).

A plausible reason: cars are already the densest radar targets, so extra or cleaner points add little. What limits Car AP is box quality and ranking, not detection.

This project's diagnosis supports that reading and sharpens it. On the full validation set, radar finds about **81% of vehicles moving faster than 4 m/s but only about 32% of standing ones**, at matched point count and distance. Standing-vehicle Vehicle AP is **0.026 (scratch) vs 0.042 (Livox-ft) vs 0.878 (LiDAR)**. Livox fine-tuning does not fix standing cars; its AP gain comes mostly from better ranking.

Height is the largest error budget. 3D recall is 58% against 76% BEV recall, and z/height errors cost about 18 points. Yet 87.5% of radar points over a vehicle footprint lie inside the box, so the detector, not the data, loses the height. Ranking costs more than duplicates. Re-ordering scores per vehicle would lift ft from 0.356 to 0.407, and keeping the best-fitting box per vehicle would reach 0.478, while offline BEV NMS at 0.5 adds only +2.2 to +2.6. The authors' post-processing applies **no NMS to vehicle boxes** after the second stage.

None of the catalogued papers reports static vs. moving AP, a z-error breakdown or duplicate rates. That makes these four failure modes both the project's main problem and its main room to contribute.

## Ten experiments mapped to DSERT-RoLL's four failure modes

The ordering below puts input hygiene first, because LiDAR-transfer comparisons are confounded while radar is fed raw Doppler through intensity-trained weights. Cheap, code-backed LiDAR baselines come next; the novel slots come last. Every row is evaluated with the unchanged official protocol (Waymo-style AP, Vehicle 3D IoU 0.5, X in [0, 70] m, Level 1, no extra score cut-off), as **Vehicle AP mean ± std over 3 seeds**, plus the notebook's standing/moving AP split.

| Priority | Experiment | Literature basis | Failure mode targeted | Expected effect / why | Cost |
|---|---|---|---|---|---|
| 1 | Ego-compensated Doppler (DSERT-RoLL odometry), then v_r split into vx/vy, then add `power` as a 5th channel; rerun scratch and Livox-ft | [RadarPillars](https://arxiv.org/abs/2408.05020) −7.2 / +3.8 mAP; [RaSS](https://pmc.ncbi.nlm.nih.gov/articles/PMC12431280/) Doppler compensation; [review](https://arxiv.org/html/2609.19216v1) "underused Doppler" | Standing cars (compensated Doppler ≈ 0 marks them as static rather than mixing them with ego speed), ranking | Largest evidence-backed single change; precondition for fair transfer comparisons | Low |
| 2 | Transfer the full model (encoder + head, incl. IoU-score branch), not encoder only; ablate 4th-channel handling (zero intensity in pretraining vs. re-init first layer vs. copy) | [LEROjD](https://arxiv.org/html/2409.05564v1) transfers the whole same-architecture model; no paper compares channel strategies | Score ranking (head currently starts at init) | Fills a stated gap; isolates whether the +3.3 comes from geometry or channel artifacts | Low |
| 3 | LEROjD multi-stage: Livox → thinned Livox (random, kNN-to-radar, voxel; 1 → 1/16) → mixed radar+Livox → radar | [LEROjD](https://arxiv.org/html/2409.05564v1), [LiDAR-CS](https://arxiv.org/html/2301.12515) density finding | General recall, mid/far range | Literature predicts +1 to +4 mAP over plain L→R; code public | Low–medium |
| 4 | SCKD-style teacher: 4R+Livox fusion with random modality dropout; adapter feature KD (LiDAR→radar and fusion→radar) + teacher pseudo-labels on unlabeled frames | [SCKD](https://arxiv.org/html/2412.14571v1) fusion teacher +3.84 over LiDAR-only teacher; [auto-labeling](https://arxiv.org/abs/2407.04709v1) | Standing cars (teacher detects them at 0.878 AP), ranking | Strongest published 4D-radar Car gain (+4.4) | Medium |
| 5 | Response/heatmap and IoU-score distillation: match radar heatmap to teacher objectness (CRKD CSRD/RespD) and distill the teacher's predicted IoU into the radar IoU branch | [CRKD](https://arxiv.org/html/2403.19104v1) heatmap target > feature target; [RadarDistill](https://arxiv.org/html/2403.05061v3) masked AFD | Score ranking (~0.05 AP headroom from re-ordering alone) | Novel for radar-only; directly attacks the dominant loss | Medium |
| 6 | LiDAR height supervision: auxiliary BEV head regressing the Livox max-Z / ground-height map, plus box-level z distillation from the teacher | [4D-ROLLS](https://arxiv.org/html/2505.13905v1) LiDAR height maps; no detection paper does this | z / height (~18 points of 3D-vs-BEV gap; radar does not track drive ground level) | Novel; radar points are vertically fine, so a ground prior should help | Medium |
| 7 | CORENet-style LiDAR-proximity point mask (τ = 0.5 m) as an auxiliary foreground head | [CORENet](https://arxiv.org/html/2508.13485v1) | Ranking, false positives | Cheap; gains will be far below CORENet's because the baseline is stronger | Low |
| 8 | Vehicle NMS inside the model's post-processing (BEV IoU 0.5) as a model change, alongside a centre-based NMS-light head check | No radar NMS paper found ([review](https://arxiv.org/html/2609.19216v1)) | Duplicates (~5 boxes per car) | +2.2 to +2.6 measured offline; hygiene baseline before any transfer claim | Low |
| 9 | Pseudo-radar GT-sampling of standing vehicles from Livox in-box points carrying radar-mean power and compensated Doppler ≈ 0 | [HyperDet](https://arxiv.org/html/2602.11554v2/), [4D-RaDiff](https://arxiv.org/html/2512.14235) LiDAR-downsampling control | Standing cars (rebalances static examples) | Car gains in literature ≤ 0.6; worth it only as a targeted static-car rebalancing test | Medium |
| 10 | Label-free cross-modal pretraining: masked radar → Livox reconstruction (RaLi-AE) or BEV-column InfoNCE against a frozen Livox encoder (SLidR/CLLAP port); report at 1 / 5 / 10 / 25 / 100% labels; repeat with Ouster as source | [RaLi-AE](https://ieeexplore.ieee.org/document/11574675/), [CLLAP](https://arxiv.org/html/2604.24044v1), [ST-SLidR protocol](https://arxiv.org/pdf/2301.05709), [Schwarzer et al.](https://arxiv.org/html/2511.02395) as radar-only SSL control | Low-label regime, general | Fills the label-curve and Livox-vs-mechanical gaps | High |

These experiments line up with gaps no paper fills:
- the first LiDAR→radar transfer results on DSERT-RoLL;
- the first solid-state (Livox) LiDAR source or teacher;
- the first static vs. moving AP breakdown for transfer, answering the review's call for velocity-stratified benchmarks;
- the first analysis of height error and duplicate boxes in radar transfer;
- the first controlled comparison of how to handle the radar/LiDAR channel mismatch;
- the first label-fraction curve for LiDAR→radar pretraining;
- the first test of point-cloud foundation models (PTv3/Sonata) on radar;
- the first distillation aimed at score ranking (IoU score) rather than features.

The review's warning that LiDAR-like completion suppresses Doppler also sets a design constraint: any distillation should run through adapters that leave the compensated-Doppler and power channels free ([review](https://arxiv.org/html/2609.19216v1)).

## Conclusion

The literature's real lesson is that LiDAR knowledge reaches radar best through what the radar student can actually reproduce. Teachers that have seen radar, gradual density curricula, and masked or response-level targets carry over. Raw LiDAR weights, LiDAR logits and LiDAR-like point completion mostly do not, and least of all for cars. For DSERT-RoLL, the evidence suggests the current +3.3 Vehicle AP is near the ceiling of plain weight transfer. More importantly, the project's dominant failure modes (static cars, height, score ranking) are ones the transfer literature has never measured. Several of them trace back to radar-native inputs (uncompensated Doppler, unused power) rather than missing LiDAR knowledge.

That reframes the research question. It is less "does LiDAR pretraining help radar?" than "which LiDAR signal fixes which radar failure?" Distilled scores and heatmaps should target ranking, LiDAR ground-height maps should target z, and fusion-teacher pseudo-labels with compensated Doppler should target standing vehicles. Answering it with velocity-stratified Vehicle AP over seeds would be a contribution the field currently lacks, on a dataset where no one else has reported radar transfer.
