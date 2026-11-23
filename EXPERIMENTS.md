# Experiment index

Table numbers follow the JEI manuscript. Historical file names retain the version used when an experiment was run. The [table guide](Table%20guide.csv) gives the matching weights, cohorts and record paths inside S1–S4.

| Table | Experiment | Core implementation or evaluator |
|---|---|---|
| 1 | Roles of external models | [Frozen segmentation and reference implementations](research/model_code/scripts/idd_fog_models_v100.py) and [matched operators](research/model_code/scripts/idd_peer_setfit_v100.py). This table is a method inventory. |
| 2 | Inference latency and memory | [CUDA profiler](experiments/S4/inference-costs/profile_mask.py) and [timing replay](experiments/S4/inference-costs/replay.py) |
| 3 | Native RGB, constant and shifted guides on IDD-AW | [Guide transfer](research/model_code/scripts/idd_guides_transfer_v100.py) and [recording-level replay](experiments/S1/recompute_tables.py) |
| 4 | Fixed kernel, Gaussian JBU and fitted kernels | [Frozen evaluation](research/experiment_exports/JEI-four-decoder-check-2026-11-23/frozen_inference.py), [source refitting](research/experiment_exports/kernel-attribution-2026-11-23/fit_control.py), [four-way replay](experiments/S4/four-decoder-fog/replay.py) and [refit replay](experiments/S4/kernel-ablation/replay.py) |
| 5 | JBU, DySample, SAPA and the matched road-scene comparison | [Matched fits](research/model_code/scripts/idd_peer_setfit_v100.py) and [operator evaluation](research/model_code/scripts/idd_peer_rescore_v100.py) |
| 6 | Transfer to BDD100K validation images at 512 pixels | [Hash-based cohort selection](research/model_code/scripts/axis_reserve_v101.py), [four-range training](research/model_code/scripts/fit_range_v101.py) and [native metric replay](experiments/S2/rescore.py) |
| 7 | Boundary scores across kernels and weather strata | [Inference across kernels](research/model_code/scripts/eval_range_v101.py) and [grouped scores](experiments/S2/rescore.py) |
| 8 | Colour-axis and matched kernel controls | [Axis control](research/model_code/scripts/axis_range_v101.py) and [paired scoring](experiments/S2/rescore.py) |
| 9 | Original fit, fixed kernel and matched FeatUp | [FeatUp training and evaluation](research/model_code/scripts/featup_control_v107.py), [fixed range ablation](research/model_code/scripts/audit_fixed_ranges_v106.py) and [replay](experiments/S3/recalculate.py) |
| 10 | Transfer to reserved BDD100K training-partition prefixes at 768 pixels | [Cohort](research/experiment_exports/resolution-transfer-2026-11-21/cohort.py), [scale-specific training](research/experiment_exports/resolution-transfer-2026-11-21/scale_training.py), [evaluation](research/experiment_exports/resolution-transfer-2026-11-21/evaluate.py) and [replay](experiments/S4/replay.py) |
| 11 | SemanticSTF projected LiDAR labels in real fog | [Projection and labels](research/experiment_exports/resolution-transfer-2026-11-21/stf_data.py), [semantic scoring](research/experiment_exports/resolution-transfer-2026-11-21/stf_score.py) and [point-level replay](experiments/S4/replay.py) |
| 12 | Local restoration on held-out KITTI drives | [Utility projection](research/model_code/scripts/relative_utility_v56.py), [source model](research/model_code/scripts/joint_tone_prototype_v99.py) and [source checks](experiments/S1/source-diagnostics/check_supplement.py) |
| 13 | Luminance-dependent versus constant blending | [Luminance calibration](research/model_code/scripts/calibrate_tone_v99.py) and [source checks](experiments/S1/source-diagnostics/check_supplement.py) |
| 14 | Frozen photographic models and held-city calibration | [BeDDE calibration](research/model_code/scripts/photo_calibration_v101.py), [photographic replay](experiments/S2/rescore_photographs.py) and [source checks](experiments/S1/source-diagnostics/check_supplement.py) |
| 15 | Frozen LiDSeg and matched operators in real fog | [LiDSeg source decoder](research/model_code/scripts/semantic_subpixel_v94.py) and [operator comparison](research/model_code/scripts/published_affiliation_v95.py) and [native real-fog scoring](experiments/S1/source-diagnostics/real_joint_check.py) |
| 16 | Class-weighted donors in mixed semantic cells | [Subpixel transport](research/model_code/scripts/subpixel_transport_v92.py), [class-weighted decoder](research/model_code/scripts/semantic_subpixel_v94.py), [cell measurements](experiments/S1/source-diagnostics/mechanism_tables/measure.py) and [source checks](experiments/S1/source-diagnostics/check_supplement.py) |
| 17 | Paired photographic and semantic outcomes | [Original dual-output predictor](research/model_code/scripts/predict_calibrated_joint_v99.py), [paired evaluation](research/model_code/scripts/paired_manuscript_v102.py) and [source records](experiments/S1/source-diagnostics/check_supplement.py) |

Additional matched comparisons include [SegFix composition](research/experiment_exports/native-guide-segfix-2026-11-20/rescore_composition.py), the [released postprocessor check](experiments/S4/postprocessor-check/check_postprocessor.py) and the [sparse-sampling control](research/experiment_exports/resolution-transfer-2026-11-21/sampling_bridge.py).

## Cohort definitions

Table 6 uses the official BDD100K validation partition. All 1,000 RGB thumbnails were screened before the draw. The 32 possible-fog candidates represented 31 capture prefixes. Those and previous study prefixes were excluded before selecting 128 images. The four accepted visual examples are descriptive. The screening records identify a single assistant but do not record an exact model identifier or version. The [selection check](manuscript_tables.py) reconstructs the 128 identifiers and compares them with the archived manifest.

Table 10 uses 128 different prefixes from the BDD100K training partition, reserved for evaluation at a 768-pixel input long edge. The final hash draw used a previously thumbnail-ranked pool. Table 4 uses IDD-AW development drives. Table 11 uses sparse reference points in real fog on 11 acquisition dates.
