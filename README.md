# Raw RGB-Guided Semantic Boundary Refinement Across Frozen Segmentation Backbones

Research code accompanying the manuscript by Lei Zhao, Yixuan Ren and Qiang Gao.

The decoder uses the original camera image to redistribute neighbouring class scores where a frozen road segmenter predicts a class transition. Coefficients fitted with LiDSeg are transferred to Mask2Former and SegFormer-B1. The photographic branch produces a separate RGB output. The study measures boundary localization, whole-image segmentation and photographic fidelity separately.

| Task | Start here |
|---|---|
| Find the code used for a manuscript table | [Experiment index](EXPERIMENTS.md) · [Table guide with record paths](Table%20guide.csv) |
| Run the original decoder on a frozen score grid | [Score-grid prediction](predict_scores.py) · [Decoder](boundary_refinement/inference/score_grid.py) · [Colour range operator](boundary_refinement/models/operators/range_kernel.py) |
| Predict a mask and restored photograph from one image | [Camera-image predictor](boundary_refinement/inference/camera.py) |
| Train or inspect the source models | [Original RGB and LiDSeg prediction](boundary_refinement/inference/lidseg_restoration.py) · [MUSES refits](boundary_refinement/training/fit_range_kernel.py) |
| Recalculate reported results from saved counts | [Numerical replay](reproduce.py) · [Published table cells](Published%20table%20cells.csv) |
| Find hardware and software versions | [Measured CUDA environment](environment.json) |

## Code layout

| Directory | Contents |
|---|---|
| [Models](boundary_refinement/models) | Boundary operators, restoration heads and joint-model controls |
| [Training](boundary_refinement/training) | Source fitting, loss functions and luminance calibration |
| [Inference](boundary_refinement/inference) | Native-image prediction and decoding of frozen score grids |
| [Data](boundary_refinement/data) | Benchmark preparation, recording splits and verified model assets |
| [Evaluation](boundary_refinement/evaluation) · [Metrics](boundary_refinement/metrics) | Semantic, photographic and paired protocols and their measures |
| [Baselines](boundary_refinement/baselines) | Adapters for published comparison networks |
| [Experiments](experiments) | Named comparison studies and the S1–S4 replay entry points |

## Models used in the experiments

| Part of the paper | Model or operator | Implementation |
|---|---|---|
| Source semantic model | Frozen LiDSeg (PIDNet-L), two-scale scores and class-weighted donor colors | [Source decoder](boundary_refinement/models/operators/donor_colors.py) · [Source inference](boundary_refinement/inference/lidseg_restoration.py) |
| Transfer to frozen targets | Mask2Former and SegFormer-B1 supply class scores. The decoder uses aligned raw RGB to refine boundaries. | [Score decoder](boundary_refinement/inference/score_grid.py) · [Backbone inference](boundary_refinement/inference/camera.py) |
| Photographic branch | MCT supplies restored views. The road model selects a local correction and fits a luminance-dependent blend. The BeDDE follow-up calibrates the MCT output separately. | [Road heads](boundary_refinement/training/road_heads.py) · [Luminance fit](boundary_refinement/training/luminance_blend.py) · [BeDDE calibration](boundary_refinement/evaluation/photographic/photo_calibration.py) |
| Published comparisons | JBU, DySample and SAPA share Mask2Former scores in Table 5. Table 9 uses a separately matched FeatUp fit. | [Table 5 evaluator](boundary_refinement/evaluation/semantic/idd_upsampling.py) · [FeatUp fit](boundary_refinement/evaluation/controls/matched_featup.py) |

## Recalculate a table

The manuscript supplement supplies the four numbered archives, including the image manifests, per-frame counts or projected point predictions, fitted parameters and bootstrap records. Extract the outer publication ZIP to a directory such as `supplement` so that it contains `archives/Supplementary Data S1.zip` through `S4.zip`. From the repository root, use Python with NumPy.

```sh
python -m pip install numpy
python reproduce.py --list
python reproduce.py --supplement supplement --table 4 5 6 10 11 17
python reproduce.py --supplement supplement --all
```

The launcher checks archive hashes and executes the original numerical replays in a temporary directory. For Tables 4, 5, 6, 10, 11 and 17 it also prints every numerical cell at the precision used in the manuscript. Table 6 replays the RGB-screen exclusions and hash-based selection of the 128 validation identifiers. These computations use CPU and saved measurements. They do not require a dataset download or a GPU.

## Apply the original decoder to scores

The small source-fitted colour and spatial coefficients for all three seeds are included. Supply floating-point scores with shape `[C,h,w]` or `[1,C,h,w]` in an NPZ array named `scores`. For Mask2Former these are the logarithms of the aggregated positive class scores. The RGB guide must cover the same field of view as the grid. The result is a PNG of class IDs in the upstream model's class order.

```sh
python predict_scores.py scores.npz aligned_rgb.png native_ids.png --size 720 1280 --method raw --seed 101 --device cuda
```

The grid entry point supports CPU and the fixed kernel. The original and fixed settings use the factorized decoder. Near exact score ties, floating-point evaluation order can change an argmax on some inputs. To apply the matched 19-class FeatUp comparison, use `--method featup --s3-assets work/S3` after expanding the paper supplement. The directory supplies the pinned external implementation and its fitted weights. The quantitative replay uses the original sufficient statistics for each published protocol.

## Predict from a camera image

Prepare the archived coefficients and the separately distributed provider weights. The asset utility supports verified local copies through `--from-cache`. Its default network mode fetches the pinned model releases. The whole-image predictor uses CUDA with bfloat16 support. The recorded environment used PyTorch 2.11.0 with CUDA 12.8.

```sh
python prepare_supplement.py supplement work
python -m boundary_refinement.data.model_assets --backbone both --output assets
python -m boundary_refinement.inference.camera input.png predictions --assets assets --fit-dir work/S2/portable --backbone mask2former --decoder original --seed 101
```

Choose `segformer` for the second backbone. The public predictor exposes three semantic conditions. `original` reuses the coefficients fitted with LiDSeg, `fitted` applies the matched four-range MUSES refit, and `fixed` uses the unfitted initialization. All three return a native-resolution class-ID mask, a colour mask, the BeDDE-calibrated MCT photograph and a manifest of output hashes. For the road-trained local utility and luminance heads shown in the original photographic branch, use the separate source predictor linked above and its fifteen fitted-head files in expanded S1. The two photographic configurations have distinct results in the manuscript.

## Source fitting and external baselines

The [experiment index](EXPERIMENTS.md) connects each of the 17 tables to its implementation, weights and evaluator. Research modules run from the checkout root with `python -m boundary_refinement.<package>.<module>`. To use an existing research workspace, set `RGB_BOUNDARY_MODEL_DATA` to its directory containing `results/` and `pretrained/`. Its parent contains `benchmark_data/` and `experiment_exports/`. The default layout is `work/model/` with `work/benchmark_data/` and `work/experiment_exports/` beside it. Training requires the corresponding provider assets and registered manifests.

The [archive source map](docs/archived_source_paths.csv) translates historical filenames in S1–S4 to the current package. Saved runs retain their original record names and source fingerprints. When loading a frozen fit, the [source fingerprint record](boundary_refinement/data/released_source_digests.json) identifies verified historical and current implementations. Dataset and checkpoint hashes are checked separately. New fits record the code actually run. The [supplement preparation script](prepare_supplement.py) expands the numbered archives and nested source diagnostics into a local directory.

Third-party backbones and benchmark assets are distributed by their respective providers. Model revisions and file hashes for the portable predictor are given in its [asset fetcher](boundary_refinement/data/model_assets.py). The exact original measurements, auxiliary fits and published baseline inputs are in the manuscript supplement. The included source-fit coefficients are indexed by seeds 101, 211 and 307.
