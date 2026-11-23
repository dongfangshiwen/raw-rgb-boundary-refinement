# Raw RGB-Guided Semantic Boundary Refinement Across Frozen Segmentation Backbones

Research code accompanying the manuscript by Lei Zhao, Yixuan Ren and Qiang Gao.

The decoder uses the original camera image to redistribute neighbouring class scores where a frozen road segmenter predicts a class transition. Coefficients fitted with LiDSeg are transferred to Mask2Former and SegFormer-B1. The photographic branch produces a separate RGB output. The study measures boundary localization, whole-image segmentation and photographic fidelity separately.

| Task | Start here |
|---|---|
| Find the code used for a manuscript table | [Experiment index](EXPERIMENTS.md) · [Table guide with record paths](Table%20guide.csv) |
| Run the original decoder on a frozen score grid | [Score-grid prediction](predict_scores.py) · [Decoder](experiments/S3/decoder.py) · [Colour range operator](experiments/S3/range_decoder.py) |
| Predict a mask and restored photograph from one image | [Camera-image predictor](experiments/S2/portable/predict.py) |
| Train or inspect the source models | [Original RGB and LiDSeg prediction](research/model_code/scripts/predict_calibrated_joint_v99.py) · [MUSES refits](research/model_code/scripts/fit_range_v101.py) |
| Recalculate reported results from saved counts | [Numerical replay](reproduce.py) · [Published table cells](Published%20table%20cells.csv) |
| Find hardware and software versions | [Measured CUDA environment](environment.json) |

## Recalculate a table

The manuscript supplement supplies the four numbered archives, including the image manifests, per-frame counts or projected point predictions, fitted parameters and bootstrap records. Extract the outer publication ZIP to a directory such as `supplement` so that it contains `archives/Supplementary Data S1.zip` through `S4.zip`. From the repository root, use Python with NumPy.

```sh
python -m pip install numpy
python reproduce.py --list
python reproduce.py --supplement supplement --table 4 6 10 11
python reproduce.py --supplement supplement --all
```

The launcher checks archive hashes and executes the original numerical replays in a temporary directory. For Tables 4, 6, 10 and 11 it also prints every numerical cell at the precision used in the manuscript. Table 6 replays the RGB-screen exclusions and hash-based selection of the 128 validation identifiers. These computations use CPU and saved measurements. They do not require a dataset download or a GPU.

## Apply the original decoder to scores

The small source-fitted colour and spatial coefficients for all three seeds are included. Supply floating-point scores with shape `[C,h,w]` or `[1,C,h,w]` in an NPZ array named `scores`. For Mask2Former these are the logarithms of the aggregated positive class scores. The RGB guide must cover the same field of view as the grid. The result is a PNG of class IDs in the upstream model's class order.

```sh
python predict_scores.py scores.npz aligned_rgb.png native_ids.png --size 720 1280 --method raw --seed 101 --device cuda
```

The grid entry point also supports CPU and the fixed kernel. It uses the factorized decoder. Near exact score ties, floating-point evaluation order can change an argmax on some inputs. The quantitative replay uses the original sufficient statistics for each published protocol.

## Predict from a camera image

Prepare the archived coefficients and the separately distributed provider weights. The asset utility supports verified local copies through `--from-cache`. Its default network mode fetches the pinned model releases. The whole-image predictor uses CUDA with bfloat16 support. The recorded environment used PyTorch 2.11.0 with CUDA 12.8.

```sh
python prepare_supplement.py supplement work
python experiments/S2/portable/fetch_assets.py --backbone both --output assets
python experiments/S2/portable/predict.py input.png predictions --assets assets --fit-dir work/S2/portable --backbone mask2former --decoder original --seed 101
```

Choose `segformer` for the second backbone. The public predictor exposes three semantic conditions. `original` reuses the coefficients fitted with LiDSeg, `fitted` applies the matched four-range MUSES refit, and `fixed` uses the unfitted initialization. All three return a native-resolution class-ID mask, a colour mask, the BeDDE-calibrated MCT photograph and a manifest of output hashes. For the road-trained local utility and luminance heads shown in the original photographic branch, use the separate source predictor linked above and its fifteen fitted-head files in expanded S1. The two photographic configurations have distinct results in the manuscript.

## Source fitting and external baselines

The [experiment index](EXPERIMENTS.md) connects each current table to its source and evaluator. `research/model_code/` contains the training and model implementations used in the paper, including the RGB decoder, restoration heads and comparison operators. The study-specific scripts in `research/experiment_exports/` import these implementations. For historical source paths in S1–S4, resolve `scripts/` and `dehaze_seg/` relative to `research/model_code/` in this repository. Research training additionally requires the provider datasets and checkpoints, their manifests, and the staging layout referenced by those scripts. Numerical replay and the two inference entry points above have separate input requirements. The [supplement preparation script](prepare_supplement.py) expands S1–S4 and the nested source diagnostics into a new local directory.

Third-party backbones and benchmark assets are distributed by their respective providers. Model revisions and file hashes for the portable predictor are given in its [asset fetcher](experiments/S2/portable/fetch_assets.py). The exact original measurements, auxiliary fits and published baseline inputs are in the manuscript supplement. The included source-fit coefficients are indexed by seeds 101, 211 and 307.
