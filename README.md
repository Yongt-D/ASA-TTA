# ASA-TTA: Anchored Set Aggregation for test-time augmentation in oriented object detection

This repository holds the code and the results of the paper *Learning When and How to Trust
Test-Time Augmentation in Oriented Object Detection* (under review).

Test-time augmentation (TTA) runs a detector on several views of an image (flips, quarter
turns, rescaled copies) and fuses the predictions. Fixed fusion rules such as weighted boxes
fusion (WBF) treat every view and every cluster of boxes alike. **Anchored Set Aggregation
(ASA)** learns the fusion from the detector's own predictions on its training images:

* the boxes of all views are clustered by rotated WBF (IoU 0.55), as in standard TTA;
* a small set encoder (DeepSets, about 4.3e4 parameters) reads each cluster's members
  (view, confidence, rank, IoU and offset relative to the fused box) and a cluster descriptor;
* the **score** is anchored to the original WBF score p0, `s = sigmoid(logit p0 + delta)`,
  with `delta` zero-initialised and penalised, so the untrained model is exactly WBF;
* the **box** is a learned weighting of the aligned member vertices,
  `softmax_i(log c_i + gamma_i)`, again equal to the WBF box before training.

ASA trains on a CPU (under a minute to under half an hour per detector and family, on eight
threads), needs no change to the detector and no extra forward passes, and works with any
detector and any set of views. The paper evaluates it on
HRSC2016-MS, DIOR-R, DOTA-v1.0 and FAIR1M-2.0 with 19 detectors (YOLO11 n–x and Oriented
R-CNN) and four TTA families (D2, C4, D4 and a multi-scale family), 76 settings in all.

## Contents

```
asa/            the method and everything it shares with the baselines
  views.py        TTA views, their exact inverse mapping, the families (python -m asa.views: self-test)
  obb.py          oriented boxes: polygon IoU, file reading and writing
  fusion.py       rotated WBF, rotated NMS, Soft-NMS
  clusters.py     WBF clusters of the pooled views, cluster and member descriptors
  model.py        ASA: network, training, prediction, saving and loading
  baselines.py    GBDT re-scoring and OBBStacking
  evaluate.py     AP50, AP75, AP50:95 (python -m asa.evaluate)
  bootstrap.py    paired bootstrap over images or DOTA scenes
  config.py       reads configs/experiments.json
tools/          command-line tools (each file starts with its usage)
configs/        experiments.json (datasets, the 19 detectors, view layout),
                yolo11/*.yaml (training arguments), mmrotate/*.py (Oriented R-CNN configs)
splits/         our FAIR1M-2.0 train/val split
results/        the paper's numbers: per detector and family, bootstrap intervals, latency
```

## Installation

```bash
pip install -r requirements.txt
```

ASA, the baselines, the evaluation and the tables need Python ≥ 3.10, PyTorch (CPU is enough),
NumPy, scikit-learn and Pillow. Exporting views from YOLO11 and training it need
Ultralytics 8.4.53 and a GPU. The Oriented R-CNN detectors need
[MMRotate](https://github.com/open-mmlab/mmrotate) dev-1.x (we used MMCV 2.0.1, MMDetection
3.0.0, PyTorch 2.0.1, on Linux), installed from source so that its configs can be referenced
as `mmrotate::...`:

```bash
git clone -b dev-1.x https://github.com/open-mmlab/mmrotate.git && pip install -e mmrotate
```

## ASA on your own detector

1. **Export views** of the training images and of the images to process. For YOLO11-OBB:

   ```bash
   python tools/export_views_yolo.py --model best.pt --source data/mydata/images/train --out-root views/mydet/train/base --imgsz 1024 --views r0 r90 r180 r270
   ```

   Repeat for the evaluation images with `--source .../images/val --out-root views/mydet/val/base`.
   Other detectors only need to write the same files: one text file per image and view with
   lines `cls x1 y1 x2 y2 x3 y3 x4 y4 conf`, corners **in the original image**, normalised
   by its width and height, a confidence floor of 0.001, and an `export.json` with
   `"back_mapping": "continuous"` next to the view directories (see `asa/views.py`).

2. **Train ASA** on the training views (CPU):

   ```bash
   python tools/train_asa.py --family c4 --labels data/mydata/labels/train --images data/mydata/images/train --num-classes 15 --out models/mydet_c4.pt --views views/mydet/train/base/r0 views/mydet/train/base/r90 views/mydet/train/base/r180 views/mydet/train/base/r270
   ```

3. **Apply it** and evaluate:

   ```bash
   python tools/apply_asa.py --model models/mydet_c4.pt --images data/mydata/images/val --out preds/mydet_c4_asa --views views/mydet/val/base/r0 views/mydet/val/base/r90 views/mydet/val/base/r180 views/mydet/val/base/r270
   ```

   ```bash
   python -m asa.evaluate --labels data/mydata/labels/val --preds preds/mydet_c4_asa --num-classes 15
   ```

Labels use the YOLO-OBB format (`cls x1 y1 ... x4 y4`, normalised). The families are defined
in `asa/views.py`: Flip `[r0, h0]`, D2 `[r0, h0, h180, r180]`, C4 `[r0, r90, r180, r270]`,
D4 (all eight) and MS `[r0, h0@0.83, r0@0.67]`; `r` is a counter-clockwise quarter turn,
`h` the same turn followed by a left-right flip, and `@s` a pass at s times the input size.

## Reproducing the paper

All commands run from the repository root. Datasets go to `data/`, exported views to
`views/<detector>/<split>/{base,0.83,0.67}/<view>`, as `configs/experiments.json` expects.

### 1. Data

| Dataset | Classes | Train / val images | Command |
|---|---|---|---|
| HRSC2016-MS | 1 | 603 / 454 | `python tools/convert_hrsc2016_ms.py --root <HRSC2016-MS> --out-root data/HRSC2016_MS_yolo_obb` |
| DIOR-R | 20 | 5,862 / 5,863 | `python tools/convert_dior_r.py --download-root <archives> --out-root data/DIOR-R/yolo_obb` |
| DOTA-v1.0 | 15 | 1024-pixel crops, gap 200; 3,066 labelled val crops | `python tools/prepare_dota.py --src-root <DOTA_V1.0> --out-root data/dota_yolo_obb_1024` |
| FAIR1M-2.0 | 37 | 6,630 / 1,657, split of the official train set | `python tools/convert_fair1m2.py --images <images> --xmls <labelXml> --out-root data/FAIR1M2_yolo_obb_rgb --splits splits` |

Each tool's docstring describes the expected input layout. Only images with a label file
are evaluated.

### 2. Detectors

**YOLO11-OBB.** On DOTA we use the released `yolo11{n,s,m,l,x}-obb.pt`. The other detectors
are fine-tuned from them with the recorded training arguments:

```bash
python tools/train_yolo11.py configs/yolo11/hrsc_yolo11s.yaml
```

The weights land in `work_dirs/yolo11/<name>/weights/best.pt`. Training is not bit-reproducible
on GPUs, so retrained detectors differ slightly from ours.

**Oriented R-CNN** (R50-FPN, MMRotate's DOTA recipe, trained on the train split only):

```bash
python tools/yolo_obb_to_dota_annfiles.py --yaml data/HRSC2016_MS_yolo_obb/hrsc2016_ms_obb.yaml --split train --out data/mmrotate/hrsc/annfiles/train --link-images
```

```bash
python <mmrotate>/tools/train.py configs/mmrotate/oriented_rcnn_r50_hrsc2016_ms.py
```

The other datasets work the same way: `dota`, `dior_r` (add `--min-side 2`) and `fair1m2`
under `data/mmrotate/`.

### 3. Views

Export every detector's views on the train and val splits: the eight base views at the
base input size, and the two scaled passes of the multi-scale family. The base size is
1024 (1536 for the DOTA YOLO11 detectors), and the scaled passes run at 832 and 672
(1280 and 1024 for the DOTA YOLO11 detectors):

```bash
python tools/export_views_yolo.py --model work_dirs/yolo11/hrsc_yolo11s/weights/best.pt --source data/HRSC2016_MS_yolo_obb/images/val --out-root views/hrsc_yolo11s/val/base --imgsz 1024 --views r0 r90 r180 r270 h0 h90 h180 h270
```

```bash
python tools/export_views_yolo.py --model work_dirs/yolo11/hrsc_yolo11s/weights/best.pt --source data/HRSC2016_MS_yolo_obb/images/val --out-root views/hrsc_yolo11s/val/0.83 --imgsz 832 --views h0
```

```bash
python tools/export_views_yolo.py --model work_dirs/yolo11/hrsc_yolo11s/weights/best.pt --source data/HRSC2016_MS_yolo_obb/images/val --out-root views/hrsc_yolo11s/val/0.67 --imgsz 672 --views r0
```

Oriented R-CNN uses `tools/export_views_mmrotate.py` with `--config`, `--checkpoint`,
`--classes data/mmrotate/<name>/annfiles/classes.json` and `--img-scale` (1024, 832, 672)
in place of `--model` and `--imgsz`. On DOTA, `--only-labelled data/dota_yolo_obb_1024/labels/<split>`
skips the crops without labels.

### 4. Aggregation rules

```bash
python tools/build_cache.py --jobs 8
```

```bash
python tools/run_cell.py
```

```bash
python tools/run_cell.py --ablations --detectors hrsc_yolo11s dior_r_yolo11s dota_yolo11s fair1m2_yolo11s
```

`build_cache.py` writes the clusters of every detector, family and split to `caches/`
(36 GB for all 152 caches). `run_cell.py` then fits and scores every rule on every cell,
one process per cell. It writes the rescored predictions to `preds/` and the metrics to
`results/`. Both take `--detectors` and `--families` to run a subset. The ablations run on
the four development detectors, which were also used to choose the settings of ASA.

| Paper | Rule in the code |
|---|---|
| WBF, max score | `wbf_max` |
| WBF, original score (p0, the prior of ASA) | `wbf_orig` |
| GBDT re-scoring | `gbdt` |
| OBBStacking / OBBStacking score, WBF box | `obbstacking` / `obbstacking_wbfbox` |
| ASA, score only / ASA | `asa_score` / `asa` |
| Ablations: without the penalty on delta, without the WBF prior, without the member encoder, Gaussian box average | `asa_no_penalty`, `asa_no_prior`, `asa_no_members`, `wbf_orig_gaussbox` |
| Rotated NMS / Soft-NMS (C4) | `python tools/fuse.py --rule nms softnms --families c4` |

### 5. Statistics, tables and cost

```bash
python tools/fuse.py --rule wbf_max --families flip
```

```bash
python tools/bootstrap.py vs-r0 --rule wbf_max --families flip d2 c4 d4 ms
```

```bash
python tools/bootstrap.py contrasts
```

```bash
python tools/make_tables.py --out tables
```

`make_tables.py` writes the LaTeX tables of the paper and `facts.txt`. The `results/`
folder ships with the paper's numbers, so this step runs right after cloning. The cost
tables come from `tools/benchmark_views.py` (GPU, batch size 1) and
`tools/benchmark_aggregator.py` (CPU, one thread).

### Exactness

* From our exported views, `tools/run_cell.py` with PyTorch 2.8.0 on the CPU (eight threads,
  the default) reproduced the numbers in `results/` exactly in our check (HRSC2016-MS,
  YOLO11s, C4). Other PyTorch versions change the last digits, by about 0.002 AP with
  PyTorch 2.11.
* Views exported on another GPU, or on the CPU, differ in the last digits of the boxes.
* Retrained detectors differ by the run-to-run variation of detector training.

## Citation

```bibtex
@misc{deng2026asa,
  title  = {Learning When and How to Trust Test-Time Augmentation in Oriented Object Detection},
  author = {Deng, Yongtao and Lei, Dajiang and Zhang, Liping and Li, Jiaxin and Li, Weisheng},
  year   = {2026},
  note   = {Under review}
}
```

## License

MIT, see [LICENSE](LICENSE). The datasets and the Ultralytics and MMRotate code keep their
own licenses.
