# Cloud Segmentation with CMX (RGB + NIR)

**Four-class cloud segmentation from paired RGB and near-infrared satellite imagery, built on the
CMX cross-modal architecture.**

Kaggle: [Clouds Segmentation 2025](https://www.kaggle.com/competitions/clouds-segmentation-2025)
— *Clouds Semantic Segmentation, 2025 Fall, Hanbat National University.*

| My best submission | Private | Public |
|---|---:|---:|
| CMX (MiT-B2, RGB+NIR) | **0.78462** | **0.76443** |

This was a team course project in which each member developed a separate model independently.
**This repository is my model**, and the scores above are from my own submission.

---

## Task

Each sample is a co-registered pair: an RGB image and a NIR (near-infrared) band. Every pixel is
assigned one of four classes.

| Index | Class | Label colour (BGR) |
|---:|---|---|
| 0 | Background | `[0, 0, 0]` |
| 1 | Thick cloud | `[0, 0, 255]` |
| 2 | Thin cloud | `[0, 255, 0]` |
| 3 | Cloud shadow | `[0, 255, 255]` |

---

## Architecture

[CMX](https://github.com/huaaaliu/RGBX_Semantic_Segmentation) runs **two parallel MiT encoders** and exchanges information between them at every stage:

```
RGB ──► MiT encoder ──┐
                      ├─► FRM (rectify) ─► FFM (fuse) ─► stage output   × 4 stages
NIR ──► MiT encoder ──┘
                                            ↓
                                    MLP decoder ─► 4-class logits
```

- **FRM (Feature Rectify Module)** — channel-wise and spatial-wise gating; each modality corrects
  the other before fusion.
- **FFM (Feature Fusion Module)** — cross-attention between the two streams, then channel embedding.


---

## What This Repository Adapts

CMX is designed for RGB-D and RGB-thermal indoor benchmarks. Moving it to satellite RGB+NIR
required changes at the input, the pretraining, and the augmentation. The table separates upstream
code from what was written for this task.

| Component | Origin |
|---|---|
| `models/modules.py` — FRM, FFM, CrossAttention, CrossPath, ChannelEmbed | **Upstream CMX** |
| `models/backbone.py` — `RGBXTransformer` dual-encoder, MiT blocks | **Upstream CMX** |
| `models/decoder.py` — MLP decoder | **Upstream CMX (SegFormer-style)** |
| `models/backbone.py` — `load_pretrained_from_transformers()` | Written here |
| `models/cmx.py` — NIR channel adapter, model factory | Written here |
| `data/augmentations.py` — cloud-instance Copy-Paste, 4-channel joint transform | Written here |
| `data/dataset.py` — colour-mask decoding, in-RAM loading | Written here |
| `utils/losses.py` — OHEM+Dice and the loss registry | Written here |
| `train.py`, `test.py`, `config.py` | Written here |

### Pretrained weights from HuggingFace instead of local checkpoints

Upstream CMX expects MiT checkpoints (`mit_b2.pth`) downloaded from the authors' links. Here the
backbone instead loads [`nvidia/mit-b*`](https://huggingface.co/nvidia/mit-b2) through
`transformers` and remaps the keys.

Two mismatches had to be handled:

1. **Naming.** HuggingFace SegFormer uses `patch_embeddings.{i}` / `block.{i}` / `attention.self.query`;
   CMX uses `patch_embed{i+1}` / `block{i+1}` / `attn.q`.
2. **Fused QKV.** HuggingFace stores `key` and `value` as separate tensors; CMX uses a single fused
   `attn.kv`. The loader collects both and concatenates them.

```python
if 'attention.self.key' in k:
    kv_weights.setdefault(base, {})['key'] = v      # collect
...
w = torch.cat([kv['key'], kv['value']], dim=0)      # fuse into attn.kv
```

Every loaded tensor is also **copied into the `extra_*` branch**, so the NIR encoder starts from
the same ImageNet-pretrained weights as the RGB encoder rather than from random initialisation.

### NIR is one channel, CMX expects three

```python
if nir.shape[1] == 1:
    nir = nir.repeat(1, 3, 1, 1)
```

The band is replicated so the second encoder receives the 3-channel input it expects.

### Copy-Paste for cloud instances

- Connected components are extracted from the source mask (`cv2.connectedComponentsWithStats`),
  and components under 100 px are skipped.
- One component is chosen at random, rescaled by a factor in `[0.4, 1.2]`, and pasted at a random
  position with boundary clamping.
- **RGB, NIR and mask are pasted together in the same region.**

### Geometric augmentation on a 4-channel stack

RGB and NIR are concatenated into a single 4-channel array before the geometric transforms, then
split back afterwards:

```python
combined = np.concatenate([rgb, nir], axis=-1)     # (H, W, 4)
aug = self.geom(image=combined, mask=mask)         # flips/rotations/distortions stay aligned
...
rgb_t, nir_t = img4[:3], img4[3:4]
```

Normalisation uses ImageNet statistics for RGB and separate statistics for NIR
(`mean=[0.485, 0.456, 0.406, 0.5]`, `std=[0.229, 0.224, 0.225, 0.25]`).

---

## Training Setup

| | |
|---|---|
| Backbone | MiT-B2 (`mit_b1`–`mit_b4` selectable), 66.6 M parameters |
| Input | 512×512 random crop |
| Batch | 4, with **gradient accumulation ×4** → effective 16 |
| Optimizer | AdamW, `weight_decay=2e-2` |
| Learning rate | **backbone 3e-5, head/decoder 3e-4** (10× apart) |
| Schedule | Cosine annealing, `eta_min=1e-6` |
| Loss | `ohem+dice` = 0.7 · OHEM-CE (hardest 25 % of pixels) + 0.3 · Dice |
| Split | 829 images → 663 train / 166 val (8:2) |
| Hardware | Kaggle, **Tesla P100-PCIE-16GB** |

---

## Results

**The submitted result came from an earlier version of this code.**

| | Scored run | Current code in this repository |
|---|---|---|
| Loss | `dice+ce` (0.7 CE + 0.3 Dice) | `ohem+dice` |
| OHEM | not implemented | implemented |
| Epochs | 100 | 60 |
| Copy-Paste scale | `[0.3, 1.0]`, no boundary clamp | `[0.4, 1.2]`, clamped |
| Status | completed, submitted | **never run to completion** |

Scored run, on the 166-image validation split:

| Metric | Value |
|---|---:|
| Best val mIoU | **0.5287** (epoch 94) |
| Val pixel accuracy | 0.8260 |
| Runtime | ~6 min 33 s / epoch × 100 ≈ **10.9 hours** |

Validation mIoU over training: 0.3678 (ep 4) → 0.4071 (ep 9) → 0.4427 (ep 19) → 0.4961 (ep 34)
→ 0.5152 (ep 64) → 0.5237 (ep 79) → **0.5287 (ep 94)**. Gains after epoch 60 were small
(+0.013 over the last 35 epochs).

Kaggle score for that run: **Private 0.78462 / Public 0.76443**.

OHEM, the wider Copy-Paste range and the boundary fix were added *after* that submission. They
are in the code but **were never evaluated**, so no score is claimed for them.

---

## Setup

```bash
pip install -r requirements.txt
```

PyTorch ≥ 2.0, torchvision ≥ 0.15, albumentations ≥ 1.3, timm ≥ 0.9, transformers ≥ 4.30,
opencv-python ≥ 4.7.

MiT weights download automatically from HuggingFace on first run.

### Data layout

```
<workspace>/
├── train/
│   ├── rgb/      # RGB images
│   ├── ngr/      # NIR in channel 2 (BGR index 2)
│   └── label/    # colour-coded masks
└── test/
    ├── rgb/
    └── ngr/
```

Paths are set in `config.py` (`workspace_path`, `output_path`); the defaults are the Kaggle
input/working directories.

---

## Usage

```bash
# Train
python train.py
python train.py --workspace /path/to/data --output /path/to/out \
                --epochs 100 --batch_size 8 --backbone mit_b3 --seed 42

# Predict and build submission.csv
python test.py --checkpoint ckpt/cmx_best.pt --backbone mit_b2
```

Validation runs every 5 epochs, writes RGB / NIR / prediction / ground-truth panels to
`ckpt/visuals/`, and keeps the best-mIoU checkpoint at `ckpt/cmx_best.pt`.

`CloudSeg.ipynb` is the single-file Kaggle version of the same pipeline — the modules under
`models/`, `data/` and `utils/` inlined into one notebook. It matches the current code, not the
scored run.

---

## Limitations

- **The code in this repository has no measured result.** The reported scores belong to an earlier
  version; the current configuration was never trained to completion.
- **No experiment tracking**, so the effect of OHEM, of the Copy-Paste range, and of the split
  learning rates was never measured.

---

## Repository Structure

```
.
├── config.py                 # all hyperparameters
├── train.py                  # training loop, gradient accumulation, validation, visualisation
├── test.py                   # inference + RLE submission
├── CloudSeg.ipynb            # single-file Kaggle version
├── models/
│   ├── cmx.py                # model assembly, NIR channel adapter
│   ├── backbone.py           # dual MiT encoder + HuggingFace weight loader
│   ├── modules.py            # FRM / FFM (upstream CMX)
│   └── decoder.py            # MLP decoder
├── data/
│   ├── dataset.py            # colour-mask decoding, RGB/NIR pairing
│   └── augmentations.py      # Copy-Paste, 4-channel geometric transforms
├── utils/
│   ├── losses.py             # OHEM-CE, Dice, Jaccard, combinations
│   ├── metrics.py            # mIoU, pixel accuracy, Dice
│   └── utils.py
└── requirements.txt
```

---

## References

```bibtex
@article{zhang2023cmx,
  title={CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation with Transformers},
  author={Zhang, Jiaming and Liu, Huayao and Yang, Kailun and Hu, Xinxin and Liu, Ruiping and Stiefelhagen, Rainer},
  journal={arXiv preprint arXiv:2203.04838},
  year={2023}
}

@inproceedings{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  booktitle={NeurIPS},
  year={2021}
}
```

The CMX architecture code is based on the official implementation at
[huaaaliu/RGBX_Semantic_Segmentation](https://github.com/huaaaliu/RGBX_Semantic_Segmentation).
MiT backbone weights come from the [NVIDIA SegFormer](https://huggingface.co/nvidia/mit-b2)
release on HuggingFace.
