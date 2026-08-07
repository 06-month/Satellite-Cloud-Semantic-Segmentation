# CMX 기반 Cloud Semantic Segmentation (RGB + NIR)

**본 프로젝트는 3학년 2학기 컴퓨터 비전 수업의 기말 팀 프로젝트다.**

RGB와 근적외선(NIR) 위성 영상 쌍으로부터 구름을 4개 class로 분할한다. CMX cross-modal architecture
위에 구현했다.

Kaggle: [Clouds Segmentation 2025](https://www.kaggle.com/competitions/clouds-segmentation-2025)
— *Clouds Semantic Segmentation, 2025 가을학기, 한밭대학교.*

| 본인 최고 제출 | Private | Public |
|---|---:|---:|
| CMX (MiT-B2, RGB+NIR) | **0.78462** | **0.76443** |

수업 팀 프로젝트였으나 팀원 각자가 별도의 모델을 독립적으로 개발했다.
**이 저장소는 본인의 모델이며**, 위 점수는 본인이 직접 제출한 결과다.

---

## 과제

각 샘플은 정합된 한 쌍, 즉 RGB 이미지와 NIR(근적외선) 밴드로 이뤄진다. 모든 픽셀에 4개 class 중 하나를
할당한다.

| 인덱스 | Class | 라벨 색상 (BGR) |
|---:|---|---|
| 0 | 배경 | `[0, 0, 0]` |
| 1 | 두꺼운 구름 | `[0, 0, 255]` |
| 2 | 얇은 구름 | `[0, 255, 0]` |
| 3 | 구름 그림자 | `[0, 255, 255]` |

---

## Architecture

[CMX](https://github.com/huaaaliu/RGBX_Semantic_Segmentation)는 **두 개의 MiT encoder를 병렬로**
돌리면서 매 stage마다 두 encoder 사이에 정보를 교환한다.

```
RGB ──► MiT encoder ──┐
                      ├─► FRM (보정) ─► FFM (융합) ─► stage 출력   × 4 stage
NIR ──► MiT encoder ──┘
                                            ↓
                                    MLP decoder ─► 4-class logit
```

- **FRM (Feature Rectify Module)** — channel 단위와 공간 단위 gating. 융합 전에 각 modality가 상대편을
  보정한다.
- **FFM (Feature Fusion Module)** — 두 stream 사이의 cross-attention 후 channel embedding.

---

## 이 저장소가 적용한 변경

CMX는 RGB-D와 RGB-thermal 실내 벤치마크를 위해 설계되었다. 이를 위성 RGB+NIR로 옮기려면 입력, 사전학습,
augmentation 세 지점에서 변경이 필요했다. 아래 표는 upstream 코드와 이 과제를 위해 작성한 코드를
구분한 것이다.

| 구성 요소 | 출처 |
|---|---|
| `models/modules.py` — FRM, FFM, CrossAttention, CrossPath, ChannelEmbed | **Upstream CMX** |
| `models/backbone.py` — `RGBXTransformer` dual-encoder, MiT block | **Upstream CMX** |
| `models/decoder.py` — MLP decoder | **Upstream CMX (SegFormer 방식)** |
| `models/backbone.py` — `load_pretrained_from_transformers()` | 여기서 작성 |
| `models/cmx.py` — NIR channel adapter, model factory | 여기서 작성 |
| `data/augmentations.py` — 구름 instance Copy-Paste, 4채널 joint transform | 여기서 작성 |
| `data/dataset.py` — 색상 mask 디코딩, RAM 상주 로딩 | 여기서 작성 |
| `utils/losses.py` — OHEM+Dice 및 loss registry | 여기서 작성 |
| `train.py`, `test.py`, `config.py` | 여기서 작성 |

### 로컬 checkpoint 대신 HuggingFace 사전학습 가중치

Upstream CMX는 저자들이 제공하는 링크에서 MiT checkpoint(`mit_b2.pth`)를 받아 쓰도록 되어 있다. 여기서는
backbone이 `transformers`를 통해 [`nvidia/mit-b*`](https://huggingface.co/nvidia/mit-b2)를 불러오고
key를 다시 매핑한다.

처리해야 했던 불일치가 두 가지다.

1. **이름 규칙.** HuggingFace SegFormer는 `patch_embeddings.{i}` / `block.{i}` /
   `attention.self.query`를 쓰고, CMX는 `patch_embed{i+1}` / `block{i+1}` / `attn.q`를 쓴다.
2. **QKV 결합 방식.** HuggingFace는 `key`와 `value`를 별도 텐서로 저장하지만 CMX는 하나로 합친
   `attn.kv`를 쓴다. 로더가 둘을 모아 concat한다.

```python
if 'attention.self.key' in k:
    kv_weights.setdefault(base, {})['key'] = v      # 수집
...
w = torch.cat([kv['key'], kv['value']], dim=0)      # attn.kv로 결합
```

불러온 모든 텐서는 **`extra_*` 분기에도 복사**한다. 그래서 NIR encoder가 무작위 초기화가 아니라 RGB
encoder와 동일한 ImageNet 사전학습 가중치에서 출발한다.

### NIR은 1채널인데 CMX는 3채널을 기대한다

```python
if nir.shape[1] == 1:
    nir = nir.repeat(1, 3, 1, 1)
```

두 번째 encoder가 기대하는 3채널 입력을 맞추기 위해 밴드를 복제한다.

### 구름 instance에 대한 Copy-Paste

- source mask에서 연결 요소를 추출하고(`cv2.connectedComponentsWithStats`), 100 px 미만인 요소는
  건너뛴다.
- 요소 하나를 무작위로 골라 `[0.4, 1.2]` 범위의 배율로 크기를 조정한 뒤, 경계를 clamp하면서 임의 위치에
  붙인다.
- **RGB, NIR, mask를 같은 영역에 함께 붙인다.**

### 4채널 스택에 대한 기하 augmentation

기하 변환 전에 RGB와 NIR을 하나의 4채널 배열로 concat하고, 변환 후 다시 분리한다.

```python
combined = np.concatenate([rgb, nir], axis=-1)     # (H, W, 4)
aug = self.geom(image=combined, mask=mask)         # flip/rotation/distortion이 정렬을 유지한다
...
rgb_t, nir_t = img4[:3], img4[3:4]
```

정규화는 RGB에 ImageNet 통계를, NIR에 별도 통계를 사용한다
(`mean=[0.485, 0.456, 0.406, 0.5]`, `std=[0.229, 0.224, 0.225, 0.25]`).

---

## 학습 설정

| | |
|---|---|
| Backbone | MiT-B2 (`mit_b1`–`mit_b4` 선택 가능), 파라미터 66.6 M |
| 입력 | 512×512 random crop |
| Batch | 4, **gradient accumulation ×4** → 유효 배치 16 |
| Optimizer | AdamW, `weight_decay=2e-2` |
| Learning rate | **backbone 3e-5, head/decoder 3e-4** (10배 차이) |
| 스케줄 | Cosine annealing, `eta_min=1e-6` |
| Loss | `ohem+dice` = 0.7 · OHEM-CE (상위 25% 어려운 픽셀) + 0.3 · Dice |
| 분할 | 829장 → train 663 / val 166 (8:2) |
| 하드웨어 | Kaggle, **Tesla P100-PCIE-16GB** |

---

## 결과

**제출된 결과는 이 코드의 이전 버전에서 나온 것이다.**

| | 점수를 낸 실행 | 이 저장소의 현재 코드 |
|---|---|---|
| Loss | `dice+ce` (0.7 CE + 0.3 Dice) | `ohem+dice` |
| OHEM | 미구현 | 구현됨 |
| Epoch | 100 | 60 |
| Copy-Paste 배율 | `[0.3, 1.0]`, 경계 clamp 없음 | `[0.4, 1.2]`, clamp 적용 |
| 상태 | 완료, 제출됨 | **끝까지 실행한 적 없음** |

점수를 낸 실행의 결과다. 166장 validation 분할 기준이다.

| 지표 | 값 |
|---|---:|
| 최고 val mIoU | **0.5287** (epoch 94) |
| Val pixel accuracy | 0.8260 |
| 실행 시간 | epoch당 약 6분 33초 × 100 ≈ **10.9시간** |

학습에 따른 validation mIoU 추이: 0.3678 (ep 4) → 0.4071 (ep 9) → 0.4427 (ep 19) → 0.4961 (ep 34)
→ 0.5152 (ep 64) → 0.5237 (ep 79) → **0.5287 (ep 94)**. epoch 60 이후 향상 폭은 작았다
(마지막 35 epoch 동안 +0.013).

해당 실행의 Kaggle 점수: **Private 0.78462 / Public 0.76443**.

OHEM, 확장된 Copy-Paste 범위, 경계 처리 수정은 모두 그 제출 *이후*에 추가했다.

---

## 설치

```bash
pip install -r requirements.txt
```

PyTorch ≥ 2.0, torchvision ≥ 0.15, albumentations ≥ 1.3, timm ≥ 0.9, transformers ≥ 4.30,
opencv-python ≥ 4.7.

MiT 가중치는 최초 실행 시 HuggingFace에서 자동으로 내려받는다.

### 데이터 배치

```
<workspace>/
├── train/
│   ├── rgb/      # RGB 이미지
│   ├── ngr/      # NIR (BGR 인덱스 2번 채널)
│   └── label/    # 색상으로 부호화된 mask
└── test/
    ├── rgb/
    └── ngr/
```

경로는 `config.py`에서 설정한다(`workspace_path`, `output_path`). 기본값은 Kaggle의 input/working
디렉터리다.

---

## 사용법

```bash
# 학습
python train.py
python train.py --workspace /path/to/data --output /path/to/out \
                --epochs 100 --batch_size 8 --backbone mit_b3 --seed 42

# 추론 및 submission.csv 생성
python test.py --checkpoint ckpt/cmx_best.pt --backbone mit_b2
```

validation은 5 epoch마다 실행되며, RGB / NIR / 예측 / ground-truth 패널을 `ckpt/visuals/`에 저장하고
mIoU가 가장 높은 checkpoint를 `ckpt/cmx_best.pt`에 유지한다.

`CloudSeg.ipynb`는 같은 파이프라인의 Kaggle 단일 파일 버전이다. `models/`, `data/`, `utils/` 아래
모듈을 노트북 하나에 인라인한 것이며, 점수를 낸 실행이 아니라 현재 코드와 일치한다.

---

## 한계

- **이 저장소의 코드에는 측정된 결과가 없다.** 보고한 점수는 이전 버전의 것이고, 현재 설정은 끝까지
  학습한 적이 없다.
- **정확한 변인 통제 실험이 이루어지지 않았다.** 그래서 OHEM, Copy-Paste 범위, encoder/decoder를 분리한 learning rate의
  효과를 각각 측정하지 못했다.

---

## 저장소 구조

```
.
├── config.py                 # 모든 하이퍼파라미터
├── train.py                  # 학습 루프, gradient accumulation, validation, 시각화
├── test.py                   # 추론 + RLE 제출 파일 생성
├── CloudSeg.ipynb            # Kaggle 단일 파일 버전
├── models/
│   ├── cmx.py                # 모델 조립, NIR channel adapter
│   ├── backbone.py           # dual MiT encoder + HuggingFace 가중치 로더
│   ├── modules.py            # FRM / FFM (upstream CMX)
│   └── decoder.py            # MLP decoder
├── data/
│   ├── dataset.py            # 색상 mask 디코딩, RGB/NIR 페어링
│   └── augmentations.py      # Copy-Paste, 4채널 기하 변환
├── utils/
│   ├── losses.py             # OHEM-CE, Dice, Jaccard 및 조합
│   ├── metrics.py            # mIoU, pixel accuracy, Dice
│   └── utils.py
└── requirements.txt
```

---

## 참고 문헌

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

CMX architecture 코드는 공식 구현
[huaaaliu/RGBX_Semantic_Segmentation](https://github.com/huaaaliu/RGBX_Semantic_Segmentation)을
기반으로 한다. MiT backbone 가중치는 HuggingFace의
[NVIDIA SegFormer](https://huggingface.co/nvidia/mit-b2) 릴리스에서 가져왔다.
