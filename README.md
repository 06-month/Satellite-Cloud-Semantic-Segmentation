# Cloud Segmentation - CMX Model

CMX (Cross-Modal Fusion) 모델을 사용한 구름 세그멘테이션 프로젝트입니다.

**Kaggle Competition:** [Clouds Segmentation 2025](https://www.kaggle.com/competitions/clouds-segmentation-2025)

## 📋 목차

- [프로젝트 개요](#프로젝트-개요)
- [주요 기능](#주요-기능)
- [프로젝트 구조](#프로젝트-구조)
- [설치 방법](#설치-방법)
- [사용 방법](#사용-방법)
- [모델 아키텍처](#모델-아키텍처)
- [성능](#성능)
- [참고 문헌](#참고-문헌)

## 🎯 프로젝트 개요

이 프로젝트는 RGB 이미지와 NIR(Near-Infrared) 이미지를 활용하여 구름을 세그멘테이션하는 딥러닝 모델입니다. CMX(Cross-Modal Fusion) 아키텍처를 기반으로 하며, 다음과 같은 클래스를 예측합니다:

- **Class 0**: Background (배경)
- **Class 1**: Thick Cloud (두꺼운 구름)
- **Class 2**: Thin Cloud (얇은 구름)
- **Class 3**: Cloud Shadow (구름 그림자)

## ✨ 주요 기능

### 모델
- **CMX (Cross-Modal Fusion)**: RGB-X 세그멘테이션을 위한 Cross-Modal 융합 아키텍처
- **MiT Backbone**: Mix Transformer 백본 (B1, B2, B3, B4 variants)
- **FRM & FFM**: Feature Rectify Module과 Feature Fusion Module
- **Pretrained Weights**: HuggingFace SegFormer 사전학습 가중치 지원

### 데이터 증강
- **Copy-Paste Augmentation**: 구름 인스턴스 복사-붙여넣기
- **Geometric Transforms**: Crop, Flip, Rotation, ShiftScaleRotate
- **Color Augmentation**: Brightness, Contrast, HSV, CLAHE
- **Separate Normalization**: RGB와 NIR 채널 별도 정규화

### 학습 기법
- **Gradient Accumulation**: 효과적인 배치 크기 증가
- **Mixed Loss**: OHEM + Dice Loss 조합
- **Learning Rate Scheduling**: Cosine Annealing / ReduceLROnPlateau
- **Separated Learning Rate**: Backbone과 Head에 다른 학습률 적용

## 📁 프로젝트 구조

```
.
├── config.py              # 설정 파일
├── train.py              # 학습 스크립트
├── test.py               # 테스트 및 제출 스크립트
├── requirements.txt      # 패키지 의존성
├── models/               # 모델 관련 모듈
│   ├── __init__.py
│   ├── modules.py        # FRM, FFM 모듈
│   ├── decoder.py        # MLP Decoder
│   ├── backbone.py       # MiT Transformer Backbone
│   └── cmx.py           # CMX 메인 모델
├── data/                # 데이터 관련 모듈
│   ├── __init__.py
│   ├── augmentations.py # 데이터 증강
│   └── dataset.py       # 데이터셋 클래스
└── utils/               # 유틸리티 모듈
    ├── __init__.py
    ├── losses.py        # 손실 함수
    ├── metrics.py       # 평가 메트릭
    └── utils.py         # 기타 유틸리티
```

## 🔧 설치 방법

### 1. 저장소 클론

```bash
git clone <repository-url>
cd cloud-segmentation
```

### 2. 가상환경 생성 (권장)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

### 필수 패키지
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- albumentations >= 1.3.0
- timm >= 0.9.0
- transformers >= 4.30.0
- opencv-python >= 4.7.0
- pandas, numpy, matplotlib, tqdm

## 🚀 사용 방법

### 설정 변경

`config.py` 파일에서 하이퍼파라미터를 수정할 수 있습니다:

```python
# Paths
workspace_path = '/path/to/dataset'
output_path = '/path/to/output'

# Training
batch_size = 4
epochs = 60
patch_size = 512

# Model
cmx_backbone = 'mit_b2'  # 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4'

# Loss
loss_func = 'ohem+dice'

# Optimizer
lr_head = 3e-4
lr_backbone = 3e-5
accumulation_steps = 4

# Augmentation
use_copy_paste = True
```

### 학습

```bash
# 기본 설정으로 학습
python train.py

# 커스텀 설정으로 학습
python train.py \
    --workspace /path/to/dataset \
    --output /path/to/output \
    --epochs 100 \
    --batch_size 8 \
    --backbone mit_b3 \
    --seed 42
```

**학습 과정:**
- 데이터 로딩 및 전처리
- 모델 생성 및 사전학습 가중치 로드
- Epoch마다 학습 및 검증
- 5 에포크마다 검증 수행 및 시각화
- Best 모델 자동 저장 (`ckpt/cmx_best.pt`)

### 테스트 및 제출

```bash
# 기본 설정으로 테스트
python test.py

# 커스텀 checkpoint 사용
python test.py \
    --workspace /path/to/dataset \
    --output /path/to/output \
    --checkpoint /path/to/checkpoint.pt \
    --backbone mit_b2
```

**출력:**
- 예측 결과 이미지: `output/results/`
- 제출 파일: `output/submission.csv`

## 🏗️ 모델 아키텍처

### CMX (Cross-Modal Fusion)

CMX는 RGB와 NIR 두 가지 모달리티를 효과적으로 융합하는 아키텍처입니다.

```
Input: RGB (3 channels) + NIR (1 channel)
   ↓
[Dual MiT Encoders]
   ├─ RGB Encoder (MiT-B2)
   └─ NIR Encoder (MiT-B2)
   ↓
[4-Stage Feature Extraction]
   └─ Each stage:
      ├─ Patch Embedding
      ├─ Transformer Blocks
      ├─ FRM (Feature Rectify Module)
      └─ FFM (Feature Fusion Module)
   ↓
[MLP Decoder]
   └─ Multi-scale feature fusion
   ↓
Output: Segmentation Map (4 classes)
```

### 주요 컴포넌트

1. **MiT Backbone**: Hierarchical Vision Transformer
   - 4단계 피라미드 구조
   - Efficient Self-Attention with Spatial Reduction
   - Overlapping Patch Merging

2. **FRM (Feature Rectify Module)**
   - Channel-wise attention
   - Spatial-wise attention
   - Cross-modal feature refinement

3. **FFM (Feature Fusion Module)**
   - Cross-path attention
   - Channel embedding
   - Dual-stream feature fusion

4. **MLP Decoder**
   - Multi-scale feature aggregation
   - Lightweight head design

### Backbone Variants

| Model | Params | Depths | Embed Dims | Heads |
|-------|--------|--------|------------|-------|
| MiT-B1 | ~13M | [2,2,2,2] | [64,128,320,512] | [1,2,5,8] |
| MiT-B2 | ~25M | [3,4,6,3] | [64,128,320,512] | [1,2,5,8] |
| MiT-B3 | ~45M | [3,4,18,3] | [64,128,320,512] | [1,2,5,8] |
| MiT-B4 | ~62M | [3,8,27,3] | [64,128,320,512] | [1,2,5,8] |

## 📊 성능

### 학습 환경
- GPU: NVIDIA GTX 1080 Ti
- Batch Size: 4 (Effective: 16 with gradient accumulation)
- Epochs: 60
- Training Time: ~11 hours

### 평가 메트릭
- **mIOU**: Mean Intersection over Union
- **Pixel Accuracy**: Pixel-wise classification accuracy
- **Dice Score**: F1 score for segmentation

### 시각화

학습 중 검증 샘플이 자동으로 시각화되어 `ckpt/visuals/`에 저장됩니다:
- RGB 입력
- NIR 입력
- 예측 마스크
- Ground Truth 마스크

## 🛠️ 고급 사용법

### 커스텀 데이터셋

데이터셋 구조:
```
dataset/
├── train/
│   ├── rgb/        # RGB 이미지
│   ├── ngr/        # NIR 이미지 (채널 2에 NIR 데이터)
│   └── label/      # 라벨 이미지 (BGR 컬러)
└── test/
    ├── rgb/
    └── ngr/
```

라벨 색상 매핑:
- Background: `[0, 0, 0]` (Black)
- Thick Cloud: `[0, 0, 255]` (Red in BGR)
- Thin Cloud: `[0, 255, 0]` (Green in BGR)
- Cloud Shadow: `[0, 255, 255]` (Yellow in BGR)

### 손실 함수 커스터마이징

`utils/losses.py`에서 새로운 손실 함수를 추가할 수 있습니다:

```python
def custom_loss(preds, targets):
    # Your custom loss implementation
    return loss_value

# config.py에서 사용
loss_func = 'custom'
```

### 증강 기법 추가

`data/augmentations.py`에서 증강 파이프라인을 수정할 수 있습니다.

## 📝 참고 문헌

### CMX Model
```
@article{zhang2023cmx,
  title={CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation with Transformers},
  author={Zhang, Jiaming and Liu, Huayao and Yang, Kailun and Hu, Xinxin and Liu, Ruiping and Stiefelhagen, Rainer},
  journal={arXiv preprint arXiv:2203.04838},
  year={2023}
}
```

### SegFormer (MiT Backbone)
```
@inproceedings{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  booktitle={NeurIPS},
  year={2021}
}
```

## 🤝 기여

버그 리포트, 기능 요청, Pull Request는 언제나 환영합니다!

## 📄 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다.

## 🔗 링크

- **Kaggle Competition**: [Clouds Segmentation 2025](https://www.kaggle.com/competitions/clouds-segmentation-2025)
- **CMX Paper**: [arXiv:2203.04838](https://arxiv.org/abs/2203.04838)
- **SegFormer**: [Hugging Face](https://huggingface.co/docs/transformers/model_doc/segformer)

---

**Happy Cloud Segmentation! ☁️**

