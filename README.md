# APTOS 2019 Diabetic Retinopathy Classification

> **당뇨망막병증 자동 등급 분류 — 버그 디버깅부터 Ordinal Regression 앙상블까지의 체계적 개선 과정**

망막 안저 영상으로 당뇨망막병증의 중증도(0~4단계)를 판별하는 딥러닝 모델을 개발하고,
**v1(Private 0.800) → v2 5-fold 앙상블(Private 0.892)** 으로 점진적으로 성능을 끌어올린 프로젝트입니다.

---

## Performance Summary

| Version | Model | Strategy | Private Score | Public Score |
|---------|-------|----------|:---:|:---:|
| v1 | EfficientNet-B0 (단일) | CrossEntropyLoss, augmentation 없음 | 0.800 | 0.572 |
| v2 single | EfficientNet-B3 (단일 fold) | CORAL Loss + WeightedSampler + Augmentation | 0.883 | 0.712 |
| v2 3-fold | EfficientNet-B3 (3-fold) | 위 전략 + 앙상블 | 0.890 | 0.736 |
| v2 4-fold | EfficientNet-B3 (4-fold) | 위 전략 + 앙상블 | 0.891 | 0.739 |
| **v2 5-fold** | **EfficientNet-B3 (5-fold)** | **위 전략 + 앙상블** | **0.892** | **0.735** |

> 평가지표: **Quadratic Weighted Kappa (QWK)** — 클래스 간 순서와 거리를 고려하는 의료 영상 표준 지표

**v1 → v2 최종: Private +0.092 향상**

---

## Problem Solving Process

### Phase 1. 베이스라인 구축 및 디버깅

#### 1-1. 가중치 로드 실패 해결
- **Problem:** 학습 환경(Colab)과 제출 환경(Kaggle)의 `timm` 버전 차이로 EfficientNet 내부 레이어 이름이 달라 `size mismatch` 에러 발생.
- **Solution:** 이름 끝부분(suffix) 매칭 + shape 검증을 결합한 유연한 가중치 로더를 구현.
- **Result:** 362/362 파라미터 100% 매칭 달성, 예측값 쏠림 현상 해소.

#### 1-2. 학습-추론 전처리 불일치 발견
- **Problem:** 학습 코드에서 `/255.0` 스케일링과 ImageNet Normalize가 누락되어 0~255 범위로 학습되었으나, 추론 코드에서는 Normalize를 적용하여 입력 분포가 완전히 불일치.
- **Solution:** 학습 코드의 `Dataset.__getitem__`을 한 줄씩 추적하여 전처리 파이프라인을 정확히 복원.
- **Result:** v1 Private Kappa **0.800** 달성. (전처리 일치 전에는 예측이 특정 클래스로 쏠림)

#### 1-3. 모델 구조 버그 발견
- **Problem:** `self.model.classfier` (오타)로 인해 새로운 Linear 레이어가 사용되지 않고, timm 내부의 1000개 출력 classifier를 통과하는 비정상 구조.
- **Solution:** 1000개 출력에서 앞 5개를 슬라이싱하는 방식으로 학습된 지능을 최대한 복원.
- **Lesson:** Python의 동적 속성 생성 특성상 오타가 런타임 에러를 발생시키지 않아 발견이 어려움.

### Phase 2. 전략적 성능 개선 (v1 → v2)

평가지표(QWK)와 데이터 특성을 분석하여 4가지 개선 전략을 우선순위에 따라 적용.

#### 2-1. 데이터 불균형 해소 (최우선)
- **Problem:** 클래스 분포 극심한 불균형 (class 0: 1,805개 vs class 3: 193개, 약 9:1).
- **Solution:** `WeightedRandomSampler`로 소수 클래스 샘플링 확률 보정 + Loss에 클래스별 역빈도 가중치 적용.
- **Result:** 소수 클래스(1, 3, 4)의 재현율 개선, 예측 분포가 실제 분포에 근접.

#### 2-2. 순서형 손실 함수 도입
- **Problem:** CrossEntropyLoss는 클래스 간 순서를 무시 (0→4 오분류와 0→1 오분류를 동일 취급). QWK는 먼 클래스 오분류에 제곱 페널티(0→4: 16배)를 부과하므로 근본적 불일치.
- **Solution:** **CORAL (Consistent Rank Logits) Ordinal Regression** 도입. 5-class 분류를 4개의 이진 질문("등급이 k보다 높은가?")으로 변환하여 순서 관계를 자연스럽게 학습.
- **Result:** 인접 클래스 혼동 감소, QWK 지표와 직접 연결되는 학습 신호 확보.

#### 2-3. Augmentation 및 학습 전략 정규화
- **Problem:** 기존 코드에 데이터 증강 전혀 없음. 고정 lr로 10 epoch만 학습.
- **Solution:** Albumentations 기반 종합 증강 (Flip, Rotate, ShiftScaleRotate, GaussNoise, OpticalDistortion, CoarseDropout) + ImageNet Normalize 적용 + CosineAnnealingLR 스케줄러 (1e-4 → 1e-6, 25 epochs).
- **Result:** Train-Valid Kappa 격차 0.04 수준으로 과적합 억제.

#### 2-4. 모델 업그레이드
- **Problem:** EfficientNet-B0 (5.3M params, 224px)의 표현력 한계.
- **Solution:** EfficientNet-B3 (12M params, 300px)로 업그레이드. T4 15GB GPU 제약 내에서 batch_size=16으로 안정 구동.
- **Result:** 미세 병변(microaneurysm 등) 감지력 향상.

### Phase 3. 앙상블 및 안정성

#### 3-1. 5-Fold 앙상블
- 5-Fold Stratified CV 구성, 각 fold별 독립 학습 후 로짓 평균 앙상블.
- fold 수 증가에 따른 성능 변화:

| 앙상블 | Folds | Private | 단일 모델 대비 |
|--------|:-----:|:-------:|:----------:|
| 단일 모델 | 1 | 0.883 | — |
| 3-fold | 0, 1, 2 | 0.890 | +0.007 |
| 4-fold | 0, 1, 2, 3 | 0.891 | +0.008 |
| **5-fold** | **0, 1, 2, 3, 4** | **0.892** | **+0.009** |

- 각 fold별 Validation Kappa:

| Fold | Best Kappa | Best Epoch |
|------|:---------:|:---------:|
| Fold 0 | 0.8975 | 24 |
| Fold 1 | 0.9000 | 16 |
| Fold 2 | 0.8984 | 18 |
| Fold 3 | 0.8919 | 15 |
| Fold 4 | 0.8937 | 20 |
| **평균** | **0.8963** | — |

#### 3-2. 학습 안정성 설계
- 무료 Colab 환경의 런타임 끊김에 대비하여 매 epoch Google Drive 체크포인트 저장 + 자동 이어하기(resume) 구현.
- Gradient Clipping (max_norm=1.0)으로 학습 안정성 확보.

---

## 1등 솔루션 대비 Gap 분석

1등(Guanshuo Xu, Private 0.936)의 핵심 전략과 본 프로젝트의 차이를 분석하여 추가 개선 방향을 도출.

| 항목 | 1등 솔루션 | 본 프로젝트 (v2) |
|------|----------|---------------|
| 데이터 | 2019 + 2015 DR + IDRiD + Messidor (~40,000장) | 2019만 (3,662장) |
| Pseudo labeling | ✅ test set soft label로 재학습 | 미적용 |
| 모델 앙상블 | 4종 아키텍처 × 8개 모델 | 1종 × 5-fold |
| 입력 크기 | 512×512 | 300×300 |
| Pooling | GeM (Generalized Mean Pooling) | Average Pooling |
| 전처리 | 없음 (단순 resize) | Ben Graham |
| Loss | SmoothL1Loss (회귀) | CORAL Ordinal |
| 후처리 | QWK threshold 수동 최적화 | 고정 (0.5) |

**핵심 인사이트:** 0.89 → 0.93 구간의 성능 향상은 모델 아키텍처가 아닌 **데이터 전략**(외부 데이터 확보, pseudo labeling)이 지배적.

---

## Visualization

### Grad-CAM (Attention Map)

<img width="794" height="394" alt="Grad-CAM" src="https://github.com/user-attachments/assets/ce4c303d-d6e9-423f-9e80-9ed028f1f2d2" />

모델이 안구 내 혈관 및 실제 병변 의심 부위를 주요 특징으로 추출하고 있음을 시각적으로 확인.

### Confusion Matrix

<img width="788" height="701" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/ed7e225a-1e6f-4d41-9736-a5ffa940711b" />

인접 클래스 간 오진을 제외하면 중증도 구분이 논리적으로 이루어지고 있음.

---

## Tech Stack

| Category | Details |
|----------|---------|
| Language | Python 3.12 |
| Framework | PyTorch |
| Architecture | EfficientNet-B3 (timm) |
| Loss Function | CORAL Ordinal Regression |
| Augmentation | Albumentations |
| Preprocessing | Ben Graham's method (Gaussian blur contrast enhancement) |
| Visualization | Grad-CAM (pytorch-grad-cam) |
| Training Infra | Google Colab (T4 GPU) + Google Drive checkpoint |

---

## Project Structure

```
├── APTOS_v2_CORAL.ipynb          # 재학습 코드 (Colab)
├── aptos_kaggle_inference.py      # 캐글 제출 코드 (단일 모델)
├── aptos_kaggle_ensemble.py       # 캐글 제출 코드 (5-fold 앙상블)
├── aptos_checkpoints/
│   ├── best_fold0.pth
│   ├── best_fold1.pth
│   ├── best_fold2.pth
│   ├── best_fold3.pth
│   └── best_fold4.pth
└── README.md
```

---

## Lessons Learned

1. **데이터 파이프라인을 먼저 의심하라** — 성능 문제의 80%는 모델이 아니라 전처리 불일치, 정규화 누락에 있었다.
2. **평가지표가 전략을 결정한다** — QWK의 순서형 특성이 CORAL Loss 선택을 직접 결정했다.
3. **제약 조건 안에서 우선순위를 정하라** — 무료 Colab이라는 한계 내에서 가장 효과적인 4가지 전략을 선별 적용했다.
4. **천장을 알아야 방향이 보인다** — 1등 솔루션 분석으로 다음 단계의 핵심이 데이터 확보임을 파악했다.
5. **앙상블의 수확 체감** — 1→3 fold에서 +0.007, 3→5 fold에서 +0.002. 같은 아키텍처의 fold 추가보다 다른 아키텍처 추가가 더 효과적.
