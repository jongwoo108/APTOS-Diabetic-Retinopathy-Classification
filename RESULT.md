# APTOS 2019 — 1등 솔루션 (Guanshuo Xu) vs 우리 v2 비교

## 1등 최종 점수: Private 0.936 | 우리 v2: Private 0.883

---

## 1. 전처리
| 항목 | 1등 | 우리 v2 |
|------|-----|---------|
| 방식 | **전처리 없음**, 단순 resize만 | Ben Graham (가우시안 차감) + crop |
| 근거 | "이미지 품질이 충분히 좋아서 딥러닝 입력에 전처리 불필요" | 2015 대회 우승 기법 차용 |
| 교훈 | 과한 전처리는 오히려 정보를 손실시킬 수 있음. 모델이 충분히 크면 원본이 더 나음 |

## 2. 모델
| 항목 | 1등 | 우리 v2 |
|------|-----|---------|
| 아키텍처 | inception_resnet_v2, inception_v4, SE-ResNeXt50, SE-ResNeXt101 (4종) | EfficientNet-B3 (1종) |
| 모델 수 | 8개 (각 아키텍처 × 2 시드) | 1개 (단일 fold) |
| 입력 크기 | **512×512** (일부 384) | 300×300 |
| Pooling | **GeM (Generalized Mean Pooling)** — 학습 가능한 파라미터 p로 avg와 max 사이를 자동 조절 | 기본 Average Pooling |
| 교훈 | 다양한 아키텍처 앙상블이 핵심. 입력 크기는 클수록 좋음. GeM은 간단하지만 효과적 |

## 3. 손실 함수
| 항목 | 1등 | 우리 v2 |
|------|-----|---------|
| Loss | **SmoothL1Loss** (회귀 접근) | CORAL Ordinal Loss |
| 출력 | 1개 연속값 (회귀) | 4개 이진 출력 (순서형) |
| 후처리 | QWK threshold 최적화 [0.7, 1.5, 2.5, 3.5] | sigmoid > 0.5 고정 |
| 교훈 | 회귀로 풀면 threshold 조정으로 추가 점수 확보 가능. 마지막에 [0.5→0.7] 변경만으로 Private +0.001 |

## 4. 데이터 전략 (가장 큰 차이)
| 항목 | 1등 | 우리 v2 |
|------|-----|---------|
| 학습 데이터 | 2019 train + **2015 대회 전체 데이터** + **IDRiD** + **Messidor** | 2019 train만 (3,662장) |
| 총 데이터량 | **약 40,000장 이상** | 3,662장 |
| Pseudo labeling | ✅ Stage1 모델로 test set 예측 → soft label로 재학습 | 없음 |
| 외부 데이터 라벨 처리 | IDRiD: 원본 라벨과 예측값 평균. Messidor: 그룹 평균으로 보정 | 해당 없음 |
| 교훈 | **데이터 양이 모델 성능의 가장 큰 결정 요인**. 같은 도메인의 외부 데이터를 적극 활용 |

## 5. 학습 전략
| 항목 | 1등 | 우리 v2 |
|------|-----|---------|
| 단계 | **2단계** (Stage1: 기본 학습 → Stage2: pseudo label + 외부 데이터 추가 10 epoch) | 1단계 |
| Validation | 2015+2019 전체를 train으로 쓰고 **Public LB만으로 검증** | 5-fold CV |
| Augmentation | Contrast, Brightness, Hue, Saturation, Blur/Sharpen, Rotate 180°, Scale, Shear, Shift, Mirror | Flip, Rotate, ShiftScale, Noise, Distortion, CoarseDropout |
| 교훈 | LB 피드백을 전략적으로 활용. Pseudo labeling은 2단계 학습의 핵심 |

## 6. 앙상블 & 후처리
| 항목 | 1등 | 우리 v2 |
|------|-----|---------|
| 앙상블 | 8개 모델 단순 평균 | 단일 모델 |
| Threshold | **수동 최적화** (마지막에 0.5→0.7으로 변경) | 고정 (0.5) |
| 교훈 | 회귀 출력의 threshold 튜닝은 무료 점수. 앙상블은 다양한 아키텍처일수록 효과적 |

---

## 핵심 교훈 요약 (임팩트 순)

### 1. 외부 데이터 + Pseudo labeling (최대 임팩트)
1등의 가장 큰 무기. 3,662장 → 40,000장+으로 데이터를 10배 이상 늘림.
2015 DR 대회 데이터는 같은 태스크라 직접 활용 가능.
Pseudo label은 test set의 분포를 학습에 반영하는 효과.

### 2. 다중 아키텍처 앙상블
서로 다른 구조(Inception 계열 + ResNeXt 계열)의 앙상블이
같은 구조 여러 개보다 훨씬 효과적.
"Inceptions and ResNets usually blend well"이라는 경험칙.

### 3. 큰 입력 크기 (512×512)
우리 300 vs 1등 512. 망막 이미지는 미세한 병변(microaneurysm 등)이 중요해서
해상도가 높을수록 유리. GPU 메모리가 허락하는 한 키우는 게 좋음.

### 4. 회귀 접근 + Threshold 최적화
SmoothL1Loss로 연속값을 예측하고, QWK에 최적화된 threshold를 찾는 방식.
[0.5,1.5,2.5,3.5] 대신 [0.7,1.5,2.5,3.5]로 바꾸는 것만으로 점수 상승.

### 5. GeM Pooling
Average Pooling → GeM으로 교체하는 것은 코드 3줄 변경으로
일관되게 성능 향상. 거의 공짜 점수.

### 6. 전처리는 최소화
1등은 Ben Graham 전처리를 쓰지 않음.
"모델이 충분히 크고 데이터가 충분하면 원본이 최선"이라는 관점.
