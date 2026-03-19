# APTOS 2019 Diabetic Retinopathy Classification

> **EfficientNet-B0를 활용한 당뇨망막병증 다중 클래스 분류 및 진단 근거 시각화 프로젝트**

본 프로젝트는 안저 영상을 분석하여 질병의 중증도(0~4단계)를 판별하고, 의료 AI의 신뢰성을 높이기 위해 Grad-CAM을 통한 진단 근거 시각화를 구현하였습니다.

## Key Performance
- **Quadratic Weighted Kappa: 0.8508**
- **Validation Accuracy: 80.22%**

## Key Problem Solving

### 1. 시각적 노이즈 및 장비 편차 해결
- **Problem:** 촬영 환경에 따른 조명/색상 불균형 및 불필요한 배경 노이즈.
- **Solution:** **Ben Graham's Preprocessing** (Auto-Crop, Gaussian Blur 기반 대비 향상) 적용.
- **Result:** 장비 간 편차를 제거하고 혈관 및 미세 병변의 대비를 극대화함.

### 2. 학습 데이터 불균형 대응
- **Problem:** 정상(Label 0) 데이터 편중으로 인한 모델의 다수 클래스 편향 위험.
- **Solution:** **Weighted CrossEntropy Loss** 도입 (희귀 클래스에 높은 가중치 부여).
- **Result:** 데이터가 적은 Label 3, 4 단계에 대한 판별력 강화.

### 3. 미세 병변 정보 손실 방지
- **Problem:** 리사이징 시 초기 진단의 핵심인 미세 특징 소실 가능성.
- **Solution:** **EfficientNet-B0** 아키텍처 채택 및 입력 해상도 최적화.
- **Result:** 육안으로 구분이 어려운 초기 단계(Label 1)의 특징 포착 성능 향상.

### 4. 모델 판단 근거의 투명성 확보 (XAI)
- **Problem:** 딥러닝 모델의 블랙박스 특성으로 인한 진단 근거 확인 불가.
- **Solution:** **Grad-CAM** 시각화 기법 구현.
- **Result:** 모델이 실제 병변 부위에 집중하고 있음을 히트맵으로 증명하여 설명 가능성 확보.

## 📊 Visualization

### Confusion Matrix
![Confusion Matrix](./results/confusion_matrix.png)  
*주석: 인접 클래스 간 오진을 제외하면 중증도 구분이 논리적으로 이루어지고 있음을 확인.*

### Grad-CAM (Attention Map)
![Grad-CAM](./results/grad_cam_result.png)  
*주석: 모델이 안구 내 혈관 및 실제 병변 의심 부위를 주요 특징으로 추출하고 있음.*

## Tech Stack
- **Language:** Python 3.12
- **Framework:** PyTorch
- **Architecture:** EfficientNet-B0
- **Library:** timm, pytorch-grad-cam, OpenCV, Scikit-learn
