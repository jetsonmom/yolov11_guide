# yolov11_guide

# YOLOv11 초보자 가이드 🚀

## 목차
- [YOLOv11이란?](#yolov11이란)
- [주요 특징](#주요-특징)
- [YOLOv8 vs YOLOv11 비교](#yolov8-vs-yolov11-비교)
- [YOLO 용어 정리](#yolo-용어-정리)
- [설치 방법](#설치-방법)
- [모델 종류](#모델-종류)
- [기본 사용법](#기본-사용법)
- [지원하는 작업](#지원하는-작업)
- [참고 자료](#참고-자료)

---

## YOLOv11이란?

YOLOv11은 Ultralytics에서 2024년에 출시한 **최신 실시간 객체 탐지 모델**입니다. "You Only Look Once"의 약자로, 이미지를 단 한 번만 확인하여 객체를 탐지하는 혁신적인 방식을 사용합니다.

### 왜 YOLOv11을 사용해야 할까요?

- ⚡ **더 빠른 속도**: YOLOv10보다 약 2% 빠른 추론 시간
- 🎯 **높은 정확도**: YOLOv8m보다 22% 적은 파라미터로 더 높은 mAP 달성
- 💪 **효율성**: 적은 컴퓨팅 리소스로 뛰어난 성능
- 🌐 **다양한 플랫폼**: 엣지 디바이스, 클라우드, NVIDIA GPU 모두 지원
- 🎨 **다목적**: 탐지, 분류, 세그멘테이션, 포즈 추정 등 다양한 작업 지원

---

## 주요 특징

### 1. 향상된 아키텍처
- **C3K2 블록**: 더 빠르고 효율적인 특징 추출
- **C2PSA 블록**: 공간 주의 메커니즘으로 중요한 영역에 집중
- **SPPF 모듈**: 다중 스케일 특징 추출 유지

### 2. 새로운 기능
- **NMS-Free Training**: Non-Maximum Suppression 대체로 추론 시간 단축
- **Dual Label Assignment**: 겹치는 객체 탐지 개선
- **Large Kernel Convolutions**: 적은 리소스로 더 나은 특징 추출
- **Partial Self-Attention (PSA)**: 선택적 주의 메커니즘

---

## YOLOv8 vs YOLOv11 비교

| 항목 | YOLOv8 | YOLOv11 |
|------|--------|---------|
| **출시 연도** | 2023 | 2024 |
| **백본** | CSPDarkNet | 개선된 CSPDarkNet |
| **핵심 블록** | C2f 블록 | C3K2 블록 (Neck), C2PSA 블록 추가 |
| **파라미터 수** | 기준 (예: YOLOv8m) | 22% 감소 (YOLOv11m) |
| **정확도 (mAP)** | 기준 | 더 높음 (파라미터 감소에도) |
| **추론 속도** | 빠름 | YOLOv10 대비 2% 더 빠름 |
| **주의 메커니즘** | 제한적 | C2PSA를 통한 공간 주의 강화 |
| **NMS** | 사용 | NMS-Free 옵션 |
| **OBB (회전 객체 탐지)** | 제한적 | 완전 지원 |
| **최적화** | 양호 | 향상된 학습 파이프라인 |
| **적용 환경** | 다양함 | 엣지 디바이스에 더 최적화 |

### 성능 비교 요약
```
YOLOv11m vs YOLOv8m:
✅ 파라미터: 22% 감소
✅ mAP: 증가
✅ 속도: 더 빠름
✅ 메모리: 더 효율적
```

---

## YOLO 용어 정리

| 용어 | 영문 | 설명 | 예시/활용 |
|------|------|------|-----------|
| **객체 탐지** | Object Detection | 이미지에서 객체의 위치와 종류를 찾는 작업 | 자율주행차의 보행자/차량 탐지 |
| **바운딩 박스** | Bounding Box | 객체를 감싸는 사각형 영역 | (x, y, width, height) 좌표 |
| **mAP** | mean Average Precision | 모델의 정확도를 측정하는 지표 (0~1, 높을수록 좋음) | mAP@0.5 = 0.85 |
| **FPS** | Frames Per Second | 초당 처리 가능한 프레임 수 (높을수록 빠름) | 30 FPS = 실시간 처리 가능 |
| **추론** | Inference | 학습된 모델로 예측을 수행하는 과정 | 새로운 이미지에 모델 적용 |
| **백본** | Backbone | 이미지에서 기본 특징을 추출하는 신경망 부분 | CSPDarkNet, ResNet |
| **넥** | Neck | 백본과 헤드를 연결하며 다중 스케일 특징 통합 | FPN, PAN 구조 |
| **헤드** | Head | 최종 예측 출력을 생성하는 부분 | 바운딩 박스 + 클래스 예측 |
| **앵커** | Anchor | 객체 탐지를 위한 사전 정의된 바운딩 박스 | Anchor-Free (YOLOv11은 사용 안 함) |
| **NMS** | Non-Maximum Suppression | 중복된 탐지 박스를 제거하는 후처리 기법 | IoU 임계값으로 필터링 |
| **IoU** | Intersection over Union | 두 박스의 겹침 정도를 측정 (0~1) | IoU > 0.5 = 같은 객체로 판단 |
| **세그멘테이션** | Segmentation | 픽셀 단위로 객체를 구분하는 작업 | 의료 영상에서 종양 영역 분리 |
| **포즈 추정** | Pose Estimation | 사람의 관절 위치를 찾는 작업 | 스포츠 동작 분석, 피트니스 앱 |
| **클래스** | Class | 객체의 종류/카테고리 | 사람, 자동차, 고양이, 개 등 |
| **신뢰도** | Confidence Score | 예측의 확실성 정도 (0~1) | 0.9 = 90% 확신 |
| **배치 크기** | Batch Size | 한 번에 처리하는 이미지 수 | Batch Size = 16 |
| **에포크** | Epoch | 전체 데이터셋을 한 번 학습하는 과정 | 100 epochs 학습 |
| **전이 학습** | Transfer Learning | 사전 학습된 모델을 활용하는 기법 | COCO로 학습된 모델 사용 |
| **파인튜닝** | Fine-tuning | 사전 학습 모델을 특정 데이터로 재학습 | 내 데이터셋으로 추가 학습 |
| **데이터 증강** | Data Augmentation | 학습 데이터를 인위적으로 늘리는 기법 | 회전, 크롭, 색상 변경 등 |
| **OBB** | Oriented Bounding Box | 회전된 사각형 박스로 객체 탐지 | 드론 이미지의 차량 각도 탐지 |
| **SPPF** | Spatial Pyramid Pooling Fast | 다양한 크기의 특징을 효율적으로 통합 | 다중 스케일 객체 탐지 |
| **CSP** | Cross Stage Partial | 네트워크의 연산량을 줄이는 구조 | 메모리 효율성 향상 |
| **PSA** | Partial Self-Attention | 일부 영역에만 주의 메커니즘 적용 | 중요 영역 집중, 연산량 절감 |

---

## 설치 방법

### 1. 필수 요구사항
- Python 3.8 이상
- PyTorch 1.8 이상
- CUDA (GPU 사용 시, 선택사항)

### 2. Ultralytics 패키지 설치

```bash
# pip를 이용한 설치
pip install ultralytics

# 또는 최신 개발 버전 설치
pip install git+https://github.com/ultralytics/ultralytics.git
```

### 3. 설치 확인

```python
from ultralytics import YOLO

# 모델 로드 및 버전 확인
model = YOLO('yolo11n.pt')
print("YOLOv11 설치 완료!")
```

---

## 모델 종류

YOLOv11은 다양한 크기의 모델을 제공합니다. 용도에 맞게 선택하세요!

| 모델 | 파라미터 수 | 용도 | 추천 사용처 |
|------|-------------|------|-------------|
| **YOLOv11n** | 최소 (Nano) | 경량 작업 | 모바일, 임베디드 디바이스 |
| **YOLOv11s** | 소형 (Small) | 일반적인 실시간 작업 | 웹캠, 저사양 PC |
| **YOLOv11m** | 중형 (Medium) | 범용 목적 | 일반적인 프로젝트, 균형잡힌 성능 |
| **YOLOv11l** | 대형 (Large) | 높은 정확도 필요 | 고성능 서버, 정밀 탐지 |
| **YOLOv11x** | 초대형 (Extra-Large) | 최고 정확도 | 연구, 최고 성능이 필요한 프로젝트 |

### 작업별 모델

각 크기별로 다양한 작업을 지원하는 모델이 있습니다:

- **Detect**: 일반 객체 탐지 (`yolo11n.pt`, `yolo11s.pt`, ...)
- **Segment**: 인스턴스 세그멘테이션 (`yolo11n-seg.pt`, ...)
- **Pose**: 포즈 추정 (`yolo11n-pose.pt`, ...)
- **Classify**: 이미지 분류 (`yolo11n-cls.pt`, ...)
- **OBB**: 회전 객체 탐지 (`yolo11n-obb.pt`, ...)

---

## 기본 사용법

### 1. 객체 탐지 (Object Detection)

```python
from ultralytics import YOLO

# 모델 로드
model = YOLO('yolo11n.pt')

# 이미지에서 객체 탐지
results = model('image.jpg')

# 결과 시각화
results[0].show()

# 결과 저장
results[0].save('result.jpg')
```

### 2. 비디오 처리

```python
# 비디오 파일 처리
results = model('video.mp4')

# 웹캠 실시간 처리
results = model(source=0, show=True)  # 0 = 기본 웹캠
```

### 3. 커스텀 학습

```python
# 내 데이터셋으로 모델 학습
model = YOLO('yolo11n.pt')

# 학습 시작
results = model.train(
    data='custom_data.yaml',  # 데이터셋 설정 파일
    epochs=100,               # 학습 반복 횟수
    imgsz=640,               # 이미지 크기
    batch=16,                # 배치 크기
    device=0                 # GPU 번호 (CPU는 'cpu')
)
```

### 4. 학습된 모델로 예측

```python
# 학습된 모델 로드
model = YOLO('runs/detect/train/weights/best.pt')

# 예측 수행
results = model.predict('test_image.jpg', conf=0.5)  # 신뢰도 50% 이상만 표시
```

### 5. 배치 예측

```python
# 여러 이미지 동시 처리
images = ['image1.jpg', 'image2.jpg', 'image3.jpg']
results = model(images)

for i, result in enumerate(results):
    result.save(f'result_{i}.jpg')
```

---

## 지원하는 작업

### 1. Object Detection (객체 탐지)
이미지나 비디오에서 객체의 위치와 종류 탐지

**활용 예시**: 자율주행, 보안 감시, 소매 분석

### 2. Instance Segmentation (인스턴스 세그멘테이션)
픽셀 단위로 객체 분리 및 구분

**활용 예시**: 의료 영상 분석, 로봇 비전

### 3. Image Classification (이미지 분류)
이미지를 사전 정의된 카테고리로 분류

**활용 예시**: 제품 분류, 품질 검사

### 4. Pose Estimation (포즈 추정)
이미지/비디오에서 주요 관절 포인트 탐지

**활용 예시**: 피트니스 추적, 스포츠 분석, 동작 인식

### 5. Object Tracking (객체 추적)
비디오에서 객체의 움직임 추적

**활용 예시**: 교통 모니터링, 스포츠 분석

### 6. Oriented Object Detection (OBB)
회전된 객체를 각도와 함께 탐지

**활용 예시**: 항공 이미지 분석, 창고 자동화

---

## 참고 자료

### 공식 문서
- [Ultralytics YOLOv11 공식 문서](https://docs.ultralytics.com/models/yolo11/)
- [GitHub 저장소](https://github.com/ultralytics/ultralytics)

### 데이터셋
- [COCO Dataset](https://cocodataset.org/) - 객체 탐지
- [ImageNet](https://www.image-net.org/) - 이미지 분류
- [DOTA](https://captain-whu.github.io/DOTA/) - 회전 객체 탐지

### 커뮤니티
- [Ultralytics Community Forum](https://community.ultralytics.com/)
- [GitHub Issues](https://github.com/ultralytics/ultralytics/issues)

---

## 자주 묻는 질문 (FAQ)

### Q1: YOLOv11과 YOLOv8 중 어떤 것을 사용해야 하나요?
**A**: 새 프로젝트라면 YOLOv11을 추천합니다. 더 빠르고 정확하며 파라미터도 적습니다.

### Q2: GPU 없이도 사용할 수 있나요?
**A**: 네! CPU에서도 작동하지만, GPU를 사용하면 훨씬 빠릅니다.

### Q3: 내 데이터로 학습하려면?
**A**: 커스텀 데이터셋을 YOLO 형식으로 준비하고 `model.train()` 함수를 사용하세요.

### Q4: 모바일에서도 사용 가능한가요?
**A**: YOLOv11n (Nano) 모델은 모바일/엣지 디바이스에 최적화되어 있습니다.

### Q5: 상업적 용도로 사용할 수 있나요?
**A**: AGPL-3.0 라이선스를 따르며, 상업적 사용은 Ultralytics 라이선스가 필요할 수 있습니다.

---

## 라이선스

YOLOv11은 [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 하에 배포됩니다.

---

## 기여하기

버그 리포트, 기능 제안, 풀 리퀘스트는 언제나 환영합니다!

---

**만든 날짜**: 2025년 11월  
**업데이트**: 정기적으로 업데이트 예정

---

⭐ 이 가이드가 도움이 되었다면 Star를 눌러주세요!

📧 질문이나 피드백이 있으시면 Issue를 열어주세요.
