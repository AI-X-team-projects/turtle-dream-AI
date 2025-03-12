# AI 기반 자세 교정 시스템 - 테스트 버전

## 설치 방법

1. 필요한 패키지 설치:

```bash
pip install -r requirements.txt
```

2. MediaPipe 모델 다운로드:

- [pose_landmarker_heavy.task](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task) 파일을 다운로드
- 다운로드한 파일을 프로젝트 루트 디렉토리에 저장

## 실행 방법

```bash
python pose_detection_test.py
```

## 사용 방법

- 프로그램이 실행되면 웹캠이 활성화됩니다.
- 화면에 포즈 랜드마크가 녹색 점으로 표시됩니다.
- 종료하려면 'q' 키를 누르세요.

## 요구사항

- Python 3.8 이상
- 웹캠
- 충분한 조명
