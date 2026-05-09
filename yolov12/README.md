# conda `test` 환경 구축 가이드

이 저장소에서는 환경 정보를 아래 두 파일로 관리합니다.

## 1. `pyproject.toml`

- 용도: Python 버전과 핵심 학습 의존성을 문서화합니다.
- 사용 시점: 프로젝트의 기본 환경 구성을 빠르게 파악하고 싶을 때 사용합니다.

## 2. `requirements.tst`

- 용도: 원래 `conda test` 환경에서 추출한 `pip freeze` 스냅샷입니다.
- 사용 시점: 다른 컴퓨터에서 최대한 동일한 환경을 재현하고 싶을 때 사용합니다.

## 권장 설치 순서

1. 저장소 루트로 이동합니다.

```bash
cd /path/to/yolov12
```

2. 같은 Python 버전으로 conda 환경을 생성합니다.

```bash
conda create -n test python=3.10.9 -y
```

3. 환경을 활성화합니다.

```bash
conda activate test
```

4. `pip` 관련 도구를 먼저 업데이트합니다.

```bash
python -m pip install --upgrade pip setuptools wheel
```

5. 로컬 YOLOv12 패키지를 editable 모드로 설치합니다.

```bash
python -m pip install -e ./yolov12
```

6. 원본 환경 스냅샷을 설치합니다.

```bash
python -m pip install -r requirements.tst
```

## 가벼운 설치 방법

정확히 같은 환경까지는 필요 없고, 핵심 프로젝트 의존성만 설치하고 싶다면 아래처럼 진행할 수 있습니다.

```bash
python -m pip install .
python -m pip install -e ./yolov12
```

참고:

- `python -m pip install .` 는 루트의 `pyproject.toml` 을 사용합니다.
- 이 프로젝트는 `./yolov12` 아래의 로컬 Ultralytics 포크를 사용하므로 editable 설치를 권장합니다.

## 주의사항

1. `requirements.tst` 안에는 아래 항목이 포함되어 있습니다.

```text
-e ./yolov12
```

이 때문에 설치 명령은 반드시 저장소 루트에서 실행해야 합니다.

2. `requirements.tst` 는 ROS 관련 Python 패키지가 함께 설치된 환경에서 추출되었습니다.

- 다른 컴퓨터에서는 운영체제, CUDA, 드라이버, ROS 구성 차이 때문에 일부 패키지 설치가 실패할 수 있습니다.
- 완전 재현이 안 될 경우에는 먼저 "가벼운 설치 방법"으로 시작한 뒤, 필요한 패키지만 추가하는 방식이 더 안정적입니다.

3. `requirements.tst` 에는 GPU 관련 패키지도 포함되어 있습니다.

- 대상 컴퓨터에 호환되는 NVIDIA 드라이버와 CUDA 런타임이 있어야 합니다.

4. `torch` 나 `onnxruntime-gpu` 설치가 실패하면, 해당 머신에 맞는 버전을 먼저 설치한 뒤 아래 명령을 다시 실행하세요.

```bash
python -m pip install -r requirements.tst
```

## 설치 확인

설치가 끝난 뒤에는 아래 명령으로 환경을 확인할 수 있습니다.

```bash
python -V
python -m pip show torch torchvision timm albumentations
```

간단한 추가 확인:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "from ultralytics import YOLO; print('ultralytics import ok')"
```

## 요약

- 최대한 동일하게 재현하려면 `requirements.tst` 를 사용합니다.
- 더 가볍게 설치하려면 `pyproject.toml` 기반으로 설치합니다.
- 이 저장소를 제대로 사용하려면 `./yolov12` 를 editable 모드로 설치하는 것을 권장합니다.
