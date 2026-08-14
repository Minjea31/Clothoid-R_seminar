너는 ROS, 자율주행 perception, YOLO 계열 객체검출, 모델 프루닝을 잘 설명하는 세미나 PPT 기획자이자 발표 스크립트 작성자야.

나는 아래 코드베이스를 기반으로 세미나 PPT를 만들려고 한다.

목표:
- `perception_ws/src/detect_msgs`의 object 관련 ROS 메시지 구조를 설명
- `perception_ws/src/yolo26`의 YOLO ROS 노드가 어떻게 실행되고 어떤 로직으로 bbox를 발행하는지 설명
- `perception_ws/yolo26`의 Ultralytics/YOLO26 runtime이 ROS 노드에서 어떻게 사용되는지 설명
- `~/yolo26_pruning/yolo26`에서 YOLO26 베이스라인 학습, 프루닝 기준, 프루닝 실행, fine-tuning, 평가 흐름을 설명
- 발표 대상은 코드를 직접 다 보지 않은 팀원/교수님이라고 가정하고, 슬라이드별 핵심 메시지, 발표 스크립트, 도식 아이디어, 예상 질문/답변까지 만들어줘.

코드 위치:
- ROS workspace: `/home/a/Clothoid-R/perception_ws`
- 메시지 패키지: `/home/a/Clothoid-R/perception_ws/src/detect_msgs`
- ROS YOLO wrapper 패키지: `/home/a/Clothoid-R/perception_ws/src/yolo26`
- ROS에서 import하는 YOLO26 라이브러리: `/home/a/Clothoid-R/perception_ws/yolo26`
- 프루닝/학습용 YOLO26 저장소: `/home/a/yolo26_pruning/yolo26`
- 데이터셋 yaml: `/home/a/yolo26_pruning/dataset.yaml`

중요한 이름 혼동:
- 폴더명은 `src/yolo26`이지만 ROS 패키지명은 `yolov12`로 되어 있다.
- 그래서 실행은 `rosrun yolov12 yolo_detect.py` 형태다.
- 코드 로그에는 `YOLOv12 MODEL LOADED`라고 나오지만, 실제 구조/설정은 YOLO26 계열 yaml/weight를 사용한다.

0. Python 3.10 conda 환경 생성 및 설치:
- ROS YOLO wrapper의 shebang이 `/home/a/anaconda3/envs/yolo/bin/python`으로 고정되어 있으므로 conda 환경 이름은 `yolo`로 만드는 것이 가장 자연스럽다.
- 환경 이름을 다르게 만들면 `src/yolo26/scripts/yolo_detect.py` 첫 줄 shebang도 새 환경 경로로 수정해야 한다.
- conda 환경 생성:
  - `conda create -n yolo python=3.10 -y`
  - `conda activate yolo`
  - `python -m pip install --upgrade pip setuptools wheel`
- ROS inference용 YOLO26 라이브러리 설치:
  - `cd /home/a/Clothoid-R/perception_ws/yolo26`
  - `pip install -r requirements.txt`
  - `pip install -e .`
- 프루닝/학습용 YOLO26 저장소도 같은 환경을 써도 된다:
  - `cd /home/a/yolo26_pruning/yolo26`
  - `pip install -r requirements.txt`
  - `pip install -e .`
- 프루닝 코드 실행 전, repository의 patched prune utility를 conda 환경 안의 PyTorch prune.py로 교체해야 한다:
  - `cp "$CONDA_PREFIX/lib/python3.10/site-packages/torch/nn/utils/prune.py" "$CONDA_PREFIX/lib/python3.10/site-packages/torch/nn/utils/prune.py.bak"`
  - `cp /home/a/yolo26_pruning/asset/prune.py "$CONDA_PREFIX/lib/python3.10/site-packages/torch/nn/utils/prune.py"`
  - 의미: `/home/a/yolo26_pruning/asset/prune.py`를 현재 활성화된 conda 환경의 `torch/nn/utils/prune.py` 위치에 덮어써서, 이 프로젝트의 pruning 코드와 PyTorch pruning API 동작을 맞춘다.
- ROS Python helper 패키지:
  - `pip install rospkg catkin_pkg empy pyyaml`
- ROS 실행 전 터미널 source 순서:
  - `source /opt/ros/noetic/setup.bash`
  - `cd /home/a/Clothoid-R/perception_ws`
  - `catkin_make`
  - `source devel/setup.bash`
  - `conda activate yolo`
- 주의:
  - `yolo_detect.py`는 `cv_bridge`를 직접 쓰지 않고 `CompressedImage`를 OpenCV로 직접 decode하므로 conda Python 3.10 환경에서 실행하기 쉽다.
  - 반면 `livox_camera_fusion` 같은 C++/시스템 ROS 노드는 기존처럼 시스템 ROS 환경에서 실행하는 것을 전제로 한다.
  - PyTorch CUDA 버전이 맞지 않으면 `requirements.txt` 설치 후 GPU 인식이 안 될 수 있으니, 장비 CUDA에 맞는 PyTorch wheel로 재설치할 수 있다고 설명하면 좋다.

1. `detect_msgs` 메시지 패키지 요약:
- `package.xml`: catkin 메시지 패키지. `message_generation`, `message_runtime`, `std_msgs`, `geometry_msgs`, `sensor_msgs`에 의존한다.
- `CMakeLists.txt`: 아래 4개 msg를 `add_message_files()`에 등록하고 `generate_messages()`로 생성한다.
- `msg/Objects.msg`
  - `int32 Class`: 검출 클래스 id
  - `int32 id`: 프레임 내 객체 id
  - `int32 x1`, `int32 x2`, `int32 y1`, `int32 y2`: bbox 픽셀 좌표. 코드에서는 의미상 `(x1, y1, x2, y2)`로 사용한다.
- `msg/Yolo_Objects.msg`
  - `std_msgs/Header header`
  - `Objects[] yolo_objects`
  - YOLO 노드가 `/perception/camera/yolo`로 발행하고 fusion 노드가 구독하는 핵심 메시지다.
- `msg/detected_object.msg`
  - `int64 id`
  - `geometry_msgs/Pose world_point`
  - bbox가 아니라 월드 좌표 객체 포즈용 메시지로 보인다.
- `msg/detected_array.msg`
  - `std_msgs/Header header`
  - `detected_object[] objects`
  - 현재 확인한 YOLO/fusion 코드에서는 `Yolo_Objects`를 주로 쓰고, `detected_array` 계열은 향후/다른 노드용 월드좌표 인터페이스처럼 보인다.

2. ROS YOLO 노드 `src/yolo26/scripts/yolo_detect.py`:
- shebang: `/home/a/anaconda3/envs/yolo/bin/python`
- `sys.path.insert(0, "/home/a/Clothoid-R/perception_ws/yolo26")`로 워크스페이스 루트의 YOLO26/Ultralytics 포크를 import한다.
- `from ultralytics import YOLO`
- 기본 입력 토픽: `/camera/image_raw/compressed`
- 기본 출력 토픽: `/perception/camera/yolo`
- 기본 frame id: `camera_link`
- 기본 모델 yaml: `/home/a/Clothoid-R/perception_ws/src/yolo26/models/0524.yaml`
- 기본 weight: `/home/a/Clothoid-R/perception_ws/src/yolo26/models/0524.pt`
- ROS params:
  - `~source`
  - `~output_topic`
  - `~yaml_cfg`
  - `~pt_weights`
  - `~frame_id`
  - `~erp42_confidence`, `~drum_confidence`, `~cone_confidence`
- 클래스 설정:
  - class 0: `ERP-42`, publish true, confidence 0.5
  - class 1: `drum`, publish false, confidence 0.5
  - class 2: `cone`, publish false, confidence 0.5
  - 설정에 없거나 publish false인 클래스는 검출되어도 발행하지 않는다.
- 실행 흐름:
  - `rospy.init_node("yolo_detect_node")`
  - `YOLO(yaml_cfg, task="detect").load(pt_weights)`로 모델 로드
  - `sensor_msgs/CompressedImage`를 구독
  - callback에서 `cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)`로 이미지를 OpenCV BGR frame으로 변환
  - 원본 이미지 크기 `(h0, w0)`를 사용해 `self.model(frame, imgsz=(h0, w0), conf=self.conf_thres)[0]` 추론
  - 결과 bbox마다 `cls`, `conf`, `xyxy`를 읽음
  - 클래스별 publish 여부와 confidence threshold로 필터링
  - 통과한 bbox를 `detect_msgs/Objects`로 만들고 `Yolo_Objects.yolo_objects`에 append
  - header stamp는 입력 이미지 stamp를 사용하고, frame_id는 입력 header에 있으면 그대로 쓰고 없으면 `camera_link`를 사용
  - 결과가 없어도 빈 `Yolo_Objects` 메시지를 publish한다.
- 실행 예시:
  - 최초 1회 환경 생성은 위의 "Python 3.10 conda 환경 생성 및 설치" 섹션을 따른다.
  - `cd /home/a/Clothoid-R/perception_ws`
  - `catkin_make`
  - `source devel/setup.bash`
  - `conda activate yolo`
  - `rosrun yolov12 yolo_detect.py`
  - 파라미터 지정 예: `rosrun yolov12 yolo_detect.py _source:=/camera/image_raw/compressed _output_topic:=/perception/camera/yolo _erp42_confidence:=0.5`
  - 확인: `rostopic echo /perception/camera/yolo`

3. perception stack에서 YOLO 메시지가 쓰이는 흐름:
- `perception_bringup/launch/perception.launch`는 `livox_camera_fusion`과 `livox_clustering`을 launch한다.
- README 기준으로 `velodyne_detection`, `yolov12`는 launch에 포함하지 않고 별도 터미널에서 실행한다.
- `livox_camera_fusion`은 `/livox/lidar`, `/camera/image_raw/compressed`, `/perception/camera/yolo`를 `message_filters::ApproximateTime`으로 50ms 안에서 동기화한다.
- fusion 요약:
  - LiDAR point cloud를 카메라 이미지 좌표로 projection
  - 각 YOLO bbox 안에 들어온 LiDAR point를 모음
  - bbox 중심 기준 ROI 반경 컷
  - RANSAC으로 지면 제거
  - Euclidean clustering 후 가장 큰 cluster의 centroid 계산
  - Kalman tracker로 centroid 추적
  - `/perception/fusion/centroids`에 `sensor_msgs/PointCloud` 발행
  - `/perception/fusion/filtered_cloud`에 디버그 point cloud 발행

4. ROS 배포용 모델:
- `src/yolo26/models/0524.yaml`, `0524.pt`가 기본값이다.
- `0516_prune.yaml/.pt`, `prune_0510.yaml/.pt`도 같이 있다.
- yaml 구조는 `nc: 8`, `scales: prune [1, 1, 1024]`, backbone/head에 pruned channel 수가 직접 적힌 형태다.
- backbone은 Conv, C3k2, A2C2f 중심이고, head는 Upsample, Concat, A2C2f, Conv, C3k2, Detect 흐름이다.
- Detect는 `[14, 17, 20]` feature를 받아 `nc=8` detection을 수행한다.

5. 프루닝 저장소 `~/yolo26_pruning/yolo26` 구조:
- Ultralytics 기반 YOLO26 포크다.
- 핵심 실행 스크립트:
  - `train_baseline_model.py`: baseline 학습
  - `prune_finetune.py`: pruning 후 fine-tuning
  - `eval.py`: val/test 평가
- 핵심 프루닝 구현:
  - `compress/Compress.py`: PruneHandler
  - `compress/GM.py`: geometric median 기반 structured pruning
  - `ultralytics/nn/modules/conv.py`: `Conv.recon()` 추가
  - `ultralytics/nn/modules/block.py`: `Bottleneck`, `SPPF`, `C3k2`, `C2PSA` 등의 `recon()` 추가
  - `ultralytics/nn/modules/head.py`: `Detect.recon()` 추가
  - `ultralytics/nn/tasks.py`: `exact_channels`, `reg_max`, `end2end` 처리
  - `ultralytics/engine/model.py`: `_use_current_model_for_train`이면 pruned in-memory model을 그대로 trainer에 넘김

6. 데이터셋:
- `/home/a/yolo26_pruning/dataset.yaml`
- `train: dataset/train/images`
- `val: dataset/val/images`
- `test: dataset/test/images`
- `nc: 1`
- `names: ['CAR']`
- Roboflow project: `clothoid-ryhpb/crash_test`, version 1, license CC BY 4.0

7. Baseline 학습:
- `train_baseline_model.py` args:
  - `--name baseline`
  - `--bs 4`
  - `--epoch 100`
  - `--model_pt ../yolo26n.pt`
  - `--resume`
  - `--data ../dataset.yaml`
  - `--device 0`
- 내부 로직:
  - `model = YOLO(args.model_pt)`
  - `model.train(data=args.data, epochs=args.epoch, imgsz=640, device=args.device, name=args.name, batch=args.bs, workers=4, save_period=5, project=checkpoints)`
  - `--resume`이면 같은 train 옵션에 `resume=True` 추가
- 실행 예시:
  - `cd /home/a/yolo26_pruning/yolo26`
  - `conda activate yolo`
  - 최초 1회 설치가 아직 안 됐다면 `pip install -r requirements.txt && pip install -e .`
  - `python train_baseline_model.py --model_pt ../yolo26n.pt --data ../dataset.yaml --name baseline --epoch 100 --bs 4 --device 0`
- 현재 baseline 결과:
  - `checkpoints/baseline-2/weights/best.pt`
  - `results.csv` 기준 best mAP50은 epoch 60의 0.98255
  - best mAP50-95는 epoch 66의 0.83339
  - 마지막 확인 epoch 74는 precision 0.95449, recall 0.97826, mAP50 0.97313, mAP50-95 0.82894

8. 프루닝 기준과 실행:
- `prune_finetune.py` 주요 args:
  - `--bmodel`: baseline weight. 기본값은 오래된 경로라 실제 실행 때 override 필요
  - `--pruning_ratio`: 전체적으로 줄이고 싶은 비율. 기본 0.5
  - `--prune_type`: `ALL`, `H`, `B`
  - `--method`: `GM`, `L1`, `L2`
  - `--cfg_output_path`: pruned `.pt/.yaml` 저장 폴더
  - `--epoch`: pruning 후 fine-tuning epoch. 기본 600
  - `--name`: fine-tuning run 이름
  - `--bs`: batch size
  - `--resume_path`: fine-tuning resume용
  - `--align`: 남길 채널 수를 이 값의 배수로 맞춤. 기본 8
  - `--data`: 기본값은 `hackathon.yaml`이지만 현재는 `../dataset.yaml`로 주는 것이 안전
- `pruning_ratio` 변환:
  - 코드에서 `compression_ratio = 1 - sqrt(1 - pruning_ratio)`로 바꾼다.
  - 이유: conv는 input channel과 output channel을 같이 줄이면 대략 곱으로 연산량/파라미터가 줄기 때문에, 목표 전체 감소율을 per-channel 감소율로 환산한다.
- `prune_type`:
  - `ALL`: Detect layer를 제외한 대부분 Conv pruning
  - `H`: head만 pruning. backbone layer 0~10과 detect layer 제외
  - `B`: backbone layer 0~10만 pruning
- `method`:
  - `L1`: `torch.nn.utils.prune.ln_structured(..., n=1, dim=0)`
  - `L2`: `ln_structured(..., n=2, dim=0)`
  - `GM`: custom FPGM/geometric median 방식. 필터를 flatten하고 Euclidean distance matrix를 만들며, 거리합이 작은 필터를 다른 필터와 유사한 redundant filter로 보고 제거한다.
- `align`:
  - `_amt()`에서 keep channel 수를 `align` 배수로 반올림한다.
  - 기본 8은 양자화/NPU 커널 정렬 안정화를 위한 설정이다.
- 실행 예시:
  - `cd /home/a/yolo26_pruning/yolo26`
  - `python prune_finetune.py --bmodel checkpoints/baseline-2/weights/best.pt --data ../dataset.yaml --pruning_ratio 0.5 --prune_type ALL --method GM --cfg_output_path prune_gm50 --epoch 600 --name prune_gm50 --bs 4 --align 8 --device 0`

9. `Compress.py` 프루닝 내부 로직:
- `PruneHandler.__init__`
  - YOLO 모델을 CPU로 이동한다. GM 방식이 numpy/scipy 거리 계산을 사용하기 때문이다.
  - `self.cr`, `self.method`, `self.prune_type`, `self.align` 저장
- `_amt(module)`
  - 현재 Conv output channel 수 `num`
  - 남길 채널 수 `keep = round(num * (1 - cr) / align) * align`
  - 최소 `align`, 최대 `num`, 최소 1채널 보장
  - return은 `num - keep`, 즉 잘라낼 channel 개수
- `prune()`
  - Conv2d output channel dim=0에 structured pruning mask 적용
  - 같은 mask를 뒤따르는 BatchNorm2d weight/bias에도 적용
  - `prune.remove()`로 mask를 실제 weight에 반영
  - detect layer는 pruning하지 않는다.
- `reconstruct()`
  - weight가 0으로 남아 있는 sparse 모델이 아니라, 실제 channel dimension을 줄인 dense 모델로 재구성한다.
  - 각 Conv/C3k2/SPPF/C2PSA/Detect/Concat을 순서대로 돌며 살아남은 channel index를 전달한다.
  - concat layer에서는 이전 feature map과 skip connection의 channel offset을 보정한다.
  - Detect input은 layer 16, 19, 22의 살아남은 channel을 기준으로 head 첫 conv를 slice한다.
- `model_to_yaml()`
  - pruned 모델의 실제 channel 수를 yaml로 저장한다.
  - `nc`, `end2end`, `reg_max`를 baseline 모델에서 가져온다.
  - `exact_channels: True`를 넣어 parse_model이 channel을 width multiplier로 다시 반올림하지 않게 한다.
  - 출력: `{cfg_output_path}/best_model_prune.yaml`
- `compress_yolo26()`
  - output folder 생성
  - prune → reconstruct → yaml 저장
  - 모든 floating parameter `requires_grad=True`
  - `_use_current_model_for_train=True`
  - `{cfg_output_path}/best_model_prune.pt` 저장
  - pruned model 반환 후 바로 fine-tuning

10. 평가:
- `eval.py` args:
  - `--model_pt`
  - `--data`
  - `--split train|val|test`
  - `--imgsz 640`
  - `--batch 2`
  - `--workers 4`
- 실행 예시:
  - `python eval.py --model_pt prune_gm50/best_model_prune.pt --data ../dataset.yaml --split test --imgsz 640 --batch 2`

PPT 요청:
- 15~20분 발표용 슬라이드 12~15장으로 구성해줘.
- 각 슬라이드마다 제목, 핵심 bullet, 발표자가 말할 스크립트, 넣으면 좋은 그림/도식 아이디어를 줘.
- 전체 흐름은 다음 순서가 좋다:
  1. 발표 목적과 전체 perception pipeline
  2. ROS 메시지 인터페이스 `detect_msgs`
  3. YOLO ROS 노드 실행 구조
  4. 이미지 입력부터 bbox publish까지 callback 흐름
  5. YOLO 결과가 LiDAR-camera fusion으로 들어가는 방식
  6. YOLO26 모델 구조 개요
  7. 왜 프루닝이 필요한가
  8. Baseline 학습 방식
  9. 프루닝 방법 L1/L2/GM 비교
  10. pruning_ratio, prune_type, align의 의미
  11. prune → reconstruct → yaml/pt 저장 → fine-tuning 흐름
  12. 평가 방식과 baseline 결과
  13. ROS 배포 모델로 연결하는 방법
  14. 한계/주의사항
  15. 예상 질문과 답변
- 발표에서는 코드 함수명을 적당히 보여주되, 너무 긴 코드는 넣지 말고 흐름도와 메시지 구조 그림 위주로 설명해줘.
- 특히 `prune()`와 `reconstruct()`의 차이를 강조해줘. `prune()`는 제거할 channel을 고르고 mask를 반영하는 단계이고, `reconstruct()`는 실제 tensor shape을 줄여 dense pruned model로 만드는 단계다.
- 최종적으로 내가 바로 PPT로 옮길 수 있게 한국어로 깔끔하게 작성해줘.
