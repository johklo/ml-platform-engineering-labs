# 02. 합성 카드 탐지에서 기준선과 학습 모델 비교하기

이 과정은 신분증의 글자를 읽는 OCR이 아니라 이미지 안에서 카드 영역을 찾는 객체 탐지입니다. 개인정보 없이 위치 검출을 배울 수 있도록 글자 대신 선과 도형이 그려진 합성 카드를 사용합니다.

## 데이터와 색상 기준선

```powershell
uv run ml-labs cards prepare --output data/cards --train 80 --validation 20 --test 20
uv run ml-labs cards train --data data/cards --output .ml-labs/runs/cards
uv run ml-labs cards evaluate --candidate .ml-labs/runs/cards/candidate.json --data data/cards --report .ml-labs/reports/cards.json --min-iou 0.75
uv run ml-labs register --candidate .ml-labs/runs/cards/candidate.json --report .ml-labs/reports/cards.json --name card-detector
```

`images/train`, `images/val`, `images/test`에는 PNG가 있고, 대응하는 `labels` 폴더에는 YOLO 형식의 텍스트가 있습니다. 한 줄은 클래스 번호, 중심 x, 중심 y, 폭과 높이이며 좌표는 이미지 크기로 나눈 값입니다. 빈 라벨은 카드가 없는 음성 예제입니다.

기본 `color_component_baseline`은 가장 넓게 반복되는 색으로 카드 영역을 찾는 규칙입니다. 학습 가능한 파라미터가 없는 비교 기준이지 YOLO를 흉내 낸 학습 모델이 아닙니다.

IoU는 정답 영역과 예측 영역의 겹치는 면적을 두 영역의 합집합 면적으로 나눈 값입니다. 평균 IoU와 `box_recall_iou_0_5`는 카드가 있는 이미지에서 계산합니다. 카드가 없는 이미지의 수와 잘못 검출한 수는 `negative_images`, `false_positives`로 따로 남깁니다. 이 단일 객체 평가를 표준 객체 탐지 mAP와 같게 해석하면 안 됩니다.

## 실제 YOLO 학습

```powershell
uv sync --locked --extra vision
uv run --extra vision ml-labs cards train --data data/cards --output .ml-labs/runs/cards-yolo --backend yolov8 --epochs 10
uv run --extra vision ml-labs cards evaluate --candidate .ml-labs/runs/cards-yolo/candidate.json --data data/cards --report .ml-labs/reports/cards-yolo.json --min-iou 0.5
```

처음 실행하면 Ultralytics가 `yolov8n.pt` 가중치를 내려받을 수 있습니다. 외부 네트워크와 디스크 공간이 필요하며 CPU에서는 시간이 걸립니다. 현재 CLI는 비교를 위해 CPU를 명시합니다.

`yolo/synthetic-cards/weights/best.pt`와 학습 로그를 확인합니다. `best.pt`는 검증 결과로 선택한 가중치입니다. 학습이 끝났다는 사실과 평가 게이트 통과는 별개이고, 열 번의 epoch나 위 임계치가 품질을 보장하지 않습니다.

자동 점검에서 실제 CPU 학습과 MLflow 모델 재로딩을 확인하려면 선택 테스트를 실행합니다. 1 epoch의 매우 작은 학습이므로 결과의 정확도를 인증하는 검사가 아닙니다.

```powershell
$env:ML_LABS_RUN_VISION = "1"
uv run --extra dev --extra vision pytest -q tests/test_vision.py
Remove-Item Env:ML_LABS_RUN_VISION
```

검출 모델을 MLflow에 저장할 때는 설정과 가중치, 추론 코드, 필요한 Ultralytics 의존성도 포함합니다. `candidate.json`에서 원천 run으로 이동해 어떤 backend를 사용했는지 확인하세요.

## 원격 추론의 입력 계약

로컬 파일 경로는 다른 서버에서 같은 파일을 가리키지 않습니다. 모델은 `image_path` 또는 `image_base64` 중 정확히 하나를 받습니다. Azure ML의 MLflow 추론에서는 다음처럼 실제 이미지 바이트를 보냅니다.

```powershell
$image = Get-ChildItem data/cards/images/test/*.png | Select-Object -First 1
$encoded = [Convert]::ToBase64String([IO.File]::ReadAllBytes($image.FullName))
$request = @{ input_data = @{ columns = @("image_base64"); data = @(@($encoded)) } }
$request | ConvertTo-Json -Depth 8 | Set-Content .ml-labs/card-request.json
```

관리형 엔드포인트에 보내는 단계는 [Azure 과정](07-azure.md)에 있습니다. 서버가 접근할 수 없는 개인 PC 경로를 보내는 방식은 사용하지 않습니다.

## 실패를 관찰하기

먼저 검증 IoU 상한을 1보다 크게 두어 게이트 거절과 보고서 생성을 확인합니다. 이후에는 별도 데이터 사본에 밝기 변화, 회전, 가림을 적용하고 새 manifest를 생성하는 전처리 단계를 구현해 보세요. 기존 파일을 바꾸고 원래 manifest로 평가하면 지문 오류로 먼저 차단되는 것이 정상입니다.

색상 기준선이 잘 맞는 것은 데이터 생성기의 특성을 이용했기 때문입니다. 실제 사진의 일반화를 보여 주지 않습니다. 실제 데이터로 확장할 때는 촬영 세션 또는 문서 단위로 분할하고, 동영상의 인접 프레임이 학습과 테스트에 나뉘지 않게 해야 합니다. 실제 신분증은 이 저장소에 추가하지 않습니다.

한 줄 정리: 쉬운 합성 데이터의 성공, 학습 모델의 성공과 실제 촬영 환경의 성공을 구분해야 합니다.

[이전: 영화 추천](01-movies.md) / [다음: Feast](03-feast.md)
