# 05. 선택한 모델을 서비스하고 변화 관찰하기

등록된 모델은 창고에 보관한 상품과 비슷합니다. 고객이 주문할 수 있게 만드는 서빙은 별도의 과정입니다. 모델 등록에 성공해도 HTTP 요청, 입력 검사와 운영 관측까지 저절로 만들어지지는 않습니다.

## 고정 버전으로 로컬 API 실행하기

[영화 추천](01-movies.md), [카드 탐지](02-cards.md), [RAG](06-rag-llmops.md) 중 준비한 모델만 전달하면 됩니다. 먼저 영화 추천으로 시작합니다.

```powershell
uv run ml-labs serve --movies .ml-labs/runs/movies/candidate.json --host 127.0.0.1 --port 8000
```

다른 PowerShell에서 요청합니다.

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
$body = @{ user_id = "u000" } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/recommend -Method Post -ContentType application/json -Body $body
$body = @{ user_id = "new-user" } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/recommend -Method Post -ContentType application/json -Body $body
```

기존 사용자는 `personalized`, 새 사용자는 `cold_start_popularity`가 나와야 합니다. 학습 때 본 영화는 기존 사용자의 추천에서 빠집니다.

등록 후 candidate의 `registered_uri`는 `models:/이름/버전`입니다. `champion` 별칭이 나중에 이동해도 이미 선택한 후보가 몰래 다른 버전으로 바뀌지 않습니다. 새 버전은 명시적으로 후보를 선택하고 서비스를 다시 시작해야 반영됩니다. 모델을 메모리에 올린 뒤 별칭만 바꿔도 실행 중 서비스가 갱신되지는 않습니다.

## 세 모델 함께 실행하기

```powershell
uv run ml-labs serve --movies .ml-labs/runs/movies/candidate.json --cards .ml-labs/runs/cards/candidate.json --rag .ml-labs/runs/rag/candidate.json
```

`http://127.0.0.1:8000/docs`에서 요청 스키마를 볼 수 있습니다. `/predict/cards`는 서버가 읽을 수 있는 합성 이미지 경로를 받습니다. `/rag/query`는 `{"query": "What is a feature store?"}`를 받습니다.

인증 없는 학습용 API입니다. `0.0.0.0`으로 공개하거나 다른 사람의 파일 경로를 읽는 서비스로 사용하지 마세요. 실서비스에는 인증, 업로드 검증, 요청 크기 제한과 접근 정책이 필요합니다.

## BentoML로 같은 모델 패키징하기

```powershell
uv sync --locked --extra bento
uv run --extra bento ml-labs tools bento build --candidate .ml-labs/runs/movies/candidate.json --output integrations/bento-movies
uv run --extra bento bentoml serve service:PlatformLabService --working-dir integrations/bento-movies --port 3000
```

이 명령의 `build`는 MLflow 모델을 BentoML 모델 저장소에 가져오고 `service.py`, `bentofile.yaml`을 생성합니다. Bento 패키지를 만드는 명령은 별도로 실행합니다.

```powershell
uv run --extra bento bentoml build integrations/bento-movies
uv run --extra bento bentoml list
```

BentoML의 `/predict`에서 다음처럼 호출합니다.

```powershell
$body = @{ request = @{ user_id = "u000" } } | ConvertTo-Json -Depth 4
Invoke-RestMethod http://127.0.0.1:3000/predict -Method Post -ContentType application/json -Body $body
```

생성된 코드를 읽고 모델 태그가 어디서 고정되는지 확인합니다. 이 단계는 movie 또는 rag PyFunc를 대상으로 합니다. Bento를 만들었다고 컨테이너 이미지가 빌드되거나 클러스터에 배포된 것은 아닙니다.

## 지연 시간과 요청 수

기본 API의 `/metrics`는 Prometheus 형식입니다.

```powershell
(Invoke-WebRequest http://127.0.0.1:8000/metrics).Content
```

`ml_labs_inference_requests_total`과 `ml_labs_inference_seconds`를 찾아 실제 요청 후 값이 바뀌는지 확인합니다. 이는 최소 계측이며 모든 실패 유형이나 SLO를 구현한 것은 아닙니다. Prometheus 서버, Grafana 대시보드와 알림 배포도 현재 저장소의 자동 설치 범위 밖입니다.

모델 정확도가 떨어져도 HTTP는 200을 반환할 수 있습니다. 서비스 지연과 예측 품질을 따로 관찰해야 하는 이유입니다.

## Evidently로 데이터 변화 확인하기

현재 평점 분포를 의도적으로 바꾼 비교 파일을 만듭니다. 아래는 별도 파일을 만드는 실험이지 학습 데이터 원본을 수정하는 명령이 아닙니다.

```powershell
uv run python -c "import pandas as pd; d=pd.read_csv('data/movies/ratings.csv'); d[['rating']].to_csv('.ml-labs/reference.csv',index=False); d['rating']=1.0; d[['rating']].to_csv('.ml-labs/current.csv',index=False)"
uv run --extra monitor ml-labs tools drift --reference .ml-labs/reference.csv --current .ml-labs/current.csv --output .ml-labs/reports/drift.json --backend evidently
```

`drift.html`을 브라우저에서 열어 reference와 current의 분포를 비교합니다. `--backend psi`는 직접 계산하는 PSI 기준선이며 Evidently를 실행한 것과 같지 않습니다.

드리프트는 입력 분포의 변화입니다. 그 자체로 모델이 틀렸다는 증거나 자동 재학습 승인 사유가 아닙니다. 성능 라벨, 사용자 영향, 데이터 오류 여부를 조사한 뒤 대응해야 합니다.

## 복구 실험과 정리

같은 모델 이름으로 두 버전을 등록하고 이전 candidate 사본의 고정 URI로 서비스를 다시 시작해 보세요. 추천 결과가 어떤 버전에서 나온 것인지 확인합니다. 이 과정은 로컬 재시작 방식이며 트래픽 분할이나 무중단 복구를 구현하지는 않습니다.

API, BentoML과 MLflow UI는 각각 실행한 터미널에서 Ctrl+C로 종료합니다. 생성 모델은 `.ml-labs`, `mlruns`, BentoML의 사용자 모델 저장소에 남을 수 있습니다. 삭제 전에 실행 기록의 참조 관계를 확인하세요.

한 줄 정리: 모델 버전 고정, 요청 동작, 서비스 지연과 데이터 변화를 각각 확인해야 배포 이후를 설명할 수 있습니다.

[이전: 파이프라인](04-pipelines.md) / [다음: RAG와 LLMOps](06-rag-llmops.md)
