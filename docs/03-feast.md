# 03. 같은 사용자 피처를 과거와 현재에서 조회하기

오늘까지 쌓인 평점 수를 작년의 예측에 넣으면 미래 정보를 미리 알려 준 셈입니다. 피처 스토어에서는 값 자체뿐 아니라 어느 시점에 알 수 있었는지를 함께 다룹니다.

이번 과정은 앞에서 만든 영화 평점으로 사용자 평균 평점과 평점 수를 계산합니다. Feast라는 이름의 파일을 만드는 데 그치지 않고 실제 SDK로 정의 등록, 과거 시점 조회, materialization과 온라인 조회를 실행합니다.

## 영화 데이터에서 피처 정의 만들기

[영화 추천 과정](01-movies.md)의 `data/movies`가 필요합니다. 없다면 먼저 `movies prepare`를 실행합니다.

```powershell
uv run --extra feast ml-labs tools feast init --data data/movies --output integrations/feast
uv run --extra feast ml-labs tools feast run --repo integrations/feast --action apply
```

`feature_store.yaml`, `features.py`, 피처 Parquet와 메타데이터가 만들어집니다. 기존 정의를 덮어쓰지 않으므로 다시 만들 때는 새 출력 폴더를 사용합니다.

사용자마다 학습 이벤트를 시간 순으로 누적해 `average_rating`, `ratings_count`를 계산합니다. 검증과 테스트의 평점은 이 집계에 넣지 않습니다. 최신 통계 하나를 과거 날짜로 붙여 넣는 것이 아니라 각 이벤트 시점의 스냅샷을 남깁니다.

SVD 모델 자체는 사용자별 점수 행렬을 사용합니다. 피처를 조회했다고 가중치가 자동으로 바뀌는 것은 아닙니다. 대신 [통합 운영 과정](09-integrated-operations.md)의 실제 추천 API는 Feast에서 평점 수와 관측 시점을 조회하고, 충분한 이력이 있으면 개인화 추천, 적으면 인기 추천으로 분기합니다. 모델과 피처 원천의 지문, 누락과 최신성도 검사합니다.

## 먼저 온라인 조회가 실패하는지 보기

```powershell
uv run --extra feast ml-labs tools feast run --repo integrations/feast --action online --user-id u007
```

아직 materialization을 하지 않았다면 오류로 종료되어야 합니다. 정의를 등록하는 `apply`와 온라인 값을 채우는 materialization은 다른 작업입니다. 피처가 없는데 0점이나 빈 정상 결과로 바꾸면 문제를 숨길 수 있습니다.

## 과거 시점 조회

```powershell
uv run --extra feast ml-labs tools feast run --repo integrations/feast --action historical --user-id u007
```

이 명령은 같은 사용자를 서로 다른 기준 시점에서 조회합니다. 첫 평점 이전에는 값이 없고, 이후 시점에서는 그때까지 쌓인 평점 수가 나와야 합니다. 기본 합성 데이터에서 중간과 마지막 시점의 수가 달라지는지 확인하세요.

Feast의 과거 조회가 모든 시간 의미를 자동 해결하는 것은 아닙니다. 이 실습은 데이터가 이벤트 발생 시점에 바로 사용 가능하다고 가정합니다. 실제 데이터의 수집 지연, 정정과 backfill을 모델링하려면 이벤트 시점과 실제 사용 가능 시점을 구분해야 합니다.

## 일부 시점까지만 온라인으로 채우기

```powershell
uv run --extra feast ml-labs tools feast run --repo integrations/feast --action materialize --end-date "2025-01-05T00:00:00+00:00"
uv run --extra feast ml-labs tools feast run --repo integrations/feast --action online --user-id u007
```

현재 생성 데이터의 이벤트는 2025년 1월에 시작합니다. 기본 데이터에서 `u007`의 평점 수는 이 시점까지 5건입니다. 데이터를 바꾸었다면 숫자를 외우기보다 원천 이벤트와 비교해야 합니다.

나머지 이벤트도 채우고 조회합니다.

```powershell
uv run --extra feast ml-labs tools feast run --repo integrations/feast --action materialize
uv run --extra feast ml-labs tools feast run --repo integrations/feast --action online --user-id u007
```

기본 합성 데이터에서는 학습 평점 14건까지 반영됩니다. 검증과 테스트 평점을 더한 18건이 나오는 것이 아닙니다.

온라인 저장소는 로컬 SQLite입니다. Redis 없이도 실제 Feast 흐름을 확인하기 위한 선택이며, 처리량과 지연을 운영 환경 수준으로 검증한 것은 아닙니다.

## 무엇을 남겨야 할까요

| 대상 | 확인할 내용 |
| --- | --- |
| entity | 원천 데이터에 실제 존재하는 사용자 ID인지 |
| source | 어떤 학습 데이터와 지문에서 만들어졌는지 |
| event timestamp | 통계가 어느 시점까지의 이벤트를 포함하는지 |
| historical query | 조회 시점보다 뒤의 평점이 섞이지 않는지 |
| online query | materialization 범위와 조회값이 일치하는지 |

Feast FeatureView의 TTL과 온라인 저장소 자체의 키 만료는 같은 개념이 아닙니다. 온라인 최신성, 늦게 도착한 이벤트와 피처 누락을 어떻게 처리할지는 별도 정책이 필요합니다.

## 연결을 더 깊게 만들기

평점 수에 따른 라우터는 [통합 운영 과정](09-integrated-operations.md)에서 실행합니다. 이어서 사용자 평균 평점과 장르 선호도를 모델 입력으로 활용하는 별도 순위 모델을 구현해 보세요. 그때는 피처 계약과 학습, 서빙 변환을 함께 바꾸어야 합니다.

한 줄 정리: 피처 정의, 과거 시점 조회와 온라인 값 공급은 서로 다른 단계이며 각각의 시점과 누락을 확인해야 합니다.

[이전: 객체 탐지](02-cards.md) / [다음: 파이프라인](04-pipelines.md)
