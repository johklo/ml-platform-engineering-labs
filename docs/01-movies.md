# 01. 영화 추천을 시간 순서대로 학습하고 평가하기

내일 볼 영화를 추천하는 모델이 내일의 평점까지 보고 학습했다면 시험 답안을 미리 본 것과 같습니다. 추천 데이터의 분할은 행을 무작위로 나누는 것보다 언제 알 수 있었던 정보인지가 중요합니다.

이번 과정에서는 사용자마다 먼저 발생한 평점을 학습에 사용하고, 이후 평점을 검증과 테스트에 사용합니다. 생성 데이터에서는 사용자와 영화의 숨은 취향 벡터로 평점을 만들지만, 모델에는 그 벡터를 알려 주지 않습니다.

## 작은 데이터로 전체 연결하기

```powershell
uv run ml-labs movies prepare --output data/movies
uv run ml-labs movies train --data data/movies --output .ml-labs/runs/movies --rank 4
uv run ml-labs movies evaluate --candidate .ml-labs/runs/movies/candidate.json --data data/movies --report .ml-labs/reports/movies.json --max-rmse 1.5
uv run ml-labs register --candidate .ml-labs/runs/movies/candidate.json --report .ml-labs/reports/movies.json --name movie-recommender
```

`ratings.csv`에는 `user_id`, `item_id`, `rating`, `timestamp`, `split`이 필요합니다. 같은 사용자와 영화 쌍의 중복, 범위를 벗어난 평점, 사용자별로 뒤섞인 분할 시점을 검사합니다.

`train`은 사용자 평균에서 벗어난 취향을 저차원 행렬로 근사하는 SVD를 사용합니다. 본 영화는 추천 목록에서 제외하고, 처음 보는 사용자는 학습 평점으로 계산한 영화 평균 순위를 사용합니다. 이 인기 기준선은 노출 수와 신뢰도를 보정한 서비스용 순위가 아닙니다.

## 결과를 읽는 순서

`candidate.json`의 `validation_rmse`를 `baseline_rmse`와 비교합니다. RMSE는 평점 예측 오차이고, 낮을수록 좋습니다. `rank`를 키웠다고 항상 좋아지지는 않습니다. 현재 등록 게이트는 검증 RMSE의 절대 상한이며 기준선보다 반드시 좋아야 한다는 조건은 아닙니다.

검증 게이트를 통과한 뒤에만 테스트 평점에서 Recall@5를 계산합니다. 여기서는 평점 4 이상을 관련 영화로 정의하고, 그 영화 중 상위 다섯 추천에 포함된 비율을 합산합니다. 관련 영화가 없으면 지표는 `null`이며 실패를 0점처럼 숨기지 않습니다.

평점 오차가 낮아도 추천 순서가 좋다는 보장은 없습니다. 테스트를 반복해서 보고 rank를 고르면 테스트도 사실상 모델 선택에 사용됩니다. 후보 비교는 검증 데이터로 하고, 테스트는 고른 후보의 마지막 확인에 사용하세요.

## 실제 MovieLens로 교체하기

합성 데이터는 연결 확인용입니다. 현실의 희소성과 인기 편향을 보려면 [GroupLens MovieLens latest-small](https://grouplens.org/datasets/movielens/latest/)에서 이용 조건을 읽고 ZIP을 직접 내려받습니다. 자동 다운로드와 재배포는 하지 않습니다.

```powershell
uv run ml-labs movies import-movielens --archive .\ml-latest-small.zip --output data/movielens
uv run ml-labs movies train --data data/movielens --output .ml-labs/runs/movielens --rank 16
uv run ml-labs movies evaluate --candidate .ml-labs/runs/movielens/candidate.json --data data/movielens --report .ml-labs/reports/movielens.json
```

가져오기는 사용자당 최소 10건과 서로 다른 시점 세 개 이상을 요구합니다. 동일 타임스탬프의 이벤트는 같은 분할에 묶습니다. 시점 그룹의 약 80%, 10%, 10%로 나누므로 행 개수 비율은 정확히 같지 않습니다.

MovieLens의 0.5~5점은 학습 데이터 계약에 맞춰 `1 + (원래 평점 - 0.5) × 4 / 4.5`로 변환합니다. 원본 ZIP의 지문과 변환식은 manifest에 남습니다. 변환 후 4점 기준은 원래 4점과 같지 않다는 점도 평가 해석에 포함해야 합니다.

이 과정은 latest-small을 대상으로 합니다. 수천만 건을 현재의 조밀 행렬 구현에 바로 넣으면 메모리 문제가 생깁니다. 대규모 과정으로 확장하려면 희소 행렬과 ANN, 후보 생성 및 순위화를 별도로 설계해야 합니다.

## 실패 실험

아주 낮은 RMSE 상한으로 기존 후보를 평가합니다.

```powershell
uv run ml-labs movies evaluate --candidate .ml-labs/runs/movies/candidate.json --data data/movies --report .ml-labs/reports/movies-rejected.json --max-rmse 0.01
```

종료 코드가 0이 아니고, 보고서에 `passed: false`가 기록되어야 합니다. 그 보고서를 `register`에 넘겼을 때도 등록이 거절되어야 합니다. 이것이 단순히 경고를 출력하는 것과 평가 게이트의 차이입니다.

이어서 새 사용자와 기존 사용자를 HTTP로 조회합니다. [서빙 과정](05-serving-monitoring.md)에서 두 응답의 `status`와 추천 목록을 비교할 수 있습니다.

## 더 깊이 연습하기

rank 2, 4, 8을 같은 데이터에서 비교하고 검증 기준으로 하나만 고릅니다. 학습 데이터만으로 만든 인기 추천의 Recall@5도 계산해 비교해 보세요. 이어 NDCG@5, 장르별 커버리지와 사용자 활동량별 성능을 추가하면 평균 RMSE가 숨기는 문제를 볼 수 있습니다. 이 추가 지표와 세그먼트 평가는 현재 자동 구현 범위가 아닌 확장 과제입니다.

한 줄 정리: 추천에서는 평점 오차, 순위 품질, 처음 보는 사용자와 시간 누수를 서로 다른 문제로 확인합니다.

[이전: 구조와 준비](00-foundation.md) / [다음: 객체 탐지](02-cards.md)
